from flask import Flask, render_template, jsonify, request, send_from_directory
import json
import os
import importlib.util
import sys
import traceback
import random

app = Flask(__name__, static_folder='webapp', template_folder='webapp')

class GameManager:
    def __init__(self):
        self.current_game_env = None
        self.current_game_module = None
        self.game_scores = {}
        self.game_descriptions = {}
        self.current_game_name = None
        self.fallback_games = {}
        self.load_game_scores()
        self.load_game_descriptions()
        self.initialize_fallback_games()
        
    def load_game_scores(self):
        """Load game scores from JSON file"""
        try:
            with open('game_app/cache/001/game_scores.json', 'r') as f:
                self.game_scores = json.load(f)
            print(f"Loaded game scores: {self.game_scores}")
        except FileNotFoundError:
            print("Game scores file not found, using default scores")
            self.game_scores = {
                "0_AllyCraft_g1": 0.5,
                "EcoSync_g2": 0.09,
                "Elemental_Synergy_Quest_g3": 0.31,
                "Tile_Strategy_Quest_g4": 0.44
            }
    
    def load_game_descriptions(self):
        """Load game descriptions from JSON file"""
        try:
            with open('game_app/cache/001/game_descriptions.json', 'r') as f:
                self.game_descriptions = json.load(f)
            print(f"Loaded {len(self.game_descriptions)} game descriptions")
        except FileNotFoundError:
            print("Game descriptions file not found")
            self.game_descriptions = {}
    
    def initialize_fallback_games(self):
        """Initialize fallback game states"""
        self.fallback_games = {
            "0_AllyCraft_g1": {
                "map": [
                    ['A', 'A', '#', 'A', 'A', 'A', 'A', 'O'],
                    ['A', '@', 'A', 'A', '&', 'A', 'R', 'A'],
                    ['A', 'A', 'A', 'S', 'A', 'A', 'A', 'A'],
                    ['R', 'A', 'A', 'A', 'A', '#', 'A', 'A'],
                    ['A', 'A', 'O', 'A', 'A', 'A', 'A', 'I']
                ],
                "player_pos": [1, 1],
                "health": 100,
                "score": 0,
                "mechanics": {
                    "move_player_up": 0,
                    "move_player_down": 1,
                    "move_player_left": 2,
                    "move_player_right": 3,
                    "summon_ally": 4,
                    "attack": 5
                },
                "tile_paths": {}
            },
            "EcoSync_g2": {
                "map": [
                    ['A', 'M', 'G', 'A', 'M', 'A', 'R', 'A'],
                    ['G', '@', 'M', 'A', '#', 'A', 'A', 'O'],
                    ['A', 'G', 'A', 'A', 'A', '&', 'M', 'A'],
                    ['M', 'A', 'R', 'A', 'G', 'A', 'A', 'A'],
                    ['A', 'A', 'A', 'M', 'A', 'A', 'G', 'R']
                ],
                "player_pos": [1, 1],
                "health": 80,
                "score": 0,
                "mechanics": {
                    "move_player_up": 0,
                    "move_player_down": 1,
                    "move_player_left": 2,
                    "move_player_right": 3,
                    "cleanse_pollution": 4,
                    "teleport_enemy": 5
                },
                "tile_paths": {}
            },
            "Elemental_Synergy_Quest_g3": {
                "map": [
                    ['A', 'F', 'W', 'A', 'E', 'A', 'R', 'O'],
                    ['F', '@', 'A', 'W', 'A', '#', 'A', 'I'],
                    ['A', 'E', 'T', 'A', 'C', 'A', 'D', 'A'],
                    ['W', 'A', 'A', 'F', 'A', '&', 'A', 'R'],
                    ['A', 'R', 'A', 'A', 'I', 'A', 'O', 'A']
                ],
                "player_pos": [1, 1],
                "health": 90,
                "score": 0,
                "mechanics": {
                    "move_player_up": 0,
                    "move_player_down": 1,
                    "move_player_left": 2,
                    "move_player_right": 3,
                    "elemental_link": 4,
                    "burst_explosion": 5
                },
                "tile_paths": {}
            },
            "Tile_Strategy_Quest_g4": {
                "map": [
                    ['A', 'X', 'Y', 'A', 'C', 'A', 'O', 'A'],
                    ['Z', '@', 'A', 'S', 'A', '#', 'A', 'R'],
                    ['A', 'A', 'C', 'A', 'A', 'A', '&', 'A'],
                    ['R', 'A', 'A', 'X', 'Y', 'A', 'A', 'O'],
                    ['A', 'S', 'A', 'A', 'Z', 'A', 'A', 'A']
                ],
                "player_pos": [1, 1],
                "health": 60,
                "mana": 30,
                "score": 0,
                "mechanics": {
                    "move_player_up": 0,
                    "move_player_down": 1,
                    "move_player_left": 2,
                    "move_player_right": 3,
                    "craft_combination": 4,
                    "create_shield": 5,
                    "area_blast": 6
                },
                "tile_paths": {}
            }
        }
    
    def find_closest_games(self, target_score):
        """Find games with scores closest to the target score"""
        if not self.game_scores:
            return []
        
        differences = {name: abs(score - target_score) 
                      for name, score in self.game_scores.items()}
        
        min_diff = min(differences.values())
        closest_games = [name for name, diff in differences.items() 
                        if diff == min_diff]
        
        return [(name, self.game_scores[name]) for name in closest_games]
    
    def load_game(self, game_name):
        """Load a game dynamically or use fallback"""
        try:
            game_file = f"game_app/cache/001/{game_name}.py"
            
            if not os.path.exists(game_file):
                print(f"Game file {game_file} not found, using fallback")
                return self.use_fallback_game(game_name)
            
            # Try dynamic loading but fall back gracefully
            try:
                spec = importlib.util.spec_from_file_location("game_module", game_file)
                game_module = importlib.util.module_from_spec(spec)
                sys.modules["game_module"] = game_module
                spec.loader.exec_module(game_module)
                
                if hasattr(game_module, 'make_game'):
                    game_env, _, _, _, _ = game_module.make_game()
                    self.current_game_env = game_env
                    self.current_game_module = game_module
                    self.current_game_name = game_name
                    print(f"Successfully loaded game: {game_name}")
                    return True
                else:
                    return self.use_fallback_game(game_name)
                    
            except Exception as e:
                print(f"Error loading game module {game_name}: {e}")
                return self.use_fallback_game(game_name)
                
        except Exception as e:
            print(f"Error loading game {game_name}: {e}")
            return self.use_fallback_game(game_name)
    
    def use_fallback_game(self, game_name):
        """Use fallback game state"""
        if game_name in self.fallback_games:
            print(f"Using fallback game state for {game_name}")
            self.current_game_env = None
            self.current_game_module = None
            self.current_game_name = game_name
            return True
        return False
    
    def get_game_state(self):
        """Get current game state"""
        if self.current_game_env:
            try:
                state = self.current_game_env.get_state()
                return state
            except:
                pass
        
        if self.current_game_name and self.current_game_name in self.fallback_games:
            fallback = self.fallback_games[self.current_game_name]
            return {
                "map": fallback["map"],
                "tile_paths": fallback["tile_paths"]
            }
        
        return {"error": "No game loaded"}
    
    def get_game_mechanics(self):
        """Get game mechanics"""
        if self.current_game_env:
            try:
                return self.current_game_env.mechanic_to_action
            except:
                pass
        
        if self.current_game_name and self.current_game_name in self.fallback_games:
            return self.fallback_games[self.current_game_name]["mechanics"]
        
        return {}
    
    def take_action(self, action):
        """Take an action in the current game"""
        if self.current_game_env:
            try:
                observation, reward, done, info = self.current_game_env.step(action)
                return {
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "info": info
                }
            except:
                pass
        
        # Fallback action handling
        if self.current_game_name and self.current_game_name in self.fallback_games:
            return self.fallback_take_action(action)
        
        return {"error": "No game loaded"}
    
    def fallback_take_action(self, action):
        """Simple fallback action handling"""
        game_state = self.fallback_games[self.current_game_name]
        current_pos = game_state["player_pos"]
        game_map = game_state["map"]
        
        new_pos = current_pos.copy()
        reward = 0
        
        if action == 0:  # Move up
            new_pos[0] = max(0, current_pos[0] - 1)
        elif action == 1:  # Move down
            new_pos[0] = min(len(game_map) - 1, current_pos[0] + 1)
        elif action == 2:  # Move left
            new_pos[1] = max(0, current_pos[1] - 1)
        elif action == 3:  # Move right
            new_pos[1] = min(len(game_map[0]) - 1, current_pos[1] + 1)
        elif action >= 4:  # Special actions
            reward = random.randint(1, 10)
        
        if new_pos != current_pos:
            game_map[current_pos[0]][current_pos[1]] = 'A'
            
            target_tile = game_map[new_pos[0]][new_pos[1]]
            if target_tile in ['O', 'I', 'R']:
                reward = 20
                game_state["score"] += reward
            elif target_tile in ['#', '&']:
                reward = -5
                game_state["health"] = max(0, game_state["health"] - 10)
            
            game_map[new_pos[0]][new_pos[1]] = '@'
            game_state["player_pos"] = new_pos
        
        return {
            "observation": game_map,
            "reward": reward,
            "done": False,
            "info": {}
        }
    
    def reset_game(self):
        """Reset the current game"""
        if self.current_game_env:
            try:
                initial_state = self.current_game_env.reset()
                return {"initial_state": initial_state}
            except:
                pass
        
        if self.current_game_name and self.current_game_name in self.fallback_games:
            self.initialize_fallback_games()
            return {"initial_state": self.fallback_games[self.current_game_name]["map"]}
        
        return {"error": "No game to reset"}

game_manager = GameManager()

# Serve your existing website files (don't change main page)
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('.', filename)

# Games routes are at /games/ path - won't interfere with your main site
@app.route('/games/')
def games_index():
    return render_template('game_search.html')

@app.route('/games/get_all_games')
def get_all_games():
    return jsonify(game_manager.game_scores)

@app.route('/games/search_games', methods=['POST'])
def search_games():
    data = request.get_json()
    target_score = data.get('score', 0.5)
    closest_games = game_manager.find_closest_games(target_score)
    
    return jsonify({
        "success": True,
        "target_score": target_score,
        "games": closest_games
    })

@app.route('/games/load_game/<game_name>')
def load_game(game_name):
    success = game_manager.load_game(game_name)
    if success:
        return render_template('index.html')
    else:
        return f"Game {game_name} not found", 404

@app.route('/games/get_game_state')
def get_game_state():
    return jsonify(game_manager.get_game_state())

@app.route('/games/get_game_mechanics')
def get_game_mechanics():
    mechanics = game_manager.get_game_mechanics()
    return jsonify({"mechanics": mechanics})

@app.route('/games/get_game_description')
def get_game_description():
    if game_manager.current_game_name:
        description = game_manager.game_descriptions.get(game_manager.current_game_name, {})
        return jsonify({
            "narrative": description.get("Game narrative", ""),
            "win_condition": description.get("Win condition", "")
        })
    return jsonify({"error": "No game loaded"})

@app.route('/games/take_action/<int:action>')
def take_action(action):
    return jsonify(game_manager.take_action(action))

@app.route('/games/reset_game')
def reset_game():
    return jsonify(game_manager.reset_game())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port) 