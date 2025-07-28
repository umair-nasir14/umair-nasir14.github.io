from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import os
import importlib.util
import io
import base64
import sys
import re
import ast
import shutil
import glob

app = Flask(__name__, static_folder='webapp', template_folder='webapp')

class CodeCleaner(ast.NodeVisitor):
    """AST visitor that extracts only imports, classes, and functions"""
    
    def __init__(self):
        self.kept_nodes = []
    
    def visit_Import(self, node: ast.Import) -> None:
        """Keep import statements"""
        self.kept_nodes.append(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Keep from...import statements"""
        self.kept_nodes.append(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Keep class definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire class as-is
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Keep function definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire function as-is
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Keep async function definitions"""
        self.kept_nodes.append(node)
        # Don't visit children - we want the entire function as-is

class GameManager:
    def __init__(self):
        self.current_game_env = None
        self.current_game_module = None
        self.game_scores = {}
        self.game_descriptions = {}
        self.current_game_name = None
        self.load_game_scores()
        self.load_game_descriptions()
        
    def load_game_scores(self):
        """Load game scores from JSON file"""
        try:
            with open('cache/001/game_scores.json', 'r') as f:
                self.game_scores = json.load(f)
        except FileNotFoundError:
            print("Game scores file not found")
            self.game_scores = {}
    
    def load_game_descriptions(self):
        """Load game descriptions from JSON file"""
        try:
            with open('cache/001/game_descriptions.json', 'r') as f:
                self.game_descriptions = json.load(f)
        except FileNotFoundError:
            print("Game descriptions file not found")
            self.game_descriptions = {}
    
    def find_closest_games(self, target_score):
        """Find games with scores closest to the target score"""
        if not self.game_scores:
            return []
        
        # Calculate absolute differences
        differences = {name: abs(score - target_score) 
                      for name, score in self.game_scores.items()}
        
        # Find minimum difference
        min_diff = min(differences.values())
        
        # Return all games with the minimum difference
        closest_games = [name for name, diff in differences.items() 
                        if diff == min_diff]
        
        return [(name, self.game_scores[name]) for name in closest_games]
    
    def load_game(self, game_name):
        """Dynamically load a game module and create environment"""
        try:
            game_file = f"cache/001/{game_name}.py"
            
            if not os.path.exists(game_file):
                raise FileNotFoundError(f"Game file {game_file} not found")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("game_module", game_file)
            game_module = importlib.util.module_from_spec(spec)
            
            # Add the module to sys.modules to handle imports
            sys.modules["game_module"] = game_module
            spec.loader.exec_module(game_module)
            
            # Create the game environment
            if hasattr(game_module, 'make_game'):
                game_env, str_world, tile_mapping, env_image, mechanics_to_actions = game_module.make_game()
                self.current_game_env = game_env
                self.current_game_module = game_module
                self.current_game_name = game_name
                return True
            else:
                print(f"Game {game_name} does not have make_game function")
                return False
                
        except Exception as e:
            print(f"Error loading game {game_name}: {e}")
            return False
    
    def get_game_state(self):
        """Get current game state for rendering"""
        if not self.current_game_env:
            return None
            
        try:
            # Get the current state
            state = self.current_game_env.get_state()
            
            # Get tile mapping for rendering
            if hasattr(self.current_game_module, 'env_dict'):
                env_images, tile_mapping = self.current_game_module.env_dict()
                
                # Convert PIL images to base64 for web display
                tile_paths = {}
                for tile, img in env_images.items():
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    tile_paths[tile] = f"data:image/png;base64,{img_str}"
                
                return {
                    "map": state["map"],
                    "tile_paths": tile_paths
                }
            
            return {"map": state["map"], "tile_paths": {}}
            
        except Exception as e:
            print(f"Error getting game state: {e}")
            return None
    
    def take_action(self, action):
        """Execute action in current game"""
        if not self.current_game_env:
            return {"error": "No game loaded"}
            
        try:
            state, reward, done, truncated, info = self.current_game_env.step(action)
            
            return {
                "observation": state,
                "reward": reward,
                "done": done,
                "info": info
            }
        except Exception as e:
            print(f"Error taking action: {e}")
            return {"error": str(e)}
    
    def reset_game(self):
        """Reset current game to initial state"""
        if not self.current_game_env:
            return {"error": "No game loaded"}
            
        try:
            initial_state = self.current_game_env.reset()
            return {"initial_state": initial_state}
        except Exception as e:
            print(f"Error resetting game: {e}")
            return {"error": str(e)}
    
    def clean_python_file(self, file_path, output_path=None, create_backup=True, path_replacement=None):
        """
        Clean a Python file by keeping only imports, classes, and functions.
        
        Args:
            file_path: Path to the input Python file
            output_path: Path for the output file (if None, overwrites original)
            create_backup: If True, create a backup before cleaning
            path_replacement: Tuple of (old_path, new_path) for string replacement
        
        Returns:
            dict: Status and message
        """
        try:
            # Read the original file
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except Exception as e:
            return {"success": False, "error": f"Error reading file: {e}"}
        
        try:
            # Parse the AST
            tree = ast.parse(original_code)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error in file: {e}"}
        
        # Extract only the nodes we want to keep
        cleaner = CodeCleaner()
        cleaner.visit(tree)
        
        if not cleaner.kept_nodes:
            return {"success": False, "error": "No imports, classes, or functions found"}
        
        # Create a new AST with only the kept nodes
        new_tree = ast.Module(body=cleaner.kept_nodes, type_ignores=[])
        
        # Convert back to Python code
        try:
            import astor
            cleaned_code = astor.to_source(new_tree)
        except ImportError:
            # Fallback: use ast.unparse (Python 3.9+)
            try:
                cleaned_code = ast.unparse(new_tree)
            except AttributeError:
                return {"success": False, "error": "Requires 'astor' package or Python 3.9+"}
        
        # Apply path replacement if specified
        if path_replacement:
            old_path, new_path = path_replacement
            
            # Try to replace the original path first
            replacement_made = False
            original_cleaned_code = cleaned_code
            
            # Multiple replacement attempts with different quote styles
            replacements = [
                (f"'{old_path}'", f"r'{new_path}'"),
                (f'"{old_path}"', f'r"{new_path}"'),
                (f"'{old_path}/", f"r'{new_path}/"),
                (f'"{old_path}/', f'r"{new_path}/'),
                (old_path, new_path),  # Unquoted version
                (f"{old_path}/", f"{new_path}/"),  # With trailing slash
            ]
            
            for old_pattern, new_pattern in replacements:
                if old_pattern in cleaned_code:
                    cleaned_code = cleaned_code.replace(old_pattern, new_pattern)
                    replacement_made = True
                    print(f"   Replaced: {old_pattern} -> {new_pattern}")
            
            if not replacement_made:
                # Check if the path appears in any form
                if old_path in original_cleaned_code or old_path.replace('/', '\\') in original_cleaned_code:
                    print(f"   Warning: Path variations found but not replaced in file")
                else:
                    print(f"   Note: Path '{old_path}' not found in this file")
        
        # Determine output path
        if output_path is None:
            output_path = file_path
        
        try:
            # Create backup if requested
            if create_backup and output_path == file_path:
                backup_path = f"{file_path}.bak"
                shutil.copy2(file_path, backup_path)
            
            # Write cleaned code
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_code)
            
            return {
                "success": True, 
                "message": f"File cleaned successfully",
                "original_size": len(original_code),
                "cleaned_size": len(cleaned_code),
                "reduction_percent": round((1 - len(cleaned_code)/len(original_code)) * 100, 1)
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error writing file: {e}"}
    
    def clean_game_directory(self, directory_path="cache/001", create_backup=True, path_replacement=None):
        """
        Clean all Python game files in the specified directory.
        
        Args:
            directory_path: Directory containing game files
            create_backup: If True, create backup files
            path_replacement: Tuple of (old_path, new_path) for string replacement
        
        Returns:
            dict: Cleaning results
        """
        if not os.path.isdir(directory_path):
            return {"success": False, "error": f"Directory '{directory_path}' not found"}
        
        # Find all Python files
        pattern = os.path.join(directory_path, "*.py")
        python_files = glob.glob(pattern)
        
        if not python_files:
            return {"success": False, "error": "No Python files found"}
        
        results = {
            "success": True,
            "files_processed": 0,
            "files_failed": 0,
            "total_size_reduction": 0,
            "details": []
        }
        
        for file_path in python_files:
            filename = os.path.basename(file_path)
            result = self.clean_python_file(file_path, None, create_backup, path_replacement)
            
            if result["success"]:
                results["files_processed"] += 1
                results["total_size_reduction"] += result["reduction_percent"]
                results["details"].append({
                    "file": filename,
                    "status": "success",
                    "reduction": f"{result['reduction_percent']}%",
                    "size_before": result["original_size"],
                    "size_after": result["cleaned_size"]
                })
            else:
                results["files_failed"] += 1
                results["details"].append({
                    "file": filename,
                    "status": "failed",
                    "error": result["error"]
                })
        
        if results["files_processed"] > 0:
            results["average_reduction"] = round(
                results["total_size_reduction"] / results["files_processed"], 1
            )
        
        return results
    
    def clean_single_game(self, game_name, create_backup=True, path_replacement=None):
        """
        Clean a specific game file.
        
        Args:
            game_name: Name of the game (without .py extension)
            create_backup: If True, create backup file
            path_replacement: Tuple of (old_path, new_path) for string replacement
        
        Returns:
            dict: Cleaning result
        """
        game_file = f"cache/001/{game_name}.py"
        
        if not os.path.exists(game_file):
            return {"success": False, "error": f"Game file '{game_name}.py' not found"}
        
        return self.clean_python_file(game_file, None, create_backup, path_replacement)
    
    def auto_clean_on_startup(self):
        """
        Automatically clean all game files on application startup.
        Includes path replacement from lustre to current Windows path.
        """
        print("🧹 Auto-cleaning game files on startup...")
        
        # Define path replacement (from lustre to current Windows path)
        old_path = "/mnt/lustre/users/mnasir/gmd"
        new_path = os.path.abspath(".").replace("\\", "/")  # Use forward slashes for Python compatibility
        path_replacement = (old_path, new_path)
        
        print(f"   Path replacement: '{old_path}' -> '{new_path}'")
        
        # Only clean cache/001 directory
        cache_dir = "cache/001"
        
        if not os.path.isdir(cache_dir):
            print(f"   Cache directory '{cache_dir}' not found.")
            return
        
        # Check if it contains Python files
        pattern = os.path.join(cache_dir, "*.py")
        python_files = glob.glob(pattern)
        
        if not python_files:
            print(f"   No Python files found in {cache_dir}")
            return
        
        print(f"   Found {len(python_files)} Python files in {cache_dir}")
        print(f"   Cleaning {cache_dir}...")
        
        result = self.clean_game_directory(
            directory_path=cache_dir, 
            create_backup=True, 
            path_replacement=path_replacement
        )
        
        if result.get("success"):
            files_processed = result.get("files_processed", 0)
            files_failed = result.get("files_failed", 0)
            avg_reduction = result.get("average_reduction", 0)
            
            print(f"     ✅ {files_processed} files cleaned (avg {avg_reduction}% reduction)")
            if files_failed > 0:
                print(f"     ⚠️  {files_failed} files failed")
            
            print(f"🎉 Startup cleaning complete! {files_processed} files cleaned, {files_failed} failed")
        else:
            print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
            print(f"🎉 Startup cleaning complete! 0 files cleaned, 1 failed")

# Global game manager instance
game_manager = GameManager()

# Auto-clean games on startup
game_manager.auto_clean_on_startup()

@app.route('/mortar')
def index():
    """Main page for game score search"""
    return render_template('game_search.html')

@app.route('/search_games', methods=['POST'])
def search_games():
    """API endpoint to search games by score"""
    try:
        data = request.get_json()
        target_score = float(data.get('score', 0))
        
        closest_games = game_manager.find_closest_games(target_score)
        
        return jsonify({
            "success": True,
            "target_score": target_score,
            "games": closest_games
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/load_game/<game_name>')
def load_game(game_name):
    """Load and start a specific game"""
    success = game_manager.load_game(game_name)
    
    if success:
        return redirect(url_for('play_game'))
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to load game {game_name}"
        })

@app.route('/play')
def play_game():
    """Game playing interface"""
    return render_template('index.html')

@app.route('/get_game_state')
def get_game_state():
    """API endpoint for getting current game state"""
    state = game_manager.get_game_state()
    
    if state:
        return jsonify(state)
    else:
        return jsonify({"error": "No game loaded"})

@app.route('/get_game_mechanics')
def get_game_mechanics():
    """API endpoint for getting current game mechanics to action mapping"""
    if not game_manager.current_game_env:
        return jsonify({"error": "No game loaded"})
    
    try:
        if hasattr(game_manager.current_game_env, 'get_mechanics_to_action'):
            mechanics = game_manager.current_game_env.get_mechanics_to_action()
            return jsonify({"mechanics": mechanics})
        else:
            return jsonify({"error": "Game does not support mechanics mapping"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/take_action/<int:action>')
def take_action(action):
    """API endpoint for taking actions in the game"""
    result = game_manager.take_action(action)
    return jsonify(result)

@app.route('/reset_game')
def reset_game():
    """API endpoint for resetting the game"""
    result = game_manager.reset_game()
    return jsonify(result)

@app.route('/get_all_games')
def get_all_games():
    """API endpoint to get all available games and their scores"""
    return jsonify(game_manager.game_scores)

@app.route('/get_game_description')
def get_game_description():
    """API endpoint to get current game's description"""
    if not game_manager.current_game_name:
        return jsonify({"error": "No game loaded"})
    
    game_desc = game_manager.game_descriptions.get(game_manager.current_game_name)
    if game_desc:
        return jsonify({
            "game_name": game_manager.current_game_name,
            "narrative": game_desc.get("Game narrative", ""),
            "win_condition": game_desc.get("Win condition", "")
        })
    else:
        return jsonify({"error": "Description not found for current game"})

# Hidden API endpoints for code cleaning (not exposed in UI)
@app.route('/admin/clean_all_games', methods=['POST'])
def clean_all_games():
    """Hidden API endpoint to clean all game files"""
    try:
        data = request.get_json() or {}
        create_backup = data.get('backup', True)
        path_replacement = None
        
        # Check if path replacement is requested
        if 'old_path' in data and 'new_path' in data:
            path_replacement = (data['old_path'], data['new_path'])
        
        result = game_manager.clean_game_directory(create_backup=create_backup, path_replacement=path_replacement)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/admin/clean_game/<game_name>', methods=['POST'])
def clean_single_game(game_name):
    """Hidden API endpoint to clean a specific game file"""
    try:
        data = request.get_json() or {}
        create_backup = data.get('backup', True)
        path_replacement = None
        
        # Check if path replacement is requested
        if 'old_path' in data and 'new_path' in data:
            path_replacement = (data['old_path'], data['new_path'])
        
        result = game_manager.clean_single_game(game_name, create_backup, path_replacement)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/admin/clean_status')
def clean_status():
    """Hidden API endpoint to check which games have backups (indicating they were cleaned)"""
    try:
        backup_files = glob.glob("cache/001/*.py.bak")
        cleaned_games = [os.path.basename(f).replace('.py.bak', '') for f in backup_files]
        
        return jsonify({
            "success": True,
            "cleaned_games": cleaned_games,
            "total_cleaned": len(cleaned_games)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 