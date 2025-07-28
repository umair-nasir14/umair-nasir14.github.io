import os
import sys
from mech_mcts import PlayMCTS, calculate_mechanic_fitness, RandomAgent
from utils import extract_list
import numpy as np
import json
import time
import imageio
from PIL import Image, ImageDraw, ImageFont
from rembg import remove
import math
import copy
import random
import openai
import traceback
from dotenv import load_dotenv
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import ray
import gc


class GameEnv(gym.Env):

    def __init__(self, walkable_tiles, tiles_without_char, tiles,
        str_map_without_chars, str_map, interactive_object_tiles,
        enemy_tiles, collectible_tiles, extra_tiles, render_mode='rgb_array'):
        super(GameEnv, self).__init__()
        self.map_str_without_chars = str_map_without_chars.strip().split('\n')
        self.map_str = str_map.strip().split('\n')
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.
            map_str_without_chars]
        self.tiles = tiles
        self.tiles_without_char = tiles_without_char
        self.action_space = spaces.Discrete(self.get_action_space())
        self.char_set = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
            'W': 6, 'R': 7, 'T': 8, 'O': 9, 'I': 10, 'X': 11, 'P': 12, 'Z':
            13, 'M': 14, '@': 15, '#': 16, '&': 17}
        self.char_to_int = lambda c: self.char_set.get(c, 0)
        self.mechanic_to_action = self.get_mechanics_to_action()
        max_width = max(len(row) for row in self.map_str)
        self.tile_size = 16
        self.char_tile_size = 16
        self.frames = []
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.
            char_set), len(self.map_str), max_width), dtype=np.int32)
        self.render_mode = 'rgb_array'
        self.default_walkable_tile = 'A'
        self.walkable_tiles = walkable_tiles
        self.interactive_object_tiles = interactive_object_tiles
        self.enemy_tiles = enemy_tiles
        self.npc_tiles = ['&']
        self.enemy_tiles = ['#']
        self.extra_tiles = extra_tiles
        self.collectible_tiles = collectible_tiles
        self.current_score = 0
        self.objects_on_target = 0
        self.collected_items = 0
        self.player_health = 100
        self.enemy_health = 100
        self.inventory = []
        self.chests_collected = 0
        self.map = [list(row) for row in self.map_str]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    self.player_position = i, j
        self.current_tile = self.default_walkable_tile
        self.reset()

    def reset(self, seed=None):
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.
            map_str_without_chars]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    self.player_position = i, j
        self.current_tile = self.map_without_chars[self.player_position[0]][
            self.player_position[1]]
        self.inventory = []
        self.chests_collected = 0
        self.current_score = 0
        return self.get_state()['map']

    def move_player(self, action):
        moves = {(0): (-1, 0), (1): (1, 0), (2): (0, -1), (3): (0, 1)}
        dx, dy = moves[action]
        new_row = self.player_position[0] + dx
        new_col = self.player_position[1] + dy
        reward = 0
        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            new_tile = self.map[new_row][new_col]
            if (new_tile in self.walkable_tiles or new_tile in self.
                collectible_tiles or new_tile in self.interactive_object_tiles
                ):
                if new_tile in self.interactive_object_tiles:
                    reward += self.pickup_object(new_row, new_col, new_tile)
                self.update_player_position(new_row, new_col, new_tile)
        return reward

    def get_state(self):
        return {'map': self.map, 'player_position': self.player_position,
            'inventory': self.inventory, 'chests_collected': self.
            chests_collected, 'collected_items': self.collected_items,
            'current_score': self.current_score}

    def step(self, action):
        reward = 0
        if action < 4:
            reward += self.move_player(action)
        elif action == 4:
            reward += self.enhanced_elemental_link()
        elif action == 5:
            reward += self.collaborative_tile_enrichment_with_strategy()
        elif action == 6:
            reward += self.manual_pickup()
        elif action == 7:
            reward += len(self.inventory)
        reward += self.cooperative_burst_explosion()
        reward += self.teleport_synergy_bonus()
        reward += self.clone_collaboration()
        self.current_score += reward
        done = self.is_terminal()
        if done:
            reward += 10
            self.current_score += 10
        info = {'inventory': self.inventory.copy(), 'chests_collected':
            self.chests_collected, 'total_items': self.collected_items,
            'current_score': self.current_score}
        return self.get_state()['map'], reward, done, False, info

    def render(self, mode='human'):
        env_img = Image.new('RGBA', (len(self.map[0]) * self.tile_size, len
            (self.map) * self.tile_size))
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                tile_img = self.tiles[self.default_walkable_tile].resize((
                    self.tile_size, self.tile_size))
                env_img.paste(tile_img, (j * self.tile_size, i * self.
                    tile_size), tile_img)
        for i, row in enumerate(self.map_without_chars):
            for j, tile in enumerate(row):
                if tile in self.tiles and tile != self.default_walkable_tile:
                    tile_img = self.tiles[tile].resize((self.tile_size,
                        self.tile_size))
                    env_img.paste(tile_img, (j * self.tile_size, i * self.
                        tile_size), tile_img)
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile in self.tiles and tile not in self.walkable_tiles:
                    if tile.isalpha():
                        tile_img = self.tiles[tile].resize((self.tile_size,
                            self.tile_size))
                    else:
                        tile_img = self.tiles[tile].resize((self.
                            char_tile_size, self.char_tile_size))
                        x_offset = (self.tile_size - self.char_tile_size) // 2
                        y_offset = (self.tile_size - self.char_tile_size) // 2
                        env_img.paste(tile_img, (j * self.tile_size +
                            x_offset, i * self.tile_size + y_offset), tile_img)
        for clone_row in range(len(self.map)):
            for clone_col in range(len(self.map[0])):
                if self.map[clone_row][clone_col] == 'C':
                    clone_img = self.tiles['C'].resize((self.tile_size,
                        self.tile_size))
                    env_img.paste(clone_img, (clone_col * self.tile_size, 
                        clone_row * self.tile_size), clone_img)
        frame = np.array(env_img.convert('RGB'))
        self.frames.append(frame)
        return frame

    def get_action_space(self):
        return 8

    def get_mechanics_to_action(self):
        return {'move_up': 0, 'move_down': 1, 'move_left': 2, 'move_right':
            3, 'enhanced_elemental_link': 4,
            'collaborative_tile_enrichment_with_strategy': 5,
            'manual_pickup': 6, 'check_inventory': 7}

    def enhanced_elemental_link(self):
        reward = 0
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        elemental_tiles = ['F', 'W', 'E']
        elemental_connections = []
        nearby_players = []
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] == '@':
                    nearby_players.append((new_row, new_col))
                    reward += 3
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] in elemental_tiles:
                    reward += 10
                    elemental_connections.append((new_row, new_col))
                    self.map[new_row][new_col] = '.'
        if nearby_players and elemental_connections:
            reward += len(nearby_players) * 5
        return reward

    def collaborative_tile_enrichment_with_strategy(self):
        reward = 0
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        resource_tiles = ['R', 'F', 'W']
        adjacent_players = []
        strategic_enrichments = []
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] == '@':
                    adjacent_players.append((new_row, new_col))
                    reward += 3
        for dx, dy in adjacency_offsets:
            for row_offset in range(-1, 2):
                for col_offset in range(-1, 2):
                    new_row = self.player_position[0] + dx + row_offset
                    new_col = self.player_position[1] + dy + col_offset
                    if 0 <= new_row < len(self.map) and 0 <= new_col < len(self
                        .map[0]):
                        if self.map[new_row][new_col] in resource_tiles:
                            reward += 10
                            strategic_enrichments.append((new_row, new_col))
                            self.map[new_row][new_col] = '.'
        if adjacent_players:
            reward += len(adjacent_players) * 2
        if len(strategic_enrichments) > 1:
            reward += 5
        return reward

    def cooperative_burst_explosion(self):
        reward = 0
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        explosive_tiles = ['E']
        nearby_players = []
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] == '@':
                    nearby_players.append((new_row, new_col))
                    reward += 3
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] in explosive_tiles:
                    reward += 15
                    if nearby_players:
                        reward += len(nearby_players) * 5
                    self.map[new_row][new_col] = '.'
        return reward

    def teleport_synergy_bonus(self):
        """Rewards players for teleporting and enhances synergies with nearby players and resources."""
        reward = 0
        if self.map[self.player_position[0]][self.player_position[1]] == 'T':
            return self.teleport_player()
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] == '@':
                    reward += 5
                elif self.map[new_row][new_col] in {'A', 'B', 'C', 'D'}:
                    reward += 10
        return reward

    def clone_collaboration(self):
        reward = 0
        clone_positions = [(row, col) for row in range(len(self.map)) for
            col in range(len(self.map[0])) if self.map[row][col] == 'C']
        for clone_row, clone_col in clone_positions:
            adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            for dx, dy in adjacency_offsets:
                new_row, new_col = clone_row + dx, clone_col + dy
                if 0 <= new_row < len(self.map) and 0 <= new_col < len(self
                    .map[0]):
                    if self.map[new_row][new_col] in {'A', 'B', 'C', 'D'}:
                        reward += 5
                        self.map[new_row][new_col] = '.'
        if self.map[self.player_position[0]][self.player_position[1]] == '@':
            for clone_row, clone_col in clone_positions:
                if abs(self.player_position[0] - clone_row) <= 1 and abs(
                    self.player_position[1] - clone_col) <= 1:
                    reward += 3
        return reward

    def pickup_object(self, row, col, tile_type):
        """Handle pickup mechanics for interactive objects like chests"""
        reward = 0
        if tile_type == 'O':
            chest_item = f'Chest_{self.chests_collected + 1}'
            self.inventory.append(chest_item)
            self.chests_collected += 1
            self.collected_items += 1
            reward += 20
        elif tile_type == 'I':
            item = f'Item_{len(self.inventory) + 1}'
            self.inventory.append(item)
            self.collected_items += 1
            reward += 15
        self.map[row][col] = self.default_walkable_tile
        return reward

    def manual_pickup(self):
        """Manual pickup action for objects at current position"""
        reward = 0
        current_tile = self.map[self.player_position[0]][self.
            player_position[1]]
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                tile = self.map[new_row][new_col]
                if tile in self.interactive_object_tiles:
                    reward += self.pickup_object(new_row, new_col, tile)
                    break
        return reward

    def update_player_position(self, new_row, new_col, new_tile):
        if not (0 <= new_row < self.grid_height and 0 <= new_col < self.
            grid_width):
            return
        if not (0 <= self.player_position[0] < self.grid_height and 0 <=
            self.player_position[1] < self.grid_width):
            self.player_position = new_row, new_col
            self.current_tile = new_tile
            self.map[new_row][new_col] = '@'
            return
        if (new_tile not in self.walkable_tiles and new_tile not in self.
            collectible_tiles and new_tile not in self.interactive_object_tiles
            ):
            return
        self.map[self.player_position[0]][self.player_position[1]
            ] = self.current_tile
        self.map_without_chars[self.player_position[0]][self.player_position[1]
            ] = self.current_tile
        self.player_position = new_row, new_col
        if new_tile in self.interactive_object_tiles:
            self.current_tile = self.default_walkable_tile
        else:
            self.current_tile = new_tile
        self.map[new_row][new_col] = '@'

    def find_player_position(self):
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    return i, j
        return None

    def clone(self):
        new_env = object.__new__(GameEnv)
        for attr, value in self.__dict__.items():
            if attr not in ['action_space', 'observation_space']:
                setattr(new_env, attr, copy.deepcopy(value))
        new_env.action_space = spaces.Discrete(new_env.get_action_space())
        new_env.observation_space = spaces.Box(low=0, high=1, shape=(len(
            new_env.char_set), len(new_env.map_str), max(len(row) for row in
            new_env.map_str)), dtype=np.int32)
        return new_env

    def teleport_player(self):
        import random
        walkable_positions = []
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] in self.walkable_tiles:
                    if not (abs(self.player_position[0] - row) <= 1 and abs
                        (self.player_position[1] - col) <= 1):
                        walkable_positions.append((row, col))
        if walkable_positions:
            new_position = random.choice(walkable_positions)
            self.update_player_position(new_position[0], new_position[1],
                self.map[new_position[0]][new_position[1]])
            return 10
        return 0

    def is_terminal(self):
        target_items = 3
        if self.collected_items >= target_items:
            return True
        collectible_tiles_remaining = any(self.map[row][col] in ['F', 'W',
            'E', 'R', 'D', 'T', 'O', 'I'] for row in range(len(self.map)) for
            col in range(len(self.map[0])))
        if not collectible_tiles_remaining:
            return True
        return False


def env_dict():
    env_image = dict()
    image_paths = dict()

    def load_image(char, path):
        env_image[char] = Image.open(path).convert('RGBA')
        image_paths[char] = path
    base_path = 'C:\\Users\\DELL\\Projects\\Research\\gmd'
    load_image('A',
        f'{base_path}/world_tileset_data/td_world_floor_grass_c.png')
    load_image('B',
        f'{base_path}/world_tileset_data/td_world_wall_stone_h_a.png')
    load_image('X',
        f'{base_path}/world_tileset_data/td_world_floor_grass_c.png')
    load_image('O', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('I', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('@',
        f'{base_path}/character_sprite_data/td_monsters_archer_d1.png')
    load_image('#',
        f'{base_path}/character_sprite_data/td_monsters_witch_d1.png')
    load_image('&',
        f'{base_path}/character_sprite_data/td_monsters_goblin_captain_d1.png')
    load_image('F', f'{base_path}/world_tileset_data/td_items_amulet_gold.png')
    load_image('W', f'{base_path}/world_tileset_data/td_items_gem_ruby.png')
    load_image('E', f'{base_path}/world_tileset_data/td_world_crate.png')
    load_image('R', f'{base_path}/world_tileset_data/tg_world_barrel.png')
    load_image('D', f'{base_path}/world_tileset_data/tg_world_floor_moss_e.png'
        )
    load_image('T', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('C',
        f'{base_path}/character_sprite_data/td_monsters_angel_d2.png')
    load_image('P',
        f'{base_path}/character_sprite_data/td_monsters_archer_u2.png')
    load_image('Z',
        f'{base_path}/character_sprite_data/td_monsters_berserker_d1.png')
    load_image('M',
        f'{base_path}/character_sprite_data/td_monsters_demon_l1.png')
    return env_image, image_paths


def str_map():
    str_world = """AAAAAAAAAAAAAAAAAA
AAAAAAAAAAAAAAAAAA
AAA@OAXEAAAAAAAAAA
AAAA#IRAAAAAAAAAAA
AAAAAA#AAAAAAAAAAA
AAAAAAAAAADTFAAAAA
AAAAAAAAAAAEAAAAAA
AAAAAAAAAAAAAAAAAA"""
    return str_world


def important_tiles():
    walkables = ['A', 'X', 'T']
    non_walkables = ['B']
    interactive_object_tiles = ['O', 'I']
    collectible_tiles = ['F', 'W', 'E', 'R', 'D']
    npc_tiles = ['&']
    player_tile = ['@']
    enemy_tiles = ['#']
    extra_tiles = ['C', 'P', 'Z', 'M']
    return (walkables, non_walkables, interactive_object_tiles,
        collectible_tiles, npc_tiles, player_tile, enemy_tiles, extra_tiles)


def create_all_a_map(str_world, objects):
    lines = str_world.split('\n')
    new_lines = []
    for line in lines:
        new_line = ''.join(char if char in objects else 'A' for char in line)
        new_lines.append(new_line)
    return '\n'.join(new_lines)


def pad_rows_to_max_length(text):
    lines = text.strip().split('\n')
    max_length = max(len(line) for line in lines)
    padded_lines = [(line + line[-1] * (max_length - len(line)) if line else
        '') for line in lines]
    return '\n'.join(padded_lines)


def remove_spaces(map_str):
    lines = map_str.strip().split('\n')
    cleaned_lines = [line.replace(' ', '') for line in lines]
    return '\n'.join(cleaned_lines)


def remove_extra_players(input_string):
    special_chars = '@'
    first_occurrences = {char: (False) for char in special_chars}
    new_string = []
    for char in input_string:
        if char in special_chars:
            if not first_occurrences[char]:
                new_string.append(char)
                first_occurrences[char] = True
        else:
            new_string.append(char)
    return ''.join(new_string)


def ensure_player_exists(str_world):
    lines = str_world.strip().split('\n')
    has_player = any('@' in line for line in lines)
    if not has_player:
        grid = [list(line.strip()) for line in lines]
        a_positions = []
        for i, row in enumerate(grid):
            for j, char in enumerate(row):
                if char == 'A':
                    a_positions.append((i, j))
        if a_positions:
            import random
            i, j = random.choice(a_positions)
            grid[i][j] = '@'
            return '\n'.join(''.join(row) for row in grid)
    return str_world


def make_game():
    str_world = str_map()
    str_world = remove_spaces(str_world)
    str_world = ensure_player_exists(str_world)
    str_world = remove_extra_players(str_world)
    str_world = pad_rows_to_max_length(str_world)
    (walkables, non_walkables, interactive_object_tiles, collectible_tiles,
        npc_tiles, player_tile, enemy_tiles, extra_tiles) = important_tiles()
    tile_mapping = {'walkable_tiles': walkables, 'non_walkable_tiles':
        non_walkables, 'interactive_object_tiles': interactive_object_tiles,
        'collectible_tiles': collectible_tiles, 'npc_tiles': npc_tiles,
        'player_tile': player_tile, 'enemy_tiles': enemy_tiles,
        'extra_tiles': extra_tiles}
    str_map_wo_chars = create_all_a_map(str_world, interactive_object_tiles)
    env_image, _ = env_dict()
    env = GameEnv(walkable_tiles=walkables, tiles_without_char=
        str_map_wo_chars, tiles=env_image, str_map_without_chars=
        str_map_wo_chars, str_map=str_world, interactive_object_tiles=
        interactive_object_tiles, enemy_tiles=enemy_tiles,
        collectible_tiles=collectible_tiles, extra_tiles=extra_tiles,
        render_mode='rgb_array')
    mechanics_to_actions = env.get_mechanics_to_action()
    return env, str_world, tile_mapping, env_image, mechanics_to_actions


class GymCompatibilityWrapper(gym.Wrapper):

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def clone(self):
        new_env = GymCompatibilityWrapper(self.env.clone())
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        new_env.metadata = self.metadata
        return new_env


def visualize_action_sequence(action_sequence, env, output_path):
    env.reset()
    frames = []
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    for action in action_sequence:
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if done:
            break
    if frames:
        frames.extend([frames[-1]] * 10)
        imageio.mimsave(output_path, frames, fps=2)
    else:
        print('No frames were captured')
