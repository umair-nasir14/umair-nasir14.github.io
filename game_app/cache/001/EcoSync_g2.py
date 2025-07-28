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
        self.char_set = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'O': 4, '@': 5,
            '#': 6, '&': 7}
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
        self.npc_tiles = []
        self.enemy_tiles = ['#', '&', 'E']
        self.extra_tiles = extra_tiles
        self.collectible_tiles = collectible_tiles
        self.current_score = 0
        self.objects_on_target = 0
        self.collected_items = 0
        self.player_health = 100
        self.enemy_health = 100
        self.step_count = 0
        self.teleport_count = 0
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
        self.step_count = 0
        self.player_health = 100
        self.teleport_count = 0
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    self.player_position = i, j
        self.current_tile = self.map_without_chars[self.player_position[0]][
            self.player_position[1]]
        return self.get_state()['map']

    def move_player(self, action):
        moves = {(0): (-1, 0), (1): (1, 0), (2): (0, -1), (3): (0, 1)}
        dx, dy = moves[action]
        new_row = self.player_position[0] + dx
        new_col = self.player_position[1] + dy
        reward = 0
        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            new_tile = self.map[new_row][new_col]
            if new_tile in self.walkable_tiles:
                self.update_player_position(new_row, new_col, new_tile)
        return reward

    def get_state(self):
        return {'map': self.map, 'player_position': self.player_position}

    def step(self, action):
        reward = 0
        self.step_count += 1
        if action < 4:
            reward += self.move_player(action)
        elif action == 4:
            reward += self.resource_rally()
        elif action == 5:
            reward += self.EnvironmentalCleansing()
        elif action == 6:
            reward += self.ambush_and_blast()
        elif action == 7:
            reward += self.confuse_and_teleport_enemies()
        elif action == 8:
            reward += self.resource_synergy_bonus()
        current_tile = self.map[self.player_position[0]][self.
            player_position[1]]
        if current_tile == 'M':
            self.player_health -= 2
            reward -= 1
        if current_tile == 'G':
            self.player_health = min(100, self.player_health + 1)
            reward += 0.5
        done = self.is_terminal()
        if done:
            if self.player_health > 0:
                all_enemy_types = ['#', '&', 'E']
                enemy_count = sum(row.count(enemy) for row in self.map for
                    enemy in all_enemy_types)
                polluted_count = sum(row.count('M') for row in self.map)
                cleansed_count = sum(row.count('G') for row in self.map)
                if hasattr(self, 'teleport_count'
                    ) and self.teleport_count >= 3:
                    reward += 75
                elif enemy_count == 0 and polluted_count == 0:
                    reward += 50
                elif cleansed_count >= 8:
                    reward += 30
                elif enemy_count == 0:
                    reward += 25
                else:
                    reward += 15
            else:
                reward -= 50
        info = {'step_count': self.step_count, 'player_health': self.
            player_health, 'teleport_count': self.teleport_count}
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
        resource_tile_types = ['T', 'R', 'M', 'F']
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile in resource_tile_types:
                    tile_img = self.tiles[tile].resize((self.tile_size,
                        self.tile_size))
                    env_img.paste(tile_img, (j * self.tile_size, i * self.
                        tile_size), tile_img)
        frame = np.array(env_img.convert('RGB'))
        self.frames.append(frame)
        return frame

    def get_mechanics_to_action(self):
        return {'move_up': 0, 'move_down': 1, 'move_left': 2, 'move_right':
            3, 'resource_rally': 4, 'EnvironmentalCleansing': 5,
            'ambush_and_blast': 6, 'confuse_and_teleport_enemies': 7,
            'resource_synergy_bonus': 8}

    def resource_rally(self):
        reward = 0
        rally_positions = []
        resource_tile = 'R'
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == resource_tile:
                    rally_positions.append((row, col))
        if rally_positions:
            for rally_position in rally_positions:
                distance = abs(rally_position[0] - self.player_position[0]
                    ) + abs(rally_position[1] - self.player_position[1])
                if distance <= 2:
                    self.map[rally_position[0]][rally_position[1]] = '.'
                    reward += 4
        if reward > 0:
            new_resource_position = self.player_position[0
                ], self.player_position[1] + 1
            if 0 <= new_resource_position[0] < len(self.map
                ) and 0 <= new_resource_position[1] < len(self.map[0]):
                self.map[new_resource_position[0]][new_resource_position[1]
                    ] = resource_tile
                reward += 2
        return reward

    def EnvironmentalCleansing(self):
        reward = 0
        cleansing_positions = []
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in adjacency_offsets:
            adjacent_row = self.player_position[0] + dx
            adjacent_col = self.player_position[1] + dy
            if 0 <= adjacent_row < len(self.map) and 0 <= adjacent_col < len(
                self.map[0]):
                if self.map[adjacent_row][adjacent_col] == 'M':
                    cleansing_positions.append((adjacent_row, adjacent_col))
        if cleansing_positions:
            cleanse_position = random.choice(cleansing_positions)
            self.map[cleanse_position[0]][cleanse_position[1]] = 'G'
            reward += 8
            for dx, dy in adjacency_offsets:
                bonus_check_row = cleanse_position[0] + dx
                bonus_check_col = cleanse_position[1] + dy
                if 0 <= bonus_check_row < len(self.map
                    ) and 0 <= bonus_check_col < len(self.map[0]):
                    if self.map[bonus_check_row][bonus_check_col] in ['T',
                        'R', 'F']:
                        reward += 3
                        break
            adjacent_cleansed = 0
            for dx, dy in adjacency_offsets:
                check_row = cleanse_position[0] + dx
                check_col = cleanse_position[1] + dy
                if 0 <= check_row < len(self.map) and 0 <= check_col < len(self
                    .map[0]):
                    if self.map[check_row][check_col] == 'G':
                        adjacent_cleansed += 1
            if adjacent_cleansed >= 2:
                reward += 2
        else:
            reward = 0
        return reward

    def ambush_and_blast(self):
        reward = 0
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        enemy_tile = 'E'
        blast_tile = 'Z'
        for dx, dy in adjacent_positions:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                if self.map[new_row][new_col] == enemy_tile:
                    self.map[new_row][new_col] = '.'
                    reward += 2
        if reward > 0:
            blast_positions = []
            for dx, dy in adjacent_positions:
                blast_row = self.player_position[0] + dx
                blast_col = self.player_position[1] + dy
                if 0 <= blast_row < len(self.map) and 0 <= blast_col < len(self
                    .map[0]):
                    if self.map[blast_row][blast_col] == '.':
                        blast_positions.append((blast_row, blast_col))
            if blast_positions:
                blast_position = random.choice(blast_positions)
                self.map[blast_position[0]][blast_position[1]] = blast_tile
                reward += 3
        return reward

    def confuse_and_teleport_enemies(self):
        reward = 0
        adjacent_enemies = []
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        enemy_tiles = ['#', '&', 'E']
        for dx, dy in adjacency_offsets:
            adjacent_row = self.player_position[0] + dx
            adjacent_col = self.player_position[1] + dy
            if 0 <= adjacent_row < len(self.map) and 0 <= adjacent_col < len(
                self.map[0]):
                current_tile = self.map[adjacent_row][adjacent_col]
                if current_tile in enemy_tiles:
                    adjacent_enemies.append((adjacent_row, adjacent_col,
                        current_tile))
        if adjacent_enemies:
            enemy_row, enemy_col, enemy_type = random.choice(adjacent_enemies)
            teleport_positions = []
            for row in range(len(self.map)):
                for col in range(len(self.map[0])):
                    distance_from_player = abs(row - self.player_position[0]
                        ) + abs(col - self.player_position[1])
                    if (self.map[row][col] in self.walkable_tiles and 
                        distance_from_player >= 3 and (row, col) != (
                        enemy_row, enemy_col)):
                        teleport_positions.append((row, col))
            if teleport_positions:
                teleport_destination = random.choice(teleport_positions)
                self.map[teleport_destination[0]][teleport_destination[1]
                    ] = enemy_type
                self.map[enemy_row][enemy_col] = 'A'
                self.teleport_count += 1
                reward += 10
                if self.teleport_count == 1:
                    reward += 5
                elif self.teleport_count == 2:
                    reward += 10
                elif self.teleport_count >= 3:
                    reward += 20
            else:
                reward = 0
        else:
            reward = 0
        return reward

    def resource_synergy_bonus(self):
        reward = 0
        resource_tile_types = ['T', 'R', 'M', 'F']
        if self.map[self.player_position[0]][self.player_position[1]
            ] in resource_tile_types:
            reward += 10
            adjacent_positions = [(dx, dy) for dx in range(-1, 2) for dy in
                range(-1, 2) if dx != 0 or dy != 0]
            for dx, dy in adjacent_positions:
                new_x = self.player_position[0] + dx
                new_y = self.player_position[1] + dy
                if 0 <= new_x < len(self.map) and 0 <= new_y < len(self.map[0]
                    ):
                    if self.map[new_x][new_y] in resource_tile_types:
                        reward += 5
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
        if new_tile not in self.walkable_tiles:
            return
        self.map[self.player_position[0]][self.player_position[1]
            ] = self.current_tile
        self.map_without_chars[self.player_position[0]][self.player_position[1]
            ] = self.current_tile
        self.player_position = new_row, new_col
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

    def get_action_space(self):
        return 9

    def is_terminal(self):
        all_enemy_types = ['#', '&', 'E']
        enemy_count = sum(row.count(enemy) for row in self.map for enemy in
            all_enemy_types)
        polluted_count = sum(row.count('M') for row in self.map)
        resource_types = ['R', 'T', 'F']
        resources_on_map = sum(row.count(tile) for row in self.map for tile in
            resource_types)
        cleansed_count = sum(row.count('G') for row in self.map)
        if hasattr(self, 'player_health') and self.player_health <= 0:
            return True
        if hasattr(self, 'teleport_count') and self.teleport_count >= 3:
            return True
        if enemy_count == 0 and polluted_count == 0 and resources_on_map <= 3:
            return True
        if cleansed_count >= 8 and enemy_count <= 1 and polluted_count <= 2:
            return True
        if enemy_count == 0 and cleansed_count >= 4 and resources_on_map <= 5:
            return True
        if resources_on_map <= 2 and polluted_count <= 3 and enemy_count <= 1:
            return True
        if hasattr(self, 'step_count'):
            if self.step_count > 200:
                if polluted_count >= 6 and cleansed_count <= 2:
                    return True
            if self.step_count > 500:
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
    load_image('C', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('@',
        f'{base_path}/character_sprite_data/td_monsters_archer_d1.png')
    load_image('#',
        f'{base_path}/character_sprite_data/td_monsters_witch_d1.png')
    load_image('&',
        f'{base_path}/character_sprite_data/td_monsters_goblin_captain_d1.png')
    load_image('M', f'{base_path}/world_tileset_data/td_world_crate.png')
    load_image('R', f'{base_path}/world_tileset_data/td_items_gem_ruby.png')
    load_image('T', f'{base_path}/world_tileset_data/tg_world_floor_moss_e.png'
        )
    load_image('F', f'{base_path}/world_tileset_data/tg_world_barrel.png')
    load_image('E',
        f'{base_path}/character_sprite_data/td_monsters_berserker_d1.png')
    load_image('G',
        f'{base_path}/world_tileset_data/tg_world_floor_panel_steel_c.png')
    load_image('Z', f'{base_path}/world_tileset_data/td_items_amulet_gold.png')
    return env_image, image_paths


def str_map():
    str_world = """BBBBBBBBBBBBBBBBBB
BAAAAAAAAAAAAAAAAAB
BA@OAAXAAAATRFAAAAB
BAAAAAAICAAAAAAAAB
BAAAMMMMAAARRRTAAB
BAA#MMMMAAA&AAAAB
BAAAMMMMAAAAAEAAAB
BAAAGGGGAAAAAAAAAAB
BAAAAAAAAAAAAAAAAAAB
BBBBBBBBBBBBBBBBBB"""
    return str_world


def important_tiles():
    walkables = ['A', 'R', 'T', 'F', 'G', 'Z']
    non_walkables = ['B', 'M']
    interactive_object_tiles = ['O', 'I', 'C']
    collectible_tiles = ['R', 'T', 'F']
    npc_tiles = ['&']
    player_tile = ['@']
    enemy_tiles = ['#', '&', 'E']
    extra_tiles = ['X', 'Z']
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
