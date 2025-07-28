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
        self.npc_tiles = ['&']
        self.enemy_tiles = ['#']
        self.extra_tiles = extra_tiles
        self.collectible_tiles = collectible_tiles
        self.current_score = 0
        self.objects_on_target = 0
        self.collected_items = 0
        self.player_health = 100
        self.enemy_health = 100
        self.map = [list(row) for row in self.map_str]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    self.player_position = i, j
        self.current_tile = self.default_walkable_tile
        self.allies = []
        self.max_allies = 3
        self.selected_ally = 0
        self.ally_symbol = 'S'
        self.enemies = []
        self.enemy_move_cooldown = 2
        self.turn_counter = 0
        self.ally_health = {}
        self._initialize_enemies()
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
        self.allies = []
        self.selected_ally = 0
        self.ally_health = {}
        self.enemies = []
        self.turn_counter = 0
        self._initialize_enemies()
        return self.get_state()['map']

    def _initialize_enemies(self):
        """Initialize enemy data structures from the map"""
        self.enemies = []
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile in ['#', '&', 'D', 'E', 'Z']:
                    enemy_data = {'pos': (i, j), 'type': tile, 'health':
                        self._get_enemy_health(tile), 'last_move_turn': 0,
                        'damage': self._get_enemy_damage(tile)}
                    self.enemies.append(enemy_data)

    def _get_enemy_health(self, enemy_type):
        """Get health based on enemy type"""
        health_map = {'#': 50, '&': 75, 'D': 100, 'E': 60, 'Z': 30}
        return health_map.get(enemy_type, 50)

    def _get_enemy_damage(self, enemy_type):
        """Get damage based on enemy type"""
        damage_map = {'#': 15, '&': 20, 'D': 25, 'E': 30, 'Z': 10}
        return damage_map.get(enemy_type, 15)

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
        if action < 4:
            self.move_player(action)
        elif action == 4:
            reward += self.summon_ally()
        elif action >= 5 and action <= 8:
            reward += self.move_ally(action - 5)
        elif action == 9:
            reward += self.select_next_ally()
        elif action == 10:
            reward += self.confuse_and_teleport_enemies()
        elif action == 11:
            reward += self.activate_and_combine_resources()
        elif action == 12:
            reward += self.player_attack()
        elif action == 13:
            reward += self.ally_attack()
        elif action == 14:
            reward += self.heal_ally()
        elif action == 15:
            reward += self.defensive_stance()
        self.turn_counter += 1
        reward += self._process_enemy_turn()
        done = self.is_terminal()
        if done:
            if self.player_health > 0:
                reward += 100
            else:
                reward += 0
        info = {}
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
        if self.allies and self.selected_ally < len(self.allies):
            selected_row, selected_col = self.allies[self.selected_ally]
            from PIL import ImageDraw
            draw = ImageDraw.Draw(env_img)
            x1 = selected_col * self.tile_size
            y1 = selected_row * self.tile_size
            x2 = x1 + self.tile_size
            y2 = y1 + self.tile_size
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        frame = np.array(env_img.convert('RGB'))
        self.frames.append(frame)
        return frame

    def get_action_space(self):
        return 16

    def get_mechanics_to_action(self):
        return {'move_player_up': 0, 'move_player_down': 1,
            'move_player_left': 2, 'move_player_right': 3, 'summon_ally': 4,
            'move_ally_up': 5, 'move_ally_down': 6, 'move_ally_left': 7,
            'move_ally_right': 8, 'select_next_ally': 9,
            'confuse_and_teleport_enemies': 10,
            'activate_and_combine_resources': 11, 'player_attack': 12,
            'ally_attack': 13, 'heal_ally': 14, 'defensive_stance': 15}

    def summon_ally(self):
        """Summon an ally at an adjacent empty position"""
        reward = 0
        if len(self.allies) >= self.max_allies:
            return 0
        player_row, player_col = self.player_position
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for dx, dy in adjacency_offsets:
            new_row = player_row + dx
            new_col = player_col + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                base_tile = self.map_without_chars[new_row][new_col]
                current_tile = self.map[new_row][new_col]
                if (base_tile in self.walkable_tiles and (new_row, new_col)
                     not in self.allies and (new_row, new_col) != self.
                    player_position and current_tile not in self.
                    enemy_tiles and current_tile in self.walkable_tiles):
                    ally_pos = new_row, new_col
                    self.allies.append(ally_pos)
                    self.ally_health[ally_pos] = 100
                    self.map[new_row][new_col] = self.ally_symbol
                    reward = 1
                    break
        return reward

    def move_ally(self, direction):
        """Move the currently selected ally"""
        reward = 0
        if not self.allies:
            return 0
        if self.selected_ally >= len(self.allies):
            self.selected_ally = 0
        ally_row, ally_col = self.allies[self.selected_ally]
        moves = {(0): (-1, 0), (1): (1, 0), (2): (0, -1), (3): (0, 1)}
        dx, dy = moves[direction]
        new_row = ally_row + dx
        new_col = ally_col + dy
        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            base_tile = self.map_without_chars[new_row][new_col]
            current_tile = self.map[new_row][new_col]
            if base_tile in self.walkable_tiles and (new_row, new_col
                ) not in self.allies and (new_row, new_col
                ) != self.player_position and current_tile not in self.enemy_tiles:
                old_base_tile = self.map_without_chars[ally_row][ally_col]
                self.map[ally_row][ally_col] = old_base_tile
                self.allies[self.selected_ally] = new_row, new_col
                self.map[new_row][new_col] = self.ally_symbol
                if base_tile in self.interactive_object_tiles:
                    reward = 5
                elif base_tile in ['R', 'F', 'W']:
                    reward = 2
        return reward

    def select_next_ally(self):
        """Select the next ally for movement"""
        if self.allies:
            self.selected_ally = (self.selected_ally + 1) % len(self.allies)
            return 0
        return 0

    def get_ally_status(self):
        """Get information about current ally status"""
        status = {'total_allies': len(self.allies), 'max_allies': self.
            max_allies, 'selected_ally': self.selected_ally,
            'ally_positions': self.allies.copy()}
        return status

    def remove_ally(self, ally_index):
        """Remove an ally (for when they're defeated or dismissed)"""
        if 0 <= ally_index < len(self.allies):
            ally_row, ally_col = self.allies[ally_index]
            original_tile = self.map_without_chars[ally_row][ally_col]
            self.map[ally_row][ally_col] = original_tile
            self.allies.pop(ally_index)
            if self.selected_ally >= len(self.allies) and self.allies:
                self.selected_ally = len(self.allies) - 1
            elif not self.allies:
                self.selected_ally = 0

    def confuse_and_teleport_enemies(self):
        reward = 0
        enemy_positions = []
        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] in self.enemy_tiles:
                    enemy_positions.append((row, col))
        if enemy_positions:
            enemies_confused = 0
            for enemy_row, enemy_col in enemy_positions:
                direction = random.choice(['up', 'down', 'left', 'right'])
                teleport_possible = False
                new_enemy_row, new_enemy_col = enemy_row, enemy_col
                if direction == 'up' and enemy_row > 0:
                    new_enemy_row -= 1
                elif direction == 'down' and enemy_row < len(self.map) - 1:
                    new_enemy_row += 1
                elif direction == 'left' and enemy_col > 0:
                    new_enemy_col -= 1
                elif direction == 'right' and enemy_col < len(self.map[0]) - 1:
                    new_enemy_col += 1
                distance_to_player = abs(enemy_row - self.player_position[0]
                    ) + abs(enemy_col - self.player_position[1])
                if distance_to_player <= 2:
                    enemies_confused += 1
            if enemies_confused >= 2:
                reward = 1
        return reward

    def activate_and_combine_resources(self):
        """Activates a signature ability to gather resources and create barricades in the environment."""
        reward = 0
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        resource_tiles = ['R', 'F', 'W']
        adjacent_resources = 0
        adjacent_objects = 0
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                adjacent_tile = self.map[new_row][new_col]
                if adjacent_tile in resource_tiles:
                    adjacent_resources += 1
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                adjacent_tile = self.map[new_row][new_col]
                if adjacent_tile in self.interactive_object_tiles:
                    adjacent_objects += 1
        if adjacent_resources >= 2 and adjacent_objects >= 1:
            reward = 5
        return reward

    def CheckTileColor(self):
        tile_color = self.get_current_tile_color()
        player_status = self.get_player_status()
        if tile_color == 'R':
            return -1
        else:
            return 0

    def collaborative_tile_enrichment(self):
        reward = 0
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        resource_tiles = ['R', 'F', 'W']
        nearby_resources = 0
        nearby_objects = 0
        nearby_allies = 0
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ):
                tile = self.map[new_row][new_col]
                if tile in resource_tiles:
                    nearby_resources += 1
                elif tile in self.interactive_object_tiles:
                    nearby_objects += 1
                elif tile == self.ally_symbol:
                    nearby_allies += 1
        if (nearby_resources >= 2 and nearby_objects >= 1 and nearby_allies >=
            2):
            reward = 10
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

    def get_current_tile(self):
        return self.map[self.player_position[0]][self.player_position[1]]

    def get_current_tile_color(self):
        current_tile = self.current_tile
        tile_colors = {'A': 'G', 'O': 'Y', 'I': 'Y', 'C': 'Y', 'R': 'R',
            'F': 'G', 'W': 'G', 'B': 'B', '#': 'R', '&': 'R', 'D': 'R', 'E':
            'R', 'X': 'N', 'Z': 'R', 'S': 'B'}
        return tile_colors.get(current_tile, 'N')

    def get_player_status(self):
        return '@' if self.map[self.player_position[0]][self.player_position[1]
            ] == '@' else None

    def is_terminal(self):
        if self.player_health <= 0:
            return True
        current_tile = self.current_tile
        if current_tile in self.interactive_object_tiles:
            return True
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        enemy_count = 0
        for dx, dy in adjacency_offsets:
            new_row = self.player_position[0] + dx
            new_col = self.player_position[1] + dy
            if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]
                ) and self.map[new_row][new_col] in self.enemy_tiles:
                enemy_count += 1
        if enemy_count >= 3:
            return True
        return False

    def _process_enemy_turn(self):
        """Process all enemy actions for this turn"""
        reward = 0
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                self._remove_enemy(enemy)
                continue
            if self.turn_counter - enemy['last_move_turn'
                ] >= self.enemy_move_cooldown:
                enemy['last_move_turn'] = self.turn_counter
                if enemy['type'] == 'Z':
                    reward += self._archer_behavior(enemy)
                elif enemy['type'] == 'E':
                    reward += self._berserker_behavior(enemy)
                elif enemy['type'] == 'D':
                    reward += self._demon_behavior(enemy)
                else:
                    reward += self._default_enemy_behavior(enemy)
        return reward

    def _archer_behavior(self, enemy):
        """Archer tries to maintain distance and shoot"""
        reward = 0
        enemy_pos = enemy['pos']
        targets = [self.player_position] + self.allies
        for target_pos in targets:
            distance = abs(enemy_pos[0] - target_pos[0]) + abs(enemy_pos[1] -
                target_pos[1])
            if 2 <= distance <= 3 and self._has_line_of_sight(enemy_pos,
                target_pos):
                reward += self._enemy_attack(enemy, target_pos, ranged=True)
                return reward
        reward += self._move_enemy_smart(enemy, maintain_distance=True)
        return reward

    def _berserker_behavior(self, enemy):
        """Berserker moves aggressively toward targets"""
        reward = 0
        enemy_pos = enemy['pos']
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        targets = [self.player_position] + self.allies
        for dx, dy in adjacency_offsets:
            attack_pos = enemy_pos[0] + dx, enemy_pos[1] + dy
            if attack_pos in targets:
                reward += self._enemy_attack(enemy, attack_pos)
                return reward
        reward += self._move_enemy_aggressive(enemy)
        return reward

    def _demon_behavior(self, enemy):
        """Demon uses smart tactics and powerful attacks"""
        reward = 0
        enemy_pos = enemy['pos']
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        targets = [self.player_position] + self.allies
        for dx, dy in adjacency_offsets:
            attack_pos = enemy_pos[0] + dx, enemy_pos[1] + dy
            if attack_pos in targets:
                reward += self._enemy_attack(enemy, attack_pos)
                return reward
        reward += self._move_enemy_smart(enemy)
        return reward

    def _default_enemy_behavior(self, enemy):
        """Default enemy behavior - basic movement toward player"""
        reward = 0
        enemy_pos = enemy['pos']
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        targets = [self.player_position] + self.allies
        for dx, dy in adjacency_offsets:
            attack_pos = enemy_pos[0] + dx, enemy_pos[1] + dy
            if attack_pos in targets:
                reward += self._enemy_attack(enemy, attack_pos)
                return reward
        reward += self._move_enemy_toward_target(enemy, self.player_position)
        return reward

    def _has_line_of_sight(self, start_pos, end_pos):
        """Check if there's a clear line of sight between two positions"""
        start_row, start_col = start_pos
        end_row, end_col = end_pos
        row_diff = end_row - start_row
        col_diff = end_col - start_col
        if row_diff == 0:
            step = 1 if col_diff > 0 else -1
            for col in range(start_col + step, end_col, step):
                if self.map[start_row][col] in ['B']:
                    return False
        elif col_diff == 0:
            step = 1 if row_diff > 0 else -1
            for row in range(start_row + step, end_row, step):
                if self.map[row][start_col] in ['B']:
                    return False
        return True

    def _enemy_attack(self, enemy, target_pos, ranged=False):
        """Enemy attacks target at given position"""
        reward = 0
        damage = enemy['damage']
        if target_pos == self.player_position:
            self.player_health -= damage
            reward = -5
            if ranged:
                print(
                    f"Player hit by {enemy['type']} ranged attack for {damage} damage! Health: {self.player_health}"
                    )
            else:
                print(
                    f"Player hit by {enemy['type']} for {damage} damage! Health: {self.player_health}"
                    )
        elif target_pos in self.allies:
            ally_index = self.allies.index(target_pos)
            if target_pos not in self.ally_health:
                self.ally_health[target_pos] = 100
            self.ally_health[target_pos] -= damage
            reward = -2
            if ranged:
                print(
                    f"Ally at {target_pos} hit by {enemy['type']} ranged attack for {damage} damage! Health: {self.ally_health[target_pos]}"
                    )
            else:
                print(
                    f"Ally at {target_pos} hit by {enemy['type']} for {damage} damage! Health: {self.ally_health[target_pos]}"
                    )
            if self.ally_health[target_pos] <= 0:
                self.remove_ally(ally_index)
                print(f'Ally at {target_pos} has been defeated!')
                reward = -10
        return reward

    def _move_enemy_toward_target(self, enemy, target_pos):
        """Move enemy one step toward target"""
        reward = 0
        enemy_pos = enemy['pos']
        enemy_row, enemy_col = enemy_pos
        target_row, target_col = target_pos
        row_diff = target_row - enemy_row
        col_diff = target_col - enemy_col
        move_row, move_col = 0, 0
        if abs(row_diff) > abs(col_diff):
            move_row = 1 if row_diff > 0 else -1
        else:
            move_col = 1 if col_diff > 0 else -1
        new_row = enemy_row + move_row
        new_col = enemy_col + move_col
        if self._is_valid_enemy_move(enemy_pos, (new_row, new_col)):
            self._execute_enemy_move(enemy, (new_row, new_col))
        return reward

    def _move_enemy_aggressive(self, enemy):
        """Move enemy aggressively toward closest target"""
        reward = 0
        enemy_pos = enemy['pos']
        targets = [self.player_position] + self.allies
        closest_target = min(targets, key=lambda t: abs(enemy_pos[0] - t[0]
            ) + abs(enemy_pos[1] - t[1]))
        return self._move_enemy_toward_target(enemy, closest_target)

    def _move_enemy_smart(self, enemy, maintain_distance=False):
        """Smart enemy movement with tactical considerations"""
        reward = 0
        enemy_pos = enemy['pos']
        if maintain_distance:
            target_pos = self.player_position
            distance = abs(enemy_pos[0] - target_pos[0]) + abs(enemy_pos[1] -
                target_pos[1])
            if distance < 2:
                row_diff = enemy_pos[0] - target_pos[0]
                col_diff = enemy_pos[1] - target_pos[1]
                move_row = 1 if row_diff > 0 else -1 if row_diff < 0 else 0
                move_col = 1 if col_diff > 0 else -1 if col_diff < 0 else 0
                new_pos = enemy_pos[0] + move_row, enemy_pos[1] + move_col
                if self._is_valid_enemy_move(enemy_pos, new_pos):
                    self._execute_enemy_move(enemy, new_pos)
            elif distance > 3:
                return self._move_enemy_toward_target(enemy, target_pos)
        else:
            return self._move_enemy_toward_target(enemy, self.player_position)
        return reward

    def _is_valid_enemy_move(self, current_pos, new_pos):
        """Check if enemy move is valid"""
        new_row, new_col = new_pos
        if not (0 <= new_row < len(self.map) and 0 <= new_col < len(self.
            map[0])):
            return False
        current_tile = self.map[new_row][new_col]
        if (current_tile not in self.walkable_tiles or new_pos == self.
            player_position or new_pos in self.allies or any(e['pos'] ==
            new_pos for e in self.enemies)):
            return False
        return True

    def _execute_enemy_move(self, enemy, new_pos):
        """Execute enemy movement"""
        old_pos = enemy['pos']
        old_row, old_col = old_pos
        new_row, new_col = new_pos
        original_tile = self.map_without_chars[old_row][old_col]
        self.map[old_row][old_col] = original_tile
        self.map[new_row][new_col] = enemy['type']
        enemy['pos'] = new_pos

    def _remove_enemy(self, enemy):
        """Remove defeated enemy from game"""
        enemy_pos = enemy['pos']
        enemy_row, enemy_col = enemy_pos
        original_tile = self.map_without_chars[enemy_row][enemy_col]
        self.map[enemy_row][enemy_col] = original_tile
        if enemy in self.enemies:
            self.enemies.remove(enemy)

    def player_attack(self):
        """Player attacks adjacent enemies"""
        reward = 0
        player_row, player_col = self.player_position
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        enemies_attacked = 0
        for dx, dy in adjacency_offsets:
            attack_row = player_row + dx
            attack_col = player_col + dy
            attack_pos = attack_row, attack_col
            for enemy in self.enemies:
                if enemy['pos'] == attack_pos:
                    damage = 25
                    enemy['health'] -= damage
                    enemies_attacked += 1
                    print(
                        f"Player attacks {enemy['type']} for {damage} damage! Enemy health: {enemy['health']}"
                        )
                    if enemy['health'] <= 0:
                        print(f"{enemy['type']} defeated!")
                        reward += 10
                    else:
                        reward += 2
        if enemies_attacked == 0:
            reward = -1
        return reward

    def ally_attack(self):
        """Selected ally attacks adjacent enemies"""
        reward = 0
        if not self.allies or self.selected_ally >= len(self.allies):
            return -1
        ally_pos = self.allies[self.selected_ally]
        ally_row, ally_col = ally_pos
        adjacency_offsets = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        enemies_attacked = 0
        for dx, dy in adjacency_offsets:
            attack_row = ally_row + dx
            attack_col = ally_col + dy
            attack_pos = attack_row, attack_col
            for enemy in self.enemies:
                if enemy['pos'] == attack_pos:
                    damage = 20
                    enemy['health'] -= damage
                    enemies_attacked += 1
                    print(
                        f"Ally at {ally_pos} attacks {enemy['type']} for {damage} damage! Enemy health: {enemy['health']}"
                        )
                    if enemy['health'] <= 0:
                        print(f"{enemy['type']} defeated by ally!")
                        reward += 8
                    else:
                        reward += 1
        if enemies_attacked == 0:
            reward = -1
        return reward

    def heal_ally(self):
        """Heal the selected ally"""
        reward = 0
        if not self.allies or self.selected_ally >= len(self.allies):
            return -1
        ally_pos = self.allies[self.selected_ally]
        if ally_pos in self.ally_health:
            old_health = self.ally_health[ally_pos]
            self.ally_health[ally_pos] = min(100, self.ally_health[ally_pos
                ] + 30)
            if old_health < self.ally_health[ally_pos]:
                heal_amount = self.ally_health[ally_pos] - old_health
                print(
                    f'Ally at {ally_pos} healed for {heal_amount} HP! Health: {self.ally_health[ally_pos]}'
                    )
                reward = 1
            else:
                print(f'Ally at {ally_pos} is already at full health!')
                reward = -1
        return reward

    def defensive_stance(self):
        """Enter defensive stance to reduce incoming damage"""
        reward = 0
        self.player_health = min(100, self.player_health + 10)
        print(
            f'Player enters defensive stance! Health boosted to: {self.player_health}'
            )
        return 1


from PIL import Image


def env_dict():
    env_image = dict()
    image_paths = dict()

    def load_image(char, path):
        try:
            env_image[char] = Image.open(path).convert('RGBA')
            image_paths[char] = path
        except FileNotFoundError:
            img = Image.new('RGBA', (32, 32), (128, 128, 128, 255))
            env_image[char] = img
            image_paths[char] = f'placeholder_for_{char}'
    base_path = 'C:/Users/DELL/Projects/Research/gmd'
    load_image('A',
        f'{base_path}/world_tileset_data/td_world_floor_grass_c.png')
    load_image('B',
        f'{base_path}/world_tileset_data/td_world_wall_stone_h_a.png')
    load_image('O', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('I', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('C', f'{base_path}/world_tileset_data/td_world_chest.png')
    load_image('@',
        f'{base_path}/character_sprite_data/td_monsters_archer_d1.png')
    load_image('#',
        f'{base_path}/character_sprite_data/td_monsters_witch_d1.png')
    load_image('&',
        f'{base_path}/character_sprite_data/td_monsters_goblin_captain_d1.png')
    load_image('R', f'{base_path}/world_tileset_data/td_items_gem_ruby.png')
    load_image('F', f'{base_path}/world_tileset_data/td_world_crate.png')
    load_image('W', f'{base_path}/world_tileset_data/tg_world_barrel.png')
    load_image('D',
        f'{base_path}/character_sprite_data/td_monsters_demon_l1.png')
    load_image('E',
        f'{base_path}/character_sprite_data/td_monsters_berserker_d1.png')
    load_image('X', f'{base_path}/world_tileset_data/tg_world_floor_sand_f.png'
        )
    load_image('Z',
        f'{base_path}/character_sprite_data/td_monsters_archer_u2.png')
    load_image('S',
        f'{base_path}/character_sprite_data/td_monsters_archer_d1.png')
    return env_image, image_paths


def str_map():
    str_world = """AAAXXXXXXXXXAAAAAA
AWAF@RFWAXAAZAAAEA
AAAORAAAXAAAAAFWAA
XAAAAAICAAAXAAAAAX
AAA#AAWFAAA&ADARAA
AAAAXAAAAAXAAAAAAA
AAFWAAAXAAAAAOAAAA
AAAAAAXXXXXAAAAEAA"""
    return str_world


def important_tiles():
    walkables = ['A', 'X', 'R', 'F', 'W']
    non_walkables = ['B']
    interactive_object_tiles = ['O', 'I', 'C']
    collectible_tiles = []
    npc_tiles = []
    player_tile = ['@']
    enemy_tiles = ['#', '&', 'D', 'E']
    extra_tiles = []
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
