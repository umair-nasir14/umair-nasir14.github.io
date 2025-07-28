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
        self.action_space = spaces.Discrete(9)
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
        self.player_health = 60
        self.max_player_health = 60
        self.player_mana = 30
        self.max_player_mana = 30
        self.enemy_health = 100
        self.map = [list(row) for row in self.map_str]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        self.player_tiles = ['X', 'Y']
        self.rare_tiles = []
        self.combinations_made_this_turn = 0
        self.consecutive_combinations = 0
        self.moves_made = 0
        self.last_action_feedback = ''
        self.achievements_unlocked = set()
        self.total_combinations_made = 0
        self.enemies_defeated = 0
        self.secrets_found = 0
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
        self.player_tiles = ['X', 'Y']
        self.rare_tiles = []
        self.combinations_made_this_turn = 0
        self.consecutive_combinations = 0
        self.moves_made = 0
        self.last_action_feedback = ''
        self.current_score = 0
        self.objects_on_target = 0
        self.collected_items = 0
        self.player_health = self.max_player_health
        self.player_mana = self.max_player_mana
        self.enemy_health = 100
        self.total_combinations_made = 0
        self.enemies_defeated = 0
        self.secrets_found = 0
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
                self.last_action_feedback = '🚶 Moved successfully'
                if new_tile == 'C':
                    self.player_mana += 2
                    reward += 2
                    self.last_action_feedback += (
                        ' (+2 mana from combination tile)')
            elif new_tile in ['O', 'I']:
                if random.random() < 0.3:
                    found_items = random.choice(['X', 'Y', 'A', 'R'])
                    self.player_tiles.append(found_items)
                    reward += 4
                    self.last_action_feedback = (
                        f'🎁 Found {found_items} tile in interactive object!')
                else:
                    self.last_action_feedback = (
                        '🔍 Searched object but found nothing this time')
            elif new_tile in ['#', '&']:
                combat_result = self.handle_combat(new_tile)
                reward += combat_result
            elif new_tile == 'S':
                self.update_player_position(new_row, new_col, new_tile)
                self.player_health = min(self.max_player_health, self.
                    player_health + 3)
                reward += 1
                self.last_action_feedback = (
                    '🛡️ Moved to shield tile (+3 health)')
            elif new_tile in ['X', 'Y', 'Z']:
                self.player_tiles.append(new_tile)
                self.map[new_row][new_col] = 'A'
                self.update_player_position(new_row, new_col, 'A')
                reward += 3
                self.last_action_feedback = f'💎 Collected {new_tile} tile!'
            else:
                self.last_action_feedback = f'❌ Cannot move to {new_tile} tile'
        else:
            self.last_action_feedback = (
                '❌ Cannot move outside the map boundaries')
        return reward

    def handle_combat(self, enemy_type):
        """Handle combat encounters with different enemy types."""
        reward = 0
        if enemy_type == '#':
            if self.player_health > 12:
                self.player_health -= 9
                self.enemies_defeated += 1
                reward = 8
                self.last_action_feedback = (
                    '⚔️ Defeated weak enemy! (-9 health)')
                new_row, new_col = self.player_position[0] + (-1 if self.
                    player_position[0] > 0 else 1), self.player_position[1]
                if (0 <= new_row < self.grid_height and 0 <= new_col < self
                    .grid_width):
                    if self.map[new_row][new_col] == '#':
                        self.map[new_row][new_col] = 'A'
            else:
                self.player_health -= 15
                reward = -5
                self.last_action_feedback = (
                    '💀 Badly hurt by enemy! Need more health')
        elif enemy_type == '&':
            if self.player_health > 24 and len(self.rare_tiles) > 0:
                self.player_health -= 15
                self.rare_tiles.pop()
                self.enemies_defeated += 1
                reward = 15
                self.last_action_feedback = (
                    '⚔️ Used rare tile to defeat strong enemy! (-15 health)')
            else:
                self.player_health -= 21
                reward = -8
                self.last_action_feedback = (
                    '💀 Strong enemy overpowered you! Need rare tiles')
        return reward

    def is_terminal(self):
        if self.player_health <= 0:
            self.last_action_feedback = '💀 Game Over: Health depleted!'
            return True
        win_conditions = []
        if 'LEGENDARY' in self.player_tiles:
            win_conditions.append(
                '🌟 LEGENDARY MASTER - You crafted the ultimate power!')
        chests_looted = 0
        for row in self.map:
            chests_looted += row.count('O')
        original_chests = 8
        if original_chests - chests_looted >= 3:
            win_conditions.append(
                "💰 TREASURE HUNTER - You've claimed the ancient riches!")
        if self.consecutive_combinations >= 4:
            win_conditions.append(
                '⛓️ CHAIN MASTER - Unstoppable combination streak!')
        if self.enemies_defeated >= 3 and self.player_health >= 42:
            win_conditions.append('⚔️ COMBAT LEGEND - Warrior supreme!')
        if self.player_mana >= 24 and 'C' in [tile for row in self.map for
            tile in row]:
            win_conditions.append('🔮 MANA WIZARD - Master of magical arts!')
        if self.moves_made <= 20 and 'Z' in self.player_tiles:
            win_conditions.append('⚡ SPEED DEMON - Lightning fast victory!')
        total_resources = len(self.player_tiles) + len(self.rare_tiles)
        if total_resources >= 8:
            win_conditions.append('💎 RESOURCE BARON - Master of abundance!')
        if ('Z' in self.player_tiles and self.player_mana >= 18 and self.
            enemies_defeated <= 1):
            win_conditions.append(
                '🕊️ PEACEFUL SCHOLAR - Knowledge over violence!')
        combination_tiles = sum(row.count('C') for row in self.map)
        shield_tiles = sum(row.count('S') for row in self.map)
        if combination_tiles >= 3 and shield_tiles >= 5:
            win_conditions.append('🏗️ MASTER ARCHITECT - Builder of wonders!')
        if self.player_health <= 18 and self.enemies_defeated >= 2 and len(self
            .rare_tiles) >= 1:
            win_conditions.append('🎲 DAREDEVIL - Victory through courage!')
        if (self.moves_made >= 30 and total_resources >= 6 and self.
            enemies_defeated >= 1):
            win_conditions.append('🗺️ GRAND EXPLORER - Every corner conquered!'
                )
        if (self.player_health >= 36 and self.player_mana >= 15 and 'Z' in
            self.player_tiles and self.enemies_defeated >= 1 and self.
            consecutive_combinations >= 2):
            win_conditions.append('☯️ PERFECT HARMONY - Master of all aspects!'
                )
        player_created_tiles = combination_tiles + shield_tiles
        if player_created_tiles >= 8 and self.enemies_defeated >= 4:
            win_conditions.append('👑 DOMINATION - Ruler of the realm!')
        if hasattr(self, 'min_health_reached'
            ) and self.min_health_reached <= 15 and self.player_health >= 48:
            win_conditions.append('🔥 PHOENIX RISING - From ashes to glory!')
        if not hasattr(self, 'min_health_reached'):
            self.min_health_reached = self.player_health
        else:
            self.min_health_reached = min(self.min_health_reached, self.
                player_health)
        if self.moves_made <= 15 and len(self.player_tiles
            ) >= 5 and random.random() < 0.1:
            win_conditions.append('🍀 LUCKY STRIKE - Fortune favors the bold!')
        unique_tiles = set(self.player_tiles + self.rare_tiles)
        if len(unique_tiles) >= 5:
            win_conditions.append(
                '🏆 MASTER COLLECTOR - Gatherer of all things!')
        if win_conditions:
            victory_priorities = {'LEGENDARY MASTER': 10, 'DOMINATION': 9,
                'PHOENIX RISING': 8, 'PERFECT HARMONY': 7,
                'MASTER COLLECTOR': 6, 'COMBAT LEGEND': 5,
                'TREASURE HUNTER': 5, 'MASTER ARCHITECT': 4, 'CHAIN MASTER':
                4, 'SPEED DEMON': 3, 'MANA WIZARD': 3, 'DAREDEVIL': 3,
                'RESOURCE BARON': 2, 'GRAND EXPLORER': 2,
                'PEACEFUL SCHOLAR': 2, 'LUCKY STRIKE': 1}
            best_victory = win_conditions[0]
            best_priority = 0
            for victory in win_conditions:
                for key, priority in victory_priorities.items():
                    if key in victory:
                        if priority > best_priority:
                            best_priority = priority
                            best_victory = victory
                        break
            self.last_action_feedback = f'🎉 {best_victory}'
            if len(win_conditions) > 1:
                self.last_action_feedback += (
                    f' (Plus {len(win_conditions) - 1} other victories!)')
            return True
        return False

    def get_state(self):
        return {'map': self.map, 'player_tiles': self.player_tiles,
            'rare_tiles': self.rare_tiles, 'player_position': self.
            player_position, 'grid_height': self.grid_height, 'grid_width':
            self.grid_width, 'player_health': self.player_health,
            'player_mana': self.player_mana, 'current_score': self.
            current_score, 'moves_made': self.moves_made,
            'consecutive_combinations': self.consecutive_combinations,
            'last_feedback': self.last_action_feedback,
            'achievements_unlocked': list(self.achievements_unlocked)}

    def step(self, action):
        reward = 0
        self.moves_made += 1
        self.combinations_made_this_turn = 0
        if action < 4:
            reward += self.move_player(action)
        elif action == 4:
            if self.player_mana >= 5:
                reward += self.tile_exchange_bonus()
                self.player_mana -= 5
            else:
                self.last_action_feedback = (
                    '❌ Not enough mana for enhanced exchange!')
        elif action == 5:
            reward += self.tile_exchange()
        elif action == 6:
            if self.player_mana >= 10:
                reward += self.strategic_blast()
                self.player_mana -= 10
            else:
                self.last_action_feedback = (
                    '❌ Not enough mana for strategic blast!')
        elif action == 7:
            if self.player_mana >= 15:
                reward += self.create_combination_field()
                self.player_mana -= 15
            else:
                self.last_action_feedback = (
                    '❌ Not enough mana for combination field!')
        elif action == 8:
            if self.player_mana >= 20:
                reward += self.create_resource_shield()
                self.player_mana -= 20
            else:
                self.last_action_feedback = (
                    '❌ Not enough mana for resource shield!')
        self.player_mana = min(self.max_player_mana, self.player_mana + 1)
        self.check_achievements()
        if self.combinations_made_this_turn > 0:
            self.consecutive_combinations += self.combinations_made_this_turn
            chain_bonus = min(self.consecutive_combinations * 2, 15)
            reward += chain_bonus
            if chain_bonus > 0:
                self.last_action_feedback += f' 🔥 Chain Bonus: +{chain_bonus}!'
        else:
            self.consecutive_combinations = 0
        done = self.is_terminal()
        if done:
            reward += self.calculate_completion_bonus()
        info = {'feedback': self.last_action_feedback, 'mana': self.
            player_mana, 'health': self.player_health}
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
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                tile = self.map[i][j]
                if tile == 'C':
                    tile_img = self.tiles[tile].resize((self.tile_size,
                        self.tile_size))
                    env_img.paste(tile_img, (j * self.tile_size, i * self.
                        tile_size), tile_img)
                elif tile == 'S':
                    tile_img = self.tiles[tile].resize((self.tile_size,
                        self.tile_size))
                    env_img.paste(tile_img, (j * self.tile_size, i * self.
                        tile_size), tile_img)
        frame = np.array(env_img.convert('RGB'))
        self.frames.append(frame)
        return frame

    def get_action_space(self):
        return 9

    def get_mechanics_to_action(self):
        return {'move_up': 0, 'move_down': 1, 'move_left': 2, 'move_right':
            3, 'tile_exchange_bonus': 4, 'tile_exchange': 5,
            'strategic_blast': 6, 'create_combination_field': 7,
            'create_resource_shield': 8}

    def tile_exchange_bonus(self):
        reward = 0
        available_tiles = self.get_available_tiles()
        if 'X' in available_tiles and 'Y' in available_tiles:
            self.player_tiles.remove('X')
            self.player_tiles.remove('Y')
            self.player_tiles.append('Z')
            base_reward = 8
            self.combinations_made_this_turn += 1
            self.last_action_feedback = '✨ Enhanced Tile Exchange! X + Y → Z'
            if '@' in self.player_tiles:
                self.player_tiles.append('A')
                base_reward += 3
                self.last_action_feedback += ' + Bonus tile A!'
            mana_restore = 3
            self.player_mana = min(self.max_player_mana, self.player_mana +
                mana_restore)
            self.last_action_feedback += f' (+{mana_restore} mana)'
            reward = base_reward
        elif self.can_make_rare_combinations():
            reward = self.create_rare_combination()
        else:
            self.last_action_feedback = (
                '❌ Cannot perform enhanced exchange - need X and Y tiles')
        return reward

    def tile_exchange(self):
        reward = 0
        available_tiles = self.get_available_tiles()
        if 'X' in available_tiles and 'Y' in available_tiles:
            self.player_tiles.remove('X')
            self.player_tiles.remove('Y')
            self.player_tiles.append('Z')
            base_reward = 5
            self.combinations_made_this_turn += 1
            self.last_action_feedback = '🔄 Basic Tile Exchange: X + Y → Z'
            reward = base_reward
        elif available_tiles.count('Z') >= 2:
            self.player_tiles.remove('Z')
            self.player_tiles.remove('Z')
            self.player_tiles.append('M')
            self.rare_tiles.append('M')
            reward = 12
            self.combinations_made_this_turn += 1
            self.last_action_feedback = (
                '🌟 Rare Combination: Z + Z → Master Tile M!')
        elif 'A' in available_tiles and 'Z' in available_tiles:
            self.player_tiles.remove('A')
            self.player_tiles.remove('Z')
            self.player_tiles.append('P')
            reward = 9
            self.combinations_made_this_turn += 1
            self.last_action_feedback = (
                '⚡ Power Combination: A + Z → Power Tile P!')
        else:
            self.last_action_feedback = (
                '❌ No valid tile combinations available')
        return reward

    def strategic_blast(self):
        reward = 0
        enemies_hit = 0
        special_tiles_affected = 0
        blast_radius = 2
        player_row, player_col = self.player_position
        self.last_action_feedback = '💥 Strategic Blast activated!'
        for dx in range(-blast_radius, blast_radius + 1):
            for dy in range(-blast_radius, blast_radius + 1):
                if abs(dx) + abs(dy) <= blast_radius:
                    target_row = player_row + dx
                    target_col = player_col + dy
                    if (0 <= target_row < self.grid_height and 0 <=
                        target_col < self.grid_width):
                        tile = self.map[target_row][target_col]
                        if tile == '@':
                            reward += 1
                        elif tile in ['#', '&']:
                            self.map[target_row][target_col] = 'A'
                            enemies_hit += 1
                            reward += 6
                        elif tile in ['A', 'B', 'C', 'D']:
                            special_tiles_affected += 1
                            reward += 2
                        elif tile in ['O', 'I']:
                            if random.random() < 0.2:
                                self.player_tiles.append('R')
                                self.last_action_feedback += (
                                    ' Found hidden resource!')
                                reward += 4
        if enemies_hit > 0:
            self.enemies_defeated += enemies_hit
            self.last_action_feedback += f' Hit {enemies_hit} enemies!'
        if special_tiles_affected > 0:
            self.last_action_feedback += (
                f' Affected {special_tiles_affected} special tiles!')
        if enemies_hit >= 2:
            reward += 8
            self.last_action_feedback += ' 🎯 Multi-target bonus!'
        return reward

    def create_combination_field(self):
        reward = 0
        field_size = 2
        tiles_placed = 0
        synergy_bonus = 0
        self.last_action_feedback = '🔮 Creating Combination Field...'
        for dx in range(-field_size, field_size + 1):
            for dy in range(-field_size, field_size + 1):
                if abs(dx) + abs(dy) <= field_size:
                    new_row = self.player_position[0] + dx
                    new_col = self.player_position[1] + dy
                    if (0 <= new_row < self.grid_height and 0 <= new_col <
                        self.grid_width):
                        current_tile = self.map[new_row][new_col]
                        if current_tile in self.walkable_tiles:
                            self.map[new_row][new_col] = 'C'
                            tiles_placed += 1
                            reward += 1
                        elif current_tile in ['O', 'I']:
                            synergy_bonus += 2
                            self.last_action_feedback += (
                                ' ⚡ Enhanced interactive object!')
        if tiles_placed >= 5:
            effectiveness_bonus = min(tiles_placed * 1, 8)
            reward += effectiveness_bonus
            self.last_action_feedback += (
                f' Large field bonus: +{effectiveness_bonus}!')
        combination_bonus = self.tile_combination_bonus()
        reward += combination_bonus + synergy_bonus
        if tiles_placed > 0:
            self.last_action_feedback += (
                f' Placed {tiles_placed} combination tiles!')
        else:
            self.last_action_feedback = (
                '❌ No space available for combination field!')
        return reward

    def create_resource_shield(self):
        reward = 0
        shield_radius = 3
        shield_tiles_created = 0
        protected_objects = 0
        self.last_action_feedback = '🛡️ Creating Resource Shield...'
        player_row, player_col = self.player_position
        for dx in range(-shield_radius, shield_radius + 1):
            for dy in range(-shield_radius, shield_radius + 1):
                distance = abs(dx) + abs(dy)
                if distance <= shield_radius and distance >= 2:
                    new_row = player_row + dx
                    new_col = player_col + dy
                    if (0 <= new_row < self.grid_height and 0 <= new_col <
                        self.grid_width):
                        current_tile = self.map[new_row][new_col]
                        if current_tile in self.walkable_tiles:
                            self.map[new_row][new_col] = 'S'
                            shield_tiles_created += 1
                            reward += 1
                        elif current_tile in ['O', 'I', 'C']:
                            protected_objects += 1
                            reward += 3
        if shield_tiles_created >= 8:
            completeness_bonus = 8
            reward += completeness_bonus
            self.last_action_feedback += (
                f' Complete shield bonus: +{completeness_bonus}!')
        if protected_objects > 0:
            protection_bonus = protected_objects * 2
            reward += protection_bonus
            self.last_action_feedback += (
                f' Protected {protected_objects} valuable objects!')
        if shield_tiles_created >= 12:
            self.achievements_unlocked.add('Shield Master')
            reward += 15
            self.last_action_feedback += ' 🏆 Achievement: Shield Master!'
        if shield_tiles_created > 0:
            self.last_action_feedback += (
                f' Created {shield_tiles_created} shield tiles!')
        else:
            self.last_action_feedback = (
                '❌ Cannot create shield - no suitable positions!')
        return reward

    def can_make_rare_combinations(self):
        """Check if player can make advanced rare combinations."""
        available = self.get_available_tiles()
        if len(set(available) & {'Z', 'M', 'P'}) >= 2:
            return True
        if 'M' in available and available.count('A') >= 2:
            return True
        return False

    def create_rare_combination(self):
        """Create advanced rare tile combinations."""
        available = self.get_available_tiles()
        reward = 0
        if 'Z' in available and 'M' in available and 'P' in available:
            self.player_tiles.remove('Z')
            self.player_tiles.remove('M')
            self.player_tiles.remove('P')
            self.rare_tiles.append('LEGENDARY')
            self.player_tiles.append('LEGENDARY')
            reward = 25
            self.combinations_made_this_turn += 1
            self.last_action_feedback = (
                '🌟 LEGENDARY COMBINATION! Z + M + P → LEGENDARY TILE!')
            self.achievements_unlocked.add('Legendary Crafter')
        elif 'M' in available and available.count('A') >= 2:
            self.player_tiles.remove('M')
            self.player_tiles.remove('A')
            self.player_tiles.remove('A')
            self.player_tiles.append('ENHANCED_M')
            reward = 15
            self.combinations_made_this_turn += 1
            self.last_action_feedback = (
                '⚡ Master Enhancement: M + A + A → Enhanced Master!')
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
        new_env.action_space = spaces.Discrete(9)
        new_env.observation_space = spaces.Box(low=0, high=1, shape=(len(
            new_env.char_set), len(new_env.map_str), max(len(row) for row in
            new_env.map_str)), dtype=np.int32)
        return new_env

    def get_available_tiles(self):
        return self.player_tiles

    def is_terminal(self):
        if self.player_health <= 0:
            self.last_action_feedback = '💀 Game Over: Health depleted!'
            return True
        win_conditions = []
        exchange_completed = 'Z' in self.player_tiles
        has_rare_tiles = len(self.rare_tiles) > 0
        used_abilities = (self.consecutive_combinations > 0 or self.
            enemies_defeated > 0 or 'C' in [tile for row in self.map for
            tile in row])
        if exchange_completed and has_rare_tiles and used_abilities:
            win_conditions.append('Master Alchemist Victory')
        if self.enemies_defeated >= 2 and self.player_health > 50:
            win_conditions.append('Combat Mastery Victory')
        if 'LEGENDARY' in self.player_tiles:
            win_conditions.append('Legendary Crafter Victory')
        if (self.moves_made <= 25 and self.consecutive_combinations >= 3 and
            self.current_score >= 100):
            win_conditions.append('Efficiency Master Victory')
        if len(self.achievements_unlocked) >= 3:
            win_conditions.append('Achievement Master Victory')
        if win_conditions:
            victory_type = win_conditions[0]
            self.last_action_feedback = f'🏆 VICTORY: {victory_type}!'
            return True
        return False

    def tile_combination_bonus(self):
        """Calculate bonus reward for tile combinations in the combination field"""
        reward = 0
        combination_count = 0
        for row in self.map:
            for tile in row:
                if tile == 'C':
                    combination_count += 1
        if combination_count >= 3:
            reward += 15
        elif combination_count >= 1:
            reward += 5
        return reward

    def calculate_completion_bonus(self):
        """Calculate more realistic completion bonus based on performance."""
        bonus = 15
        if self.moves_made <= 15:
            bonus += 12
            self.last_action_feedback += ' ⚡ Efficiency bonus!'
        elif self.moves_made <= 25:
            bonus += 6
            self.last_action_feedback += ' 🎯 Good efficiency bonus!'
        health_percentage = self.player_health / self.max_player_health
        health_bonus = int(health_percentage * 10)
        bonus += health_bonus
        achievement_bonus = len(self.achievements_unlocked) * 3
        bonus += achievement_bonus
        rare_bonus = len(self.rare_tiles) * 5
        bonus += rare_bonus
        if bonus > 15:
            self.last_action_feedback += (
                f' 🌟 Total completion bonus: +{bonus}!')
        return bonus

    def check_achievements(self):
        """Enhanced achievement checking with meaningful rewards."""
        new_achievements = []
        if (self.total_combinations_made >= 1 and 'First Alchemist' not in
            self.achievements_unlocked):
            self.achievements_unlocked.add('First Alchemist')
            new_achievements.append('🧪 First Alchemist')
        if (self.consecutive_combinations >= 5 and 'Chain Master' not in
            self.achievements_unlocked):
            self.achievements_unlocked.add('Chain Master')
            new_achievements.append('⛓️ Chain Master')
        if (self.enemies_defeated >= 3 and 'Enemy Slayer' not in self.
            achievements_unlocked):
            self.achievements_unlocked.add('Enemy Slayer')
            new_achievements.append('⚔️ Enemy Slayer')
        if self.moves_made <= 20 and self.is_terminal(
            ) and 'Speed Runner' not in self.achievements_unlocked:
            self.achievements_unlocked.add('Speed Runner')
            new_achievements.append('🏃 Speed Runner')
        if len(self.rare_tiles
            ) >= 3 and 'Rare Collector' not in self.achievements_unlocked:
            self.achievements_unlocked.add('Rare Collector')
            new_achievements.append('💎 Rare Collector')
        if new_achievements:
            achievement_text = ' | '.join(new_achievements)
            self.last_action_feedback += f' 🏆 NEW: {achievement_text}!'


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
    load_image('Y',
        f'{base_path}/world_tileset_data/td_world_floor_grass_c.png')
    load_image('Z',
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
    load_image('S', f'{base_path}/world_tileset_data/td_world_crate.png')
    load_image('D',
        f'{base_path}/character_sprite_data/td_monsters_demon_l1.png')
    return env_image, image_paths


def str_map():
    str_world = """BBBBBBBBBBBBBBBBBBBBBBBBBBBBB
BXAA&AAABBBAAAOYAAAABBBOAAXB
BBB&AAABBBAAABBBAAAAAAAXBBB
B#AAAICBAAAXYAAAABICAA&OAAB
BBBXAAABBBAAABBBAAAAAAAAABB
BAAAXAAAAAAAAAAAAAAXYAAAAAB
BAAA#BBAAA&BBBBBAABAAABBAAB
BAAA#OBAABICAAA##ABYACXBAAB
BXAAB##BAABAAAA&##BAY&AABAAB
BAAABBXAAABAAAAABBAAAAABBAAB
BA@AXAAABBBAAAYAAAABBBOAAYB
BBBBBBBBBBBBBBBBBBBBBBBBBBBBB"""
    return str_world


def important_tiles():
    walkables = ['A', 'C']
    non_walkables = ['B']
    interactive_object_tiles = ['O', 'I', 'C']
    collectible_tiles = []
    npc_tiles = []
    player_tile = ['@']
    enemy_tiles = ['#', '&']
    extra_tiles = ['X', 'Y', 'Z', 'S']
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
