import pandas as pd
import numpy as np
import os
import re
import ast
from collections import defaultdict
from tqdm.notebook import tqdm
import pickle

class ActionLogLoader:
    def __init__(self, root_dir, cache_dir="classifiers/data"):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        
        
        try:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_script_dir = os.getcwd()
            
        self.cache_dir = os.path.join(current_script_dir, cache_dir)
        self.cache_file = os.path.join(self.cache_dir, "action_logs_data.pkl")
        
        self.action_data = []

    def _parse_stats_csv(self, file_path):
        """Reads the stats.csv to map Agent_X to their specific LLM model."""
        if not os.path.exists(file_path):
            return {}
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.strip().lower() for c in df.columns]
            
            agent_map = {}
            for _, row in df.iterrows():
                agent_key = row['agent_name'].strip() 
                raw_model = str(row['model_name']).strip()
                
                if "Apertus-70B-Instruct-2509" in raw_model:
                    model_name = "Apertus-70B-Instruct-2509"
                else:
                    model_name = raw_model
                    
                agent_map[agent_key] = model_name
            return agent_map
        except Exception:
            return {}

    def _parse_occupants(self, occ_str):
        """Safely evaluates the occupants string into a Python list."""
        if occ_str in ('None', '[]'):
            return []
        try:
            return ast.literal_eval(occ_str)
        except (ValueError, SyntaxError):
            return []

    def _parse_single_action_log(self, file_path, agent_id, role, model_name):
        """Parses a single agent's action.log into a sequence of moves."""
        history = []
        if not os.path.exists(file_path):
            return history

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f.readlines()]

        current_entry = {}
        in_adjacent = False

        for line in lines:
            if not line:
                continue
                
            if line.startswith("Round"):
                match = re.search(r"Round (\d+)", line)
                if match:
                    current_entry['round'] = int(match.group(1))
                    
            elif line.startswith("Current Location:"):
                match = re.search(r"\[([^\]]+)\]", line)
                if match:
                    current_entry['location'] = match.group(1)
                in_adjacent = False
                
            elif line.startswith("-> Occupants:") and not in_adjacent:
                occ_str = line.split("-> Occupants:")[1].strip()
                current_entry['occupants'] = self._parse_occupants(occ_str)
                
            elif line.startswith("Adjacent Location(s):"):
                in_adjacent = True
                current_entry['adjacent_locations'] = {}
                
            elif in_adjacent and line.startswith("["):
                match = re.search(r"\[([^\]]+)\] -> Occupants: (.*)", line)
                if match:
                    room = match.group(1)
                    occ_str = match.group(2).strip()
                    current_entry['adjacent_locations'][room] = self._parse_occupants(occ_str)
                    
            elif line.startswith("Bodies Seen:"):
                in_adjacent = False
                
            elif "Selected:" in line:
                match = re.search(r"Selected: (.+)", line)
                if match:
                    current_entry['action'] = match.group(1).strip()
                    
            elif line.startswith("======================"):
                if current_entry:
                    current_entry['agent_id'] = agent_id
                    current_entry['role'] = role
                    current_entry['model'] = model_name
                    history.append(current_entry)
                    current_entry = {}

        return history

    def discover_games(self):
        """Finds all valid game directories across the experiment folders."""
        game_paths = []
        for root, dirs, files in tqdm(os.walk(self.root_dir), desc="Scanning Folders"):
            if "experiment_5" in root.lower(): continue
            if 'stats.csv' in files and 'Byz' in dirs and 'Honest' in dirs:
                game_paths.append(root)
        return game_paths

    def _save_to_cache(self):
        """Saves the parsed action data to a pickle file."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        print(f"--- Saving spatial data to cache: {self.cache_file} ---")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.action_data, f)
        print("Save complete.")

    def _load_from_cache(self):
        """Loads the action data from the pickle cache if it exists."""
        if os.path.exists(self.cache_file):
            print(f"--- Cache found at {self.cache_file}. Loading... ---")
            with open(self.cache_file, 'rb') as f:
                self.action_data = pickle.load(f)
            print(f"Loaded spatial data for {len(self.action_data)} games from cache.")
            return True
        return False

    def load_all_actions(self, force_reload=False):
        """
        Main entry point. Loads from cache if available, otherwise parses 
        all game logs and saves to cache.
        """
        if not force_reload and self._load_from_cache():
            return self.action_data

        all_game_paths = self.discover_games()
        self.action_data = []

        print(f"--- Processing Action Logs for {len(all_game_paths)} Games ---")
        for root in tqdm(all_game_paths, desc="Parsing Spatial Data"):
            path_parts = os.path.normpath(root).split(os.sep)
            game_id = path_parts[-1]
            
            stats_path = os.path.join(root, 'stats.csv')
            agent_model_map = self._parse_stats_csv(stats_path)
            
            game_record = {
                'game_id': game_id,
                'Byzantine': [],
                'Honest': []
            }
            
            for sub_root, _, files in os.walk(root):
                if "action.log" in files:
                    agent_id = os.path.basename(sub_root)
                    
                    if agent_id in ["Agent_0", "Agent_1"]:
                        role = "Byzantine"
                    else:
                        role = "Honest"
                        
                    model_name = agent_model_map.get(agent_id, "Unknown")
                    file_path = os.path.join(sub_root, "action.log")
                    agent_history = self._parse_single_action_log(file_path, agent_id, role, model_name)
                    
                    game_record[role].extend(agent_history)
            
            self.action_data.append(game_record)

        self._save_to_cache()
        return self.action_data
    
    
class ActionAnalysis:
    def __init__(self, action_data):
        self.action_data = action_data
        
        # Initialize trackers that hold lists of percentages per game
        def new_tracker():
            return {
                'iso_pcts': [], 'soc_move_pcts': [], 'max_crowd_pcts': [],
                'total_actions': 0, 'total_moves': 0
            }
            
        self.role_stats = defaultdict(new_tracker)
        self.model_stats = defaultdict(new_tracker)
        self.model_role_stats = defaultdict(new_tracker) 
        self.size_stats = defaultdict(new_tracker)
        
        self._process_data()

    @staticmethod
    def normalize_model_name(name):
        clean_name = str(name).split('/')[-1]
        shorthand_map = {
            "Llama-3.3-70B-Instruct": "Llama3.3-70B",
            "DeepSeek-R1-Distill-Llama-70B": "DeepSeek-70B",
            "L3.3-GeneticLemonade-Final-v2-70B": "GeneticLemonade-70B",
            "Hermes-4-70B": "Hermes4-70B",
            "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
            "Qwen3-Next-80B-A3B-Instruct": "Qwen3Next-80B",
            "Apertus-70B-Instruct-2509": "Apertus-70B",
            "Arcee-Nova": "ArceeNova-73B", 
            "Mixtral-8x7B-Instruct-v0.1-upscaled": "MixtralUpscaled-82B",
            "Athene-V2-Chat": "AtheneV2-73B",
            "HyperNova-60B": "HyperNova-60B",
            "Meta-Llama-3-8B-Instruct": "Llama3-8B",
            "Llama-3.1-8B-Instruct": "Llama3.1-8B",
            "Olmo-3-7B-Instruct": "Olmo3-7B",
            "gemma-2-9b-it": "Gemma2-9B",
            "Qwen2-7B-Instruct": "Qwen2-7B",
            "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
            "Qwen3-14B-Instruct": "Qwen3-14B",
            "gpt-oss-20b": "gpt-oss-20B",
            "Apertus-8B-Instruct-2509": "Apertus-8B",
            "Arcee-Agent": "ArceeAgent-8B",
        }
        for key, val in shorthand_map.items():
            if key in clean_name: return val
        return shorthand_map.get(clean_name, clean_name)

    def _get_weight_class(self, norm_name):
        match = re.search(r'(\d+)B', norm_name, re.IGNORECASE)
        if not match: return "Unknown"
        size_val = int(match.group(1))
        return "Small" if size_val <= 30 else "Medium"

    def _process_data(self):
        #  Aggregate data per game for Roles, Models, and Model+Role combinations
        for game in self.action_data:
            game_role = defaultdict(lambda: defaultdict(int))
            game_model = defaultdict(lambda: defaultdict(int))
            game_model_role = defaultdict(lambda: defaultdict(int))

            for role in ['Byzantine', 'Honest']:
                for action in game[role]:
                    raw_model = action.get('model', 'Unknown')
                    norm_model = self.normalize_model_name(raw_model)
                    
                    self._increment_stats(game_role[role], action)
                    self._increment_stats(game_model[norm_model], action)
                    
                    # Track this specific model's performance in this specific role
                    self._increment_stats(game_model_role[(norm_model, role)], action)

            self._append_game_rates(self.role_stats, game_role)
            self._append_game_rates(self.model_stats, game_model)
            self._append_game_rates(self.model_role_stats, game_model_role)

        # Phase 2: Build Size Stats grouped by Role, based on individual model means
        for (model_name, role), stats in self.model_role_stats.items():
            # Lowered the threshold slightly since actions are now split by role
            if stats['total_actions'] < 30: 
                continue

            weight_class = self._get_weight_class(model_name)
            if weight_class == "Unknown":
                continue

            display_role = "Imposter" if role == 'Byzantine' else "Crew"
            category_key = f"{weight_class} {display_role}"

            
            iso_mean = np.mean(stats['iso_pcts']) if stats['iso_pcts'] else 0
            soc_mean = np.mean(stats['soc_move_pcts']) if stats['soc_move_pcts'] else 0
            max_mean = np.mean(stats['max_crowd_pcts']) if stats['max_crowd_pcts'] else 0

            
            self.size_stats[category_key]['iso_pcts'].append(iso_mean)
            self.size_stats[category_key]['soc_move_pcts'].append(soc_mean)
            self.size_stats[category_key]['max_crowd_pcts'].append(max_mean)
            self.size_stats[category_key]['total_actions'] += stats['total_actions']

    def _increment_stats(self, tracker, action):
        tracker['total_actions'] += 1
        if len(action.get('occupants', [])) == 0:
            tracker['isolated_states'] += 1
            
        action_str = action.get('action', '')
        if action_str.startswith('move ->'):
            target_room = action_str.split('move ->')[1].strip()
            adj_locs = action.get('adjacent_locations', {})
            
            if adj_locs:
                adj_counts = {room: len(occ) for room, occ in adj_locs.items()}
                tracker['total_moves'] += 1
                
                if target_room in adj_counts:
                    target_count = adj_counts[target_room]
                    max_adj_count = max(adj_counts.values()) if adj_counts else 0
                    
                    if target_count > 0: tracker['moves_to_crowd'] += 1
                    if target_count > 0 and target_count == max_adj_count: tracker['moves_to_max_crowd'] += 1

    def _append_game_rates(self, main_tracker, game_tracker):
        for key, stats in game_tracker.items():
            total_act = stats['total_actions']
            if total_act > 0:
                iso_pct = (stats['isolated_states'] / total_act) * 100
                main_tracker[key]['iso_pcts'].append(iso_pct)
                main_tracker[key]['total_actions'] += total_act 

            total_moves = stats['total_moves']
            if total_moves > 0:
                soc_move_pct = (stats['moves_to_crowd'] / total_moves) * 100
                max_crowd_pct = (stats['moves_to_max_crowd'] / total_moves) * 100
                main_tracker[key]['soc_move_pcts'].append(soc_move_pct)
                main_tracker[key]['max_crowd_pcts'].append(max_crowd_pct)
                main_tracker[key]['total_moves'] += total_moves

    def _print_table(self, title, stats_dict, sort_method="iso", show_sd=True):
        if not stats_dict: return
            
        max_key_len = max(len(str(k)) for k in stats_dict.keys())
        col1_width = max(max_key_len, len('Category')) + 2 
        
        if show_sd:
            w_iso, w_soc, w_max = 18, 20, 23
            h_iso, h_soc, h_max = "Isolation %", "Social Move %", "Max Crowd Move %"
        else:
            w_iso, w_soc, w_max = 12, 15, 18
            h_iso, h_soc, h_max = "Isolation %", "Social Move %", "Max Crowd Move %"

        total_width = col1_width + 3 + w_iso + 3 + w_soc + 3 + w_max
        
        print("\n" + "=" * total_width)
        print(f"{title:^{total_width}}")
        print("=" * total_width)
        print(f"{'Category':<{col1_width}} | {h_iso:>{w_iso}} | {h_soc:>{w_soc}} | {h_max:>{w_max}}")
        print("-" * total_width)
        
        if sort_method == "iso":
            sorted_keys = sorted(
                stats_dict.keys(), 
                key=lambda k: np.mean(stats_dict[k]['iso_pcts']) if stats_dict[k]['iso_pcts'] else 0, 
                reverse=True
            )
        elif sort_method == "custom_size":
            # Forces the specific grouping order requested
            order = ["Medium Imposter", "Small Imposter", "Medium Crew", "Small Crew"]
            sorted_keys = [k for k in order if k in stats_dict]
        else:
            sorted_keys = sorted(stats_dict.keys())
        
        for key in sorted_keys:
            stats = stats_dict[key]
            
            # Skip if low data count (but never skip our aggregate cohorts)
            is_aggregate = any(k in key for k in ["Small", "Medium"])
            if stats['total_actions'] < 50 and not is_aggregate:
                continue
                
            iso_mean = np.mean(stats['iso_pcts']) if stats['iso_pcts'] else 0
            soc_mean = np.mean(stats['soc_move_pcts']) if stats['soc_move_pcts'] else 0
            max_mean = np.mean(stats['max_crowd_pcts']) if stats['max_crowd_pcts'] else 0
            
            if show_sd:
                iso_std = np.std(stats['iso_pcts']) if stats['iso_pcts'] else 0
                soc_std = np.std(stats['soc_move_pcts']) if stats['soc_move_pcts'] else 0
                max_std = np.std(stats['max_crowd_pcts']) if stats['max_crowd_pcts'] else 0
                
                iso_str = f"{iso_mean:5.2f}% ±{iso_std:5.2f}%"
                soc_str = f"{soc_mean:5.2f}% ±{soc_std:5.2f}%"
                max_str = f"{max_mean:5.2f}% ±{max_std:5.2f}%"
            else:
                iso_str = f"{iso_mean:5.2f}%"
                soc_str = f"{soc_mean:5.2f}%"
                max_str = f"{max_mean:5.2f}%"
            
            print(f"{key:<{col1_width}} | {iso_str:>{w_iso}} | {soc_str:>{w_soc}} | {max_str:>{w_max}}")
            
        print("=" * total_width)

    def print_reports(self):
        """Prints all three analysis tables with customized sorting and SD rules."""
        self._print_table("SPATIAL BEHAVIOR BY ROLE (IMPOSTER VS CREWMATE)", self.role_stats, sort_method="alpha", show_sd=True)
        
        self._print_table("SPATIAL BEHAVIOR BY MODEL WEIGHT CLASS & ROLE", self.size_stats, sort_method="custom_size", show_sd=True)
        
        self._print_table("SPATIAL BEHAVIOR BY SPECIFIC LLM", self.model_stats, sort_method="iso", show_sd=False)


if __name__ == "__main__":

    PROJECT_ROOT = "experiments/"

    action_loader = ActionLogLoader(root_dir=PROJECT_ROOT)
    action_games = action_loader.load_all_actions(force_reload=False)

    analyzer = ActionAnalysis(action_games)
    analyzer.print_reports()

    if action_games:
        first_game = action_games[0]
        print(f"\nAnalyzing Game: {first_game['game_id']}")
        
        if first_game['Byzantine']:
            first_imposter_move = first_game['Byzantine'][0]
            print(f"Agent: {first_imposter_move['agent_id']} ({first_imposter_move['model']})")
            print(f"Round {first_imposter_move['round']} Location: {first_imposter_move['location']}")
            print(f"Action Taken: {first_imposter_move['action']}")