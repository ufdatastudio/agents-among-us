import json
import os
import re
import warnings
import ast
import pickle
import statistics
import concurrent
import multiprocessing as mp
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import xgboost as xgb
import matplotlib.pyplot as plt
from config.prompts import SUSPECT_JUDGE_SYSTEM, SUSPECT_JUDGE_USER
os.environ["LLM_MODE"] = "CONTROLLER"
from dotenv import load_dotenv
load_dotenv()
import torch
import scipy.stats as stats
from core.stopwords import ENGLISH_STOP_WORDS
warnings.filterwarnings('ignore')

def _gpu_evaluation_worker(gpu_id, tasks_chunk, eval_model_id, batch_size, base_checkpoint_name):
    """
    Independent worker process that locks itself to a specific GPU, 
    loads a full replica of the model, and chews through its chunk of tasks.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    checkpoint_file = base_checkpoint_name.replace(".csv", f"_gpu_{gpu_id}.csv")
    
    completed_tasks = set()
    raw_eval_logs = []
    
    if os.path.exists(checkpoint_file):
        try:
            df_prev = pd.read_csv(checkpoint_file)
            for _, row in df_prev.iterrows():
                sig = (str(row['game_id']), str(row['agent_id']), int(row['round_1']), int(row['round_2']))
                completed_tasks.add(sig)
            raw_eval_logs = df_prev.to_dict('records')
        except Exception:
            pass

    # Filter out already completed tasks for this specific chunk
    remaining_tasks = [t for t in tasks_chunk if (t['g_id'], t['a_id'], t['r1'], t['r2']) not in completed_tasks]
    
    if not remaining_tasks:
        return raw_eval_logs

    # Load tokenizer and quantizer with 4-bit q
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(eval_model_id, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        eval_model_id,
        quantization_config=quantization_config,
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Batch Evaluation Loop
    SAVE_INTERVAL = 100 
    completed_this_session = 0

    for i in tqdm(range(0, len(remaining_tasks), batch_size), desc=f"GPU {gpu_id}", position=gpu_id):
        batch_tasks = remaining_tasks[i:i + batch_size]
        batch_prompts = [t['prompt'] for t in batch_tasks]
        
        try:
            inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[:, input_length:]
            responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for task, raw_text in zip(batch_tasks, responses):
                raw_text = raw_text.strip()
                try:
                    clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                    eval_data = json.loads(clean_json)
                    
                    result = {
                        'game_id': task['g_id'], 'agent_id': task['a_id'],
                        'round_1': task['r1'], 'round_2': task['r2'],
                        'vote_1': task['vote1'], 'vote_2': task['vote2'],
                        'target_changed': eval_data.get("target_changed"),
                        'justified': eval_data.get("justification_provided"),
                        'unjustified_reversal': eval_data.get("unjustified_reversal"),
                        'raw_llm_response': raw_text,
                        'parse_failed': False
                    }
                except json.JSONDecodeError:
                    result = {
                        'game_id': task['g_id'], 'agent_id': task['a_id'],
                        'round_1': task['r1'], 'round_2': task['r2'],
                        'raw_llm_response': raw_text,
                        'parse_failed': True
                    }
                
                raw_eval_logs.append(result)
                completed_this_session += 1
            
            if completed_this_session >= SAVE_INTERVAL:
                pd.DataFrame(raw_eval_logs).to_csv(checkpoint_file, index=False)
                completed_this_session = 0

        except Exception as e:
            tqdm.write(f"\n[GPU {gpu_id} Error] Batch failed: {e}")
            continue

    pd.DataFrame(raw_eval_logs).to_csv(checkpoint_file, index=False)
    return raw_eval_logs


class GameLogLoader:
    def __init__(self, root_dir, cache_dir="classifiers/data"):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(current_script_dir, cache_dir)
        self.cache_file = os.path.join(self.cache_dir, "active_games_data.pkl")
        self.games_data = []      
        self.silent_games = []    

    def _parse_stats_csv(self, file_path):
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
                
                agent_map[agent_key] = {
                    'model': model_name,
                    'role': row.get('alignment', 'Unknown'), 
                    'won': row['won_game']
                }
            return agent_map
        except Exception:
            return None

    def _parse_round_results(self, file_path):
        results = {}
        if not os.path.exists(file_path):
            return results

        current_round = 0
        round_header = re.compile(r"=== Round (\d+) ===")
        votes_received = re.compile(r"Votes Received: (.+)")
        ejection = re.compile(r"EJECTION: (.+) was ejected")

        round_data = {'tally': {}, 'ejected': "None", 'consensus': 0.0}

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                r_match = round_header.search(line)
                if r_match:
                    if current_round > 0:
                        results[current_round] = round_data
                    current_round = int(r_match.group(1))
                    round_data = {'tally': {}, 'ejected': "None", 'consensus': 0.0}
                    continue

                v_match = votes_received.search(line)
                if v_match:
                    try:
                        tally = ast.literal_eval(v_match.group(1))
                        round_data['tally'] = tally
                        if tally:
                            total_votes = sum(tally.values())
                            max_votes = max(tally.values())
                            round_data['consensus'] = max_votes / total_votes if total_votes > 0 else 0.0
                        else:
                            round_data['consensus'] = 0.0
                    except:
                        round_data['tally'] = {}
                        round_data['consensus'] = 0.0
                
                e_match = ejection.search(line)
                if e_match:
                    round_data['ejected'] = e_match.group(1)

            if current_round > 0:
                results[current_round] = round_data
        
        return results

    def _find_and_parse_votes(self, game_root):
        agent_votes = defaultdict(dict)
        vote_pattern = re.compile(r"Round (\d+): Voted for (.+)")

        for root, dirs, files in os.walk(game_root):
            if "vote.log" in files:
                agent_id = os.path.basename(root)
                with open(os.path.join(root, "vote.log"), 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        match = vote_pattern.search(line)
                        if match:
                            r_num = int(match.group(1))
                            target = match.group(2).strip()
                            agent_votes[agent_id][r_num] = target
        return agent_votes

    def _parse_discussion_log(self, file_path, agent_map, round_results, all_agent_votes):
        discussions = []
        if not os.path.exists(file_path):
            return []

        current_round = 0
        current_reporter = None
        statement_counts = defaultdict(int)
        
        round_pattern = re.compile(r"=== Round (\d+) ===")
        meeting_pattern = re.compile(r"\*\* MEETING CALLED by (Agent_\d+)")
        talk_pattern = re.compile(r"^(Agent_\d+):\s*(.+)$")
        
        current_agent = None
        current_text = []
        
        def save_turn():
            if current_agent is not None:
                meta = agent_map.get(current_agent, {})
                agent_role = meta.get('role', 'Unknown')
                my_vote_target = all_agent_votes.get(current_agent, {}).get(current_round, "None")
                
                vote_is_correct = False
                target_role = "Unknown"
                if my_vote_target not in ["None", "SKIP"]:
                    target_meta = agent_map.get(my_vote_target, {})
                    target_role = target_meta.get('role', 'Unknown')
                    if agent_role == 'H' and target_role == 'B':
                        vote_is_correct = True
                    if agent_role == 'B' and target_role == 'H':
                        vote_is_correct = True
                        
                r_res = round_results.get(current_round, {'tally': {}, 'ejected': "None", 'consensus': 0.0})
                
                statement_counts[current_agent] += 1
                s_num = min(statement_counts[current_agent], 2)
                
                is_reporter = 1 if (current_agent == current_reporter and s_num == 1) else 0
                
                discussions.append({
                    'round': current_round,
                    'agent': current_agent,
                    'model': meta.get('model', 'Unknown'),
                    'role': agent_role,
                    'won': meta.get('won', 0),
                    'text': " ".join(current_text),
                    'vote_target': my_vote_target,
                    'vote_target_role': target_role,
                    'vote_correct': vote_is_correct,
                    'round_tally': r_res['tally'],
                    'round_consensus': r_res['consensus'],
                    'round_ejected': r_res['ejected'],
                    'reported': is_reporter,
                    'statement_num': s_num
                })

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                r_match = round_pattern.search(line)
                if r_match:
                    save_turn()
                    current_agent = None
                    current_text = []
                    current_round = int(r_match.group(1))
                    statement_counts.clear()
                    current_reporter = None
                    continue
                    
                m_match = meeting_pattern.search(line)
                if m_match:
                    current_reporter = m_match.group(1)
                    continue

                t_match = talk_pattern.match(line)
                if t_match:
                    save_turn() 
                    current_agent = t_match.group(1)
                    current_text = [t_match.group(2)]
                else:
                    if current_agent and not line.startswith("**"):
                        current_text.append(line)
                        
        save_turn() 
        return discussions

    def discover_games(self):
        print(f"--- Scanning directory structure in: {self.root_dir} ---")
        game_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            if "experiment_5" in root.lower(): continue
            if 'stats.csv' in files:
                game_paths.append(root)
        print(f"Found {len(game_paths)} potential games (excluding Experiment 5).")
        return game_paths

    def _save_to_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        data = {'active': self.games_data, 'silent': self.silent_games}
        print(f"--- Saving enriched data to cache: {self.cache_file} ---")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        print("Save complete.")

    def _load_from_cache(self):
        if os.path.exists(self.cache_file):
            print(f"--- Cache found at {self.cache_file}. Loading... ---")
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.games_data = data['active']
                self.silent_games = data['silent']
            print(f"Loaded {len(self.games_data)} active games and {len(self.silent_games)} silent games from cache.")
            return True
        return False

    def load_all(self, force_reload=False):
        if not force_reload and self._load_from_cache():
            return self.games_data, self.silent_games

        all_game_paths = self.discover_games()
        print(f"--- Processing Logs (Parsing Votes & Round Results) ---")
        for root in tqdm(all_game_paths, desc="Loading Games", unit="game"):
            path_parts = os.path.normpath(root).split(os.sep)
            game_id = path_parts[-1]
            composition_id = path_parts[-2]
            
            exp_id = next((part for part in path_parts if "experiment" in part.lower()), "Unknown")

            agent_map = self._parse_stats_csv(os.path.join(root, 'stats.csv'))
            if not agent_map: continue

            round_results = self._parse_round_results(os.path.join(root, 'roundResults.log'))
            avg_game_consensus = sum(r['consensus'] for r in round_results.values()) / len(round_results) if round_results else 0.0

            all_agent_votes = self._find_and_parse_votes(root)
            turns = self._parse_discussion_log(os.path.join(root, 'discussion.log'), agent_map, round_results, all_agent_votes)
            
            if not turns:
                self.silent_games.append({'experiment': exp_id, 'composition': composition_id, 'game_id': game_id})
            else:
                self.games_data.append({
                    'experiment_id': exp_id,
                    'composition_id': composition_id,
                    'game_id': game_id,
                    'game_consensus': avg_game_consensus,
                    'turns': turns,
                    'discussion_count': len(set(t['round'] for t in turns))
                })
        
        self._save_to_cache()
        return self.games_data, self.silent_games

class DatasetBuilder:
    def __init__(self):
        self.stop_words = ENGLISH_STOP_WORDS
        self.locations = [
            "Reactor", "Security", "UpperEngine", "LowerEngine", "MedBay", 
            "Cafeteria", "Electrical", "Storage", "Admin", "Weapons", 
            "Shields", "O2", "Navigation", "Communications"
        ]
        self.loc_pattern = re.compile(r'\b(?:' + '|'.join(self.locations) + r')\b', flags=re.IGNORECASE)
        self.agent_pattern = re.compile(r'\bagent_\d+\b', flags=re.IGNORECASE)
        self.groups = {
            'experiment_1': ('Heavyweight', 'Homogenous'),
            'experiment_2': ('Heavyweight', 'Heterogenous'),
            'experiment_3': ('Lightweight', 'Homogenous'),
            'experiment_4': ('Lightweight', 'Heterogenous')
        }

    def _preprocess_text(self, text):
        text = text.lower()
        text = self.loc_pattern.sub('place', text)
        text = self.agent_pattern.sub('agent_x', text)
        text = re.sub(r'[^a-z0-9\s_]', '', text)
        tokens = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)

    def build(self, active_games, save_path="observer_dataset.csv"):
        if os.path.exists("virtual_observer_dataset.csv"):
            print("Loading Virtual Observer dataset from 'virtual_observer_dataset.csv'")
            return pd.read_csv("virtual_observer_dataset.csv")

        dataset = []
        for game in active_games:
            exp_id = game['experiment_id'].lower()
            exp_key = next((k for k in self.groups.keys() if k in exp_id), None)
            
            if not exp_key: continue
                
            weight_class, composition = self.groups[exp_key]
            
            for turn in game['turns']:
                if 'olmo' in turn['model'].lower(): continue
                    
                clean_text = self._preprocess_text(turn['text'])
                dataset.append({
                    'Game_ID': game['game_id'],
                    'Model_Name': turn['model'],
                    'Round': turn['round'],
                    'Agent': turn['agent'],
                    'Text': clean_text,
                    'Reported': turn['reported'],
                    'Statement_Num': turn['statement_num'],
                    'Composition': composition,
                    'WeightClass': weight_class,
                    'Role': turn['role']
                })
                
        df = pd.DataFrame(dataset)
        df['Text'] = df['Text'].replace('', pd.NA)    
        df = df.dropna(subset=['Text'])  
        df.to_csv(save_path, index=False)
        return df

class GameAnalytics:

    @staticmethod
    def normalize_model_name(name):
        """Normalizes raw model names and injects accurate parameter counts."""
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
    
    @staticmethod
    def calculate_total_discussions(active_games):
        """
        Calculates the total number of unique discussion rounds 
        that occurred across every game in the dataset.
        """
        total_discussions = sum(game.get('discussion_count', 0) for game in active_games)
        
        print("\n" + "="*40)
        print(f"Global Discussion Statistics")
        print("-" * 40)
        print(f"Total Games Processed:    {len(active_games)}")
        print(f"Total Discussions Held:   {total_discussions}")
        if len(active_games) > 0:
            avg = total_discussions / len(active_games)
            print(f"Avg Discussions/Game:     {avg:.2f}")
        print("="*40)
        
        return total_discussions
    
    @staticmethod
    def calculate_average_game_length(active_games):
        """
        Computes the average number of rounds played per game.
        This represents the game duration regardless of whether 
        discussions occurred in every round.
        """
        if not active_games:
            return 0.0

        durations = []
        for game in active_games:
            # Each game dict contains 'turns'. We find the highest round index
            # present in those turns to determine how long the game lasted.
            if game['turns']:
                last_round = max(t['round'] for t in game['turns'])
                durations.append(last_round)
        
        avg_rounds = sum(durations) / len(durations) if durations else 0
        
        print("\n" + "="*40)
        print(f"Game Duration Statistics")
        print("-" * 40)
        print(f"Total Games:         {len(durations)}")
        print(f"Max Rounds (Longest): {max(durations) if durations else 0}")
        print(f"Min Rounds (Shortest):{min(durations) if durations else 0}")
        print(f"Avg Rounds Per Game:  {avg_rounds:.2f}")
        print("="*40)
        
        return avg_rounds
    
    @staticmethod
    def calculate_win_rates(active_games):
        stats = defaultdict(lambda: {'total': 0, 'crew_wins': 0, 'imp_wins': 0})
        print("\nCalculating Win Rates...")
        for game in active_games:
            exp = game['experiment_id']
            comp = game['composition_id']
            turns = game['turns']
            game_winner = None
            
            for turn in turns:
                role = turn['role']
                won = turn['won']
                
                if role == 'H':
                    game_winner = 'Crew' if won == 1 else 'Imposter'
                    break
                elif role == 'B':
                    game_winner = 'Imposter' if won == 1 else 'Crew'
                    break
            
            if game_winner:
                stats[exp]['total'] += 1
                if game_winner == 'Crew':
                    stats[exp]['crew_wins'] += 1
                else:
                    stats[exp]['imp_wins'] += 1
                
                comp_key = f"{exp} :: {comp}"
                stats[comp_key]['total'] += 1
                if game_winner == 'Crew':
                    stats[comp_key]['crew_wins'] += 1
                else:
                    stats[comp_key]['imp_wins'] += 1

        return stats

    @staticmethod
    def print_win_rate_report(stats):
        print("\n" + "="*80)
        print(f"{'EXPERIMENT / COMPOSITION':<50} | {'GAMES':<6} | {'CREW %':<8} | {'IMP %':<8}")
        print("="*80)
        
        sorted_keys = sorted(stats.keys())
        for key in sorted_keys:
            data = stats[key]
            total = data['total']
            c_rate = (data['crew_wins'] / total) * 100 if total > 0 else 0
            i_rate = (data['imp_wins'] / total) * 100 if total > 0 else 0
            
            if "::" not in key:
                print("-" * 80)
                print(f"{key.upper():<50} | {total:<6} | {c_rate:6.2f}% | {i_rate:6.2f}%")
                print("-" * 80)
            else:
                clean_name = key.split("::")[1].strip()
                print(f"  {clean_name:<48} | {total:<6} | {c_rate:6.2f}% | {i_rate:6.2f}%")

    @staticmethod
    def calculate_population_shifts(active_games):
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
            'played': 0, 'won': 0, 'total_votes': 0, 'correct_votes': 0
        })))

        print("\nCalculating Population Shifts (Win Rate & Accuracy)...")
        for game in active_games:
            exp_id = game['experiment_id']
            if 'experiment_1' in exp_id.lower(): exp_key = 'exp_1'
            elif 'experiment_2' in exp_id.lower(): exp_key = 'exp_2'
            elif 'experiment_3' in exp_id.lower(): exp_key = 'exp_3'
            elif 'experiment_4' in exp_id.lower(): exp_key = 'exp_4'
            else: continue

            seen_agents_for_winrate = set()
            for turn in game['turns']:
                agent = turn['agent']
                model = turn['model']
                role = 'Crew' if turn['role'] == 'H' else 'Imposter'

                if agent not in seen_agents_for_winrate:
                    stats[exp_key][model][role]['played'] += 1
                    stats[exp_key][model][role]['won'] += turn['won']
                    seen_agents_for_winrate.add(agent)
                if role == 'Crew':
                    target = turn.get('vote_target', 'None')
                    if target not in ['None', 'SKIP']:
                        stats[exp_key][model][role]['total_votes'] += 1
                        if turn['vote_correct']:
                            stats[exp_key][model][role]['correct_votes'] += 1
        return stats

    @staticmethod
    def print_shift_report(stats):
        comparisons = [
            ("HEAVYWEIGHT SHIFT (Homogenous Exp 1 -> Heterogenous Exp 2)", 'exp_1', 'exp_2'),
            ("LIGHTWEIGHT SHIFT (Homogenous Exp 3 -> Heterogenous Exp 4)", 'exp_3', 'exp_4')
        ]
        
        for title, src_exp, tgt_exp in comparisons:
            print("\n" + "="*115)
            print(f"{title}")
            print("="*115)
            print(f"\n--- CREW SHIFT (Sorted by Win Rate Delta) ---")
            print(f"{'MODEL NAME':<40} | {'WIN(Hom)':<10} {'WIN(Het)':<10} {'Δ WIN':<7} | {'ACC(Hom)':<10} {'ACC(Het)':<10} {'Δ ACC':<10}")
            print("-" * 115)
            
            src_models = set(stats[src_exp].keys())
            tgt_models = set(stats[tgt_exp].keys())
            all_models = list(src_models | tgt_models)
            
            rows = []
            for model in all_models:
                s_data = stats[src_exp].get(model, {}).get('Crew', {'played':0, 'won':0, 'total_votes':0, 'correct_votes':0})
                s_win = (s_data['won'] / s_data['played'] * 100) if s_data['played'] > 0 else 0.0
                s_acc = (s_data['correct_votes'] / s_data['total_votes'] * 100) if s_data['total_votes'] > 0 else 0.0
                
                t_data = stats[tgt_exp].get(model, {}).get('Crew', {'played':0, 'won':0, 'total_votes':0, 'correct_votes':0})
                t_win = (t_data['won'] / t_data['played'] * 100) if t_data['played'] > 0 else 0.0
                t_acc = (t_data['correct_votes'] / t_data['total_votes'] * 100) if t_data['total_votes'] > 0 else 0.0
                
                if s_data['played'] == 0 and t_data['played'] == 0: continue

                rows.append({
                    'model': model,
                    's_win': s_win, 't_win': t_win, 'd_win': t_win - s_win,
                    's_acc': s_acc, 't_acc': t_acc, 'd_acc': t_acc - s_acc
                })
                
            rows.sort(key=lambda x: x['d_win'], reverse=True)
            for r in rows:
                d_win_str = f"{r['d_win']:+.1f}%"
                d_acc_str = f"{r['d_acc']:+.1f}%"
                print(f"{r['model']:<40.30} | {r['s_win']:8.1f}%  {r['t_win']:8.1f}%  {d_win_str:<7} | {r['s_acc']:8.1f}%  {r['t_acc']:8.1f}%  {d_acc_str:<7}")

            print(f"\n--- IMPOSTER SHIFT (Sorted by Win Rate Delta) ---")
            print(f"{'MODEL NAME':<40} | {'WIN(Hom)':<10} {'WIN(Het)':<10} {'Δ WIN':<7}")
            print("-" * 80)
            
            rows = []
            for model in all_models:
                s_data = stats[src_exp].get(model, {}).get('Imposter', {'played':0, 'won':0})
                s_win = (s_data['won'] / s_data['played'] * 100) if s_data['played'] > 0 else 0.0
                t_data = stats[tgt_exp].get(model, {}).get('Imposter', {'played':0, 'won':0})
                t_win = (t_data['won'] / t_data['played'] * 100) if t_data['played'] > 0 else 0.0
                if s_data['played'] == 0 and t_data['played'] == 0: continue
                rows.append({'model': model, 's_win': s_win, 't_win': t_win, 'd_win': t_win - s_win})
                
            rows.sort(key=lambda x: x['d_win'], reverse=True)
            for r in rows:
                d_win_str = f"{r['d_win']:+.1f}%"
                print(f"{r['model']:<40.30} | {r['s_win']:8.1f}%  {r['t_win']:8.1f}%  {d_win_str:<7}")

    @staticmethod
    def calculate_voting_metrics(active_games):
        """
        Computes TP, FP, FN, Precision, Recall, and F1 for crewmate voting.
        """
        
        # Store global accumulations per model
        model_metrics = defaultdict(lambda: {
            'TP': 0, 'FP': 0, 'FN': 0, 
            'base_TP': 0.0, 'base_FP': 0.0, 'base_FN': 0.0
        })

        for game in active_games:
            rounds = defaultdict(list)
            for turn in game['turns']:
                rounds[turn['round']].append(turn)

            for round_num, turns in rounds.items():
                agents_in_round = {}
                for t in turns:
                    agents_in_round[t['agent']] = {
                        'role': t['role'], 
                        'model': t['model'], 
                        'vote': t['vote_target'], 
                        'vote_role': t['vote_target_role']
                    }

                # V = Valid targets (Total alive - 1 for self)
                V = max(1, len(agents_in_round) - 1) 
                I = sum(1 for a in agents_in_round.values() if a['role'] == 'B')

                # Random baseline expectations for this round state
                e_tp = I / (V + 1)
                e_fp = (V - I) / (V + 1)
                e_fn = 1 / (V + 1)

                for agent_id, data in agents_in_round.items():
                    if data['role'] == 'H' and 'olmo' not in data['model'].lower(): 
                        model = data['model']
                        vote = data['vote']
                        vote_role = data['vote_role']

                        # Global accumulators
                        model_metrics[model]['base_TP'] += e_tp
                        model_metrics[model]['base_FP'] += e_fp
                        model_metrics[model]['base_FN'] += e_fn
                        
                        if vote == 'None' or vote == 'SKIP':
                            model_metrics[model]['FN'] += 1
                        elif vote_role == 'B':
                            model_metrics[model]['TP'] += 1
                        elif vote_role == 'H':
                            model_metrics[model]['FP'] += 1

        # Aggregate Global Results
        results = {}
        ov_TP = ov_FP = ov_FN = 0
        ov_base_TP = ov_base_FP = ov_base_FN = 0

        # Arrays to hold final model scores to calculate std dev for the overall average
        model_Ps, model_Rs, model_F1s = [], [], []
        base_Ps, base_Rs, base_F1s = [], [], []

        for model, counts in model_metrics.items():
            TP, FP, FN = counts['TP'], counts['FP'], counts['FN']
            
            ov_TP += TP
            ov_FP += FP
            ov_FN += FN
            ov_base_TP += counts['base_TP']
            ov_base_FP += counts['base_FP']
            ov_base_FN += counts['base_FN']

            P = TP / (TP + FP) if (TP + FP) > 0 else 0
            R = TP / (TP + FN) if (TP + FN) > 0 else 0
            F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0

            b_TP, b_FP, b_FN = counts['base_TP'], counts['base_FP'], counts['base_FN']
            bP = b_TP / (b_TP + b_FP) if (b_TP + b_FP) > 0 else 0
            bR = b_TP / (b_TP + b_FN) if (b_TP + b_FN) > 0 else 0
            bF1 = 2 * (bP * bR) / (bP + bR) if (bP + bR) > 0 else 0

            results[model] = {
                'TP': TP, 'FP': FP, 'FN': FN, 
                'P': P, 'R': R, 'F1': F1,
                'bP': bP, 'bR': bR, 'bF1': bF1,
            }
            
            # Store for global std dev calculation
            model_Ps.append(P)
            model_Rs.append(R)
            model_F1s.append(F1)
            base_Ps.append(bP)
            base_Rs.append(bR)
            base_F1s.append(bF1)

        def get_std(lst): return statistics.stdev(lst) if len(lst) > 1 else 0.0

        # Calculate overall averages strictly based on sums
        avg_P = ov_TP / (ov_TP + ov_FP) if (ov_TP + ov_FP) > 0 else 0
        avg_R = ov_TP / (ov_TP + ov_FN) if (ov_TP + ov_FN) > 0 else 0
        avg_F1 = 2 * (avg_P * avg_R) / (avg_P + avg_R) if (avg_P + avg_R) > 0 else 0

        avg_bP = ov_base_TP / (ov_base_TP + ov_base_FP) if (ov_base_TP + ov_base_FP) > 0 else 0
        avg_bR = ov_base_TP / (ov_base_TP + ov_base_FN) if (ov_base_TP + ov_base_FN) > 0 else 0
        avg_bF1 = 2 * (avg_bP * avg_bR) / (avg_bP + avg_bR) if (avg_bP + avg_bR) > 0 else 0

        results['AVERAGE'] = {
            'TP': ov_TP, 'FP': ov_FP, 'FN': ov_FN, 
            'P': avg_P, 'R': avg_R, 'F1': avg_F1,
            'std_P': get_std(model_Ps),
            'std_R': get_std(model_Rs),
            'std_F1': get_std(model_F1s),
            'bP': avg_bP, 'bR': avg_bR, 'bF1': avg_bF1,
            'std_bP': get_std(base_Ps),
            'std_bR': get_std(base_Rs),
            'std_bF1': get_std(base_F1s)
        }

        return results

    @classmethod
    def calculate_grouped_f1(cls, voting_results):      
        small_f1s = []
        medium_f1s = []
        baseline_f1s = []
        
        for model_raw, metrics in voting_results.items():
            if model_raw == 'AVERAGE':
                continue
                
            norm_name = cls.normalize_model_name(model_raw)
            
            # Extract the integer parameter count
            match = re.search(r'(\d+)B', norm_name, re.IGNORECASE)
            if not match:
                continue
                
            size_val = int(match.group(1))
            
            # Convert F1 to percentage
            f1_pct = metrics['F1'] * 100
            bf1_pct = metrics['bF1'] * 100
            
            # Bucket the models
            if 7 <= size_val <= 20:
                small_f1s.append(f1_pct)
            elif size_val >= 60:
                medium_f1s.append(f1_pct)

            baseline_f1s.append(bf1_pct)

        def get_stats(data):
            if not data:
                return 0.0, 0.0
            mean_val = sum(data) / len(data)
            std_val = statistics.stdev(data) if len(data) > 1 else 0.0
            return mean_val, std_val

        small_mean, small_std = get_stats(small_f1s)
        med_mean, med_std = get_stats(medium_f1s)
        base_mean, base_std = get_stats(baseline_f1s)

        print("\n" + "="*90)
        print(f"{'GROUPED F1 SCORES (Small vs Medium)':^90}")
        print("="*90)
        print(f"{'Model Group':<18} | {'F1':<15} |")
        print("-" * 90)
        print(f"{'Small (7B-20B)':<18} | {small_mean:5.1f}% ± {small_std:<5.1f}% |")
        print(f"{'Medium (60B+)':<18} | {med_mean:5.1f}% ± {med_std:<5.1f}% |")
        print(f"{'Random baseline':<18} | {base_mean:5.1f}% ± {base_std:<5.1f}% |")
        print("="*90)
    
    
    @classmethod
    def calculate_round_level_f1_significance(cls, active_games):
        """
        Computes F1 scores for every individual round to perform a 
        high-powered one-sided Mann-Whitney U test on the performance distribution.
        """

        small_round_f1s = []
        medium_round_f1s = []

        for game in active_games:
            # Experiments 1/2 are Medium (Heavyweight); 3/4 are Small (Lightweight)
            exp_id = game['experiment_id'].lower()
            is_medium = any(x in exp_id for x in ['experiment_1', 'experiment_2'])
            is_small = any(x in exp_id for x in ['experiment_3', 'experiment_4'])
            
            # Aggregate voting outcomes by round within the game 
            round_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
            
            for turn in game['turns']:
                if turn['role'] == 'H' and 'olmo' not in turn['model'].lower():
                    r_num = turn['round']
                    target = turn.get('vote_target', 'None')
                    
                    if target in ['None', 'SKIP']:
                        round_stats[r_num]['FN'] += 1
                    elif turn['vote_correct']:
                        round_stats[r_num]['TP'] += 1
                    else:
                        round_stats[r_num]['FP'] += 1
            
            for r_num, s in round_stats.items():
                tp, fp, fn = s['TP'], s['FP'], s['FN']
                # F1 = 2TP / (2TP + FP + FN) 
                denominator = (2 * tp) + fp + fn
                
                if denominator > 0:
                    f1 = (2 * tp) / denominator
                    if is_medium:
                        medium_round_f1s.append(f1 * 100)
                    elif is_small:
                        small_round_f1s.append(f1 * 100)

        print("\n" + "="*90)
        print(f"{'STATISTICAL SIGNIFICANCE (ONE-TAILED): ROUND-LEVEL F1':^90}")
        print("="*90)

        # UPDATED: Changed to alternative='less' to test if Small < Medium
        u_stat, p_value = stats.mannwhitneyu(small_round_f1s, medium_round_f1s, alternative='less')

        print(f"Small Model Rounds  (n={len(small_round_f1s):<6}): "
            f"Median F1 = {np.median(small_round_f1s):.1f}% | Mean F1 = {np.mean(small_round_f1s):.1f}%")
        print(f"Medium Model Rounds (n={len(medium_round_f1s):<6}): "
            f"Median F1 = {np.median(medium_round_f1s):.1f}% | Mean F1 = {np.mean(medium_round_f1s):.1f}%")
        print("-" * 90)
        print(f"U-Statistic : {u_stat:.4f}")
        print(f"P-Value     : {p_value:.4e}") 
        print("-" * 90)

        if p_value < 0.05:
            print("CONCLUSION: Small models are significantly less effective than Medium models (p < 0.05).")
        else:
            print("CONCLUSION: No significant directional difference found.")
        print("="*90)

        return u_stat, p_value

    @staticmethod
    def print_voting_metrics_report(results):
        print("\n" + "="*125)
        print(f"{'CREWMATE VOTING PERFORMANCE (vs Random Baseline)':^125}")
        print("="*125)
        print(f"{'MODEL NAME':<32} | {'TP':<5} {'FP':<5} {'FN':<5} | {'P':<6} {'R':<6} {'F1':<6} | {'Base P':<7} {'Base R':<7} {'Base F1':<7}")
        print("-" * 125)

        avg_res = results.pop('AVERAGE')
        sorted_models = sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True)

        for model, metrics in sorted_models:
            print(f"{model:<32.30} | {metrics['TP']:<5} {metrics['FP']:<5} {metrics['FN']:<5} | "
                  f"{metrics['P']*100:5.1f}% {metrics['R']*100:5.1f}% {metrics['F1']*100:5.1f}% | "
                  f"{metrics['bP']*100:6.1f}% {metrics['bR']*100:6.1f}% {metrics['bF1']*100:6.1f}%")

        print("-" * 125)
        
        avg_P_str = f"{avg_res['P']*100:.1f}±{avg_res['std_P']*100:.1f}"
        avg_R_str = f"{avg_res['R']*100:.1f}±{avg_res['std_R']*100:.1f}"
        avg_F1_str = f"{avg_res['F1']*100:.1f}±{avg_res['std_F1']*100:.1f}"
        
        base_P_str = f"{avg_res['bP']*100:.1f}±{avg_res['std_bP']*100:.1f}"
        base_R_str = f"{avg_res['bR']*100:.1f}±{avg_res['std_bR']*100:.1f}"
        base_F1_str = f"{avg_res['bF1']*100:.1f}±{avg_res['std_bF1']*100:.1f}"

        print(f"{'AVERAGE':<32} | {avg_res['TP']:<5} {avg_res['FP']:<5} {avg_res['FN']:<5} | "
              f"{avg_P_str:<11} {avg_R_str:<10} {avg_F1_str:<10} | "
              f"{base_P_str:<10} {base_R_str:<10} {base_F1_str:<10}")
        print("=" * 125)

    @classmethod
    def plot_voting_accuracy_vs_size(cls, voting_results, save_path="voting_accuracy_vs_scale.png"):
        """
        Aggregates precision by model parameter count and generates a bar chart 
        with error bars representing the standard deviation across models of that size.
        """
        
        size_accuracies = defaultdict(list)
        size_data = defaultdict(lambda: {'TP': 0, 'FP': 0, 'votes': 0, 'bP_sum': 0.0})
        
        for model_raw, metrics in voting_results.items():
            if model_raw == 'AVERAGE': continue
            norm_name = cls.normalize_model_name(model_raw)
            match = re.search(r'(\d+)B', norm_name, re.IGNORECASE)
            
            if match:
                size_val = int(match.group(1))
                votes = metrics['TP'] + metrics['FP']
                if votes > 0:
                    acc = metrics['TP'] / votes
                    size_accuracies[size_val].append(acc * 100)
                    
                    size_data[size_val]['TP'] += metrics['TP']
                    size_data[size_val]['FP'] += metrics['FP']
                    size_data[size_val]['votes'] += votes
                    size_data[size_val]['bP_sum'] += metrics['bP'] * votes

        rows = []
        for size in sorted(size_data.keys()):
            d = size_data[size]
            acc_list = size_accuracies[size]
            mean_acc = d['TP'] / d['votes'] * 100
            std_dev = statistics.stdev(acc_list) if len(acc_list) > 1 else 0.0
            
            rows.append({
                'Model Scale': f"{size}B",
                'Accuracy': mean_acc,
                'Baseline': (d['bP_sum'] / d['votes']) * 100,
                'StdDev': std_dev
            })
            
        df = pd.DataFrame(rows)
        global_baseline = df['Baseline'].mean()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Model Scale'], df['Accuracy'], yerr=df['StdDev'],
                       color='#1f77b4', edgecolor='black', capsize=5, zorder=3)
        
        plt.axhline(y=global_baseline, color='#d62728', linestyle='--', linewidth=2, 
                    label=f'Avg Random Baseline ({global_baseline:.1f}%)', zorder=4)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        plt.xlabel('Model Scale (Parameters)', fontsize=12, fontweight='bold')
        plt.ylabel('Crew Voting Precision (%)', fontsize=12, fontweight='bold')
        plt.title('Voting Precision vs. Model Scale (with StdDev)', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.legend(loc='upper left')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.1f}%', 
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        return df
    
    @classmethod
    def plot_voting_accuracy_vs_sizegroup(cls, voting_results, save_path="voting_accuracy_vs_scale_grouped.png"):
        """
        Aggregates precision by size group with error bars showing model variability.
        """
        group_accuracies = defaultdict(list)
        group_totals = {
            "Small (7B-20B)": {'TP': 0, 'FP': 0, 'votes': 0, 'bP_sum': 0.0},
            "Medium (60B+)": {'TP': 0, 'FP': 0, 'votes': 0, 'bP_sum': 0.0}
        }
        
        for model_raw, metrics in voting_results.items():
            if model_raw == 'AVERAGE': continue
            norm_name = cls.normalize_model_name(model_raw)
            match = re.search(r'(\d+)B', norm_name, re.IGNORECASE)
            if not match: continue
            
            size_val = int(match.group(1))
            group_key = None
            if 7 <= size_val <= 20: group_key = "Small (7B-20B)"
            elif size_val >= 60: group_key = "Medium (60B+)"
            
            if group_key:
                votes = metrics['TP'] + metrics['FP']
                if votes > 0:
                    acc = (metrics['TP'] / votes) * 100
                    group_accuracies[group_key].append(acc)
                    group_totals[group_key]['TP'] += metrics['TP']
                    group_totals[group_key]['FP'] += metrics['FP']
                    group_totals[group_key]['votes'] += votes
                    group_totals[group_key]['bP_sum'] += metrics['bP'] * votes

        rows = []
        for group in ["Small (7B-20B)", "Medium (60B+)"]:
            d = group_totals[group]
            acc_list = group_accuracies[group]
            if not acc_list: continue
            
            mean_acc = d['TP'] / d['votes'] * 100
            std_dev = statistics.stdev(acc_list) if len(acc_list) > 1 else 0.0
            
            rows.append({
                'Model Group': group,
                'Accuracy': mean_acc,
                'Baseline': (d['bP_sum'] / d['votes']) * 100,
                'StdDev': std_dev
            })
            
        df = pd.DataFrame(rows)
        plt.figure(figsize=(8, 6))
        
        # Plot with yerr (Standard Deviation)
        bars = plt.bar(df['Model Group'], df['Accuracy'], yerr=df['StdDev'],
                       color=['#1f77b4', '#ff7f0e'], edgecolor='black', 
                       capsize=10, zorder=3, width=0.6)
        
        global_baseline = df['Baseline'].mean()
        plt.axhline(y=global_baseline, color='#d62728', linestyle='--', linewidth=2, 
                    label=f'Avg Random Baseline ({global_baseline:.1f}%)', zorder=4)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        plt.ylabel('Crew Voting Precision (%)', fontsize=12, fontweight='bold')
        plt.title('Voting Precision: Small vs. Medium (with StdDev)', fontsize=14, fontweight='bold')
        plt.ylim(0, 110)
        plt.legend(loc='upper left')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        return df

class ObserverPipeline:
    def __init__(self, output_dir="results/classifiers"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.experiments = {
            'Train_Homogenous_Test_Heterogenous': {
                'train': lambda df: df['Composition'] == 'Homogenous',
                'test': lambda df: df['Composition'] == 'Heterogenous'
            },
            'Train_Heterogenous_Test_Homogenous': {
                'train': lambda df: df['Composition'] == 'Heterogenous',
                'test': lambda df: df['Composition'] == 'Homogenous'
            },
            'Train_Lightweight_Test_Heavyweight': {
                'train': lambda df: df['WeightClass'] == 'Lightweight',
                'test': lambda df: df['WeightClass'] == 'Heavyweight'
            },
            'Train_Heavyweight_Test_Lightweight': {
                'train': lambda df: df['WeightClass'] == 'Heavyweight',
                'test': lambda df: df['WeightClass'] == 'Lightweight'
            }
        }
        
        self.models = {
            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            'MLP_Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SGD_Classifier': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42),
            'SVM': CalibratedClassifierCV(LinearSVC(dual='auto', random_state=42), cv=3),
            #'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        }

    def _build_pipeline(self, model):
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 3)), 'Text'),
                ('num', MinMaxScaler(), ['Reported', 'Statement_Num'])
            ],
            remainder='drop'
        )
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

    def _get_round_predictions(self, df, mask, clf):
        df_masked = df[mask].copy()
        X_all = df_masked[['Text', 'Reported', 'Statement_Num']]
        df_masked['Suspicion_Prob'] = clf.predict_proba(X_all)[:, 1]
        
        round_predictions = []
        grouped = df_masked.groupby(['Game_ID', 'Round'])
        
        for (game_id, round_num), round_df in grouped:
            agent_meta = round_df.set_index('Agent')[['Role', 'Model_Name']].drop_duplicates()
            agent_roles = round_df.set_index('Agent')['Role'].to_dict()
            agent_models = agent_meta['Model_Name'].to_dict()
            imposter_models = [m for a, m in agent_models.items() if agent_roles[a] == 'B']
            imposters_alive = any(role == 'B' for role in agent_roles.values())
            
            agent_scores = round_df.groupby('Agent')['Suspicion_Prob'].mean().to_dict()
            if not agent_scores: continue
                
            top_suspect = max(agent_scores, key=agent_scores.get)
            highest_score = agent_scores[top_suspect]
            top_suspect_role = agent_roles.get(top_suspect)
            
            round_predictions.append({
                'highest_score': highest_score,
                'suspect_name': top_suspect,
                'suspect_role': top_suspect_role,
                'suspect_model': agent_models.get(top_suspect),
                'imposters_alive': imposters_alive,
                'live_imposter_models': imposter_models
            })
            
        return round_predictions

    def _find_optimal_threshold(self, round_predictions):
        best_f1 = -1
        best_thresh = 0.50
        thresholds = np.arange(0.50, 1.00, 0.01)
        
        for thresh in thresholds:
            tp, fp, fn = 0, 0, 0
            for rnd in round_predictions:
                if rnd['highest_score'] >= thresh:
                    if rnd['suspect_role'] == 'B': tp += 1
                    else: fp += 1
                else:
                    if rnd['imposters_alive']: fn += 1
                        
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def _print_deception(self, evasion_counts, total_encounters, model_name, exp_name):
        out_dir = os.path.join(self.output_dir, "misclassifications")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{model_name.lower()}_{exp_name}_deception_report.txt")
        
        sorted_models = sorted(total_encounters.keys(), 
                               key=lambda m: evasion_counts[m] / total_encounters[m], 
                               reverse=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"--- Misclassifications (Deception): {model_name} ---\n")
            f.write(f"{'LLM MODEL':<60} | {'EVASIONS':<10} | {'TOTAL':<8} | {'RATE':<8}\n")
            f.write("-" * 85 + "\n")
            
            for m in sorted_models:
                evasions = evasion_counts[m]
                total = total_encounters[m]
                rate = (evasions / total) * 100 if total > 0 else 0
                f.write(f"{m:<50} | {evasions:<10} | {total:<8} | {rate:>6.1f}%\n")

    def _simulate_ml_observer(self, df, train_mask, test_mask, model_obj, exp_name, n_runs=1):
        run_metrics = {'Precision': [], 'Recall': [], 'F1': []}
        thresholds = []
        evasion_counts = Counter()
        total_encounters = Counter()

        for i in range(n_runs):
            current_seed = 42 + i
            df_train = df[train_mask].copy()
            imposters = df_train[df_train['Role'] == 'B']
            crewmates = df_train[df_train['Role'] == 'H']
            
            crew_downsampled = crewmates.sample(n=len(imposters), random_state=current_seed)
            df_train_balanced = pd.concat([imposters, crew_downsampled]).sample(frac=1, random_state=current_seed)
            
            X_train = df_train_balanced[['Text', 'Reported', 'Statement_Num']]
            y_train = (df_train_balanced['Role'] == 'B').astype(int)

            if hasattr(model_obj, 'random_state'):
                model_obj.random_state = current_seed
                
            clf = self._build_pipeline(model_obj)
            clf.fit(X_train, y_train)
            
            train_round_preds = self._get_round_predictions(df, train_mask, clf)
            optimal_threshold = self._find_optimal_threshold(train_round_preds)
            thresholds.append(optimal_threshold)
            
            test_round_preds = self._get_round_predictions(df, test_mask, clf)
            
            tp, fp, fn = 0, 0, 0
            for rnd in test_round_preds:
                for m in rnd['live_imposter_models']:
                    total_encounters[m] += 1

                if rnd['highest_score'] >= optimal_threshold:
                    if rnd['suspect_role'] == 'B': tp += 1
                    else: 
                        fp += 1
                        for m in rnd['live_imposter_models']:
                            evasion_counts[m] += 1
                else:
                    if rnd['imposters_alive']: 
                        fn += 1
                        for m in rnd['live_imposter_models']:
                            evasion_counts[m] += 1
                        
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            
            run_metrics['Precision'].append(prec)
            run_metrics['Recall'].append(rec)
            run_metrics['F1'].append(f1)

            self._print_deception(evasion_counts, total_encounters, exp_name, type(model_obj).__name__)

        def get_stats(vals):
            return np.median(vals), np.std(vals)

        med_p, std_p = get_stats(run_metrics['Precision'])
        med_r, std_r = get_stats(run_metrics['Recall'])
        med_f, std_f = get_stats(run_metrics['F1'])
        med_th = np.median(thresholds)

        return med_th, med_p, std_p, med_r, std_r, med_f, std_f, clf

    def run_suite(self, df):
        print("\n" + "="*115)
        print(f"{'Classifier Results':^115}")
        print("="*115)
        
        for model_name, model_obj in self.models.items():
            results_to_save = []
            raw_metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}
            print(f"\n### {model_name.upper()} ###")
            print(f"{'EXPERIMENT':<48} | {'THRESHOLD':<10} | {'PRECISION':<14} | {'RECALL':<14} | {'F1':<14}")
            print("-" * 115)

            trained_clfs = []
            for exp_name, config in self.experiments.items():
                train_mask = config['train'](df)
                test_mask = config['test'](df)       

                res = self._simulate_ml_observer(df, train_mask, test_mask, model_obj, exp_name)
                med_th, p_med, p_std, r_med, r_std, f_med, f_std, clf = res
                
                p_str = f"{p_med:.2f} ± {p_std:.2f}"
                r_str = f"{r_med:.2f} ± {r_std:.2f}"
                f_str = f"{f_med:.2f} ± {f_std:.2f}"

                raw_metrics['threshold'].append(med_th)
                raw_metrics['precision'].append(p_med)
                raw_metrics['recall'].append(r_med)
                raw_metrics['f1'].append(f_med)
                trained_clfs.append(clf)

                print(f"{exp_name:<48} | > {med_th:<8.2f} | "
                      f"{p_str:<14} | "
                      f"{r_str:<14} | "
                      f"{f_str:<14}")
                results_to_save.append({
                    'Experiment': exp_name, 'Precision': p_str, 'Recall': r_str, 'F1': f_str
                })

            avg_th = np.mean(raw_metrics['threshold'])
            avg_p = np.mean(raw_metrics['precision'])
            avg_r = np.mean(raw_metrics['recall'])
            avg_f = np.mean(raw_metrics['f1'])
            
            print("-" * 115)
            print(f"{'AVERAGE PERFORMANCE':<48} | > {avg_th:<8.2f} | "
                  f"{avg_p:<7.2f}        | {avg_r:<7.2f}        | {avg_f:<7.2f}")
            
            results_to_save.append({
                'Experiment': 'AVERAGE',
                'Precision': f"{avg_p:.2f}",
                'Recall': f"{avg_r:.2f}",
                'F1': f"{avg_f:.2f}"
            })

            csv_filename = f"{model_name}_results.csv"
            pd.DataFrame(results_to_save).to_csv(os.path.join(self.output_dir, csv_filename), index=False)            

    def print_results(self):
        if not os.path.exists(self.output_dir):
            print(f"\n[Error] Directory '{self.output_dir}' not found. Have you run the suite yet?")
            return

        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('_results.csv')]
        if not csv_files:
            print(f"\n[Error] No result CSVs found in '{self.output_dir}'.")
            return

        print("\n" + "="*110)
        print(f"{'CLASSIFIER RESULTS':^110}")
        print("="*110)

        for file in sorted(csv_files):
            model_name = file.replace('_results.csv', '').replace('_', ' ')
            file_path = os.path.join(self.output_dir, file)
            
            try:
                df = pd.read_csv(file_path)
                avg_row = {'Experiment': 'AVERAGE'}
                
                for col in df.columns:
                    if col == 'Experiment': continue
                    try:
                        vals = df[col].apply(lambda x: float(str(x).split(' ')[0]))
                        avg_val = vals.mean()
                        avg_row[col] = f"{avg_val:.3f}"
                    except Exception:
                        avg_row[col] = "N/A"
                
                df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
                
                print(f"\n### {model_name.upper()} ###")
                print(df.to_string(index=False, justify='left'))
            except Exception as e:
                print(f"Could not read {file}: {e}")
                
        print("\n" + "="*110)

class LogAnalysis:
    @classmethod
    def count_confessions(cls, active_games, save_path="imposter_confessions.csv"):
        """
        Analyzes Imposter statements for self-incrimination using NLP dependency parsing.
        Saves off the exact statements, game IDs, and rounds for traceability.
        """
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm", disable=["ner"])

        incriminating_lemmas = {"tag", "eliminate", "kill"}
        
        # Added 'confession_logs' list to store the exact text and metadata
        stats = defaultdict(lambda: {'imposter_statements': 0, 'confessions': 0, 'confession_logs': []})
        
        print("\nExtracting Imposter Statements...")
        
        statements_to_process = []
        for game in active_games:
            game_id = game['game_id']
            for turn in game['turns']:
                if turn['role'] == 'B':
                    agent_id = turn['agent'].lower()
                    
                    num_match = re.search(r'(\d+)', agent_id)
                    my_num = num_match.group(1) if num_match else None
                    
                    # Pack the traceability metadata into the context
                    context = {
                        'model_name': turn['model'],
                        'my_num': my_num,
                        'agent_id': agent_id,
                        'game_id': game_id,
                        'round': turn['round'],
                        'text': turn['text']
                    }
                    statements_to_process.append((turn['text'], context))

        print(f"Found {len(statements_to_process)} Imposter statements. Parsing with spaCy pipeline...")

        pipeline = nlp.pipe(statements_to_process, as_tuples=True, batch_size=500)
        
        for doc, context in tqdm(pipeline, total=len(statements_to_process), desc="Parsing NLP Dependencies"):
            model_name = context['model_name']
            my_num = context['my_num']
            agent_id = context['agent_id']
            
            stats[model_name]['imposter_statements'] += 1
            is_confession = False
            
            for token in doc:
                if token.lemma_.lower() in incriminating_lemmas and token.pos_ == "VERB":
                    is_negated = any(child.dep_ == "neg" for child in token.children)
                    if is_negated:
                        continue
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass"): 
                            
                            # 1st person
                            if child.text.lower() in ("i", "we", "my"):
                                is_confession = True
                                break
                                
                            # 3rd person
                            subj_text = " ".join(t.text.lower() for t in child.subtree)
                            mentioned_nums = re.findall(r'agent\s*[_]*\s*(\d+)', subj_text)
                            
                            if my_num and my_num in mentioned_nums:
                                is_confession = True
                                break
                if is_confession:
                    break
            
            if is_confession:
                stats[model_name]['confessions'] += 1
                # Save off the exact statement and metadata for later review
                stats[model_name]['confession_logs'].append({
                    'model': model_name,
                    'game_id': context['game_id'],
                    'round': context['round'],
                    'agent': context['agent_id'],
                    'text': context['text']
                })

        # --- Calculate Totals ---
        # Note: If you want to exclude OLMo from the totals, we filter it out here
        total_stmts = sum(d['imposter_statements'] for m, d in stats.items() if 'olmo' not in m.lower())
        total_confs = sum(d['confessions'] for m, d in stats.items() if 'olmo' not in m.lower())
        overall_rate = (total_confs / total_stmts * 100) if total_stmts > 0 else 0.0

        # --- Print the Report ---
        print("\n" + "="*90)
        print(f"{'IMPOSTER CONFESSION RATES':^90}")
        print("="*90)
        print(f"{'MODEL NAME':<45} | {'STATEMENTS':<12} | {'CONFESSIONS':<12} | {'RATE (%)':<8}")
        print("-" * 90)
        
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: (x[1]['confessions'] / x[1]['imposter_statements']) if x[1]['imposter_statements'] > 0 else 0, 
            reverse=True
        )
        
        for model, data in sorted_stats:
            stmts = data['imposter_statements']
            confs = data['confessions']
            rate = (confs / stmts * 100) if stmts > 0 else 0.0
            print(f"{model:<45.43} | {stmts:<12} | {confs:<12} | {rate:.2f}%")
            
        print("-" * 90)
        print(f"{'OVERALL TOTALS (Excluding OLMo)':<45} | {total_stmts:<12} | {total_confs:<12} | {overall_rate:.2f}%")
        print("="*90)
        
        # --- Export the Logs ---
        all_logs = []
        for model, data in stats.items():
            all_logs.extend(data['confession_logs'])
            
        if all_logs:
            df_logs = pd.DataFrame(all_logs)
            df_logs.to_csv(save_path, index=False)
            print(f"\nSaved {len(all_logs)} full confession transcripts to '{save_path}'")
        
        return stats

    @classmethod
    def compute_confession_response_rate(cls, active_games, confessions_csv="imposter_confessions.csv"):
        """
        Loads the saved confessions CSV and cross-references it with the active_games logs
        to determine how often the crew successfully ejected the confessing imposter.
        """

        if not os.path.exists(confessions_csv):
            print(f"\n[Error] '{confessions_csv}' not found. Please run count_confessions() first.")
            return None

        print(f"\nLoading confessions from {confessions_csv}...")
        df_confessions = pd.read_csv(confessions_csv)

        # Build a fast lookup dictionary for round results
        # Structure: round_results_lookup[game_id][round_num] = ejected_agent
        round_results_lookup = defaultdict(dict)
        for game in active_games:
            g_id = game['game_id']
            for turn in game['turns']:
                r_num = turn['round']
                # We only need to grab the ejection result once per round
                if r_num not in round_results_lookup[g_id]:
                    round_results_lookup[g_id][r_num] = turn.get('round_ejected', 'None')

        # Track ejections per model
        stats = defaultdict(lambda: {'confessions': 0, 'ejections': 0})
        
        for _, row in df_confessions.iterrows():
            model = row['model']
            g_id = row['game_id']
            r_num = row['round']
            confessor_id = str(row['agent']).strip().lower() # e.g., "agent_0"

            stats[model]['confessions'] += 1
            
            # Check who was actually ejected at the end of this round
            ejected_agent = round_results_lookup.get(g_id, {}).get(r_num, "None")
            ejected_agent_clean = str(ejected_agent).strip().lower()

            # If the ejected agent matches the confessor, the crew successfully punished them
            if ejected_agent_clean == confessor_id:
                stats[model]['ejections'] += 1

        # --- Print the Report ---
        print("\n" + "="*85)
        print(f"{'EJECTION RATE AFTER CONFESSION':^85}")
        print("="*85)
        print(f"{'MODEL NAME':<45} | {'CONFESSIONS':<12} | {'EJECTED':<10} | {'RATE (%)':<8}")
        print("-" * 85)
        
        total_confessions = 0
        total_ejections = 0
        
        # Sort by ejection rate (highest penalty first)
        sorted_stats = sorted(
            stats.items(), 
            key=lambda x: (x[1]['ejections'] / x[1]['confessions']) if x[1]['confessions'] > 0 else 0, 
            reverse=True
        )
        
        for model, data in sorted_stats:
            confs = data['confessions']
            ejects = data['ejections']
            rate = (ejects / confs * 100) if confs > 0 else 0.0
            print(f"{model:<45.43} | {confs:<12} | {ejects:<10} | {rate:.2f}%")
            
            # Keep running totals, excluding OLMo
            if 'olmo' not in model.lower():
                total_confessions += confs
                total_ejections += ejects
                
        overall_rate = (total_ejections / total_confessions * 100) if total_confessions > 0 else 0.0
        
        print("-" * 85)
        print(f"{'OVERALL TOTALS (Excluding OLMo)':<45} | {total_confessions:<12} | {total_ejections:<10} | {overall_rate:.2f}%")
        print("="*85)
        
        return stats

    @classmethod
    def sentiment_analysis(cls, active_games):
        """
        Uses VADER sentiment analysis to compare the emotional affect of the agent 
        who reported the body vs. the agents giving routine status updates.
        """

        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            print("Downloading NLTK VADER lexicon...")
            nltk.download('vader_lexicon', quiet=True)

        sia = SentimentIntensityAnalyzer()

        # Track metrics: Reporter (Body Discovery) vs Bystander (Routine Update)
        # We will track both the 'compound' score and the 'neu' (neutral) score.
        stats = {
            'Reporter': {'compound': [], 'neutral': [], 'count': 0},
            'Bystander': {'compound': [], 'neutral': [], 'count': 0}
        }

        print("\nRunning VADER Sentiment Analysis on Discussion Logs...")

        for game in active_games:
            for turn in game['turns']:
                text = turn['text']
                is_reporter = turn['reported'] == 1
                
                scores = sia.polarity_scores(text)
                
                if is_reporter:
                    stats['Reporter']['compound'].append(scores['compound'])
                    stats['Reporter']['neutral'].append(scores['neu'])
                    stats['Reporter']['count'] += 1
                else:
                    stats['Bystander']['compound'].append(scores['compound'])
                    stats['Bystander']['neutral'].append(scores['neu'])
                    stats['Bystander']['count'] += 1

        # --- Print the Report ---
        print("\n" + "="*85)
        print(f"{'EMOTIONAL FLATNESS ANALYSIS (VADER SENTIMENT)':^85}")
        print("="*85)
        print(f"{'GAME STATE (AGENT ROLE)':<30} | {'COUNT':<10} | {'AVG COMPOUND':<15} | {'AVG NEUTRALITY':<15}")
        print("-" * 85)

        for category, data in stats.items():
            count = data['count']
            if count == 0:
                continue
                
            avg_compound = sum(data['compound']) / count
            std_compound = statistics.stdev(data['compound']) if count > 1 else 0
            
            avg_neutral = sum(data['neutral']) / count
            std_neutral = statistics.stdev(data['neutral']) if count > 1 else 0

            # VADER compound is -1 to 1. Neutral is 0 to 1 (percentage of text that is unemotional)
            compound_str = f"{avg_compound:+.3f} ± {std_compound:.2f}"
            neutral_str = f"{avg_neutral:.3f} ± {std_neutral:.2f}"

            print(f"{category:<30} | {count:<10} | {compound_str:<15} | {neutral_str:<15}")

        print("="*85)
        
        return stats

    @classmethod
    def suspect_analysis(cls, active_games):
        eval_model_id = "meta-llama/Llama-3.1-8B-Instruct" 
        checkpoint_file = "llm_suspect_evaluations_log.csv"

        completed_task_signatures = set()
        if os.path.exists(checkpoint_file):
            print(f"\nFound master log! Loading {checkpoint_file} to skip completed tasks...")
            try:
                df_prev = pd.read_csv(checkpoint_file)
                for _, row in df_prev.iterrows():
                    # Create a unique signature for every completed task
                    sig = (str(row['game_id']), str(row['agent_id']), int(row['round_1']), int(row['round_2']))
                    completed_task_signatures.add(sig)
                print(f"Successfully loaded {len(completed_task_signatures)} previously completed evaluations.")
            except Exception as e:
                print(f"Warning: Could not read checkpoint. Starting fresh. Error: {e}")
        
        # --- 1. TRACK ELIMINATIONS ---
        alive_tracker = defaultdict(lambda: defaultdict(set))
        for game in active_games:
            g_id = str(game['game_id'])
            for turn in game['turns']:
                alive_tracker[g_id][int(turn['round'])].add(str(turn['agent']))
        
        # --- 2. STRUCTURE TIMELINES ---
        agent_timelines = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'statements': [], 'vote': 'None'})))
        
        print("\nStructuring crewmate timelines...")
        for game in active_games:
            g_id = str(game['game_id'])
            for turn in game['turns']:
                if turn.get('role') != 'H': continue
                    
                a_id = str(turn['agent'])
                r_num = int(turn['round'])
                text = turn['text']
                s_num = turn.get('statement_num', 1) 
                vote = turn.get('vote_target', 'None')
                
                agent_timelines[g_id][a_id][r_num]['statements'].append(f"[Message {s_num}]: {text}")
                if vote and vote not in ('None', 'SKIP'):
                    agent_timelines[g_id][a_id][r_num]['vote'] = vote

        # --- 3. BUILD WORKLOAD TASKS ---
        tasks = []
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(eval_model_id, trust_remote_code=True)
        
        for g_id, agents in agent_timelines.items():
            for a_id, rounds_dict in agents.items():
                sorted_rounds = sorted(rounds_dict.keys())
                if len(sorted_rounds) < 2: continue
                
                for i in range(len(sorted_rounds) - 1):
                    r1, r2 = sorted_rounds[i], sorted_rounds[i+1]

                    if (g_id, a_id, r1, r2) in completed_task_signatures:
                        continue

                    eliminated_agents = alive_tracker[g_id][r1] - alive_tracker[g_id][r2]
                    eliminated_str = ", ".join(eliminated_agents) if eliminated_agents else "None"
                    
                    stmt1 = "\n".join(rounds_dict[r1]['statements'])
                    stmt2 = "\n".join(rounds_dict[r2]['statements'])
                    
                    user_prompt = SUSPECT_JUDGE_USER.format(
                        agent_id=a_id, r1=r1, stmt1=stmt1, vote1=rounds_dict[r1]['vote'], eliminated_r1=eliminated_str,
                        r2=r2, stmt2=stmt2, vote2=rounds_dict[r2]['vote']
                    )
                    
                    messages = [
                        {"role": "system", "content": SUSPECT_JUDGE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    tasks.append({
                        'g_id': g_id, 'a_id': a_id, 'r1': r1, 'r2': r2, 
                        'vote1': rounds_dict[r1]['vote'], 'vote2': rounds_dict[r2]['vote'], 
                        'prompt': formatted_prompt
                    })

        if not tasks:
            print("No valid multi-round pairs found to evaluate.")
            return 0.0

        # --- 4. MULTI-GPU DISTRIBUTION ---
        BATCH_SIZE = 8 
        
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs detected! Aborting.")
            return 0.0
            
        print(f"\nDetected {num_gpus} GPUs. Splitting {len(tasks)} tasks across them...")
        
        task_chunks = np.array_split(tasks, num_gpus)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        all_results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id in range(num_gpus):
                chunk = task_chunks[gpu_id].tolist()
                print(f"Assigning {len(chunk)} tasks to GPU {gpu_id}...")
                futures.append(
                    executor.submit(_gpu_evaluation_worker, gpu_id, chunk, eval_model_id, BATCH_SIZE, checkpoint_file)
                )
            
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        # --- 5. MERGE & FINAL REPORTING ---           
        df_new = pd.DataFrame(all_results)
        
        # Safely concatenate the newly processed tasks with the 15k already in the master file
        if os.path.exists(checkpoint_file):
            df_master = pd.read_csv(checkpoint_file)
            # Only concatenate if the new dataframe actually has data
            if not df_new.empty:
                df_final = pd.concat([df_master, df_new], ignore_index=True)
            else:
                df_final = df_master
        else:
            df_final = df_new

        df_final.to_csv(checkpoint_file, index=False)
        print(f"\nFinal merge complete. Master log updated and saved to {checkpoint_file}.")
        
        final_total = len(df_final[df_final['parse_failed'] == False])
        final_reversals = len(df_final[df_final['unjustified_reversal'] == True])
        failed_parses = len(df_final[df_final['parse_failed'] == True])
        
        print("\n" + "="*85)
        print(f"{'CONTRADICTORY REASONING / SUSPICION REVERSAL RATE (CREWMATES ONLY)':^85}")
        print("="*85)
        print(f"Total Consecutive Pairs Evaluated : {final_total}")
        print(f"Total Unjustified Reversals       : {final_reversals}")
        if failed_parses > 0:
            print(f"Session Parse Failures            : {failed_parses} (Ignored)")
        print("-" * 85)
        
        reversal_rate = (final_reversals / final_total * 100) if final_total > 0 else 0.0
        print(f"Final Reversal Rate               : {reversal_rate:.2f}%")
        print("="*85)

        return reversal_rate

    
if __name__ == "__main__":
    #ROOT_DIRECTORY = "experiments/MixedWeight_Full_Hybrid"
    #ROOT_DIRECTORY = "experiments/MixedWeight_Half_Hybrid"
    ROOT_DIRECTORY = "results/"

    #cache = "classifiers/data/MixedWeigh_Full_Hybrid"
    #cache = "classifiers/data/MixedWeigh_Half_Hybrid"
    cache = "classifiers/data/"
   
    
    #loader = GameLogLoader(ROOT_DIRECTORY, cache_dir="")
    loader = GameLogLoader(ROOT_DIRECTORY, cache_dir=cache)
    active_games, silent_games = loader.load_all(force_reload=False)

    # MOVEMENT ANALYSIS
    # movement_stats = LogAnalysis.movement_analysis(active_games)
    # SUSPECT ANALYSIS
    # suspect_analysis = LogAnalysis.suspect_analysis(active_games)

    # SENTIMENT ANALYSIS
    # sentiment_stats = LogAnalysis.sentiment_analysis(active_games)

    # CONFESSION METRICS 
    # confession_stats = LogAnalysis.count_confessions(active_games)
    #response_stats = LogAnalysis.compute_confession_response_rate(active_games)


    #csv_path = "imposter_confessions.csv"
    # if os.path.exists(csv_path):
    #     df_confessions = pd.read_csv(csv_path)
    #     print(f"\nSuccessfully loaded {len(df_confessions)} total confessions from '{csv_path}'.")
        
    #     print("\n" + "="*90)
    #     print(f"{'SAMPLE IMPOSTER CONFESSIONS':^90}")
    #     print("="*90)
        
    #     # Grab 5 random examples (or fewer if there aren't 5 total)
    #     sample_size = min(5, len(df_confessions))
    #     sample_df = df_confessions.sample(n=sample_size, random_state=42) 
        
    #     for idx, row in sample_df.iterrows():
    #         print(f"MODEL : {row['model']}")
    #         print(f"GAME  : {row['game_id']} (Round {row['round']})")
    #         print(f"AGENT : {row['agent']}")
    #         # Textwrap makes long dialogue strings easier to read in the terminal
    #         wrapped_text = textwrap.fill(str(row['text']), width=85)
    #         print(f"TEXT  :\n{wrapped_text}")
    #         print("-" * 90)
    

    
    # WIN STATS   
    # win_stats = GameAnalytics.calculate_win_rates(active_games)
    # GameAnalytics.print_win_rate_report(win_stats)
    # total_discussions = GameAnalytics.calculate_total_discussions(active_games)
    # avg_length = GameAnalytics.calculate_average_game_length(active_games)

    # Voting STATS
    # voting_results = GameAnalytics.calculate_voting_metrics(active_games)
    # GameAnalytics.print_voting_metrics_report(voting_results)
    
    # Statistical Test: Mann-Whitney U test to compare F1 scores across models
    #tat, p_value = GameAnalytics.calculate_round_level_f1_significance(active_games)
    
    # MISC
    #grouped_f1_results = GameAnalytics.calculate_grouped_f1(voting_results)

    # df_scale = GameAnalytics.plot_voting_accuracy_vs_sizegroup(
    #     voting_results, 
    #     save_path="Voting_Accuracy_Vs_Model_Scale_Grouped.png"
    # )

    # plot the voting accuracy vs. model scale for all models (not grouped)
    # df_scale_all = GameAnalytics.plot_voting_accuracy_vs_size(
    #     voting_results,
    #     save_path="Voting_Accuracy_Vs_Model_Scale.png",
    # )


    # TRAIN & TEST CLASSIFIERS
    #dataset_builder = DatasetBuilder()
    #df = dataset_builder.build(active_games)
    #pipeline = ObserverPipeline(output_dir="results/classifiers")
    #pipeline.run_suite(df)
    # pipeline.print_results()