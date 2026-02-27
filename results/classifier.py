import os
import re
import warnings
import ast
import pickle
import statistics
import pandas as pd
import numpy as np
import joblib
from collections import defaultdict, Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
warnings.filterwarnings('ignore')


class GameLogLoader:
    def __init__(self, root_dir, cache_dir="results/classifiers/data"):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
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
        self.stop_words = set(stopwords.words('english'))
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
            'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
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


if __name__ == "__main__":
    ROOT_DIRECTORY = "results"
    
    loader = GameLogLoader(ROOT_DIRECTORY)
    active_games, silent_games = loader.load_all(force_reload=False)
    
    dataset_builder = DatasetBuilder()
    df = dataset_builder.build(active_games)
    
    pipeline = ObserverPipeline(output_dir="results/classifiers")
    pipeline.run_suite(df)
    # pipeline.print_results()