import os
import sys
import re
import warnings
import ast
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.stopwords import ENGLISH_STOP_WORDS
warnings.filterwarnings('ignore')

class GameLogLoader:
    def __init__(self, root_dir, cache_dir="classifiers/data"):
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
                model_name = "Apertus-70B-Instruct-2509" if "Apertus-70B-Instruct-2509" in raw_model else raw_model
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
        if not os.path.exists(file_path): return results
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
                    if current_round > 0: results[current_round] = round_data
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
                            round_data['consensus'] = max(tally.values()) / total_votes if total_votes > 0 else 0.0
                        else:
                            round_data['consensus'] = 0.0
                    except:
                        round_data['tally'] = {}
                        round_data['consensus'] = 0.0
                e_match = ejection.search(line)
                if e_match: round_data['ejected'] = e_match.group(1)
            if current_round > 0: results[current_round] = round_data
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
                            agent_votes[agent_id][int(match.group(1))] = match.group(2).strip()
        return agent_votes

    def _parse_discussion_log(self, file_path, agent_map, round_results, all_agent_votes):
        discussions = []
        if not os.path.exists(file_path): return []
        current_round, current_reporter = 0, None
        statement_counts = defaultdict(int)
        round_pattern = re.compile(r"=== Round (\d+) ===")
        meeting_pattern = re.compile(r"\*\* MEETING CALLED by (Agent_\d+)")
        talk_pattern = re.compile(r"^(Agent_\d+):\s*(.+)$")
        current_agent, current_text = None, []
        
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
                    if (agent_role == 'H' and target_role == 'B') or (agent_role == 'B' and target_role == 'H'):
                        vote_is_correct = True
                        
                r_res = round_results.get(current_round, {'tally': {}, 'ejected': "None", 'consensus': 0.0})
                statement_counts[current_agent] += 1
                s_num = min(statement_counts[current_agent], 2)
                is_reporter = 1 if (current_agent == current_reporter and s_num == 1) else 0
                
                discussions.append({
                    'round': current_round, 'agent': current_agent, 'model': meta.get('model', 'Unknown'),
                    'role': agent_role, 'won': meta.get('won', 0), 'text': " ".join(current_text),
                    'vote_target': my_vote_target, 'vote_target_role': target_role,
                    'vote_correct': vote_is_correct, 'round_tally': r_res['tally'],
                    'round_consensus': r_res['consensus'], 'round_ejected': r_res['ejected'],
                    'reported': is_reporter, 'statement_num': s_num
                })

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r_match = round_pattern.search(line)
                if r_match:
                    save_turn()
                    current_agent, current_text, current_reporter = None, [], None
                    current_round = int(r_match.group(1))
                    statement_counts.clear()
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
        game_paths = []
        for root, dirs, files in os.walk(self.root_dir):
            if "experiment_5" in root.lower(): continue
            if 'stats.csv' in files: game_paths.append(root)
        return game_paths

    def _save_to_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.cache_file, 'wb') as f:
            pickle.dump({'active': self.games_data, 'silent': self.silent_games}, f)

    def _load_from_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                data = pickle.load(f)
                self.games_data, self.silent_games = data['active'], data['silent']
            return True
        return False

    def load_all(self, force_reload=False):
        if not force_reload and self._load_from_cache():
            
            return self.games_data, self.silent_games
        print("--- Processing Logs ---")
        for root in tqdm(self.discover_games(), desc="Loading Games"):
            path_parts = os.path.normpath(root).split(os.sep)
            game_id, composition_id = path_parts[-1], path_parts[-2]
            exp_id = next((part for part in path_parts if "experiment" in part.lower()), "Unknown")
            agent_map = self._parse_stats_csv(os.path.join(root, 'stats.csv'))
            if not agent_map: continue
            round_results = self._parse_round_results(os.path.join(root, 'roundResults.log'))
            avg_consensus = sum(r['consensus'] for r in round_results.values()) / len(round_results) if round_results else 0.0
            all_agent_votes = self._find_and_parse_votes(root)
            turns = self._parse_discussion_log(os.path.join(root, 'discussion.log'), agent_map, round_results, all_agent_votes)
            
            if not turns:
                self.silent_games.append({'experiment': exp_id, 'composition': composition_id, 'game_id': game_id})
            else:
                self.games_data.append({
                    'experiment_id': exp_id, 'composition_id': composition_id, 'game_id': game_id,
                    'game_consensus': avg_consensus, 'turns': turns,
                    'discussion_count': len(set(t['round'] for t in turns))
                })
        self._save_to_cache()
        return self.games_data, self.silent_games


class DatasetBuilder:
    def __init__(self):
        self.stop_words = ENGLISH_STOP_WORDS
        locations = ["Reactor", "Security", "UpperEngine", "LowerEngine", "MedBay", "Cafeteria", "Electrical", "Storage", "Admin", "Weapons", "Shields", "O2", "Navigation", "Communications"]
        self.loc_pattern = re.compile(r'\b(?:' + '|'.join(locations) + r')\b', flags=re.IGNORECASE)
        self.agent_pattern = re.compile(r'\bagent_\d+\b', flags=re.IGNORECASE)

    def _preprocess_text(self, text):
        text = self.loc_pattern.sub('place', text.lower())
        text = self.agent_pattern.sub('agent_x', text)
        text = re.sub(r'[^a-z0-9\s_]', '', text)
        return ' '.join([word for word in text.split() if word not in self.stop_words])

    def build(self, active_games, save_path="observer_dataset.csv"):
        if os.path.exists("observer_dataset.csv"):
            return pd.read_csv("observer_dataset.csv")
        dataset = []
        for game in active_games:
            for turn in game['turns']:
                if 'olmo' in turn['model'].lower(): continue
                dataset.append({
                    'Game_ID': game['game_id'], 'Model_Name': turn['model'], 'Round': turn['round'],
                    'Agent': turn['agent'], 'Text': self._preprocess_text(turn['text']),
                    'Reported': turn['reported'], 'Statement_Num': turn['statement_num'],
                    'Role': turn['role']
                })
        df = pd.DataFrame(dataset)
        df['Text'] = df['Text'].replace('', pd.NA)    
        df = df.dropna(subset=['Text'])  
        df.to_csv(save_path, index=False)
        return df

# -------------------------------------------------------------------
# Core Pruning Class 
# -------------------------------------------------------------------
class ContextPruner:
    def __init__(self):
        self.models = {
            #'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
            #'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'MLP_Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'SVM': CalibratedClassifierCV(LinearSVC(dual='auto', random_state=42), cv=3),
            #'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        }
        self.best_model_name = None
        self.best_pipeline = None
        self.optimal_threshold = 0.5
        self.global_avg_importance = 0.0
        self.dynamic_thresholds = {}
        
    def report_discussion_lengths(self, active_games):
        print("\n" + "="*50)
        print(f"{'DISCUSSION LENGTH FREQUENCIES':^50}")
        print("="*50)
        lengths = [game['discussion_count'] for game in active_games]
        freq = Counter(lengths)
        
        print(f"{'Discussions per Game':<25} | {'Frequency':<10}")
        print("-" * 50)
        for length in sorted(freq.keys()):
            print(f"{length:<25} | {freq[length]:<10}")
        print("="*50 + "\n")

    def _build_pipeline(self, model):
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 3)), 'Text'),
                ('num', MinMaxScaler(), ['Reported', 'Statement_Num'])
            ],
            remainder='drop'
        )
        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    def _find_optimal_threshold(self, y_true, y_probs):
        best_f1 = -1
        best_thresh = 0.50
        thresholds = np.arange(0.50, 0.95, 0.01)
        
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def train_and_evaluate_all(self, df):
        print("="*85)
        print(f"{'TRAINING CLASSIFIERS ON ALL GAMES (GLOBAL THRESHOLD OPTIMIZATION)':^85}")
        print("="*85)
        print(f"{'MODEL':<25} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'THRESHOLD':<10}")
        print("-" * 85)

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Role'])
        imposters = df_train[df_train['Role'] == 'B']
        crewmates = df_train[df_train['Role'] == 'H']
        crew_downsampled = crewmates.sample(n=len(imposters), random_state=42)
        df_train_balanced = pd.concat([imposters, crew_downsampled]).sample(frac=1, random_state=42)

        X_train = df_train_balanced[['Text', 'Reported', 'Statement_Num']]
        y_train = (df_train_balanced['Role'] == 'B').astype(int)

        X_test = df_test[['Text', 'Reported', 'Statement_Num']]
        y_test = (df_test['Role'] == 'B').astype(int)
        
        best_overall_f1 = -1

        for model_name, model_obj in self.models.items():
            clf = self._build_pipeline(model_obj)
            clf.fit(X_train, y_train)
            
            y_probs = clf.predict_proba(X_test)[:, 1]
            y_pred_default = clf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred_default)
            
            thresh = self._find_optimal_threshold(y_test, y_probs)
            y_pred_optimal = (y_probs >= thresh).astype(int)
            
            prec = precision_score(y_test, y_pred_optimal, zero_division=0)
            rec = recall_score(y_test, y_pred_optimal, zero_division=0)
            f1 = f1_score(y_test, y_pred_optimal, zero_division=0)

            print(f"{model_name:<25} | {acc:<10.3f} | {prec:<10.3f} | {rec:<10.3f} | > {thresh:<8.2f}")

            if f1 > best_overall_f1:
                best_overall_f1 = f1
                self.best_model_name = model_name
                self.best_pipeline = clf
                self.optimal_threshold = thresh
        
       
        print("-" * 85)
        print(f"Selected Best Model: {self.best_model_name} with Threshold: {self.optimal_threshold:.2f}")
        self.dynamic_thresholds = self._calculate_importance(df, self.best_pipeline)
        return self.best_pipeline, self.dynamic_thresholds

    def _calculate_importance(self, df, pipeline):
        """
        Simulates state tracking to find thresholds and reports pruning
        performance with Median and Standard Deviation stats.
        """
        print("\nCalculating Dynamic Thresholds & Per-Round Statistics...")
        X_all = df[['Text', 'Reported', 'Statement_Num']]
        
        df_eval = df.copy()
        df_eval['Prob'] = pipeline.predict_proba(X_all)[:, 1]
        
        rounds_by_n = defaultdict(list)
        all_shifts_flat = []
        
        for (game_id, round_num), round_df in df_eval.groupby(['Game_ID', 'Round']):
            unique_agents = round_df['Agent'].unique()
            n_agents = len(unique_agents)
            if n_agents <= 1: continue 
            
            prior = 1.0 / n_agents
            suspicion_state = {agent: prior for agent in unique_agents}
            current_round_shifts = []
            
            for _, row in round_df.iterrows():
                speaker = row['Agent']
                new_prob = row['Prob']
                old_prob = suspicion_state.get(speaker, prior)
                
                shift = abs(new_prob - old_prob)
                current_round_shifts.append(shift)
                all_shifts_flat.append(shift)
                suspicion_state[speaker] = new_prob
            
            rounds_by_n[n_agents].append(current_round_shifts)

        fallback_threshold = np.mean(all_shifts_flat) if all_shifts_flat else 0.5
        dynamic_thresholds = {n: np.mean([s for r in rs for s in r]) for n, rs in rounds_by_n.items()}
        dynamic_thresholds['fallback'] = fallback_threshold

        print("\n" + "="*110)
        print(f"{'PRUNING PERFORMANCE BY ROUND SIZE (n)':^110}")
        print("="*110)
        header = f"{'n':<4} | {'Threshold':<10} | {'Msgs (Med ± Std)':<22} | {'Pruned (Med ± Std)':<22} | {'Retention %':<12}"
        print(header)
        print("-" * 110)

        for n in sorted(rounds_by_n.keys(), reverse=True):
            dyn_thresh = dynamic_thresholds[n]
            round_list = rounds_by_n[n]
            
            total_msgs = []
            pruned_counts = []
            
            for r_shifts in round_list:
                r_shifts = np.array(r_shifts)
                total_msgs.append(len(r_shifts))
                pruned_counts.append(np.sum(r_shifts < dyn_thresh))
            
            # Message Stats
            msg_med, msg_std = np.median(total_msgs), np.std(total_msgs)
            # Pruned Stats
            pruned_med, pruned_std = np.median(pruned_counts), np.std(pruned_counts)
            # Average Retention for context
            avg_retention = ((np.mean(total_msgs) - np.mean(pruned_counts)) / np.mean(total_msgs)) * 100
            
            msg_stat_str = f"{msg_med:>4.1f} (±{msg_std:>4.1f})"
            pruned_stat_str = f"{pruned_med:>4.1f} (±{pruned_std:>4.1f})"
            
            print(f"{n:<4} | {dyn_thresh:<10.4f} | {msg_stat_str:<22} | {pruned_stat_str:<22} | {avg_retention:<12.1f}%")

        print("-" * 110)
        print(f"GLOBAL FALLBACK BASELINE: {fallback_threshold:.4f}")
        print("="*110 + "\n")

        return dynamic_thresholds
    

    def prune(self, statement_data, suspicion_state):
       """
       Live function for in game pruning
       """
       pass



if __name__ == "__main__":
    ROOT_DIRECTORY = "results"
    
    loader = GameLogLoader(ROOT_DIRECTORY)
    active_games, silent_games = loader.load_all(force_reload=False)
    print(f"DEBUG: Found {len(active_games)} active games.")
    dataset_builder = DatasetBuilder()
    df = dataset_builder.build(active_games)
    
    pruner = ContextPruner()
    pruner.report_discussion_lengths(active_games)
    #pruner.train_and_evaluate_all(df)

    
    best_pipeline, dynamic_thresholds = pruner.train_and_evaluate_all(df)
    fallback_baseline = dynamic_thresholds['fallback']
    
    print("\n" + "="*50)
    print(f"{'FINAL PRUNING THRESHOLD':^50}")
    print("="*50)
    print(f"Target Baseline (Fallback Avg): {fallback_baseline:.4f}")
    print("="*50 + "\n")