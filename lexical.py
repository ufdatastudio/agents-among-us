import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ast
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
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
warnings.filterwarnings('ignore')

def win_rates():
    def calculate_win_rates(active_games):
        """
        Iterates through active games to calculate Win Rates for Crew vs Imposter.
        Grouped by Experiment and Composition.
        """
        stats = defaultdict(lambda: {'total': 0, 'crew_wins': 0, 'imp_wins': 0})
        
        print("\nCalculating Win Rates...")
        
        for game in active_games:
            exp = game['experiment_id']
            comp = game['composition_id']
            turns = game['turns']
            
            # We only need to check one valid turn to determine the game outcome
            # Iterate until we find an agent with a known role and win status
            game_winner = None # 'Crew' or 'Imposter'
            
            for turn in turns:
                role = turn['role']
                won = turn['won']
                
                # 1 = Won, 0 = Lost
                if role == 'H':
                    if won == 1:
                        game_winner = 'Crew'
                    else:
                        game_winner = 'Imposter'
                    break
                elif role == 'B':
                    if won == 1:
                        game_winner = 'Imposter'
                    else:
                        game_winner = 'Crew'
                    break
            
            if game_winner:
                # Aggregate by Experiment
                stats[exp]['total'] += 1
                if game_winner == 'Crew':
                    stats[exp]['crew_wins'] += 1
                else:
                    stats[exp]['imp_wins'] += 1
                
                # Aggregate by Composition (Model)
                comp_key = f"{exp} :: {comp}"
                stats[comp_key]['total'] += 1
                if game_winner == 'Crew':
                    stats[comp_key]['crew_wins'] += 1
                else:
                    stats[comp_key]['imp_wins'] += 1

        return stats

    def print_win_rate_report(stats):
        print("\n" + "="*80)
        print(f"{'EXPERIMENT / COMPOSITION':<50} | {'GAMES':<6} | {'CREW %':<8} | {'IMP %':<8}")
        print("="*80)
        
        # Sort keys: Experiments first, then Compositions alphabetically
        sorted_keys = sorted(stats.keys())
        
        current_exp = ""
        
        for key in sorted_keys:
            data = stats[key]
            total = data['total']
            c_rate = (data['crew_wins'] / total) * 100 if total > 0 else 0
            i_rate = (data['imp_wins'] / total) * 100 if total > 0 else 0
            
            if "::" not in key:
                # It's an Experiment Total
                print("-" * 80)
                print(f"{key.upper():<50} | {total:<6} | {c_rate:6.2f}% | {i_rate:6.2f}%")
                print("-" * 80)
                current_exp = key
            else:
                # It's a Composition
                clean_name = key.split("::")[1].strip()
                print(f"  {clean_name:<48} | {total:<6} | {c_rate:6.2f}% | {i_rate:6.2f}%")

    def calculate_population_shifts(active_games):
        """
        Calculates shift in Win Rates (Crew & Imposter) and Voting Accuracy (Crew Only).
        """
        # Structure: stats[experiment][model][role] = {played, won, total_votes, correct_votes}
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

            # We track seen agents per game so we don't double-count "Games Played/Won"
            # But we DO want to count every single vote for accuracy
            seen_agents_for_winrate = set()

            for turn in game['turns']:
                agent = turn['agent']
                model = turn['model']
                role = 'Crew' if turn['role'] == 'H' else 'Imposter'
                
                # --- 1. Win Rate Stats (Once per agent per game) ---
                if agent not in seen_agents_for_winrate:
                    stats[exp_key][model][role]['played'] += 1
                    stats[exp_key][model][role]['won'] += turn['won']
                    seen_agents_for_winrate.add(agent)

                # --- 2. Voting Accuracy Stats (Every valid vote) ---
                # Only track for Crew (Imposter voting is strategic/deceptive)
                if role == 'Crew':
                    target = turn.get('vote_target', 'None')
                    if target not in ['None', 'SKIP']:
                        stats[exp_key][model][role]['total_votes'] += 1
                        if turn['vote_correct']:
                            stats[exp_key][model][role]['correct_votes'] += 1

        return stats

    def print_shift_report(stats):
        """
        Prints the delta comparison including Voting Accuracy for Crew.
        """
        comparisons = [
            ("HEAVYWEIGHT SHIFT (Homogenous Exp 1 -> Heterogenous Exp 2)", 'exp_1', 'exp_2'),
            ("LIGHTWEIGHT SHIFT (Homogenous Exp 3 -> Heterogenous Exp 4)", 'exp_3', 'exp_4')
        ]
        
        for title, src_exp, tgt_exp in comparisons:
            print("\n" + "="*115)
            print(f"{title}")
            print("="*115)
            
            # --- CREW REPORT (Win Rate + Accuracy) ---
            print(f"\n--- CREW SHIFT (Sorted by Win Rate Delta) ---")
            # Header: Model | Win Rates | Accuracy Stats
            print(f"{'MODEL NAME':<40} | {'WIN(Hom)':<10} {'WIN(Het)':<10} {'Δ WIN':<7} | {'ACC(Hom)':<10} {'ACC(Het)':<10} {'Δ ACC':<10}")
            print("-" * 115)
            
            src_models = set(stats[src_exp].keys())
            tgt_models = set(stats[tgt_exp].keys())
            all_models = list(src_models | tgt_models)
            
            rows = []
            for model in all_models:
                # Source Data
                s_data = stats[src_exp].get(model, {}).get('Crew', {'played':0, 'won':0, 'total_votes':0, 'correct_votes':0})
                s_win = (s_data['won'] / s_data['played'] * 100) if s_data['played'] > 0 else 0.0
                s_acc = (s_data['correct_votes'] / s_data['total_votes'] * 100) if s_data['total_votes'] > 0 else 0.0
                
                # Target Data
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

            # --- IMPOSTER REPORT (Win Rate Only) ---
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

    def generate_word_heatmap(active_games, top_n=50):

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        """
        Generates a heatmap of the most common words across all game discussions.
        Excludes 'Agent_#' identifiers and common stopwords.
        """
        print("\n" + "="*80)
        print(" GENERATING WORD FREQUENCY HEATMAP ")
        print("="*80)

        # 1. Aggregate Text
        all_text = ""
        for game in active_games:
            for turn in game['turns']:
                all_text += " " + turn['text']

        # 2. Preprocessing
        # Convert to lower case
        text_lower = all_text.lower()
        
        # Remove "Agent_#" patterns (e.g., agent_1, agent_12)
        # \b ensures we match whole words, \d+ matches numbers
        text_cleaned = re.sub(r'\bagent_\d+\b', '', text_lower)
        
        # Remove non-alphabetic characters (keep spaces)
        text_cleaned = re.sub(r'[^a-z\s]', '', text_cleaned)

        # Tokenize
        tokens = text_cleaned.split()

        # 3. Filtering
        stop_words = set(stopwords.words('english'))
        # Add custom game-specific noise words if needed
        custom_stops = {'vote', 'voting', 'think', 'believe', 'suspect', 'im', 'would', 'skip'} 
        stop_words.update(custom_stops)

        filtered_tokens = [
            word for word in tokens 
            if word not in stop_words and len(word) > 2
        ]

        # 4. Count Frequencies
        word_counts = Counter(filtered_tokens)
        most_common = word_counts.most_common(top_n)
        
        if not most_common:
            print("No words found.")
            return

        # 5. Prepare Data for Heatmap (Grid Layout)
        # Calculate grid dimensions (e.g., 5 rows x 10 cols for top 50)
        cols = 10
        rows = int(np.ceil(top_n / cols))
        
        # Pad the list to fit the grid if necessary
        pad_len = (rows * cols) - len(most_common)
        most_common += [("", 0)] * pad_len
        
        # Separate labels (words) and values (counts)
        labels = np.array([f"{w}\n({c})" if c > 0 else "" for w, c in most_common]).reshape(rows, cols)
        values = np.array([c for _, c in most_common]).reshape(rows, cols)
        
        # 6. Plotting
        plt.figure(figsize=(16, 8))
        sns.heatmap(
            values, 
            annot=labels, 
            fmt="", 
            cmap="YlOrRd", 
            linewidths=1, 
            cbar_kws={'label': 'Frequency Count'},
            xticklabels=False,
            yticklabels=False
        )
        
        plt.title(f"Top {top_n} Most Frequent Words in Game Discussions\n(Excluding Agent IDs & Stopwords)", fontsize=16)
        plt.tight_layout()
        plt.savefig("word_frequency_heatmap.png")
        print(f"Heatmap saved to 'word_frequency_heatmap.png'.")
        
        # Print list for verification
        print("\nTop 10 Words:")
        for w, c in most_common[:10]:
            print(f"  - {w}: {c}")
def single_sample_tests():

    def build_classifier_dataset(active_games):
        """
        Builds a preprocessed dataset for Imposter vs. Crew classification.
        Excludes the 'olmo' model and normalizes locations and agent identifiers.
        """
        # if dataframe already exists, load and return it
        if os.path.exists("classifier_dataset.csv"):
            print("Loading preprocessed dataset from 'classifier_dataset.csv'")
            return pd.read_csv("classifier_dataset.csv")
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        stop_words = set(stopwords.words('english'))
        
        locations = [
            "Reactor", "Security", "UpperEngine", "LowerEngine", "MedBay", 
            "Cafeteria", "Electrical", "Storage", "Admin", "Weapons", 
            "Shields", "O2", "Navigation", "Communications"
        ]
        
        # Pre-compile regex for speed
        loc_pattern = re.compile(r'\b(?:' + '|'.join(locations) + r')\b', flags=re.IGNORECASE)
        agent_pattern = re.compile(r'\bagent_\d+\b', flags=re.IGNORECASE)
        
        def preprocess_text(text):
            text = text.lower()
            # Normalize locations and agents
            text = loc_pattern.sub('place', text)
            text = agent_pattern.sub('agent_x', text)
            
            # Strip punctuation to ensure clean stopword filtering
            text = re.sub(r'[^a-z0-9\s_]', '', text)
            
            # Remove stopwords
            tokens = [word for word in text.split() if word not in stop_words]
            return ' '.join(tokens)

        groups = {
            'experiment_1': ('Heavyweight', 'Homogenous'),
            'experiment_2': ('Heavyweight', 'Heterogenous'),
            'experiment_3': ('Lightweight', 'Homogenous'),
            'experiment_4': ('Lightweight', 'Heterogenous')
        }

        dataset = []

        for game in active_games:
            exp_id = game['experiment_id'].lower()
            exp_key = next((k for k in groups.keys() if k in exp_id), None)
            
            if not exp_key:
                continue
                
            weight_class, composition = groups[exp_key]
            
            for turn in game['turns']:
                # Exclude Olmo from the dataset
                if 'olmo' in turn['model'].lower():
                    continue
                    
                clean_text = preprocess_text(turn['text'])
                
                dataset.append({
                    'Text': clean_text,
                    'Reported': turn['reported'],
                    'Statement_Num': turn['statement_num'],
                    'Composition': composition,
                    'WeightClass': weight_class,
                    'Model_Name': turn['model'],
                    'Role': turn['role']
                })
        df = pd.DataFrame(dataset)
        df['Text'] = df['Text'].replace('', pd.NA)    
        df = df.dropna(subset=['Text'])  
        df.to_csv("classifier_dataset.csv", index=False)
        return df

    def get_models(random_seed):
        """Instantiates models with a specific random seed for variance testing."""
        return {
            # 'Linear_SVC': LinearSVC(random_state=random_seed, class_weight='balanced', dual=False),
            # 'LightGBM': LGBMClassifier(random_state=random_seed, class_weight='balanced', n_jobs=-1, verbose=-1),
            # 'MLP_Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_seed, early_stopping=True),
            # 'Complement_NB': ComplementNB(),
            # 'Balanced_RF': BalancedRandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1, replacement=True),
            # # Contamination updated to 0.5 because the dataset is now perfectly balanced 50/50
            # 'Isolation_Forest': IsolationForest(n_estimators=100, contamination=0.5, random_state=random_seed, n_jobs=-1) 

            'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=random_seed),
            'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_seed)

        }

    def evaluate_split(df, train_mask, test_mask, model_name, n_runs=3, threshold=0.5):
        """Trains and evaluates a model n times using predict_proba where applicable."""
        X_train = df[train_mask][['Text', 'Reported', 'Statement_Num']]
        y_train = (df[train_mask]['Role'] == 'B').astype(int)
        
        X_test = df[test_mask][['Text', 'Reported', 'Statement_Num']]
        y_test = (df[test_mask]['Role'] == 'B').astype(int)
        
        metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}
        
        for i in range(n_runs):
            models = get_models(random_seed=42 + i)
            clf = build_pipeline(models[model_name])
            
            # Isolation Forest is unsupervised, it ignores y_train but pipeline accepts it safely
            clf.fit(X_train, y_train)
            
            # --- DYNAMIC PREDICTION LOGIC ---
            if model_name == 'Isolation_Forest':
                # Returns 1 for inlier, -1 for outlier. Map -1 to 1 (Imposter)
                raw_preds = clf.predict(X_test)
                y_pred = (raw_preds == -1).astype(int)
                
            elif hasattr(clf.named_steps['classifier'], "predict_proba"):
                # Use probability confidence thresholding
                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
                
            else:
                # Fallback for models like LinearSVC that use decision functions/hard boundaries
                y_pred = clf.predict(X_test)
            
            # Calculate metrics for the positive class (Imposter)
            metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['F1'].append(f1_score(y_test, y_pred, zero_division=0))
            
        return metrics

    def format_stats(metric_list):
    
        mean_val = np.mean(metric_list)
        std_val = np.std(metric_list)
        
        med_val = np.median(metric_list)
        q75, q25 = np.percentile(metric_list, [75, 25])
        iqr_val = q75 - q25
        
        return f"{mean_val:.3f} ± {std_val:.3f}", f"{med_val:.3f} ± {iqr_val:.3f}"

    def run_classifier_suite(df, output_dir="results/classifiers", prediction_threshold=0.5):
        """
        Runs cross-environment experiments and saves results.
        """
        # Fix the NaN issue caused by pd.read_csv loading empty strings
        df['Text'] = df['Text'].fillna('')
        
        # --- UNDERSAMPLE TO BALANCE THE DATASET ---
        imposters = df[df['Role'] == 'B']
        crewmates = df[df['Role'] == 'H']
        
        # Randomly sample crewmates to exactly match the number of imposters
        crewmates_downsampled = crewmates.sample(n=len(imposters), random_state=42)
        
        # Recombine and shuffle the dataframe
        df = pd.concat([imposters, crewmates_downsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
        # ------------------------------------------

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        experiments = {
            'Train_Homogenous_Test_Heterogenous': {
                'train': df['Composition'] == 'Homogenous',
                'test': df['Composition'] == 'Heterogenous'
            },
            'Train_Heterogenous_Test_Homogenous': {
                'train': df['Composition'] == 'Heterogenous',
                'test': df['Composition'] == 'Homogenous'"""Returns Mean ± Std and Median ± IQR for a list of metrics."""
            },
            'Train_Lightweight_Test_Heavyweight': {
                'train': df['WeightClass'] == 'Lightweight',
                'test': df['WeightClass'] == 'Heavyweight'
            },
            'Train_Heavyweight_Test_Lightweight': {
                'train': df['WeightClass'] == 'Heavyweight',
                'test': df['WeightClass'] == 'Lightweight'
            }
        }
        
        model_names = list(get_models(42).keys())
        
        print("\n" + "="*80)
        print(" INITIATING CLASSIFIER EXPERIMENT SUITE ")
        print(f" Positive Class: Imposter (B) | Prediction Threshold: {prediction_threshold}")
        print(f" Dataset Balanced: {len(df)} total samples ({len(imposters)} per class)")
        print("="*80)

        all_results = {model: [] for model in model_names}

        for exp_name, masks in experiments.items():
            print(f"\n--- Running Experiment: {exp_name} ---")
            train_mask = masks['train']
            test_mask = masks['test']
            
            for model_name in tqdm(model_names, desc=f"Evaluating Models", leave=False):
                # Pass the threshold directly to the evaluator
                raw_metrics = evaluate_split(df, train_mask, test_mask, model_name, n_runs=3, threshold=prediction_threshold)
                
                acc_mean, acc_med = format_stats(raw_metrics['Accuracy'])
                prec_mean, prec_med = format_stats(raw_metrics['Precision'])
                rec_mean, rec_med = format_stats(raw_metrics['Recall'])
                f1_mean, f1_med = format_stats(raw_metrics['F1'])
                
                all_results[model_name].append({
                    'Experiment': exp_name,
                    'F1 (Mean ± Std)': f1_mean,
                    'F1 (Med ± IQR)': f1_med,
                    'Precision (Mean ± Std)': prec_mean,
                    'Precision (Med ± IQR)': prec_med,
                    'Recall (Mean ± Std)': rec_mean,
                    'Recall (Med ± IQR)': rec_med,
                    'Accuracy (Mean ± Std)': acc_mean,
                    'Accuracy (Med ± IQR)': acc_med
                })
                
        print("\n" + "="*80)
        print(" RESULTS GENERATED ")
        print("="*80)
        for model_name, results in all_results.items():
            results_df = pd.DataFrame(results)
            output_path = os.path.join(output_dir, f"{model_name}_results.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")

    return None



class GameLogLoader:
    def __init__(self, root_dir, cache_dir="results/data"):
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
                
                # Accurately increment and strictly cap statements to 2
                statement_counts[current_agent] += 1
                s_num = min(statement_counts[current_agent], 2)
                
                # Flag the reporter ONLY on their very first statement
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
                
                # Check for Round Header
                r_match = round_pattern.search(line)
                if r_match:
                    save_turn()
                    current_agent = None
                    current_text = []
                    current_round = int(r_match.group(1))
                    statement_counts.clear() # Reset for new round
                    current_reporter = None
                    continue
                    
                # Identify the Reporter natively
                m_match = meeting_pattern.search(line)
                if m_match:
                    current_reporter = m_match.group(1)
                    continue

                # Check for Speaker
                t_match = talk_pattern.match(line)
                if t_match:
                    save_turn() 
                    current_agent = t_match.group(1)
                    current_text = [t_match.group(2)]
                else:
                    # Buffer multi-line output so we don't lose text
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

def build_pipeline(model):
    """Builds a scikit-learn pipeline combining text TF-IDF and numeric scaling."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=5000), 'Text'),
            ('num', MinMaxScaler(), ['Reported', 'Statement_Num'])
        ],
        remainder='drop'
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
def calculate_crew_voting_metrics(active_games):
    """
    Calculates TP, FP, FN, Precision, Recall, and F1 for Crew voting.
    Sorts results by Experiment and by Model.
    Accurately calculates empirical baselines for Precision, Recall, and F1.
    """
    from collections import defaultdict
    
    groups = {
        'experiment_1': '1) Heavyweight Models, Homogenous Composition',
        'experiment_2': '2) Heavyweight Models, Heterogenous Composition',
        'experiment_3': '3) Lightweight Models, Homogenous Composition',
        'experiment_4': '4) Lightweight Models, Heterogenous Composition'
    }

    # Data structures to hold results
    group_results = defaultdict(lambda: {
        'tp': 0, 'fp': 0, 'fn': 0, 
        'base_prec_sum': 0, 'base_rec_sum': 0, 'baseline_count': 0
    })
    model_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

    for game in active_games:
        exp_id = game['experiment_id'].lower()
        group_key = next((k for k in groups.keys() if k in exp_id), None)
        if not group_key: continue
        
        # 1. Map Agent Roles for Baseline Calculation
        agent_roles = {}
        for turn in game['turns']:
            agent_roles[turn['agent']] = turn['role']
                
        # 2. Build the active pool dynamically per round based on who is present
        active_agents_per_round = defaultdict(set)
        for turn in game['turns']:
            active_agents_per_round[turn['round']].add(turn['agent'])

        # 3. Process Votes and Confusion Matrix
        for turn in game['turns']:
            if turn['role'] != 'H': continue 
            
            model = turn['model']
            if 'olmo' in model.lower(): continue

            target = turn.get('vote_target')
            is_correct = turn.get('vote_correct') 
            r_num = turn['round']
            
            # --- BASELINE CALCULATION ---
            active_pool = active_agents_per_round[r_num]
            total_active = len(active_pool)
            
            # Count how many Imposters ('B') are in the active pool
            active_imposters = sum(1 for agent in active_pool if agent_roles.get(agent) == 'B')
            
            # Valid targets = (total_active - 1)
            valid_targets = max(total_active - 1, 1) 
            
            # Precision Baseline: Randomly guessing a player (Active Imposters / Valid Targets)
            baseline_prec = active_imposters / valid_targets if valid_targets > 0 else 0.0
            
            # Recall Baseline: Randomly guessing a player OR Skip (Active Imposters / (Active Imposters + 1 [Skip]))
            baseline_rec = active_imposters / (active_imposters + 1) if active_imposters > 0 else 0.0
            
            group_results[group_key]['base_prec_sum'] += baseline_prec
            group_results[group_key]['base_rec_sum'] += baseline_rec
            group_results[group_key]['baseline_count'] += 1
            # ----------------------------

            # --- CONFUSION MATRIX ---
            if target in ['None', 'SKIP']:
                group_results[group_key]['fn'] += 1
                model_results[model]['fn'] += 1
            elif is_correct:
                group_results[group_key]['tp'] += 1
                model_results[model]['tp'] += 1
            else:
                group_results[group_key]['fp'] += 1
                model_results[model]['fp'] += 1

    # --- HELPER TO CALCULATE METRICS ---
    def calc_metrics(res):
        tp, fp, fn = res['tp'], res['fp'], res['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
        
    # --- PRINT EXPERIMENT RESULTS ---
    print("\n" + "="*125)
    print(f"{'LLM CREW VOTING PERFORMANCE VS. EMPIRICAL BASELINES':^125}")
    print("="*125)
    print(f"{'EXPERIMENT':<48} | {'PRECISION':<9} | {'BASE_PREC':<9} | {'RECALL':<8} | {'BASE_REC':<8} | {'F1':<8} | {'BASE_F1':<8}")
    print("-" * 125)

    for key, name in groups.items():
        res = group_results[key]
        precision, recall, f1 = calc_metrics(res)
        
        avg_base_prec = (res['base_prec_sum'] / res['baseline_count']) * 100 if res['baseline_count'] > 0 else 0
        avg_base_rec = (res['base_rec_sum'] / res['baseline_count']) * 100 if res['baseline_count'] > 0 else 0
        
        # Calculate Baseline F1 using the harmonic mean of the baseline precision and recall
        avg_base_f1 = 2 * (avg_base_prec * avg_base_rec) / (avg_base_prec + avg_base_rec) if (avg_base_prec + avg_base_rec) > 0 else 0

        print(f"{name:<48} | {precision*100:8.2f}% | {avg_base_prec:8.2f}% | {recall*100:7.2f}% | {avg_base_rec:7.2f}% | {f1*100:7.2f}% | {avg_base_f1:7.2f}%")

    # --- PRINT MODEL RESULTS ---
    print("\n" + "="*95)
    print(f"{'PERFORMANCE BY MODEL (ACROSS ALL EXPERIMENTS)':^95}")
    print("="*95)
    print(f"{'MODEL':<50} | {'PRECISION':<10} | {'RECALL':<8} | {'F1':<8}")
    print("-" * 95)
    
    # Sort models by F1 Score (Highest to Lowest)
    for model in sorted(model_results.keys(), key=lambda m: calc_metrics(model_results[m])[2], reverse=True):
        res = model_results[model]
        precision, recall, f1 = calc_metrics(res)
        print(f"{model:<50} | {precision*100:8.2f}% | {recall*100:7.2f}% | {f1*100:7.2f}%")

######################################## NEW TEST SUITE ######################################################
def save_trained_model(clf, model_name, exp_name):
    """
    Saves the trained pipeline (preprocessor + model) to a file.
    """
    model_dir = "results/classifiers/models/"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create a clean filename
    filename = f"{model_name}_{exp_name}.joblib".lower().replace(" ", "_")
    save_path = os.path.join(model_dir, filename)
    
    # Save the entire pipeline
    joblib.dump(clf, save_path)

def build_dataset(active_games):
    """
    Builds a dataset that preserves Game_ID, Round, and Agent identifiers
    so ML models can be evaluated as Virtual Observers on live game states.
    """
    if os.path.exists("virtual_observer_dataset.csv"):
        print("Loading Virtual Observer dataset from 'virtual_observer_dataset.csv'")
        return pd.read_csv("virtual_observer_dataset.csv")
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        
    stop_words = set(stopwords.words('english'))
    
    locations = [
        "Reactor", "Security", "UpperEngine", "LowerEngine", "MedBay", 
        "Cafeteria", "Electrical", "Storage", "Admin", "Weapons", 
        "Shields", "O2", "Navigation", "Communications"
    ]
    
    loc_pattern = re.compile(r'\b(?:' + '|'.join(locations) + r')\b', flags=re.IGNORECASE)
    agent_pattern = re.compile(r'\bagent_\d+\b', flags=re.IGNORECASE)
    
    def preprocess_text(text):
        text = text.lower()
        text = loc_pattern.sub('place', text)
        text = agent_pattern.sub('agent_x', text)
        text = re.sub(r'[^a-z0-9\s_]', '', text)
        tokens = [word for word in text.split() if word not in stop_words]
        return ' '.join(tokens)

    groups = {
        'experiment_1': ('Heavyweight', 'Homogenous'),
        'experiment_2': ('Heavyweight', 'Heterogenous'),
        'experiment_3': ('Lightweight', 'Homogenous'),
        'experiment_4': ('Lightweight', 'Heterogenous')
    }

    dataset = []

    for game in active_games:
        exp_id = game['experiment_id'].lower()
        exp_key = next((k for k in groups.keys() if k in exp_id), None)
        
        if not exp_key: continue
            
        weight_class, composition = groups[exp_key]
        
        for turn in game['turns']:
            if 'olmo' in turn['model'].lower(): continue
                
            clean_text = preprocess_text(turn['text'])
            
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
    df.to_csv("observer_dataset.csv", index=False)
    return df

def print_deception(evasion_counts, total_encounters, model_name, exp_name, output_dir="results/classifiers/misclassifications/"):
    """
    Ranks LLMs by missclassification count as a way to model how deceptive a model is
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{model_name.lower()}_{exp_name}_deception_report.txt")
    
    # Sort by evasion rate (False Negatives / Total rounds as an Imposter)
    # This places the most 'deceptive' models at the top
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
            
def get_round_predictions(df, mask, clf):
    """
    Takes a trained classifier and a dataset mask, groups by Game and Round,
    and extracts the top suspect and their suspicion score for each round.
    """
    df_masked = df[mask].copy()
    
    # Predict probabilities for all statements
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
        
        # Aggregate statement probabilities into Agent Suspicion Scores
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

def find_optimal_threshold(round_predictions):
    """
    Sweeps thresholds from 0.50 to 0.99 to find the threshold 
    that maximizes the F1 Score specifically on the provided data.
    """
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


def analyze_feature_importance(clf, model_name, exp_name, top_n=20):
    """
    Extracts and saves the most influential features (Text, Reported, Statement_Num)
    for a trained pipeline, mapping linear coefficients to their correlated roles.
    """
    # 1. Extract feature names from the ColumnTransformer
    preprocessor = clf.named_steps['preprocessor']
    
    # Get TF-IDF feature names (words)
    tfidf = preprocessor.named_transformers_['text']
    text_features = tfidf.get_feature_names_out()
    
    # Combined feature list: [words] + [Reported, Statement_Num]
    all_feature_names = np.concatenate([text_features, ['Reported', 'Statement_Num']])
    
    # 2. Extract weights based on model type
    model = clf.named_steps['classifier']
    importances = None
    is_linear = False
    
    # Handle Linear Models (Logistic Regression, LinearSVC, SGD)
    if hasattr(model, 'coef_'):
        importances = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
        is_linear = True
        
    # Handle Tree-based Models (XGBoost, Random Forest, LightGBM)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        is_linear = False
        
    # Handle Calibrated Models (SVM)
    elif hasattr(model, 'calibrated_classifiers_'):
            coefs = []
            for est in model.calibrated_classifiers_:
                if hasattr(est, 'estimator'):
                    coefs.append(est.estimator.coef_[0])
                else:
                    coefs.append(est.base_estimator.coef_[0])
            importances = np.mean(coefs, axis=0)
            is_linear = True

    if importances is None:
        return

    # 3. Create DataFrame
    feature_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    })
    
    # 4. Map the Correlated Role
    if is_linear:
        # Positive weights push probability toward 1 (Imposter 'B')
        # Negative weights push probability toward 0 (Crew 'H')
        feature_df['Correlated_Role'] = np.where(
            feature_df['Importance'] > 0, 'Imposter (B)', 
            np.where(feature_df['Importance'] < 0, 'Crew (H)', 'Neutral')
        )
    else:
        # Trees only output positive Gini/Gain importance values
        feature_df['Correlated_Role'] = 'Magnitude Only (Tree)'
    
    # 5. Sort by absolute value to find strongest predictors overall
    feature_df['Abs_Importance'] = feature_df['Importance'].abs()
    top_features = feature_df.sort_values(by='Abs_Importance', ascending=False).head(top_n)

    # 6. Save
    output_dir = "results/classifiers/features"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{model_name}_{exp_name}_features.csv".lower().replace(" ", "_")
    top_features.to_csv(os.path.join(output_dir, filename), index=False)

def simulate_ml_observer(df, train_mask, test_mask, model_obj, exp_name, n_runs=1):
    """
    Trains and evaluates the model n_runs times. 
    Returns the median threshold used and the Median ± Std Dev for metrics.
    """
    run_metrics = {'Precision': [], 'Recall': [], 'F1': []}
    thresholds = []

    evasion_counts = Counter()
    total_encounters = Counter()

    for i in range(n_runs):
        current_seed = 42 + i
        
        # --- 1. TRAIN ON BALANCED STATEMENTS (Learn Vocabulary) ---
        df_train = df[train_mask].copy()
        imposters = df_train[df_train['Role'] == 'B']
        crewmates = df_train[df_train['Role'] == 'H']
        
        # Undersample with varying seed
        crew_downsampled = crewmates.sample(n=len(imposters), random_state=current_seed)
        df_train_balanced = pd.concat([imposters, crew_downsampled]).sample(frac=1, random_state=current_seed)
        
        X_train = df_train_balanced[['Text', 'Reported', 'Statement_Num']]
        y_train = (df_train_balanced['Role'] == 'B').astype(int)
        
        # Re-instantiate model to apply the new seed if the model supports it
        if hasattr(model_obj, 'random_state'):
            model_obj.random_state = current_seed
            
        clf = build_pipeline(model_obj)
        clf.fit(X_train, y_train)
        # save model
        model_name = type(model_obj).__name__
        # exp name is the name of train mask
        if exp_name == 'Train_Heterogenous_Test_Homogenous':
            analyze_feature_importance(clf, model_name, exp_name)
        #     save_trained_model(clf, model_name, exp_name)

        # --- 2. TUNE THRESHOLD ON IMBALANCED TRAINING ROUNDS ---
        train_round_preds = get_round_predictions(df, train_mask, clf)
        optimal_threshold = find_optimal_threshold(train_round_preds)
        thresholds.append(optimal_threshold)
        
        # --- 3. TEST ON UNSEEN IMBALANCED TEST ROUNDS ---
        test_round_preds = get_round_predictions(df, test_mask, clf)
        
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

        print_deception(evasion_counts, total_encounters, exp_name, type(model_obj).__name__)

    # --- 4. CALCULATE STATS ---
    def get_stats(vals):
        return np.median(vals), np.std(vals)

    med_p, std_p = get_stats(run_metrics['Precision'])
    med_r, std_r = get_stats(run_metrics['Recall'])
    med_f, std_f = get_stats(run_metrics['F1'])
    med_th = np.median(thresholds)

    return med_th, med_p, std_p, med_r, std_r, med_f, std_f

def verify_model_logic(df, train_mask, test_mask, model_obj, num_samples=1):
    """
    Prints a detailed breakdown of a live round to verify that statement 
    probabilities are aggregating correctly into an agent-level vote.
    """
    # 1. Train the model exactly as we do in the suite
    df_train = df[train_mask].copy()
    imposters = df_train[df_train['Role'] == 'B']
    crew = df_train[df_train['Role'] == 'H']
    df_balanced = pd.concat([imposters, crew.sample(n=len(imposters), random_state=42)])
    
    clf = build_pipeline(model_obj)
    clf.fit(df_balanced[['Text', 'Reported', 'Statement_Num']], (df_balanced['Role'] == 'B').astype(int))
    
    # 2. Pick a few random rounds from the TEST set
    df_test = df[test_mask].copy()
    df_test = df_test[df_test['Round'] == 1]
    grouped = list(df_test.groupby(['Game_ID', 'Round']))
    sample_rounds = [grouped[i] for i in np.random.choice(len(grouped), num_samples)]
    

    for (game_id, round_num), round_df in sample_rounds:
        print(f"\n--- VERIFICATION: Game {game_id} | Round {round_num} ---")
        
        # Calculate probabilities
        round_df = round_df.copy()
        round_df['Prob'] = clf.predict_proba(round_df[['Text', 'Reported', 'Statement_Num']])[:, 1]
        
        # Aggregate scores
        agent_stats = round_df.groupby('Agent').agg({
            'Prob': 'mean',
            'Role': 'first',
            'Text': 'count'
        }).rename(columns={'Text': 'Statements'})
        
        # Sort by most suspicious
        agent_stats = agent_stats.sort_values(by='Prob', ascending=False)
        
        print(agent_stats.to_string())
        
        top_suspect = agent_stats.index[0]
        actual_role = agent_stats.loc[top_suspect, 'Role']
        print(f"\nMODEL CHOICE: {top_suspect} | ACTUAL ROLE: {actual_role}")
        print(f"CONFIDENCE SCORE: {agent_stats.loc[top_suspect, 'Prob']:.2%}")
        
        if actual_role == 'B':
            print("RESULT: SUCCESS (True Positive)")
        else:
            print("RESULT: FAILURE (False Positive)")

def run_classifier_suite(active_games):
    output_dir = "results/classifiers"
    os.makedirs(output_dir, exist_ok=True)
    df = build_dataset(active_games)
    
    experiments = {
        # 'Train_Homogenous_Test_Heterogenous': {
        #     'train': df['Composition'] == 'Homogenous',
        #     'test': df['Composition'] == 'Heterogenous'
        # },
        'Train_Heterogenous_Test_Homogenous': {
            'train': df['Composition'] == 'Heterogenous',
            'test': df['Composition'] == 'Homogenous'
        },
        # 'Train_Lightweight_Test_Heavyweight': {
        #     'train': df['WeightClass'] == 'Lightweight',
        #     'test': df['WeightClass'] == 'Heavyweight'
        # },
        # 'Train_Heavyweight_Test_Lightweight': {
        #     'train': df['WeightClass'] == 'Heavyweight',
        #     'test': df['WeightClass'] == 'Lightweight'
        # }
    }
    
    models = {
        #'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        #'MLP_Net': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42, early_stopping=True),
        #'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        #'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        #'SGD_Classifier': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42),
        'SVM': CalibratedClassifierCV(LinearSVC(dual='auto', random_state=42), cv=3),
        #'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    }
    
    print("\n" + "="*115)
    print(f"{'Classifier Results':^115}")
    print("="*115)
    
   
    for model_name, model_obj in models.items():
        results_to_save = []
        raw_metrics = {'threshold': [], 'precision': [], 'recall': [], 'f1': []}
        print(f"\n### {model_name.upper()} ###")
        print(f"{'EXPERIMENT':<48} | {'THRESHOLD':<10} | {'PRECISION':<14} | {'RECALL':<14} | {'F1':<14}")
        print("-" * 115)
        
        for exp_name, masks in experiments.items():
            train_mask = masks['train']
            test_mask = masks['test']
            
            # The simulator now handles the honest 3-step train/validate/test split internally
            res = simulate_ml_observer(df, train_mask, test_mask, model_obj, exp_name)
            med_th, p_med, p_std, r_med, r_std, f_med, f_std = res
            p_str = f"{p_med:.2f} ± {p_std:.2f}"
            r_str = f"{r_med:.2f} ± {r_std:.2f}"
            f_str = f"{f_med:.2f} ± {f_std:.2f}"

            raw_metrics['threshold'].append(med_th)
            raw_metrics['precision'].append(p_med)
            raw_metrics['recall'].append(r_med)
            raw_metrics['f1'].append(f_med)


            print(f"{exp_name:<48} | > {med_th:<8.2f} | "
              f"{p_str:<14} | "
              f"{r_str:<14} | "
              f"{f_str:<14}")
            results_to_save.append({
                'Experiment': exp_name, 'Precision': p_str, 'Recall': r_str, 'F1': f_str
            })

        # Save results to CSV
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

        #csv_filename = f"{model_name}_results.csv"
        #pd.DataFrame(results_to_save).to_csv(os.path.join(output_dir, csv_filename), index=False)            

def print_classifier_results(results_dir="results/classifiers"):
    """
    Reads and prints the classifier evaluation results from the generated CSV files,
    adding a final row that averages the metrics across all experiments.
    """
    if not os.path.exists(results_dir):
        print(f"\n[Error] Directory '{results_dir}' not found. Have you run the suite yet?")
        return

    csv_files = [f for f in os.listdir(results_dir) if f.endswith('_results.csv')]
    
    if not csv_files:
        print(f"\n[Error] No result CSVs found in '{results_dir}'.")
        return

    print("\n" + "="*110)
    print(f"{'CLASSIFIER RESULTS':^110}")
    print("="*110)

    # Sort files alphabetically so they print in a consistent order
    for file in sorted(csv_files):
        model_name = file.replace('_results.csv', '').replace('_', ' ')
        file_path = os.path.join(results_dir, file)
        
        try:
            df = pd.read_csv(file_path)
            
            # 1. Calculate the averages for the new row
            avg_row = {'Experiment': 'AVERAGE'}
            
            for col in df.columns:
                if col == 'Experiment':
                    continue
                try:
                    # Extract the first float from the formatted string (e.g., "0.606" from "0.606 ± 0.000")
                    vals = df[col].apply(lambda x: float(str(x).split(' ')[0]))
                    avg_val = vals.mean()
                    
                    # Format it cleanly to sit alongside the rest of the column
                    avg_row[col] = f"{avg_val:.3f}"
                except Exception:
                    avg_row[col] = "N/A"
            
            # 2. Append the new row to the dataframe
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            
            # 3. Print the result
            print(f"\n### {model_name.upper()} ###")
            print(df.to_string(index=False, justify='left'))
            
        except Exception as e:
            print(f"Could not read {file}: {e}")
            
    print("\n" + "="*110)
            
    print("\n" + "="*110)


if __name__ == "__main__":
    # CONFIGURATION
    ROOT_DIRECTORY = "results"     
    loader = GameLogLoader(ROOT_DIRECTORY)
    active_games, silent_games = loader.load_all(force_reload=False)
    run_classifier_suite(active_games)
    #print_classifier_results()

