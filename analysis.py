import pandas as pd
import glob
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import joblib
import xgboost as xgb
import re
from scipy.stats import pearsonr, spearmanr
from scipy.special import expit

plt.switch_backend('Agg')

# ==========================================
# 1. CONFIGURATION
# ==========================================
LOGS_DIR = "results"
GLOBAL_MODEL_PATH = "global_game_model.json"
CACHE_FILE_PATH = "full_experiments_game_level.csv"

shift_dir = os.path.join(LOGS_DIR, "shift_analysis")
os.makedirs(shift_dir, exist_ok=True)
# TRAINING CONFIG
TRAIN_EXPERIMENTS = ["experiment_1", "experiment_2", "experiment_3", "experiment_4"]

# INFERENCE CONFIG
TARGET_EXPERIMENTS = ["experiment_1", "experiment_2", "experiment_3", "experiment_4", "experiment_5"] 

# FUZZIFICATION CONFIG
FUZZIFY_THRESHOLD = 0.75 
NUM_BINS = 5            

# --- PLOTTING CONFIG ---
USE_MEDIAN = False  
HW_THRESHOLD = 25  

# Game Constants
MAX_IMPOSTORS = 2
MAX_CREW = 8

MODEL_SIZES = {
    "ArceeNova-72B": 72, "MixtralUpscaled-82B": 82, "AtheneV2-73B": 73,
    "Apertus-70B": 70, "Llama3.3-70B": 70, "Hermes4-70B": 70,
    "HyperNova-60B": 60, "GeneticLemonade-70B": 70, "Qwen2.5-72B": 72,      
    "Qwen3Next-80B": 80, "Qwen3-14B": 14, "gpt-oss-20B": 20, 
    "Llama3.1-8B": 8, "Gemma2-9B": 9, "Qwen2-7B": 7, 
    "Apertus-8B": 8, "Olmo3-7B": 7, "ArceeAgent-7B": 7, "Qwen2.5-7B": 7
}

def normalize_model_name(name):
    clean_name = str(name).split('/')[-1]
    shorthand_map = {
        "Llama-3.3-70B-Instruct": "Llama3.3-70B",
        "L3.3-GeneticLemonade-Final-v2-70B": "GeneticLemonade-70B",
        "Hermes-4-70B": "Hermes4-70B",
        "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen3Next-80B",
        "Apertus-70B-Instruct-2509": "Apertus-70B",
        "Arcee-Nova": "ArceeNova-72B",
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
        "Arcee-Agent": "ArceeAgent-7B",
    }
    for key, val in shorthand_map.items():
        if key in clean_name: return val
    return shorthand_map.get(clean_name, clean_name)

def get_weight_class(model_name):
    size = MODEL_SIZES.get(model_name, 0)
    return "Heavyweight" if size > HW_THRESHOLD else "Lightweight"

# ==========================================
# 2. FEATURE ENGINEERING & PIPELINE
# ==========================================
def engineer_features(df):
    epsilon = 1e-6
    df['Effective_Rounds'] = df['rounds_survived'].replace(0, 1)
    
    total_votes_cast = df['correct_votes'] + df['incorrect_votes']
    df['Voting_Precision'] = df['correct_votes'] / (total_votes_cast + epsilon)
    
    total_ops = total_votes_cast + df['skipped_votes']
    df['Indecision_Rate'] = df['skipped_votes'] / (total_ops + epsilon)
    
    df['Num_Moves'] = df['num_moves']
    df['Votes_Received'] = df['votes_received']
    df['Emergency_Meetings'] = df['emergency_meetings']
    df['Body_Reporting'] = df['bodies_reported']
    
    if 'eliminations' in df.columns:
        df['Eliminations'] = df['eliminations']
    else:
        df['Eliminations'] = 0.0
    return df

def pivot_to_game_level(player_df):
    features_to_use = ["Effective_Rounds", "Voting_Precision", "Indecision_Rate", "Num_Moves", "Votes_Received", 
                       "Emergency_Meetings", "Body_Reporting", "Eliminations"]
    
    games_list = []
    grouped = player_df.groupby(['game_id', 'experiment_id'])
    
    for (game_id, exp_id), group in grouped:
        row_dict = {'game_id': game_id, 'experiment_id': exp_id, 'Crew_Win': 0}
        
        impostors = group[group['alignment'].str.contains("Imposter", case=False)].copy().reset_index(drop=True)
        crew = group[group['alignment'].str.contains("Crew", case=False)].copy().reset_index(drop=True)
        
        if crew['won_game'].sum() > 0: row_dict['Crew_Win'] = 1
            
        for i in range(MAX_IMPOSTORS):
            slot = f"Imp{i+1}"
            if i < len(impostors):
                row_dict[f"{slot}_Model"] = impostors.loc[i, 'model_name']
                for feat in features_to_use: row_dict[f"{slot}_{feat}"] = impostors.loc[i, feat]
            else:
                row_dict[f"{slot}_Model"] = "None"
                for feat in features_to_use: row_dict[f"{slot}_{feat}"] = 0.0

        for i in range(MAX_CREW):
            slot = f"Crew{i+1}"
            if i < len(crew):
                row_dict[f"{slot}_Model"] = crew.loc[i, 'model_name']
                for feat in features_to_use: row_dict[f"{slot}_{feat}"] = crew.loc[i, feat]
            else:
                row_dict[f"{slot}_Model"] = "None"
                for feat in features_to_use: row_dict[f"{slot}_{feat}"] = 0.0
                    
        games_list.append(row_dict)
    return pd.DataFrame(games_list)

def process_data_pipeline(raw_df):
    df_eng = engineer_features(raw_df)
    df_game = pivot_to_game_level(df_eng)
    return df_game

def load_data_from_dirs(root_dir, experiments_list):
    all_data = []
    print(f"\n--- Loading Data for: {experiments_list} ---")
    
    for exp_id in experiments_list:
        exp_path = os.path.join(root_dir, exp_id)
        if not os.path.exists(exp_path):
            print(f"WARNING: Directory not found: {exp_path}")
            continue
            
        for root, dirs, files in os.walk(exp_path):
            if "stats.csv" in files:
                try:
                    df = pd.read_csv(os.path.join(root, "stats.csv"))
                    df.columns = df.columns.str.strip()
                    df["experiment_id"] = exp_id
                    df["game_id"] = exp_id + "_" + os.path.basename(root)
                    
                    if 'model_name' in df.columns: df['model_name'] = df['model_name'].apply(normalize_model_name)
                    alignment_map = {'H': 'Crew', 'B': 'Imposter'}
                    if 'alignment' in df.columns: df['alignment'] = df['alignment'].replace(alignment_map)
                    all_data.append(df)
                except Exception as e: 
                    print(f"Error loading {root}: {e}")

    if not all_data: return pd.DataFrame()
    full_raw_df = pd.concat(all_data, ignore_index=True).fillna(0)
    return process_data_pipeline(full_raw_df)

def load_feature_columns_from_cache():
    if os.path.exists(CACHE_FILE_PATH):
        df = pd.read_csv(CACHE_FILE_PATH, nrows=1)
        return [c for c in df.columns if re.match(r'^(Imp|Crew)\d+_', c) and not c.endswith("_Model")]
    return None 

# ==========================================
# 3. CACHE & MODEL MANAGEMENT
# ==========================================
def build_and_save_cache():
    print(f"\n[SYSTEM] Building dataset from scratch using: {TRAIN_EXPERIMENTS}...")
    df = load_data_from_dirs(LOGS_DIR, TRAIN_EXPERIMENTS)
    if not df.empty:
        df.to_csv(CACHE_FILE_PATH, index=False)
        return df
    return None

def train_and_save_model(training_df):
    print(f"\n[SYSTEM] Retraining new model...")
    feature_cols = [c for c in training_df.columns if re.match(r'^(Imp|Crew)\d+_', c) and not c.endswith("_Model")]
    X, y = training_df[feature_cols], training_df['Crew_Win']
    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    model.save_model(GLOBAL_MODEL_PATH)
    return model

def load_trained_model():
    model = xgb.XGBClassifier()
    model.load_model(GLOBAL_MODEL_PATH)
    return model

# ==========================================
# 4. SHAP & AGGREGATION (M3)
# ==========================================
def apply_war_normalization(results_df):
    replacement_levels = results_df.groupby('role')['impact_score'].mean().to_dict()
    results_df['WAR'] = results_df.apply(lambda row: row['impact_score'] - replacement_levels.get(row['role'], 0), axis=1)
    return results_df

def compute_shap_and_probs(df, model, feature_cols):
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0
            
    X = df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray): base_val = base_val[0]
    
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols)
    results = []
    
    for idx, row in df.iterrows():
        exp_id, won, game_id = row['experiment_id'], row['Crew_Win'], row['game_id']
        
        for role_prefix, role_name, mult in [('Imp', 'Imposter', -100), ('Crew', 'Crew', 100)]:
            max_slots = MAX_IMPOSTORS if role_prefix == 'Imp' else MAX_CREW
            for i in range(1, max_slots + 1):
                slot = f"{role_prefix}{i}"
                model_name = row.get(f"{slot}_Model", "None")
                if model_name == "None": continue
                
                cols = [c for c in feature_cols if c.startswith(slot+"_")]
                prob_impact = expit(base_val + shap_df.iloc[idx][cols].sum()) - expit(base_val)
                
                # Fetch playstyle SHAP
                feat_map = {'Eliminations': 'SHAP_Eliminations', 'Votes_Received': 'SHAP_Votes_Received',
                            'Num_Moves': 'SHAP_Num_Moves', 'Voting_Precision': 'SHAP_Voting_Precision'}
                
                entry = {"game_id": game_id, "experiment_id": exp_id, "model_name": model_name, "role": role_name, 
                         "impact_score": prob_impact * mult, "won_game": won if role_name == 'Crew' else (1 if won==0 else 0)}
                
                for f_name, col_key in feat_map.items():
                    col_id = f"{slot}_{f_name}"
                    if col_id in shap_df.columns:
                        entry[col_key] = shap_df.at[idx, col_id] * (1 if role_name == 'Crew' else -1)
                
                results.append(entry)
    return pd.DataFrame(results)

def calculate_m3_stats(df):
    """Method 3: MVP determined by largest average contribution."""
    return df.groupby(['model_name', 'role']).agg(
        Win_Rate=('won_game', 'mean'),
        Avg_WAR=('WAR', 'mean'),
        Games=('won_game', 'count')
    ).reset_index()

# ==========================================
# 5. ANALYSIS & REPORTING
# ==========================================
def analyze_correlations(df, context_name="Global"):
    print(f"\n--- Correlation Analysis ({context_name}) ---")
    stats = calculate_m3_stats(df)
    for role in ['Crew', 'Imposter']:
        rd = stats[stats['role'] == role]
        if len(rd) < 2: continue
        r_shap, _ = pearsonr(rd['Win_Rate'], rd['Avg_WAR'])
        print(f"Role: {role}\n  Win Rate vs Avg WAR (M3) Correlation: {r_shap:.4f}")

def generate_ranking_text_report(df, output_dir, filename="ranking_report.txt", glob=False):
    stats = df.groupby(['model_name', 'role']).agg(
        Win_Rate=('won_game', 'mean'),
        Avg_WAR=('WAR', 'mean'),
        Std_WAR=('WAR', 'std'), 
        Games=('won_game', 'count')
    ).reset_index()

    stats['Weight_Class'] = stats['model_name'].apply(get_weight_class)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "w") as f:
        f.write(f"MVP RANKING REPORT\nMethod: M3 (Average RAW contribution)\n" + "="*75 + "\n\n")
        for role in ['Crew', 'Imposter']:
            f.write(f"--- {role.upper()} RANKINGS ---\n")
            role_data = stats[stats['role'] == role].sort_values('Avg_WAR', ascending=False).reset_index(drop=True)
            
            # Added Std Dev to header
            header = f"{'Rank':<5} {'Model Name':<30} {'Avg WAR(%) ± Std':<20} {'Win Rate(%)':<12} {'Games':<5}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for idx, row in role_data.iterrows():
                war_str = f"{row['Avg_WAR']:>6.2f} ± {row['Std_WAR']:<5.2f}"
                f.write(f"{idx+1:<5} {row['model_name']:<30} {war_str:<20} {row['Win_Rate']*100:<12.1f} {row['Games']:<5}\n")
            f.write("\n")

        f.write("="*75 + "\n")
        f.write("WEIGHT CLASS PERFORMANCE SUMMARY\n")
        f.write("(Aggregated Average WAR & Std Dev per Class)\n")
        f.write("-" * 75 + "\n")
        
        summary = stats.groupby(['role', 'Weight_Class'])[['Avg_WAR', 'Std_WAR']].mean().reset_index()
        summary = summary.sort_values(by=['role', 'Weight_Class'])
        
        sum_header = f"{'Role':<10} {'Weight Class':<15} {'Mean Avg WAR':<15} {'Mean Std Dev':<15}"
        f.write(sum_header + "\n")
        f.write("-" * len(sum_header) + "\n")
        
        for idx, row in summary.iterrows():
            f.write(f"{row['role']:<10} {row['Weight_Class']:<15} {row['Avg_WAR']:<15.2f} {row['Std_WAR']:<15.2f}\n")
    # 2. ROLE-SEPARATED P4P SCATTER PLOTS (1x2)
    if glob:        
        stats['Size'] = stats['model_name'].map(MODEL_SIZES)
        stats['Weight_Class'] = stats['model_name'].apply(get_weight_class)
        plot_df = stats.dropna(subset=['Size'])

        # Create 1x2 Subplots with independent Y-axes
        fig, axes = plt.subplots(1, 2, figsize=(22, 10))
        sns.set_theme(style="whitegrid")
        
        wc_colors = {"Heavyweight": "#4C72B0", "Lightweight": "#DD8452"} # Muted Blue and Orange
        
        for i, role in enumerate(['Crew', 'Imposter']):
            ax = axes[i]
            role_subset = plot_df[plot_df['role'] == role].copy()
            
            if role_subset.empty: continue
            
            # Using marker size (s) to represent Standard Deviation
            # We scale the Std_WAR to make it visually discernible
            # Larger circle = Higher Variance (Less Stable)
            # Smaller circle = Lower Variance (Highly Consistent)
            sns.scatterplot(
                data=role_subset,
                x="Size",
                y="Avg_WAR",
                hue="Weight_Class",
                palette=wc_colors,
                size="Std_WAR",
                sizes=(100, 1000), # Size range for the "Stability" bubbles
                marker="o",        # Uniform icons as requested
                alpha=0.6,
                ax=ax,
                legend=False       # We will create a custom unified legend
            )
            
            # Annotate model names
            for _, row in role_subset.iterrows():
                ax.text(
                    row['Size'] + 1.2, 
                    row['Avg_WAR'], 
                    row['model_name'], 
                    fontsize=9, fontweight='bold', alpha=0.8, va='center'
                )
                
            ax.set_title(f"{role.upper()} Strategic Impact", fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel("Model Parameters (Billions)", fontsize=12)
            ax.set_ylabel("Average WAR (%)", fontsize=12)
            ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)

        hw_color = mlines.Line2D([], [], color=wc_colors["Heavyweight"], marker='o', linestyle='None', 
                                  markersize=12, label='Heavyweight (60B+)')
        lw_color = mlines.Line2D([], [], color=wc_colors["Lightweight"], marker='o', linestyle='None', 
                                  markersize=12, label='Lightweight (7B-20B)')
        
        
        fig.legend(handles=[hw_color, lw_color], 
                   title="Legend", title_fontsize='13',
                   loc='center right', bbox_to_anchor=(0.99, 0.5), fontsize=11)

        plt.tight_layout()
        plt.subplots_adjust(right=0.88, top=0.9) 
        
        plot_filename = filename.replace(".txt", "_p4p_role_split.png")
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()
        print(f"Size Role-split scatter plot saved to: {os.path.join(output_dir, plot_filename)}")
# ==========================================
# 6. PLOTTING
# ==========================================
def create_unified_legend(unique_models, model_color_map):
    legend_handles = []
    markers = {"Heavyweight": "o", "Lightweight": "s"}
    for model in unique_models:
        wc = get_weight_class(model)
        legend_handles.append(mlines.Line2D([], [], color='white', marker=markers.get(wc, 'o'), 
                               markerfacecolor=model_color_map.get(model, 'gray'), markersize=10, label=model))
    return legend_handles

def plot_shap_impact_matrix(results_df, output_dir, suffix=""):
    summary = results_df.groupby(['model_name', 'role'])['WAR'].mean().unstack().reset_index()
    summary["Size_Param"] = summary["model_name"].map(MODEL_SIZES)
    summary = summary.dropna(subset=["Size_Param"]).rename(columns={'Crew': 'Crew_Impact', 'Imposter': 'Imp_Impact'})
    summary["Weight_Class"] = summary["model_name"].apply(get_weight_class)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    models = sorted(summary['model_name'].unique())
    color_map = dict(zip(models, sns.color_palette("husl", n_colors=len(models))))
    
    sns.scatterplot(data=summary, x="Imp_Impact", y="Crew_Impact", size="Size_Param", sizes=(200, 1200),
                    hue="model_name", style="Weight_Class", markers={"Heavyweight": "o", "Lightweight": "s"}, 
                    palette=color_map, alpha=0.9, legend=False)
    
    for _, row in summary.iterrows():
        plt.text(row["Imp_Impact"]+0.01, row["Crew_Impact"]+0.01, row["model_name"], fontsize=9, weight='bold')
    
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5); plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.legend(handles=create_unified_legend(models, color_map), title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"impact_matrix{suffix}.png")); plt.close()

def plot_playstyle_clusters(results_df, output_dir, suffix=""):
    stats = results_df.groupby(['model_name', 'role']).mean(numeric_only=True).reset_index()
    stats['Weight_Class'] = stats['model_name'].apply(get_weight_class)
    models = sorted(stats['model_name'].unique())
    color_map = dict(zip(models, sns.color_palette("husl", n_colors=len(models))))
    
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    for i, (role, x_col, y_col, title, lbl_tr, lbl_bl) in enumerate([
        ('Imposter', 'SHAP_Votes_Received', 'SHAP_Eliminations', 'Impostor: Aggression vs. Stealth', 'Assassin', 'Liability'),
        ('Crew', 'SHAP_Num_Moves', 'SHAP_Voting_Precision', 'Crew: Activity vs. Intelligence', 'Detective', 'Passenger')]):
        
        rd = stats[stats['role'] == role]
        if rd.empty: continue
        sns.scatterplot(data=rd, x=x_col, y=y_col, hue="model_name", style="Weight_Class", 
                        markers={"Heavyweight": "o", "Lightweight": "s"}, palette=color_map, s=300, ax=axes[i], legend=False)
        axes[i].axhline(0, color='black', alpha=0.3); axes[i].axvline(0, color='black', alpha=0.3)
        axes[i].text(0.95, 0.95, lbl_tr, transform=axes[i].transAxes, ha='right', color='green', fontweight='bold')
        axes[i].text(0.05, 0.05, lbl_bl, transform=axes[i].transAxes, color='red', fontweight='bold')
        for _, row in rd.iterrows(): axes[i].text(row[x_col], row[y_col], row['model_name'], fontsize=8)

    fig.legend(handles=create_unified_legend(models, color_map), loc='center right', bbox_to_anchor=(0.99, 0.5))
    plt.tight_layout(); plt.subplots_adjust(right=0.85); plt.savefig(os.path.join(output_dir, f"playstyle_clusters{suffix}.png")); plt.close()

def plot_shap_difference(df, exp_a, exp_b, target_models, title_suffix):
    """
    Generates a Diverging Bar Chart showing the gain/loss in Strategic Impact (WAR)
    when moving from Experiment A to Experiment B.
    """
    # 1. Filter Data
    subset = df[
        (df['experiment_id'].isin([exp_a, exp_b])) & 
        (df['model_name'].isin(target_models))
    ].copy()
    
    if subset.empty:
        print(f"Skipping {title_suffix}: No data found.")
        return

    # 2. Pivot to Calculate Delta
    pivot = subset.groupby(['model_name', 'role', 'experiment_id'])['WAR'].mean().unstack()
    
    if exp_a not in pivot.columns or exp_b not in pivot.columns:
        print(f"Skipping {title_suffix}: Models do not overlap between experiments.")
        return

    pivot['Delta'] = pivot[exp_b] - pivot[exp_a]
    pivot = pivot.dropna(subset=['Delta']).reset_index()
    
    if pivot.empty:
        print(f"Skipping {title_suffix}: No matching roles found.")
        return

    # 3. Create Plot (1x2 for Crew/Imposter)
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    sns.set_theme(style="whitegrid")

    for i, role in enumerate(['Crew', 'Imposter']):
        ax = axes[i]
        role_data = pivot[pivot['role'] == role].copy()
        
        if role_data.empty:
            ax.set_visible(False)
            continue
                    
        # Color Logic: Green for Gain, Red for Loss
        role_data['Color'] = role_data['Delta'].apply(lambda x: '#2ca02c' if x >= 0 else '#d62728')
        
        # Plot Horizontal Bars
        bars = ax.barh(
            y=role_data['model_name'], 
            width=role_data['Delta'], 
            color=role_data['Color'], 
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        padding = x_range * 0.01
        # Add Value Labels next to bars
        for bar, value in zip(bars, role_data['Delta']):
            # Position text to the right of positive bars, left of negative bars
            x_pos = value + (padding if value >= 0 else -padding)
            # Adjust alignment based on sign
            ha = 'left' if value >= 0 else 'right'
            
            ax.text(
                x_pos, 
                bar.get_y() + bar.get_height()/2, 
                f"{value:+.2f}%", 
                va='center', 
                ha=ha, 
                fontsize=10,
                fontweight='bold',
                color='black'
            )

        # Formatting
        ax.axvline(0, color='black', linewidth=1.5, alpha=0.8) # Zero line
        ax.set_title(f"{role} Shift", fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel(f"Change in Win Probability Added (WAR %)\n(Exp {exp_a[-1]} $\\rightarrow$ Exp {exp_b[-1]})", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Remove Y label as model names are self-explanatory
        ax.set_ylabel("")
        
        # Add Quadrant Labels
        ymin, ymax = ax.get_ylim()
        xmax = max(abs(role_data['Delta'].max()), abs(role_data['Delta'].min())) * 1.1
        ax.set_xlim(-xmax, xmax) # Center the zero line visually if possible, or just pad
        
    plt.suptitle(f"Population Shift Impact: {title_suffix}", fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save
    shift_dir = os.path.join(LOGS_DIR, "shift_analysis")
    os.makedirs(shift_dir, exist_ok=True)
    filename = f"shift_bars_{title_suffix}.png"
    plt.savefig(os.path.join(shift_dir, filename), bbox_inches='tight')
    plt.close()




def plot_feature_breakdown(df, model, feature_cols, target_models, output_dir, filename_suffix=""):
    """
    Calculates behavioral rankings (1-10) for the entire population but 
    filters the final heatmap to only display the specified outlier models.
    """
    print(f"--- Generating Feature Rankings for Outliers: {target_models} ---")
    
    if df.empty:
        print("Empty DataFrame. Skipping breakdown.")
        return

    # 1. Compute Raw SHAP Values for the context
    X = df[feature_cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 2. Map SHAP values to every model in the dataset to establish global ranks
    breakdown_data = []
    feat_idx_map = {feat: i for i, feat in enumerate(feature_cols)}

    for i, (idx, row) in enumerate(df.iterrows()):
        # Iterate through all slots (Imposter 1-2, Crew 1-8)
        for role_prefix, role_name, max_slots, mult in [('Imp', 'Imposter', MAX_IMPOSTORS, -1), 
                                                        ('Crew', 'Crew', MAX_CREW, 1)]:
            for s in range(1, max_slots + 1):
                m_name = row.get(f"{role_prefix}{s}_Model")
                if pd.isna(m_name) or m_name == "None": continue
                
                for f_name in feature_cols:
                    if f_name.startswith(f"{role_prefix}{s}_"):
                        clean_feat = f_name.replace(f"{role_prefix}{s}_", "")
                        val = shap_values[i][feat_idx_map[f_name]] * mult
                        breakdown_data.append({
                            "model_name": m_name, "role": role_name, 
                            "Feature": clean_feat, "Impact": val
                        })

    # 3. Aggregate and Rank across the WHOLE population
    full_summary = pd.DataFrame(breakdown_data)
    summary = full_summary.groupby(['model_name', 'role', 'Feature'])['Impact'].mean().reset_index()
    summary['Weight_Class'] = summary['model_name'].apply(get_weight_class)

    # Calculate ranks 1-10 within each Role/Weight Class/Feature group
    summary['Feature_Rank'] = summary.groupby(['role', 'Weight_Class', 'Feature'])['Impact'].rank(ascending=False, method='min')

    # 4. FILTER for only the specified Outlier Models
    outlier_summary = summary[summary['model_name'].isin(target_models)].copy()

    # 5. Plot Heatmaps for Outliers
    for role in ['Crew', 'Imposter']:
        for wc in ['Heavyweight', 'Lightweight']:
            subset = outlier_summary[(outlier_summary['role'] == role) & (outlier_summary['Weight_Class'] == wc)]
            if subset.empty: continue
            
            # Pivot: Rows=Model Name, Cols=Game Behavior, Values=Rank
            pivot = subset.pivot(index='model_name', columns='Feature', values='Feature_Rank')
            
            plt.figure(figsize=(14, len(pivot) * 1.2 + 2))
            sns.set_theme(style="white")
            
            # Heatmap with reversed colormap (1 is Green/Best, 10 is Red/Worst)
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", center=5.5, 
                        cbar_kws={'label': 'Feature Rank (1 = Top Performance)'})
            
            plt.title(f"Strategic Profile: {wc} {role} Outliers", fontsize=16, fontweight='bold', pad=20)
            plt.ylabel("Outlier Model", fontweight='bold')
            plt.xlabel("Behavioral Rank (Lower is Better)", fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"outlier_ranks_{wc.lower()}_{role.lower()}{filename_suffix}.png")
            plt.savefig(save_path)
            plt.close()
    print(f"Filtered outlier heatmaps saved to {output_dir}")

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    if os.path.exists(CACHE_FILE_PATH):
        training_df = pd.read_csv(CACHE_FILE_PATH)
    else:
        training_df = build_and_save_cache()

    if not os.path.exists(GLOBAL_MODEL_PATH):
        train_and_save_model(training_df)

    cached_exps = training_df['experiment_id'].unique() if training_df is not None else []
    missing_exps = [e for e in TARGET_EXPERIMENTS if e not in cached_exps]
    inference_df = training_df[training_df['experiment_id'].isin(TARGET_EXPERIMENTS)].copy()
    
    if missing_exps:
        new_data = load_data_from_dirs(LOGS_DIR, missing_exps)
        inference_df = pd.concat([inference_df, new_data], ignore_index=True)
        
    if not inference_df.empty:
        feature_cols = load_feature_columns_from_cache()
        model = load_trained_model()
        res = compute_shap_and_probs(inference_df, model, feature_cols)
        res = apply_war_normalization(res)
        
        # 1. PER-EXPERIMENT
        for exp_id in TARGET_EXPERIMENTS:
            subset = res[res['experiment_id'] == exp_id].copy()
            if not subset.empty:
                exp_dir = os.path.join(LOGS_DIR, exp_id, "shap_analysis")
                os.makedirs(exp_dir, exist_ok=True)
                plot_shap_impact_matrix(subset, exp_dir)
                plot_playstyle_clusters(subset, exp_dir)
                generate_ranking_text_report(subset, exp_dir, filename=f"ranking_{exp_id}.txt")

    # 2. HEAVYWEIGHT GLOBAL (Exp 1 & 2)
    hw_res = res[res['experiment_id'].isin(['experiment_1', 'experiment_2'])].copy()
    if not hw_res.empty:
        hw_dir = os.path.join(LOGS_DIR, "heavyweight_global")
        os.makedirs(hw_dir, exist_ok=True)
        plot_shap_impact_matrix(hw_res, hw_dir, suffix="_hw_global")
        plot_playstyle_clusters(hw_res, hw_dir, suffix="_hw_global")
        generate_ranking_text_report(hw_res, hw_dir, filename="hw_global_ranking.txt")

    # 3. LIGHTWEIGHT GLOBAL (Exp 3 & 4)
    lw_res = res[res['experiment_id'].isin(['experiment_3', 'experiment_4'])].copy()
    if not lw_res.empty:
        lw_dir = os.path.join(LOGS_DIR, "lightweight_global")
        os.makedirs(lw_dir, exist_ok=True)
        plot_shap_impact_matrix(lw_res, lw_dir, suffix="_lw_global")
        plot_playstyle_clusters(lw_res, lw_dir, suffix="_lw_global")
        generate_ranking_text_report(lw_res, lw_dir, filename="lw_global_ranking.txt")

    p4p_res = res[res['experiment_id'].isin(['experiment_1', 'experiment_2', 'experiment_3', 'experiment_4'])].copy()
    print(f"\n=== Generating Final Rankings (Experiments 1 thru 4 Combined) ===")
    total_dir = os.path.join(LOGS_DIR, "full_study_analysis")
    os.makedirs(total_dir, exist_ok=True)
    generate_ranking_text_report(p4p_res, total_dir, filename="full_study_ranking.txt", glob=True)

    # --- DEFINE GROUPS ---
    # Identifying models based on your metadata
    all_models = res['model_name'].unique()
    hw_models = [m for m in all_models if get_weight_class(m) == "Heavyweight"]
    lw_models = [m for m in all_models if get_weight_class(m) == "Lightweight"]        
    # 1. Heavyweight Shift (Exp 1 -> Exp 2)
    plot_shap_difference(res, "experiment_1", "experiment_2", hw_models, "Heavyweights")
    
    # 2. Lightweight Shift (Exp 3 -> Exp 4)
    plot_shap_difference(res, "experiment_3", "experiment_4", lw_models, "Lightweights")

    
    print("\n=== Running Grouped Feature Breakdown for Outliers ===")
        
    outliers = [
        "ArceeAgent-7B", 
        "GeneticLemonade-70B", 
        "Apertus-70B", 
        "Qwen3Next-80B"
    ]
    norm_outliers = [normalize_model_name(x) for x in outliers]

        # GROUP 1: Heavyweight Context (Exp 1 & 2)
        # ------------------------------------------------------
    hw_exps = ['experiment_1', 'experiment_2']
    hw_df = inference_df[inference_df['experiment_id'].isin(hw_exps)].copy()
    
    if not hw_df.empty:
        hw_dir = os.path.join(LOGS_DIR, "heavyweight_global")
        if not os.path.exists(hw_dir): os.makedirs(hw_dir)
        
        print(f"\n>> Generating Feature Breakdown for Heavyweight Context (Exp 1 & 2)...")
        plot_feature_breakdown(hw_df, model, feature_cols, norm_outliers, hw_dir)

    # GROUP 2: Lightweight Context (Exp 3 & 4)
    # ------------------------------------------------------
    lw_exps = ['experiment_3', 'experiment_4']
    lw_df = inference_df[inference_df['experiment_id'].isin(lw_exps)].copy()
    
    if not lw_df.empty:
        lw_dir = os.path.join(LOGS_DIR, "lightweight_global")
        if not os.path.exists(lw_dir): os.makedirs(lw_dir)
        
        print(f"\n>> Generating Feature Breakdown for Lightweight Context (Exp 3 & 4)...")
        plot_feature_breakdown(lw_df, model, feature_cols, norm_outliers, lw_dir)