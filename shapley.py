import pandas as pd
import glob
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from config.model_composition import COMPOSITION

plt.switch_backend('Agg')
USE_NEURAL_NET = False 

def normalize_model_name(name):
    clean_name = str(name).split('/')[-1]
    if "Apertus-70B-Instruct-2509" in clean_name:
        clean_name = "Apertus-70B-Instruct-2509"
    
    shorthand_map = {
        "Llama-3.3-70B-Instruct": "Llama 3.3",
        "L3.3-GeneticLemonade-Final-v2-70B": "GeneticLemonade",
        "Hermes-4-70B": "Hermes 4",
        "Qwen2.5-72B-Instruct": "Qwen 2.5",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen 3 Next",
        "Apertus-70B-Instruct-2509": "Apertus 70B",
        "Arcee-Nova": "Arcee Nova",
        "Mixtral-8x7B-Instruct-v0.1-upscaled": "Mixtral Upscaled",
        "Athene-V2-Chat": "Athene V2",
        "HyperNova-60B": "HyperNova"
    }
    return shorthand_map.get(clean_name, clean_name)

def run_shap_analysis(logs_dir="logs", use_neural_net=USE_NEURAL_NET):
    method_name = "Neural Net + Clustering" if use_neural_net else "XGBoost"
    print(f"--- Starting SHAP Contribution Analysis ({method_name}) ---")
    
    # 1. LOAD DATA
    all_data = []
    
    for config in COMPOSITION:
        scen_name = config["name"]
        search_path = os.path.join(logs_dir, scen_name, f"*_{scen_name}_Run*", "stats.csv")
        files = glob.glob(search_path)
        files.sort()
        if len(files) > 100: files = files[:100]
        for filename in files:
            try:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip()
                df["scenario"] = scen_name
                df["game_id"] = os.path.basename(os.path.dirname(filename))
                
                if 'model_name' in df.columns:
                    df['model_name'] = df['model_name'].apply(normalize_model_name)

                alignment_map = {'H': 'Crew', 'B': 'Impostor'}
                if 'alignment' in df.columns:
                    df['alignment'] = df['alignment'].replace(alignment_map)
                
                if 'won_game' in df.columns:
                    df['won_game'] = df['won_game'].astype(int)
                
                all_data.append(df)
            except:
                continue
                
    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.fillna(0) 

    # 2. DEFINE PIPELINES
    roles = ["Crew", "Impostor"]
    
    output_dir = os.path.join(logs_dir, "shap")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for role in roles:
        print(f"\n=== Analyzing Role: {role} ===")
        
        role_df = full_df[full_df['alignment'] == role].copy()
        
        if role_df.empty:
            print(f"No data for {role}, skipping.")
            continue

        features = [
            "correct_votes", "incorrect_votes", "skipped_votes",
            "emergency_meetings", "bodies_reported", 
            "rounds_survived", "num_moves", "votes_received"
        ]
        
        if role == "Crew":
            if "eliminations" in features: features.remove("eliminations")
            features.append("times_eliminated") 
        else:
            if "times_eliminated" in features: features.remove("times_eliminated")
            features.append("eliminations")

        valid_features = [f for f in features if f in role_df.columns]
        X = role_df[valid_features]
        y = role_df['won_game']

        print(f"Training Model on {len(X)} instances using {method_name}...")

        # 3. TRAIN & EXPLAIN
        if use_neural_net:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X) 
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            model.fit(X_scaled, y)
            
            background = shap.kmeans(X, 10)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values_raw = explainer.shap_values(X_scaled, nsamples='auto')
            shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else np.array(shap_values_raw)
        else:
            model = xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=9,
                min_child_weight=3, gamma=0.2, subsample=0.8,
                colsample_bytree=0.8, random_state=42, eval_metric='logloss'
            )
            model.fit(X, y)
            print(f"Accuracy: {model.score(X, y)*100:.2f}%")
            
            explainer = shap.TreeExplainer(model.get_booster(), data=X, model_output="probability", feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)

        # 5. ASSIGN SCORES (SCALED BY 100)
        role_df["contribution_score"] = np.sum(shap_values, axis=1) * 100
        role_df["won_game"] = role_df["won_game"] * 100
        
        # 6. AGGREGATE STATISTICS
        # Define Median Absolute Deviation (MAD) for  variance
        def mad(x): return np.median(np.abs(x - np.median(x)))

        stats = role_df.groupby("model_name").agg(
            Utility_Mean=("contribution_score", "mean"),
            Utility_Median=("contribution_score", "median"),
            Utility_Std=("contribution_score", "std"),
            Utility_MAD=("contribution_score", mad), 
            Win_Rate=("won_game", "mean"),
            Games=("game_id", "count")
        )
        
        # --- 7A. GENERATE LEADERBOARD 1: MEAN ---
        stats_mean = stats.sort_values(by="Utility_Mean", ascending=False).copy()
        stats_mean["Score"] = stats_mean["Utility_Mean"].map('{:+.4f}'.format) + " ± " + stats_mean["Utility_Std"].map('{:.4f}'.format)
        
        table_mean = stats_mean[["Score", "Win_Rate", "Games"]].copy()
        table_mean["Win_Rate"] = table_mean["Win_Rate"].map('{:.1f}%'.format)

        print(f"\n--- {role} LEADERBOARD (MEAN SHAP) ---")
        print(table_mean)
        stats_mean.to_csv(os.path.join(output_dir, f"leaderboard_{role}_MEAN.csv"))

        plt.figure(figsize=(10, len(table_mean) * 0.5 + 2))
        plt.axis('off')
        tbl = plt.table(cellText=table_mean.values, colLabels=table_mean.columns, 
                        rowLabels=table_mean.index, loc='center', cellLoc='center', colColours=["#e6f2ff"] * 3)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 1.5)
        plt.title(f"{role} Leaderboard: Mean SHAP", fontsize=16)
        plt.savefig(os.path.join(output_dir, f"leaderboard_{role}_MEAN.png"), bbox_inches='tight')
        plt.close()

        # --- 7B. GENERATE LEADERBOARD 2: MEDIAN ---
        stats_median = stats.sort_values(by="Utility_Median", ascending=False).copy()
        stats_median["Score"] = stats_median["Utility_Median"].map('{:+.4f}'.format) + " ± " + stats_median["Utility_MAD"].map('{:.4f}'.format)
        
        table_median = stats_median[["Score", "Win_Rate", "Games"]].copy()
        # FIX: Use {:.1f}% here as well
        table_median["Win_Rate"] = table_median["Win_Rate"].map('{:.1f}%'.format)

        print(f"\n--- {role} LEADERBOARD (MEDIAN SHAP) ---")
        print(table_median)
        stats_median.to_csv(os.path.join(output_dir, f"leaderboard_{role}_MEDIAN.csv"))

        plt.figure(figsize=(10, len(table_median) * 0.5 + 2))
        plt.axis('off')
        tbl = plt.table(cellText=table_median.values, colLabels=table_median.columns, 
                        rowLabels=table_median.index, loc='center', cellLoc='center', colColours=["#fff2e6"] * 3)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.2, 1.5)
        plt.title(f"{role} Leaderboard: Median SHAP", fontsize=16)
        plt.savefig(os.path.join(output_dir, f"leaderboard_{role}_MEDIAN.png"), bbox_inches='tight')
        plt.close()

        # --- 8. PLOT 1: WIN RATE vs MEAN SHAP ---
        plt.figure(figsize=(12, 8))
        plt.scatter(stats["Win_Rate"], stats["Utility_Mean"], s=150, c='royalblue', edgecolors='black', alpha=0.8, zorder=3)
        for model_name, row in stats.iterrows():
            plt.annotate(model_name, (row["Win_Rate"], row["Utility_Mean"]), xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
        
        plt.axhline(stats["Utility_Mean"].mean(), color='blue', linestyle='--', alpha=0.3, label="Avg Mean SHAP")
        plt.axvline(stats["Win_Rate"].mean(), color='gray', linestyle='--', alpha=0.3, label="Avg Win Rate")
        
        plt.title(f"Impact vs. Winning ({role})\n(Mean SHAP)", fontsize=16)
        plt.xlabel("Win Rate (%)", fontsize=13)
        plt.ylabel("SHAP Contribution (% Win Probability Added)", fontsize=13)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(output_dir, f"scatter_{role}_MEAN.png"), bbox_inches='tight')
        plt.close()

        # --- 9. PLOT 2: WIN RATE vs MEDIAN SHAP ---
        plt.figure(figsize=(12, 8))
        plt.scatter(stats["Win_Rate"], stats["Utility_Median"], s=150, c='darkorange', edgecolors='black', alpha=0.8, zorder=3)
        for model_name, row in stats.iterrows():
            plt.annotate(model_name, (row["Win_Rate"], row["Utility_Median"]), xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
            
        plt.axhline(stats["Utility_Median"].mean(), color='orange', linestyle='--', alpha=0.3, label="Avg Median SHAP")
        plt.axvline(stats["Win_Rate"].mean(), color='gray', linestyle='--', alpha=0.3, label="Avg Win Rate")
        
        plt.title(f"Impact vs. Winning ({role})\n(Median SHAP)", fontsize=16)
        plt.xlabel("Win Rate (%)", fontsize=13)
        plt.ylabel("SHAP Contribution (% Win Probability Added)", fontsize=13)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(output_dir, f"scatter_{role}_MEDIAN.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    run_shap_analysis(logs_dir='results/experiment_1')