import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from config.model_composition import COMPOSITION 

plt.switch_backend('Agg')

def normalize_model_name(name):
    """
    Normalizes model names for cleaner plotting labels.
    """
    clean_name = str(name).split('/')[-1]
    
    # Handle specific edge cases
    if "Apertus-70B-Instruct-2509" in clean_name:
        clean_name = "Apertus-70B-Instruct-2509"
    
    shorthand_map = {
        "Llama-3.3-70B-Instruct": "Llama3.3-70B",
        "L3.3-GeneticLemonade-Final-v2-70B": "GeneticLemonade-70B",
        "Hermes-4-70B": "Hermes4-70B",
        "Qwen2.5-72B-Instruct": "Qwen2.5-72B",
        "Qwen3-Next-80B-A3B-Instruct": "Qwen3Next",
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

    return shorthand_map.get(clean_name, clean_name)

def render_mpl_table(data, col_width=6.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
    """
    Renders a pandas dataframe as a matplotlib image.
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax


def plot_win_rates(df, output_dir):
    """Generates the Double Bar Chart for Win Rates."""
    print("Generating Win Rate Charts...")
    stats = (
        df.groupby(["model_name", "alignment"])["Games Won"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    stats.rename(columns={"mean": "Win_Rate_Num"}, inplace=True)

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Sort for cleaner chart
    stats = stats.sort_values(by="Win_Rate_Num", ascending=False)

    chart = sns.barplot(
        data=stats,
        x="model_name",
        y="Win_Rate_Num",
        hue="alignment",
        palette={"Crew": "#3498db", "Imposter": "#e74c3c"} 
    )

    plt.title("Win Rate Comparison: Crew vs Imposter", fontsize=18)
    plt.ylabel("Win Rate (0.0 - 1.0)", fontsize=14)
    plt.xlabel("Model Name", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(title="Role")

    for container in chart.containers:
        chart.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "chart_win_rates.png")
    plt.savefig(chart_path)
    plt.close()

    # Save CSV
    csv_path = os.path.join(output_dir, "HW_Homogenous_WR.csv")
    stats[["model_name", "alignment",  "Win_Rate_Num"]].to_csv(csv_path, index=False)

def plot_stat_tables(df, output_dir):
    """Generates PNG tables for Crew and Imposter stats."""
    print("Generating Stat Tables...")
    stats = (
        df.groupby(["model_name", "alignment"])["Games Won"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    stats.rename(columns={
        "model_name": "Model",
        "alignment": "Role",
        "count": "Total Games",
        "sum": "Wins",
        "mean": "Win_Rate_Num"
    }, inplace=True)
    
    stats["Losses"] = stats["Total Games"] - stats["Wins"]
    stats["Win Rate"] = stats["Win_Rate_Num"].apply(lambda x: f"{x:.2%}")
    table_df = stats[["Model", "Role", "Wins", "Losses", "Win Rate"]]

    # Crew Table
    crew_data = table_df[table_df["Role"].isin(["Crew"])].sort_values(by="Wins", ascending=False)
    if not crew_data.empty:
        render_mpl_table(crew_data, header_columns=0, col_width=6.0)
        p = os.path.join(output_dir, "table_stats_crew.png")
        plt.savefig(p, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # Imposter Table
    imp_data = table_df[table_df["Role"].isin(["Imposter"])].sort_values(by="Wins", ascending=False)
    if not imp_data.empty:
        render_mpl_table(imp_data, header_columns=0, col_width=6.0)
        p = os.path.join(output_dir, "table_stats_imposter.png")
        plt.savefig(p, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def plot_imposter_pairings(full_df, output_dir):
    """Generates Imposter Synergy Heatmap and Pairing Table."""
    print("Generating Imposter Pairings Analysis...")
    imp_df = full_df[full_df["alignment"].isin(["Impostor", "Imposter", "Byzantine"])].copy()
    
    if imp_df.empty:
        print("No Imposter data found.")
        return

    # Create Pairing Data
    game_pairings = []
    for game_id, group in imp_df.groupby("game_id"):
        models = sorted(group["model_name"].tolist())
        if len(models) == 2:
            game_pairings.append({
                "model_a": models[0],
                "model_b": models[1],
                "won": group["Games Won"].iloc[0]
            })
    
    if not game_pairings:
        return

    pairings_df = pd.DataFrame(game_pairings)

    model_win_rates = imp_df.groupby("model_name")["Games Won"].mean().sort_values(ascending=False)
    
    paired_models = set(pairings_df["model_a"].unique()) | set(pairings_df["model_b"].unique())
    sorted_models = [m for m in model_win_rates.index if m in paired_models]

    # Heatmap Logic
    matrix = pd.DataFrame(index=sorted_models, columns=sorted_models, dtype=float)
    agg_pairs = pairings_df.groupby(["model_a", "model_b"])["won"].mean()
    
    for (m1, m2), win_rate in agg_pairs.items():
        matrix.at[m1, m2] = win_rate
        matrix.at[m2, m1] = win_rate 

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white") 
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    
    sns.heatmap(
        matrix.iloc[1:, :-1], 
        mask=mask[1:, :-1],
        annot=True, fmt=".0%", cmap="RdYlGn",    
        linewidths=.5, linecolor='gray', square=True
    )
    plt.title("Impostor Synergy Heatmap (Sorted by Win Rate)", fontsize=16)
    plt.tight_layout()
    p = os.path.join(output_dir, "chart_imposter_heatmap.png")
    plt.savefig(p)
    plt.close()


def perform_feature_analysis(df, output_dir, use_mean=False, 
                             show_bar=True, show_point=True, show_table=True):
    """
    Analyzes numerical features.
    
    Args:
        use_mean (bool): True=Mean/Std, False=Median/MAD.
        show_bar (bool): Toggle Voting Accuracy Bar Chart.
        show_point (bool): Toggle Point Plots for features.
        show_table (bool): Toggle the summary table.
    """
    metric_label = "Mean" if use_mean else "Median"
    variation_label = "StdDev" if use_mean else "MAD"
    print(f"Generating Feature Analysis ({metric_label} & {variation_label})...")
    
    # 1. Feature Engineering
    df['total_votes'] = df['correct_votes'] + df['incorrect_votes']
    df['voting_accuracy'] = np.where(
        df['total_votes'] > 0, 
        df['correct_votes'] / df['total_votes'], 
        np.nan 
    )

    features = [
        "voting_accuracy",      
        "votes_received", 
        "skipped_votes",
        "num_moves", 
        "rounds_survived", 
        "emergency_meetings", 
        "bodies_reported",
        "eliminations",         
        "times_eliminated",  
        "ejections"      
    ]
    
    valid_features = [f for f in features if f in df.columns]
    
    # Store data for Table
    table_data_collector = []

    # Get unique sorted models
    unique_models = sorted(df['model_name'].unique())
    x_indices = np.arange(len(unique_models))

    for feature in valid_features:

        if feature == "voting_accuracy":
            # Aggregate for Calculation
            agg_df = df.groupby(["model_name", "alignment"])[["correct_votes", "incorrect_votes"]].sum().reset_index()
            agg_df["total"] = agg_df["correct_votes"] + agg_df["incorrect_votes"]
            agg_df["accuracy"] = np.where(agg_df["total"] > 0, agg_df["correct_votes"] / agg_df["total"], 0.0)
            
            # Store for Table (Accuracy has no variation in this Aggregated view)
            for _, row in agg_df.iterrows():
                table_data_collector.append({
                    "Model": row["model_name"],
                    "Role": row["alignment"],
                    "Feature": "Voting Acc.", # Shorten for table
                    "Value": row["accuracy"],
                    "Variation": 0.0 # No std/mad for single agg number
                })

            if show_bar:
                plt.figure(figsize=(14, 8))
                sns.set_theme(style="whitegrid")
                sns.barplot(
                    data=agg_df, x="model_name", y="accuracy", hue="alignment",
                    palette={"Crew": "#3498db", "Imposter": "#e74c3c", "Impostor": "#e74c3c", "Byzantine": "#e74c3c"},
                    edgecolor="white"
                )
                plt.title("Global Voting Accuracy (Aggregated)", fontsize=16)
                plt.ylim(0, 1.05)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title="Role", loc='best')
                plt.tight_layout()
                p = os.path.join(output_dir, f"feature_bar_{feature}.png")
                plt.savefig(p)
                plt.close()

        else:
            if feature in ["correct_votes", "incorrect_votes"]: continue

            # Calculate Stats
            if use_mean:
                stats_df = df.groupby(["model_name", "alignment"])[feature].agg(
                    Center='mean', Variation='std'
                ).reset_index()
            else:
                stats_df = df.groupby(["model_name", "alignment"])[feature].agg(
                    Center='median',
                    Variation=lambda x: stats.median_abs_deviation(x, scale='normal') if len(x) > 0 else 0
                ).reset_index()
            
            # Store for Table
            for _, row in stats_df.iterrows():
                table_data_collector.append({
                    "Model": row["model_name"],
                    "Role": row["alignment"],
                    "Feature": feature,
                    "Value": row["Center"],
                    "Variation": row["Variation"]
                })

            if show_point:
                plt.figure(figsize=(14, 8))
                sns.set_theme(style="whitegrid")
                offset_map = {"Crew": -0.15, "Imposter": 0.15, "Impostor": 0.15, "Byzantine": 0.15}
                colors = {"Crew": "#3498db", "Imposter": "#e74c3c", "Impostor": "#e74c3c", "Byzantine": "#e74c3c"}
                model_map = {name: i for i, name in enumerate(unique_models)}
                has_data = False
                
                for role in ["Crew", "Imposter"]:
                    role_data = stats_df[stats_df["alignment"] == role]
                    if role_data.empty: continue
                    has_data = True
                    xs = [model_map[m] + offset_map.get(role, 0) for m in role_data["model_name"]]
                    centers = role_data["Center"].values
                    vars_ = role_data["Variation"].values
                    lower_errors = [min(v, c) for c, v in zip(centers, vars_)]
                    plt.errorbar(x=xs, y=centers, yerr=[lower_errors, vars_], fmt='none', 
                                 ecolor=colors.get(role, 'gray'), elinewidth=2, capsize=5, alpha=0.7)
                    plt.scatter(xs, centers, s=100, c=colors.get(role, 'gray'), label=role, edgecolor='white', zorder=5)

                plt.title(f"{feature} ({metric_label} ± {variation_label})", fontsize=16)
                plt.xticks(x_indices, unique_models, rotation=45, ha='right')
                plt.grid(axis='x', alpha=0.3)
                if feature in ["num_moves", "votes_received"]: plt.ylim(bottom=-0.5)
                if has_data: plt.legend(title="Role", loc='best')
                plt.tight_layout()
                tag = "MEAN" if use_mean else "MEDIAN"
                p = os.path.join(output_dir, f"feature_point_{feature}_{tag}.png")
                plt.savefig(p)
                plt.close()

    wr_df = df.groupby(["model_name", "alignment"])["Games Won"].mean().reset_index()
    for _, row in wr_df.iterrows():
        table_data_collector.append({
            "Model": row["model_name"], "Role": row["alignment"],
            "Feature": "Win Rate", "Value": row["Games Won"], "Variation": 0.0
        })
    
    
    if show_table and table_data_collector:
        print("Generating Summary Table...")
        full_data = pd.DataFrame(table_data_collector)
        
        # Create Formatted String "Center ± Variation"
        full_data["Formatted"] = full_data.apply(
            lambda x: f"{x['Value']:.2f}" if x['Feature'] in ['Voting Acc.', 'Win Rate'] else f"{x['Value']:.2f} ± {x['Variation']:.2f}", 
            axis=1
        )
        
        # Pivot: Rows=Model+Role, Cols=Feature (Formatted Strings)
        pivot_table = full_data.pivot(index=["Model", "Role"], columns="Feature", values="Formatted")
        
        # Numeric pivot for calculating max values (Raw Floats)
        pivot_numeric = full_data.pivot(index=["Model", "Role"], columns="Feature", values="Value")
        
        # Dynamic height setup
        row_count = len(pivot_table)
        col_count = len(pivot_table.columns) + 2 
        fig_height = max(4, row_count * 0.5 + 1)
        fig_width = max(12, col_count * 1.5)
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        # Sort by Role then Model
        pivot_table = pivot_table.sort_index(level=["Role", "Model"])
        
        # Prepare Data for Matplotlib Table
        table_vals = []
        col_labels = ["Model", "Role"] + list(pivot_table.columns)
        
        for (model, role), row in pivot_table.iterrows():
            r_data = [model, role] + row.tolist()
            table_vals.append(r_data)

        # Draw Table
        the_table = ax.table(
            cellText=table_vals,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1, 1.5)

        # 1. Calculate Max Values per Role/Feature using RAW numbers
        max_map = {}
        for feature in pivot_numeric.columns:
            for role in pivot_numeric.index.get_level_values("Role").unique():
                # Get all values for this specific Role and Feature
                subset = pivot_numeric.xs(role, level="Role")[feature]
                if not subset.empty:
                    max_map[(role, feature)] = subset.max()

        # 2. Apply Styling
        # Iterate over the table cells
        cells = the_table.get_celld()
        
        for (row_idx, col_idx), cell in cells.items():
            cell.set_edgecolor('black')
            
            # Header Row
            if row_idx == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e') 
            else:
                # Data Rows
                cell.set_facecolor('#f5f5f5' if row_idx % 2 == 0 else 'white')
                
                # Check for Highlighting (Skip Model/Role columns 0 and 1)
                if col_idx >= 2:
                    # Retrieve identifiers from the table data to look up the raw value
                    # Note: row_idx is 1-based in table, so we use row_idx-1 for data list
                    current_model = table_vals[row_idx-1][0]
                    current_role = table_vals[row_idx-1][1]
                    current_feature = col_labels[col_idx]

                    if current_role == 'Crew' and current_feature == 'eliminations':
                        continue

                    if current_role == 'Imposter' and current_feature == 'times_eliminated':
                        continue
                    
                    # Look up the RAW value and the MAX value
                    try:
                        raw_val = pivot_numeric.loc[(current_model, current_role), current_feature]
                        max_val = max_map.get((current_role, current_feature), -999)
                        
                        # Compare raw floats with a tiny epsilon for safety
                        if abs(raw_val - max_val) < 1e-9:
                            cell.set_text_props(weight='bold')
                            cell.set_facecolor('#d1e7dd') # Light Green
                    except KeyError:
                        pass # Handle cases where keys might be missing safely

        plt.title(f"Summary ({metric_label} ± {variation_label})", fontsize=16, pad=10)
        plt.tight_layout()
        
        tag = "MEAN" if use_mean else "MEDIAN"
        p = os.path.join(output_dir, f"table_summary_{tag}.png")
        plt.savefig(p, bbox_inches='tight', dpi=300)
        plt.close()


def aggregate_logs(logs_dir, toggles):

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # 1. Load Data
    all_data = []
    # for config in COMPOSITION:
    #     scen_name = config["name"]
    #     search_path = os.path.join(logs_dir, scen_name, f"*_{scen_name}_Run*", "stats.csv")
    #     files = glob.glob(search_path)
    #     files.sort()
    #     if len(files) > 100: files = files[:100] # Limit for speed if needed

    #     for filename in files:
    #         try:
    #             df = pd.read_csv(filename)
    #             df.columns = df.columns.str.strip()
    #             df["scenario"] = scen_name
    #             df["game_id"] = os.path.basename(os.path.dirname(filename))
                
    #             # Normalize Names immediately
    #             if 'model_name' in df.columns:
    #                 df['model_name'] = df['model_name'].apply(normalize_model_name)
                
    #             alignment_map = {'H': 'Crew', 'B': 'Imposter'}
    #             if 'alignment' in df.columns:
    #                  df['alignment'] = df['alignment'].replace(alignment_map)
                     
    #             all_data.append(df)
    #         except:
    #             continue

    for entry in os.scandir(logs_dir):
        if entry.is_dir():
            model_folder_name = entry.name  # e.g., "qwen3_80B"
            
            # 2. Search for all stats.csv files nested within this model's game folders
            # Path pattern: logs_dir / model_name / *run_folder* / stats.csv
            search_path = os.path.join(entry.path, "*_Run*", "stats.csv")
            files = glob.glob(search_path)
            print(f"Found {len(files)} files for model {model_folder_name}")
            files.sort()
            
            # if len(files) > 100: 
            #     files = files[:100]

            for filename in files:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip()
                
                # Use the folder names as metadata
                df["scenario"] = model_folder_name
                df["game_id"] = os.path.basename(os.path.dirname(filename))
                
                # Normalize Names
                if 'model_name' in df.columns:
                    df['model_name'] = df['model_name'].apply(normalize_model_name)
                
                alignment_map = {'H': 'Crew', 'B': 'Imposter'}
                if 'alignment' in df.columns:
                        df['alignment'] = df['alignment'].replace(alignment_map)
                        
                all_data.append(df)

    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.fillna(0)
        
    if 'won_game' in full_df.columns:
        full_df.rename(columns={'won_game': 'Games Won'}, inplace=True)

    # 2. Execute Toggled Visuals
    print("\n--- STARTING VISUALIZATION PIPELINE ---")
    
    if toggles.get("Win_Rates_Chart"):
        plot_win_rates(full_df, logs_dir)
        
    if toggles.get("Stats_Tables"):
        plot_stat_tables(full_df, logs_dir)
        
    if toggles.get("Imposter_Pairings"):
        plot_imposter_pairings(full_df, logs_dir)
        
    if toggles.get("Feature_Analysis"):
        perform_feature_analysis(full_df, logs_dir, use_mean=True, show_bar=True, show_point=False, show_table=True) 

    print("\n--- PIPELINE COMPLETE ---")

if __name__ == "__main__":
    print("STARTING")
    VISUALIZATION_CONFIG = {
        "Win_Rates_Chart": True,      # The double bar chart
        "Stats_Tables": True,         # The PNG tables (Crew/Imposter wins)
        "Imposter_Pairings": False,    # Heatmap and pairing list
        "Feature_Analysis": True      # Feature breakdown
    }
    
    aggregate_logs(logs_dir="mixed/", toggles=VISUALIZATION_CONFIG)