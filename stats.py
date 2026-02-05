import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table 
import numpy as np
from config.model_composition import COMPOSITION 

plt.switch_backend('Agg')

def normalize_model_name(name):
    """
    1. Removes organization prefix (everything before /).
    2. Groups specific variants together (e.g. Apertus).
    3. Returns a shorthand version for plotting.
    """
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

def render_mpl_table(data, col_width=6.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Helper function to render a Pandas DataFrame as a matplotlib table.
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
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def save_model_stats_visuals(full_df, output_dir):
    """
    Generates:
    1. Double Bar Chart (Crew vs Imposter Win Rates per Model)
    2. Scatter Plot (Model Performance: Crew Win Rate vs Imposter Win Rate)
    3. Separate PNG Tables for Crew and Imposter stats
    """
    print("Generating Model Stats Visuals...")
    
    if 'model_name' not in full_df.columns:
        print("Error: 'model_name' column not found.")
        return

    df = full_df.copy()
    df['model_name'] = df['model_name'].apply(normalize_model_name)

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

    # --- PART A: DOUBLE BAR CHART ---
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    stats = stats.sort_values(by="Win_Rate_Num", ascending=False)

    chart = sns.barplot(
        data=stats,
        x="Model",
        y="Win_Rate_Num",
        hue="Role",
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
    chart_path = os.path.join(output_dir, "chart_model_win_rates_double.png")
    plt.savefig(chart_path)
    print(f"Saved Double Bar Chart to: {chart_path}")
    plt.close()

    # --- PART B: SCATTER PLOT (CREW vs IMPOSTER) ---
    pivot_df = stats.pivot(index="Model", columns="Role", values="Win_Rate_Num").reset_index()
        
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # Plot with Hue for distinct colors
    scatter = sns.scatterplot(
        data=pivot_df,
        x="Imposter",
        y="Crew",
        hue="Model",        
        s=200,              
        palette="tab10",    
        edgecolor="black",
        alpha=0.9
    )
    
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)

    plt.title("Crew vs. Imposter Win Rate", fontsize=18)
    plt.xlabel("Imposter Win Rate", fontsize=14)
    plt.ylabel("Crew Win Rate", fontsize=14)
    plt.xlim(.5, .85)
    plt.ylim(0.25, .35)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Model Name", fontsize=15)
    
    plt.tight_layout() 
    scatter_path = os.path.join(output_dir, "chart_scatter_win_rates.png")
    plt.savefig(scatter_path)
    print(f"Saved Scatter Plot to: {scatter_path}")
    plt.close()

    print("Generating Overall Win Rate Chart...")
    overall_stats = (
        df.groupby("model_name")["Games Won"]
        .mean()
        .reset_index()
        .rename(columns={"Games Won": "Overall_Win_Rate", "model_name": "Model"})
        .sort_values(by="Overall_Win_Rate", ascending=False)
    )

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    chart_overall = sns.barplot(
        data=overall_stats,
        x="Model",
        y="Overall_Win_Rate",
        color="#00000000")

    plt.title("Overall Win Rate", fontsize=18)
    plt.ylabel("Win Rate (0.0 - 1.0)", fontsize=14)
    plt.xlabel("Model Name", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)

    for container in chart_overall.containers:
        chart_overall.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    overall_path = os.path.join(output_dir, "chart_model_win_rates_overall.png")
    plt.savefig(overall_path)
    print(f"Saved Overall Win Rate Chart to: {overall_path}")
    plt.close()

    # --- PART D: TABLES ---
    stats["Win Rate"] = stats["Win_Rate_Num"].apply(lambda x: f"{x:.2%}")
    table_df = stats[["Model", "Role", "Wins", "Losses", "Win Rate"]]

    # 1. CREW TABLE
    crew_data = table_df[table_df["Role"].isin(["Crew"])].sort_values(by="Wins", ascending=False)
    if not crew_data.empty:
        ax = render_mpl_table(crew_data, header_columns=0, col_width=6.0)
        p = os.path.join(output_dir, "table_model_stats_crew.png")
        plt.savefig(p, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved Crew Stats Table to: {p}")
        plt.close()

    # --- PART C: PREPARE TABLES ---
    stats["Win Rate"] = stats["Win_Rate_Num"].apply(lambda x: f"{x:.2%}")
    table_df = stats[["Model", "Role", "Wins", "Losses", "Win Rate"]]

    # 1. CREW TABLE
    crew_data = table_df[table_df["Role"].isin(["Crew"])].sort_values(by="Wins", ascending=False)
    if not crew_data.empty:
        ax = render_mpl_table(crew_data, header_columns=0, col_width=6.0)
        p = os.path.join(output_dir, "table_model_stats_crew.png")
        plt.savefig(p, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved Crew Stats Table to: {p}")
        plt.close()

    # 2. IMPOSTER TABLE
    imp_data = table_df[table_df["Role"].isin(["Imposter"])].sort_values(by="Wins", ascending=False)
    if not imp_data.empty:
        ax = render_mpl_table(imp_data, header_columns=0, col_width=6.0)
        p = os.path.join(output_dir, "table_model_stats_imposter.png")
        plt.savefig(p, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved Imposter Stats Table to: {p}")
        plt.close()

    csv_path = os.path.join(output_dir, "table_model_win_rates.csv")
    table_df.to_csv(csv_path, index=False)


def save_imposter_pairing_table(full_df, output_dir):
    """
    Generates:
    1. PNG Table for Imposter Pairings.
    2. Performance Heatmap (Model vs Model Win Rate).
    """
    print("Generating Imposter Pairing Analysis...")
    
    imp_df = full_df[full_df["alignment"].isin(["Impostor", "Imposter", "Byzantine"])].copy()
    
    if imp_df.empty:
        print("No Imposter data found.")
        return

    imp_df['model_name'] = imp_df['model_name'].apply(normalize_model_name)

    game_pairings = []
    
    for game_id, group in imp_df.groupby("game_id"):
        models = sorted(group["model_name"].tolist())
        if len(models) == 2:
            pairing_key = "|".join(models) 
            won = group["Games Won"].iloc[0]
            
            game_pairings.append({
                "model_a": models[0],
                "model_b": models[1],
                "pairing": f"{models[0]} & {models[1]}",
                "won": won
            })
    
    if not game_pairings:
        print("No valid pairings found.")
        return

    pairings_df = pd.DataFrame(game_pairings)

    pairing_stats = (
        pairings_df.groupby("pairing")["won"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    pairing_stats.rename(columns={
        "pairing": "Impostor Duo",
        "count": "Total Games",
        "sum": "Wins",
        "mean": "Win Rate"
    }, inplace=True)

    table_stats = pairing_stats.copy()
    table_stats["Losses"] = table_stats["Total Games"] - table_stats["Wins"]
    table_stats = table_stats[["Impostor Duo", "Win Rate"]]
    table_stats["Win Rate"] = table_stats["Win Rate"].apply(lambda x: f"{x:.2%}")
    table_stats = table_stats.sort_values(by="Win Rate", ascending=False)

    ax = render_mpl_table(table_stats, header_columns=0, col_width=11.0)
    save_path = os.path.join(output_dir, "table_imposter_pairings.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved Imposter Pairing Table to: {save_path}")
    plt.close()

    csv_path = os.path.join(output_dir, "table_imposter_pairings.csv")
    table_stats.to_csv(csv_path, index=False)

    print("Generating Heatmap...")

    individual_stats = imp_df.groupby("model_name")["Games Won"].mean().sort_values(ascending=False)
    sorted_models = individual_stats.index.tolist()

    models = sorted(list(set(pairings_df["model_a"].unique()) | set(pairings_df["model_b"].unique())))
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)

    agg_pairs = pairings_df.groupby(["model_a", "model_b"])["won"].mean()
    
    for (m1, m2), win_rate in agg_pairs.items():
        matrix.at[m1, m2] = win_rate
        matrix.at[m2, m1] = win_rate # Symmetric

    valid_sorter = [m for m in sorted_models if m in matrix.index]
    matrix = matrix.loc[valid_sorter, valid_sorter]

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white") 
    
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    matrix_cropped = matrix.iloc[1:, :-1]
    mask_cropped = mask[1:, :-1] 
    
    ax = sns.heatmap(
        matrix_cropped, 
        mask=mask_cropped,
        annot=True, 
        fmt=".0%",        
        cmap="RdYlGn",    
        cbar_kws={'label': 'Win Rate'},
        linewidths=.5,
        linecolor='gray',
        square=True
    )

    ax.set_facecolor('white') 
    ax.grid(False)
    plt.title("Impostor Synergy Heatmap", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tick_params(left=False, bottom=False)
    
    heatmap_path = os.path.join(output_dir, "chart_imposter_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    print(f"Saved Imposter Heatmap to: {heatmap_path}")
    plt.close()


def aggregate_logs(logs_dir="results/experiment_1"):
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    all_scenarios_data = []

    for config in COMPOSITION:
        scen_name = config["name"]
        search_path = os.path.join(logs_dir, scen_name, f"*_{scen_name}_Run*", "stats.csv")
        files = glob.glob(search_path)
        print("Found files for scenario", scen_name, ":", len(files))
        files.sort()
        if len(files) > 100:
            files = files[:100]

        for filename in files:
            try:
                df = pd.read_csv(filename)
                df.columns = df.columns.str.strip()
                df["scenario"] = scen_name
                df["game_id"] = os.path.basename(os.path.dirname(filename))
                
                alignment_map = {'H': 'Crew', 'B': 'Imposter'}
                if 'alignment' in df.columns:
                     df['alignment'] = df['alignment'].replace(alignment_map)

                all_scenarios_data.append(df)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
    if not all_scenarios_data:
        print("No data found.")
        return

    full_df = pd.concat(all_scenarios_data, ignore_index=True)

    numeric_cols = [
        "correct_votes", "incorrect_votes", "skipped_votes",
        "emergency_meetings", "bodies_reported", "rounds_survived",
        "eliminations", "times_eliminated", "ejections",
        "num_moves", "votes_received", "won_game"
    ]
    
    for col in numeric_cols:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce').fillna(0)

    if 'won_game' in full_df.columns:
        full_df.rename(columns={'won_game': 'Games Won'}, inplace=True)
        
        # print("\n--- WIN RATE SUMMARY (SCENARIO) ---")
        # win_rates = (
        #     full_df
        #     .groupby(["scenario", "alignment"])["Games Won"]
        #     .mean()
        #     .reset_index()
        #     .sort_values(["scenario", "alignment"])
        # )
        # for _, row in win_rates.iterrows():
        #     print(f"Scenario: {row['scenario']:<25} Alignment: {row['alignment']:<10} Win Rate: {row['Games Won']:.3f}")
    

        print("\n--- GENERATING VISUALS ---")
        save_model_stats_visuals(full_df, logs_dir)
        save_imposter_pairing_table(full_df, logs_dir)

if __name__ == "__main__":
    aggregate_logs()