from collections import defaultdict
import os
import matplotlib.pyplot as plt
from results.classifier import GameLogLoader, GameAnalytics
import pandas as pd

import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np

def run_scaling_memory_stats(active_games, target_exp_type="Hybrid", target_context="C2"):
    """
    Performs Kruskal-Wallis and Dunn's post-hoc test on GAME-LEVEL F1 scores
    grouped by the number of agents with context memory.
    """
    # Dictionary to hold lists of game-level F1 scores keyed by num_agents
    f1_distributions = defaultdict(list)

    for game in active_games:
        folder_name = game['composition_id']
        
        # Filter for the specific experimental line
        is_hybrid = "Hybrid" in folder_name
        exp_type = "Hybrid" if is_hybrid else "Default"
        context = "C2" if "C2" in folder_name else "C1"
        
        if exp_type != target_exp_type or context != target_context:
            continue
            
        # Determine number of agents
        num_agents = 0
        if "2Agents" in folder_name: num_agents = 2
        elif "4Agents" in folder_name: num_agents = 4
        elif "6Agents" in folder_name: num_agents = 6
        elif "8Agents" in folder_name or "Full" in folder_name: num_agents = 8
        elif "Control" in folder_name: num_agents = 0
        else: continue

        # Aggregate voting outcomes for the ENTIRE game
        game_tp = 0
        game_fp = 0
        game_fn = 0
        
        for turn in game['turns']:
            if turn['role'] == 'H' and 'olmo' not in turn['model'].lower():
                target = turn.get('vote_target', 'None')
                
                if target in ['None', 'SKIP']:
                    game_fn += 1
                elif turn['vote_correct']:
                    game_tp += 1
                else:
                    game_fp += 1
                    
        # Calculate F1 for the entire game and append to the specific num_agents group
        denominator = (2 * game_tp) + game_fp + game_fn
        if denominator > 0:
            f1 = (2 * game_tp) / denominator
            f1_distributions[num_agents].append(f1 * 100)
        else:
            # If the crew never voted or skipped (0 denominator), F1 is 0
            f1_distributions[num_agents].append(0.0)

    # Prepare data for Kruskal-Wallis
    agent_counts = sorted(f1_distributions.keys())
    data_arrays = [f1_distributions[count] for count in agent_counts]

    print("\n" + "="*80)
    print(f"STATISTICAL SIGNIFICANCE: {target_exp_type} ({target_context}) Memory Scaling")
    print("="*80)
    
    # Check if we have enough groups to compare
    if len(data_arrays) < 2:
        print("Not enough data groups found to run statistical tests.")
        return

    for count in agent_counts:
        scores = f1_distributions[count]
        print(f"Group {count} Agents: n={len(scores):<4} | Median F1={np.median(scores):.1f}%")

    print("-" * 80)
    
    # 1. Kruskal-Wallis H Test
    h_stat, p_kw = stats.kruskal(*data_arrays)
    print(f"Kruskal-Wallis H-Statistic: {h_stat:.4f}")
    print(f"Kruskal-Wallis P-Value:     {p_kw:.4e}")
    
    if p_kw < 0.05:
        print("\nSignificant difference found across groups (p < 0.05). Running Dunn's Post-Hoc...")
        print("-" * 80)
        
        # 2. Dunn's Test (with Bonferroni correction)
        # Convert dictionary to flat lists for scikit-posthocs
        flat_data = []
        flat_groups = []
        for count in agent_counts:
            flat_data.extend(f1_distributions[count])
            flat_groups.extend([count] * len(f1_distributions[count]))
            
        dunn_p_values = sp.posthoc_dunn(flat_data, p_adjust='bonferroni', groups=flat_groups)
        
        print("Dunn's Test P-Values (Bonferroni Corrected Matrix):")
        print(dunn_p_values.round(4))
        
        # Highlight significant pairwise differences
        print("\nSignificant Pairwise Differences (p < 0.05):")
        found_sig = False
        for i in range(len(agent_counts)):
            for j in range(i + 1, len(agent_counts)):
                g1 = agent_counts[i]
                g2 = agent_counts[j]
                p_val = dunn_p_values.loc[g1, g2]
                if p_val < 0.05:
                    print(f" - {g1} Agents vs {g2} Agents (p = {p_val:.4e})")
                    found_sig = True
        if not found_sig:
            print(" - None")
    else:
        print("\nNo significant difference found across groups. Post-hoc test aborted.")
    print("="*80)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from collections import defaultdict
    import os
    
    BASE_DIRECTORY = "results/experiments"
    CSV_PATH = "f1_context_comparison_table.csv"
    
    READ_FROM_CSV = False  # Toggle: True to load from CSV, False to parse game logs
    
    styles = {
        ("Hybrid", "C2"): {"color": "#1f77b4", "marker": "o", "label": "Hybrid (C=2)"},
        ("Hybrid", "C1"): {"color": "#6baed6", "marker": "s", "label": "Hybrid (C=1)", "linestyle": "--"},
        ("Default", "C2"): {"color": "#d62728", "marker": "o", "label": "Default (C=2)"},
        ("Default", "C1"): {"color": "#fb6a4a", "marker": "s", "label": "Default (C=1)", "linestyle": "--"},
    }

    plt.figure(figsize=(10, 6))

    if READ_FROM_CSV:
        print(f"Loading data directly from {CSV_PATH}...")
        df_pivot = pd.read_csv(CSV_PATH, header=[0, 1], index_col=0)
        
        for col in df_pivot.columns:
            exp_type, context = col
            y_vals = df_pivot[col].dropna()
            x_vals = y_vals.index
            style = styles.get((exp_type, context), {"marker": "x"})
            plt.plot(x_vals, y_vals, linewidth=2.5, markersize=8, **style)
            
    else:
        graph_data = defaultdict(dict)
        all_active_games = []
        
        if os.path.exists(BASE_DIRECTORY):
            for folder_name in os.listdir(BASE_DIRECTORY):
                if folder_name == "AmongUs10k" or "MixedWeight" in folder_name: continue
                
                folder_path = os.path.join(BASE_DIRECTORY, folder_name)
                if not os.path.isdir(folder_path): continue
                    
                is_hybrid = "Hybrid" in folder_name
                exp_type = "Hybrid" if is_hybrid else "Default"
                context = "C2" if "C2" in folder_name else "C1"
                
                num_agents = 0
                if "2Agents" in folder_name: num_agents = 2
                elif "4Agents" in folder_name: num_agents = 4
                elif "6Agents" in folder_name: num_agents = 6
                elif "8Agents" in folder_name or "Full" in folder_name: num_agents = 8
                elif "Control" in folder_name: num_agents = 0
                else: continue 
                
                print(f"Processing {folder_name}...")
                loader = GameLogLoader(folder_path, cache_dir=f"classifiers/data/{folder_name}")
                active_games, _ = loader.load_all(force_reload=False)
                
                if not active_games: continue
                
                # Add this folder's games to the master list
                all_active_games.extend(active_games) 
                    
                voting_results = GameAnalytics.calculate_voting_metrics(active_games)
                graph_data[(exp_type, context)][num_agents] = voting_results['AVERAGE']['F1'] * 100
                
            # --- RUN STATS OUTSIDE THE LOOP USING THE MASTER LIST ---
            if all_active_games:
                run_scaling_memory_stats(all_active_games, target_exp_type="Hybrid", target_context="C2")
                run_scaling_memory_stats(all_active_games, target_exp_type="Default", target_context="C2")
                run_scaling_memory_stats(all_active_games, target_exp_type="Hybrid", target_context="C1")
                run_scaling_memory_stats(all_active_games, target_exp_type="Default", target_context="C1")

            print(f"Exporting data to {CSV_PATH}...")
            table_rows = [{"Agent_Type": et, "Context_Window": cx, "Agents_With_Memory": na, "F1_Score": round(f1, 2)} 
                          for (et, cx), data in graph_data.items() for na, f1 in data.items()]
            
            if table_rows:
                df_pivot = pd.DataFrame(table_rows).pivot_table(
                    index="Agents_With_Memory", columns=["Agent_Type", "Context_Window"], values="F1_Score"
                )
                df_pivot.to_csv(CSV_PATH)
                
            for (exp_type, context), data_points in graph_data.items():
                if not data_points: continue
                sorted_items = sorted(data_points.items())
                x_vals = [item[0] for item in sorted_items]
                y_vals = [item[1] for item in sorted_items]
                style = styles.get((exp_type, context), {"marker": "x"})
                plt.plot(x_vals, y_vals, linewidth=2.5, markersize=8, **style)

    plt.title('Impact of Context Window & Agent Type on F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Agents with Context Memory', fontsize=12, fontweight='bold')
    plt.ylabel('Overall Crew F1 Score (%)', fontsize=12, fontweight='bold')
    plt.xticks([0, 2, 4, 6, 8])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("f1_context_comparison.png", dpi=300)
    print("Graph saved to f1_context_comparison.png")