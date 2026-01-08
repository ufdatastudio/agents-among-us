import pandas as pd
import glob
import os
import sys
from config.model_composition import COMPOSITION

def aggregate_logs(logs_dir="logs"):
    all_data = []
    
    # Iterate through compositions in config
    for config in COMPOSITION:
        scen_name = config["name"]
        pattern = os.path.join(logs_dir, f"Game_{scen_name}_Job*", "stats.csv")
        files = glob.glob(pattern)
        print(f"Scenario: {scen_name} -> Found {len(files)} files.")
        
        for filename in files:
            df = pd.read_csv(filename)
            df["scenario"] = scen_name
            df["game_id"] = os.path.basename(os.path.dirname(filename))
            all_data.append(df)


    if not all_data:
        print("No data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)

    # Metrics to process
    numeric_cols = [
        "correct_votes", "incorrect_votes", "skipped_votes",
        "emergency_meetings", "bodies_reported", "rounds_survived",
        "eliminations", "won_game", "times_eliminated", "ejections",
        "num_moves", "votes_received"
    ]
    
    # Filter to columns that actually exist in the CSVs
    valid_numeric_cols = [c for c in numeric_cols if c in full_df.columns]
    
    group_cols = ["scenario", "model_name", "agent_name", "alignment"]

    # PART 1: SUM TOTALS 
    # ==========================================
    summary_sum = full_df.groupby(group_cols)[valid_numeric_cols].sum().reset_index()

    # Calculate total games per scenario
    game_counts = full_df.groupby("scenario")["game_id"].nunique().reset_index()
    game_counts.rename(columns={"game_id": "total_games_played"}, inplace=True)

    # Merge total games count
    summary_sum = pd.merge(summary_sum, game_counts, on="scenario")
    summary_sum.rename(columns={"model_name": "LLM"}, inplace=True)
    
    output_sum = "aggregated_results_by_agent.csv"
    summary_sum.to_csv(output_sum, index=False)
    print(f"Sums saved to: {output_sum}")

    # PART 2: MEAN ± STD DEV 
    # ==========================================
    stats_cols = [c for c in valid_numeric_cols if c != "won_game"]

    # Calculate Mean and Std
    stats_agg = full_df.groupby(group_cols)[stats_cols].agg(['mean', 'std'])

    stats_agg.columns = [f"{col}_{stat}" for col, stat in stats_agg.columns]
    stats_agg = stats_agg.reset_index()

    # Create the final formatted DataFrame
    stats_formatted = stats_agg[group_cols].copy()

    for col in stats_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        
        stats_formatted[col] = stats_agg.apply(
            lambda x: f"{x[mean_col]:.2f} ± {x.get(std_col, 0.0):.2f}", axis=1
        )

    # Add game counts for context
    stats_formatted = pd.merge(stats_formatted, game_counts, on="scenario")
    stats_formatted.rename(columns={"model_name": "LLM"}, inplace=True)

    output_stats = "aggregated_stats_mean_std.csv"
    stats_formatted.to_csv(output_stats, index=False)
    print(f"Means ± Std saved to: {output_stats}")

if __name__ == "__main__":
    aggregate_logs()