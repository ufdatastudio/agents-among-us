import pandas as pd
import os

base_path = os.path.join(os.path.dirname(__file__), "database")
df_agents = pd.read_csv(os.path.join(base_path, "dim_agent.csv"))
df_events = pd.read_csv(os.path.join(base_path, "fact_game_events.csv"))
df_game = pd.read_csv(os.path.join(base_path, "dim_game.csv"))
df_consensus = pd.read_csv(os.path.join(base_path, "dim_consensus.csv"))

# Join data to get model names and selected model type
merged = df_events.merge(df_agents[["game_id", "agent_id", "model_name", "role"]], on=["game_id", "agent_id"])
merged = merged.merge(df_game[["game_id", "selected_model"]], on="game_id")
merged = merged[merged["role"] == "honest"]
merged = merged[merged["vote_target"].isin(df_agents[df_agents["role"] == "byzantine"]["agent_id"].values)]

# Overall model-wise vote accuracy
model_stats = merged.groupby("model_name")["vote_correct"].agg(["count", "sum"]).reset_index()
model_stats.rename(columns={"count": "total_votes", "sum": "correct_votes"}, inplace=True)
model_stats["correct_votes"] = model_stats["correct_votes"].astype(int)
model_stats["accuracy"] = (model_stats["correct_votes"] / model_stats["total_votes"]).round(3)
total_row = pd.DataFrame([{
    "model_name": "TOTAL",
    "total_votes": model_stats["total_votes"].sum(),
    "correct_votes": model_stats["correct_votes"].sum(),
    "accuracy": round(model_stats["correct_votes"].sum() / model_stats["total_votes"].sum(), 3)
}])
model_stats = pd.concat([model_stats.sort_values("accuracy", ascending=False), total_row], ignore_index=True)

print("\n=== Overall Vote Accuracy by Model ===")
print(model_stats.to_string(index=False))

# Filter to only votes where a byzantine was actually voted for
byz_votes = merged

# Accuracy when all models were vs. one another
byz_all = byz_votes[byz_votes["selected_model"] == "All"]
byz_all_stats = byz_all.groupby("model_name")["vote_correct"].agg(["count", "sum"]).reset_index()
byz_all_stats.rename(columns={"count": "total_votes", "sum": "correct_votes"}, inplace=True)
byz_all_stats["correct_votes"] = byz_all_stats["correct_votes"].astype(int)
byz_all_stats["accuracy"] = (byz_all_stats["correct_votes"] / byz_all_stats["total_votes"]).round(3)
total_row = pd.DataFrame([{
    "model_name": "TOTAL",
    "total_votes": byz_all_stats["total_votes"].sum(),
    "correct_votes": byz_all_stats["correct_votes"].sum(),
    "accuracy": round(byz_all_stats["correct_votes"].sum() / byz_all_stats["total_votes"].sum(), 3)
}])
byz_all_stats = pd.concat([byz_all_stats.sort_values("accuracy", ascending=False), total_row], ignore_index=True)

print("\n=== Byzantine Vote Accuracy (Randomized Games - 'All') ===")
print(byz_all_stats.to_string(index=False))

# Accuracy by isolating models (single-model games)
byz_single = byz_votes[byz_votes["selected_model"] != "All"]
byz_single_stats = byz_single.groupby("model_name")["vote_correct"].agg(["count", "sum"]).reset_index()
byz_single_stats.rename(columns={"count": "total_votes", "sum": "correct_votes"}, inplace=True)
byz_single_stats["correct_votes"] = byz_single_stats["correct_votes"].astype(int)
byz_single_stats["accuracy"] = (byz_single_stats["correct_votes"] / byz_single_stats["total_votes"]).round(3)
total_row = pd.DataFrame([{
    "model_name": "TOTAL",
    "total_votes": byz_single_stats["total_votes"].sum(),
    "correct_votes": byz_single_stats["correct_votes"].sum(),
    "accuracy": round(byz_single_stats["correct_votes"].sum() / byz_single_stats["total_votes"].sum(), 3)
}])
byz_single_stats = pd.concat([byz_single_stats.sort_values("accuracy", ascending=False), total_row], ignore_index=True)

print("\n=== Byzantine Vote Accuracy (Single Model Games) ===")
print(byz_single_stats.to_string(index=False))

# Decisiveness chart with rounds as columns and percent correct per model
round_model_accuracy = merged.groupby(["model_name", "round_id"])["vote_correct"].mean().reset_index()
round_model_accuracy["percent_correct"] = (round_model_accuracy["vote_correct"] * 100).round(1)
pivot = round_model_accuracy.pivot(index="model_name", columns="round_id", values="percent_correct").fillna(0).round(1)

print("\n=== Decisiveness by Model and Round (Percent of Correct Votes) ===")
print(pivot.to_string())
