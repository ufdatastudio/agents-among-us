import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set path to your data folder
base_path = os.path.join(os.path.dirname(__file__), "database")

# Load data
df_agents = pd.read_csv(os.path.join(base_path, "dim_agent.csv"))
df_events = pd.read_csv(os.path.join(base_path, "fact_game_events.csv"))
df_game = pd.read_csv(os.path.join(base_path, "dim_game.csv"))
df_consensus_raw = pd.read_csv(os.path.join(base_path, "dim_consensus.csv"))
df_trust = pd.read_csv(os.path.join(base_path, "dim_trust.csv"))

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

# Decisiveness chart with rounds as columns and percent correct per model
round_model_accuracy = merged.groupby(["model_name", "round_id"])["vote_correct"].mean().reset_index()
round_model_accuracy["percent_correct"] = (round_model_accuracy["vote_correct"] * 100).round(1)
pivot = round_model_accuracy.pivot(index="model_name", columns="round_id", values="percent_correct").fillna(0).round(1)

# Chart data prep: TRUST
df_trust = df_trust.merge(df_agents[["game_id", "agent_id", "model_name"]], on=["game_id", "agent_id"])
df_trust["direction"] = df_trust["trust_delta"].apply(lambda x: "Increase" if x > 0 else "Decrease")

# Chart data prep: CONSENSUS (needs vote_correct + model_name)
df_consensus = df_consensus_raw.merge(
    df_events[["game_id", "round_id", "vote_correct"]].drop_duplicates(),
    on=["game_id", "round_id"],
    how="left"
)
df_consensus = df_consensus.merge(
    df_game[["game_id"]].drop_duplicates(),
    on="game_id",
    how="left"
)
df_consensus = df_consensus.merge(
    df_agents[["game_id", "model_name"]].drop_duplicates(),
    on="game_id",
    how="left"
)
df_consensus["Correctness"] = df_consensus["vote_correct"].map({1: "Correct", 0: "Incorrect"})

fig = plt.figure(figsize=(18, 16))
gs = fig.add_gridspec(3, 2)

# Text Panel 1: Overall model accuracy
ax_text1 = fig.add_subplot(gs[0, 0])
ax_text1.axis("off")
text1 = f"""=== Overall Vote Accuracy by Model ===
{model_stats.to_string(index=False)}"""
ax_text1.text(0, 1, text1, fontsize=10, fontfamily="monospace", va="top")

# Text Panel 2: Accuracy - Randomized Games
ax_text2 = fig.add_subplot(gs[1, 0])
ax_text2.axis("off")
text2 = f"""=== Byzantine Vote Accuracy (Randomized Games - 'All') ===
{byz_all_stats.to_string(index=False)}"""
ax_text2.text(0, 1, text2, fontsize=10, fontfamily="monospace", va="top")

# Text Panel 3: Accuracy - Single Model Games
ax_text3 = fig.add_subplot(gs[2, 0])
ax_text3.axis("off")
text3 = f"""=== Byzantine Vote Accuracy (Single Model Games) ===
{byz_single_stats.to_string(index=False)}

=== Decisiveness by Model and Round ===
{pivot.to_string()}"""
ax_text3.text(0, 1, text3, fontsize=10, fontfamily="monospace", va="top")

# Chart 1: Trust Changes by Model
ax1 = fig.add_subplot(gs[0, 1])
sns.countplot(
    data=df_trust,
    x="model_name",
    hue="direction",
    order=df_trust["model_name"].value_counts().index,
    ax=ax1
)
ax1.set_title("Trust Changes by Model")
ax1.set_xlabel("Model Name")
ax1.set_ylabel("Count")
ax1.tick_params(axis='x', rotation=45)

# Chart 2: Consensus Agreement by Model and Vote Correctness
ax2 = fig.add_subplot(gs[1, 1])
sns.boxplot(
    data=df_consensus,
    x="model_name",
    y="agreement_level",
    hue="Correctness",
    order=df_consensus["model_name"].value_counts().index,
    ax=ax2
)
ax2.set_title("Consensus Agreement Level by Model and Vote Correctness")
ax2.set_xlabel("Model Name")
ax2.set_ylabel("Agreement Level")
ax2.tick_params(axis='x', rotation=45)

# Chart 3: Vote Volume vs Accuracy
df_vote = pd.read_csv(os.path.join(base_path, "dim_vote.csv"))
vote_counts = df_vote.groupby("game_id").size().reset_index(name="num_votes")
vote_accuracy = merged.groupby("game_id")["vote_correct"].mean().reset_index(name="vote_accuracy")
votes_vs_accuracy = vote_counts.merge(vote_accuracy, on="game_id")

ax3 = fig.add_subplot(gs[2, 1])
sns.scatterplot(data=votes_vs_accuracy, x="num_votes", y="vote_accuracy", ax=ax3)
sns.regplot(data=votes_vs_accuracy, x="num_votes", y="vote_accuracy", scatter=False, ax=ax3, color="red")
ax3.set_title("Vote Volume vs Vote Accuracy")
ax3.set_xlabel("Number of Votes")
ax3.set_ylabel("Vote Accuracy")

fig.suptitle("Byzantine Brains Simulation Analysis", fontsize=20, fontweight="bold", ha="center", va="top", y=0.995)
plt.tight_layout()
fig.canvas.manager.set_window_title("Analysis")
plt.show()
