from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from data.database import log_agent_metadata
import random

def create_agents(game_id):
    model_assignments = [
        ("gpt-4o", "🔴"),
        ("gpt-4", "🔵"),
        ("claude-3-opus-20240229", "🟢"),
        ("claude-3-haiku-20240307", "💗"),
        ("gpt-4-turbo", "🟠"),
        ("gpt-4o", "🟡"),
        ("gpt-4-turbo", "⚫"),
        ("gpt-4", "⚪"),
    ]

    all_indices = list(range(len(model_assignments)))
    eligible_indices = [i for i, (model, _) in enumerate(model_assignments) if "claude" not in model] # Claude models refuse to act deceptively
    byzantine_indices = random.sample(eligible_indices, 2)

    agents_state = {
        f"Agent_{i+1}": {
            "role": "byzantine" if i in byzantine_indices else "honest",
            "trust_scores": {},
            "messages": [],
            "perception": []
        }
        for i in range(len(model_assignments))
    }

    agents = []
    for i, (model, color) in enumerate(model_assignments):
        name = f"Agent_{i+1}"
        role = agents_state[name]["role"]
        agent_class = ByzantineAgent if role == "byzantine" else HonestAgent
        agent = agent_class(name, agents_state, model_name=model, color=color)
        agents.append(agent)
        log_agent_metadata(game_id, name, role, model, color)

    return agents, agents_state
