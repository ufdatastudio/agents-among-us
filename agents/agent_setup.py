# Creates and initializes agents with assigned models, roles, and colors.
from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from data.database import log_agent_metadata
import random

def create_agents(game_id, model_name='meta-llama/Meta-Llama-3-8B-Instruct', num_agents=10):
    """
    Initializes a list of agents for a game simulation.

    Args:
        game_id (str): The unique identifier for the game session.
        model_name (str): The Hugging Face model identifier to be used for all agents.
        num_agents (int): The total number of agents to create.

    Returns:
        tuple: A tuple containing the list of agent objects and the initial agent state dictionary.
    """
    # Define available agent colors.
    colors = ["🔴", "🔵", "🟢", "💗", "🟠", "🟡", "⚫", "⚪", "🟣", "🟤"]
    
    # Ensure there are at least two agents to assign roles correctly.
    if num_agents < 2:
        raise ValueError("Cannot create a game with fewer than 2 agents.")

    # Randomly choose 2 agents to be Byzantine.
    byzantine_indices = random.sample(range(num_agents), 2)

    # Initialize agent state with role and perception fields.
    agents_state = {
        f"Agent_{i+1}": {
            "role": "byzantine" if i in byzantine_indices else "honest", # The agent's true alignment for the game
            "survived": True, # Tracks if the agent is survived the game
            "messages": [], # A log of messages sent by this agent during meetings
            "perception": [], # A log of game events this agent has observed (e.g., "saw Red vent")
            "game_stats": {
                "votes_cast": [], # Dictionary of agents this agent voted for (e.g., ["target": "skip", "was_correct": False])]
                "votes_recieved": [], # [{"round": 1, "voters": ["Agent_2", "Agent_5"]}]
                "meeting_called": False, # Boolean for emergency meetings initiated
                "bodies_reported": 0, # Counter for bodies reported
                "eliminations_made": 0 # Counter for eliminations by a byzantine agent
            }
        }
        for i in range(num_agents)
    }

    agents = []
    # Instantiate agent objects with role-based classes and assign metadata.
    for i in range(num_agents):
        name = f"Agent_{i+1}"
        role = agents_state[name]["role"]
        agent_class = ByzantineAgent if role == "byzantine" else HonestAgent # <-- Go to this next
        color = colors[i % len(colors)]
        
        # Create the agent instance with the specified model
        agent = agent_class(name, agents_state, model_name=model_name, color=color)
        agents.append(agent)
        
        # Log the agent's metadata
        log_agent_metadata(game_id, name, role, model_name, color)

    return agents, agents_state

