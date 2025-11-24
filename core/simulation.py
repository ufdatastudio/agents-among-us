from agents.agent_setup import create_agents
from game.game_loopv3 import rooms
from config.settings import NUM_ROUNDS
from data.database import log_round_metadata, write_csv, log_game_model_selection
import random
import time

NUM_ROUNDS = 10  # Increased to 10 rounds for a clearer end condition.
_current_game = {}

# Initializes the game state, agents, and room assignments for a new simulation.
def setup_game(game_id, selected_model="All"):
    agents, agents_state = create_agents(game_id, selected_model)
    all_rooms = list(rooms.keys())
    state = {
        agent.name: {
            "room": random.choice(all_rooms),
            "eliminated": False, # changed from Killed 
            "room_body": None,
            "task_room": random.choice(all_rooms) if agent.__class__.__name__ == "HonestAgent" else None,
            "task_done": False,
            "doing_task": False,
            "ejected": False,         
        } for agent in agents
    }

    state["body_is_reported"] = False
    state["_reported_bodies"] = set()
    state["emergency_meeting_called"] = False
    
    

    alive = sum(1 for v in state.values() if isinstance(v, dict) and not v.get("eliminated"))
    dead = sum(1 for v in state.values() if isinstance(v, dict) and v.get("eliminated"))

    log_round_metadata(game_id, 0, alive, dead)
    log_game_model_selection(game_id, selected_model)
    state["start_time"] = time.time() 
    _current_game.update({"state": state, "agents": agents, "agents_state": agents_state, "round": 0})
   
    return agents, agents_state, state

# Returns the current list of agents and their game state.
def get_current_state():
    return _current_game.get("agents", []), _current_game.get("state", {})
