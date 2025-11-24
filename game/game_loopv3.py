# game/game_loop.py
import random
import json
from datetime import datetime
import os
import platform
import time
from config.settings import NUM_ROUNDS
from data.database import (
    log_game_event, log_reflection, log_vote,
    log_consensus, log_round_metadata, log_game_summary
)

# --- UTILITY FUNCTIONS ---

def clear_terminal():
    os.system("cls" if platform.system() == "Windows" else "clear")

# --- GAME DATA ---

rooms = {
    "Cafeteria": ["Weapons", "Navigation", "Storage", "Admin", "MedBay", "UpperEngine"],
    "Weapons": ["Cafeteria", "O2"],
    "Navigation": ["Shields", "O2"],
    "O2": ["Weapons", "Navigation", "Cafeteria"],
    "Shields": ["Navigation", "Communications", "Storage"],
    "Communications": ["Shields"],
    "Admin": ["Cafeteria", "Storage"],
    "Storage": ["Admin", "Shields", "Electrical", "LowerEngine", "Cafeteria"],
    "Electrical": ["Storage", "LowerEngine"],
    "LowerEngine": ["Storage", "Electrical", "Security", "Reactor"],
    "Security": ["LowerEngine", "UpperEngine", "Reactor"],
    "Reactor": ["Security", "UpperEngine", "LowerEngine"],
    "UpperEngine": ["Reactor", "Security", "MedBay", "Cafeteria"],
    "MedBay": ["UpperEngine", "Cafeteria"]
}
# --- LOGGING SETUP ---

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"simulation_log_{timestamp}.json")
log_data = { "start_time": timestamp, "events": [] }

# --- DISPLAY FUNCTIONS ---

def show_ship_map(state, agents):
    mapping = {room: [] for room in rooms}
    bodies = {room: [] for room in rooms}
    for agent_name, info in state.items():
        if not isinstance(info, dict): continue
        agent_obj = next((a for a in agents if a.name == agent_name), None)
        if not agent_obj: continue

        if not info.get("eliminated", False):
            if "room" in info:
                mapping[info["room"]].append(f"{agent_obj.color}{agent_name}\033[0m")
        elif info.get("room_body"):
            bodies[info["room_body"]].append(f"💀{agent_obj.color}{agent_name}\033[0m")

    def fmt(room):
        entries = mapping.get(room, []) + bodies.get(room, [])
        return f"[{room}]" if not entries else f"[{room}: {' '.join(entries)}]"

    print("\n---  Ship Map ---\n")
    print(f"{fmt('UpperEngine')} {' ' * 15} {fmt('Cafeteria')} {' ' * 15} {fmt('Weapons')}")
    print(f"{fmt('Reactor')} {' ' * 5} {fmt('Security')} {' ' * 5} {fmt('MedBay')} {' ' * 5} {fmt('O2')} {' ' * 5} {fmt('Navigation')}")
    print(f"{fmt('LowerEngine')} {' ' * 2} {fmt('Electrical')} {' ' * 5} {fmt('Storage')} {' ' * 5} {fmt('Admin')} {' ' * 5} {fmt('Shields')}")
    print(f"{' ' * 65} {fmt('Communications')}")
    print("\n" + "-"*40)

def show_agent_status(agents, state, agents_state):
    print("\n--- Agent Status ---")
    print(f"{'Name':<25} | {'Role':<10} | {'Status':<12}")
    print("-" * 50)
    sorted_agents = sorted(agents, key=lambda a: a.name)
    for agent in sorted_agents:
        status = "Alive"
        info = state.get(agent.name, {})
        if info.get("ejected"): status = "Ejected"
        elif info.get("eliminated"): status = "Eliminated"
        colored_name = f"{agent.color}{agent.name}\033[0m"
        print(f"{colored_name:<35} | {agents_state[agent.name]['role']:<10} | {status:<12}")
    print("-" * 50)

# --- CORE GAME LOGIC ---

def _handle_agent_action(agent, agents, state, agents_state, step, previous_room=None):
    """
    Handles a single agent's action for one turn.
    Returns True if an elimination occurred, False otherwise.
    """
    current = state[agent.name]["room"]
    adj = rooms[current]
    
    dest, raw_response = agent.choose_room(current, adj, state, agents_state, step, previous_room)
    print(f"\n-- {agent.color}{agent.name}| {agent.agents_state[agent.name]['role']} ---")
    print(f"Raw response: {raw_response}")
    print(f"Action chosen: {dest}")
    
    elimination_made = False

    if dest.startswith("Eliminate "):
        target_name = dest.split(" ")[1]
        if target_name in state and state[target_name]["room"] == current and not state[target_name]["eliminated"]:
            state[target_name]["eliminated"] = True
            agents_state[agent.name]["game_stats"]["eliminations_made"] += 1
            state[target_name]["room_body"] = current
            print(f"🔪 {agent.color}{agent.name}\033[0m eliminated {target_name} in {current}")
            elimination_made = True
            
            # After an elimination, the agent automatically flees to a random adjacent room # or make llm decide?
            # flee_destination = random.choice(adj)
            # state[agent.name]["room"] = flee_destination
            # print(f"👣 {agent.color}{agent.name}\033[0m eliminated a target and fled from {current} to {flee_destination}")
        else:
            print(f"ERROR: {agent.color}{agent.name}\033[0m failed to eliminate {target_name}. Moving randomly.")
            state[agent.name]["room"] = random.choice(adj)

    elif dest == "Press Emergency Button" and current == "Cafeteria" and not agents_state[agent.name]["game_stats"]["meeting_called"]:
        print(f"🚨 {agent.color}{agent.name}\033[0m pressed the emergency button!")
        agents_state[agent.name]["game_stats"]["meeting_called"] = True
        state["emergency_meeting_called"] = True
    elif dest in rooms:
        state[agent.name]["room"] = dest
        if dest != current:
            print(f"👣 {agent.color}{agent.name}\033[0m moved from {current} to {dest}")
        else: # Agent chose to stay
             print(f"🧘{agent.color}{agent.name}\033[0m stayed in {current}")
    else: # Invalid move
        state[agent.name]["room"] = current
        print(f"❓ {agent.color}{agent.name}\033[0m made an invalid choice and stayed in {current}")

    # --- Post-Action Perception and Tasks ---
    new_room = state[agent.name]["room"]
    seen_agents = [a.name for a in agents if state[a.name].get("room") == new_room and a.name != agent.name and not state[a.name].get("eliminated", False)]
    seen_bodies = [a.name for a in agents if state[a.name].get("eliminated") and state[a.name].get("room_body") == new_room and a.name not in state["_reported_bodies"]]
    
    agents_state[agent.name]["perception"].append({"room": new_room, "agents_seen": seen_agents, "bodies_seen": seen_bodies})

    if seen_bodies and not state[agent.name]["eliminated"]:
        if agent.decide_to_report(seen_bodies, new_room, state, agents_state, round_num=step): 
            bodies_str = ", ".join(seen_bodies)
            print(f"📢 {agent.color}{agent.name}\033[0m reports the body of {bodies_str}!")
            agents_state[agent.name]["game_stats"]["bodies_reported"] += 1
            state["_reported_bodies"].update(seen_bodies)
            state["body_is_reported"] = True
    
    if agent.__class__.__name__ == "HonestAgent" and not state[agent.name]["task_done"]:
        if state[agent.name]["room"] == state[agent.name]["task_room"]:
            if state[agent.name]["doing_task"]:
                state[agent.name]["task_done"] = True
                print(f"✅ {agent.color}{agent.name}\033[0m completed their task.")
            else:
                state[agent.name]["doing_task"] = True
                print(f"🛠️ {agent.color}{agent.name}\033[0m started a task.")
        else:
            state[agent.name]["doing_task"] = False
            
    return elimination_made

def movement_phase(state, agents, agents_state, step):
    if "_reported_bodies" not in state:
        state["_reported_bodies"] = set()

    shuffled_agents = list(agents)
    random.shuffle(shuffled_agents)
    
    for agent in shuffled_agents:
        if state[agent.name]["eliminated"]:
            continue
        
        if agents_state[agent.name]['role'] == 'byzantine':
            max_moves = 5
            moves_taken = 0
            elimination_made = False
            last_room = None
            while moves_taken < max_moves and not elimination_made:
                current_room_before_move = state[agent.name]["room"]
                elimination_made = _handle_agent_action(agent, agents, state, agents_state, step, previous_room=last_room)
                last_room = current_room_before_move
                moves_taken += 1
                # If a meeting is triggered by the action, stop this agent's turn
                if state.get("body_is_reported") or state.get("emergency_meeting_called"):
                    break
        else: # Honest Agent
            max_moves = 2
            moves_taken = 0
            last_room = None
            while moves_taken < max_moves:
                current_room_before_move = state[agent.name]["room"]
                
                # Execute action
                _handle_agent_action(agent, agents, state, agents_state, step, previous_room=last_room)
                
                # Update memory to prevent immediate backtracking in the next sub-move
                last_room = current_room_before_move
                moves_taken += 1

                # Stop moving if a meeting starts (Body Reported or Button Pressed)
                if state.get("body_is_reported") or state.get("emergency_meeting_called"):
                    break
                
def run_game_round(game_id, step, state, agents, agents_state):
    max_movement_phases = 3
    meeting_triggered = False

    for movement_stage in range(1, max_movement_phases + 1):
        print(f"\n--- Movement Phase {step}.{movement_stage} ---")
        time.sleep(1)
        
        movement_phase(state, agents, agents_state, f"{step}.{movement_stage}")
        show_ship_map(state, agents)

        alive_byzantines = [a for a in agents if agents_state[a.name]["role"] == "byzantine" and not state[a.name]["eliminated"]]
        alive_honest = [a for a in agents if agents_state[a.name]["role"] == "honest" and not state[a.name]["eliminated"]]
        if not alive_byzantines or len(alive_byzantines) >= len(alive_honest):
            show_agent_status(agents, state, agents_state)
            return

        if state.get("body_is_reported") or state.get("emergency_meeting_called"):
            meeting_triggered = True
            break
    
    if meeting_triggered:
        state["body_is_reported"] = False
        state["emergency_meeting_called"] = False
        
        print(f"\n--- DISCUSSION (Round {step}) ---")
        messages = {}
        for agent in agents:
            if not state[agent.name]["eliminated"]:
                history = agents_state[agent.name].get("perception", [])
                message = agent.simulate_message(history, state, agents_state, step)
                if message:
                    print(f"🗣️ {agent.color}{agent.name}\033[0m says: {message}")
                    messages[agent.name] = message
                    log_reflection(game_id, step, agent.name, message)

        for agent in agents:
            if not state[agent.name]["eliminated"]:
                response = agent.respond_to_message(messages, agents_state[agent.name].get("perception", []), state, agents_state)
                if response:
                    print(f"💬 {agent.color}{agent.name}\033[0m responds: {response}")
                    log_reflection(game_id, step, agent.name, response)
        
        print("\n--- VOTING ---")
        votes = {}
        for agent in agents:
            if not state[agent.name]["eliminated"]:
                voter, target = agent.vote_for_ejection(state, agents_state, step)
                votes[voter] = target
                log_vote(game_id, step, voter, target)
                print(f"🗳️ {agent.color}{voter}\033[0m voted for {target}")

                agents_state[voter]["game_stats"]["votes_cast"].append({
                    "round": step,
                    "target": target})

        vote_counts = {target: 0 for target in votes.values()}
        for target in votes.values(): vote_counts[target] += 1
        
        if not vote_counts:
             print("\n--- VOTE RESULT: No votes were cast. ---")
        else:
            ejected = max(vote_counts, key=vote_counts.get)
            agreement_level = vote_counts[ejected] / len(votes)

            if ejected in agents_state and ejected != "No Ejection":
                correct = (agents_state[ejected]["role"] == "byzantine")
                print(f"\n--- VOTE RESULT: {ejected} was ejected. ---")
                print(f"The vote was {'CORRECT' if correct else 'INCORRECT'}. Their role was: {agents_state[ejected]['role']}.")
                state[ejected]["eliminated"] = True
                state[ejected]["ejected"] = True
            else:
                print("\n--- VOTE RESULT: No one was ejected. ---")
            log_consensus(game_id, step, f"Eject {ejected}", agreement_level)
    
    alive_agents = [a for a in agents if not state[a.name]["eliminated"]]
    log_round_metadata(game_id, step, len(alive_agents), len(agents) - len(alive_agents))
    show_agent_status(agents, state, agents_state)

def finalize_log(game_id, duration, victory_condition):
    log_game_summary(game_id, victory_condition, duration)
    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nSimulation results saved to database for game_id: {game_id}")
    print(f"Full log saved to {log_file_path}")
    if duration is not None:
        print(f"Total game time: {duration:.2f} seconds")

