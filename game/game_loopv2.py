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

# Clears the terminal for fresh map display.
def clear_terminal():
    os.system("cls" if platform.system() == "Windows" else "clear")

# Defines adjacency map of the ship layout.
rooms = {
    "Cafeteria": ["Weapons", "Navigation", "Storage", "Admin", "MedBay", "UpperEngine"],
    "Weapons": ["Cafeteria", "O2"],
    "Navigation": ["O2", "Shields"],
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

# Sets up logging directory and file for storing simulation output.
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"simulation_log_{timestamp}.json")
log_data = {
    "start_time": timestamp,
    "events": []
}

# Prints the current ship map showing agents and bodies in each room.
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

# Prints the status of all agents including role and elimination status.
def show_agent_status(agents, state, agents_state):
    print("\n--- Agent Status ---")
    print(f"{'Name':<25} | {'Role':<10} | {'Status':<12}")
    print("-" * 50)
    sorted_agents = sorted(agents, key=lambda a: a.name)
    for agent in sorted_agents:
        info = state.get(agent.name, {})
        agent_s_state = agents_state.get(agent.name, {})
        status = "Alive"
        if info.get("ejected"): status = "Ejected"
        elif info.get("eliminated"): status = "Eliminated"
        colored_name = f"{agent.color}{agent.name}\033[0m"
        print(f"{colored_name:<25} | {agents_state[agent.name]['role']:<10} | {status:<12}")
    print("-" * 50)
    
# Handles agent movement, task completion, body discovery, and body reporting.
def movement_phase(state, agents, agents_state, step):
    if "_reported_bodies" not in state:
        state["_reported_bodies"] = set()

    shuffled_agents = list(agents)
    random.shuffle(shuffled_agents)
    
    for agent in shuffled_agents:
        if state[agent.name]["eliminated"]:
            continue
        
        current = state[agent.name]["room"]
        adj = rooms[current]
        

        dest, raw_response = agent.choose_room(current, adj, state, step)
        print(f"\n-- {agent.color}{agent.name}| {agent.agents_state[agent.name]['role']} ---")
        print(f"Raw response: {raw_response}")
        print(f"Destination chosen: {dest}")
        if dest.startswith("Eliminate "):
            target_name = dest.split(" ")[1]
            if target_name in state and state[target_name]["room"] == current and not state[target_name]["eliminated"]:
                state[target_name]["eliminated"] = True
                agents_state[agent.name]["game_stats"]["eliminations_made"] += 1
                state[target_name]["room_body"] = current
                print(f"🔪 {agent.color}{agent.name}\033[0m eliminated {target_name} in {current}")
                
                move_dest, response = agent.choose_room(current, rooms[current], state, step)
                if move_dest in rooms[current] and not move_dest.startswith("Eliminate"):
                    state[agent.name]["room"] = move_dest
                    print(f"👣 {agent.color}{agent.name}\033[0m moved from {current} to {move_dest}")
                else:
                    print(f"🧘{agent.name} stayed in {current}<--- Actual Response: {response}")
            else:
                 print(f"ERROR: {agent.color}{agent.name}\033[0m failed to eliminate {target_name}. <-- Actual Response: {raw_response}")
                 state[agent.name]["room"] = random.choice(adj)

        elif dest == "Press Emergency Button" and current == "Cafeteria" and not agents_state[agent.name]["game_stats"]["meeting_called"]:
            print(f"🚨 {agent.color}{agent.name}\033[0m pressed the emergency button in the Cafeteria!")
            agents_state[agent.name]["game_stats"]["meeting_called"] = True
            state["emergency_meeting_called"] = True
        elif dest in rooms:
            state[agent.name]["room"] = dest
            if dest != current:
                print(f"👣 {agent.color}{agent.name}\033[0m moved from {current} to {dest}")
            else:
                print(f"🧘{agent.color}{agent.name}\033[0m stayed in {current}) <--- Actual Response: {raw_response}")
                state[agent.name]["room"] = current

        new_room = state[agent.name]["room"]
        seen = [ a.name for a in agents if state[a.name].get("room") == new_room and a.name != agent.name and not state[a.name].get("eliminated", False) ]
        seen_bodies = [ a.name for a in agents if state[a.name].get("eliminated") and state[a.name].get("room_body") == new_room and a.name not in state["_reported_bodies"] ]
        
        agents_state[agent.name]["perception"].append({ "room": new_room, "agents_seen": seen, "bodies_seen": seen_bodies })

        if seen_bodies and not state[agent.name]["eliminated"]:
            if agent.decide_to_report(seen_bodies, new_room, state): 
                bodies_str = ", ".join(seen_bodies)
                agent_msg = f"I just found the bodie(s) of {bodies_str} in {new_room}! Reporting it now!"
                print(f"📢 {agent.color}{agent.name}\033[0m reports: {agent_msg}") 
                agents_state[agent.name]["messages"].append(agent_msg)
                agents_state[agent.name]["game_stats"]["bodies_reported"] += 1
                state["_reported_bodies"].update(seen_bodies)
                state["body_is_reported"] = True
        
        # Honest agents perform their tasks if in the correct room
        if agent.__class__.__name__ == "HonestAgent":
            if not state[agent.name]["task_done"]:
                if state[agent.name]["room"] == state[agent.name]["task_room"]:
                    if state[agent.name]["doing_task"]:
                        state[agent.name]["task_done"] = True
                        print(f"✅ {agent.color}{agent.name}\033[0m completed their task in {state[agent.name]['room']}.")
                    else:
                        state[agent.name]["doing_task"] = True
                        print(f"🛠️ {agent.color}{agent.name}\033[0m started task in {state[agent.name]['room']}.")
                else:
                    state[agent.name]["doing_task"] = False




# Executes a full round including movement, discussion, voting, and ejection logic.
def run_game_round(game_id, step, state, agents, agents_state):
    max_movement_phases = 5
    meeting_triggered = False

    for movement_stage in range(1, max_movement_phases + 1):
        print(f"\n--- Movement Phase {step}.{movement_stage} ---")
        time.sleep(1) # Pause for readability
        
        movement_phase(state, agents, agents_state, f"{step}.{movement_stage}")
        show_ship_map(state, agents)

        # Check for win conditions immediately after movement
        alive_byzantines = [a for a in agents if agents_state[a.name]["role"] == "byzantine" and not state[a.name]["eliminated"]]
        alive_honest = [a for a in agents if agents_state[a.name]["role"] == "honest" and not state[a.name]["eliminated"]]
        if not alive_byzantines or len(alive_byzantines) >= len(alive_honest):
            show_agent_status(agents, state, agents_state)
            return

        # If a meeting is called, break the movement loop
        if state.get("body_is_reported") or state.get("emergency_meeting_called"):
            meeting_triggered = True
            break
    
    # If a meeting was triggered, proceed to discussion and voting
    if meeting_triggered:
        state["body_is_reported"] = False
        state["emergency_meeting_called"] = False
        
        print(f"\n--- DISCUSSION (Round {step}) ---")
        messages = {}
        for agent in agents:
            if not state[agent.name]["eliminated"]:
                history = agents_state[agent.name].get("perception", [])
                message = agent.simulate_message(history, state)
                if message:
                    print(f"🗣️ {agent.color}{agent.name}\033[0m says: {message}")
                    messages[agent.name] = message
                    log_reflection(game_id, step, agent.name, message)

        # Allow agents to respond to the initial messages
        for agent in agents:
            if not state[agent.name]["eliminated"]:
                response = agent.respond_to_message(messages, agents_state[agent.name].get("perception", []), state)
                if response:
                    print(f"💬   {agent.color}{agent.name}\033[0m responds: {response}")
                    log_reflection(game_id, step, agent.name, response)
        
        print("\n--- VOTING ---")
        votes = {}
        for agent in agents:
            if not state[agent.name]["eliminated"]:
                voter, target = agent.vote_for_ejection(state, step)
                votes[voter] = target
                log_vote(game_id, step, voter, target)
                print(f"🗳️ {agent.color}{voter}\033[0m voted for {target}")

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
                print(f"Consensus Agreement Level: {agreement_level:.2f}")
                state[ejected]["eliminated"] = True
                state[ejected]["ejected"] = True
            else:
                print("\n--- VOTE RESULT: No one was ejected. ---")
                print(f"Consensus Agreement Level: {agreement_level:.2f}")

            log_consensus(game_id, step, f"Eject {ejected}", agreement_level)
    
    # Log metadata at the end of the entire round
    alive_agents = [a for a in agents if not state[a.name]["eliminated"]]
    log_round_metadata(game_id, step, len(alive_agents), len(agents) - len(alive_agents))
    
    # Show final status for the round
    show_agent_status(agents, state, agents_state)



# Saves all collected log data to disk as JSON.
def finalize_log(game_id, duration, victory_condition):
    log_game_summary(game_id, victory_condition, duration)
    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Simulation results saved to database for game_id: {game_id}")
    print(f"Full log saved to {log_file_path}")
    if duration is not None:
        print(f"Total game time: {duration:.2f} seconds")

