# Handles simulation logic, including movement, voting, visualization, and game state updates.
import random
import config
from agents.agent_setup import create_agents
import json
from datetime import datetime
import os
import platform
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
    "Cafeteria": ["Weapons", "Navigation", "Storage", "Admin", "MedBay", "Upper Engine"],
    "Weapons": ["Cafeteria", "O2"],
    "Navigation": ["O2", "Shields"],
    "O2": ["Weapons", "Navigation", "Cafeteria"],
    "Shields": ["Navigation", "Communications", "Storage"],
    "Communications": ["Shields"],
    "Admin": ["Cafeteria", "Storage"],
    "Storage": ["Admin", "Shields", "Electrical", "Lower Engine", "Cafeteria"],
    "Electrical": ["Storage", "Lower Engine"],
    "Lower Engine": ["Storage", "Electrical", "Security", "Reactor"],
    "Security": ["Lower Engine", "Upper Engine", "Reactor"],
    "Reactor": ["Security", "Upper Engine", "Lower Engine"],
    "Upper Engine": ["Reactor", "Security", "MedBay", "Cafeteria"],
    "MedBay": ["Upper Engine", "Cafeteria"]
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
    clear_terminal()
    mapping = {room: [] for room in rooms}
    bodies = {room: [] for room in rooms}
    for agent, info in state.items():
        if not isinstance(info, dict):
            continue
        if not info.get("eliminated", False):
            agent_obj = next((a for a in agents if a.name == agent), None)
            mapping[info["room"]].append(f"{agent_obj.color}{agent}" if agent_obj else agent)
        elif info.get("room_body"):
            agent_obj = next((a for a in agents if a.name == agent), None)
            color = agent_obj.color if agent_obj else ""
            bodies[info["room_body"]].append(f"💀{color}{agent}")

    def fmt(room):
        entries = mapping.get(room, []) + bodies.get(room, [])
        return f"[{room}]" if not entries else f"[{room}: {' '.join(entries)}]"

    print("\n--- Ship Map ---\n")
    print(f"{fmt('Upper Engine')}               {fmt('Cafeteria')}              {fmt('Weapons')}")
    print("")
    print(f"{fmt('Reactor')}        {fmt('Security')}       {fmt('MedBay')}        {fmt('O2')}      {fmt('Navigation')}")
    print("")
    print(f"{fmt('Lower Engine')}   {fmt('Electrical')}     {fmt('Storage')}       {fmt('Admin')}   {fmt('Shields')}")
    print("")
    print(f"{' ' * 55}{fmt('Communications')}")

    print("\nPlayer Locations:")
    for room, occupants in mapping.items():
        if occupants:
            print(f"{room}: {' '.join(occupants)}")

# Handles agent movement, task completion, body discovery, and body reporting.
def movement_phase(state, agents, agents_state, stream,step):
    if "_reported_bodies" not in state:
        state["_reported_bodies"] = set()

    # Randomize turn order 
    shuffled_agents = list(agents)
    random.shuffle(shuffled_agents)
    
    for agent in shuffled_agents:
        if state[agent.name]["eliminated"]:
            continue
        current = state[agent.name]["room"]
        adj = rooms[current]
        dest = agent.choose_room(current, adj, state,step)

        # Handle kill action if applicable
        if dest.startswith("Eliminate "):
            target_name = dest.split(" ")[1]
            if target_name in state and state[target_name]["room"] == current and not state[target_name]["eliminated"]:
                state[target_name]["eliminated"] = True
                agents_state[agent.name]["game_stats"]["eliminations_made"] += 1 
                state[target_name]["room_body"] = current
                print(f"{agent.name} eliminated {target_name} in {current}")
                yield from stream.flush()
            move_dest = agent.choose_room(current, rooms[current], state,step)
            if move_dest in rooms[current]:
                state[agent.name]["room"] = move_dest
                print(f"{agent.name} moved from {current} to {move_dest}")
                yield from stream.flush()
            else:
                print(f"{agent.name} stayed in {current}")
                yield from stream.flush()
        
        # Emergency meeting button press
        elif dest == "Press Emergency Button" and current == "Cafeteria" and not agents_state[agent.name]["game_stats"]["meeting_called"]:
            print(f"{agent.name} pressed the emergency button in the Cafeteria!")
            agents_state[agent.name]["game_stats"]["meeting_called"] = True
            state["emergency_meeting_called"] = True # Set a temporary flag for this round
            yield from stream.flush()
        else:
            state[agent.name]["room"] = dest
            if dest == current:
                print(f"{agent.name} stayed in {current}")
            else:
                print(f"{agent.name} moved from {current} to {dest}")
            yield from stream.flush()

        # Update perception of other agents and bodies
        seen = [
            a for a, info in state.items()
            if isinstance(info, dict) and info.get("room") == dest and a != agent.name and not info.get("eliminated", False)
        ]
        room = state[agent.name]["room"]
        seen_bodies = [
            a for a, info in state.items()
            if isinstance(info, dict)
               and info.get("eliminated")
               and info.get("room_body") == room
               and a not in state["_reported_bodies"]
        ]
        agents_state[agent.name]["perception"].append({
            # Add the round number here 
            "room": room,
            "agents_seen": seen,
            "bodies_seen": seen_bodies
        })


        if seen_bodies and not state[agent.name]["eliminated"]:
            # Pass the whole list to the agent's decision method
            if agent.decide_to_report(seen_bodies, room, state): 
                # Update the message to include all bodies
                bodies_str = ", ".join(seen_bodies)
                agent_msg = f"I just found the bodie(s) of {bodies_str} in {room}! Reporting it now!"
                print(f"{agent.name} reports: {agent_msg}") 
                agents_state[agent.name]["messages"].append(agent_msg)
                agents_state[agent.name]["game_stats"]["bodies_reported"] += 1 
                # Update the reported set with all the bodies found
                state["_reported_bodies"].update(seen_bodies)
                state["body_is_reported"] = True
                yield from stream.flush()


        # Honest agents perform their tasks if in the correct room
        if agent.__class__.__name__ == "HonestAgent":
            if not state[agent.name]["task_done"]:
                if state[agent.name]["room"] == state[agent.name]["task_room"]:
                    if state[agent.name]["doing_task"]:
                        state[agent.name]["task_done"] = True
                        print(f"{agent.name} completed their task in {state[agent.name]['room']}.")
                        yield from stream.flush()
                    else:
                        state[agent.name]["doing_task"] = True
                        print(f"{agent.name} started task in {state[agent.name]['room']}.")
                        yield from stream.flush()
                else:
                    state[agent.name]["doing_task"] = False

# Executes a full round including movement, discussion, voting, and ejection logic.
def run_game_round(game_id, step, state, agents, agents_state, stream):
    yield from movement_phase(state, agents, agents_state, stream,step)
    show_ship_map(state, agents)
    yield from stream.flush()

    # Check for endgame conditions
    alive = [a for a in state if isinstance(state[a], dict) and not state[a]["eliminated"]]
    alive_byzantines = [a for a in alive if agents_state[a]["role"] == "byzantine"]
    alive_honest = [a for a in alive if agents_state[a]["role"] == "honest"]

    log_round_metadata(game_id, step, len(alive), len(state) - len(alive))

    victory = None
    if len(alive_byzantines) == 0:
        victory = "Honest Agents Win"
    elif len(alive_byzantines) >= len(alive_honest):
        victory = "Byzantine Agents Win"
    elif step >= NUM_ROUNDS:
        victory = "Max Rounds Reached – Honest Agents Win"

    if victory:
        log_consensus(game_id, step, victory, 1.0)
        print(f"\nGame Over: {victory}")
        # finalize_log()
        yield from stream.flush()
        return

    # Discussion and voting phase (if any body was seen)
    messages = {}
    # any_body_seen = any(
    #     isinstance(agent_state, dict)
    #     and agent_state.get("perception")
    #     and len(agent_state["perception"][-1].get("bodies_seen", [])) > 0
    #     and not agent_state.get("eliminated", False)
    #     for agent_state in state.values()
    # )

    if state.get("body_is_reported") or state.get("emergency_meeting_called"):
        state["emergency_meeting_called"] = False # Reset for next round
        print(f"\n--- DISCUSSION (Round {step}) ---")
        yield from stream.flush()
        for agent in agents:
            if state[agent.name]["eliminated"]:
                continue
            history = agents_state[agent.name].get("perception", [])
            message = agent.simulate_message(history)
            yield from stream.flush()
            if message:
                messages[agent.name] = message
                log_reflection(game_id, step, agent.name, message)

        for agent in agents:
            if state[agent.name]["eliminated"]:
                continue
            response = agent.respond_to_message(messages, agents_state[agent.name].get("perception", []))
            yield from stream.flush()
            if response:
                log_reflection(game_id, step, agent.name, response)

        # Voting and ejection logic
        print(f"\n--- VOTING (Round {step}) ---")
        yield from stream.flush()
        votes = {}
        for agent in agents:
            if state[agent.name]["eliminated"]:
                continue
            voter, target = agent.vote_for_ejection(state, step)
            votes[voter] = target
            log_vote(game_id, step, voter, target)
            print(f"{voter} voted to eject {target}")
            yield from stream.flush()

        vote_counts = {}
        for target in votes.values():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # Track who recieved votes
        for agent_name in state:
            if isinstance(state[agent_name], dict):
                voters = [voter for voter, target in votes.items() if target == agent_name]
                if voters:
                    agents_state[agent_name]["game_stats"]["votes_recieved"].append({
                        "round": step,
                        "voters": voters
                    })
        ejected = max(vote_counts, key=vote_counts.get)
        agreement_level = vote_counts[ejected] / len(votes)



        if ejected in agents_state:
            correct = (agents_state[ejected]["role"] == "byzantine")
            print(f"\nEjected: {ejected}")
            print(f"Vote {'correct' if correct else 'incorrect'} — Role was: {agents_state[ejected]['role']}")
            print(f"Consensus Agreement Level: {agreement_level:.2f}")
            yield from stream.flush()
            state[ejected]["eliminated"] = True
            state[ejected]["ejected"] = True
        else:
            correct = False
            print(f"\nNo one was ejected.")
            print(f"Consensus Agreement Level: {agreement_level:.2f}")
            yield from stream.flush()

        log_consensus(game_id, step, f"Eject {ejected}", agreement_level)
        for voter, target in votes.items():
            agents_state[voter]["game_stats"]["votes_cast"].append({
                "target": target,
                "was_correct": (target == ejected and correct)
            })

        # for agent in agents:
        #     if agent.name in votes:
        #         voted_correctly = (votes[agent.name] == ejected and correct)
        #         # agent.update_trust(ejected, voted_correctly)
        #         # delta = 20 if voted_correctly else -20
        #         # log_trust_change(game_id, step, agent.name, ejected, delta)

        for agent in agents:
            if state[agent.name]["eliminated"]:
                continue
            log_game_event(game_id, step, agent.name, state[agent.name]["room"],
                           state[agent.name]["eliminated"], True, votes[agent.name],
                           ejected, votes[agent.name] == ejected, 0, True, agreement_level)

# Saves all collected log data to disk as JSON.
def finalize_log(game_id, duration, victory_condition):
    log_game_summary(game_id, victory_condition, duration)
    with open(log_file_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Simulation results saved to {log_file_path}")
    if duration is not None:
        print(f"Total Game time: {duration:.2f} seconds")

# Generates the current map as HTML for frontend display.
def generate_map_html(state=None, agents=None):
    mapping = {room: [] for room in rooms}
    bodies = {room: [] for room in rooms}
    if state and agents:
        for agent, info in state.items():
            if not isinstance(info, dict):
                continue
            if not info.get("eliminated", False):
                agent_obj = next((a for a in agents if a.name == agent), None)
                mapping[info["room"]].append(f"{agent_obj.color}{agent}" if agent_obj else agent)
            elif info.get("room_body"):
                agent_obj = next((a for a in agents if a.name == agent), None)
                color = agent_obj.color if agent_obj else ""
                bodies[info["room_body"]].append(f"💀{color}{agent}")

    def fmt(room):
        entries = mapping.get(room, []) + bodies.get(room, [])
        return f"<strong>{room}</strong>" if not entries else f"<strong>{room}</strong>: {' '.join(entries)}"

    return f"""
    <div>
        <p>{fmt('Upper Engine')} | {fmt('Cafeteria')} | {fmt('Weapons')}</p>
        <p>{fmt('Reactor')} | {fmt('Security')} | {fmt('MedBay')} | {fmt('O2')} | {fmt('Navigation')}</p>
        <p>{fmt('Lower Engine')} | {fmt('Electrical')} | {fmt('Storage')} | {fmt('Admin')} | {fmt('Shields')}</p>
        <p>{fmt('Communications')}</p>
    </div>
    """

# Generates agent status summary table in HTML.
def generate_agent_status_html(state, agents):
    rows = []
    for agent in agents:
        info = state.get(agent.name, {})
        color = agent.color
        model = agent.model_name
        role = agent.agents_state[agent.name]["role"]

        if info.get("ejected"):
            status = "Ejected"
        elif info.get("eliminated"):
            status = "Eliminated"
        else:
            status = "Alive"

        rows.append(f"<tr><td>{color}{agent.name}</td><td>{role.capitalize()}</td><td>{model}</td><td>{status}</td></tr>")

    return f"""
    <h3>Agent Status</h3>
    <table border="1" style='color: #0f0; border-collapse: collapse; width: 100%;'>
        <tr><th>Name</th><th>Role</th><th>Model</th><th>Status</th></tr>
        {''.join(rows)}
    </table>
    """
