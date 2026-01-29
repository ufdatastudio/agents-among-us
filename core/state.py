# core/state.py
import json
import random
from datetime import datetime
from config.settings import ROOMS, NUM_ROUNDS

class GameState:
    def __init__(self, agents, log_manager):
        self.agents = agents
        self.logger = log_manager
        self.live_state_file = "live_state.json"
        self.world_data = {
            "game_id": self.logger.game_id,
            "global": {
                "round": 0,
                "current_phase": "movement",
                "ui_event_log": [],
                "meeting_called": False,
                "body_reported": False,
                "reported_victims": [],
                "meeting_caller": None,
                "meeting_reason_log": ""
            },
            "agents": {},
            "rooms": {room_name: {"occupants": [], "bodies": []} for room_name in ROOMS}
        }

        # Initialize Data
        for agent in agents:
            start_room = random.choice(list(ROOMS.keys()))
            #start_room = "Cafeteria"
            alignment = 'B' if agent.role == 'byzantine' else 'H'
            self.world_data["agents"][agent.name] = {
                "name": agent.name,
                "role": agent.role,
                "color": agent.color,
                "status": "active",
                "location": start_room,
                "last_round_seen": 0,  
                "button_used": False,
                "action_num": agent.action_num,
                "last_action": None,
                "known_bodies": [],
                # --- STATS TRACKING ---
                "stats": {
                    "model_name": agent.model_name,
                    "alignment": alignment,
                    "correct_votes": 0,
                    "incorrect_votes": 0,
                    "skipped_votes": 0,
                    "emergency_meetings": 0,
                    "bodies_reported": 0,
                    "rounds_survived": 0, # whether they made it to game end
                    "eliminations": 0,
                    "won_game": 0,
                    "rounds_survived": 0,
                    "times_eliminated": 0, # if the agent was eliminated, not ejections
                    "ejections": 0,
                    "num_moves": 0, # if stays in place, does not count as move
                    "votes_received": 0
                    
                }
            }
            self.world_data["rooms"][start_room]["occupants"].append(agent.name)
            
    def add_ui_event(self, message, category="info"):
        """Adds a message to the live UI log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {"time": timestamp, "msg": message, "type": category}
        
        # Keep only last 15 events to prevent JSON bloat
        log = self.world_data["global"]["ui_event_log"]
        log.append(entry)
        if len(log) > 15:
            log.pop(0)

    def update_phase(self, phase_name):
        self.world_data["global"]["current_phase"] = phase_name

    def update_round(self, round_num):
        """Updates the current round number in the state."""
        self.world_data["global"]["round"] = round_num
        if round_num > 1: # everyone starts alive
            for agent_name, agent_data in self.world_data["agents"].items():
                if agent_data["status"] == "active":
                    agent_data["stats"]["rounds_survived"] += 1
    
    def record_chat(self, agent_name, message):
        """Pushes an agent's chat message to the UI log."""
        self.add_ui_event(f"{agent_name}: {message}", "chat")
        
    def get_agent_view(self, agent_name, round_num, log_to_file=True):
        """
        Generates the view. 
        If log_to_file is True (Movement Phase), it writes the Observation Block to the log.
        If log_to_file is False (Discussion Phase), it returns the data without writing.
        """
        agent_data = self.world_data["agents"][agent_name]
        loc = agent_data["location"]
        occupants = self.world_data["rooms"][loc]["occupants"]
        current_bodies = self.world_data["rooms"][loc]["bodies"]

        if agent_data["last_round_seen"] < round_num:
            agent_data["known_bodies"] = []
        

        for body in current_bodies:
            # Check if we already know about this body in this specific room
            already_known = False
            for kb in agent_data["known_bodies"]:
                if kb["name"] == body and kb["room"] == loc:
                    already_known = True
                    break
            
            if not already_known:
                agent_data["known_bodies"].append({"name": body, "room": loc})


        # --- 1. Calculate Stats ---
        active_honest = len([a for n, a in self.world_data["agents"].items() 
                             if a["role"] == "honest" and a["status"] == "active"])
        total_active = len([a for n, a in self.world_data["agents"].items() 
                             if a["status"] == "active"])

        if agent_data["role"] == "byzantine":
            count_str = f"Honest Agents Remaining: {active_honest}"
        else:
            count_str = f"Players Remaining: {total_active}"

        current_room_bodies = self.world_data["rooms"][loc]["bodies"]
        
        # Structure the data with room info as requested
        visible_bodies = []
        for body in current_room_bodies:
            visible_bodies.append({"name": body, "room": loc})
    
        # --- 2. Build Surroundings ---
        surroundings = {loc: {"occupants": occupants, "bodies": current_room_bodies}}
        adj_log_str = ""
        if loc in ROOMS:
            for neighbor in ROOMS[loc]:
                neighbor_data = self.world_data["rooms"][neighbor].copy()
                neighbor_data["bodies"] = []
                surroundings[neighbor] = neighbor_data
                # Determine occupants string for log
                occ = [p for p in self.world_data["rooms"][neighbor]["occupants"] if p != agent_name]
                adj_log_str += f"\n    [{neighbor}] -> Occupants: {occ if occ else 'None'}"

        # --- 3. Role Specific (Teammates) ---
        teammate_str = ""
        if agent_data["role"] == "byzantine":
            teammates = [a.name for a in self.agents if a.role == "byzantine" and a.name != agent_name]
            tm_status = []
            for tm in teammates:
                status = self.world_data["agents"][tm]["status"]
                tm_status.append(f"{tm}: {status}")
            teammate_str = f"Teammate(s) Status: {' || '.join(tm_status)}\n"

        # --- 4. Conditional Header ---
        header_str = ""
        # Only check/update header if we are actually logging
        if log_to_file and agent_data["last_round_seen"] < round_num:
            header_str = (
                f"Round {round_num}/{NUM_ROUNDS}\n"
                f"{count_str}\n"
                f"{teammate_str}"
            )
            self.world_data["agents"][agent_name]["last_round_seen"] = round_num



        # --- 5. Construct Observation Block ---
        # Only construct and write this string if we are in the movement phase
        if log_to_file:

            if agent_data["known_bodies"]:
                bodies_log_str = ", ".join([f"{b['name']} (in {b['room']})" for b in agent_data['known_bodies']])
            else:
                bodies_log_str = "None"
            log_entry = (
                f"\n{header_str}"
                f"Current Location: [{loc}]\n"
                f"  -> Occupants: {[o for o in occupants if o != agent_name]}\n"
                f"Adjacent Location(s):{adj_log_str}\n"
                f"Bodies Seen: {bodies_log_str}\n"
            )
            self.logger.write_log("agent", agent_name, log_entry)

        # Return Data Structure
        view = {
            "self": agent_data.copy(),
            "surroundings": surroundings,
            "known_bodies": agent_data["known_bodies"],
            "log_path": self.logger.get_agent_log_path(agent_name),
            "discussion_log_path": self.logger.get_discussion_log_path(agent_data["role"]),
            "results_log_path": self.logger.get_results_log_path()
        }
        return view

    def record_action(self, agent_name, action_text, raw_response=None):
        """Records the selected action to the log file, ensuring it is one line."""
        # Post-process: Remove newlines to keep log clean
        clean_action = action_text.replace("\n", " ").replace("\r", "").strip()
        current_loc = self.world_data["agents"][agent_name]["location"]
        if '->' in clean_action:
            parts = clean_action.split('->')
            action_type = parts[0].strip().lower()
            target_dest = parts[1].strip()
            self.world_data["agents"][agent_name]["last_action"] = action_type
        # record how many actions have been taken by this agent in this round (correlates with round tick)
        self.world_data["agents"][agent_name]["action_num"] += 1
        current_action_num = self.world_data["agents"][agent_name]["action_num"]
        
        self.logger.write_log("agent", agent_name, f"Action {current_action_num} Selected: {clean_action}\n======================\n")
        if raw_response is not None:
            self.logger.write_log("debug", None, f"[DEBUG] {agent_name} | {self.world_data['agents'][agent_name]['role']} | raw response: {raw_response}\n")

        # Stat Tracking
        if action_type == "move" and target_dest.lower() != current_loc.lower():
            self.world_data["agents"][agent_name]["stats"]["num_moves"] += 1

        

    def update_location(self, agent_name, new_room):
        old_room = self.world_data["agents"][agent_name]["location"]
        if agent_name in self.world_data["rooms"][old_room]["occupants"]:
            self.world_data["rooms"][old_room]["occupants"].remove(agent_name)
        self.world_data["rooms"][new_room]["occupants"].append(agent_name)
        self.world_data["agents"][agent_name]["location"] = new_room

    def eliminate_agent(self, target_name, location):
        self.world_data["agents"][target_name]["status"] = "eliminated"
        self.world_data["agents"][target_name]["stats"]["times_eliminated"] += 1

        self.world_data["rooms"][location]["occupants"].remove(target_name)

        self.world_data["rooms"][location]["bodies"].append(target_name)

        self.logger.write_log("results", None, f"ELIMINATION: {target_name}.")
        self.add_ui_event(f"{target_name} eliminated in {location}", "kill")
    
    def report_body(self, reporter_name, body_name):
        self.world_data["global"]["body_reported"] = True
        self.world_data["global"]["reported_victims"].append(body_name)
        self.world_data["global"]["meeting_caller"] = reporter_name
        self.world_data["agents"][reporter_name]["stats"]["bodies_reported"] += 1
        self.update_phase("DISCUSSION")
        

        newly_discovered = []
        if body_name not in self.world_data["global"]["reported_victims"]:
            newly_discovered.append(body_name)
            self.world_data["global"]["reported_victims"].append(body_name)

        for room_data in self.world_data["rooms"].values():
            for body in room_data["bodies"]:
                if body not in self.world_data["global"]["reported_victims"]:
                    newly_discovered.append(body)
                    self.world_data["global"]["reported_victims"].append(body)

        newly_discovered = list(set(newly_discovered))

        # --- FIND BODY LOCATION ---
        body_location = "Unknown Location"
        # Search all rooms to find where the body is currently located
        for room_name, room_data in self.world_data["rooms"].items():
            if body_name in room_data["bodies"]:
                body_location = room_name
                break
        # --------------------------

        # Log to Discussion file
        victims_str = ", ".join(newly_discovered)
        if victims_str == "":
            victims_str = "None"
        #self.logger.write_log("discussion", None, f"** MEETING CALLED by {reporter_name}. Body reported: {body_name}. Additional victims confirmed eliminated: {victims_str} **")
        reason= f"** MEETING CALLED by {reporter_name}. Body reported: {body_name} located in {body_location}. Additional victims confirmed eliminated: {victims_str} **"
        self.world_data["global"]["meeting_reason_log"] = reason

        # Trigger Round End Sequence for Agents
        self._log_round_end(f"{reporter_name}: Body Reported: {body_name}. Additional victims confirmed eliminations: {victims_str}")
        self.add_ui_event(f"Body Report! {reporter_name} found {body_name}", "meeting")
        self._clear_all_bodies()

    def call_emergency_meeting(self, agent_name):
        """Called when button is pressed."""
        self.world_data["global"]["meeting_called"] = True
        self.world_data["global"]["meeting_caller"] = agent_name
        self.world_data["agents"][agent_name]["button_used"] = True
        self.world_data["agents"][agent_name]["stats"]["emergency_meetings"] += 1
        self.update_phase("DISCUSSION")
        
        # Identify any unreported deaths that are revealed by the meeting start
        newly_discovered = []
        for room_data in self.world_data["rooms"].values():
            for body in room_data["bodies"]:
                if body not in self.world_data["global"]["reported_victims"]:
                    newly_discovered.append(body)
                    self.world_data["global"]["reported_victims"].append(body)
        
        victims_str = ", ".join(newly_discovered) if newly_discovered else "None"

        # Log to Discussion file
        #self.logger.write_log("discussion", None, f"** MEETING CALLED by {agent_name} via Emergency Button. Unreported eliminations confirmed this round: {victims_str} **")
        reason = f"** MEETING CALLED by {agent_name} via Emergency Button. Unreported eliminations confirmed this round: {victims_str} **"
        self.world_data["global"]["meeting_reason_log"] = reason
        
        # Trigger Round End Sequence for Agents
        self._log_round_end(f"{agent_name} via Button. Additional unreported eliminations confirmed: {victims_str}")
        self.add_ui_event(f"Emergency Meeting called by {agent_name}", "meeting")
        self._clear_all_bodies()

    def _log_round_end(self, reason):
        """Writes the Round End marker to all active agent logs."""
        
        end_msg = (
            f"Discussion Started because {reason}\n"
            f"Round End.\n"
            f"======================\n"
        )
        for name, data in self.world_data["agents"].items():
            # We log this for all agents so they know why the movement phase stopped
            self.logger.write_log("agent", name, end_msg)

    def _clear_all_bodies(self):
        """Resets bodies on the map so they aren't seen in the next round."""
        for room in self.world_data["rooms"].values():
            room["bodies"] = []
        
    def record_vote(self, agent_name, target, round_num):
        """Logs the agent's vote to their private vote.log file."""
        self.logger.write_log("vote", agent_name, f"Round {round_num}: Voted for {target}")
        self.add_ui_event(f"{agent_name} voted for {target}", "vote")

    def eject_agent(self, agent_name):
        self.world_data["agents"][agent_name]["status"] = "ejected"
        current_loc = self.world_data["agents"][agent_name]["location"]
        self.world_data["agents"][agent_name]["stats"]["ejections"] += 1

        if agent_name in self.world_data["rooms"][current_loc]["occupants"]:
            self.world_data["rooms"][current_loc]["occupants"].remove(agent_name)
        
        self.logger.write_log("results", None, f"EJECTION: {agent_name} was ejected.")
        self.add_ui_event(f"{agent_name} was EJECTED.", "eject")
    
    def save_json(self):
        """Exports the current state to a JSON file for the Live Map."""        
        try:
            with open(self.live_state_file, "w", encoding="utf-8") as f:
                json.dump(self.world_data, f, indent=4)
        except Exception as e:
            print(f"[Warning] Could not save live state: {e}")