# game/engine.py
import random
import re
from config.settings import MAX_MOVEMENT_PHASES, ROOMS, NUM_BYZ, NUM_HONEST
from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from core.state import GameState
from core.logger import LogManager

class GameEngine:
    def __init__(self, game_id, num_agents=NUM_BYZ + NUM_HONEST):
        self.game_id = game_id
        self.num_agents = num_agents
        self.agents = []
        self.state = None
        self.logger = None

    def setup(self, model_list):
        num_byz = NUM_BYZ
        colors = ["🔴", "🔵", "🟢", "💗", "🟠", "🟡", "⚫", "⚪", "🟣", "🟤"]
        indices = list(range(self.num_agents))
        byz_indices = random.sample(indices, num_byz)
        byz_names = [f"Agent_{i}" for i in byz_indices]
        
        for i in range(self.num_agents):
            name = f"Agent_{i}"
            color = colors[i % len(colors)]

            # Here  potentially assign different models based on agent index or role
            agent_model = model_list[0]
            if i in byz_indices:
                self.agents.append(ByzantineAgent(name, color, [b for b in byz_names if b != name], agent_model))
            else:
                self.agents.append(HonestAgent(name, color, agent_model))

        self.logger = LogManager(self.game_id, self.agents)
        self.state = GameState(self.agents, self.logger)
        print(f"--- Game Setup Complete. Logs at: logs/Game_{self.game_id} ---")

    def run_movement_phase(self, round_num):
        self.logger.write_log("results", None, f"\n=== Round {round_num} ===")
        print(f"\n--- Round {round_num} Movement Phase ---")
        event_occurred_in_round = False
        for phase_tick in range(1, MAX_MOVEMENT_PHASES + 1):
            print(f"Tick {phase_tick}...")
            active_agents = [a for a in self.agents if self.state.world_data["agents"][a.name]["status"] == "active"]
            
            # --- 1. GATHER DECISIONS ---
            decisions = []
            for agent in active_agents:
                view = self.state.get_agent_view(agent.name, round_num, log_to_file=True)
                decision = agent.think_and_act(view, round_num)
                decisions.append((agent, decision))
            
            reports, kills, buttons, moves = [], [], [], []
            
            # Sort decisions into categories
            for agent, result in decisions:
                action, target, raw_response = result
                self.state.record_action(agent.name, f"{action} -> {target}", raw_response)
                
                if action == "report": reports.append((agent, target))
                elif action == "tag": kills.append((agent, target))
                elif action == "button": buttons.append(agent)
                elif action == "move": moves.append((agent, target))

            # --- 2. EXECUTE KILLS (Highest Priority) ---
            # Kills happen based on positions at START of tick. 
            # If you are targeted, you die before you can move/report/press button.
            newly_dead_agents = set()
            
            for killer, victim_name in kills:
                # Validation: Killer must be active, Victim must be active, Must be same room
                k_data = self.state.world_data["agents"][killer.name]
                v_data = self.state.world_data["agents"][victim_name]
                
                if (k_data["status"] == "active" and 
                    v_data["status"] == "active" and 
                    k_data["location"] == v_data["location"]):
                    
                    self.state.eliminate_agent(victim_name, k_data["location"])
                    k_data["stats"]["eliminations"] += 1
                    newly_dead_agents.add(victim_name)
                    event_occurred_in_round = True

            # --- 3. EXECUTE MEETINGS (Report / Button) ---
            meeting_triggered = False
            
            # Check Reports (Filter out reporters who just died)
            valid_reports = [r for r in reports if r[0].name not in newly_dead_agents]
            if valid_reports:
                # Use the first valid report
                reporter, body = valid_reports[0]
                self.state.report_body(reporter.name, body)
                meeting_triggered = True
                event_occurred_in_round = True
            
            # Check Buttons (Filter out pressers who just died)
            # Only trigger button if no body report happened this tick (Body report takes precedence usually, or equal)
            if not meeting_triggered:
                valid_buttons = [b for b in buttons if b.name not in newly_dead_agents]
                if valid_buttons:
                    self.state.call_emergency_meeting(valid_buttons[0].name)
                    meeting_triggered = True
                    event_occurred_in_round = True

            # If a meeting occurred, we STOP here. Moves are cancelled.
            if meeting_triggered:
                # Cleanup action counts for discussion
                self._reset_action_counts()
                return True

            # --- 4. EXECUTE MOVES (Lowest Priority) ---
            # Only move agents who are NOT dead
            for mover, room in moves:
                if mover.name not in newly_dead_agents:
                    if room in ROOMS:
                        self.state.update_location(mover.name, room)
            
            # --- End of Tick ---
        # End of movement phase cleanup
        self._reset_action_counts()
        
        if not event_occurred_in_round:
            self.logger.write_log("results", None, f"No Eliminations or Discussions in Round {round_num}")

        return False

    def _reset_action_counts(self):
        for agent in self.agents:
            if self.state.world_data["agents"][agent.name]["status"] == "active":
                self.state.world_data["agents"][agent.name]["action_num"] = 0
                self.state.world_data["agents"][agent.name]["last_action"] = None

    def run_discussion_phase(self, round_num):
        self.logger.write_log("discussion", None, f"\n=== Round {round_num} ===")
        reason = self.state.world_data["global"]["meeting_reason_log"]
        if reason:
            self.logger.write_log("discussion", None, reason)
        
        # 1. Identify Active Agents
        active_agents = [a for a in self.agents if self.state.world_data["agents"][a.name]["status"] == "active"]
        
        # 2. Determine Speaking Order (Caller goes first)
        caller_name = self.state.world_data["global"]["meeting_caller"]
        discussion_order = []
        
        # If the caller is alive/active, put them first
        caller_obj = next((a for a in active_agents if a.name == caller_name), None)
        if caller_obj:
            discussion_order.append(caller_obj)
            
        # Add everyone else
        for agent in active_agents:
            if agent.name != caller_name:
                discussion_order.append(agent)

        # 3. Conversation (2 Rounds)
        for discussion_round in range(2):            
            for agent in discussion_order:
                view = self.state.get_agent_view(agent.name, round_num, log_to_file=False) 
                msg = agent.participate_in_discussion("", view, round_num)
                
                clean_msg = msg.replace("\n", " ").replace("\r", "").strip()
                clean_msg = re.sub(r"^(\*\*)?Agent_\d+:?(\*\*)?:?\s*", "", clean_msg, flags=re.IGNORECASE)
                clean_msg = clean_msg.strip('"').strip("'")
                
                formatted_msg = f"{agent.name}: {clean_msg}"
                self.logger.write_log("discussion", None, formatted_msg)
                print(formatted_msg)

        # 2. Voting
        votes = {}
        for agent in active_agents:
            view = self.state.get_agent_view(agent.name, round_num, log_to_file=False)
            vote = agent.vote(view, [a.name for a in active_agents] + ["SKIP"])
            votes[agent.name] = vote
            self.state.record_vote(agent.name, vote, round_num)

            voter_stats = self.state.world_data["agents"][agent.name]["stats"]
            voter_role = self.state.world_data["agents"][agent.name]["role"]

            if vote == "SKIP":
                voter_stats["skipped_votes"] += 1
            elif vote in self.state.world_data["agents"]:
                self.state.world_data["agents"][vote]["stats"]["votes_recieved"] += 1
                target_role = self.state.world_data["agents"][vote]["role"]
                # Honest -> Byzantine = Correct
                # Byzantine -> Honest = Correct (Success for Byz)
                # Honest -> Honest = Incorrect
                # Byzantine -> Byzantine = Incorrect (Betrayal/Mistake)
                is_correct = False
                if voter_role == "honest" and target_role == "byzantine":
                    is_correct = True
                elif voter_role == "byzantine" and target_role == "honest":
                    is_correct = True
                if is_correct:
                    voter_stats["correct_votes"] += 1
                else:
                    voter_stats["incorrect_votes"] += 1

        # Tally
        tally = {}
        for v in votes.values(): tally[v] = tally.get(v, 0) + 1
        
        self.logger.write_log("results", None, f"Round {round_num} Votes Received: {tally}")
        
        if tally:
            # Sort by vote count descending
            sorted_votes = sorted(tally.items(), key=lambda x: x[1], reverse=True)
            winner, score = sorted_votes[0]
            is_tie = False
            if len(sorted_votes) > 1 and sorted_votes[1][1] == score:
                is_tie = True
            
            if is_tie:
                self.logger.write_log("discussion", None, "** No one was ejected (Tie) **")
            elif winner == "SKIP":
                self.logger.write_log("discussion", None, "** No one was ejected (Skipped) **")
            else:
                self.state.eject_agent(winner)
                self.logger.write_log("discussion", None, f"** {winner} was EJECTED **")
        else:
            self.logger.write_log("discussion", None, "** No votes cast **")
        
        # Reset flags
        self.state.world_data["global"]["body_reported"] = False
        self.state.world_data["global"]["meeting_called"] = False
        
        status_snapshot = {n: d["status"] for n, d in self.state.world_data["agents"].items()}
        self.logger.write_log("results", None, f"Player Statuses: {status_snapshot}")
        
    def check_win_condition(self):
            active = [d for n, d in self.state.world_data["agents"].items() if d["status"] == "active"]
            byz = [a for a in active if a["role"] == "byzantine"]
            honest = [a for a in active if a["role"] == "honest"]
            print(f"[DEBUG] Check Win: Byz={len(byz)} | Honest={len(honest)}")
            
            result = None
            if not byz: 
                result = "Honest Agents Win"
            elif len(byz) >= len(honest): 
                result = "Byzantines Win"
                
            if result:
                self.finalize_stats(result)
                return result
                
            return None
    
    def finalize_stats(self, result):
        """Calculates final game stats (won/loss) and exports to CSV."""
        winning_team_role = "honest" if "Honest" in result else "byzantine"
        
        for agent_name, data in self.state.world_data["agents"].items():
            stats = data["stats"]
            
            # Determine Win
            if data["role"] == winning_team_role:
                stats["won_game"] = 1
            else:
                stats["won_game"] = 0
                        
        # Export
        self.logger.export_stats(self.state.world_data["agents"])