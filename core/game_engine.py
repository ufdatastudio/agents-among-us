import random
import re
import time
from config.settings import MAX_MOVEMENT_PHASES, ROOMS, NUM_BYZ, NUM_HONEST, NUM_ROUNDS as DEFAULT_NUM_ROUNDS
from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from core.state import GameState
from core.logger import LogManager
import os 
import joblib
from nltk.corpus import stopwords
import pandas as pd

class Observer:
    def __init__(self, model_dir="results/classifiers/models/"):
        self.models = {}
        paths = {
            "LogisticRegression": os.path.join(model_dir, "lr.joblib"),
            "SGD": os.path.join(model_dir, "sgd.joblib"),
            "SVM": os.path.join(model_dir, "svm.joblib")
        }
        
        for name, path in paths.items():
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"[Observer] Loaded {name} model from {path}")
        
        self.locations = [
            "Reactor", "Security", "UpperEngine", "LowerEngine", "MedBay", 
            "Cafeteria", "Electrical", "Storage", "Admin", "Weapons", 
            "Shields", "O2", "Navigation", "Communications",

            "Clock", "Air cooling", "Liquid cooling", "Logs", "Diagnostics",
            "Bus", "Cpu", "Bios", "Ssd", "Io", "Gpu", "Vrm", "Network", "Firewall"
        ]
        self.loc_pattern = re.compile(r'\b(?:' + '|'.join(self.locations) + r')\b', flags=re.IGNORECASE)
        self.agent_pattern = re.compile(r'\bagent_\d+\b', flags=re.IGNORECASE)
        self.stop_words = set(stopwords.words('english'))

    def _preprocess(self, text):
        text = text.lower()
        text = self.loc_pattern.sub('place', text)
        text = self.agent_pattern.sub('agent_x', text)
        text = re.sub(r'[^a-z0-9\s_]', '', text)
        tokens = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)

    def analyze_round(self, statements):
        """
        statements: List of dicts {'Agent': str, 'Text': str, 'Reported': int, 'S_Num': int}
        Returns: Dict of {agent_name: {model_name: probability}} or None
        """
        if not statements or not self.models:
            return None

        df = pd.DataFrame(statements)
        df['Clean_Text'] = df['Text'].apply(self._preprocess)
        
        # Rename columns to match what the pipeline expects
        input_df = df[['Clean_Text', 'Reported', 'S_Num']].rename(
            columns={'Clean_Text': 'Text', 'S_Num': 'Statement_Num'}
        )

        print("\n" + "="*60)
        print(f"{'OBSERVER: SUSPICION PROBABILITY':^60}")
        print("="*60)
        print(f"{'AGENT':<12} | {'LR %':<8} | {'SGD %':<8} | {'SVM %':<8}")
        print("-" * 60)

        results = {}
        scores_by_agent = {}  
        
        for name, clf in self.models.items():
            probs = clf.predict_proba(input_df)[:, 1]
            df[f'{name}_Prob'] = probs
            # Aggregate: Mean probability per agent
            results[name] = df.groupby('Agent')[f'{name}_Prob'].mean()

        agents = results["LogisticRegression"].index if "LogisticRegression" in results else []
        
        for agent_name in agents:
            lr_p = results.get("LogisticRegression", {}).get(agent_name, 0)
            sgd_p = results.get("SGD", {}).get(agent_name, 0)
            svm_p = results.get("SVM", {}).get(agent_name, 0)
            
            scores_by_agent[agent_name] = {
                "LogisticRegression": float(lr_p),
                "SGD": float(sgd_p),
                "SVM": float(svm_p)
            }
            
            print(f"{agent_name:<12} | {lr_p*100:>6.1f}% | {sgd_p*100:>6.1f}% | {svm_p*100:>6.1f}%")
        
        print("="*60 + "\n")
        
        return scores_by_agent  

class GameEngine:
    def __init__(self, game_id, num_agents=NUM_BYZ + NUM_HONEST, num_rounds=DEFAULT_NUM_ROUNDS):
        self.game_id = game_id
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.agents = []
        self.state = None
        self.logger = None
        self.observer = Observer()
        
        # ML Classifier config (will be set during setup)
        self.enabled_classifiers = {}

    def setup(self, composition):
        scen_name = composition.get("name", "Unknown_Scenario")
        
        # check if composition has exact agent configuration
        if "agents" in composition:
            # use exact agent configuration from frontend
            agents_config = composition["agents"]
            
            byz_names = [f"Agent_{a['agent_num']}" for a in agents_config if a['role'] == 'byzantine']
            
            for agent_config in agents_config:
                agent_num = agent_config['agent_num']
                agent_name = f"Agent_{agent_num}"
                model = agent_config['model']
                role = agent_config['role']
                color = agent_config['color']
                
                if role == 'byzantine':
                    teammates = [name for name in byz_names if name != agent_name]
                    self.agents.append(
                        ByzantineAgent(agent_name, color, teammates, model)
                    )
                else:  # honest
                    self.agents.append(
                        HonestAgent(agent_name, color, model)
                    )
            
            print(f"Created {len(self.agents)} agents with EXACT configuration from frontend")
            
        else:
            honest_models = composition["honest_model"]
            byz_models = composition["byzantine_model"]
            n_honest = composition["honest_count"]
            n_byz = composition["byzantine_count"]
            
            colors = ["🔴", "🟠", "🟡", "🟩", "🟢", "🔷", "🔵", "🟣", "🟤", "💗", "⚪", "⚫"]
            
            # Create Byzantine Agents
            byz_names = [f"Agent_{i}" for i in range(n_byz)]
            for i, name in enumerate(byz_names):
                assigned_model = byz_models[i % len(byz_models)]
                teammates = [b for b in byz_names if b != name]
                self.agents.append(
                    ByzantineAgent(name, colors[i], teammates, assigned_model)
                )

            # Create Honest Agents
            start_index = n_byz 
            for i in range(n_honest):
                name = f"Agent_{start_index + i}"
                color = colors[(start_index + i) % len(colors)]
                assigned_model = honest_models[i % len(honest_models)]
                self.agents.append(
                    HonestAgent(name, color, assigned_model)
                )

        # random.shuffle(self.agents)  # commented out bc we don't want to shuffle agents

        # Optional per-role, per-phase prompt overrides from composition
        prompts_cfg = composition.get("prompts")
        if isinstance(prompts_cfg, dict):
            for agent in self.agents:
                try:
                    role_key = "honest" if getattr(agent, "role", "") == "honest" else "byzantine"
                    role_prompts = prompts_cfg.get(role_key)
                    if isinstance(role_prompts, dict):
                        setattr(agent, "prompt_overrides", role_prompts)
                except Exception:
                    # Fail silently; fallback to default prompts for that agent
                    pass

        self.logger = LogManager(self.game_id, self.agents, scen_name)
        self.state = GameState(self.agents, self.logger)
        
        # Set up ML classifiers from composition
        if "enabled_classifiers" in composition:
            self.enabled_classifiers = composition["enabled_classifiers"]
            self.state.set_classifiers(self.enabled_classifiers)
            classifiers_enabled = [k.upper() for k, v in self.enabled_classifiers.items() if v]
            if classifiers_enabled:
                print(f"Observer initialized with classifiers: {', '.join(classifiers_enabled)}")
        
        self.state.save_json()
        print(f"--- Game Setup Complete. Logs at: {self.logger.base_dir} ---")
        
    def run_movement_phase(self, round_num):
        self.logger.write_log("results", None, f"\n=== Round {round_num} ===")
        print(f"\n--- Round {round_num} Movement Phase ---")
        self.state.update_round(round_num)
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
                # wait a bit between agent actions to be watchable
                time.sleep(1)
            
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
            newly_dead_agents = set()
            
            for killer, victim_name in kills:
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
            
            valid_reports = [r for r in reports if r[0].name not in newly_dead_agents]
            if valid_reports:
                reporter, body = valid_reports[0]
                self.state.report_body(reporter.name, body)
                meeting_triggered = True
                event_occurred_in_round = True
            
            if not meeting_triggered:
                valid_buttons = [b for b in buttons if b.name not in newly_dead_agents]
                if valid_buttons:
                    self.state.call_emergency_meeting(valid_buttons[0].name)
                    meeting_triggered = True
                    event_occurred_in_round = True

            if meeting_triggered:
                self._reset_action_counts()
                return True

            # --- 4. EXECUTE MOVES (Lowest Priority) ---
            for mover, room in moves:
                if mover.name not in newly_dead_agents:
                    if room in ROOMS:
                        self.state.update_location(mover.name, room)

            self.state.save_json()
            
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
        
        active_agents = [a for a in self.agents if self.state.world_data["agents"][a.name]["status"] == "active"]
        
        caller_name = self.state.world_data["global"]["meeting_caller"]
        discussion_order = []
        
        caller_obj = next((a for a in active_agents if a.name == caller_name), None)
        if caller_obj:
            discussion_order.append(caller_obj)
            
        for agent in active_agents:
            if agent.name != caller_name:
                discussion_order.append(agent)

        # 3. Conversation (2 Rounds)
        round_statements = []
        statement_counts = {a.name: 0 for a in active_agents}
        for discussion_round in range(2):            
            for agent in discussion_order:
                view = self.state.get_agent_view(agent.name, round_num, log_to_file=False) 
                msg = agent.participate_in_discussion("", view, round_num)
                
                clean_msg = msg.replace("\n", " ").replace("\r", "").strip()
                clean_msg = re.sub(r"^(\*\*)?Agent_\d+:?(\*\*)?:?\s*", "", clean_msg, flags=re.IGNORECASE)
                clean_msg = clean_msg.strip('"').strip("'")
                
                statement_counts[agent.name] += 1
                is_reporter = 1 if (agent.name == caller_name and statement_counts[agent.name] == 1) else 0

                round_statements.append({
                    'Agent': agent.name,
                    'Text': clean_msg,
                    'Reported': is_reporter,
                    'S_Num': min(statement_counts[agent.name], 2)
                })

                formatted_msg = f"{agent.name}: {clean_msg}"
                self.logger.write_log("discussion", None, formatted_msg)
                
               
                self.logger.log_discussion_chat(
                    discussion_num=round_num,
                    reason=reason,
                    agent_name=agent.name,
                    model_name=agent.model_name,
                    role=agent.role,
                    message=clean_msg
                )
                
                self.state.record_chat(agent.name, clean_msg)
                self.state.save_json()

        # After Discussion, Use Classifier to see probabilities and store results
        suspicion_scores = self.observer.analyze_round(round_statements)
        if suspicion_scores:
            self.state.update_suspicion_scores(suspicion_scores)

        self.state.update_phase("VOTING") 
        votes = {}
        for agent in active_agents:
            view = self.state.get_agent_view(agent.name, round_num, log_to_file=False)
            candidates = [a.name for a in active_agents if a.name != agent.name] + ["SKIP"]
            vote = agent.vote(view, candidates, round_num)
            votes[agent.name] = vote
            self.state.record_vote(agent.name, vote, round_num)
            self.state.save_json()
            voter_stats = self.state.world_data["agents"][agent.name]["stats"]
            voter_role = self.state.world_data["agents"][agent.name]["role"]

            if vote == "SKIP":
                voter_stats["skipped_votes"] += 1
            elif vote in self.state.world_data["agents"]:
                self.state.world_data["agents"][vote]["stats"]["votes_received"] += 1
                target_role = self.state.world_data["agents"][vote]["role"]
                is_correct = False
                if voter_role == "honest" and target_role == "byzantine":
                    is_correct = True
                elif voter_role == "byzantine" and target_role == "honest":
                    is_correct = True
                if is_correct:
                    voter_stats["correct_votes"] += 1
                else:
                    voter_stats["incorrect_votes"] += 1

        tally = {}
        for v in votes.values(): tally[v] = tally.get(v, 0) + 1
        
        self.logger.write_log("results", None, f"Round {round_num} Votes Received: {tally}")
        
        if tally:
            sorted_votes = sorted(tally.items(), key=lambda x: x[1], reverse=True)
            winner, score = sorted_votes[0]
            is_tie = False
            if len(sorted_votes) > 1 and sorted_votes[1][1] == score:
                is_tie = True
            
            if is_tie:
                self.logger.write_log("discussion", None, "** No one was ejected (Tie) **")
                self.state.add_ui_event("⚖️ Tie Vote. No one ejected.", "info")
                
            elif winner == "SKIP":
                self.logger.write_log("discussion", None, "** No one was ejected (Skipped) **")
                self.state.add_ui_event("⏩ Vote Skipped. No one ejected.", "info")
                
            else:
                self.state.eject_agent(winner)
                self.logger.write_log("discussion", None, f"** {winner} was EJECTED **")

                
        else:
            self.logger.write_log("discussion", None, "** No votes cast **")
        
        self.state.world_data["global"]["body_reported"] = False
        self.state.world_data["global"]["meeting_called"] = False

        self.state.update_phase("MOVEMENT")
        self.state.save_json()
        
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
        from core.llm import ModelManager

        winning_team_role = "honest" if "Honest" in result else "byzantine"
        token_usage = ModelManager.get_instance().get_token_usage()

        for agent_name, data in self.state.world_data["agents"].items():
            stats = data["stats"]

            if data["role"] == winning_team_role:
                stats["won_game"] = 1
            else:
                stats["won_game"] = 0

            # Add final classifier scores to stats (if any exist)
            if self.state.suspicion_scores and agent_name in self.state.suspicion_scores:
                agent_scores = self.state.suspicion_scores[agent_name]
                stats["sgd_score"] = agent_scores.get("SGD", None)
                stats["svm_score"] = agent_scores.get("SVM", None)
                stats["lr_score"] = agent_scores.get("LogisticRegression", None)
            else:
                stats["sgd_score"] = None
                stats["svm_score"] = None
                stats["lr_score"] = None

            # Add API token usage per agent's model
            model_tokens = token_usage.get(stats.get("model_name", ""), {})
            stats["api_input_tokens"] = model_tokens.get("input_tokens", 0)
            stats["api_output_tokens"] = model_tokens.get("output_tokens", 0)

        self.state.update_phase("GAME OVER")
        self.state.add_ui_event(f"{result.upper()}", "info")
        self.state.save_json()

        self.logger.export_stats(self.state.world_data["agents"])