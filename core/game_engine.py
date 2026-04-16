import random
import re
import time
import json
from config.settings import MAX_MOVEMENT_PHASES, ROOMS, NUM_BYZ, NUM_HONEST, NUM_ROUNDS as DEFAULT_NUM_ROUNDS
from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from core.state import GameState
from core.logger import LogManager
import os 
import joblib
import pandas as pd

from core.stopwords import ENGLISH_STOP_WORDS
from results.context_pruner import ContextPruner

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
        self.stop_words = ENGLISH_STOP_WORDS

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

        print("\nOBSERVER: SUSPICION PROBABILITY")

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
            
            print(f"  {agent_name}: LR {lr_p*100:.1f}%, SGD {sgd_p*100:.1f}%, SVM {svm_p*100:.1f}%")
        
        print()
        
        return scores_by_agent  

class GameEngine:
    def __init__(self, game_id, num_agents=NUM_BYZ + NUM_HONEST, num_rounds=DEFAULT_NUM_ROUNDS, num_ticks=None, num_discussion_messages=2):
        self.game_id = game_id
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.num_ticks = num_ticks if num_ticks is not None else MAX_MOVEMENT_PHASES
        self.num_discussion_messages = num_discussion_messages
        self.agents = []
        self.state = None
        self.logger = None
        self.observer = Observer()
        self.pruner = ContextPruner()

        # double check this, add n = 5
        importance_thresholds = {10: 0.2210, 9: 0.2516, 8: 0.2600, 7: 0.2625, 6: 0.2720, 5: 0.2828, 4:0.2718, 'fallback': 0.2661}
        model_file = "results/classifiers/models/mlp_net.joblib"
        self.pruner.load_live_model(model_file, importance_thresholds)

        
        # ML Classifier config (will be set during setup)
        self.enabled_classifiers = {}

    def setup(self, composition):
        scen_name = composition.get("name", "Unknown_Scenario")
        human_experiment = bool(composition.get("human_experiment", False))
        human_agent = composition.get("human_agent", "Agent_0" if human_experiment else None)
        
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
                is_hybrid = agent_config.get('is_hybrid', False)
                is_human = bool(agent_config.get('is_human', False) or (human_experiment and agent_name == human_agent))
                
                if role == 'byzantine':
                    teammates = [name for name in byz_names if name != agent_name]
                    agent_obj = ByzantineAgent(agent_name, color, teammates, model, max_moves=self.num_ticks, max_discussion_messages=self.num_discussion_messages)
                    setattr(agent_obj, "is_human", is_human)
                    self.agents.append(
                        agent_obj
                    )
                else:
                    agent_obj = HonestAgent(agent_name, color, model, max_moves=self.num_ticks, max_discussion_messages=self.num_discussion_messages, is_hybrid=is_hybrid)
                    setattr(agent_obj, "is_human", is_human)
                    self.agents.append(agent_obj)
            
            print(f"Created {len(self.agents)} agents with EXACT configuration from frontend")
            
        else:
            honest_models = composition["honest_model"]
            byz_models = composition["byzantine_model"]
            n_honest = composition["honest_count"]
            n_byz = composition["byzantine_count"]
            hybrid_count = composition.get("hybrid_count", 0)
            hybrid_flags = composition.get("honest_hybrid", [])
            
            colors = ["🔴", "🟠", "🟡", "🟩", "🟢", "🔷", "🔵", "🟣", "🟤", "💗", "⚪", "⚫"]
            
            # Create Byzantine Agents
            byz_names = [f"Agent_{i}" for i in range(n_byz)]
            for i, name in enumerate(byz_names):
                assigned_model = byz_models[i % len(byz_models)]
                teammates = [b for b in byz_names if b != name]
                agent_obj = ByzantineAgent(name, colors[i], teammates, assigned_model, max_moves=self.num_ticks, max_discussion_messages=self.num_discussion_messages)
                setattr(agent_obj, "is_human", bool(human_experiment and name == human_agent))
                self.agents.append(agent_obj)

            start_index = n_byz
            for i in range(n_honest):
                name = f"Agent_{start_index + i}"
                color = colors[(start_index + i) % len(colors)]
                assigned_model = honest_models[i % len(honest_models)]
                if hybrid_flags:
                    # Map the boolean flag directly to the model
                    is_hybrid = hybrid_flags[i % len(hybrid_flags)]
                else:
                    is_hybrid = (i < hybrid_count)

                agent_obj = HonestAgent(name, color, assigned_model, max_moves=self.num_ticks, max_discussion_messages=self.num_discussion_messages, is_hybrid=is_hybrid)
                setattr(agent_obj, "is_human", bool(human_experiment and name == human_agent))
                self.agents.append(agent_obj)

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
                    # fallback to default prompts for that agent
                    pass

        self.logger = LogManager(self.game_id, self.agents, scen_name)
        self.state = GameState(self.agents, self.logger)
        self.state.world_data["global"]["human_experiment"] = human_experiment
        self.state.world_data["global"]["human_agent"] = human_agent
        
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
        for phase_tick in range(1, self.num_ticks + 1):
            print(f"Tick {phase_tick}...")
            active_agents = [a for a in self.agents if self.state.world_data["agents"][a.name]["status"] == "active"]
            
            # --- 1. GATHER DECISIONS ---
            decisions = []
            for agent in active_agents:
                view = self.state.get_agent_view(agent.name, round_num, log_to_file=True)
                if getattr(agent, "is_human", False):
                    decision = self._await_human_movement_action(agent, view, round_num, phase_tick, timeout_s=25)
                else:
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

    def _human_input_file(self):
        return os.path.join("logs", "human_inputs", f"{self.game_id}.jsonl")

    def _read_human_inputs(self):
        fp = self._human_input_file()
        if not os.path.exists(fp):
            return []
        out = []
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
        return out

    def _await_human_movement_action(self, agent, view, round_num, tick, timeout_s=25):
        """
        Waits for a human-submitted movement action (via Flask /api/human/action),
        persisted to logs/human_inputs/<game_id>.jsonl.
        Timeout fallback is to stay in place (move -> current location).
        """
        start = time.time()
        seen = set()
        agent_name = agent.name
        phase = "MOVEMENT"
        current_loc = view["self"]["location"]
        options = self._build_human_movement_options(agent, view)

        g = self.state.world_data["global"]
        g["awaiting_human_action"] = True
        g["awaiting_human_agent"] = agent_name
        g["awaiting_human_round"] = int(round_num)
        g["awaiting_human_tick"] = int(tick)
        g["awaiting_human_options"] = options
        self.state.save_json()

        def fallback(reason):
            return ("move", current_loc, reason)
        try:
            while time.time() - start < timeout_s:
                inputs = self._read_human_inputs()
                for item in inputs:
                    if not isinstance(item, dict):
                        continue
                    if item.get("kind") != "action":
                        continue
                    key = (
                        str(item.get("received_at")),
                        str(item.get("game_id")),
                        str(item.get("agent_name")),
                        str(item.get("phase")),
                        str(item.get("round")),
                        str(item.get("tick")),
                        str(item.get("action")),
                        str(item.get("target")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    if str(item.get("game_id")) != str(self.game_id):
                        continue
                    if str(item.get("agent_name")) != str(agent_name):
                        continue
                    if str(item.get("phase", "")).strip().upper() != phase:
                        continue
                    try:
                        if int(item.get("round")) != int(round_num):
                            continue
                        if int(item.get("tick")) != int(tick):
                            continue
                    except Exception:
                        continue

                    action = str(item.get("action", "")).strip().lower()
                    target = item.get("target")

                    # stay -> map to move to current room
                    if action == "stay":
                        return ("move", current_loc, "HUMAN_STAY")

                    # move: must be in legal targets provided to UI
                    if action == "move":
                        if isinstance(target, str) and target in options.get("move_targets", []):
                            return ("move", target, "HUMAN_MOVE")
                        continue

                    # report: target must be one of the legal report targets
                    if action == "report":
                        if isinstance(target, str) and target in options.get("report_targets", []):
                            return ("report", target, "HUMAN_REPORT")
                        continue

                    # emergency button
                    if action == "button":
                        if options.get("can_button", False):
                            return ("button", "meeting", "HUMAN_BUTTON")
                        continue

                    # kill/tag
                    if action in {"kill", "tag"}:
                        if isinstance(target, str) and target in options.get("kill_targets", []):
                            return ("tag", target, "HUMAN_TAG")
                        continue

                time.sleep(0.2)

            return fallback("HUMAN_TIMEOUT")
        finally:
            g["awaiting_human_action"] = False
            g["awaiting_human_agent"] = None
            g["awaiting_human_round"] = 0
            g["awaiting_human_tick"] = 0
            g["awaiting_human_options"] = {}
            self.state.save_json()

    def _build_human_movement_options(self, agent, view):
        current_loc = view["self"]["location"]
        neighbors = ROOMS.get(current_loc, [])
        move_targets = list(neighbors)
        report_targets = list(view.get("surroundings", {}).get(current_loc, {}).get("bodies", []) or [])
        can_button = bool(current_loc == "Cafeteria" and not view.get("self", {}).get("button_used", False))

        kill_targets = []
        if getattr(agent, "role", "") == "byzantine":
            teammates = set(getattr(agent, "teammates", []) or [])
            for other_name, other_data in self.state.world_data["agents"].items():
                if other_name == agent.name:
                    continue
                if other_name in teammates:
                    continue
                if other_data.get("status") != "active":
                    continue
                if other_data.get("location") != current_loc:
                    continue
                kill_targets.append(other_name)

        actions = ["move", "stay"]
        if report_targets:
            actions.append("report")
        if can_button:
            actions.append("button")
        if kill_targets:
            actions.append("kill")

        return {
            "actions": actions,
            "current_location": current_loc,
            "move_targets": move_targets,
            "report_targets": report_targets,
            "kill_targets": kill_targets,
            "can_button": can_button,
            "is_byzantine": bool(getattr(agent, "role", "") == "byzantine"),
        }

    def _await_human_discussion_message(self, agent, round_num, turn_idx, timeout_s=30):
        """
        Wait for a human chat submission during discussion.
        Timeout fallback is an empty message.
        """
        start = time.time()
        seen = set()
        agent_name = agent.name
        phase = "DISCUSSION"

        g = self.state.world_data["global"]
        g["awaiting_human_action"] = True
        g["awaiting_human_agent"] = agent_name
        g["awaiting_human_round"] = int(round_num)
        g["awaiting_human_tick"] = int(turn_idx)
        g["awaiting_human_options"] = {
            "mode": "discussion",
            "actions": ["say"],
            "max_chars": 400,
        }
        self.state.save_json()

        try:
            while time.time() - start < timeout_s:
                inputs = self._read_human_inputs()
                for item in inputs:
                    if not isinstance(item, dict):
                        continue
                    if item.get("kind") != "chat":
                        continue
                    key = (
                        str(item.get("received_at")),
                        str(item.get("game_id")),
                        str(item.get("agent_name")),
                        str(item.get("phase")),
                        str(item.get("round")),
                        str(item.get("tick")),
                        str(item.get("action")),
                        str(item.get("target")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    if str(item.get("game_id")) != str(self.game_id):
                        continue
                    if str(item.get("agent_name")) != str(agent_name):
                        continue
                    if str(item.get("phase", "")).strip().upper() != phase:
                        continue
                    try:
                        if int(item.get("round")) != int(round_num):
                            continue
                        if int(item.get("tick")) != int(turn_idx):
                            continue
                    except Exception:
                        continue

                    action = str(item.get("action", "")).strip().lower()
                    if action != "say":
                        continue

                    raw_msg = item.get("message", item.get("target", ""))
                    msg = str(raw_msg or "").replace("\n", " ").replace("\r", "").strip()
                    # Keep message bounded for log readability and safety.
                    if len(msg) > 400:
                        msg = msg[:400].rstrip()
                    return msg
                time.sleep(0.2)
            return ""
        finally:
            g["awaiting_human_action"] = False
            g["awaiting_human_agent"] = None
            g["awaiting_human_round"] = 0
            g["awaiting_human_tick"] = 0
            g["awaiting_human_options"] = {}
            self.state.save_json()

    def _await_human_vote(self, agent, round_num, vote_turn_idx, candidates, timeout_s=30):
        """
        Wait for a human vote submission during voting.
        Timeout fallback is SKIP.
        """
        start = time.time()
        seen = set()
        agent_name = agent.name
        phase = "VOTING"
        legal_candidates = list(candidates or [])

        g = self.state.world_data["global"]
        g["awaiting_human_action"] = True
        g["awaiting_human_agent"] = agent_name
        g["awaiting_human_round"] = int(round_num)
        g["awaiting_human_tick"] = int(vote_turn_idx)
        g["awaiting_human_options"] = {
            "mode": "voting",
            "actions": ["vote"],
            "candidates": legal_candidates,
        }
        self.state.save_json()

        try:
            while time.time() - start < timeout_s:
                inputs = self._read_human_inputs()
                for item in inputs:
                    if not isinstance(item, dict):
                        continue
                    if item.get("kind") != "vote":
                        continue
                    key = (
                        str(item.get("received_at")),
                        str(item.get("game_id")),
                        str(item.get("agent_name")),
                        str(item.get("phase")),
                        str(item.get("round")),
                        str(item.get("tick")),
                        str(item.get("action")),
                        str(item.get("target")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)

                    if str(item.get("game_id")) != str(self.game_id):
                        continue
                    if str(item.get("agent_name")) != str(agent_name):
                        continue
                    if str(item.get("phase", "")).strip().upper() != phase:
                        continue
                    try:
                        if int(item.get("round")) != int(round_num):
                            continue
                        if int(item.get("tick")) != int(vote_turn_idx):
                            continue
                    except Exception:
                        continue

                    action = str(item.get("action", "")).strip().lower()
                    target = str(item.get("target", "")).strip()
                    if action != "vote":
                        continue
                    if target in legal_candidates:
                        return target
                time.sleep(0.2)
            return "SKIP"
        finally:
            g["awaiting_human_action"] = False
            g["awaiting_human_agent"] = None
            g["awaiting_human_round"] = 0
            g["awaiting_human_tick"] = 0
            g["awaiting_human_options"] = {}
            self.state.save_json()

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

        round_statements = []
        statement_counts = {a.name: 0 for a in active_agents}
        discussion_turn_idx = 0
        for discussion_round in range(self.num_discussion_messages):            
            for agent in discussion_order:
                discussion_turn_idx += 1
                view = self.state.get_agent_view(agent.name, round_num, log_to_file=False) 
                if getattr(agent, "is_human", False):
                    msg = self._await_human_discussion_message(
                        agent,
                        round_num=round_num,
                        turn_idx=discussion_turn_idx,
                        timeout_s=30,
                    )
                else:
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
                    'S_Num': min(statement_counts[agent.name], self.num_discussion_messages)
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
        vote_turn_idx = 0
        for agent in active_agents:
            vote_turn_idx += 1
            view = self.state.get_agent_view(agent.name, round_num, log_to_file=False)
            candidates = [a.name for a in active_agents if a.name != agent.name] + ["SKIP"]
            if getattr(agent, "is_human", False):
                vote = self._await_human_vote(
                    agent,
                    round_num=round_num,
                    vote_turn_idx=vote_turn_idx,
                    candidates=candidates,
                    timeout_s=30,
                )
            elif getattr(agent, "is_hybrid", False):
                vote = agent.vote(view, candidates, round_num, pruner=self.pruner)
            else:
                vote = agent.vote(view, candidates, round_num)
            # Do not call vote() again here: a duplicate line (merge artifact) used to
            # overwrite the hybrid branch and drop the pruner, breaking hybrid voting.

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