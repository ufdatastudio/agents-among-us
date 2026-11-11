import random
import re
import time
import json
from dataclasses import replace
from pathlib import Path

from loguru import logger as log

from config.settings import MAX_MOVEMENT_PHASES, ROOMS, NUM_BYZ, NUM_HONEST, NUM_ROUNDS as DEFAULT_NUM_ROUNDS
from agents.honest_agent import HonestAgent
from agents.byzantine_agent import ByzantineAgent
from config.eligible_models import check_eligible
from core.adapters.game_agent import GameAgentAdapter
from core.situation_room import Meeting, RoomConfig
from core.state import GameState
from core.logger import LogManager
from core.timed_discussion import collect_votes, run_timed_discussion
from core.transport import InProcessTransport
from core.voting import SKIP, ask_until_valid
import os
import joblib
import pandas as pd

from core.sklearn_compat import register_legacy_loss_module
from core.stopwords import ENGLISH_STOP_WORDS
from results.context_pruner import ContextPruner
from core.thought_flags import thought_capture_flags_from_composition

class Observer:
    def __init__(self, model_dir="results/classifiers/models/"):
        register_legacy_loss_module()
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

DISCUSSION_MODES = ("turns", "timed")


class MeetingFailed(RuntimeError):
    """A meeting never ended with a vote from every agent, even after it was redone."""


class GameEngine:
    def __init__(self, game_id, num_agents=NUM_BYZ + NUM_HONEST, num_rounds=DEFAULT_NUM_ROUNDS, num_ticks=None, num_discussion_messages=2,skip_discussion=False,
                 discussion_mode="turns", room_config=None, room_workers=None, seed=None,
                 allow_skip=False, vote_retries=2, max_meeting_attempts=3):
        """Creates the engine.

        Args:
            discussion_mode: `turns` keeps the turn-taking discussion. `timed` runs the situation
                room, where every agent speaks during the same timed window and votes at once.
            room_config: RoomConfig for the timed discussion, or None for the defaults.
            room_workers: Agents that may generate at the same time in the timed discussion, or None
                for all of them. Use 1 when the model runs inside this process.
            seed: Seed of the scenario. It sets the start rooms, so games with one seed begin alike.
            allow_skip: Whether a model agent may skip its vote. By default every agent has to vote.
            vote_retries: Asks to add when a model agent's reply to the vote does not name a candidate.
            max_meeting_attempts: Times a meeting may be held. When an agent still has no valid vote after
                its retries, the meeting is rewound and held again, and the game stops with MeetingFailed
                when the last attempt also lacks a vote.
        """
        if discussion_mode not in DISCUSSION_MODES:
            raise ValueError(f"discussion_mode must be one of {DISCUSSION_MODES}, got {discussion_mode!r}")
        if max_meeting_attempts < 1 or vote_retries < 0:
            raise ValueError("max_meeting_attempts must be at least 1 and vote_retries must not be negative")
        self.game_id = game_id
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.num_ticks = num_ticks if num_ticks is not None else MAX_MOVEMENT_PHASES
        self.num_discussion_messages = num_discussion_messages
        self.skip_discussion = skip_discussion
        self.discussion_mode = discussion_mode
        self.seed = seed
        self.allow_skip = allow_skip
        self.vote_retries = vote_retries
        self.max_meeting_attempts = max_meeting_attempts
        self.room_config = room_config or RoomConfig()
        self.room_transport = InProcessTransport(room_workers)
        self.agents = []
        self.state = None
        self.logger = None
        self.observer = Observer()
        self.pruner = ContextPruner()

        # double check this, add n = 5
        importance_thresholds = {10: 0.2210, 9: 0.2516, 8: 0.2600, 7: 0.2625, 6: 0.2720, 5: 0.2828, 4:0.2718, 'fallback': 0.2661}
        model_file = "results/classifiers/models/mlp_net.joblib"
        try:
            self.pruner.load_live_model(model_file, importance_thresholds)
        except Exception as e:
            # NumPy/sklearn version skew can make the pickled hybrid pruner unloadable.
            # Non-hybrid games should still run; hybrid vote pruning is disabled until fixed.
            print(
                f"[Warning] Context pruner model failed to load ({e}). "
                "Hybrid context pruning disabled for this run."
            )

        
        # ML Classifier config (will be set during setup)
        self.enabled_classifiers = {}

        # Thought-capture flags (overridden from composition in setup)
        self.capture_thoughts = True
        self.require_think_tags = False

    def setup(self, composition):
        scen_name = composition.get("name", "Unknown_Scenario")
        human_experiment = bool(composition.get("human_experiment", False))
        human_agent = composition.get("human_agent", "Agent_0" if human_experiment else None)
        self.capture_thoughts, self.require_think_tags = (
            thought_capture_flags_from_composition(composition)
        )
        
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
                    agent_obj = ByzantineAgent(
                        agent_name,
                        color,
                        teammates,
                        model,
                        max_moves=self.num_ticks,
                        max_discussion_messages=self.num_discussion_messages,
                        skip_discussion=self.skip_discussion,
                    )
                    setattr(agent_obj, "is_human", is_human)
                    if is_human:
                        agent_obj.model_name = "Human"
                    self.agents.append(agent_obj)
                else:
                    agent_obj = HonestAgent(
                        agent_name,
                        color,
                        model,
                        max_moves=self.num_ticks,
                        max_discussion_messages=self.num_discussion_messages,
                        is_hybrid=is_hybrid,
                        skip_discussion=self.skip_discussion,
                    )
                    setattr(agent_obj, "is_human", is_human)
                    if is_human:
                        agent_obj.model_name = "Human"
                    self.agents.append(agent_obj)
            
            print(f"Created {len(self.agents)} agents with EXACT configuration from frontend")
            
        else:
            honest_models = composition["honest_model"]
            byz_models = composition["byzantine_model"]
            n_honest = composition["honest_count"]
            n_byz = composition["byzantine_count"]
            hybrid_count = composition.get("hybrid_count", 0)
            hybrid_flags = composition.get("honest_hybrid", [])
            context_windows = composition.get("context_windows", [])
            byz_context_windows = composition.get("byz_context_windows", context_windows)
            
            colors = ["🔴", "🟠", "🟡", "🟩", "🟢", "🔷", "🔵", "🟣", "🟤", "💗", "⚪", "⚫"]
            
            # Create Byzantine Agents
            byz_names = [f"Agent_{i}" for i in range(n_byz)]
            for i, name in enumerate(byz_names):
                assigned_model = byz_models[i % len(byz_models)]
                teammates = [b for b in byz_names if b != name]
                cw = byz_context_windows[i % len(byz_context_windows)] if byz_context_windows else 0
                agent_obj = ByzantineAgent(
                    name,
                    colors[i],
                    teammates,
                    assigned_model,
                    max_moves=self.num_ticks,
                    max_discussion_messages=self.num_discussion_messages,
                    skip_discussion=self.skip_discussion,
                    context_window=cw,
                )
                setattr(agent_obj, "is_human", bool(human_experiment and name == human_agent))
                if getattr(agent_obj, "is_human", False):
                    agent_obj.model_name = "Human"
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

                cw = context_windows[i % len(context_windows)] if context_windows else 0
                agent_obj = HonestAgent(
                    name,
                    color,
                    assigned_model,
                    max_moves=self.num_ticks,
                    max_discussion_messages=self.num_discussion_messages,
                    is_hybrid=is_hybrid,
                    skip_discussion=self.skip_discussion,
                    context_window=cw,
                )
                setattr(agent_obj, "is_human", bool(human_experiment and name == human_agent))
                if getattr(agent_obj, "is_human", False):
                    agent_obj.model_name = "Human"
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

        # A local vLLM model runs on HiPerGator, where only eligible models may run.
        for agent in self.agents:
            model_name = str(agent.model_name)
            if model_name.startswith("vllm:"):
                check_eligible(model_name.partition(":")[2])

        self.logger = LogManager(self.game_id, self.agents, scen_name)
        self.state = GameState(self.agents, self.logger, seed=self.seed)
        self.state.world_data["global"]["human_experiment"] = human_experiment
        self.state.world_data["global"]["human_agent"] = human_agent
        self.state.world_data["global"]["capture_thoughts"] = self.capture_thoughts
        self.state.world_data["global"]["require_think_tags"] = self.require_think_tags
        for agent in self.agents:
            agent.logger = self.logger
            agent.game_state = self.state
            agent.capture_thoughts = self.capture_thoughts
            agent.require_think_tags = self.require_think_tags
        print(
            f"Thought capture: {'ON' if self.capture_thoughts else 'OFF'} "
            f"(require_think_tags={self.require_think_tags})"
        )
        
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
                view["tick"] = phase_tick
                view["phase"] = "MOVEMENT"
                view["capture_thoughts"] = self.capture_thoughts
                view["require_think_tags"] = self.require_think_tags
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

    def _await_human_movement_action(self, agent, view, round_num, tick, timeout_s=None):
        """
        Waits for a human-submitted movement action (via Flask /api/human/action),
        persisted to logs/human_inputs/<game_id>.jsonl.
        Blocks until a valid human action is submitted.
        """
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

        try:
            while True:
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

    def _await_human_discussion_message(self, agent, round_num, turn_idx, timeout_s=None):
        """
        Wait for a human chat submission during discussion.
        Blocks until a human chat submission is received.
        """
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
            while True:
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
        finally:
            g["awaiting_human_action"] = False
            g["awaiting_human_agent"] = None
            g["awaiting_human_round"] = 0
            g["awaiting_human_tick"] = 0
            g["awaiting_human_options"] = {}
            self.state.save_json()

    def _await_human_vote(self, agent, round_num, vote_turn_idx, candidates, timeout_s=None):
        """
        Wait for a human vote submission during voting.
        Blocks until a valid human vote is submitted.
        """
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
            while True:
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
        """Holds the meeting of a round, with a vote from every agent, and ejects the agent with most votes.

        A meeting in which a model agent has no valid vote is rewound and held again, up to
        `max_meeting_attempts` times. The rewound discussion goes to voided_meetings.jsonl.

        Raises:
            MeetingFailed: If the last attempt still lacks a vote.
        """
        active_agents = [a for a in self.agents if self.state.world_data["agents"][a.name]["status"] == "active"]
        votes = {}
        for attempt in range(1, self.max_meeting_attempts + 1):
            checkpoint = self._checkpoint_meeting()
            votes = self._hold_meeting(round_num, active_agents, attempt)
            missing = self._missing_voters(active_agents, votes)
            if not missing:
                break
            if attempt == self.max_meeting_attempts:
                self._rewind_meeting(checkpoint, round_num, attempt, missing, votes)
                raise MeetingFailed(f"round {round_num}: {missing} had no valid vote in {attempt} attempts")
            self._rewind_meeting(checkpoint, round_num, attempt, missing, votes)
        for agent in active_agents:
            self._record_vote(agent, votes[agent.name], round_num)
        self._tally_and_eject(round_num, votes)

    def _hold_meeting(self, round_num, active_agents, attempt):
        """Runs the discussion of a meeting and collects every agent's vote, without recording the votes.

        Returns:
            The vote of each agent, which is SKIP where the agent gave no valid vote or skipped.
        """
        self.logger.write_log("discussion", None, f"\n=== Round {round_num} ===")
        reason = self.state.world_data["global"]["meeting_reason_log"]
        if reason:
            self.logger.write_log("discussion", None, reason)

        caller_name = self.state.world_data["global"]["meeting_caller"]
        discussion_order = []
        
        caller_obj = next((a for a in active_agents if a.name == caller_name), None)
        if caller_obj:
            discussion_order.append(caller_obj)
            
        for agent in active_agents:
            if agent.name != caller_name:
                discussion_order.append(agent)

        round_statements = []
        room = None
        adapters = []
        use_timed = not self.skip_discussion and self._use_timed_discussion(active_agents)
        if not self.skip_discussion and not use_timed:
            statement_counts = {a.name: 0 for a in active_agents}
            discussion_turn_idx = 0
            for discussion_round in range(self.num_discussion_messages):
                for agent in discussion_order:
                    discussion_turn_idx += 1
                    view = self.state.get_agent_view(agent.name, round_num, log_to_file=False)
                    view["tick"] = discussion_turn_idx
                    view["phase"] = "DISCUSSION"
                    view["capture_thoughts"] = self.capture_thoughts
                    view["require_think_tags"] = self.require_think_tags
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
                    clean_msg = re.sub(
                        rf"^(\*\*)?{re.escape(agent.name)}:?(\*\*)?:?\s*",
                        "",
                        clean_msg,
                        flags=re.IGNORECASE,
                    )
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
        elif self.skip_discussion:
            self.logger.write_log("discussion", None, "** DISCUSSION PHASE SKIPPED **")
            self.state.add_ui_event("Discussion Skipped", "info")
        else:
            adapters, room = self._run_timed_meeting(round_num, active_agents, caller_name, reason, attempt)
            round_statements = room.statements(caller_name)
            # The classifiers were trained on at most num_discussion_messages statements per agent.
            for row in round_statements:
                row["S_Num"] = min(row["S_Num"], self.num_discussion_messages)

        # After Discussion, Use Classifier to see probabilities and store results
        suspicion_scores = self.observer.analyze_round(round_statements)
        if suspicion_scores:
            self.state.update_suspicion_scores(suspicion_scores)

        self.state.update_phase("VOTING")
        votes = {}
        if room is not None:
            # Every agent votes at once after seeing the room as it ended.
            votes = collect_votes(adapters, room, self.room_transport, allow_skip=self.allow_skip)
        vote_turn_idx = 0
        for agent in ([] if room is not None else active_agents):
            vote_turn_idx += 1
            view = self.state.get_agent_view(agent.name, round_num, log_to_file=False)
            view["skip_discussion"] = self.skip_discussion
            view["tick"] = vote_turn_idx
            view["phase"] = "VOTING"
            view["capture_thoughts"] = self.capture_thoughts
            view["require_think_tags"] = self.require_think_tags
            is_human = getattr(agent, "is_human", False)
            candidates = [a.name for a in active_agents if a.name != agent.name]
            if self.allow_skip or is_human:
                candidates.append(SKIP)
            if is_human:
                vote = self._await_human_vote(
                    agent,
                    round_num=round_num,
                    vote_turn_idx=vote_turn_idx,
                    candidates=candidates,
                    timeout_s=30,
                )
            else:
                vote = self._ask_vote(agent, view, candidates, round_num)
            votes[agent.name] = vote
        return votes

    def _ask_vote(self, agent, view, candidates, round_num):
        """Asks a model agent for its vote, again when the reply names no candidate.

        Returns:
            The candidate voted for, or SKIP when no ask gave one.
        """
        def ask(temperature):
            asked = {**view, "vote_temperature": temperature}
            if getattr(agent, "is_hybrid", False):
                return agent.vote(asked, candidates, round_num, pruner=self.pruner)
            return agent.vote(asked, candidates, round_num)

        target = ask_until_valid(ask, candidates, self.vote_retries, agent.name)
        return SKIP if target is None else target

    def _missing_voters(self, active_agents, votes):
        """Lists the model agents that have no valid vote when skipping is not allowed."""
        if self.allow_skip:
            return []
        return [a.name for a in active_agents if not getattr(a, "is_human", False) and votes.get(a.name, SKIP) == SKIP]

    def _checkpoint_meeting(self):
        """Notes how far the logs a meeting writes to have grown, so the meeting can be rewound."""
        return {
            "discussion": os.path.getsize(self.logger.paths["discussion"]),
            "discussion_chat": os.path.getsize(self.logger.paths["discussion_chat"]),
            "ui_events": len(self.state.world_data["global"]["ui_event_log"]),
        }

    def _rewind_meeting(self, checkpoint, round_num, attempt, missing, votes):
        """Takes back a meeting that lacks a vote.

        The discussion and chat it wrote are cut from the logs the agents read, and kept in
        voided_meetings.jsonl with the votes it did collect. The room's record is renamed.
        """
        voided = {"round": round_num, "attempt": attempt, "missing_voters": missing, "votes": votes}
        voided["vote_reads"] = {a.name: list(a.vote_reads) for a in self.agents if getattr(a, "vote_reads", None)}
        for a in self.agents:
            if hasattr(a, "vote_reads"):
                a.vote_reads.clear()
        for key in ("discussion", "discussion_chat"):
            with open(self.logger.paths[key], "r+b") as handle:
                handle.seek(checkpoint[key])
                voided[key] = handle.read().decode("utf-8", errors="replace")
                handle.truncate(checkpoint[key])
        base = Path(self.logger.base_dir)
        with (base / "voided_meetings.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(voided) + "\n")
        room_file = base / "room" / f"room_round{round_num}.jsonl"
        if room_file.exists():
            room_file.rename(room_file.with_name(f"room_round{round_num}_voided{attempt}.jsonl"))
        del self.state.world_data["global"]["ui_event_log"][checkpoint["ui_events"]:]
        self.state.add_ui_event(f"Meeting {round_num} is held again, because {missing} had no valid vote", "info")
        log.warning("Meeting of round {} rewound after attempt {}: {} had no valid vote", round_num, attempt, missing)

    def _tally_and_eject(self, round_num, votes):
        """Counts the votes of a meeting and ejects the agent with the most, unless the vote is tied."""
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
        
    def _record_vote(self, agent, vote, round_num):
        """Stores one vote and updates the voting statistics of the voter and the target."""
        self._log_vote_readings(agent, vote, round_num)
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

    def _log_vote_readings(self, agent, vote, round_num):
        """Appends the readings of an agent's vote replies to votes.jsonl, one line per vote.

        Each reading says how the reply was read (`exact`, `marker`, `single`, `llm`, `ambiguous` or
        `none`), so a run shows how often replies needed the model. A human seat has no readings.
        """
        reads = list(getattr(agent, "vote_reads", []))
        if hasattr(agent, "vote_reads"):
            agent.vote_reads.clear()
        record = {"round": round_num, "voter": agent.name, "role": agent.role, "vote": vote, "reads": reads}
        with (Path(self.logger.base_dir) / "votes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _use_timed_discussion(self, active_agents):
        """Tells whether this meeting runs in the situation room.

        A human seat is not supported in the room yet, so a meeting with a human falls back to turns.
        """
        if self.discussion_mode != "timed":
            return False
        if any(getattr(a, "is_human", False) for a in active_agents):
            log.warning("A human player is in the game, so this meeting uses turn-taking, not the timed room")
            return False
        return True

    def _run_timed_meeting(self, round_num, active_agents, caller_name, reason, attempt=1):
        """Runs the discussion of one meeting in the situation room.

        Every utterance is written to the discussion log and the chat file in order of start time,
        as turn-taking writes them. The room's own record goes to room/room_round<N>.jsonl in the game's log folder.

        Returns:
            The adapters, which the vote reuses, and the closed room.
        """
        names = [a.name for a in active_agents]
        room_dir = Path(self.logger.get_discussion_log_path("honest")).parent / "room"
        # A meeting that is held again gets other onsets, because the same ones would repeat its timing.
        config = replace(self.room_config, seed=self.room_config.seed + round_num + 1000 * (attempt - 1))

        def view_for(name, rnd):
            view = self.state.get_agent_view(name, rnd, log_to_file=False)
            view["capture_thoughts"] = self.capture_thoughts
            view["require_think_tags"] = self.require_think_tags
            return view

        adapters = [GameAgentAdapter(a, names, config, view_for, room_dir, self.vote_retries) for a in active_agents]
        room = run_timed_discussion(adapters, Meeting(round_num, reason, caller_name), config, self.room_transport)
        by_name = {a.name: a for a in active_agents}
        for utterance in room.utterances():
            if not utterance.words:
                continue
            agent = by_name[utterance.author]
            self.logger.write_log("discussion", None, f"{agent.name}: {utterance.text}")
            self.logger.log_discussion_chat(
                discussion_num=round_num,
                reason=reason,
                agent_name=agent.name,
                model_name=agent.model_name,
                role=agent.role,
                message=utterance.text,
            )
            self.state.record_chat(agent.name, utterance.text)
        room.write_jsonl(room_dir / f"room_round{round_num}.jsonl")
        self.state.save_json()
        log.info(
            "Room round {}: {} ticks, {} utterances, closed at {:.0f} s{}",
            round_num,
            room.ticks,
            len(room.utterances()),
            room.closed_at or 0.0,
            " (early)" if room.closed_early else "",
        )
        return adapters, room

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

            if data.get("is_human"):
                stats["model_name"] = "Human"

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
            # Totals of the agent's model over the whole game. Reasoning models need these to show their cost.
            stats["api_calls"] = model_tokens.get("calls", 0)
            stats["api_reasoning_tokens"] = model_tokens.get("reasoning_tokens", 0)
            stats["api_reasoning_estimated_calls"] = model_tokens.get("reasoning_estimated_calls", 0)
            stats["api_truncated_calls"] = model_tokens.get("truncated_calls", 0)
            stats["api_empty_replies"] = model_tokens.get("empty_replies", 0)
            stats["api_errors"] = model_tokens.get("api_errors", 0)
            stats["api_max_output_tokens"] = model_tokens.get("max_output_tokens", 0)

        self.state.update_phase("GAME OVER")
        self.state.add_ui_event(f"{result.upper()}", "info")
        self.state.save_json()

        self.logger.export_stats(self.state.world_data["agents"])