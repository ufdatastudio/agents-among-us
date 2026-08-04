# agents/honest_agent.py
import os
import re
from agents.base_agent import BaseAgent
from agents.prompts import (
    HONEST_DEFAULT_MOVE_INSTRUCTIONS,
    HONEST_MOVE_INSTRUCTIONS_WITH_THOUGHTS,
    HONEST_DEFAULT_VOTE_INSTRUCTIONS,
    HONEST_DISCUSSION_INSTRUCTIONS_PREFIX,
    HONEST_DISCUSSION_INSTRUCTIONS_SUFFIX,
    HONEST_VOTE_INSTRUCTIONS_SKIP_DISCUSSION,
    honest_system_prompt,
    honest_system_prompt_skip_discussion,
)
from agents.thought_capture import capture_enabled, generate_with_optional_thoughts
from config.settings import ROOMS, MAX_MOVEMENT_PHASES

class HonestAgent(BaseAgent):
    def __init__(self, name, color, model_name, max_moves=None, max_discussion_messages=2, is_hybrid=False, 
                 skip_discussion=False, context_window=0):
        super().__init__(name, color, "honest", model_name)
        self.max_moves = max_moves if max_moves is not None else MAX_MOVEMENT_PHASES
        self.max_discussion_messages = max_discussion_messages
        self.is_hybrid = is_hybrid
        self.skip_discussion = skip_discussion
        self.context_window = context_window



    def _substitute_placeholders(self, template, extra_mapping=None):
        """
        Lightweight placeholder substitution for user-provided prompt overrides.
        Supported keys (in curly braces) inside the template:
          {self.name}  – this agent's name
          {role}       – 'honest'
          {round_num}  – current round number
          {max_moves}  – movement ticks per round
          {max_discussion_messages}  – messages per discussion
        """
        if not template:
            return template
        mapping = {
            "self.name": self.name,
            "role": self.role,
            "max_moves": str(self.max_moves),
        }
        if extra_mapping:
            mapping.update(extra_mapping)
        out = template
        for key, val in mapping.items():
            placeholder = "{" + key + "}"
            out = out.replace(placeholder, str(val))
        return out

    def _read_file(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _get_current_round_log(self, full_log, round_num):
        if not full_log: return ""
        round_num = str(round_num)
        pattern = f"(?:Round {round_num}/|=== Round {round_num} ===)"
        match = re.search(pattern, full_log)
        if match: return full_log[match.start():]
        else: return full_log

    def think_and_act(self, world_view, round_num):
        # 1. READ LOGS
        full_action_log = self._read_file(world_view["log_path"])
        results_log = self._read_file(world_view["results_log_path"])

        # 2. FILTER LOG
        current_round_log = self._get_current_round_log(full_action_log, round_num)

        # 3. Setup Context
        loc = world_view["self"]["location"]
        adj = ROOMS[loc]
        bodies = world_view["surroundings"][loc]["bodies"]
        button_used = world_view["self"].get("button_used", False)

        move_options = []
        cap_loc = loc.capitalize()
        move_options.append(f"{cap_loc}")
        
        special_actions = []
        options_str = ""
        if bodies:
            special_actions.append("REPORT")
            return "report", bodies[0], "REPORT"

        if loc == "Cafeteria" and not button_used: 
            special_actions.append("BUTTON")

        if special_actions:
            options_str += "Actions:\n" + "\n".join(f"- {a}" for a in special_actions) + "\n"
        for room in adj:
             capitalize_room = room.capitalize()
             move_options.append(f"{capitalize_room}")
        move_options_str = "\n".join(move_options)
        options_str += "Available movement actions:\n" + move_options_str + "\n"

        overrides = getattr(self, "prompt_overrides", {}) or {}
        move_override = overrides.get("movement")
        if move_override:
            move_body = self._substitute_placeholders(
                move_override,
                {"round_num": str(round_num)},
            )
        else:
            move_body = (
                HONEST_MOVE_INSTRUCTIONS_WITH_THOUGHTS
                if capture_enabled(self, world_view)
                else HONEST_DEFAULT_MOVE_INSTRUCTIONS
            )

        prompt = f"""
{results_log}
=== YOUR CURRENT ROUND ACTION LOG ===
{current_round_log}
================================

Goal: Maximize information gain. You need to witness other agents to verify their locations and identify byzantine agents. You are in a movement phase.
Options:
{options_str}

{move_body}
"""
        response = generate_with_optional_thoughts(
            self,
            self._system_prompt(),
            prompt,
            temperature=0.1,
            phase="MOVEMENT",
            round_num=round_num,
            tick=world_view.get("tick", 0),
            world_view=world_view,
        )
        clean_resp = response.strip().upper()
        
        if "REPORT" in clean_resp and bodies:
            return "report", bodies[0], response
        if "BUTTON" in clean_resp and loc == "Cafeteria" and not button_used:
            return "button", "meeting", response
        for room in adj:
            if room.upper() in clean_resp:
                return "move", room, response
        
        return "move", loc, response

    def participate_in_discussion(self, conversation_history, world_view, round_num):
        full_action_log = self._read_file(world_view["log_path"])
        recent_action_log = self._get_current_round_log(full_action_log, round_num)
        discussion_log = self._read_file(world_view["discussion_log_path"])
        recent_discussion = self._get_current_round_log(discussion_log, round_num)

        n = self.max_discussion_messages
        if n < 3:
            if n == 1:
                chances_line = "6. You get 1 chance to speak. Give your initial thought."
            else:
                chances_line = "6. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak."
        else:
            chances_line = f"6. You get {n} chances to speak: your first message is your opening statement, the middle messages are for interactive discussion (follow-ups, questions, answers), and your last message is your final stance."

        default_discussion_instructions = (
            HONEST_DISCUSSION_INSTRUCTIONS_PREFIX
            + chances_line
            + HONEST_DISCUSSION_INSTRUCTIONS_SUFFIX
        )
        overrides = getattr(self, "prompt_overrides", {}) or {}
        discussion_override = overrides.get("discussion")
        extra = {"round_num": str(round_num), "max_discussion_messages": str(self.max_discussion_messages)}
        if discussion_override:
            discussion_body = self._substitute_placeholders(discussion_override, extra)
        else:
            discussion_body = self._substitute_placeholders(default_discussion_instructions, extra)

        prompt = f"""
You are in a discussion phase.
=== Your personal memory log of what you saw in the previous round ===
{recent_action_log}
==================================

== Past rounds results ===
{self._read_file(world_view["results_log_path"])}

=== What has been said in the ongoing discussion ===
{recent_discussion}
======================

{discussion_body}
"""
        return self.llm.generate(self.model_name, self._system_prompt(), prompt, temperature=1.0)

    def vote(self, world_view, candidates, round_num, pruner=None):
        discussion_log = self._read_file(world_view["discussion_log_path"])
        round_num = int(round_num)
        recent_discussion = self._get_current_round_log(discussion_log, round_num - self.context_window) # can adjust as needed

        skip_discussion = world_view.get("skip_discussion", False)
        if skip_discussion:
            action_log = self._read_file(world_view["log_path"])
            recent_action_log = self._get_current_round_log(action_log, round_num - 1)
            recent_discussion = f"=== YOUR RECENT ACTION LOG ===\n{recent_action_log}"
        
        if self.is_hybrid and pruner is not None:
            recent_discussion, suspicion_state, surviving_agents = pruner.prune_live_log(recent_discussion)

            if suspicion_state:
                tracked_cands = [c for c in candidates if c in surviving_agents]
                candidates = tracked_cands if tracked_cands else candidates
                   
        results_log = self._read_file(world_view["results_log_path"])

        overrides = getattr(self, "prompt_overrides", {}) or {}
        vote_override = overrides.get("voting")
        if vote_override:
            vote_body = self._substitute_placeholders(
                vote_override,
                {"round_num": str(round_num), "candidates": candidates},
            )
        else:
            vote_tmpl = (
                HONEST_VOTE_INSTRUCTIONS_SKIP_DISCUSSION
                if skip_discussion
                else HONEST_DEFAULT_VOTE_INSTRUCTIONS
            )
            vote_body = self._substitute_placeholders(
                vote_tmpl,
                {"round_num": str(round_num), "candidates": candidates},
            )

        prompt = f"""
{results_log}
=====================
{recent_discussion}
=====================
You are in a voting phase.
Candidates: {candidates}.

Note: Be aware of total player count to ensure Byzantines do not equal or outnumber honest agents. Skipping your vote is an option. Your vote is ANONYMOUS. Only total counts are seen.

{vote_body}
"""
        response = self.llm.generate(self.model_name, self._system_prompt(), prompt)
        clean_resp = response.strip()
        
        for cand in candidates:
            if cand in clean_resp:
                return cand
                
        return "SKIP"

    def _system_prompt(self):
        overrides = getattr(self, "prompt_overrides", {}) or {}
        custom = overrides.get("system")
        if custom:
            return self._substitute_placeholders(
                custom,
                {
                    "round_num": "",
                },
            )

        if self.skip_discussion:
            return honest_system_prompt_skip_discussion(self.name)
        return honest_system_prompt(self.name)
