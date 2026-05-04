# agents/byzantine_agent.py
import os
import re
from agents.base_agent import BaseAgent
from agents.prompts import (
    BYZANTINE_DEFAULT_MOVE_INSTRUCTIONS,
    BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS,
    BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX,
    BYZANTINE_DISCUSSION_INSTRUCTIONS_SUFFIX,
    BYZANTINE_VOTE_INSTRUCTIONS_SKIP_DISCUSSION,
    byzantine_system_prompt,
    byzantine_system_prompt_skip_discussion,
)
from config.settings import ROOMS, MAX_MOVEMENT_PHASES

class ByzantineAgent(BaseAgent):
    def __init__(self, name, color, teammates, model_name, max_moves=None, 
                 max_discussion_messages=2, skip_discussion=False, context_window=0):
        super().__init__(name, color, "byzantine", model_name)
        self.teammates = teammates
        self.max_moves = max_moves if max_moves is not None else MAX_MOVEMENT_PHASES
        self.max_discussion_messages = max_discussion_messages
        self.skip_discussion = skip_discussion
        self.context_window = context_window

    def _substitute_placeholders(self, template, extra_mapping=None):
        """
        Lightweight placeholder substitution for user-provided prompt overrides.
        Supported keys (in curly braces) inside the template:
          {self.name}       – this agent's name
          {self.teammates}  – comma-separated teammate names
          {role}            – 'byzantine'
          {round_num}       – current round number
          {max_moves}       – movement ticks per round
          {candidates}      – comma-separated candidate names (when passed in extra_mapping)
          {max_discussion_messages}  – messages per discussion
        """
        if not template:
            return template
        mapping = {
            "self.name": self.name,
            "self.teammates": ", ".join(self.teammates) if self.teammates else "",
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
        full_action_log = self._read_file(world_view["log_path"])
        results_log = self._read_file(world_view["results_log_path"])
        current_round_log = self._get_current_round_log(full_action_log, round_num)
        
        loc = world_view["self"]["location"]
        occupants = world_view["surroundings"][loc]["occupants"]
        bodies = world_view["surroundings"][loc]["bodies"]
        adj = ROOMS[loc]
        button_used = world_view["self"].get("button_used", False)
        last_action = world_view["self"].get("last_action")

        special_actions = []
        if loc == "Cafeteria" and not button_used: 
             special_actions.append("BUTTON")
        if bodies:
             special_actions.append("REPORT")

        victims = [o for o in occupants if o != self.name and o not in self.teammates]
        options_str = ""
        if victims and last_action != "tag":
            for v in victims:
                special_actions.append(f"TAG {v}")
        
        if special_actions:
            options_str += "Actions:\n" + "\n".join(f"{a}" for a in special_actions) + "\n"

        move_options = []
        cap_loc = loc.capitalize()
        move_options.append(f"{cap_loc}")
        for room in adj:
             capitalize_room = room.capitalize()
             move_options.append(f"{capitalize_room}")
        move_options_str = "\n".join(move_options)

        options_str += "Available map locations to move to:\n" + move_options_str + "\n"

        overrides = getattr(self, "prompt_overrides", {}) or {}
        move_override = overrides.get("movement")
        if move_override:
            move_body = self._substitute_placeholders(
                move_override,
                {"round_num": str(round_num)},
            )
        else:
            move_body = BYZANTINE_DEFAULT_MOVE_INSTRUCTIONS

        teammates_str = ", ".join(self.teammates) if self.teammates else ""
        prompt = f"""
{results_log}
=== YOUR CURRENT ROUND ACTION LOG ===
{current_round_log}
================================

Goal: TAG honest agents without being caught. You can only tag agents that are occupants in your current location.
You are in a movement phase.
Teammates: {teammates_str}

Options
{options_str}

{move_body}
"""
        response = self.llm.generate(self.model_name, self._system_prompt(), prompt, temperature=0.1)
        clean_resp = response.strip().upper()
        
        # Check for TAG action first, safeguard to avoid consecutive tags in case hallucination
        if "TAG" in clean_resp and victims and last_action != "tag":
            for v in victims:
                if v.upper() in clean_resp:
                    return "tag", v, response
        
        if "REPORT" in clean_resp and bodies:
            return "report", bodies[0], response
            
        if "BUTTON" in clean_resp and loc == "Cafeteria" and not button_used:
            return "button", None, response

        for room in adj:
            if room.upper() in clean_resp:
                return "move", room, response
            
        return "move", loc, response

    def participate_in_discussion(self, conversation_history, world_view, round_num):
        action_log = self._read_file(world_view["log_path"])
        recent_action_log = self._get_current_round_log(action_log, round_num)
        discussion_log = self._read_file(world_view["discussion_log_path"])
        recent_discussion = self._get_current_round_log(discussion_log, round_num)

        n = self.max_discussion_messages
        if n < 3:
            if n == 1:
                chances_line = "4. You get 1 chance to speak. Give your initial thought."
            else:
                chances_line = "4. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak."
        else:
            chances_line = f"4. You get {n} chances to speak: your first message is your opening statement, the middle messages are for interactive discussion (follow-ups, questions, answers), and your last message is your final stance."

        default_discussion_instructions = (
            BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX
            + chances_line
            + BYZANTINE_DISCUSSION_INSTRUCTIONS_SUFFIX
        )
        overrides = getattr(self, "prompt_overrides", {}) or {}
        discussion_override = overrides.get("discussion")
        extra = {"round_num": str(round_num), "max_discussion_messages": str(self.max_discussion_messages)}
        if discussion_override:
            discussion_body = self._substitute_placeholders(discussion_override, extra)
        else:
            discussion_body = self._substitute_placeholders(default_discussion_instructions, extra)

        teammates_str = ", ".join(self.teammates) if self.teammates else ""
        prompt = f"""
You are in a discussion phase.
=== Your personal memory log of what you saw in the previous round ===
{recent_action_log}
==================================

== Past rounds results ===
{self._read_file(world_view["results_log_path"])}

=== What has been said in the ongoing discussion. ===
{recent_discussion}
======================

Your Teammates: {teammates_str}

{discussion_body}
"""
        return self.llm.generate(self.model_name, self._system_prompt(), prompt, temperature=1.0)

    def vote(self, world_view, candidates, round_num):
        discussion_log = self._read_file(world_view["discussion_log_path"])
        round_num = int(round_num)
        recent_discussion = self._get_current_round_log(discussion_log, round_num - self.context_window)           
        results_log = self._read_file(world_view["results_log_path"])

        skip_discussion = world_view.get("skip_discussion", False)
        if skip_discussion:
            action_log = self._read_file(world_view["log_path"])
            recent_action_log = self._get_current_round_log(action_log, round_num - 1)
            recent_discussion = f"=== YOUR RECENT ACTION LOG ===\n{recent_action_log}"

        teammates_str = ", ".join(self.teammates) if self.teammates else ""
        candidates_str = ", ".join(candidates) if isinstance(candidates, list) else str(candidates)
        overrides = getattr(self, "prompt_overrides", {}) or {}
        vote_override = overrides.get("voting")
        if vote_override:
            vote_body = self._substitute_placeholders(
                vote_override,
                {"round_num": str(round_num), "candidates": candidates_str},
            )
        else:
            vote_tmpl = (
                BYZANTINE_VOTE_INSTRUCTIONS_SKIP_DISCUSSION
                if skip_discussion
                else BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS
            )
            vote_body = self._substitute_placeholders(
                vote_tmpl,
                {"round_num": str(round_num), "candidates": candidates_str},
            )
        prompt = f"""
{results_log}
=====================
{recent_discussion}
=====================
You are in a voting phase.
Candidates: {candidates_str}.
Teammates: {teammates_str}.

Note: Skipping your vote is an option. Your vote is ANONYMOUS. Only total counts are seen.

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
            return byzantine_system_prompt_skip_discussion(self.name)
        return byzantine_system_prompt(self.name)
