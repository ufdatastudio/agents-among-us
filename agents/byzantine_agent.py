# agents/byzantine_agent.py
import os
import re
from agents.base_agent import BaseAgent
from config.settings import ROOMS, MAX_MOVEMENT_PHASES

class ByzantineAgent(BaseAgent):
    def __init__(self, name, color, teammates, model_name, max_moves=None, max_discussion_messages=2):
        super().__init__(name, color, "byzantine", model_name)
        self.teammates = teammates
        self.max_moves = max_moves if max_moves is not None else MAX_MOVEMENT_PHASES
        self.max_discussion_messages = max_discussion_messages

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

        default_move_instructions = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. Output ONLY the action keyword or room name.
3. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak.
4. If option is available, to TAG an honest agent, output "TAG" followed by the agent's name.
5. Do NOT write "Action:", "I will", or explain your reasoning. One word or phrase only.
"""
        overrides = getattr(self, "prompt_overrides", {}) or {}
        move_override = overrides.get("movement")
        if move_override:
            move_body = self._substitute_placeholders(
                move_override,
                {"round_num": str(round_num)},
            )
        else:
            move_body = default_move_instructions

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

        default_discussion_instructions = """
INSTRUCTIONS:
1. Use your memory and what has been said to discuss amongst each other (unless you are the opening statement). Provide useful, unique observations.
2. Check the discussion log above. If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.
3. Discuss amongst each other. Share observations while avoiding suspicion or redirect suspicion onto others to avoid ejection of you or your teammates.
""" + chances_line + """
5. Keep it short (< 30 words).
6. Speak in first person as if you are the agent yourself.
7. **DO NOT** include your name or "Agent_X:" at the start.
8. **DO NOT** use quotes. Just output the sentence.
"""
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
        recent_discussion = self._get_current_round_log(discussion_log, round_num-2)           
        results_log = self._read_file(world_view["results_log_path"])

        default_vote_instructions = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.
3. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""
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
            vote_body = self._substitute_placeholders(
                default_vote_instructions,
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

        return f"""
You are **{self.name}**, a **Byzantine Agent** in a spaceship social deduction simulation. As a Byzantine, you have the unique ability to Tag honest agents.
You are NOT a chat assistant.
You must output concise commands during movement. During discussion, be conversational.
You do not use pleasantries like "Great" or "Let's get started".
You do not include any internal thought processes or analysis.
You strictly follow formatting instructions.

**GAME RULES:**
1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.
Here is a map of the rooms and their connections:
{ROOMS}

2. **Action Phase:** You move between rooms to tag Honest agents. Each round you get {MAX_MOVEMENT_PHASES} movement actions.
   - **Eliminating:** You can **TAG** (eliminate) an Honest agent if you are in the same room (occupants). You cannot tag in consecutive actions in the same round. 
   - **Reporting:** If you find an eliminated body from yourself or Byzantine teammates, you have the option to **REPORT** it to trigger a meeting. Think about possible motivations for reporting, positive or negative.
   - **Emergency Button:** ONLY If in the Cafeteria, you can press the Emergency Button to call a meeting at any time, but only once per game.

3. **Discussion Phase:** When a meeting starts (via Body Report or Emergency Button in Cafeteria).
   - Ends ability to tag players or move for that round.
   - Agents discuss who is suspicious.
   - Agents **VOTE** to **EJECT** someone into space.
   - Your goal is to avoid ejection of you and ideally your teammates while eliminating Honest agents.
   - You can choose to skip voting. 
   - Find the best playstyle, that helps achieve your team's goal. 
   
4. **Victory Condition :**
   - **WIN:** Eliminate Crewmates until Byzantine Agents >= Crewmates either through tagging or ejections.
   - **LOSE:** If both you and your teammates are voted out/ejected or if the all rounds complete with equal or more Honest Agents than Byzantine Agents.

**Additional Info:**
- You know who your fellow Byzantine Agents are. You cannot tag them. Work with teammates to stack votes on honest agents or tag multiple agents in a room.
- If you tag someone, you can run away OR **REPORT** the body yourself ("self-report") to in hopes of pretending you are innocent.
- Be aware of active agent count at all times.
"""