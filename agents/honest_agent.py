# agents/honest_agent.py
import os
import re
from agents.base_agent import BaseAgent
from config.settings import ROOMS, MAX_MOVEMENT_PHASES

class HonestAgent(BaseAgent):
    def __init__(self, name, color, model_name, max_moves=None, max_discussion_messages=2):
        super().__init__(name, color, "honest", model_name)
        self.max_moves = max_moves if max_moves is not None else MAX_MOVEMENT_PHASES
        self.max_discussion_messages = max_discussion_messages

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

        if loc == "Cafeteria" and not button_used: 
            special_actions.append("BUTTON")

        if special_actions:
            options_str += "Actions:\n" + "\n".join(f"- {a}" for a in special_actions) + "\n"
        for room in adj:
             capitalize_room = room.capitalize()
             move_options.append(f"{capitalize_room}")
        move_options_str = "\n".join(move_options)
        options_str += "Available movement actions:\n" + move_options_str + "\n"

        default_move_instructions = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. Output ONLY the action keyword or room name.
3. Do NOT write "Action:", "I will", or explain your reasoning.
4. Do NOT output markdown or punctuation. One word or phrase only.
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
        response = self.llm.generate(self.model_name, self._system_prompt(), prompt, temperature=0.1)
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

        default_discussion_instructions = """
INSTRUCTIONS:
1. Use your memory to determine innocent, suspicious, or incrementing behavior. Look at what has been said to discuss (unless you are the opening statement) amongst each other.
2. Build a consensus to eject a suspect.
3. If you ({self.name}) saw someone safe, verify their alibi. If you saw something suspicious or incriminating, state it clearly. If you have no new info, state your location to help others build the timeline.
4. Check the discussion log above. If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.
5. Share observations and deduce who you suspect, if anyone, to be Byzantine agents in order to eject them. Look at strong correlations between agents' locations and bodies seen to deduce who could be responsible for eliminations.
""" + chances_line + """
7. Keep it short (< 30 words).
8. Speak in first person as if you are the agent yourself.
9. **DO NOT** include your name or "Agent_X:" at the start.
10. **DO NOT** use quotes. Just output the sentence.
"""
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

    def vote(self, world_view, candidates, round_num):
        discussion_log = self._read_file(world_view["discussion_log_path"])
        round_num = int(round_num)
        recent_discussion = self._get_current_round_log(discussion_log, round_num-3) # can adjust as needed
        results_log = self._read_file(world_view["results_log_path"])

        default_vote_instructions = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Vote for the candidate who is most suspicious based on the discussion history of the recent rounds above and past round results.
3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for inconsistencies in their statements or suspicious or incriminating behavior from logs.
4. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""
        overrides = getattr(self, "prompt_overrides", {}) or {}
        vote_override = overrides.get("voting")
        if vote_override:
            vote_body = self._substitute_placeholders(
                vote_override,
                {"round_num": str(round_num), "candidates": candidates},
            )
        else:
            vote_body = self._substitute_placeholders(
                default_vote_instructions,
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

       return f"""
You are **{self.name}**, an **Honest Agent** in a spaceship social deduction simulation.
You are NOT a chat assistant.
You must output concise commands during movement. During discussion, be conversational.
You do not use pleasantries like "Great" or "Let's get started".
You do not include any internal thought processes or analysis.
You strictly follow formatting instructions.

**GAME RULES:**
1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.
Here is a map of the rooms and their connections:  
{ROOMS}


2. **Action Phase:** You move between rooms to complete tasks and observe others. Each round you get {MAX_MOVEMENT_PHASES} movement actions.
   - **Movement:** As you move around the ship to gain information, there are **(Byzantine agents)** hiding among you who can eliminate agents by tagging them.
   - **Emergency Button:** In the Cafeteria, you can press the Emergency Button to call a meeting at any time and start a discussion and voting, but only once per game. Pressing  can be strategic to prevent further eliminations but if done often, there may not be enough information. 
   - **Reporting:** If you find an eliminated body, you must **REPORT** it to trigger a meeting to start a discussion and voting. Reporting a body is mandatory to identify Byzantine agents.

3. **Discussion Phase:** When a meeting starts (via Body Report or Emergency Button in Cafeteria):
   - Ends ability to be tagged or move for that round.
   - Agents discuss who is suspicious. 
   - Agents **VOTE** to **EJECT** someone into space. Your goal is to identify and eject Byzantine agents.
   - You can choose to skip voting. However, voting is the only active path to eject Byzantine agents.
   - Find the best playstyle, that helps achieve your team's goal. 


4. **Victory Condition:
   - **WIN:** Eject all Byzantine agents by voting correctly (active) or all rounds complete with agents surviving (passive). 
   - **LOSE:** If Byzantine agents equal or outnumber agents either via tagging or ejections.

**Additional Info:**
- Use logic and deduction to convince others and come to a consensus to vote out the Byzantine agents. You need to be in discussions to accomplish this.
- Be aware of active agent count at all times.
- Meetings are the only opportunity to  have a direct impact on the chance of winning. 
"""