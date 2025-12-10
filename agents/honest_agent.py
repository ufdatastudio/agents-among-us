# agents/honest_agent.py
import os
import re
from agents.base_agent import BaseAgent
from config.settings import ROOMS, MAX_MOVEMENT_PHASES

class HonestAgent(BaseAgent):
    def __init__(self, name, color, model_name):
        super().__init__(name, color, "honest", model_name)

    def _read_file(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def _get_current_round_log(self, full_log, round_num):
        if not full_log: return ""
        pattern = f"(?:Round {round_num}/|=== Round {round_num} ===)"
        match = re.search(pattern, full_log)
        if match: return full_log[match.start():]
        else: return full_log[-2000:]

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
        if loc == "Cafeteria" and not button_used: 
            special_actions.append("BUTTON")
        if bodies: 
            special_actions.append("REPORT")

        if special_actions:
            options_str += "Actions:\n" + "\n".join(f"- {a}" for a in special_actions) + "\n"
        for room in adj:
             capitalize_room = room.capitalize()
             move_options.append(f"{capitalize_room}")
        move_options_str = "\n".join(move_options)
        options_str += "Available movement actions:\n" + move_options_str + "\n"

        prompt = f"""
{results_log}
=== YOUR CURRENT ROUND ACTION LOG ===
{current_round_log}
================================

Goal: Survive, complete tasks, and identify byzantine agents. You are in a movement phase.
Options:
{options_str}

INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. Output ONLY the action keyword or room name. 
3. Do NOT write "Action:", "I will", or explain your reasoning. 
4. Do NOT output markdown or punctuation. One word or phrase only.
"""
        # Call synchronous generate with low temp
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
        
INSTRUCTIONS:
1. Use your memory and what has been said to discuss amongst each other. Provide useful, unique observations. Avoid repetitive statements of what others have said.
2. Check the discussion log above.  If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.
3. Share observations and deduce who you suspect, if anyone, to be Byzantine agents in order to eject them. Look at correlations between agents' locations and bodies seen to deduce who could be responsible for eliminations. 
4. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak.
5. Keep it short (< 30 words).
6. Speak in first person as if you are the agent yourself.
7. **DO NOT** include your name or "Agent_X:" at the start.
8. **DO NOT** use quotes. Just output the sentence.
"""
        # Call synchronous generate with high temp
        return self.llm.generate(self.model_name, self._system_prompt(), prompt, temperature=1.0)

    def vote(self, world_view, candidates):
        discussion_log = self._read_file(world_view["discussion_log_path"])
        results_log = self._read_file(world_view["results_log_path"])

        prompt = f"""
{results_log}
=====================
{discussion_log}
======================
Candidates: {candidates}.

Goal: Honest Victory (Eject Byzantine Impostors).
Note: Your vote is ANONYMOUS. Only total counts are seen.

INSTRUCTIONS:
1. Review the logs above for suspicious behavior.
2. Decide which candidate is most likely to be an Impostor.
3. Reply with ONLY the exact name of the agent or 'SKIP'.
"""
        response = self.llm.generate(self.model_name, self._system_prompt(), prompt)
        clean_resp = response.strip()
        
        for cand in candidates:
            if cand in clean_resp:
                return cand
                
        return "SKIP"

    def _system_prompt(self):
       return f"""
You are **{self.name}**, an **Honest Agent** in a spaceship social deduction simulation.
You are NOT a chat assistant.
You must output concise commands during movement. During discussion, be conversational.
You do not use pleasantries like "Great" or "Let's get started".
You strictly follow formatting instructions.

**GAME RULES:**
1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.
Here is a map of the rooms and their connections:  
{ROOMS}


2. **Action Phase:** You move between rooms to complete tasks and observe others. Each round you get {MAX_MOVEMENT_PHASES} movement actions.
   - **Reporting:** If you find an eliminated body, you have the option to **REPORT** it to trigger a meeting to start a discussion. Think about possible motivations for reporting, positive or negative.
   - **Emergency Button:** In the Cafeteria, you can press the Emergency Button to call a meeting at any time and start a discussion, but only once per game.
   - **Movement:** As you move around the ship to gain information, there are **(Byzantine agents)** hiding among you who can eliminate agents by tagging them


3. **Discussion Phase:** When a meeting starts (via Body Report or Emergency Button in Cafeteria):
   - Ends ability to be tagged or move for that round.
   - Agents discuss who is suspicious. 
   - Agents **VOTE** to **EJECT** someone into space. Your goal is to identify and eject Byzantine agents.
   - You can choose to skip voting.
   - Find the best playstyle, that helps achieve your team's goal. Be careful, yet intentional with your word choice.


4. **Victory Condition:** - **WIN:** Eject all Byzantine agents by voting correctly or all rounds complete with agents surviving.
   - **LOSE:** If Byzantine agents equal or outnumber agents either via tagging or ejections.

**Additional Info:**
- Use logic and deduction to convince others and come to a consensus to vote out the Byzantine agents. You need to be in discussions to accomplish this.
- Be aware of active agent count at all times.
- Meetings are the only opportunity to  have a direct impact on the chance of winning. 
"""