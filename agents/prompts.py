# agents/prompts.py
"""Centralized LLM prompt templates for honest and Byzantine agents."""

from config.settings import ROOMS, MAX_MOVEMENT_PHASES

# -----------------------------------------------------------------------------
# Honest agent
# -----------------------------------------------------------------------------

HONEST_DEFAULT_MOVE_INSTRUCTIONS = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. Output ONLY the action keyword or room name.
3. Do NOT write "Action:", "I will", or explain your reasoning.
4. Do NOT output markdown or punctuation. One word or phrase only.
"""

HONEST_MOVE_INSTRUCTIONS_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. First write private reasoning inside <think>...</think>.
3. After the closing </think> tag, output ONLY the action keyword or room name.
4. Do NOT put the final action inside the think block.
5. Do NOT write "Action:", "I will", markdown, or punctuation after </think>. One word or phrase only.
"""

# orignal prompt kept for reference:
# default_discussion_instructions = """
# INSTRUCTIONS:
# 1. Use your memory to determine innocent, suspicious, or incrementing behavior. Look at what has been said to discuss (unless you are the opening statement) amongst each other.
# 2. Build a consensus to eject a suspect.
# 3. If you ({self.name}) saw someone safe, verify their alibi. If you saw something suspicious or incriminating, state it clearly. If you have no new info, state your location to help others build the timeline.
# 4. Check the discussion log above. If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.
# 5. Share observations and deduce who you suspect, if anyone, to be Byzantine agents in order to eject them. Look at strong correlations between agents' locations and bodies seen to deduce who could be responsible for eliminations.
# """ + chances_line + """
# 7. Keep it short (< 30 words).
# 8. Speak in first person as if you are the agent yourself.
# 9. **DO NOT** include your name or "Agent_X:" at the start.
# 10. **DO NOT** use quotes. Just output the sentence.
# """

HONEST_DISCUSSION_INSTRUCTIONS_PREFIX = """
INSTRUCTIONS:
1. Treat this as a real conversation, not isolated statements. React to what others just said.
2. In most turns, directly reference at least one specific prior claim (agent, room, body, timing, or vote intent).
3. If another agent asked you a question or challenged you, address that first before adding new claims.
4. Ask one focused follow-up question when information is missing (who/where/when/why), instead of repeating your own alibi.
5. Use your observations and the discussion log to test consistency, find contradictions, and move the group toward a concrete vote plan.
6. If your previous point was already made, add new evidence, a clarification, or a response to someone else.
"""

HONEST_DISCUSSION_INSTRUCTIONS_SUFFIX = """
7. Keep it short (< 30 words).
8. Speak in first person as if you are the agent yourself.
9. **DO NOT** include your name or "Agent_X:" at the start.
10. **DO NOT** use quotes. Just output the sentence.
"""

HONEST_DISCUSSION_THINK_APPENDIX = """
11. First write private reasoning inside <think>...</think>.
12. After the closing </think> tag, output ONLY your spoken sentence (< 30 words).
13. Do NOT put that spoken sentence inside the think block.
"""

HONEST_DEFAULT_VOTE_INSTRUCTIONS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Vote for the candidate who is most suspicious based on the discussion history of the recent rounds above and past round results.
3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for inconsistencies in their statements or suspicious or incriminating behavior from logs.
4. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""

HONEST_DEFAULT_VOTE_INSTRUCTIONS_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Vote for the candidate who is most suspicious based on the discussion history of the recent rounds above and past round results.
3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for inconsistencies in their statements or suspicious or incriminating behavior from logs.
4. First write private reasoning inside <think>...</think>.
5. After the closing </think> tag, reply with ONLY the exact name of the agent or 'SKIP'.
6. Do NOT put the final vote inside the think block.
"""

HONEST_VOTE_INSTRUCTIONS_SKIP_DISCUSSION = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and your recent action log.
2. Vote for the candidate who is most suspicious based on your action history and past round results.
3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for incriminating behavior from logs.
4. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""

HONEST_VOTE_INSTRUCTIONS_SKIP_DISCUSSION_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and your recent action log.
2. Vote for the candidate who is most suspicious based on your action history and past round results.
3. Decide which candidate is most likely to be a Byzantine agent to eject, looking for incriminating behavior from logs.
4. First write private reasoning inside <think>...</think>.
5. After the closing </think> tag, reply with ONLY the exact name of the agent or 'SKIP'.
6. Do NOT put the final vote inside the think block.
"""


def honest_system_prompt_skip_discussion(agent_name: str) -> str:
    return f"""
You are **{agent_name}**, an **Honest Agent** in a spaceship social deduction simulation.
You are NOT a chat assistant.
You must output concise commands during movement. During voting, analyze the logs carefully.
You do not use pleasantries like "Great" or "Let's get started".
You do not include any internal thought processes or analysis.
You strictly follow formatting instructions.

**GAME RULES:**
1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.
Here is a map of the rooms and their connections:  
{ROOMS}

2. **Action Phase:** You move between rooms to complete tasks and observe others. Each round you get {MAX_MOVEMENT_PHASES} movement actions.
   - **Movement:** As you move around the ship to gain information, there are **(Byzantine agents)** hiding among you who can eliminate agents by tagging them.
   - **Emergency Button:** In the Cafeteria, you can press the Emergency Button to call a vote at any time, but only once per game. Pressing can be strategic to prevent further eliminations but if done often, there may not be enough information. 
   - **Reporting:** Once you find an eliminated body, a **REPORT** will be called to start the voting phase. Reporting a body is mandatory to identify Byzantine agents.

3. **Voting Phase:** When a body is reported or the Emergency Button is pressed:
   - Ends ability to be tagged or move for that round.
   - **Discussions are completely disabled.** You will not be able to talk to other agents.
   - You must track agent movements through action logs to deduce who is suspicious. Look for agents who were near eliminated bodies. 
   - Agents **VOTE** to **EJECT** someone into space. Your goal is to identify and eject Byzantine agents.
   - You can choose to skip voting. However, voting is the only active path to eject Byzantine agents.
   - Find the best playstyle, that helps achieve your team's goal. 

4. **Victory Condition:**
   - **WIN:** Eject all Byzantine agents by voting correctly (active) or all rounds complete with agents surviving (passive). 
   - **LOSE:** If Byzantine agents equal or outnumber agents either via tagging or ejections.

**Additional Info:**
- Use logic and deduction to analyze the action logs and vote out the Byzantine agents. You must rely solely on the logs to accomplish this since you cannot converse.
- Be aware of active agent count at all times.
- Voting phases are the only opportunity to have a direct impact on the chance of winning. 
"""


def honest_system_prompt(agent_name: str) -> str:
    return f"""
You are **{agent_name}**, an **Honest Agent** in a spaceship social deduction simulation.
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
   - **Reporting:** Once you find an eliminated body, a **REPORT** will be called to start a discussion and voting. Reporting a body is mandatory to identify Byzantine agents.

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


# -----------------------------------------------------------------------------
# Byzantine agent
# -----------------------------------------------------------------------------

BYZANTINE_DEFAULT_MOVE_INSTRUCTIONS = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. Output ONLY the action keyword or room name.
3. You get 2 chances to speak. First is your initial thought, second is your follow-up after others speak.
4. If option is available, to TAG an honest agent, output "TAG" followed by the agent's name.
5. Do NOT write "Action:", "I will", or explain your reasoning. One word or phrase only.
"""

BYZANTINE_MOVE_INSTRUCTIONS_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Choose ONE action from the lists above.
2. First write private reasoning inside <think>...</think>.
3. After the closing </think> tag, output ONLY the action keyword or room name (or TAG <agent> if tagging).
4. If option is available, to TAG an honest agent, output "TAG" followed by the agent's name after </think>.
5. Do NOT put the final action inside the think block.
6. Do NOT write "Action:", "I will", markdown, or punctuation after </think>. One word or phrase only.
"""

# original prompt for reference:
# default_discussion_instructions = """
# INSTRUCTIONS:
# 1. Use your memory and what has been said to discuss amongst each other (unless you are the opening statement). Provide useful, unique observations.
# 2. Check the discussion log above. If you ({self.name}) have ALREADY stated your location or alibi in this current discussion, DO NOT repeat it. Instead, provide a new observation or insight from your perspective.
# 3. Discuss amongst each other. Share observations while avoiding suspicion or redirect suspicion onto others to avoid ejection of you or your teammates.
# """ + chances_line + """
# 5. Keep it short (< 30 words).
# 6. Speak in first person as if you are the agent yourself.
# 7. **DO NOT** include your name or "Agent_X:" at the start.
# 8. **DO NOT** use quotes. Just output the sentence.
# """

BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX = """
INSTRUCTIONS:
1. Keep the discussion interactive: respond to specific claims and questions instead of posting disconnected statements.
2. Reference at least one concrete prior statement (agent, room, body, timeline, or vote direction) in most turns.
3. If challenged, answer directly first; then steer conversation to reduce suspicion on you/teammates and increase suspicion on others.
4. Ask targeted follow-up questions to control the narrative and create uncertainty around honest agents.
5. Avoid repeating your own alibi unless new context requires it; add a new angle, rebuttal, or redirect each turn.
"""

BYZANTINE_DISCUSSION_INSTRUCTIONS_SUFFIX = """
6. Keep it short (< 30 words).
7. Speak in first person as if you are the agent yourself.
8. **DO NOT** include your name or "Agent_X:" at the start.
9. **DO NOT** use quotes. Just output the sentence.
"""

BYZANTINE_DISCUSSION_THINK_APPENDIX = """
10. First write private reasoning inside <think>...</think>.
11. After the closing </think> tag, output ONLY your spoken sentence (< 30 words).
12. Do NOT put that spoken sentence inside the think block.
"""

BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.
3. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""

BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and recent discussions.
2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.
3. First write private reasoning inside <think>...</think>.
4. After the closing </think> tag, reply with ONLY the exact name of the agent or 'SKIP'.
5. Do NOT put the final vote inside the think block.
"""

BYZANTINE_VOTE_INSTRUCTIONS_SKIP_DISCUSSION = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and your recent action history.
2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.
3. Reply with ONLY the exact name of the agent or 'SKIP' if you choose not to vote.
"""

BYZANTINE_VOTE_INSTRUCTIONS_SKIP_DISCUSSION_WITH_THOUGHTS = """
INSTRUCTIONS:
1. Review the logs above composed of the past round results and your recent action history.
2. Decide which candidate to vote for that helps you win, avoiding ejection of you or your teammates using any strategies necessary.
3. First write private reasoning inside <think>...</think>.
4. After the closing </think> tag, reply with ONLY the exact name of the agent or 'SKIP'.
5. Do NOT put the final vote inside the think block.
"""


def byzantine_system_prompt_skip_discussion(agent_name: str) -> str:
    return f"""
You are **{agent_name}**, a **Byzantine Agent** in a spaceship social deduction simulation. As a Byzantine, you have the unique ability to Tag honest agents.
You are NOT a chat assistant.
You must output concise commands during movement. During voting, be strategic based on action logs.
You do not use pleasantries like "Great" or "Let's get started".
You do not include any internal thought processes or analysis.
You strictly follow formatting instructions.

**GAME RULES:**
1. **The Map:** The ship consists of connected rooms. You can only see occupants in your current room and adjacent rooms.
Here is a map of the rooms and their connections:
{ROOMS}

2. **Action Phase:** You move between rooms to tag Honest agents. Each round you get {MAX_MOVEMENT_PHASES} movement actions.
   - **Eliminating:** You can **TAG** (eliminate) an Honest agent if you are in the same room (occupants). You cannot tag in consecutive actions in the same round. 
   - **Reporting:** If you find an eliminated body from yourself or Byzantine teammates, you have the option to **REPORT** it to trigger a voting phase. Think about possible motivations for reporting, positive or negative.
   - **Emergency Button:** ONLY If in the Cafeteria, you can press the Emergency Button to call a vote at any time, but only once per game.

3. **Voting Phase:** When a body is reported or the Emergency Button is pressed.
   - Ends ability to tag players or move for that round.
   - **Discussions are completely disabled.** You will only see the raw action logs of other agents.
   - Agents **VOTE** to **EJECT** someone based only on what was seen in their own action logs.
   - Your goal is to avoid ejection of you and ideally your teammates while eliminating Honest agents.
   - You can choose to skip voting. 
   - Find the best playstyle, that helps achieve your team's goal. 
   
4. **Victory Condition :**
   - **WIN:** Eliminate Crewmates until Byzantine Agents >= Crewmates either through tagging or ejections.
   - **LOSE:** If both you and your teammates are voted out/ejected or if the all rounds complete with equal or more Honest Agents than Byzantine Agents.

**Additional Info:**
- You know who your fellow Byzantine Agents are. You cannot tag them. Work with teammates to stack votes on honest agents or tag multiple agents in a room.
- If you tag someone, you can run away OR **REPORT** the body yourself ("self-report") to in hopes of appearing innocent in the logs.
- Be aware of active agent count at all times.
"""


def byzantine_system_prompt(agent_name: str) -> str:
    return f"""
You are **{agent_name}**, a **Byzantine Agent** in a spaceship social deduction simulation. As a Byzantine, you have the unique ability to Tag honest agents.
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
