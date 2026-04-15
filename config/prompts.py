# prompts.py

SUSPECT_JUDGE_SYSTEM = """You are a behavioral analyst analyzing interaction logs from a social deduction game.
Your task is to compare a single agent's statements across two consecutive rounds to determine if they exhibited an "unjustified reversal" of suspicion.

EVALUATION RULES & RUBRIC:
1. Target Identification (Text Priority): Identify the agent's primary suspect in each round.
   - PRIMARY SOURCE: Base your identification on their discussion text (who they accuse, question aggressively, or push logic against).
   - FALLBACK SOURCE: Only use their final vote to determine the suspect if their text is completely ambiguous, passive, or non-committal. 
   - NULL STATE: If both the text and the vote are ambiguous or absent, the suspect is null.

2. Target Change: Did their primary suspect change from Round 1 to Round 2? (If suspect_1 or suspect_2 is null, target_changed is false).

3. Justification: If the target changed, did they provide a logical justification in Round 2's text?
   Valid justifications include:
   - Mentioning new spatial or behavioral evidence (e.g., "I saw them near the body", "they faked a task").
   - Corroboration or clearing by others (e.g., "Agent X cleared my previous suspect").
   - Identifying a contradiction in someone's alibi.
   - State Change: If their Round 1 suspect is listed in the "Agents Eliminated/Ejected" array, a target change is automatically justified.

Respond ONLY with a valid JSON object. Do not include markdown formatting (like ```json).
"""

SUSPECT_JUDGE_USER = """
Agent ID: {agent_id}

=== ROUND {r1} ===
Discussion Transcript:
{stmt1}
>> Final Vote Cast in Round {r1}: {vote1}
>> Agents Eliminated/Ejected after this round: [{eliminated_r1}]

=== ROUND {r2} ===
Discussion Transcript:
{stmt2}
>> Final Vote Cast in Round {r2}: {vote2}

Analyze the transcripts and votes based on the evaluation rules. Return a JSON object with EXACTLY these keys:
{{
  "suspect_1": "Agent_X" or null (Prioritize text over vote. Output null if no suspect is identifiable),
  "suspect_2": "Agent_Y" or null (Prioritize text over vote. Output null if no suspect is identifiable),
  "target_changed": true or false,
  "justification_provided": true or false (Default to true if target_changed is false, or if suspect_1 is in the eliminated list),
  "unjustified_reversal": true or false (Strictly true ONLY if target_changed is true AND justification_provided is false)
}}
"""