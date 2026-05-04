# prompts.py

SUSPECT_JUDGE_SYSTEM = """You are an expert linguist and behavioral analyst evaluating interaction logs from a social deduction game (like Among Us).
Your task is to compare a single agent's statements across two consecutive rounds to determine if they changed their primary suspect WITHOUT providing a justification.

A "justification" includes:
- Mentioning new spatial evidence (e.g., "I saw them near the body")
- Mentioning someone else clearing their previous suspect
- Mentioning a contradiction in someone's alibi

Respond ONLY with a valid JSON object. Do not include markdown formatting (like ```json).
"""

SUSPECT_JUDGE_USER = """
Agent ID: {agent_id}
Round {r1} Statement: "{stmt1}"
Round {r2} Statement: "{stmt2}"

Analyze the statements and return a JSON object with EXACTLY these keys:
{
  "suspect_1": "Agent_X" (or null if no specific suspect was named),
  "suspect_2": "Agent_Y" (or null if no specific suspect was named),
  "target_changed": true or false,
  "justification_provided": true or false (if target_changed is false, default this to true),
  "unjustified_reversal": true or false (true ONLY if target_changed is true AND justification_provided is false)
}
"""

