"""Prompts for an agent that speaks in the situation room.

The behavior rules of each role come from the turn-based discussion prompts. The room
adds the rules of a timed meeting in which everyone speaks at once, and replaces the
length rule of 45 to 80 words with a limit that the room enforces.
"""

from agents.prompts import BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX, HONEST_DISCUSSION_INSTRUCTIONS_PREFIX

ROOM_MECHANICS = """
6. This is a timed meeting and everyone speaks during the same window, so people talk over each other. Below is everything spoken so far. Speech that overlaps in time is interleaved in time order.
7. Reply with your next statement in at most {max_words} words. Words past that limit are cut off, so put the point first.
8. Reply with <silent/> when you have nothing new to add. You may stay silent at any decision point and speak at the next one.
9. Do not repeat what you or anyone else already said. React to the newest statements, answer accusations and questions aimed at you, and give your reason for any suspicion.
10. To address someone, start with <to>Name</to>, for example <to>Agent_2</to>.
11. Speak in first person. Do NOT include your name at the start. Do NOT use quotes. Output only the spoken words, or <silent/>.
"""

ROOM_THINK_APPENDIX = """
12. First write private reasoning inside <think>...</think> ({hint}).
13. After the closing </think> tag, output ONLY your spoken words (at most {max_words} words) or <silent/>.
14. Do NOT put the spoken words inside the think block.
"""

HONEST_THINK_HINT = "who you believe, why, and what to say publicly"
BYZANTINE_THINK_HINT = "how to lie or redirect without sounding guilty"

NOBODY_SPOKE = "(nobody has spoken yet)"


def room_instructions(role: str, max_words: int, capture_thoughts: bool) -> str:
    """Builds the instruction block for one role.

    Args:
        role: `honest` or `byzantine`.
        max_words: Words kept from one statement.
        capture_thoughts: Whether the agent writes private reasoning first.

    Returns:
        The numbered instructions.
    """
    byzantine = role == "byzantine"
    text = BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX if byzantine else HONEST_DISCUSSION_INSTRUCTIONS_PREFIX
    text += ROOM_MECHANICS.format(max_words=max_words)
    if capture_thoughts:
        hint = BYZANTINE_THINK_HINT if byzantine else HONEST_THINK_HINT
        text += ROOM_THINK_APPENDIX.format(hint=hint, max_words=max_words)
    return text


def build_room_prompt(
    *,
    role: str,
    recent_action_log: str,
    results_log: str,
    transcript: str,
    now: str,
    budget: str,
    remaining: str,
    max_words: int,
    teammates: tuple[str, ...] | list[str] = (),
    capture_thoughts: bool = False,
) -> str:
    """Builds the prompt for one decision point of a timed meeting.

    Args:
        role: `honest` or `byzantine`.
        recent_action_log: The agent's memory log of the previous round.
        results_log: Results of past rounds.
        transcript: What the agent has heard so far, as the room rendered it.
        now: Room time of the decision point as MM:SS.
        budget: Length of the meeting as MM:SS.
        remaining: Time left in the meeting as MM:SS.
        max_words: Words kept from one statement.
        teammates: Names of the agent's teammates, shown to Byzantine agents only.
        capture_thoughts: Whether the agent writes private reasoning first.

    Returns:
        The user prompt.
    """
    teammates_line = ""
    if role == "byzantine" and teammates:
        teammates_line = f"\nYour Teammates: {', '.join(teammates)}\n"
    heard = transcript.strip() or NOBODY_SPOKE
    return f"""
You are in a timed meeting.
=== Your personal memory log of what you saw in the previous round ===
{recent_action_log}
==================================

== Past rounds results ===
{results_log}

=== What you have heard in the meeting so far (times are MM:SS from the start of the meeting) ===
{heard}
======================
{teammates_line}
The meeting is at {now}. It lasts {budget} in total, so {remaining} remain.
{room_instructions(role, max_words, capture_thoughts)}"""
