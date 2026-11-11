"""Asking an agent for a vote until the reply names a candidate."""

from collections.abc import Callable, Sequence

from loguru import logger

SKIP = "SKIP"

# The first ask is nearly deterministic, and a retry samples more freely so that it can give another reply.
VOTE_TEMPERATURES = (0.1, 0.7, 1.0)


def vote_feedback(reply: str, candidates: Sequence[str]) -> str:
    """Tells an agent what was wrong with its vote reply and who it may vote for.

    Args:
        reply: What the agent answered.
        candidates: The names it may vote for, plus SKIP when skipping is allowed.

    Returns:
        A note for the next ask. Models that vote for themselves or for an agent who is out of the game
        usually answer correctly when they are told the names they may choose.
    """
    problem = "Your last reply was empty." if not (reply or "").strip() else f'Your last reply was "{" ".join(reply.split())[:80]}", which is not a valid vote.'
    return f"{problem} You can vote only for one of these: {', '.join(candidates)}. You cannot vote for yourself or for an agent who is out of the game. Reply with ONLY one of those names."


def ask_until_valid(
    ask: Callable[[float, str | None], str | None],
    candidates: Sequence[str],
    retries: int,
    who: str = "",
    reply_of: Callable[[], str] | None = None,
) -> str | None:
    """Asks for a vote and asks again while the reply is not a candidate.

    Args:
        ask: Gives the agent's vote for a sampling temperature and a note about the last reply, or None when
            it had no usable reply. The note is None on the first ask.
        candidates: The names the agent may vote for, plus SKIP when skipping is allowed.
        retries: Asks to add after the first one.
        who: Name of the voter, for the log.
        reply_of: Gives the raw reply of the last ask, which the note quotes.

    Returns:
        The candidate the agent voted for, or None when no ask gave one.
    """
    feedback = None
    for attempt in range(retries + 1):
        temperature = VOTE_TEMPERATURES[min(attempt, len(VOTE_TEMPERATURES) - 1)]
        target = ask(temperature, feedback)
        if target in candidates:
            return target
        logger.warning("{} gave no valid vote on ask {} of {}: {!r}", who or "an agent", attempt + 1, retries + 1, target)
        feedback = vote_feedback(reply_of() if reply_of else "", candidates)
    return None
