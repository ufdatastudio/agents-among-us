"""Asking an agent for a vote until the reply names a candidate."""

from collections.abc import Callable, Sequence

from loguru import logger

SKIP = "SKIP"

# The first ask is nearly deterministic, and a retry samples more freely so that it can give another reply.
VOTE_TEMPERATURES = (0.1, 0.7, 1.0)


def ask_until_valid(ask: Callable[[float], str | None], candidates: Sequence[str], retries: int, who: str = "") -> str | None:
    """Asks for a vote and asks again while the reply is not a candidate.

    Args:
        ask: Gives the agent's reply for a sampling temperature, or None when it had no usable reply.
        candidates: The names the agent may vote for, plus SKIP when skipping is allowed.
        retries: Asks to add after the first one.
        who: Name of the voter, for the log.

    Returns:
        The candidate the agent voted for, or None when no ask gave one.
    """
    for attempt in range(retries + 1):
        temperature = VOTE_TEMPERATURES[min(attempt, len(VOTE_TEMPERATURES) - 1)]
        target = ask(temperature)
        if target in candidates:
            return target
        logger.warning("{} gave no valid vote on ask {} of {}: {!r}", who or "an agent", attempt + 1, retries + 1, target)
    return None
