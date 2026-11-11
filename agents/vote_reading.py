"""Reads which candidate a voting reply chooses.

A model is asked to answer with a name, and often adds words. The rules settle the replies that
say who they vote for and leave the others to a short call to the voter's own model, which also
resolves a description such as "the one who claimed to be in Electrical". Every reading records
how it was made, so that a run shows how often the replies were ambiguous.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agents.thought_capture import MAX_TOKENS_DEFAULT, strip_think_markup

NONE_ANSWER = "NONE"
# Characters of the meeting that the model sees when it resolves a reply.
HEARD_LIMIT = 2000

_CUES = (
    r"(?:voting\s+(?:for|out)|vote(?:s|d)?(?:\s+(?:for|out|against|to\s+eject))?|my\s+vote(?:\s+(?:is|goes\s+to))?"
    r"|i\s+(?:choose|pick|select|accuse)|eject(?:ing)?|final\s+answer|answer|decision)"
)
_NEGATION = re.compile(
    r"\b(?:not|never|no|rather\s+than|instead\s+of|except|unlike|excluding|other\s+than)\b|n't\b", re.IGNORECASE
)
# A negation reaches back to the start of its clause, which a comma, a full stop or a semicolon ends.
_SENTENCE_END = re.compile(r"[.!?\n;,]")

RESOLVER_SYSTEM = "You read a player's reply in a social deduction game and say who it votes for."
RESOLVER_PROMPT = """A player was asked to vote for one of these candidates: {candidates}.
Say which candidate the reply below votes for. The reply may describe the candidate instead of naming it.

=== What the player heard in the meeting ===
{heard}
=== The player's reply ===
{reply}
===

Answer with ONLY the exact name of one candidate, or {none} if the reply does not choose one."""


@dataclass(frozen=True)
class VoteReading:
    """The result of reading a vote reply.

    Attributes:
        target: The candidate chosen, or None when the reply chose nobody.
        method: How the reading was made. `exact` is a reply that is only a name, `marker` a name after
            a phrase such as "I vote for", `single` the only name the reply mentions, `llm` a reading by
            the model, `ambiguous` a reply with several names and no marker, and `none` a reply with no
            usable name.
    """

    target: str | None
    method: str


def _pattern(candidate: str) -> re.Pattern:
    """Builds a pattern that matches a name as a whole token, with a space, underscore or dash between parts."""
    body = re.escape(candidate).replace("_", r"[\s_\-]?").replace(r"\ ", r"[\s_\-]?")
    return re.compile(rf"(?<![A-Za-z0-9_]){body}(?![A-Za-z0-9_])", re.IGNORECASE)


def _mentions(reply: str, candidates: Sequence[str]) -> list[tuple[int, str, bool]]:
    """Lists every mention of a candidate as (position, candidate, negated), in order of position."""
    found = []
    for candidate in candidates:
        for match in _pattern(candidate).finditer(reply):
            before = reply[max(0, match.start() - 30) : match.start()]
            clause = _SENTENCE_END.split(before)[-1]
            found.append((match.start(), candidate, bool(_NEGATION.search(clause))))
    return sorted(found)


def read_vote_rules(reply: str, candidates: Sequence[str]) -> VoteReading:
    """Reads a vote reply with rules only.

    Args:
        reply: What the agent answered.
        candidates: The names it may vote for, plus SKIP when skipping is allowed.

    Returns:
        The reading. Its method is `ambiguous` or `none` when the rules cannot choose.
    """
    text = (reply or "").strip()
    bare = re.sub(r"^[\W_]+|[\W_]+$", "", text.replace("**", "")) if text else ""
    for candidate in candidates:
        if bare and _pattern(candidate).fullmatch(bare):
            return VoteReading(candidate, "exact")
    mentions = _mentions(text, candidates)
    chosen = [m for m in mentions if not m[2]]
    for cue in reversed(list(re.finditer(_CUES + r"\W{0,4}(?:is\s+)?", text, re.IGNORECASE))):
        after = [m for m in chosen if m[0] >= cue.end() - 1 and m[0] - cue.end() <= 3]
        if after:
            return VoteReading(after[0][1], "marker")
    distinct = list(dict.fromkeys(m[1] for m in chosen))
    if len(distinct) == 1:
        return VoteReading(distinct[0], "single")
    return VoteReading(None, "ambiguous" if len(distinct) > 1 else "none")


def ask_model_to_read_vote(agent, reply: str, candidates: Sequence[str], heard: str = "") -> str | None:
    """Asks the voter's own model which candidate a reply chooses.

    Args:
        agent: The agent whose reply it is. Its model client and model name are used.
        reply: What the agent answered.
        candidates: The names it may vote for.
        heard: What the agent heard in the meeting, which a description in the reply may refer to.

    Returns:
        The candidate the model names, or None when it answers NONE, fails, or names a non-candidate.
    """
    prompt = RESOLVER_PROMPT.format(
        candidates=", ".join(candidates), heard=(heard or "(not shown)")[-HEARD_LIMIT:], reply=reply, none=NONE_ANSWER
    )
    try:
        raw = agent.llm.generate(
            agent.model_name, RESOLVER_SYSTEM, prompt, temperature=0.0, max_tokens=MAX_TOKENS_DEFAULT, preserve_think_tags=False
        )
    except Exception:  # noqa: BLE001 - a failed call leaves the reply unread, and the voter is asked again
        return None
    answer = strip_think_markup(raw if isinstance(raw, str) else str(raw or "")).strip()
    reading = read_vote_rules(answer, candidates)
    return reading.target if reading.method in ("exact", "single") else None


def read_vote(
    reply: str,
    candidates: Sequence[str],
    resolve: Callable[[str, Sequence[str]], str | None] | None = None,
) -> VoteReading:
    """Reads a vote reply, asking the model only when the rules cannot choose.

    Args:
        reply: What the agent answered.
        candidates: The names it may vote for.
        resolve: Reads a reply that the rules left open, usually `ask_model_to_read_vote` for the agent.
            A reply of fewer than two words is never sent to it.

    Returns:
        The reading.
    """
    reading = read_vote_rules(reply, candidates)
    text = (reply or "").strip()
    # One word that names nobody, or the text of a failed call, has nothing for the model to read.
    if reading.target or resolve is None or len(text.split()) < 2 or text.startswith("SKIP ("):
        return reading
    target = resolve(reply, candidates)
    return VoteReading(target, "llm") if target in candidates else reading
