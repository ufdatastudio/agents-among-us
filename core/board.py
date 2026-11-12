"""The board of a meeting, a reader's view of the claims set against what the reader saw itself.

The engine writes what each agent observed as facts that only that agent may read. The room writes what each
statement claims, and whom it accuses or vouches for, as claims that every agent in the meeting may read. The board
is a query over the records one reader may read. It groups the claims by the agent they are about and puts the
reader's own sightings next to each. Nothing is rejected and nothing is hidden, so the reader still has to judge
who lies, and a reader that did not see an agent has no evidence either way.

Usage:
    ledger = Ledger()
    record_observations_from_log(ledger, "Agent_2", action_log_text)
    record_statement(ledger, "Agent_5", "I saw Agent_3 in Storage.", 12.0, roster, round_num=1)
    print(render_board(ledger, "Agent_2", t=20.0, round_num=1))
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence

from config.settings import ROOMS
from core.claims import extract_claims, extract_edges, parse_adjacent, parse_observations
from core.ledger import CLAIM, FACT, NOTE, Ledger, Record

ROOM_NAMES = sorted(ROOMS, key=len, reverse=True)
MAX_SUBJECTS = 10
MAX_LINES = 6
HEADER = (
    "Board of this meeting. The claims made so far, grouped by the agent they are about, each next to what you saw yourself. "
    "You only know what you saw, so not having seen something proves nothing."
)


def record_observation(
    ledger: Ledger, agent: str, round_num: int, tick: int, room: str, beside: Iterable[str], adjacent: dict[str, Iterable[str]]
) -> Record:
    """Writes what an agent saw at one movement tick, readable by that agent only.

    Args:
        ledger: The ledger of the game.
        agent: The agent who observed.
        round_num: Round of the game.
        tick: Movement tick of the round, from 1.
        room: Room the agent was in.
        beside: Agents it saw in that room.
        adjacent: Agents it saw in each adjacent room.

    Returns:
        The fact.
    """
    content = {"room": room, "beside": sorted(beside), "adjacent": {r: sorted(a) for r, a in sorted(adjacent.items())}}
    return ledger.append(FACT, "engine", content, round_num=round_num, tick=tick, subject=agent, visible_to=(agent,))


def record_observations_from_log(ledger: Ledger, agent: str, text: str) -> list[Record]:
    """Writes the facts of an observation log, for a game that was recorded without a ledger.

    Args:
        ledger: The ledger to write to.
        agent: The agent whose log it is.
        text: The content of its `action.log`.

    Returns:
        The facts, one for each tick.
    """
    rooms, beside = parse_observations(text)
    adjacent = parse_adjacent(text)
    return [
        record_observation(ledger, agent, r, t, room, beside.get((r, t), set()), adjacent.get((r, t), {}))
        for r, t, room in rooms
    ]


def record_statement(
    ledger: Ledger,
    author: str,
    text: str,
    end_s: float,
    roster: Sequence[str],
    round_num: int,
    rooms: Sequence[str] = ROOM_NAMES,
) -> list[Record]:
    """Writes the claims and the accusations of one statement, readable by everyone in the meeting.

    Args:
        ledger: The ledger of the game.
        author: Who spoke.
        text: The words that were spoken.
        end_s: Room time at which the last word was spoken, from which a listener has heard all of it.
        roster: Agents in the meeting.
        round_num: Round of the meeting.
        rooms: Room names to look for.

    Returns:
        The records, which are empty when the statement says nothing about places or other agents.
    """
    records = []
    for claim in extract_claims(author, text, rooms):
        records.append(
            ledger.append(
                CLAIM, author, {"kind": claim.kind, "room": claim.room}, round_num=round_num, time_s=end_s,
                subject=claim.subject, visible_to=roster,
            )
        )
    for edge in extract_edges(author, text, 0):
        records.append(
            ledger.append(
                CLAIM, author, {"kind": edge.kind, "room": None}, round_num=round_num, time_s=end_s,
                subject=edge.target, visible_to=roster,
            )
        )
    return records


def retract(ledger: Ledger, records: Sequence[Record], roster: Sequence[str]) -> Record | None:
    """Corrects claims by appending a note that cites them, because a record is never removed.

    Args:
        ledger: The ledger of the game.
        records: The claims to withdraw.
        roster: Agents in the meeting.

    Returns:
        The note, or None when there was nothing to withdraw.
    """
    if not records:
        return None
    return ledger.append(NOTE, "engine", {"retracts": True}, refs=[r.id for r in records], visible_to=roster)


def _sightings(view: Sequence[Record], reader: str, round_num: int) -> tuple[list[str], dict[str, list[tuple[int, int, str, bool]]]]:
    """Collects the rooms the reader was in and where it saw each other agent, up to a round.

    Args:
        view: The records the reader may read.
        reader: The reader.
        round_num: Last round to count.

    Returns:
        The rooms the reader visited in order without repeats, and for each agent a list of
        (round, tick, room, in an adjacent room).
    """
    visited: list[str] = []
    seen: dict[str, list[tuple[int, int, str, bool]]] = defaultdict(list)
    for fact in view:
        if fact.type != FACT or fact.subject != reader or fact.round_num is None or fact.round_num > round_num:
            continue
        room = fact.content["room"]
        if room not in visited:
            visited.append(room)
        for agent in fact.content["beside"]:
            seen[agent].append((fact.round_num, fact.tick or 0, room, False))
        for neighbour, agents in fact.content["adjacent"].items():
            for agent in agents:
                seen[agent].append((fact.round_num, fact.tick or 0, neighbour, True))
    return visited, seen


def _join(items: Sequence[str]) -> str:
    """Joins names with commas and an `and`."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _evidence(subject: str, room: str | None, reader: str, visited: list[str], seen: dict) -> str:
    """Says what the reader saw that bears on a claim about an agent and, when given, a room.

    Args:
        subject: The agent the claim is about.
        room: The room the claim names, or None.
        reader: The reader.
        visited: Rooms the reader was in.
        seen: Sightings of the other agents.

    Returns:
        A sentence that starts with a space, or an empty string when the reader has nothing to say.
    """
    if subject == reader:
        if not visited:
            return ""
        if room is None:
            return f" You were in {_join(visited)}."
        return " You were there." if room in visited else f" You were in {_join(visited)}, never in {room}."
    sights = seen.get(subject, [])
    if not sights:
        return f" You did not see {subject}."
    places = list(dict.fromkeys(s[2] for s in sights))
    if room is None:
        return f" You saw {subject} in {_join(places)}."
    for round_num, tick, where, adjacent in sights:
        if where == room:
            return f" You saw {subject} {'next door' if adjacent else 'there'} (round {round_num}, tick {tick})."
    return f" You saw {subject} only in {_join(places)}, never in {room}."


def _line(claim: Record, reader: str, visited: list[str], seen: dict) -> str:
    """Writes the board line of one claim about a place."""
    kind, room = claim.content["kind"], claim.content["room"]
    who = "you" if claim.subject == reader else claim.subject
    if kind == "self":
        said = f"{claim.author} said it was in {room}."
    elif kind == "seen":
        said = f"{claim.author} said it saw {who} in {room}."
    else:
        said = f"{claim.author} said it was with {who}" + (f" in {room}." if room else ".")
    # A claim of one's own whereabouts is tested against what the reader saw of the speaker.
    about = claim.author if kind == "self" else claim.subject
    return said + _evidence(about, room, reader, visited, seen)


def render_board(ledger: Ledger, reader: str, t: float, round_num: int) -> str:
    """Builds the board a reader sees at a time in a meeting.

    Args:
        ledger: The ledger of the game.
        reader: The agent who reads.
        t: Room time. Only statements that were fully spoken by then count.
        round_num: Round of the meeting.

    Returns:
        The board as text, or an empty string when no statement has been fully spoken. It lists at most
        `MAX_SUBJECTS` agents with `MAX_LINES` lines each.
    """
    view = ledger.view(reader)
    withdrawn = {ref for r in view if r.type == NOTE and r.content.get("retracts") for ref in r.refs}
    claims = [
        r for r in view
        if r.type == CLAIM and r.round_num == round_num and r.id not in withdrawn and r.author != reader
        and r.time_s is not None and r.time_s <= t + 1e-9
    ]
    if not claims:
        return ""
    visited, seen = _sightings(view, reader, round_num)
    by_subject: dict[str, list[Record]] = defaultdict(list)
    for claim in claims:
        by_subject[claim.subject].append(claim)
    order = sorted(by_subject, key=lambda s: (-len(by_subject[s]), s == reader, s))[:MAX_SUBJECTS]
    lines = [HEADER]
    for subject in order:
        group = by_subject[subject]
        places = list(dict.fromkeys(
            (c.author, c.content["kind"], c.content["room"], c.subject) for c in group if c.content["kind"] in ("self", "seen", "with")
        ))
        entries = []
        for author, kind, room, about in places:
            claim = next(c for c in group if (c.author, c.content["kind"], c.content["room"], c.subject) == (author, kind, room, about))
            entries.append(_line(claim, reader, visited, seen))
        accused = list(dict.fromkeys(c.author for c in group if c.content["kind"] == "accuse"))
        vouched = list(dict.fromkeys(c.author for c in group if c.content["kind"] == "trust"))
        if accused:
            entries.append(f"Accused by {_join(accused)}.")
        if vouched:
            entries.append(f"Vouched for by {_join(vouched)}.")
        lines.append("You" if subject == reader else subject)
        lines.extend(f"  {entry}" for entry in entries[:MAX_LINES])
        if len(entries) > MAX_LINES:
            lines.append(f"  and {len(entries) - MAX_LINES} more.")
    return "\n".join(lines)
