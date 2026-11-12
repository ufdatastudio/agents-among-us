"""Reads claims, accusations and observations out of text, for the room and for the analysis.

A statement in a meeting says where agents were (`Claim`) and whom the speaker accuses or vouches for (`Edge`).
An agent's observation log says where it was and whom it saw there. The board of the room checks the first
against the second, and `scripts/analyze_p2_text.py` checks claims against the logs of every agent.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

AGENT = re.compile(r"\bAgents?[_ ]?(\d+)\b", re.IGNORECASE)
LOCATION = re.compile(r"^Current Location: \[(?P<room>[^\]]+)\]")
ADJACENT = re.compile(r"^\s+\[(?P<room>[^\]]+)\] -> Occupants: (?P<names>.*)$")
OCCUPANTS = re.compile(r"^\s*-> Occupants: (?P<names>.*)$")
ROUND_HEADER = re.compile(r"^Round (?P<round>\d+)/\d+")
SENTENCE = re.compile(r"[.!?;\n]+")
CLAUSE = re.compile(r",|\bbut\b|\bwhile\b|\band then\b|\bhowever\b|\bthough\b|\bsince\b|\bbecause\b", re.IGNORECASE)
SUSPICION = re.compile(
    r"suspicious|suspect|\bsus\b|\blying\b|\blie[sd]?\b|\bliar\b|impostor|imposter|byzantine|killer|guilty|shady|fishy|"
    r"dubious|doubt|distrust|(?:do not|don't|dont|not) trust|unfounded|blame|accus|culprit|fram(?:e|ing)|too convenient|"
    r"red flag|inconsistent|scrutini[sz]e|investigate|\bvote\b|\beject|don't buy|do not buy|too polished|too quick|"
    r"unaccounted|unexplained|hiding|avoiding|flipp?ing|lingering|linger|convenient|evasive|defensive|nervous|sketchy|"
    r"dodgy|contradict|add(?:s|ing)? up|likely (?:culprit|impostor|suspect)|most likely|vague|odd(?:ly)?\b|weird|strange|"
    r"unclear|shaky|weak|foggy|premature|reckless|concern(?:ing|ed)|gaps?\b",
    re.IGNORECASE,
)
NO_SUSPICION = re.compile(
    r"(?:\b(?:no|nothing|nobody|not|never|none|any|anything|anyone)\b|n't)\W+(?:\w+\W+){0,3}?(?:suspicious|unusual|wrong|odd|strange|shady)|"
    r"no reason to (?:doubt|suspect|distrust)|not (?:a )?(?:threat|suspect)|(?:cleared|innocent)",
    re.IGNORECASE,
)
TRUST = re.compile(r"\btrust|vouch|confirm|checks out|alibi holds|can attest|corroborate|innocent", re.IGNORECASE)
FIRST_PERSON = re.compile(r"\b(?:i|i've|i'm|i'd|i'll)\b", re.IGNORECASE)
SECOND_PERSON = re.compile(r"\b(?:you|your|you're|you've)\b", re.IGNORECASE)
PRESENCE = re.compile(
    r"\b(?:was|were|am|is|stayed|stood|moved|went|headed|walked|passed|came|ended up|lingered|waiting|"
    r"been|found|saw|seen|noticed|spotted|observed|with)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:not|never|no|nobody|didn't|haven't|wasn't|weren't|hadn't|couldn't)\b|n't\b", re.IGNORECASE)
VICTIM = re.compile(
    r"(?:taking out|killing|killed|eliminat\w*|tagged|body of|victim|disappearance of|death of|demise of|found|reported|report on|report of)"
    r"\W+(?:agents?[_ ]?)?(?P<before>\d+)|"
    r"agents?[_ ]?(?P<after>\d+)(?:'s|\u2019s)\s+(?:body|elimination|disappearance|death|demise|corpse)",
    re.IGNORECASE,
)
PRONOUN_OR_DEMONSTRATIVE = re.compile(r"\b(?:they|them|their|he|him|his|she|her|that|this|it)\b", re.IGNORECASE)
ADDRESS = re.compile(r"^\s*(?:agents?[_ ]?(?P<n>\d+)\s*[,:\-]+\s*)+", re.IGNORECASE)
PRONOUN = re.compile(r"\b(?:they|them|their|he|him|his|she|her)\b", re.IGNORECASE)
SIGHTING = re.compile(r"\b(?:saw|seen|noticed|spotted|observed|found)\b", re.IGNORECASE)
# Words after which a name in the same sentence is no longer a partner of "with".
BEYOND_PARTNERS = re.compile(r"\b(?:while|when|but|then|after|before|until|because|since|as|where|and saw|and noticed)\b", re.IGNORECASE)
BARE_NUMBER = re.compile(r"\b(?:vote|eject|get|target)\s+(?:for\s+)?(?:agents?[_ ]?)?(\d)\b", re.IGNORECASE)


def agent_names(text: str) -> list[str]:
    """Finds the agents a text mentions, in order and without repeats.

    Args:
        text: A statement or a part of one.

    Returns:
        Names such as `Agent_3`. The spellings `Agent 3`, `Agents_3` and `agent3` give the same name, and so does
        a bare number after `vote`, `eject`, `get` or `target`, as in "vote 8".
    """
    found = sorted([(m.start(), m.group(1)) for m in AGENT.finditer(text)] + [(m.start(1), m.group(1)) for m in BARE_NUMBER.finditer(text)])
    return list(dict.fromkeys(f"Agent_{n}" for _, n in found))


def _room_pattern(room: str) -> re.Pattern:
    """Builds a pattern for a room name that also matches its spaced spelling, as in `Upper Engine`."""
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\s?", re.escape(room).replace(r"\ ", r"\s?"))
    return re.compile(rf"\b{spaced}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    """A claim about where agents were, taken from one sentence.

    Attributes:
        kind: `self` (the speaker was in a room), `seen` (another agent was in a room) or `with` (the speaker
            was together with another agent).
        speaker: Who made the claim.
        subject: Agent the claim is about. For `with` it is the partner.
        room: Room named in the sentence, or None.
    """

    kind: str
    speaker: str
    subject: str
    room: str | None


def extract_claims(speaker: str, text: str, rooms: Sequence[str]) -> list[Claim]:
    """Finds the claims about places in a statement.

    A sentence with a negation gives no claim, because the pattern cannot tell what is denied. A sentence
    that addresses someone as "you" gives no claim about the speaker, because "you were in Admin with me"
    is about the listener.

    Args:
        speaker: Who spoke.
        text: What was said.
        rooms: Room names of the game.

    Returns:
        The claims, in order.
    """
    patterns = [(room, _room_pattern(room)) for room in rooms]
    claims: list[Claim] = []
    for sentence in SENTENCE.split(text):
        if not sentence.strip() or NEGATION.search(sentence) or not PRESENCE.search(sentence):
            continue
        named = [room for room, pattern in patterns if pattern.search(sentence)]
        first = bool(FIRST_PERSON.search(sentence)) and not SECOND_PERSON.search(sentence)
        sighting = SIGHTING.search(sentence)
        if first and named and not sighting:
            claims.extend(Claim("self", speaker, speaker, room) for room in named)
        together = re.search(r"\bwith\b(?P<rest>.*)", sentence, re.IGNORECASE)
        # "I saw Agent_5 with Agent_7" says where others were, so only "I was with ..." is an alibi.
        if first and together and not (sighting and sighting.start() < together.start()):
            partners = agent_names(BEYOND_PARTNERS.split(together["rest"])[0])
            claims.extend(Claim("with", speaker, n, named[0] if len(named) == 1 else None) for n in partners if n != speaker)
        if sighting:
            after = sentence[sighting.end() :]
            where = [room for room, pattern in patterns if pattern.search(after)]
            who = [n for n in agent_names(BEYOND_PARTNERS.split(after)[0]) if n != speaker]
            if where and who:
                where = where[:1]
                claims.extend(Claim("seen", speaker, n, room) for n in who for room in where)
            elif first and where and not who and not claims:
                claims.extend(Claim("self", speaker, speaker, room) for room in where[:1])
        elif named and not first:
            names = [n for n in agent_names(sentence) if n != speaker]
            if len(names) == 1:
                claims.append(Claim("seen", speaker, names[0], named[0]))
    return list(dict.fromkeys(claims))


@dataclass(frozen=True)
class Edge:
    """A statement about another agent.

    Attributes:
        kind: `accuse` or `trust`.
        speaker: Who spoke.
        target: Who the statement is about.
        statement: Index of the statement in the game.
    """

    kind: str
    speaker: str
    target: str
    statement: int


def _nearest(clause: str, cue: re.Match, names: Sequence[str]) -> str:
    """Picks the name a cue in a clause is about, which is the nearest one and the one after it on a tie."""
    spots = [(abs(m.start() - cue.start()) - (0.5 if m.start() > cue.start() else 0), f"Agent_{m.group(1)}") for m in AGENT.finditer(clause)]
    return min(spots)[1] if spots else names[0]


def extract_edges(speaker: str, text: str, statement: int) -> list[Edge]:
    """Finds who a statement accuses and who it vouches for.

    The text is read a clause at a time. A clause with a suspicion cue accuses the agent named nearest to the
    cue, or the agent addressed at the start of the statement when the clause speaks to "you". A clause with a
    trust cue vouches for every agent it names. A clause that denies any suspicion, such as "nothing
    suspicious", accuses nobody. The speaker's own name and the victims of the round are left out.

    Args:
        speaker: Who spoke.
        text: What was said.
        statement: Index of the statement in the game.

    Returns:
        The edges.
    """
    edges: list[Edge] = []
    opening = ADDRESS.match(text)
    addressee = f"Agent_{opening['n']}" if opening else None
    text = text[opening.end() :] if opening else text
    carried: list[str] = []
    for sentence in SENTENCE.split(text):
        clauses = CLAUSE.split(sentence)
        for clause in clauses:
            victims = {f"Agent_{m['before'] or m['after']}" for m in VICTIM.finditer(clause)}
            mentioned = agent_names(clause)
            names = [n for n in mentioned if n != speaker and n not in victims]
            if mentioned and not names:
                # The clause only names the speaker or a victim, so a pronoun in it does not point at anyone else.
                continue
            cue = SUSPICION.search(clause)
            suspicious = cue and not NO_SUSPICION.search(clause)
            distrust = re.search(r"(?:do not|don't|dont|not|never)\s+trust", clause, re.IGNORECASE)
            if not names and suspicious and SECOND_PERSON.search(clause) and addressee and addressee != speaker:
                edges.append(Edge("accuse", speaker, addressee, statement))
                continue
            if not names or (PRONOUN.search(clause) and suspicious and carried):
                # "I find that suspicious" and "they look guilty" refer to the agent named before them.
                names = carried if suspicious and PRONOUN_OR_DEMONSTRATIVE.search(clause) else []
            else:
                carried = names
            if not names:
                continue
            if NO_SUSPICION.search(clause):
                if TRUST.search(clause):
                    edges.extend(Edge("trust", speaker, n, statement) for n in names)
            elif cue and (distrust or not re.search(r"\btrust\b", clause, re.IGNORECASE)):
                target = _nearest(clause, cue, names)
                edges.append(Edge("accuse", speaker, target if target != speaker else names[0], statement))
            elif TRUST.search(clause):
                edges.extend(Edge("trust", speaker, n, statement) for n in names)
    return list(dict.fromkeys(edges))


def parse_observations(text: str) -> tuple[list[tuple[int, int, str]], dict[tuple[int, int], set[str]]]:
    """Reads where an agent was at each tick, and whom it saw there, from its observation log.

    Args:
        text: The content of the `action.log` of one agent.

    Returns:
        The rooms as (round, tick, room), and the agents seen in the same room at each (round, tick).
    """
    rooms: list[tuple[int, int, str]] = []
    beside: dict[tuple[int, int], set[str]] = {}
    current_round, tick = 0, 0
    for line in text.splitlines():
        header = ROUND_HEADER.match(line)
        if header:
            current_round, tick = int(header["round"]), 0
            continue
        place = LOCATION.match(line)
        if place:
            tick += 1
            rooms.append((current_round, tick, place["room"]))
            continue
        seen = OCCUPANTS.match(line)
        if seen and rooms and rooms[-1][:2] == (current_round, tick) and (current_round, tick) not in beside:
            beside[(current_round, tick)] = set(agent_names(seen["names"]))
    return rooms, beside


def parse_adjacent(text: str) -> dict[tuple[int, int], dict[str, set[str]]]:
    """Reads whom an agent saw in the rooms next to it at each tick, from its observation log.

    Args:
        text: The content of the `action.log` of one agent.

    Returns:
        For each (round, tick), the agents seen in each adjacent room.
    """
    seen: dict[tuple[int, int], dict[str, set[str]]] = {}
    current_round, tick = 0, 0
    for line in text.splitlines():
        header = ROUND_HEADER.match(line)
        if header:
            current_round, tick = int(header["round"]), 0
        elif LOCATION.match(line):
            tick += 1
        else:
            neighbour = ADJACENT.match(line)
            if neighbour and tick:
                seen.setdefault((current_round, tick), {})[neighbour["room"]] = set(agent_names(neighbour["names"]))
    return seen
