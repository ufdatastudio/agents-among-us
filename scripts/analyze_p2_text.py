"""Reads the transcripts of P2 games and measures what the agents say, hear and believe.

Every game folder under `logs/` holds the discussion, each agent's own observation log (where it was
and who it saw), the votes, and in a timed room the utterances with their ticks. The observation logs
give the ground truth, so a claim such as "I was in Storage with Agent_0" can be checked.

Usage:
    python -m scripts.analyze_p2_text logs/p2_ibm-granite_granite-3.3-8b-instruct
"""

import argparse
import ast
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SPACES = {0x202F: " ", 0x00A0: " ", 0x2009: " ", 0x2007: " ", 0x200B: ""}
GAME_DIR = re.compile(r"Game_(?P<tag>.+?)_(?P<protocol>sequential|none|(?:clean|interleaved|noisy)-\d+)_(?P<seed>\d+)_Run0$")
AGENT = re.compile(r"\bAgents?[_ ]?(\d+)\b", re.IGNORECASE)
LOCATION = re.compile(r"^Current Location: \[(?P<room>[^\]]+)\]")
OCCUPANTS = re.compile(r"^\s*-> Occupants: (?P<names>.*)$")
ROUND_HEADER = re.compile(r"^Round (?P<round>\d+)/\d+")
VOTE_LINE = re.compile(r"^Round (?P<round>\d+): Voted for (?P<target>\S+)")
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
VOTE_CALL = re.compile(r"\bvote\b|\beject", re.IGNORECASE)
AGREE = re.compile(r"\bi agree|agree with|same here|good point|you(?:'re| are) right|exactly|seconded|second that|concur", re.IGNORECASE)
DENY = re.compile(
    r"\bi (?:didn't|did not|wasn't|was not|haven't|have not|am not|never)\b|not me\b|i'm innocent|i swear|"
    r"your accusation|unfounded|i have no reason|i have nothing to hide",
    re.IGNORECASE,
)
FIRST_PERSON = re.compile(r"\b(?:i|i've|i'm|i'd|i'll)\b", re.IGNORECASE)
SECOND_PERSON = re.compile(r"\b(?:you|your|you're|you've)\b", re.IGNORECASE)
PRESENCE = re.compile(
    r"\b(?:was|were|am|is|stayed|stood|moved|went|headed|walked|passed|came|ended up|lingered|waiting|"
    r"been|found|saw|seen|noticed|spotted|observed|with)\b",
    re.IGNORECASE,
)
NEGATION = re.compile(r"\b(?:not|never|no|nobody|didn't|haven't|wasn't|weren't|hadn't|couldn't)\b|n't\b", re.IGNORECASE)
CHALLENGE = re.compile(r"clarify|explain|account for|where were you|whereabouts|need (?:more )?(?:transparency|evidence|alibis?)|prove|demand", re.IGNORECASE)
REDIRECT = re.compile(r"what about|focus on|look at|look into|keep (?:an )?eyes?|we should (?:focus|look|check|start)", re.IGNORECASE)
WORD = re.compile(r"[a-z']+")
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


@dataclass
class Statement:
    """One thing an agent said in a meeting.

    Attributes:
        meeting: Round of the meeting.
        index: Position in the meeting, from 0.
        speaker: Agent name.
        role: `honest` or `byzantine`.
        text: What was said.
        tick: Tick of a timed room at which the agent decided to speak, None in turn-taking.
        cursor: Room time up to which the speaker had heard, None in turn-taking.
        start_s: Room time at which the first word began, None in turn-taking.
        intended: Words the agent wanted to say, None in turn-taking.
        spoken: Words that were kept, None in turn-taking.
    """

    meeting: int
    index: int
    speaker: str
    role: str
    text: str
    tick: int | None = None
    cursor: float | None = None
    start_s: float | None = None
    intended: int | None = None
    spoken: int | None = None


@dataclass
class Game:
    """Everything one game folder says.

    Attributes:
        game_id: Folder name without `Game_` and `_Run0`.
        model: Model the agents ran.
        protocol: Discussion protocol.
        seed: Scenario seed.
        roles: Role of every agent.
        statements: Every statement in order of meeting and position.
        votes: Vote of each agent in each meeting, by round.
        active: Agents that could vote in each meeting, by round.
        ejected: Agent ejected after each meeting, by round.
        rooms: Room each agent was in at each tick, as (round, tick, room).
        beside: Agents each agent saw in the same room at each tick, by agent and (round, tick).
    """

    game_id: str
    model: str
    protocol: str
    seed: int
    roles: dict[str, str]
    statements: list[Statement] = field(default_factory=list)
    votes: dict[int, dict[str, str]] = field(default_factory=dict)
    active: dict[int, set[str]] = field(default_factory=dict)
    ejected: dict[int, str | None] = field(default_factory=dict)
    rooms: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    beside: dict[str, dict[tuple[int, int], set[str]]] = field(default_factory=dict)

    def meetings(self) -> list[int]:
        """Lists the rounds that had a meeting with statements."""
        return sorted({s.meeting for s in self.statements})


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


def parse_action_log(path: Path) -> tuple[list[tuple[int, int, str]], dict[tuple[int, int], set[str]]]:
    """Reads where an agent was at each tick, and whom it saw there.

    Args:
        path: The `action.log` of one agent.

    Returns:
        The rooms as (round, tick, room), and the agents seen in the same room at each (round, tick).
    """
    rooms: list[tuple[int, int, str]] = []
    beside: dict[tuple[int, int], set[str]] = {}
    current_round, tick = 0, 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
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


def parse_results(path: Path, roster: Sequence[str]) -> tuple[dict[int, set[str]], dict[int, str | None]]:
    """Reads who could vote in each meeting and who was ejected.

    Args:
        path: The `roundResults.log` of the game.
        roster: Every agent in the game.

    Returns:
        The agents active during each meeting, and the agent ejected after it, by round.
    """
    active: dict[int, set[str]] = {}
    ejected: dict[int, str | None] = {}
    status = {name: "active" for name in roster}
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    parts = re.split(r"^=== Round (\d+) ===$", text, flags=re.MULTILINE)
    for number, body in zip(parts[1::2], parts[2::2]):
        round_num = int(number)
        killed = set(re.findall(r"ELIMINATION: (Agent_\d+)", body))
        active[round_num] = {n for n, s in status.items() if s == "active"} - killed
        eject = re.search(r"EJECTION: (Agent_\d+)", body)
        ejected[round_num] = eject.group(1) if eject else None
        snapshot = re.search(r"Player Statuses: (\{.*\})", body)
        if snapshot:
            status = ast.literal_eval(snapshot.group(1))
    return active, ejected


def parse_game(path: Path) -> Game | None:
    """Reads one game folder.

    Args:
        path: A `Game_<id>_Run0` folder.

    Returns:
        The game, or None when the folder is not a P2 game or has no discussion.
    """
    named = GAME_DIR.search(path.name)
    chat = path / "discussion_chat.csv"
    if not named or not chat.exists():
        return None
    roles = {p.name: "byzantine" for p in (path / "Byz").glob("Agent_*")} | {p.name: "honest" for p in (path / "Honest").glob("Agent_*")}
    with chat.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    game = Game(
        game_id=f"{named['tag']}_{named['protocol']}_{named['seed']}",
        model=rows[0]["model"] if rows else "",
        protocol=named["protocol"],
        seed=int(named["seed"]),
        roles=roles,
    )
    counters: Counter = Counter()
    by_round: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_round[int(row["discussion_num"])].append(row)
    for round_num, items in sorted(by_round.items()):
        timed = _utterances(path / "room" / f"room_round{round_num}.jsonl")
        for row in items:
            agent = f"Agent_{row['agent_num']}"
            # Some models put a no-break space inside "Agent 5", which would hide the name.
            text = row["message"].translate(SPACES).strip()
            if not text:
                continue
            index = counters[round_num]
            counters[round_num] += 1
            match = next((u for u in timed if u["author"] == agent and u["text"].replace("\n", " ") == " ".join(text.split())), None)
            if match is not None:
                timed.remove(match)
            game.statements.append(
                Statement(
                    round_num, index, agent, roles.get(agent, "honest" if row["role"] == "Honest" else "byzantine"), text,
                    tick=match["tick"] if match else None, cursor=match["cursor"] if match else None,
                    start_s=match["start_s"] if match else None, intended=match["intended_words"] if match else None,
                    spoken=match["spoken_words"] if match else None,
                )
            )
    for agent in roles:
        folder = path / ("Byz" if roles[agent] == "byzantine" else "Honest") / agent
        if (folder / "action.log").exists():
            game.rooms[agent], game.beside[agent] = parse_action_log(folder / "action.log")
        if (folder / "vote.log").exists():
            for line in (folder / "vote.log").read_text(encoding="utf-8", errors="ignore").splitlines():
                found = VOTE_LINE.match(line)
                if found:
                    game.votes.setdefault(int(found["round"]), {})[agent] = found["target"]
    game.active, game.ejected = parse_results(path / "roundResults.log", sorted(roles))
    return game


def _utterances(path: Path) -> list[dict]:
    """Reads the utterance rows of a room record, or an empty list when the game had no room."""
    if not path.exists():
        return []
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return [row for row in rows if row.get("type") == "utterance" and row.get("text")]


def room_names(game: Game) -> list[str]:
    """Lists the rooms that appear in the game's observation logs, longest name first."""
    return sorted({room for rooms in game.rooms.values() for _, _, room in rooms}, key=len, reverse=True)


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


def verify_claim(game: Game, claim: Claim, meeting: int) -> bool | None:
    """Checks a claim against the observation logs, counting the rounds up to the meeting.

    A claim is true when it held at some tick, so the test only rejects claims that never held.

    Args:
        game: The game.
        claim: The claim.
        meeting: Round of the meeting in which it was made.

    Returns:
        True or False, or None when the logs of the agents involved are missing.
    """
    own = {(r, t): room for r, t, room in game.rooms.get(claim.speaker, []) if r <= meeting}
    if claim.kind == "self":
        return None if not own else claim.room in own.values()
    other = {(r, t): room for r, t, room in game.rooms.get(claim.subject, []) if r <= meeting}
    if claim.kind == "seen":
        if not other:
            return None
        return claim.room in other.values()
    if not own or not other:
        return None
    together = [key for key in own if other.get(key) == own[key] and (claim.room is None or own[key] == claim.room)]
    return bool(together)


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


def label_statement(text: str) -> set[str]:
    """Gives the kinds of speech act in a statement.

    Args:
        text: What was said.

    Returns:
        Labels from `accuse`, `vote_call`, `defend`, `report` (the speaker says what it saw or where it was),
        `vouch`, `agree`, `question`, `redirect`, `challenge` (a demand to explain or prove) and `other`. A statement can have several.
    """
    labels = set()
    for sentence in SENTENCE.split(text):
        if SUSPICION.search(sentence) and not NO_SUSPICION.search(sentence) and agent_names(sentence):
            labels.add("accuse")
        if VOTE_CALL.search(sentence) and agent_names(sentence):
            labels.add("vote_call")
        if DENY.search(sentence):
            labels.add("defend")
        if FIRST_PERSON.search(sentence) and re.search(r"\b(?:saw|noticed|found|was in|was at|was with|moved|went|headed|stayed)\b", sentence, re.IGNORECASE):
            labels.add("report")
        if TRUST.search(sentence) and agent_names(sentence) and not re.search(r"(?:not|n't) trust", sentence, re.IGNORECASE):
            labels.add("vouch")
        if AGREE.search(sentence):
            labels.add("agree")
        if REDIRECT.search(sentence) and agent_names(sentence):
            labels.add("redirect")
        if CHALLENGE.search(sentence):
            labels.add("challenge")
    if "?" in text:
        labels.add("question")
    return labels or {"other"}


def words(text: str) -> set[str]:
    """Gives the set of lower-case words of a text."""
    return set(WORD.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Gives the share of words two sets have in common."""
    return len(a & b) / len(a | b) if a | b else 0.0


def family(protocol: str) -> str:
    """Names the group of a protocol, which is the overlap mode for a timed room."""
    return protocol if protocol in ("sequential", "none") else protocol.split("-")[0]


def rate(count: int, total: int) -> float | None:
    """Gives count over total, or None for an empty total."""
    return count / total if total else None


def speech_profile(games: Iterable[Game]) -> dict[tuple[str, str, str], dict]:
    """Measures what each role says, for each model and group of protocols.

    Args:
        games: Parsed games.

    Returns:
        For each (model, group, role) the number of statements, mean words, the share of statements with each
        label, repetition of the speaker's own earlier statements, and the share cut at the word limit.
    """
    stats: dict[tuple[str, str, str], dict] = defaultdict(lambda: {"n": 0, "words": [], "labels": Counter(), "repeat": 0, "cut": 0, "timed": 0, "tagged": 0})
    for game in games:
        earlier: dict[tuple[int, str], list[set[str]]] = defaultdict(list)
        for s in game.statements:
            cell = stats[(game.model, family(game.protocol), s.role)]
            cell["n"] += 1
            cell["words"].append(len(s.text.split()))
            cell["labels"].update(label_statement(s.text))
            mine = words(s.text)
            cell["repeat"] += any(jaccard(mine, old) >= 0.6 for old in earlier[(s.meeting, s.speaker)])
            earlier[(s.meeting, s.speaker)].append(mine)
            if s.intended is not None:
                cell["timed"] += 1
                cell["cut"] += s.intended > (s.spoken or 0)
            cell["tagged"] += bool(re.match(r"^\s*Agent_\d+\s*,", s.text))
    return stats


def claim_profile(games: Iterable[Game]) -> dict[tuple[str, str, str], dict]:
    """Checks the claims about places, for each model, group and role.

    Args:
        games: Parsed games.

    Returns:
        For each (model, group, role) the claims that could be checked, how many were false, and the same
        counts for the claims that repeat a place or a partner an earlier speaker had already claimed.
    """
    stats: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for game in games:
        rooms = room_names(game)
        seen_before: dict[int, set[tuple]] = defaultdict(set)
        for s in game.statements:
            for claim in extract_claims(s.speaker, s.text, rooms):
                verdict = verify_claim(game, claim, s.meeting)
                if verdict is None:
                    continue
                key = (claim.kind, claim.room) if claim.kind != "with" else (claim.kind, claim.subject)
                echoed = key in seen_before[s.meeting]
                cell = stats[(game.model, family(game.protocol), s.role)]
                cell["claims"] += 1
                cell["false"] += not verdict
                cell["echoed"] += echoed
                cell["echoed_false"] += echoed and not verdict
                cell["fresh_false"] += (not echoed) and not verdict
                cell["fresh"] += not echoed
                cell[f"{claim.kind}_claims"] += 1
                cell[f"{claim.kind}_false"] += not verdict
            for claim in extract_claims(s.speaker, s.text, rooms):
                seen_before[s.meeting].add((claim.kind, claim.room) if claim.kind != "with" else (claim.kind, claim.subject))
    return stats


def accusation_profile(games: Iterable[Game]) -> dict[tuple[str, str, str], Counter]:
    """Counts whom each role accuses, for each model, group and role.

    Args:
        games: Parsed games.

    Returns:
        For each (model, group, role) the accusations, the share aimed at Byzantines, the share aimed at a
        teammate, and the share that a random pick among the other active agents would have hit.
    """
    stats: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for game in games:
        for s in game.statements:
            cell = stats[(game.model, family(game.protocol), s.role)]
            active = game.active.get(s.meeting, set(game.roles)) - {s.speaker}
            for edge in extract_edges(s.speaker, s.text, s.index):
                if edge.target not in game.roles or edge.target not in active | {edge.target}:
                    continue
                if edge.kind == "trust":
                    cell["trust"] += 1
                    cell["trust_byz"] += game.roles[edge.target] == "byzantine"
                    continue
                cell["accuse"] += 1
                cell["at_byz"] += game.roles[edge.target] == "byzantine"
                cell["chance_byz"] += sum(game.roles[a] == "byzantine" for a in active) / max(1, len(active)) if active else 0
    return stats


def influence_profile(games: Iterable[Game]) -> dict[tuple[str, str], Counter]:
    """Relates each honest vote to the accusations the voter heard.

    For every honest voter and every candidate the voter could choose, the candidate is placed in one cell by
    who accused them in that meeting (other than the voter and the candidate): nobody, only honest agents, only
    Byzantine agents, or both. The cell counts the pairs and the votes the candidate got from this voter.

    Args:
        games: Parsed games.

    Returns:
        For each (model, group) a counter with `<candidate role>|<accusers>|pairs` and `...|votes` keys, and
        `plurality_votes` and `honest_votes` for the share of votes that went to the most accused agent.
    """
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for game in games:
        cell = stats[(game.model, family(game.protocol))]
        for meeting in game.meetings():
            accusers: dict[str, set[str]] = defaultdict(set)
            counts: Counter = Counter()
            for s in (s for s in game.statements if s.meeting == meeting):
                for edge in extract_edges(s.speaker, s.text, s.index):
                    if edge.kind == "accuse":
                        accusers[edge.target].add(s.speaker)
                        counts[edge.target] += 1
            top = counts.most_common(2)
            plurality = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
            for voter, target in game.votes.get(meeting, {}).items():
                if game.roles.get(voter) != "honest" or target not in game.roles:
                    continue
                cell["honest_votes"] += 1
                if plurality is not None:
                    cell["plurality_meetings_votes"] += 1
                    cell["plurality_votes"] += target == plurality
                for candidate in game.active.get(meeting, set(game.roles)) - {voter}:
                    heard = accusers[candidate] - {voter, candidate}
                    by_byz = any(game.roles.get(a) == "byzantine" for a in heard)
                    by_honest = any(game.roles.get(a) == "honest" for a in heard)
                    group = "both" if by_byz and by_honest else "byzantine only" if by_byz else "honest only" if by_honest else "nobody"
                    key = f"{game.roles[candidate]}|{group}"
                    cell[f"{key}|pairs"] += 1
                    cell[f"{key}|votes"] += target == candidate
    return stats


def listening_profile(games: Iterable[Game]) -> dict[tuple[str, str], Counter]:
    """Measures what agents pick up from the room, for each model and group.

    Args:
        games: Parsed games.

    Returns:
        For each (model, group) the accusations made (`accused`), the accusations whose target spoke again in
        the meeting (`answerable`), the answers that name the accuser (`answered`), the statements made without
        having heard another statement of the same meeting (`unheard`) and the statements of a timed room.
    """
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for game in games:
        cell = stats[(game.model, family(game.protocol))]
        by_meeting: dict[int, list[Statement]] = defaultdict(list)
        for s in game.statements:
            by_meeting[s.meeting].append(s)
        for meeting, items in by_meeting.items():
            for s in items:
                for edge in extract_edges(s.speaker, s.text, s.index):
                    if edge.kind != "accuse":
                        continue
                    cell["accused"] += 1
                    later = [t for t in items if t.index > s.index and t.speaker == edge.target and (s.tick is None or (t.tick or 0) > s.tick)]
                    if later:
                        cell["answerable"] += 1
                        cell["answered"] += any(edge.speaker in agent_names(t.text) for t in later)
                if s.cursor is not None:
                    cell["timed_statements"] += 1
                    heard = [o for o in items if o is not s and o.start_s is not None and s.start_s is not None and o.start_s < s.cursor]
                    cell["heard_total"] += len(heard)
                    concurrent = [o for o in items if o is not s and o.tick == s.tick]
                    cell["concurrent_total"] += len(concurrent)
    return stats


def _fmt(value: float | None, spec: str = ".0%") -> str:
    return "-" if value is None else format(value, spec)


def short_model(model: str) -> str:
    """Shortens a model name for a table."""
    return model.split("/")[-1].removeprefix("vllm:")


def report(games: list[Game]) -> str:
    """Formats every measure of a set of games as Markdown tables.

    Args:
        games: Parsed games.

    Returns:
        The tables.
    """
    out = [f"{len(games)} games, {sum(len(g.statements) for g in games)} statements.", ""]
    profile = speech_profile(games)
    labels = ["accuse", "vote_call", "defend", "report", "vouch", "agree", "question", "redirect", "challenge", "other"]
    out += ["Share of statements with each kind of speech act, by role.", ""]
    out += ["| Model | Group | Role | Statements | Words | " + " | ".join(labels) + " | Repeats own | Cut at limit | Opens with a name |", "|" + "---|" * (len(labels) + 8)]
    for (model, group, role), c in sorted(profile.items()):
        out.append(
            f"| {short_model(model)} | {group} | {role} | {c['n']} | {statistics.mean(c['words']):.0f} | "
            + " | ".join(_fmt(rate(c["labels"][label], c["n"])) for label in labels)
            + f" | {_fmt(rate(c['repeat'], c['n']))} | {_fmt(rate(c['cut'], c['timed']))} | {_fmt(rate(c['tagged'], c['n']))} |"
        )
    out += ["", "Claims about places, checked against where each agent was. A claim is false when it never held in any earlier tick.", ""]
    out += ["| Model | Group | Role | Claims | False | Self | Seen | With | Echoes an earlier claim | False when echoed | False when new |", "|" + "---|" * 11]
    for (model, group, role), c in sorted(claim_profile(games).items()):
        out.append(
            f"| {short_model(model)} | {group} | {role} | {c['claims']} | {_fmt(rate(c['false'], c['claims']))} "
            f"| {_fmt(rate(c['self_false'], c['self_claims']))} | {_fmt(rate(c['seen_false'], c['seen_claims']))} | {_fmt(rate(c['with_false'], c['with_claims']))} "
            f"| {_fmt(rate(c['echoed'], c['claims']))} | {_fmt(rate(c['echoed_false'], c['echoed']))} | {_fmt(rate(c['fresh_false'], c['fresh']))} |"
        )
    out += ["", "Accusations by role, with the share a random pick would have aimed at a Byzantine.", ""]
    out += ["| Model | Group | Role | Accusations | At a Byzantine | By chance | Vouches | For a Byzantine |", "|" + "---|" * 8]
    for (model, group, role), c in sorted(accusation_profile(games).items()):
        out.append(
            f"| {short_model(model)} | {group} | {role} | {c['accuse']} | {_fmt(rate(c['at_byz'], c['accuse']))} "
            f"| {_fmt(rate(c['chance_byz'], c['accuse']))} | {c['trust']} | {_fmt(rate(c['trust_byz'], c['trust']))} |"
        )
    out += ["", "Honest votes by what the voter heard about the candidate. Each cell is the share of voter and candidate pairs that ended in a vote.", ""]
    cells = [(role, who) for role in ("honest", "byzantine") for who in ("nobody", "honest only", "byzantine only", "both")]
    out += ["| Model | Group | Votes | " + " | ".join(f"{r} candidate, accused by {w}" for r, w in cells) + " | Vote for the most accused |", "|" + "---|" * (len(cells) + 4)]
    for (model, group), c in sorted(influence_profile(games).items()):
        out.append(
            f"| {short_model(model)} | {group} | {c['honest_votes']} | "
            + " | ".join(f"{_fmt(rate(c[f'{r}|{w}|votes'], c[f'{r}|{w}|pairs']))} (n={c[f'{r}|{w}|pairs']})" for r, w in cells)
            + f" | {_fmt(rate(c['plurality_votes'], c['plurality_meetings_votes']))} |"
        )
    out += ["", "What agents pick up. An answer names the accuser in a later statement of the accused.", ""]
    out += ["| Model | Group | Accusations | Accused agent spoke later | Answer names the accuser | Heard per statement | Same tick, unheard |", "|" + "---|" * 7]
    for (model, group), c in sorted(listening_profile(games).items()):
        n = c["timed_statements"]
        out.append(
            f"| {short_model(model)} | {group} | {c['accused']} | {_fmt(rate(c['answerable'], c['accused']))} | {_fmt(rate(c['answered'], c['answerable']))} "
            f"| {_fmt(rate(c['heard_total'], n), '.1f')} | {_fmt(rate(c['concurrent_total'], n), '.1f')} |"
        )
    return "\n".join(out)


def load_games(folders: Iterable[Path]) -> list[Game]:
    """Reads every game folder under the given paths.

    Args:
        folders: Paths of game folders, or of folders that contain them.

    Returns:
        The games that parsed.
    """
    games = []
    for folder in folders:
        paths = [folder] if folder.name.endswith("_Run0") else sorted(folder.glob("Game_*_Run0"))
        games.extend(g for g in (parse_game(p) for p in paths) if g is not None)
    return games


def main(argv: Sequence[str] | None = None) -> None:
    """Prints the analysis of the games under the given folders.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(report(load_games(args.folders)))


if __name__ == "__main__":
    main()
