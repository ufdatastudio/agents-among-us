"""A recorded game as a timeline of agents walking the tile map of the Godot world.

The Python engine moves agents between rooms by name. This module reads the logs of a finished game and plans how each
agent would walk the tile map of `core/world_map.py` to reach each room: through doors and corridors, to a free spot of
the room. It follows the rules of `GameEngine.run_movement_phase`. At each tick a tag is carried out first, then a report
ends the round, and the moves of that tick are dropped when a report ends the round. After a report every agent that is
still in play walks to the Cafeteria for the meeting, votes are read out, and the agent with most votes is ejected.

The timeline is a function of video time. `Timeline.at(t)` returns what a frame shows, and nothing here draws.

Usage:
    from core.world_map import build_map
    replay = load_replay(Path("logs/p2_x/Game_tag_interleaved-180_3_Run0"))
    timeline = build_timeline(replay, build_map())
    frame = timeline.at(12.5)
"""

import csv
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import ROOMS
from core.claims import LOCATION, ROUND_HEADER
from core.world_map import FLOOR, PASSABLE, Room, WorldMap, display_name
from scripts.analyze_p2_text import parse_game

ACTION = re.compile(r"^Action (?P<n>\d+) Selected: (?P<verb>\w+) -> (?P<target>.+?)\s*$")
CAUSE = re.compile(r"^Discussion Started because (?P<reporter>Agent_\d+): Body Reported: (?P<body>Agent_\d+)\.")
WORDS_PER_S = 2.5
MEETING_ROOM = "Cafeteria"
SEAT_RADII = (5.6, 5.0)
SKIP = "SKIP"

Point = tuple[float, float]


@dataclass(frozen=True)
class Pacing:
    """How long each part of the video lasts, in video seconds.

    Attributes:
        round_card: Pause at the start of a round.
        kill: Time for the tags of a tick.
        settle: Pause before a walk.
        walk: Time agents take to walk to their next rooms in a tick.
        report: Time the report of a body is shown.
        gather: Time agents take to walk to the table.
        meeting_speed: Factor by which meeting time is played faster than it happened.
        min_meeting: Shortest meeting.
        per_vote: Time between two votes being shown.
        vote_hold: Time the tally stays after the last vote.
        eject: Time of the ejection.
        back: Time agents take to walk back to their rooms.
        final: Time the result card stays.
        linger: Time a caption stays after its speaker finished.
    """

    round_card: float = 1.2
    kill: float = 1.1
    settle: float = 0.3
    walk: float = 1.6
    report: float = 2.4
    gather: float = 3.0
    meeting_speed: float = 4.0
    min_meeting: float = 4.0
    per_vote: float = 0.4
    vote_hold: float = 1.6
    eject: float = 2.6
    back: float = 2.4
    final: float = 5.0
    linger: float = 2.0


@dataclass(frozen=True)
class Act:
    """What one agent did at one tick.

    Attributes:
        room: Room the agent was in when it chose.
        verb: `move`, `tag` or `report`.
        target: A room for a move, an agent for a tag, the body for a report.
    """

    room: str
    verb: str
    target: str


@dataclass(frozen=True)
class Spoken:
    """One statement of a meeting, in meeting time.

    Attributes:
        author: Who spoke.
        start_s: When the first word began.
        end_s: When the last word ended.
        text: What was said.
    """

    author: str
    start_s: float
    end_s: float
    text: str


@dataclass
class MeetingLog:
    """A meeting as it was recorded.

    Attributes:
        closed_at: Meeting time at which the room closed.
        spoken: The statements.
        votes: Vote of each agent, an agent name or `SKIP`.
        ejected: The agent ejected, or None.
    """

    closed_at: float
    spoken: list[Spoken] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    ejected: str | None = None


@dataclass
class RoundLog:
    """A round as it was recorded.

    Attributes:
        number: Round of the game.
        acts: The acts of each agent, one for each tick it took part in.
        reporter: Agent that reported a body, or None when the round had no meeting.
        body: The body it reported.
        victims: Agents eliminated during the round.
        meeting: The meeting, or None.
    """

    number: int
    acts: dict[str, list[Act]]
    reporter: str | None = None
    body: str | None = None
    victims: list[str] = field(default_factory=list)
    meeting: MeetingLog | None = None

    def ticks(self) -> int:
        """Counts the ticks of the round."""
        return max((len(a) for a in self.acts.values()), default=0)


@dataclass
class Replay:
    """Everything the timeline needs from a game folder.

    Attributes:
        game_id: Folder name without `Game_` and `_Run0`.
        model: Model the agents ran.
        protocol: Discussion protocol.
        seed: Scenario seed.
        roles: Role of each agent, `honest` or `byzantine`.
        rounds: The rounds in order.
        winner: `honest` or `byzantine`, or None when the stats say nothing.
        room: Settings of the timed room as the first meeting recorded them, or empty when no meeting had a room.
    """

    game_id: str
    model: str
    protocol: str
    seed: int
    roles: dict[str, str]
    rounds: list[RoundLog]
    winner: str | None = None
    room: dict = field(default_factory=dict)


def parse_acts(text: str) -> tuple[dict[int, list[Act]], dict[int, tuple[str, str]]]:
    """Reads the acts of one agent and why each meeting began, from its `action.log`.

    Args:
        text: The content of the log.

    Returns:
        The acts by round, and for each round that its log gives a report for, the reporter and the body.
    """
    acts: dict[int, list[Act]] = defaultdict(list)
    causes: dict[int, tuple[str, str]] = {}
    round_num, room = 0, ""
    for line in text.splitlines():
        if header := ROUND_HEADER.match(line):
            round_num = int(header["round"])
        elif place := LOCATION.match(line):
            room = place["room"]
        elif action := ACTION.match(line):
            acts[round_num].append(Act(room, action["verb"], action["target"]))
        elif cause := CAUSE.match(line):
            # A log of an agent that is out of the game gets the causes of later meetings under its last round, so only the first counts.
            causes.setdefault(round_num, (cause["reporter"], cause["body"]))
    return dict(acts), causes


def parse_victims(path: Path) -> dict[int, list[str]]:
    """Reads the agents eliminated in each round from `roundResults.log`."""
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    parts = re.split(r"^=== Round (\d+) ===$", text, flags=re.MULTILINE)
    return {int(n): re.findall(r"ELIMINATION: (Agent_\d+)", body) for n, body in zip(parts[1::2], parts[2::2])}


def read_room(path: Path) -> tuple[float | None, list[Spoken]]:
    """Reads the close time and the utterances of a timed room record, or nothing when the game had no room."""
    if not path.exists():
        return None, []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    closed = next((r["closed_at"] for r in rows if r.get("type") == "room"), None)
    spoken = [Spoken(r["author"], r["start_s"], r["end_s"], " ".join(r["text"].split())) for r in rows if r.get("type") == "utterance" and r.get("text")]
    return closed, sorted(spoken, key=lambda s: (s.start_s, s.author))


def read_room_options(path: Path) -> dict:
    """Reads the settings row of a timed room record, or an empty dict when there is none."""
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("type") == "room":
            return json.loads(line)
    return {}


def winner_of(path: Path) -> str | None:
    """Reads which side won from `stats.csv`, or None."""
    stats = path / "stats.csv"
    if not stats.exists():
        return None
    with stats.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    won = {r["alignment"] for r in rows if r.get("won_game") in ("1", "True", "true", "1.0")}
    return {"H": "honest", "B": "byzantine"}.get(next(iter(won), ""))


def load_replay(path: Path) -> Replay:
    """Reads a game folder.

    Args:
        path: A `Game_<id>_Run0` folder.

    Returns:
        The replay.

    Raises:
        ValueError: If the folder is not a P2 game.
    """
    game = parse_game(path)
    if game is None:
        raise ValueError(f"{path} is not a game folder with a discussion record")
    acts: dict[str, dict[int, list[Act]]] = {}
    causes: dict[int, Counter] = defaultdict(Counter)
    for agent, role in game.roles.items():
        log = path / ("Byz" if role == "byzantine" else "Honest") / agent / "action.log"
        if log.exists():
            acts[agent], found = parse_acts(log.read_text(encoding="utf-8", errors="ignore"))
            for number, cause in found.items():
                causes[number][cause] += 1
    victims = parse_victims(path / "roundResults.log")
    numbers = sorted({n for a in acts.values() for n in a} | set(victims))
    rounds = []
    options: dict = {}
    for number in numbers:
        entry = RoundLog(number, {agent: by_round[number] for agent, by_round in acts.items() if number in by_round}, victims=victims.get(number, []))
        if number in game.votes or number in game.meetings():
            if number in causes:
                entry.reporter, entry.body = causes[number].most_common(1)[0][0]
            closed, spoken = read_room(path / "room" / f"room_round{number}.jsonl")
            options = options or read_room_options(path / "room" / f"room_round{number}.jsonl")
            if not spoken:
                spoken = _paced(game, number)
            entry.meeting = MeetingLog(
                closed_at=closed if closed else max((s.end_s for s in spoken), default=0.0) + 2.0,
                spoken=spoken, votes=dict(game.votes.get(number, {})), ejected=game.ejected.get(number),
            )
        rounds.append(entry)
    return Replay(game.game_id, game.model, game.protocol, game.seed, dict(game.roles), rounds, winner_of(path), options)


def _paced(game, number: int) -> list[Spoken]:
    """Gives statements of a meeting without a clock one after another, for a speaking speed."""
    spoken, cursor = [], 0.0
    for s in game.statements:
        if s.meeting != number:
            continue
        length = max(2.0, len(s.text.split()) / WORDS_PER_S)
        spoken.append(Spoken(s.speaker, cursor, cursor + length, " ".join(s.text.split())))
        cursor += length + 0.5
    return spoken


class PathFinder:
    """Finds walks on the tile map that go through doors and corridors."""

    def __init__(self, world: WorldMap) -> None:
        """Creates a path finder.

        Args:
            world: The map.
        """
        self.world = world
        self._cache: dict[tuple, list[tuple[int, int]]] = {}

    def _passable(self, x: int, y: int) -> bool:
        return self.world.tile(x, y) in PASSABLE

    def _tiles(self, start: tuple[int, int], goal: tuple[int, int], rooms: Sequence[str]) -> list[tuple[int, int]]:
        """Finds the cheapest chain of tiles, where a room outside `rooms` costs 3 times as much to cross."""
        key = (start, goal, tuple(sorted(rooms)))
        if key in self._cache:
            return self._cache[key]
        best: dict[tuple[int, int], float] = {start: 0.0}
        came: dict[tuple[int, int], tuple[int, int]] = {}
        heap = [(0.0, start)]
        while heap:
            cost, tile = heapq.heappop(heap)
            if tile == goal:
                break
            if cost > best.get(tile, math.inf):
                continue
            x, y = tile
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, ny = x + dx, y + dy
                if not self._passable(nx, ny):
                    continue
                if dx and dy and not (self._passable(x + dx, y) and self._passable(x, y + dy)):
                    continue
                step = math.hypot(dx, dy)
                if self.world.tile(nx, ny) == FLOOR and self.world.room_at.get((nx, ny)) not in rooms:
                    step *= 3
                if cost + step < best.get((nx, ny), math.inf):
                    best[(nx, ny)] = cost + step
                    came[(nx, ny)] = tile
                    heapq.heappush(heap, (cost + step, (nx, ny)))
        if goal not in best:
            raise ValueError(f"No walk from {start} to {goal}")
        chain = [goal]
        while chain[-1] != start:
            chain.append(came[chain[-1]])
        chain.reverse()
        self._cache[key] = chain
        return chain

    def _clear(self, a: Point, b: Point) -> bool:
        """Tells whether a straight walk keeps 0.3 of a tile away from every wall."""
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.2))
        for k in range(steps + 1):
            x, y = a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps
            if not all(self._passable(math.floor(x + dx), math.floor(y + dy)) for dx in (-0.3, 0.3) for dy in (-0.3, 0.3)):
                return False
        return True

    def route(self, start: Point, goal: Point, rooms: Sequence[str] = ()) -> list[Point]:
        """Plans a walk.

        Args:
            start: Where the walk begins, in tile units.
            goal: Where it ends.
            rooms: Rooms the walk may cross without extra cost, which are the rooms of the two ends.

        Returns:
            Corner points from `start` to `goal`, with the corners cut where the way is clear.
        """
        s, g = (math.floor(start[0]), math.floor(start[1])), (math.floor(goal[0]), math.floor(goal[1]))
        chain = self._tiles(s, g, rooms)
        points: list[Point] = [start] + [(x + 0.5, y + 0.5) for x, y in chain[1:-1]] + [goal]
        out, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._clear(points[i], points[j]):
                j -= 1
            out.append(points[j])
            i = j
        return out


def room_slots(room: Room) -> list[Point]:
    """Lists spots of a room where agents can stand, from the middle outward and away from its label.

    Args:
        room: The room.

    Returns:
        Spots in tile units, at least 1.4 tiles apart.
    """
    cx, cy = room.center()
    half_label = 0.22 * len(display_name(room.name)) + 0.4
    floor = set(room.floor())
    free: list[Point] = []
    under_label: list[Point] = []
    for j in range(-12, 13):
        for i in range(-12, 13):
            x, y = cx + i * 1.4 + (0.7 if j % 2 else 0.0), cy + j * 1.3
            if not {(math.floor(x + dx), math.floor(y + dy)) for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)} <= floor:
                continue
            (under_label if abs(y - cy) < 0.8 and abs(x - cx) < half_label else free).append((x, y))

    def by_distance(p: Point) -> tuple[float, Point]:
        return math.hypot(p[0] - cx, (p[1] - cy) * 1.2), p

    return sorted(free, key=by_distance) + sorted(under_label, key=by_distance)


def seat_points(agents: Sequence[str], world: WorldMap) -> dict[str, Point]:
    """Places each agent at its own seat around the table of the meeting room, by agent number.

    Args:
        agents: Every agent of the game.
        world: The map.

    Returns:
        The seat of each agent in tile units.
    """
    cx, cy = world.rooms[MEETING_ROOM].center()
    order = sorted(agents, key=lambda a: int(a.rsplit("_", 1)[1]))
    return {
        a: (cx + SEAT_RADII[0] * math.sin(2 * math.pi * k / len(order)), cy - SEAT_RADII[1] * math.cos(2 * math.pi * k / len(order)))
        for k, a in enumerate(order)
    }


@dataclass
class Leg:
    """A walk along a path between two times."""

    start: float
    end: float
    path: list[Point]

    def __post_init__(self) -> None:
        self.length = sum(math.dist(a, b) for a, b in zip(self.path, self.path[1:]))

    def at(self, t: float) -> Point:
        """Returns the position at a time, with the walk easing in and out."""
        if t <= self.start or self.end <= self.start:
            return self.path[0] if t <= self.start else self.path[-1]
        if t >= self.end:
            return self.path[-1]
        u = (t - self.start) / (self.end - self.start)
        remaining = (3 * u * u - 2 * u * u * u) * self.length
        last = len(self.path) - 2
        for i, (a, b) in enumerate(zip(self.path, self.path[1:])):
            d = math.dist(a, b)
            if remaining <= d or i == last:
                f = 0.0 if d == 0 else min(1.0, remaining / d)
                return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            remaining -= d
        return self.path[-1]


@dataclass(frozen=True)
class Fate:
    """When an agent left the game.

    Attributes:
        kind: `killed` or `ejected`.
        time: Video time.
        round_num: Round of the game.
        at: Where the agent was.
    """

    kind: str
    time: float
    round_num: int
    at: Point


@dataclass(frozen=True)
class Caption:
    """A statement as it appears in the video."""

    start: float
    end: float
    speaker: str
    text: str
    round_num: int


@dataclass(frozen=True)
class Banner:
    """A line of text on a stretch of the video, with a spot on the map to point at."""

    start: float
    end: float
    title: str
    detail: str = ""
    focus: Point | None = None


@dataclass(frozen=True)
class Strike:
    """A tag, drawn from the killer to the victim."""

    start: float
    end: float
    killer: str
    victim: str
    killer_at: Point
    victim_at: Point


@dataclass(frozen=True)
class VoteMark:
    """A vote shown from a time until the end of its meeting."""

    start: float
    end: float
    voter: str
    target: str


@dataclass(frozen=True)
class Event:
    """Something that happened, for the log under the map."""

    time: float
    round_num: int
    text: str


@dataclass(frozen=True)
class Phase:
    """A stretch of the video with a name for the heading."""

    start: float
    end: float
    kind: str
    round_num: int
    label: str


@dataclass(frozen=True)
class Table:
    """A stretch with the table in the meeting room, and the clock of the meeting inside it."""

    start: float
    end: float
    clock_start: float
    clock_end: float
    length_s: float
    round_num: int


@dataclass
class Token:
    """An agent on a frame."""

    name: str
    x: float
    y: float
    scale: float
    speaking: bool


@dataclass
class Frame:
    """What one frame of the video shows.

    Attributes:
        t: Video time.
        phase: The current phase, or None.
        tokens: Agents in play.
        bodies: Bodies as (victim, x, y).
        status: For each agent, `alive`, `killed` or `ejected`, with the round.
        strikes: Tags under way as (killer position, victim position, progress).
        arrows: Votes shown as (voter, target).
        captions: Statements to show, as (speaker, text, finished).
        banner: The current banner, or None.
        table: The table and its clock as (seconds into the meeting, length), or None.
        tally: Votes by target when the vote is shown.
        events: What has happened so far, as (round, text), oldest first.
    """

    t: float
    phase: Phase | None
    tokens: list[Token]
    bodies: list[tuple[str, float, float]]
    status: dict[str, tuple[str, int | None]]
    strikes: list[tuple[Point, Point, float]]
    arrows: list[tuple[str, str]]
    captions: list[tuple[str, str, bool]]
    banner: Banner | None
    table: tuple[float, float] | None
    tally: dict[str, int]
    events: list[tuple[int, str]]


@dataclass
class Timeline:
    """The whole video as data."""

    duration: float
    roles: dict[str, str]
    start: dict[str, Point]
    seats: dict[str, Point]
    legs: dict[str, list[Leg]] = field(default_factory=dict)
    fates: dict[str, Fate] = field(default_factory=dict)
    cleared: dict[str, float] = field(default_factory=dict)
    captions: list[Caption] = field(default_factory=list)
    banners: list[Banner] = field(default_factory=list)
    strikes: list[Strike] = field(default_factory=list)
    marks: list[VoteMark] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    pacing: Pacing = field(default_factory=Pacing)
    eject_s: float = 2.6

    def position(self, agent: str, t: float) -> Point:
        """Returns where an agent is at a video time."""
        place = self.start[agent]
        for leg in self.legs.get(agent, []):
            if leg.start > t:
                break
            place = leg.at(t)
        return place

    def at(self, t: float) -> Frame:
        """Builds the frame for a video time."""
        tokens, bodies, status = [], [], {}
        speaking = {c.speaker for c in self.captions if c.start <= t < c.end}
        for agent in sorted(self.roles):
            fate = self.fates.get(agent)
            if fate is None or t < fate.time:
                tokens.append(Token(agent, *self.position(agent, t), 1.0, agent in speaking))
                status[agent] = ("alive", None)
                continue
            status[agent] = (fate.kind, fate.round_num)
            if fate.kind == "killed":
                if t < self.cleared.get(agent, math.inf):
                    bodies.append((agent, *fate.at))
            elif t < fate.time + self.eject_s:
                tokens.append(Token(agent, *self.position(agent, t), max(0.0, 1 - (t - fate.time) / self.eject_s), False))
        strikes = [(s.killer_at, s.victim_at, (t - s.start) / (s.end - s.start)) for s in self.strikes if s.start <= t < s.end]
        arrows = [(m.voter, m.target) for m in self.marks if m.start <= t < m.end]
        captions = [
            (c.speaker, _revealed(c, t), t >= c.end)
            for c in self.captions
            if c.start <= t < c.end + self.pacing.linger
        ][-3:]
        banner = next((b for b in self.banners if b.start <= t < b.end), None)
        table = next(((min(x.length_s, max(0.0, (t - x.clock_start) / (x.clock_end - x.clock_start) * x.length_s)), x.length_s) for x in self.tables if x.start <= t < x.end), None)
        tally = dict(Counter(target for _, target in arrows))
        phase = next((p for p in self.phases if p.start <= t < p.end), self.phases[-1] if self.phases else None)
        events = [(e.round_num, e.text) for e in self.events if e.time <= t]
        return Frame(t, phase, tokens, bodies, status, strikes, arrows, captions, banner, table, tally, events)


def _revealed(caption: Caption, t: float) -> str:
    """Cuts a caption to the words spoken by a time, at an even rate over its length."""
    words = caption.text.split()
    if t >= caption.end:
        return caption.text
    share = (t - caption.start) / (caption.end - caption.start)
    return " ".join(words[: max(1, math.ceil(share * len(words)))])


class Director:
    """Plans the video, one phase after another, from a replay."""

    def __init__(self, replay: Replay, world: WorldMap, pacing: Pacing) -> None:
        """Creates a director.

        Args:
            replay: The game.
            world: The map.
            pacing: The durations.
        """
        self.replay, self.world, self.pacing = replay, world, pacing
        self.finder = PathFinder(world)
        self.slots = {name: room_slots(room) for name, room in world.rooms.items()}
        self.seats = seat_points(list(replay.roles), world)
        self.t = 0.0
        self.alive: set[str] = set(replay.roles)
        self.where: dict[str, Point] = {}
        self.rooms: dict[str, str] = {}
        self.out = Timeline(0.0, dict(replay.roles), {}, self.seats, pacing=pacing, eject_s=pacing.eject)
        self.out.legs = defaultdict(list)
        self.bodies: list[str] = []

    def layout(self, rooms: dict[str, str]) -> dict[str, Point]:
        """Gives each agent a spot in its room, in name order, away from the bodies that lie there."""
        by_room: dict[str, list[str]] = defaultdict(list)
        for agent, room in rooms.items():
            by_room[room].append(agent)
        lying = [self.out.fates[v].at for v in self.bodies if v in self.out.fates]
        spots: dict[str, Point] = {}
        for room, agents in by_room.items():
            free = [p for p in self.slots[room] if all(math.dist(p, body) > 1.2 for body in lying)] or self.slots[room]
            for k, agent in enumerate(sorted(agents)):
                spots[agent] = free[k % len(free)]
        return spots

    def _walk(self, agent: str, goal: Point, start: float, end: float, rooms: Sequence[str]) -> None:
        """Adds a walk to an agent, or a short slide when the goal is near."""
        origin = self.where[agent]
        if math.dist(origin, goal) < 1e-6:
            return
        near = math.dist(origin, goal) < 3.0 and len(set(rooms)) == 1
        path = [origin, goal] if near else self.finder.route(origin, goal, rooms)
        self.out.legs[agent].append(Leg(start, end, path))
        self.where[agent] = goal

    def _phase(self, kind: str, round_num: int, label: str, length: float) -> None:
        self.out.phases.append(Phase(self.t, self.t + length, kind, round_num, label))

    def run(self) -> Timeline:
        """Plans every round and the result card.

        Returns:
            The timeline.
        """
        rounds = self.replay.rounds
        first = next((r for r in rounds if r.acts), None)
        if first is None:
            raise ValueError("The game has no acts")
        self.rooms = {a: acts[0].room for a, acts in first.acts.items()}
        self.where = self.layout(self.rooms)
        self.out.start = dict(self.where)
        for agent in self.replay.roles:
            self.out.start.setdefault(agent, self.seats[agent])
            self.where.setdefault(agent, self.seats[agent])
        for index, entry in enumerate(rounds):
            nxt = next((r for r in rounds[index + 1:] if r.acts), None)
            self._round(entry, nxt)
        self._final()
        self.out.duration = self.t
        return self.out

    def _round(self, entry: RoundLog, nxt: RoundLog | None) -> None:
        """Plans the card, the ticks and the meeting of a round."""
        pace, number = self.pacing, entry.number
        self._phase("round", number, f"Round {number}", pace.round_card)
        self.t += pace.round_card
        for tick in range(entry.ticks()):
            if self._tick(entry, tick):
                self._meeting(entry, nxt)
                return

    def _tick(self, entry: RoundLog, tick: int) -> bool:
        """Plans a tick, and says whether a report ended the round."""
        pace, number = self.pacing, entry.number
        acts = {a: v[tick] for a, v in entry.acts.items() if a in self.alive and tick < len(v)}
        for agent, act in acts.items():
            self.rooms[agent] = act.room
        before = self.layout({a: r for a, r in self.rooms.items() if a in self.alive})
        for agent, spot in before.items():
            if math.dist(self.where[agent], spot) > 0.01:
                self.out.legs[agent].append(Leg(self.t, self.t, [self.where[agent], spot]))
                self.where[agent] = spot
        label = f"Round {number}, tick {tick + 1}"
        dead: set[str] = set()
        tagged: list[str] = []
        focus: Point | None = None
        for killer, act in sorted(acts.items()):
            if act.verb != "tag" or killer in dead or act.target not in self.alive or act.target in dead:
                continue
            if self.rooms.get(act.target) != act.room:
                continue
            dead.add(act.target)
            at = self.where[act.target]
            start = self.t + 0.2
            self.out.strikes.append(Strike(start, start + 0.7, killer, act.target, self.where[killer], at))
            self.out.fates[act.target] = Fate("killed", start + 0.5, number, at)
            tagged.append(f"{killer} tagged {act.target} in {display_name(act.room)}")
            focus = focus or at
            self.bodies.append(act.target)
        self.alive -= dead
        if tagged:
            self.out.banners.append(Banner(self.t + 0.2, self.t + pace.kill, tagged[0], "; ".join(tagged[1:]), focus))
            self.out.events.extend(Event(self.t + 0.7, number, text) for text in tagged)
            self._phase("kill", number, label, pace.kill)
            # Survivors step off the spot where a body now lies.
            for agent, spot in self.layout({a: r for a, r in self.rooms.items() if a in self.alive}).items():
                if math.dist(self.where[agent], spot) > 0.01:
                    self.out.legs[agent].append(Leg(self.t + 0.6, self.t + pace.kill, [self.where[agent], spot]))
                    self.where[agent] = spot
            self.t += pace.kill
        reports = [(a, act) for a, act in sorted(acts.items()) if act.verb == "report" and a not in dead]
        if reports:
            reporter, act = reports[0]
            spot = self.out.fates[act.target].at if act.target in self.out.fates else self.where[reporter]
            self.out.banners.append(Banner(self.t, self.t + pace.report, f"{reporter} reported {act.target}'s body", f"in {display_name(self.rooms[reporter])}", spot))
            self.out.events.append(Event(self.t, number, f"{reporter} reported {act.target}'s body in {display_name(self.rooms[reporter])}"))
            self._phase("report", number, label, pace.report)
            self.t += pace.report
            return True
        after = dict(self.rooms)
        for agent, act in acts.items():
            if agent not in dead and act.verb == "move" and act.target in ROOMS:
                after[agent] = act.target
        goal = self.layout({a: r for a, r in after.items() if a in self.alive})
        start = self.t + pace.settle
        for agent, spot in goal.items():
            self._walk(agent, spot, start, start + pace.walk, (self.rooms[agent], after[agent]))
        self.rooms = after
        self._phase("tick", number, label, pace.settle + pace.walk)
        self.t += pace.settle + pace.walk
        return False

    def _meeting(self, entry: RoundLog, nxt: RoundLog | None) -> None:
        """Plans the walk to the table, the talk, the vote, the ejection and the walk back."""
        pace, number = self.pacing, entry.number
        members = sorted(self.alive)
        for victim in entry.victims:
            if victim in self.alive:
                self.alive.discard(victim)
                at = self.where.get(victim, self.seats[victim])
                self.out.fates[victim] = Fate("killed", self.t, number, at)
                members = sorted(self.alive)
        meeting = entry.meeting
        self._phase("gather", number, f"Round {number}, meeting", pace.gather)
        gather_start = self.t
        for agent in members:
            self._walk(agent, self.seats[agent], self.t, self.t + pace.gather, (self.rooms[agent], MEETING_ROOM))
        self.t += pace.gather
        if meeting is None:
            return
        length = max(pace.min_meeting, meeting.closed_at / pace.meeting_speed)
        begin = self.t
        self._phase("meeting", number, f"Round {number}, meeting", length)
        for s in meeting.spoken:
            start, end = begin + s.start_s / pace.meeting_speed, begin + s.end_s / pace.meeting_speed
            self.out.captions.append(Caption(start, max(end, start + 0.4), s.author, s.text, number))
        self.t += length
        vote_start = self.t
        voters = [v for v in sorted(meeting.votes) if v in self.replay.roles]
        shown = len(voters) * pace.per_vote + pace.vote_hold
        self._phase("vote", number, f"Round {number}, vote", shown)
        end_of_meeting = vote_start + shown + pace.eject
        for k, voter in enumerate(voters):
            self.out.marks.append(VoteMark(vote_start + k * pace.per_vote, end_of_meeting, voter, meeting.votes[voter]))
        self.out.tables.append(Table(gather_start + pace.gather * 0.6, end_of_meeting, begin, begin + length, meeting.closed_at, number))
        self.t = vote_start + shown
        self._phase("eject", number, f"Round {number}, result", pace.eject)
        ejected = meeting.ejected
        counts = Counter(meeting.votes.values())
        if ejected and ejected in self.replay.roles:
            at = self.seats[ejected]
            self.out.fates[ejected] = Fate("ejected", self.t + 0.4, number, at)
            self.out.banners.append(Banner(self.t, self.t + pace.eject, f"{ejected} was ejected", f"{self.replay.roles[ejected].capitalize()}. {counts[ejected]} of {len(voters)} votes", at))
            self.out.events.append(Event(self.t, number, f"{ejected} was ejected, {self.replay.roles[ejected]}, with {counts[ejected]} of {len(voters)} votes"))
            self.alive.discard(ejected)
        else:
            self.out.banners.append(Banner(self.t, self.t + pace.eject, "No one was ejected", "The votes tied or skipped"))
            self.out.events.append(Event(self.t, number, "No one was ejected"))
        self.t += pace.eject
        for victim in self.bodies:
            self.out.cleared[victim] = self.t
        self.bodies = []
        if nxt is not None:
            self._back(nxt, number)

    def _back(self, nxt: RoundLog, number: int) -> None:
        """Plans the walk from the table to the rooms where the next round begins."""
        pace = self.pacing
        self.rooms = {a: acts[0].room for a, acts in nxt.acts.items() if a in self.alive}
        goal = self.layout(self.rooms)
        self._phase("back", nxt.number, f"Round {number}, back to work", pace.back)
        for agent, spot in goal.items():
            self._walk(agent, spot, self.t, self.t + pace.back, (MEETING_ROOM, self.rooms[agent]))
        self.t += pace.back

    def _final(self) -> None:
        """Adds the result card."""
        pace, replay = self.pacing, self.replay
        side = {"honest": "Honest agents win", "byzantine": "Byzantine agents win"}.get(replay.winner or "", "Game over")
        left = Counter(replay.roles[a] for a in self.alive)
        detail = f"{left['honest']} honest and {left['byzantine']} Byzantine agents were left"
        self.out.banners.append(Banner(self.t, self.t + pace.final, side, detail))
        self._phase("end", self.replay.rounds[-1].number, "Game over", pace.final)
        self.t += pace.final


def build_timeline(replay: Replay, world: WorldMap, pacing: Pacing | None = None) -> Timeline:
    """Plans the video of a game.

    Args:
        replay: The game.
        world: The map.
        pacing: The durations, or None for the defaults.

    Returns:
        The timeline.
    """
    return Director(replay, world, pacing or Pacing()).run()
