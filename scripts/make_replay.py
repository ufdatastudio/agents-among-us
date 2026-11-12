"""Renders a replay of one meeting of a recorded P2 game as a video, a short GIF and still frames.

The replay shows a timed room as it happened, drawn from the logs of the game: who spoke when, what each agent said,
and how each agent voted. It reads `room/room_round<N>.jsonl` and the roles, votes and results of the game folder.
The viewer sees the roles. The agents never saw them.

Usage:
    uv run --with matplotlib python -m scripts.make_replay logs/p2_.../Game_tag_interleaved-180_3_Run0 --round 1 --out docs/media

The video needs ffmpeg on the path.
"""

import argparse
import ast
import csv
import json
import math
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from scripts.analyze_p2_text import parse_game, short_model

WIDTH, HEIGHT = 1600, 900
SCALE = 2

BG = (17, 20, 26)
PANEL = (24, 28, 36)
TABLE = (33, 39, 50)
INK = (232, 235, 240)
MUTED = (140, 148, 162)
GRID = (52, 58, 70)
SEAT = (40, 47, 60)
SEAT_OUT = (70, 76, 90)
ACCENT = (96, 165, 255)
ACCENT_DONE = (64, 104, 156)
BYZANTINE = (255, 120, 100)
GOOD = (104, 206, 146)

TABLE_CENTER = (424, 345)
SEAT_RADII = (300, 185)
LANES_TOP = 656
LANE_H = 21
AXIS = (124, 1560)


@dataclass
class Utterance:
    """One statement in the room.

    Attributes:
        author: Agent that spoke.
        start_s: Meeting time at which the speech began.
        end_s: Meeting time at which it ended.
        text: Words of the statement.
    """

    author: str
    start_s: float
    end_s: float
    text: str

    def visible(self, t: float) -> list[str]:
        """Gives the words heard by meeting time t, spread evenly from the start to the end of the speech."""
        words = self.text.split()
        if t <= self.start_s or not words:
            return []
        span = max(self.end_s - self.start_s, 1e-6)
        return words[: min(len(words), math.ceil(len(words) * (t - self.start_s) / span))]

    def speaking(self, t: float) -> bool:
        """Tells whether the agent is speaking at meeting time t."""
        return self.start_s <= t < self.end_s


@dataclass
class Meeting:
    """Everything a replay draws, read from one game folder.

    Attributes:
        game_id: Id of the game.
        model: Short name of the model.
        protocol: Protocol of the room, for example `interleaved-180`.
        seed: Scenario seed of the game.
        round: Round of the meeting.
        roster: Every agent of the game, in seat order.
        roles: Role of each agent.
        status: Status of each agent at the start of the meeting (`active`, `eliminated` or `ejected`).
        victims: Agents found dead before the meeting.
        closed_at: Meeting time at which the room closed.
        tick_s: Seconds between ticks.
        reason: Text that opened the meeting.
        utterances: Statements of the room, by start time.
        votes: Vote of each agent, by voter.
        tally: Votes received by each agent.
        ejected: Agent ejected after the meeting, or None.
    """

    game_id: str
    model: str
    protocol: str
    seed: int
    round: int
    roster: list[str]
    roles: dict[str, str]
    status: dict[str, str]
    victims: list[str]
    closed_at: float
    tick_s: float
    reason: str
    utterances: list[Utterance] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)
    tally: dict[str, int] = field(default_factory=dict)
    ejected: str | None = None

    @property
    def setting(self) -> str:
        """Describes the room in words, for example `timed room, interleaved, 180 s`."""
        mode, _, budget = self.protocol.partition("-")
        return "turns, one speaker at a time" if self.protocol == "sequential" else f"timed room, {mode}, {budget} s"

    @property
    def speakers(self) -> list[str]:
        """Agents that could speak, in seat order."""
        return [a for a in self.roster if self.status.get(a) == "active" and a not in self.victims]


def read_results(path: Path, round_num: int) -> dict:
    """Reads one round of a `roundResults.log`.

    Args:
        path: The log.
        round_num: Round to read.

    Returns:
        The victims, the tally of votes, the agent ejected and the statuses before and after the round.
    """
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    parts = re.split(r"^=== Round (\d+) ===$", text, flags=re.MULTILINE)
    before: dict[str, str] = {}
    for number, body in zip(parts[1::2], parts[2::2]):
        snapshot = re.search(r"Player Statuses: (\{.*\})", body)
        after = ast.literal_eval(snapshot.group(1)) if snapshot else before
        if int(number) == round_num:
            tally = re.search(r"Votes Received: (\{.*\})", body)
            eject = re.search(r"EJECTION: (Agent_\d+)", body)
            return {
                "victims": re.findall(r"ELIMINATION: (Agent_\d+)", body),
                "tally": ast.literal_eval(tally.group(1)) if tally else {},
                "ejected": eject.group(1) if eject else None,
                "before": before,
                "after": after,
            }
        before = after
    return {"victims": [], "tally": {}, "ejected": None, "before": before, "after": before}


def clean_reason(reason: str) -> str:
    """Shortens the line that opened a meeting for display."""
    reason = reason.replace("**", "").replace("MEETING CALLED", "Meeting called").strip()
    return re.sub(r"\s*Additional victims confirmed eliminated: None\.?", "", reason).strip()


def load_meeting(game_dir: Path, round_num: int | None = None) -> Meeting:
    """Reads one meeting of a game folder.

    Args:
        game_dir: A `Game_<id>_Run0` folder of a P2 game with a timed room.
        round_num: Round of the meeting, or None for the first one that has a room record.

    Returns:
        The meeting.

    Raises:
        ValueError: If the folder is not a P2 game or has no room record for the round.
    """
    game = parse_game(game_dir)
    if game is None:
        raise ValueError(f"{game_dir} is not a P2 game folder")
    files = sorted((game_dir / "room").glob("room_round*.jsonl"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if round_num is None:
        if not files:
            raise ValueError(f"{game_dir} has no room record")
        round_num = int(re.search(r"(\d+)", files[0].stem).group(1))
    record = game_dir / "room" / f"room_round{round_num}.jsonl"
    if not record.exists():
        raise ValueError(f"{record} does not exist")
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = next((r for r in rows if r.get("type") == "room"), {})
    utterances = sorted(
        (Utterance(r["author"], float(r["start_s"]), float(r["end_s"]), " ".join(r["text"].split())) for r in rows if r.get("type") == "utterance" and r.get("text")),
        key=lambda u: (u.start_s, u.author),
    )
    results = read_results(game_dir / "roundResults.log", round_num)
    roster = sorted(game.roles, key=lambda a: int(a.split("_")[1]))
    status = {a: results["before"].get(a, "active") for a in roster}
    reason = ""
    with (game_dir / "discussion_chat.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["discussion_num"]) == round_num:
                reason = clean_reason(row["reason"])
                break
    return Meeting(
        game_id=game.game_id,
        model=short_model(game.model),
        protocol=game.protocol,
        seed=game.seed,
        round=round_num,
        roster=roster,
        roles=game.roles,
        status=status,
        victims=results["victims"],
        closed_at=float(header.get("closed_at", max((u.end_s for u in utterances), default=0.0))),
        tick_s=float(header.get("tick_s", 10.0)),
        reason=reason,
        utterances=utterances,
        votes={v: t for v, t in sorted(game.votes.get(round_num, {}).items(), key=lambda kv: int(kv[0].split("_")[1])) if t in roster},
        tally=results["tally"],
        ejected=results["ejected"],
    )


@dataclass(frozen=True)
class State:
    """What the replay shows at one moment.

    Attributes:
        t: Meeting clock in seconds.
        votes_shown: Number of votes drawn so far.
        result: True once the result of the vote is shown.
    """

    t: float
    votes_shown: int = 0
    result: bool = False


@dataclass(frozen=True)
class Timeline:
    """Maps the time of the video to the state of the meeting.

    Attributes:
        closed_at: Length of the discussion in meeting seconds.
        n_votes: Number of votes to show.
        speed: Meeting seconds per video second.
        intro_s: Video seconds that show the table before the clock starts.
        hold_s: Video seconds that hold the closed discussion before the votes.
        per_vote_s: Video seconds between votes.
        result_s: Video seconds that show the result.
    """

    closed_at: float
    n_votes: int
    speed: float = 2.5
    intro_s: float = 2.0
    hold_s: float = 1.5
    per_vote_s: float = 0.5
    result_s: float = 4.0

    @property
    def duration(self) -> float:
        """Length of the video in seconds."""
        return self.intro_s + self.closed_at / self.speed + self.hold_s + self.n_votes * self.per_vote_s + self.result_s

    def at(self, video_t: float) -> State:
        """Gives the state at a time of the video."""
        talk_end = self.intro_s + self.closed_at / self.speed
        if video_t < self.intro_s:
            return State(0.0)
        if video_t < talk_end:
            return State((video_t - self.intro_s) * self.speed)
        vote_start = talk_end + self.hold_s
        if video_t < vote_start:
            return State(self.closed_at)
        shown = int((video_t - vote_start) / self.per_vote_s) + 1
        if shown <= self.n_votes:
            return State(self.closed_at, votes_shown=shown)
        return State(self.closed_at, votes_shown=self.n_votes, result=True)


def seat_positions(names: Sequence[str], center: tuple[float, float] = TABLE_CENTER, radii: tuple[float, float] = SEAT_RADII) -> dict[str, tuple[float, float]]:
    """Places seats on an ellipse, the first at the top and the rest clockwise."""
    n = len(names)
    return {
        name: (center[0] + radii[0] * math.cos(-math.pi / 2 + 2 * math.pi * i / n), center[1] + radii[1] * math.sin(-math.pi / 2 + 2 * math.pi * i / n))
        for i, name in enumerate(names)
    }


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    """Breaks text into lines no wider than `width` pixels at the font's size."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and font.getlength(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines


def last_words(words: Sequence[str], font: ImageFont.FreeTypeFont, width: float, lines: int = 2) -> list[str]:
    """Gives the end of a statement as at most `lines` lines that fit `width`, with an ellipsis when it was cut."""
    wrapped = wrap(" ".join(words), font, width)
    if len(wrapped) <= lines:
        return wrapped
    kept = wrapped[-lines:]
    first = kept[0]
    while font.getlength("... " + first) > width and " " in first:
        first = first.split(" ", 1)[1]
    kept[0] = "... " + first
    return kept


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Loads the DejaVu fonts that ship with matplotlib, at the scaled sizes used by the frames."""
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular, bold = str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")
    sizes = {"title": 24, "clock": 24, "meta": 15, "label": 12, "small": 11, "seat": 20, "text": 13, "head": 13, "badge": 11, "center": 14, "big": 20}
    return {
        name: ImageFont.truetype(bold if name in ("title", "clock", "seat", "head", "badge", "big") else regular, size * SCALE)
        for name, size in sizes.items()
    }


class Renderer:
    """Draws the frames of a meeting."""

    def __init__(self, meeting: Meeting) -> None:
        """Prepares the fonts, the seats and the static background.

        Args:
            meeting: The meeting to draw.
        """
        self.m = meeting
        self.f = load_fonts()
        self.seats = seat_positions(meeting.roster)
        self.speakers = meeting.speakers
        self.base = self._background()

    def _p(self, v: float) -> int:
        return round(v * SCALE)

    def _box(self, d: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float, **kw) -> None:
        d.rounded_rectangle([self._p(x0), self._p(y0), self._p(x1), self._p(y1)], radius=self._p(kw.pop("radius", 10)), **kw)

    def _text(self, d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: str, fill, anchor: str = "la") -> None:
        d.text((self._p(xy[0]), self._p(xy[1])), text, font=self.f[font], fill=fill, anchor=anchor)

    def _circle(self, d: ImageDraw.ImageDraw, xy: tuple[float, float], r: float, **kw) -> None:
        width = kw.pop("width", 0)
        d.ellipse([self._p(xy[0] - r), self._p(xy[1] - r), self._p(xy[0] + r), self._p(xy[1] + r)], width=self._p(width) if width else 0, **kw)

    def _background(self) -> Image.Image:
        image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), BG)
        d = ImageDraw.Draw(image)
        self._text(d, (28, 16), "Situation room replay", "title", INK)
        meta = f"{self.m.model}  ·  {self.m.setting}  ·  seed {self.m.seed}  ·  round {self.m.round}"
        self._text(d, (WIDTH - 28, 24), meta, "meta", MUTED, "ra")
        self._box(d, 24, 76, 824, 620, fill=PANEL, radius=14)
        self._box(d, 840, 76, 1576, 620, fill=PANEL, radius=14)
        self._box(d, 24, 632, 1576, 892, fill=PANEL, radius=14)
        cx, cy = TABLE_CENTER
        d.ellipse([self._p(cx - 190), self._p(cy - 98), self._p(cx + 190), self._p(cy + 98)], fill=TABLE, outline=GRID, width=self._p(2))
        self._text(d, (856, 88), "What the agents said, as they spoke it", "label", MUTED)
        self._text(d, (36, 640), "Who is speaking", "label", MUTED)
        self._text(d, (WIDTH - 36, 640), "B marks a Byzantine agent. The agents were not told the roles.", "label", MUTED, "ra")
        x0, x1 = AXIS
        for i, name in enumerate(self.speakers):
            y = LANES_TOP + i * LANE_H
            self._text(d, (36, y + 3), name, "small", MUTED)
            if self.m.roles.get(name) == "byzantine":
                self._badge(d, (104, y + 9))
        grid_y1 = LANES_TOP + len(self.speakers) * LANE_H
        step = self.m.tick_s
        k = 0
        while k * step <= self.m.closed_at + 1e-6:
            x = x0 + (x1 - x0) * k * step / self.m.closed_at
            d.line([self._p(x), self._p(LANES_TOP - 2), self._p(x), self._p(grid_y1)], fill=GRID if k % 3 == 0 else (38, 43, 54), width=self._p(1))
            if k % 3 == 0:
                self._text(d, (x, grid_y1 + 4), f"{int(k * step)} s", "small", MUTED, "ma")
            k += 1
        return image

    def _badge(self, d: ImageDraw.ImageDraw, xy: tuple[float, float]) -> None:
        self._circle(d, xy, 7, fill=BYZANTINE)
        self._text(d, xy, "B", "badge", BG, "mm")

    def _lane_x(self, t: float) -> float:
        x0, x1 = AXIS
        return x0 + (x1 - x0) * min(max(t, 0.0), self.m.closed_at) / self.m.closed_at

    def frame(self, state: State) -> Image.Image:
        """Draws one frame.

        Args:
            state: The moment to draw.

        Returns:
            The frame at the output size.
        """
        image = self.base.copy()
        d = ImageDraw.Draw(image)
        self._header(d, state)
        self._table(d, state)
        self._transcript(d, state)
        self._lanes(d, state)
        return image.resize((WIDTH, HEIGHT), Image.LANCZOS)

    def _header(self, d: ImageDraw.ImageDraw, state: State) -> None:
        if state.result:
            label = "Result"
        elif state.votes_shown:
            label = "Voting"
        else:
            label = f"{int(state.t) // 60}:{int(state.t) % 60:02d} / {int(self.m.closed_at) // 60}:{int(self.m.closed_at) % 60:02d}"
        self._text(d, (WIDTH / 2, 20), label, "clock", INK, "ma")
        self._box(d, 24, 62, WIDTH - 24, 68, fill=GRID, radius=3)
        fraction = 1.0 if state.votes_shown or state.result else state.t / self.m.closed_at
        if fraction > 0:
            self._box(d, 24, 62, 24 + (WIDTH - 48) * fraction, 68, fill=ACCENT, radius=3)

    def _table(self, d: ImageDraw.ImageDraw, state: State) -> None:
        m, t = self.m, state.t
        latest: dict[str, Utterance] = {}
        for u in m.utterances:
            if u.speaking(t) and not (state.votes_shown or state.result):
                latest[u.author] = u
        shown = list(m.votes.items())[: state.votes_shown]
        counts: dict[str, int] = {}
        for _, target in shown:
            counts[target] = counts.get(target, 0) + 1
        cx, cy = TABLE_CENTER
        for voter, target in shown:
            self._arrow(d, self.seats[voter], self.seats[target])
        if state.result:
            self._box(d, cx - 176, cy - 50, cx + 176, cy + 54, fill=PANEL, outline=GRID, width=self._p(2), radius=14)
            for i, (text, font, fill) in enumerate(self._result_lines()):
                self._text(d, (cx, cy - 36 + i * 26), text, font, fill, "ma")
        elif not state.votes_shown:
            for i, line in enumerate(self._wrap_logical(m.reason, "center", 300)[:4]):
                self._text(d, (cx, cy - 36 + i * 21), line, "center", MUTED, "ma")
        for name in m.roster:
            self._seat(d, name, name in latest, latest.get(name), t, counts.get(name, 0), state)

    def _wrap_logical(self, text: str, font: str, width: float) -> list[str]:
        return wrap(text, self.f[font], width * SCALE)

    def _result_lines(self) -> list[tuple[str, str, tuple[int, int, int]]]:
        m = self.m
        if not m.ejected:
            return [("No agent was ejected", "big", INK), ("The vote was tied or empty", "center", MUTED)]
        role = m.roles.get(m.ejected, "honest")
        right = role == "byzantine"
        votes = m.tally.get(m.ejected, 0)
        return [
            (f"{m.ejected} was ejected", "big", INK),
            (f"{votes} votes. Role: {'Byzantine' if right else 'honest'}", "center", GOOD if right else BYZANTINE),
            ("A Byzantine agent was removed" if right else "An honest agent was removed", "center", MUTED),
        ]

    def _arrow(self, d: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float]) -> None:
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        start = (a[0] + ux * 32, a[1] + uy * 32)
        end = (b[0] - ux * 34, b[1] - uy * 34)
        color = (86, 134, 205)
        d.line([self._p(start[0]), self._p(start[1]), self._p(end[0]), self._p(end[1])], fill=color, width=self._p(2))
        left = (end[0] - ux * 10 - uy * 5, end[1] - uy * 10 + ux * 5)
        right = (end[0] - ux * 10 + uy * 5, end[1] - uy * 10 - ux * 5)
        d.polygon([self._p(end[0]), self._p(end[1]), self._p(left[0]), self._p(left[1]), self._p(right[0]), self._p(right[1])], fill=color)

    def _seat(self, d: ImageDraw.ImageDraw, name: str, speaking: bool, utterance: Utterance | None, t: float, votes: int, state: State) -> None:
        m = self.m
        x, y = self.seats[name]
        number = name.split("_")[1]
        out = m.status.get(name) != "active" or name in m.victims
        byz = m.roles.get(name) == "byzantine"
        gone = state.result and name == m.ejected
        if speaking:
            self._circle(d, (x, y), 41, outline=(52, 84, 130), width=1.5)
            self._circle(d, (x, y), 35, outline=ACCENT, width=2)
        self._circle(d, (x, y), 28, fill=(30, 34, 42) if out or gone else SEAT, outline=ACCENT if speaking else (BYZANTINE if byz else SEAT_OUT), width=4 if speaking else (3 if byz else 2))
        self._text(d, (x, y), number, "seat", MUTED if out or gone else INK, "mm")
        if out:
            d.line([self._p(x - 14), self._p(y - 14), self._p(x + 14), self._p(y + 14)], fill=BYZANTINE if name in m.victims else MUTED, width=self._p(3))
            d.line([self._p(x - 14), self._p(y + 14), self._p(x + 14), self._p(y - 14)], fill=BYZANTINE if name in m.victims else MUTED, width=self._p(3))
        if byz:
            self._badge(d, (x + 22, y - 22))
        caption = name
        if name in m.victims:
            caption = f"{name} found dead"
        elif gone:
            caption = f"{name} ejected"
        elif m.status.get(name) == "ejected":
            caption = f"{name} ejected earlier"
        self._text(d, (x, y + 38), caption, "label", BYZANTINE if name in m.victims else (MUTED if out or gone else INK), "ma")
        if speaking and utterance is not None:
            tail = last_words(utterance.visible(t), self.f["small"], 150 * SCALE, 2)
            for i, line in enumerate(tail):
                self._text(d, (x, y + 54 + i * 14), line, "small", ACCENT, "ma")
        if votes:
            self._circle(d, (x - 24, y - 22), 11, fill=ACCENT)
            self._text(d, (x - 24, y - 22), str(votes), "badge", BG, "mm")

    def _transcript(self, d: ImageDraw.ImageDraw, state: State) -> None:
        m, t = self.m, state.t
        started = [u for u in m.utterances if u.start_s <= t]
        top, bottom = 108.0, 612.0
        y = bottom
        line_h = 17.0
        for u in reversed(started):
            shown = wrap(" ".join(u.visible(t)), self.f["text"], 690 * SCALE)
            height = 19 + line_h * len(shown) + 8
            if y - height < top:
                break
            y -= height
            fresh = t - u.end_s < 20
            speaking = u.speaking(t)
            head = f"{u.author}   {u.start_s:.0f} s"
            self._text(d, (856, y), head, "head", ACCENT if speaking else INK)
            if m.roles.get(u.author) == "byzantine":
                self._badge(d, (856 + self.f["head"].getlength(head) / SCALE + 14, y + 8))
            for i, line in enumerate(shown):
                self._text(d, (856, y + 19 + i * line_h), line, "text", INK if fresh or speaking else MUTED)

    def _lanes(self, d: ImageDraw.ImageDraw, state: State) -> None:
        m, t = self.m, state.t
        for u in m.utterances:
            if u.start_s > t or u.author not in self.speakers:
                continue
            row = self.speakers.index(u.author)
            y = LANES_TOP + row * LANE_H
            x0, x1 = self._lane_x(u.start_s), self._lane_x(min(u.end_s, t))
            self._box(d, x0, y + 3, max(x1, x0 + 2), y + 15, fill=ACCENT if u.speaking(t) else ACCENT_DONE, radius=3)
        if not (state.votes_shown or state.result):
            x = self._lane_x(t)
            d.line([self._p(x), self._p(LANES_TOP - 4), self._p(x), self._p(LANES_TOP + len(self.speakers) * LANE_H)], fill=INK, width=self._p(2))


def render_video(renderer: Renderer, timeline: Timeline, path: Path, fps: int = 20) -> None:
    """Writes the replay as an H.264 video, by piping frames to ffmpeg."""
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = int(timeline.duration * fps)
    for i in range(frames):
        process.stdin.write(renderer.frame(timeline.at(i / fps)).tobytes())
        if i % 200 == 0:
            logger.info("frame {} of {}", i, frames)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    logger.info("wrote {}", path)


def render_gif(video: Path, path: Path, start: float, seconds: float, width: int = 960, fps: int = 8) -> None:
    """Cuts a short GIF from the video, with a palette made for it."""
    filters = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-t", str(seconds), "-i", str(video), "-vf", filters, str(path)], check=True)
    logger.info("wrote {}", path)


def parse_still(token: str, meeting: Meeting) -> tuple[str, State]:
    """Turns a still request into a name and a state.

    Args:
        token: A meeting time in seconds, `votes` for the moment all votes are drawn, or `result`.
        meeting: The meeting.

    Returns:
        The name for the file and the state to draw.
    """
    if token == "votes":
        return "votes", State(meeting.closed_at, votes_shown=len(meeting.votes))
    if token == "result":
        return "result", State(meeting.closed_at, votes_shown=len(meeting.votes), result=True)
    return f"t{int(float(token)):03d}", State(float(token))


def main(argv: Sequence[str] | None = None) -> None:
    """Renders the still frames and the video of one meeting.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", type=Path, help="a Game_<id>_Run0 folder")
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--speed", type=float, default=2.5, help="meeting seconds per video second")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stills", default="0,2,85,178,votes,result", help="comma-separated meeting times, `votes` or `result`")
    parser.add_argument("--stem", default=None, help="start of the output file names, by default made from the game")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--gif", default="2,24", help="start and length in video seconds of the GIF, or empty for none")
    args = parser.parse_args(argv)
    meeting = load_meeting(args.game, args.round)
    renderer = Renderer(meeting)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"situation-room-{meeting.game_id}-round{meeting.round}"
    for token in filter(None, args.stills.split(",")):
        name, state = parse_still(token, meeting)
        path = args.out / f"{stem}-{name}.png"
        renderer.frame(state).save(path, optimize=True)
        logger.info("wrote {}", path)
    if not args.no_video:
        video = args.out / f"{stem}.mp4"
        render_video(renderer, Timeline(meeting.closed_at, len(meeting.votes), speed=args.speed), video, args.fps)
        if args.gif:
            start, seconds = (float(x) for x in args.gif.split(","))
            render_gif(video, args.out / f"{stem}.gif", start, seconds)


if __name__ == "__main__":
    main()
