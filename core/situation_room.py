"""The situation room, a shared record of everything agents say during a timed discussion.

The room is a grow-only map of per-author lanes. Each lane has a single writer and
only the controller commits to it, so the room needs no locks. A reader's view is
a rendering of the lanes up to a point in room time. Room time is virtual, so every
model is charged the same time per word whatever its inference speed.

Design notes are in docs/situation-room.md.
"""

import json
import random
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

SILENT = "<silent/>"
INAUDIBLE = "[inaudible]"
OVERLAP_MODES = ("clean", "interleaved", "noisy")
TAG_POLICIES = ("every_chunk", "first_chunk")
_EPSILON = 1e-9
_TO_TAG = re.compile(r"<to>(.*?)</to>", re.IGNORECASE | re.DOTALL)
_SILENT_TAG = re.compile(r"<silent\s*/?>", re.IGNORECASE)


@dataclass(frozen=True)
class RoomConfig:
    """Settings of a timed discussion.

    Attributes:
        budget_s: Length of the discussion in room time.
        tick_s: Time between decision points.
        words_per_s: Speaking rate in the text modality.
        max_words: Words kept from one utterance. Later words are dropped.
        onset_jitter_s: Upper bound of the seeded delay before a speaker starts within a tick.
        quiet_close_ticks: Close early after this many silent ticks, 0 to disable.
        overlap: How speech is rendered, one of OVERLAP_MODES.
        window_s: Width of one window in the interleaved renderings.
        dropout: Fraction of words replaced by INAUDIBLE in noisy mode.
        tag_policy: Speaker tags on every chunk or only a speaker's first, in noisy mode.
        barge_in: Whether an agent that is mid-utterance may speak again at a tick.
        seed: Seed for onsets, shuffles and dropout.
    """

    budget_s: float = 180.0
    tick_s: float = 10.0
    words_per_s: float = 2.5
    max_words: int = 40
    onset_jitter_s: float = 2.0
    quiet_close_ticks: int = 3
    overlap: str = "clean"
    window_s: float = 2.0
    dropout: float = 0.0
    tag_policy: str = "every_chunk"
    barge_in: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks that the settings are usable.

        Raises:
            ValueError: If a setting is out of range or not one of the allowed values.
        """
        if self.overlap not in OVERLAP_MODES:
            raise ValueError(f"overlap must be one of {OVERLAP_MODES}, got {self.overlap!r}")
        if self.tag_policy not in TAG_POLICIES:
            raise ValueError(f"tag_policy must be one of {TAG_POLICIES}, got {self.tag_policy!r}")
        if min(self.budget_s, self.tick_s, self.words_per_s, self.window_s) <= 0 or self.max_words < 1:
            raise ValueError("budget_s, tick_s, words_per_s, window_s and max_words must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if self.onset_jitter_s < 0 or self.quiet_close_ticks < 0:
            raise ValueError("onset_jitter_s and quiet_close_ticks must not be negative")


@dataclass(frozen=True)
class Meeting:
    """What the agents are told when a meeting starts.

    Attributes:
        round_num: Game round in which the meeting happens.
        reason: Why the meeting was called.
        caller: Agent who called the meeting, if any.
    """

    round_num: int
    reason: str = ""
    caller: str | None = None


@dataclass(frozen=True)
class Tick:
    """A decision point in a timed discussion.

    Attributes:
        tick: Tick number, starting at 1.
        t: Room time of the tick.
        remaining_s: Room time left in the discussion.
    """

    tick: int
    t: float
    remaining_s: float


@dataclass(frozen=True)
class Percept:
    """What one agent hears at a tick.

    Attributes:
        reader: Agent who hears.
        tick: Tick number.
        t: Room time of the percept.
        cursor: Room time up to which the reader has now heard.
        modality: Kind of percept, text in v1.
        text: The rendering of everything the reader has heard so far.
        audio_ref: Path of the reader's mix for the audio modality.
        speakers: Diarization labels the reader may see in the audio modality.
    """

    reader: str
    tick: int
    t: float
    cursor: float
    modality: str
    text: str
    audio_ref: str | None = None
    speakers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reply:
    """An agent's answer to a tick.

    Attributes:
        agent: Agent who replies.
        tick: Tick number the reply answers.
        text: What the agent says, or None for silence.
        addressed_to: Agents the reply is addressed to.
    """

    agent: str
    tick: int
    text: str | None
    addressed_to: tuple[str, ...] = ()


@dataclass(frozen=True)
class Vote:
    """An agent's vote at the end of a discussion.

    Attributes:
        agent: Agent who votes.
        target: Name of the agent voted for, or SKIP.
    """

    agent: str
    target: str


def parse_reply(agent: str, tick: int, raw: str | None, roster: list[str]) -> Reply:
    """Turns a model's raw output into a reply.

    A reply with the silent tag, or with no words, is silence. `<to>Name</to>` tags
    are removed from the text and become `addressed_to`, keeping only names in the roster.

    Args:
        agent: Agent who replies.
        tick: Tick number.
        raw: The raw output, or None.
        roster: Names of every agent in the room.

    Returns:
        The parsed reply.
    """
    if raw is None or _SILENT_TAG.search(raw):
        return Reply(agent, tick, None)
    by_lower = {name.lower(): name for name in roster}
    addressed: list[str] = []
    for tag in _TO_TAG.findall(raw):
        for part in tag.split(","):
            name = by_lower.get(part.strip().lower())
            if name and name not in addressed:
                addressed.append(name)
    text = " ".join(_TO_TAG.sub("", raw).split())
    return Reply(agent, tick, text or None, tuple(addressed))


def format_time(seconds: float) -> str:
    """Formats room time as MM:SS.

    Args:
        seconds: Room time.

    Returns:
        The time rounded down to whole seconds.
    """
    whole = int(seconds + _EPSILON)
    return f"{whole // 60:02d}:{whole % 60:02d}"


@dataclass(frozen=True)
class Utterance:
    """One thing an agent said, placed on the room's timeline.

    Words are spoken at an even pace from `start_s`.

    Attributes:
        author: Agent who spoke.
        tick: Tick at which the agent decided to speak.
        start_s: Room time at which the first word begins.
        pace_s: Seconds per word.
        words: The words actually spoken, after the word limit and any cut-off.
        intended: Number of words the agent wanted to say.
        cursor: Room time up to which the author had heard when it spoke.
        addressed_to: Agents the utterance is addressed to.
    """

    author: str
    tick: int
    start_s: float
    pace_s: float
    words: tuple[str, ...]
    intended: int
    cursor: float = 0.0
    addressed_to: tuple[str, ...] = ()

    @property
    def end_s(self) -> float:
        """Room time at which the last word ends."""
        return self.start_s + len(self.words) * self.pace_s

    @property
    def text(self) -> str:
        """The spoken words as text."""
        return " ".join(self.words)

    def word_end(self, index: int) -> float:
        """Gives the room time at which a word ends.

        Args:
            index: Position of the word, starting at 0.

        Returns:
            The end time of that word.
        """
        return self.start_s + (index + 1) * self.pace_s

    def heard_count(self, t: float) -> int:
        """Counts the words that have been completely spoken by a time.

        Args:
            t: Room time.

        Returns:
            Number of words whose end is at or before `t`.
        """
        if t <= self.start_s:
            return 0
        return min(len(self.words), int((t - self.start_s) / self.pace_s + _EPSILON))

    def cut(self, at_s: float) -> "Utterance":
        """Stops the utterance at a time, keeping only the words completed by then.

        Args:
            at_s: Room time at which the speaker stops.

        Returns:
            A shorter utterance, or this one if it already ends by `at_s`.
        """
        if self.end_s <= at_s + _EPSILON:
            return self
        return replace(self, words=self.words[: self.heard_count(at_s)])


@dataclass
class Lane:
    """The utterances of one author, in order. Only the controller appends to it.

    Attributes:
        author: Agent whose speech this is.
        utterances: Utterances in the order they were committed.
    """

    author: str
    utterances: list[Utterance] = field(default_factory=list)

    @property
    def free_at(self) -> float:
        """Room time at which the author finishes speaking."""
        return self.utterances[-1].end_s if self.utterances else 0.0


class Room:
    """The record of one discussion and the renderings of it for each reader."""

    def __init__(self, agents: list[str], config: RoomConfig) -> None:
        """Creates an empty room.

        Args:
            agents: Names of the agents in the room, in a fixed order.
            config: Settings of the discussion.
        """
        self.config = config
        self.agents = list(agents)
        self.lanes = {name: Lane(name) for name in agents}
        self.closed_at: float | None = None
        self.ticks = 0
        self.closed_early = False
        self.percept_log: list[dict] = []

    def free_at(self, author: str) -> float:
        """Gives the room time at which an author finishes speaking.

        Args:
            author: Agent name.

        Returns:
            Room time, 0 if the author has not spoken.
        """
        return self.lanes[author].free_at

    def append(
        self,
        author: str,
        text: str,
        start_s: float,
        tick: int,
        cursor: float = 0.0,
        addressed_to: tuple[str, ...] = (),
        duration_s: float | None = None,
    ) -> Utterance:
        """Commits an utterance to the author's lane.

        The room keeps the first `max_words` words. If the author is still speaking at
        `start_s`, the earlier utterance is cut there when barge-in is on.

        Args:
            author: Agent who speaks.
            text: What the agent says.
            start_s: Room time at which the first word begins.
            tick: Tick at which the agent decided to speak.
            cursor: Room time up to which the author had heard.
            addressed_to: Agents the utterance is addressed to.
            duration_s: Length of the whole text when it is known, for example from synthesized
                speech. The text modality uses `words_per_s` when this is None.

        Returns:
            The committed utterance.

        Raises:
            ValueError: If the text has no words, or the author is still speaking and barge-in is off.
        """
        words = text.split()
        if not words:
            raise ValueError("an utterance needs at least one word")
        lane = self.lanes[author]
        if start_s < lane.free_at - _EPSILON:
            if not self.config.barge_in:
                raise ValueError(f"{author} is still speaking at {start_s:.2f} s and barge_in is off")
            lane.utterances[-1] = lane.utterances[-1].cut(start_s)
        pace = (duration_s / len(words)) if duration_s else 1.0 / self.config.words_per_s
        utterance = Utterance(
            author=author,
            tick=tick,
            start_s=start_s,
            pace_s=pace,
            words=tuple(words[: self.config.max_words]),
            intended=len(words),
            cursor=cursor,
            addressed_to=addressed_to,
        )
        lane.utterances.append(utterance)
        return utterance

    def close(self, at_s: float) -> None:
        """Ends the discussion, cutting every utterance still in progress.

        Args:
            at_s: Room time at which the discussion ends.
        """
        for lane in self.lanes.values():
            if lane.utterances:
                lane.utterances[-1] = lane.utterances[-1].cut(at_s)
        self.closed_at = at_s

    def percept(self, reader: str, t: float, tick: int) -> Percept:
        """Builds what a reader hears at a time and logs it.

        Args:
            reader: Agent who hears.
            t: Room time.
            tick: Tick number.

        Returns:
            The percept for the reader.
        """
        text = render(self, reader, t)
        self.percept_log.append({"reader": reader, "tick": tick, "t": t, "text": text})
        return Percept(reader=reader, tick=tick, t=t, cursor=t, modality="text", text=text)

    def utterances(self) -> list[Utterance]:
        """Lists every utterance in order of start time, then author.

        Returns:
            The utterances.
        """
        everything = [u for lane in self.lanes.values() for u in lane.utterances]
        return sorted(everything, key=lambda u: (u.start_s, u.author))

    def statements(self, caller: str | None = None) -> list[dict]:
        """Lists the spoken statements in the shape the Observer classifiers read.

        Args:
            caller: Agent who called the meeting. Their first statement is marked as the report.

        Returns:
            One dict per statement with `Agent`, `Text`, `Reported` and `S_Num`, where `S_Num`
            counts the author's statements from 1.
        """
        counts: dict[str, int] = {}
        rows = []
        for utterance in self.utterances():
            if not utterance.words:
                continue
            counts[utterance.author] = counts.get(utterance.author, 0) + 1
            rows.append(
                {
                    "Agent": utterance.author,
                    "Text": utterance.text,
                    "Reported": int(utterance.author == caller and counts[utterance.author] == 1),
                    "S_Num": counts[utterance.author],
                }
            )
        return rows

    def ground_truth(self) -> list[dict]:
        """Lists every utterance with its timing, for analysis.

        Returns:
            One dict per utterance, in order of start time.
        """
        return [
            {
                "author": u.author,
                "tick": u.tick,
                "start_s": round(u.start_s, 3),
                "end_s": round(u.end_s, 3),
                "text": u.text,
                "intended_words": u.intended,
                "spoken_words": len(u.words),
                "cursor": u.cursor,
                "addressed_to": list(u.addressed_to),
            }
            for u in self.utterances()
        ]

    def write_jsonl(self, path: Path) -> None:
        """Writes the settings, the ground truth and every percept as JSON lines.

        Args:
            path: Output file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            summary = {"ticks": self.ticks, "closed_at": self.closed_at, "closed_early": self.closed_early}
            handle.write(json.dumps({"type": "room", "agents": self.agents, **summary, **asdict(self.config)}) + "\n")
            for row in self.ground_truth():
                handle.write(json.dumps({"type": "utterance", **row}) + "\n")
            for row in self.percept_log:
                handle.write(json.dumps({"type": "percept", **row}) + "\n")


def render(room: Room, reader: str, t: float) -> str:
    """Renders what a reader has heard up to a time.

    Args:
        room: The room to read.
        reader: Agent who hears.
        t: Room time.

    Returns:
        The rendering for the room's overlap mode.
    """
    if room.config.overlap == "clean":
        return render_clean(room, reader, t)
    return render_interleaved(room, reader, t)


def _name(author: str, reader: str) -> str:
    """Marks the reader's own speech."""
    return f"{author} (you)" if author == reader else author


def render_clean(room: Room, reader: str, t: float) -> str:
    """Renders one block per utterance, with its time span.

    Utterances still in progress show the words spoken so far followed by `...`.

    Args:
        room: The room to read.
        reader: Agent who hears.
        t: Room time.

    Returns:
        The blocks in order of start time, one per line.
    """
    lines = []
    for u in room.utterances():
        heard = u.heard_count(t)
        if heard == 0:
            continue
        name = _name(u.author, reader)
        if heard == len(u.words):
            lines.append(f"[{format_time(u.start_s)}-{format_time(u.end_s)}] {name}: {u.text}")
        else:
            lines.append(f"[{format_time(u.start_s)}-...  ] {name}: {' '.join(u.words[:heard])} ...")
    return "\n".join(lines)


def _masked(config: RoomConfig, reader: str, u: Utterance, index: int) -> bool:
    """Decides whether a reader fails to hear one word. The decision depends only on the seed and ids."""
    if config.overlap != "noisy" or config.dropout <= 0 or u.author == reader:
        return False
    key = f"{config.seed}|mask|{reader}|{u.author}|{u.start_s:.3f}|{index}"
    return random.Random(key).random() < config.dropout


def render_interleaved(room: Room, reader: str, t: float) -> str:
    """Renders speech in windows, so that simultaneous speakers appear on adjacent lines.

    Each line holds one speaker's words in one window of `window_s` seconds, and lines
    with the same timestamp were spoken at the same time. Only windows that have ended
    by `t` appear, so the rendering at a later time extends the earlier one. Lines in
    the same window are shuffled with a seed from the room seed, the reader and the
    window, which keeps that order fixed across ticks.

    In noisy mode some words become INAUDIBLE, and with the first_chunk policy only the
    first line of an utterance carries the speaker's name.

    Args:
        room: The room to read.
        reader: Agent who hears.
        t: Room time.

    Returns:
        The lines in order of window.
    """
    config = room.config
    tag_all = config.overlap != "noisy" or config.tag_policy == "every_chunk"
    chunks: dict[int, dict[str, list[tuple[int, str]]]] = {}
    for u in room.utterances():
        for index, word in enumerate(u.words):
            window = int((u.start_s + (index + 0.5) * u.pace_s) // config.window_s)
            if (window + 1) * config.window_s > t + _EPSILON:
                break
            shown = INAUDIBLE if _masked(config, reader, u, index) else word
            chunks.setdefault(window, {}).setdefault(u.author, []).append((index, shown))
    lines = []
    for window in sorted(chunks):
        authors = sorted(chunks[window])
        random.Random(f"{config.seed}|shuffle|{reader}|{window}").shuffle(authors)
        for author in authors:
            words: list[str] = []
            for _, shown in chunks[window][author]:
                if shown != INAUDIBLE or not words or words[-1] != INAUDIBLE:
                    words.append(shown)
            starts_utterance = any(index == 0 for index, _ in chunks[window][author])
            stamp = format_time(window * config.window_s)
            if author == reader or tag_all or starts_utterance:
                lines.append(f"[{stamp}] {_name(author, reader)}: {' '.join(words)}")
            else:
                lines.append(f"[{stamp}] {' '.join(words)}")
    return "\n".join(lines)
