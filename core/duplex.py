"""Agents that listen and speak in the same running context, with the ledger as their ears and their mouth.

The tick loop of the room rebuilds a whole prompt for each agent at each tick. Here each agent keeps one session with the
model. A beat of room time is the unit: the agent receives the chunks that others finished speaking since its last beat,
as text appended to the session, and the model writes a short burst of tokens that go to the ledger as the agent's own
chunks. The model keeps its context between beats, so only new tokens are processed.

An engine here is anything that can open a session and answer a chunk of input with a burst of output. The vLLM engine
is in `core/vllm_stream.py`, and the tests use a scripted one. The ledger decides who hears what, and the runner
makes every agent read the same snapshot at a beat before any burst is written.

Usage:
    fmt = TurnFormat.probe(render)
    agents = [DuplexAgent("Agent_1", engine.open("Agent_1", 0.7), fmt, system, intro, ledger.subscribe("Agent_1"), 2.0, 8)]
    meeting = await run_duplex_meeting(agents, ledger, budget_s=60.0, beat_s=2.0)
"""

import asyncio
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from loguru import logger

from core.ledger import CHUNK, Ledger, Subscription
from core.situation_room import format_time
from core.stream import Heard, StreamWriter, chunk_of, render_heard

SILENT_TAG = re.compile(r"<silent\s*/?>", re.IGNORECASE)
_NO_SPACE_BEFORE = ".,;:!?)'\""
_SENTINELS = ("<<SYSTEM>>", "<<FIRST_USER>>", "<<ASSISTANT>>", "<<SECOND_USER>>")


@dataclass(frozen=True)
class StreamPrompt:
    """One chunk of input for a session.

    Attributes:
        text: Text to append to the context of the model.
        audio: Audio arrays in the order of the placeholders in the text.
        sample_rate: Sample rate of the arrays.
    """

    text: str
    audio: tuple[np.ndarray, ...] = ()
    sample_rate: int = 16000


@dataclass(frozen=True)
class Burst:
    """What the model wrote in answer to one chunk of input.

    Attributes:
        text: The words written, without the last token when the engine drops it. The engine drops the token that it
            sampled last, because it has no cached keys and values yet, so the context keeps the text before it.
        finish_reason: `length` when the burst reached its token limit, `stop` when the model ended its message.
        prompt_tokens: Tokens added to the context by the input chunk, when the engine reports them.
        output_tokens: Tokens the model wrote.
        wall_s: Seconds the engine took.
    """

    text: str
    finish_reason: str | None
    prompt_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0


class StreamHandle(Protocol):
    """A session with a model that keeps its context between chunks."""

    async def send(self, prompt: StreamPrompt, max_tokens: int) -> Burst:
        """Appends a chunk of input and returns the burst the model writes after it.

        Args:
            prompt: The new input.
            max_tokens: Most tokens the burst may have.

        Returns:
            The burst.
        """

    async def close(self) -> None:
        """Ends the session and frees its context."""


@dataclass(frozen=True)
class TurnFormat:
    """The way a chat template joins turns, found by rendering sentinel messages.

    A burst is one assistant message and a chunk of input is one user message. The template writes the end of the
    assistant message and the start of the next user message between them, and a cue for the assistant after. The
    strings are cut out of a rendered conversation, so any template with those turns works.

    Attributes:
        render: Renders a list of messages as text, with or without the cue for the assistant at the end.
        gap: Text between the end of an assistant message and the content of the next user message.
        cue: Text after the content of a user message that makes the model write.
    """

    render: Callable[[list[dict], bool], str]
    gap: str
    cue: str

    @classmethod
    def probe(cls, render: Callable[[list[dict], bool], str]) -> "TurnFormat":
        """Finds the gap and the cue of a chat template.

        Args:
            render: Renders messages as text, given the messages and whether to add the cue for the assistant.

        Returns:
            The format.

        Raises:
            ValueError: If the rendering loses a sentinel or puts them out of order.
        """
        system, first, assistant, second = _SENTINELS
        text = render(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": first},
                {"role": "assistant", "content": assistant},
                {"role": "user", "content": second},
            ],
            True,
        )
        positions = [text.find(s) for s in _SENTINELS]
        if min(positions) < 0 or positions != sorted(positions):
            raise ValueError(f"the template did not keep the sentinels in order: {text!r}")
        gap = text[positions[2] + len(assistant) : positions[3]]
        cue = text[positions[3] + len(second) :]
        return cls(render, gap, cue)

    def opening(self, system: str, user: str) -> str:
        """Renders the start of a session.

        Args:
            system: System prompt.
            user: First user message.

        Returns:
            The text with the cue for the assistant at the end.
        """
        return self.render([{"role": "system", "content": system}, {"role": "user", "content": user}], True)

    def next_turn(self, user: str) -> str:
        """Renders a user message that follows a burst.

        Args:
            user: The message.

        Returns:
            The end of the previous assistant message, the message and the cue.
        """
        return f"{self.gap}{user}{self.cue}"


@dataclass
class AgentStats:
    """Counts for one agent in one meeting."""

    beats: int = 0
    spoke: int = 0
    silent: int = 0
    statements: int = 0
    words: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    engine_s: float = 0.0


@dataclass
class DuplexAgent:
    """One agent with a session, a place in the ledger and a way to speak.

    Attributes:
        name: Agent name.
        handle: The session with the model.
        fmt: How the chat template joins turns.
        system: System prompt.
        intro: First part of the first user message, which says what the meeting is and how to answer.
        subscription: The position of the agent in the ledger.
        beat_s: Room time of one beat.
        max_tokens: Most tokens in one burst.
        audio_placeholder: Text that stands for an audio array in the prompt, for a model that hears.
        round_num: Round of the game.
    """

    name: str
    handle: StreamHandle
    fmt: TurnFormat
    system: str
    intro: str
    subscription: Subscription
    beat_s: float
    max_tokens: int
    audio_placeholder: str | None = None
    round_num: int | None = None
    stats: AgentStats = field(default_factory=AgentStats)
    _started: bool = field(default=False, init=False, repr=False)
    _writer: StreamWriter | None = field(default=None, init=False, repr=False)
    _last: str = field(default="", init=False, repr=False)

    def hear(self, t: float) -> Heard:
        """Reads the chunks that others finished speaking since the last beat.

        Args:
            t: Room time at the start of the beat.

        Returns:
            The rendering of the new chunks.
        """
        records = [r for r in self.subscription.poll(until_s=t) if r.type == CHUNK]
        return render_heard([chunk_of(r) for r in records], self.name, self.audio_placeholder)

    async def respond(self, heard: Heard, t: float) -> Burst:
        """Sends what the agent heard to its session and returns the burst.

        Args:
            heard: The new chunks as text and audio.
            t: Room time at the start of the beat.

        Returns:
            The burst.
        """
        body = heard.text or f"[{format_time(t)}] ..."
        if self._started:
            text = self.fmt.next_turn(body)
        else:
            text = self.fmt.opening(self.system, f"{self.intro}\n\n{body}")
            self._started = True
        burst = await self.handle.send(StreamPrompt(text, heard.audio, heard.sample_rate or 16000), self.max_tokens)
        self.stats.beats += 1
        self.stats.prompt_tokens += burst.prompt_tokens
        self.stats.output_tokens += burst.output_tokens
        self.stats.engine_s += burst.wall_s
        return burst

    def commit(self, burst: Burst, t: float, ledger: Ledger) -> None:
        """Writes the burst to the ledger as chunks of the agent's own stream.

        Silence ends a statement that was going on. A burst that ends with `stop` ends the statement it belongs to,
        and a burst that ends with `length` leaves the statement open for the next beat. Every burst is a message of
        its own, so it starts without the space that a token inside a sentence has, and a burst that continues a
        statement gets a space in front unless it starts with a space or a closing mark.

        Args:
            burst: What the model wrote.
            t: Room time at the start of the beat.
            ledger: The ledger.
        """
        words = SILENT_TAG.sub("", burst.text)
        if not words.strip():
            self.stats.silent += 1
            self._end(t)
            return
        if self._writer is None:
            self._writer = StreamWriter(ledger, self.name, round_num=self.round_num)
            self.stats.statements += 1
            self._last = ""
        elif self._last and not self._last[-1].isspace() and not words[0].isspace() and words[0] not in _NO_SPACE_BEFORE:
            words = f" {words}"
        self._last = words
        self._writer.write(words, t, t + self.beat_s)
        self.stats.spoke += 1
        self.stats.words += len(words.split())
        if burst.finish_reason == "stop":
            self._end(t + self.beat_s)

    def _end(self, at_s: float) -> None:
        """Closes the stream of a statement that is going on."""
        if self._writer is not None:
            self._writer.close(at_s)
            self._writer = None
            self._last = ""

    def finish(self, at_s: float) -> None:
        """Closes the stream of a statement in progress when the meeting ends.

        Args:
            at_s: Room time at which the meeting ends.
        """
        self._end(at_s)


@dataclass(frozen=True)
class DuplexMeeting:
    """The result of a meeting.

    Attributes:
        ledger: The ledger with every chunk.
        beats: Number of beats played.
        wall_s: Seconds the meeting took.
        stats: Counts of each agent.
    """

    ledger: Ledger
    beats: int
    wall_s: float
    stats: dict[str, AgentStats]


async def run_duplex_meeting(
    agents: Sequence[DuplexAgent], ledger: Ledger, budget_s: float, beat_s: float
) -> DuplexMeeting:
    """Plays a meeting beat by beat.

    At each beat every agent reads the chunks that were finished before the beat, all agents answer at once, and
    the bursts are written to the ledger after all have arrived, in the order of the agents. Order of arrival has
    no effect. The sessions are closed at the end.

    Args:
        agents: The agents of the meeting.
        ledger: The ledger the agents follow and write to.
        budget_s: Room time of the meeting.
        beat_s: Room time of one beat.

    Returns:
        The meeting.
    """
    started = time.monotonic()
    t = 0.0
    beats = 0
    try:
        while t < budget_s - 1e-9:
            heard = [agent.hear(t) for agent in agents]
            bursts = await asyncio.gather(*(agent.respond(h, t) for agent, h in zip(agents, heard)))
            for agent, burst in zip(agents, bursts):
                agent.commit(burst, t, ledger)
            t += beat_s
            beats += 1
            logger.debug("beat {} at {:.0f} s: {} chunks in the ledger", beats, t, len(ledger))
        for agent in agents:
            agent.finish(min(t, budget_s))
    finally:
        await asyncio.gather(*(agent.handle.close() for agent in agents), return_exceptions=True)
    return DuplexMeeting(ledger, beats, time.monotonic() - started, {a.name: a.stats for a in agents})

