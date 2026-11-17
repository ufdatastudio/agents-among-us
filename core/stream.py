"""Streams of speech in the ledger: chunks written while a speaker talks, and the append-only rendering a listener reads.

A statement is no longer one record written when it ends. The speaker appends chunks to the ledger as it goes, each with
the words (or the slice of audio) of a short stretch of room time, and a last chunk marks the end. A chunk is a record
of type `CHUNK` whose `time_s` is the time it was fully spoken, so a subscription returns it only after that.

A listener turns the new chunks into text with `render_heard`. The rendering has no state: the lines it gives for one
batch do not depend on earlier batches, so the text a model has already read is never rewritten and its cached keys
and values stay valid.

Usage:
    writer = StreamWriter(ledger, "Agent_3", round_num=1)
    writer.write("I was in", 12.0, 13.0)
    writer.write("Shields with Agent_2", 13.0, 14.0)
    writer.close(14.0)
    heard = ledger.subscribe("Agent_5")
    render_heard([chunk_of(r) for r in heard.poll(until_s=14.0)], "Agent_5").text
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.audio.mix import read_wav
from core.ledger import CHUNK, Ledger, Record
from core.situation_room import format_time

TEXT = "text"
AUDIO = "audio"
IMAGE = "image"
MODALITIES = (TEXT, AUDIO, IMAGE)
_EPSILON = 1e-9


@dataclass(frozen=True)
class Chunk:
    """One piece of a stream, read back from a ledger record.

    Attributes:
        record_id: Id of the record in the ledger.
        stream: Name of the stream, which is one statement of one author.
        seq: Position in the stream, from 0.
        author: Who is speaking.
        modality: One of `MODALITIES`.
        start_s: Room time at which the piece begins.
        end_s: Room time at which the piece ends, which is when a listener can hear all of it.
        data: The text exactly as spoken, so that the chunks of a stream joined without a separator give the statement.
            For audio a dict with `path`, `rate`, `start` and `count`, and `text` in the same form when the words are known.
        final: Whether this chunk ends the stream.
    """

    record_id: int
    stream: str
    seq: int
    author: str
    modality: str
    start_s: float
    end_s: float
    data: str | dict
    final: bool

    @property
    def words(self) -> str:
        """The words of the chunk, or an empty string when an audio chunk has none."""
        if isinstance(self.data, dict):
            return str(self.data.get("text", ""))
        return str(self.data)


def chunk_of(record: Record) -> Chunk:
    """Reads a chunk record.

    Args:
        record: A ledger record of type `CHUNK`.

    Returns:
        The chunk.

    Raises:
        ValueError: If the record is not a chunk.
    """
    if record.type != CHUNK:
        raise ValueError(f"record {record.id} is a {record.type}, not a chunk")
    c = record.content
    return Chunk(
        record.id, c["stream"], c["seq"], record.author, c["modality"], c["start_s"],
        record.time_s if record.time_s is not None else c["start_s"], c["data"], c["final"],
    )


class StreamWriter:
    """Appends the chunks of one statement to a ledger."""

    def __init__(
        self,
        ledger: Ledger,
        author: str,
        modality: str = TEXT,
        *,
        round_num: int | None = None,
        visible_to: Iterable[str] | None = None,
    ) -> None:
        """Opens a stream. The stream is named after the author and the number of streams the author has begun.

        Args:
            ledger: The ledger to write to.
            author: Who is speaking.
            modality: One of `MODALITIES`.
            round_num: Round of the game.
            visible_to: Agents who may hear it, or None for everyone.

        Raises:
            ValueError: If the modality is unknown.
        """
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        begun = sum(1 for r in ledger.records if r.type == CHUNK and r.author == author and r.content["seq"] == 0)
        self.ledger = ledger
        self.author = author
        self.modality = modality
        self.name = f"{author}/{begun + 1}"
        self.closed = False
        self._seq = 0
        self._round = round_num
        self._visible = None if visible_to is None else tuple(visible_to)

    def write(self, data: str | dict, start_s: float, end_s: float, *, final: bool = False) -> Record:
        """Appends a chunk.

        Args:
            data: Words for text, or the audio reference of `audio_data`.
            start_s: Room time at which the piece begins.
            end_s: Room time at which it ends.
            final: Whether this chunk ends the stream.

        Returns:
            The record.

        Raises:
            ValueError: If the stream is closed or the piece ends before it begins.
        """
        if self.closed:
            raise ValueError(f"stream {self.name} is closed")
        if end_s < start_s - _EPSILON:
            raise ValueError(f"a chunk cannot end at {end_s} before it begins at {start_s}")
        record = self.ledger.append(
            CHUNK, self.author,
            {"stream": self.name, "seq": self._seq, "modality": self.modality, "start_s": start_s, "data": data, "final": final},
            round_num=self._round, time_s=end_s, visible_to=self._visible,
        )
        self._seq += 1
        self.closed = final
        return record

    def close(self, at_s: float) -> Record:
        """Ends the stream with an empty final chunk.

        Args:
            at_s: Room time at which the speaker stopped.

        Returns:
            The record.
        """
        return self.write("" if self.modality == TEXT else {}, at_s, at_s, final=True)


def audio_data(path: Path, rate: int, start: int, count: int, text: str = "") -> dict:
    """Describes a slice of a WAV file as the data of an audio chunk.

    Args:
        path: The WAV file.
        rate: Sample rate of the file.
        start: First sample of the slice.
        count: Number of samples in the slice.
        text: The words of the slice, when they are known.

    Returns:
        The data dict. The file is read only when `read_audio` is called.
    """
    return {"path": str(path), "rate": rate, "start": start, "count": count, "text": text}


def read_audio(data: dict) -> np.ndarray:
    """Reads the samples that an audio chunk refers to.

    Args:
        data: The data of an audio chunk.

    Returns:
        Mono float32 samples.

    Raises:
        ValueError: If the file has a different rate from the one the chunk states.
    """
    samples, rate = read_wav(Path(data["path"]))
    if rate != data["rate"]:
        raise ValueError(f"{data['path']} has rate {rate}, the chunk says {data['rate']}")
    return samples[data["start"] : data["start"] + data["count"]]


def audio_pieces(path: Path, start_s: float, chunk_s: float, text: str) -> list[tuple[float, float, dict]]:
    """Cuts a WAV file into pieces of equal length and shares the words out among them.

    Every share after the first begins with a space, as text chunks do.

    Args:
        path: The WAV file.
        start_s: Room time at which the clip begins.
        chunk_s: Length of one piece in seconds.
        text: The words of the clip.

    Returns:
        For each piece its start, its end and the data of an audio chunk.
    """
    samples, rate = read_wav(path)
    per_chunk = max(1, round(chunk_s * rate))
    starts = list(range(0, len(samples), per_chunk))
    words = text.split()
    pieces = []
    for index, first in enumerate(starts):
        count = min(per_chunk, len(samples) - first)
        share = " ".join(words[index * len(words) // len(starts) : (index + 1) * len(words) // len(starts)])
        share = f" {share}" if share and index else share
        begin = start_s + first / rate
        pieces.append((begin, begin + count / rate, audio_data(path, rate, first, count, share)))
    return pieces


def write_audio_stream(
    ledger: Ledger,
    author: str,
    path: Path,
    start_s: float,
    chunk_s: float = 1.0,
    text: str = "",
    *,
    round_num: int | None = None,
    visible_to: Iterable[str] | None = None,
) -> StreamWriter:
    """Writes a spoken clip as audio chunks of equal length, one after another.

    Args:
        ledger: The ledger to write to.
        author: Who is speaking.
        path: The WAV file of the clip.
        start_s: Room time at which the clip begins.
        chunk_s: Length of one chunk in seconds.
        text: The words of the clip.
        round_num: Round of the game.
        visible_to: Agents who may hear it, or None for everyone.

    Returns:
        The writer, closed.
    """
    writer = StreamWriter(ledger, author, AUDIO, round_num=round_num, visible_to=visible_to)
    pieces = audio_pieces(path, start_s, chunk_s, text)
    for begin, end, data in pieces:
        writer.write(data, begin, end)
    writer.close(pieces[-1][1] if pieces else start_s)
    return writer


def write_replay(
    ledger: Ledger,
    clips: Sequence[tuple[str, Path, float, str]],
    chunk_s: float = 1.0,
    *,
    round_num: int | None = None,
) -> list[StreamWriter]:
    """Writes recorded clips of several speakers as audio streams, with the chunks in the order they were spoken.

    A subscription returns records in id order and holds back a record that is not audible yet, so chunks of speakers
    who overlap must be appended in the order of their end times. This writes them that way.

    Args:
        ledger: The ledger to write to.
        clips: For each clip the author, the WAV file, the room time at which it begins and its words.
        chunk_s: Length of one chunk in seconds.
        round_num: Round of the game.

    Returns:
        One closed writer for each clip, in the order of `clips`.
    """
    writers = [StreamWriter(ledger, author, AUDIO, round_num=round_num) for author, _, _, _ in clips]
    events = []
    for index, (_, path, start_s, text) in enumerate(clips):
        pieces = audio_pieces(path, start_s, chunk_s, text)
        for begin, end, data in pieces:
            events.append((end, index, begin, data, False))
        events.append((pieces[-1][1] if pieces else start_s, index, pieces[-1][1] if pieces else start_s, {}, True))
    for end, index, begin, data, final in sorted(events, key=lambda e: (e[0], e[1], e[4])):
        writers[index].write(data, begin, end, final=final)
    return writers


@dataclass(frozen=True)
class Heard:
    """What a listener hears in one batch of chunks.

    Attributes:
        text: Lines of text, ready to append to a prompt. Empty when there was nothing to hear.
        audio: The audio arrays in the order of the placeholders in the text.
        sample_rate: Sample rate of the arrays, or 0 when there is no audio.
        chunks: The chunks that went into the text.
    """

    text: str
    audio: tuple[np.ndarray, ...] = ()
    sample_rate: int = 0
    chunks: tuple[Chunk, ...] = ()


def render_heard(chunks: Sequence[Chunk], reader: str, audio_placeholder: str | None = None) -> Heard:
    """Renders a batch of new chunks as lines to append to a listener's prompt.

    Chunks of the same author in the batch join into one line, with whitespace tidied. A listener never hears its own
    chunks, because its own words are already in its context. Text chunks give their words. Audio chunks give their
    placeholder with the audio array when the model can hear, and their words when it cannot, and nothing when they
    are unknown. A stream's empty final chunk adds nothing.

    Args:
        chunks: The new chunks, in ledger order.
        reader: Agent who listens.
        audio_placeholder: Text that stands for one audio array in the prompt of a model that takes audio, or None.

    Returns:
        The rendering. A line looks like `[00:12] Agent_5: words`.

    Raises:
        ValueError: If the audio chunks have different sample rates.
    """
    order: list[str] = []
    parts: dict[str, list[str]] = {}
    starts: dict[str, float] = {}
    arrays: dict[str, list[np.ndarray]] = {}
    used: dict[str, list[Chunk]] = {}
    rates: set[int] = set()
    for chunk in chunks:
        if chunk.author == reader:
            continue
        if chunk.modality == AUDIO and audio_placeholder is not None and isinstance(chunk.data, dict) and chunk.data:
            piece = f" {audio_placeholder} "
            array = read_audio(chunk.data)
            rates.add(chunk.data["rate"])
        elif chunk.words.strip():
            piece, array = chunk.words, None
        else:
            continue
        if chunk.author not in parts:
            order.append(chunk.author)
            parts[chunk.author] = []
            starts[chunk.author] = chunk.start_s
        parts[chunk.author].append(piece)
        used.setdefault(chunk.author, []).append(chunk)
        if array is not None:
            arrays.setdefault(chunk.author, []).append(array)
    lines = [f"[{format_time(starts[a])}] {a}: {' '.join(''.join(parts[a]).split())}" for a in order]
    if len(rates) > 1:
        raise ValueError(f"the audio chunks have different sample rates: {sorted(rates)}")
    audio = tuple(array for author in order for array in arrays.get(author, []))
    return Heard("\n".join(lines), audio, next(iter(rates), 0), tuple(c for author in order for c in used[author]))


def stream_texts(ledger: Ledger) -> dict[str, str]:
    """Joins the words of every stream in a ledger.

    Args:
        ledger: The ledger to read.

    Returns:
        The words of each stream with whitespace tidied, keyed by stream name, in the order the streams began.
    """
    texts: dict[str, list[str]] = {}
    for record in ledger.records:
        if record.type == CHUNK:
            chunk = chunk_of(record)
            texts.setdefault(chunk.stream, []).append(chunk.words)
    return {name: " ".join("".join(parts).split()) for name, parts in texts.items()}
