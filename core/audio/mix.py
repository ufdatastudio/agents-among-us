"""Per-reader audio mixing for the situation room.

Every utterance is a clip placed on the room's timeline. A reader hears the sum of
all other speakers' clips up to the current room time, so overlapping speech is
acoustic overlap and nothing is cut or reordered.
"""

import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000


@dataclass(frozen=True)
class Clip:
    """One utterance as audio on the room's timeline.

    Attributes:
        speaker: Name of the agent who spoke.
        start_s: Room time at which the clip begins.
        samples: Mono float32 samples in [-1, 1] at SAMPLE_RATE.
    """

    speaker: str
    start_s: float
    samples: np.ndarray

    @property
    def duration_s(self) -> float:
        """Length of the clip in seconds."""
        return len(self.samples) / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        """Room time at which the clip ends."""
        return self.start_s + self.duration_s


def mix_for_reader(
    clips: Sequence[Clip],
    reader: str,
    until_s: float,
    gains_db: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Mixes what a reader hears from room time 0 up to a given time.

    The reader's own voice is left out, which takes the place of marking one's own
    words in the text modality. A clip still playing at `until_s` is cut there.
    If the sum would clip, the whole mix is scaled so the relative levels stay.

    Args:
        clips: Every clip spoken so far.
        reader: Name of the listening agent.
        until_s: Room time up to which the reader hears.
        gains_db: Optional gain per speaker in decibels.

    Returns:
        A float32 array of `round(until_s * SAMPLE_RATE)` samples.
    """
    length = round(until_s * SAMPLE_RATE)
    mix = np.zeros(length, dtype=np.float32)
    for clip in clips:
        if clip.speaker == reader or clip.start_s >= until_s:
            continue
        offset = round(clip.start_s * SAMPLE_RATE)
        end = min(offset + len(clip.samples), length)
        gain = 10 ** ((gains_db or {}).get(clip.speaker, 0.0) / 20)
        mix[offset:end] += clip.samples[: end - offset] * gain
    peak = float(np.max(np.abs(mix))) if length else 0.0
    if peak > 1.0:
        mix /= peak
    return mix


def overlap_fraction(clips: Sequence[Clip]) -> float:
    """Computes the share of speaking time during which 2 or more clips play.

    Args:
        clips: The clips on the timeline.

    Returns:
        A value from 0 to 1, or 0 when nobody speaks.
    """
    events = sorted([(c.start_s, 1) for c in clips] + [(c.end_s, -1) for c in clips])
    active = 0
    speaking = 0.0
    overlapped = 0.0
    previous = 0.0
    for time_s, change in events:
        if active >= 1:
            speaking += time_s - previous
        if active >= 2:
            overlapped += time_s - previous
        active += change
        previous = time_s
    return overlapped / speaking if speaking else 0.0


def speakers_at(clips: Sequence[Clip], time_s: float, exclude: str | None = None) -> list[str]:
    """Lists the speakers whose clips cover a point in room time.

    Args:
        clips: The clips on the timeline.
        time_s: Room time to look at.
        exclude: A speaker to leave out, normally the reader.

    Returns:
        Speaker names, in clip order.
    """
    return [c.speaker for c in clips if c.speaker != exclude and c.start_s <= time_s <= c.end_s]


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Writes mono float samples as a 16-bit PCM WAV file.

    Args:
        path: Output file.
        samples: Float samples in [-1, 1].
        sample_rate: Sampling rate in hertz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Reads a mono 16-bit PCM WAV file.

    Args:
        path: Input file.

    Returns:
        Float32 samples in [-1, 1] and the sampling rate.
    """
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32767, rate
