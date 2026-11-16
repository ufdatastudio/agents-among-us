"""Speech clips for a recorded game: the words to say, the clips that fit their time slots and the track that places them.

The agents of a text game never heard a voice. A viewer can, so each logged statement is spoken in the voice of its
author and placed on the clock of its meeting. The synthesizer is passed in, so this module needs no speech model and
the tests use a fake one. `core/audio/tts.py` has the Kokoro synthesizer that the cluster job passes.

Usage:
    clips = synthesize_utterances(utterances, voices, synth, out_dir)
    track = place_clips(clips, length_s=480.0, pans={"Agent_1": -0.6})
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from core.audio.mix import SAMPLE_RATE, read_wav, write_wav

BASE_SPEED = 0.85
MAX_SPEED = 1.5
MIN_SLOT_S = 1.2
SPEED_TRIES = 3
KNEE = 0.6
OVERRUN = 1.3
OVERRUN_S = 0.5
FADE_S = 0.05
INDEX = "speech.json"
AGENT = re.compile(r"\bAgent_(\d+)\b")
JOINED = re.compile(r"(?<=[a-z])(?=[A-Z])")
OXYGEN = re.compile(r"\bO2\b")
UNSAYABLE = re.compile(r"[^\w\s.,;:!?'\"%()$&/+-]")


class Synthesizer(Protocol):
    """Anything that turns text into mono float samples at `SAMPLE_RATE`."""

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Speaks a text."""


@dataclass(frozen=True)
class Utterance:
    """A statement to be spoken.

    Attributes:
        round_num: Round of the meeting it belongs to.
        index: Position among the statements of that meeting, from 0.
        author: Who spoke.
        start_s: Meeting time at which the first word began.
        end_s: Meeting time at which the last word ended, which is the end of the slot the clip must fit.
        text: The words, as logged.
    """

    round_num: int
    index: int
    author: str
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class SpeechClip:
    """A spoken statement.

    Attributes:
        utterance: What was said and when.
        voice: Voice that said it.
        speed: Speed that made it fit its slot.
        file: Name of the WAV file, in the folder of the speech.
        samples: Mono float32 samples at `SAMPLE_RATE`.
        natural_s: Seconds the speech took at that speed before any cut.
    """

    utterance: Utterance
    voice: str
    speed: float
    file: str
    samples: np.ndarray
    natural_s: float = 0.0

    @property
    def duration_s(self) -> float:
        """Length of the clip in seconds."""
        return len(self.samples) / SAMPLE_RATE


def spoken_form(text: str) -> str:
    """Rewrites a statement so a voice says it as a person would.

    Agent names lose the underscore, room names written as one word are split, `O2` is read as two letters and a
    digit, and characters a voice cannot say are dropped.

    Args:
        text: The words as logged.

    Returns:
        The words to synthesize.
    """
    text = AGENT.sub(r"Agent \1", text)
    text = JOINED.sub(" ", text)
    text = OXYGEN.sub("O 2", text)
    text = UNSAYABLE.sub(" ", text.replace("_", " "))
    return " ".join(text.split())


def fit_clip(
    synth: Synthesizer, text: str, voice: str, slot_s: float, limit_s: float | None = None
) -> tuple[np.ndarray, float, float]:
    """Speaks a text so that it ends inside its time slot.

    The slot is at least `MIN_SLOT_S`, because a statement of one word is logged as lasting under half a second and
    no voice says it in that time. The text is spoken at `BASE_SPEED` first. While the clip runs longer than the slot,
    the speed rises by the ratio, up to `MAX_SPEED` and `SPEED_TRIES` tries, since pauses do not shrink with speed.
    A clip that still runs past the slot may play on until `limit_s`, which keeps a slow voice from losing its last
    words. A clip longer than that is cut with a short fade.

    Args:
        synth: The synthesizer.
        text: The words to say, already in spoken form.
        voice: Voice name.
        slot_s: Seconds the statement was logged as lasting.
        limit_s: Longest the clip may play, or None for the slot.

    Returns:
        The samples, the speed that was used and the seconds the speech took before any cut.
    """
    slot_s = max(slot_s, MIN_SLOT_S)
    speed = BASE_SPEED
    samples = synth.synthesize(text, voice, speed)
    for _ in range(SPEED_TRIES):
        length_s = len(samples) / SAMPLE_RATE
        if length_s <= slot_s or speed >= MAX_SPEED:
            break
        speed = min(MAX_SPEED, speed * length_s / slot_s * 1.02)
        samples = synth.synthesize(text, voice, speed)
    natural_s = len(samples) / SAMPLE_RATE
    limit = round(max(slot_s, limit_s or 0.0) * SAMPLE_RATE)
    if 0 < limit < len(samples):
        samples = samples[:limit].copy()
        fade = min(len(samples), round(FADE_S * SAMPLE_RATE))
        samples[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return samples.astype(np.float32), speed, natural_s


def synthesize_utterances(
    utterances: Sequence[Utterance], voices: Mapping[str, str], synth: Synthesizer, out_dir: Path
) -> list[SpeechClip]:
    """Speaks every statement of a game and writes the clips and an index.

    A clip may run past the end of its statement by `OVERRUN` and `OVERRUN_S`, but never into the next statement of
    the same author, which a voice could not say at the same time.

    Args:
        utterances: The statements, in the order the video shows them.
        voices: Voice of each author.
        synth: The synthesizer.
        out_dir: Folder for the WAV files and `speech.json`. It is created.

    Returns:
        The clips, in the order of the statements.

    Raises:
        KeyError: If an author has no voice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for u in utterances:
        slot = u.end_s - u.start_s
        later = [v.start_s for v in utterances if v.round_num == u.round_num and v.author == u.author and v.start_s >= u.end_s]
        limit = slot * OVERRUN + OVERRUN_S
        if later:
            limit = min(limit, min(later) - u.start_s)
        samples, speed, natural_s = fit_clip(synth, spoken_form(u.text), voices[u.author], slot, limit)
        name = f"r{u.round_num}_{u.index:03d}_{u.author}.wav"
        write_wav(out_dir / name, samples)
        clips.append(SpeechClip(u, voices[u.author], speed, name, samples, natural_s))
    rows = [{**asdict(c.utterance), "voice": c.voice, "speed": c.speed, "file": c.file, "natural_s": c.natural_s} for c in clips]
    (out_dir / INDEX).write_text(json.dumps({"voices": dict(voices), "clips": rows}, indent=1), encoding="utf-8")
    return clips


def read_voices(folder: Path) -> dict[str, str]:
    """Reads only the voice of each author from the index of a folder of speech.

    Args:
        folder: Folder with `speech.json`.

    Returns:
        The voice of each author.

    Raises:
        FileNotFoundError: If the folder has no index.
    """
    path = folder / INDEX
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist, run scripts.world_audio first")
    return json.loads(path.read_text(encoding="utf-8"))["voices"]


def read_speech(folder: Path) -> tuple[dict[str, str], list[SpeechClip]]:
    """Reads the folder that `synthesize_utterances` wrote.

    Args:
        folder: Folder with `speech.json` and the WAV files.

    Returns:
        The voice of each author and the clips in index order.

    Raises:
        FileNotFoundError: If the folder has no index.
    """
    path = folder / INDEX
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist, run scripts.world_audio first")
    index = json.loads(path.read_text(encoding="utf-8"))
    clips = []
    for row in index["clips"]:
        samples, rate = read_wav(folder / row["file"])
        if rate != SAMPLE_RATE:
            raise ValueError(f"{row['file']} has rate {rate}, expected {SAMPLE_RATE}")
        u = Utterance(row["round_num"], row["index"], row["author"], row["start_s"], row["end_s"], row["text"])
        clips.append(SpeechClip(u, row["voice"], row["speed"], row["file"], samples, row.get("natural_s", 0.0)))
    return index["voices"], clips


def place_clips(
    placements: Sequence[tuple[float, np.ndarray, float]], length_s: float, level_dbfs: float = -18.0
) -> np.ndarray:
    """Places clips on a stereo track.

    Each clip goes to both channels with a constant-power pan. The whole track is then scaled once so that the
    frames with sound have the given root mean square level, which keeps the relative loudness of the voices.
    Samples above `KNEE` are bent toward 1 so that moments with many voices do not clip.

    Args:
        placements: For each clip its start in seconds on the track, its samples and its pan from -1 (left) to 1 (right).
        length_s: Length of the track in seconds.
        level_dbfs: Level of the frames with sound, in decibels below full scale.

    Returns:
        A float32 array of shape (samples, 2).
    """
    length = round(length_s * SAMPLE_RATE)
    track = np.zeros((length, 2), dtype=np.float32)
    for start_s, samples, pan in placements:
        offset = round(start_s * SAMPLE_RATE)
        if offset >= length or offset < 0:
            continue
        end = min(offset + len(samples), length)
        angle = (min(1.0, max(-1.0, pan)) + 1.0) * np.pi / 4
        part = samples[: end - offset]
        track[offset:end, 0] += part * np.cos(angle)
        track[offset:end, 1] += part * np.sin(angle)
    active = np.abs(track).max(axis=1) > 1e-6
    if active.any():
        rms = float(np.sqrt(np.mean(track[active] ** 2)))
        track *= 10 ** (level_dbfs / 20) / rms
        over = np.abs(track) > KNEE
        track[over] = np.sign(track[over]) * (KNEE + (1 - KNEE) * np.tanh((np.abs(track[over]) - KNEE) / (1 - KNEE)))
    return track
