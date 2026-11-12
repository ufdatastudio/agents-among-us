"""Scoring of speaker labels over time, with no dependency beyond numpy.

A segment says that one speaker talks from a start to an end. The reference comes from the clips the room
placed on its timeline, and a hypothesis comes from a diarization model. The diarization error rate counts the
speaking time that is missed, invented or given to the wrong speaker, divided by the speaking time of the reference.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from core.audio.mix import SAMPLE_RATE, Clip

FRAME_S = 0.01


@dataclass(frozen=True)
class Segment:
    """A speaker talking for a stretch of time.

    Attributes:
        speaker: Label of the speaker. A hypothesis uses the labels of the model, which need not match the reference.
        start_s: Time at which the speaker starts.
        end_s: Time at which the speaker stops.
    """

    speaker: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Der:
    """The parts of a diarization error rate, all in seconds of speaking time.

    Attributes:
        miss_s: Reference speech that the hypothesis leaves out.
        false_alarm_s: Hypothesis speech that the reference does not have.
        confusion_s: Speech that the hypothesis gives to the wrong speaker.
        reference_s: Total speaking time of the reference, which counts once per speaker when speakers overlap.
    """

    miss_s: float
    false_alarm_s: float
    confusion_s: float
    reference_s: float

    @property
    def error_s(self) -> float:
        """Total time in error."""
        return self.miss_s + self.false_alarm_s + self.confusion_s

    @property
    def der(self) -> float:
        """Errors divided by reference speech, which is 0 when both are empty and 1 when only the hypothesis speaks."""
        if self.reference_s == 0.0:
            return 0.0 if self.error_s == 0.0 else 1.0
        return self.error_s / self.reference_s


def energy_segments(
    samples: np.ndarray,
    sample_rate: int,
    speaker: str,
    threshold_db: float = -35.0,
    min_pause_s: float = 0.3,
) -> list[Segment]:
    """Finds where a clean clip of one speaker has speech, using frame energy.

    A synthesized sentence has pauses at commas. A diarizer that reports silence there is right, so the reference
    follows the energy of the clip and joins stretches that a pause shorter than `min_pause_s` separates.

    Args:
        samples: Mono float samples of one speaker.
        sample_rate: Sampling rate in hertz.
        speaker: Label for the segments.
        threshold_db: Frames below the loudest frame by more than this many decibels are silence.
        min_pause_s: Shortest silence that ends a segment.

    Returns:
        Segments in time order, with times relative to the start of `samples`.
    """
    frame = round(FRAME_S * sample_rate)
    count = len(samples) // frame
    if count == 0:
        return []
    rms = np.sqrt(np.mean(samples[: count * frame].reshape(count, frame) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        return []
    active = rms >= peak * 10 ** (threshold_db / 20)
    edges = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    merged: list[list[int]] = []
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if merged and (start - merged[-1][1]) * FRAME_S < min_pause_s:
            merged[-1][1] = int(end)
        else:
            merged.append([int(start), int(end)])
    return [Segment(speaker, start * FRAME_S, end * FRAME_S) for start, end in merged]


def oracle_segments(clips: Sequence[Clip], threshold_db: float = -35.0, min_pause_s: float = 0.3) -> list[Segment]:
    """Lists the true speech segments of a set of clips on the room timeline.

    Args:
        clips: The clips the room placed.
        threshold_db: Silence threshold, see `energy_segments`.
        min_pause_s: Shortest silence that ends a segment.

    Returns:
        Segments in room time, labeled with the speaker of each clip.
    """
    return [
        Segment(clip.speaker, clip.start_s + found.start_s, clip.start_s + found.end_s)
        for clip in clips
        for found in energy_segments(clip.samples, SAMPLE_RATE, clip.speaker, threshold_db, min_pause_s)
    ]


def activity(segments: Sequence[Segment], speakers: Sequence[str], frames: int) -> np.ndarray:
    """Marks the frames in which each speaker talks.

    Args:
        segments: The segments to draw.
        speakers: Row order of the result.
        frames: Number of frames of `FRAME_S` seconds.

    Returns:
        A boolean array of shape (speakers, frames).
    """
    rows = {name: index for index, name in enumerate(speakers)}
    grid = np.zeros((len(speakers), frames), dtype=bool)
    for segment in segments:
        grid[rows[segment.speaker], round(segment.start_s / FRAME_S) : round(segment.end_s / FRAME_S)] = True
    return grid


def best_assignment(overlap: np.ndarray) -> list[tuple[int, int]]:
    """Pairs rows with columns so that the sum of their overlaps is as large as possible.

    The search runs over subsets of the shorter axis, which holds at most about 10 speakers here.

    Args:
        overlap: A matrix of overlap in frames between each row and column.

    Returns:
        Pairs of a row index and a column index, each used at most once.
    """
    transposed = overlap.shape[0] < overlap.shape[1]
    values = overlap.T if transposed else overlap
    items, slots = values.shape
    best: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for item in range(items):
        step = dict(best)
        for mask, (score, pairs) in best.items():
            for slot in range(slots):
                if mask >> slot & 1:
                    continue
                grown = mask | 1 << slot
                total = score + float(values[item, slot])
                if grown not in step or total > step[grown][0]:
                    step[grown] = (total, [*pairs, (item, slot)])
        best = step
    chosen = max(best.values(), key=lambda state: state[0])[1]
    return [(slot, item) if transposed else (item, slot) for item, slot in chosen]


def diarization_error_rate(reference: Sequence[Segment], hypothesis: Sequence[Segment]) -> Der:
    """Scores a hypothesis against a reference, frame by frame with no forgiveness collar.

    Overlapped speech counts once for each speaker. The hypothesis labels are matched to the reference labels in
    the way that agrees most, so a model may call the first speaker 3 and the second speaker 0.

    Args:
        reference: The true segments.
        hypothesis: The segments that the model produced.

    Returns:
        The parts of the error and the reference speaking time.
    """
    ends = [s.end_s for s in (*reference, *hypothesis)]
    frames = round(max(ends) / FRAME_S) if ends else 0
    ref_names = sorted({s.speaker for s in reference})
    hyp_names = sorted({s.speaker for s in hypothesis})
    ref = activity(reference, ref_names, frames)
    hyp = activity(hypothesis, hyp_names, frames)
    n_ref, n_hyp = ref.sum(axis=0), hyp.sum(axis=0)
    correct = np.zeros(frames, dtype=int)
    if ref_names and hyp_names:
        overlap = ref.astype(int) @ hyp.astype(int).T
        for r, h in best_assignment(overlap):
            correct += ref[r] & hyp[h]
    return Der(
        miss_s=float(np.maximum(n_ref - n_hyp, 0).sum()) * FRAME_S,
        false_alarm_s=float(np.maximum(n_hyp - n_ref, 0).sum()) * FRAME_S,
        confusion_s=float((np.minimum(n_ref, n_hyp) - correct).sum()) * FRAME_S,
        reference_s=float(n_ref.sum()) * FRAME_S,
    )
