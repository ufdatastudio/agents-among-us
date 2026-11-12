"""Tests for the scoring of speaker labels over time."""

import numpy as np
import pytest

from core.audio.der import Segment, best_assignment, diarization_error_rate, energy_segments, oracle_segments
from core.audio.mix import SAMPLE_RATE, Clip


def tone(seconds: float, level: float = 0.5) -> np.ndarray:
    """Makes a steady tone that stands in for speech."""
    t = np.arange(round(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_perfect_labels_score_zero_even_when_the_names_differ():
    reference = [Segment("Red", 0.0, 2.0), Segment("Blue", 2.0, 4.0)]
    hypothesis = [Segment("3", 0.0, 2.0), Segment("0", 2.0, 4.0)]
    result = diarization_error_rate(reference, hypothesis)
    assert result.der == 0.0
    assert result.reference_s == pytest.approx(4.0)


def test_missed_speech_is_counted_against_the_reference_time():
    result = diarization_error_rate([Segment("Red", 0.0, 4.0)], [Segment("0", 0.0, 3.0)])
    assert result.miss_s == pytest.approx(1.0)
    assert result.false_alarm_s == 0.0
    assert result.der == pytest.approx(0.25)


def test_invented_speech_is_a_false_alarm():
    result = diarization_error_rate([Segment("Red", 0.0, 2.0)], [Segment("0", 0.0, 3.0)])
    assert result.false_alarm_s == pytest.approx(1.0)
    assert result.der == pytest.approx(0.5)


def test_merging_two_speakers_into_one_label_is_confusion():
    reference = [Segment("Red", 0.0, 3.0), Segment("Blue", 3.0, 4.0)]
    hypothesis = [Segment("0", 0.0, 4.0)]
    result = diarization_error_rate(reference, hypothesis)
    assert result.confusion_s == pytest.approx(1.0)
    assert result.der == pytest.approx(0.25)


def test_overlapped_speech_counts_once_for_each_speaker():
    reference = [Segment("Red", 0.0, 2.0), Segment("Blue", 1.0, 3.0)]
    hypothesis = [Segment("0", 0.0, 2.0)]
    result = diarization_error_rate(reference, hypothesis)
    assert result.reference_s == pytest.approx(4.0)
    assert result.miss_s == pytest.approx(2.0)
    assert result.confusion_s == 0.0


def test_two_hypothesis_labels_for_one_speaker_leave_confusion():
    reference = [Segment("Red", 0.0, 4.0)]
    hypothesis = [Segment("0", 0.0, 3.0), Segment("1", 3.0, 4.0)]
    result = diarization_error_rate(reference, hypothesis)
    assert result.confusion_s == pytest.approx(1.0)


def test_empty_reference_and_hypothesis_score_zero():
    assert diarization_error_rate([], []).der == 0.0
    assert diarization_error_rate([], [Segment("0", 0.0, 1.0)]).der == 1.0


def test_best_assignment_beats_taking_the_largest_overlap_first():
    overlap = np.array([[10, 9], [9, 0]])
    pairs = sorted(best_assignment(overlap))
    assert pairs == [(0, 1), (1, 0)]


def test_best_assignment_handles_more_columns_than_rows():
    overlap = np.array([[0, 5, 1]])
    assert best_assignment(overlap) == [(0, 1)]


def test_energy_segments_find_speech_and_join_a_short_pause():
    samples = np.concatenate([tone(1.0), np.zeros(round(0.2 * SAMPLE_RATE), dtype=np.float32), tone(1.0)])
    found = energy_segments(samples, SAMPLE_RATE, "Red", min_pause_s=0.3)
    assert len(found) == 1
    assert found[0].start_s == pytest.approx(0.0, abs=0.02)
    assert found[0].end_s == pytest.approx(2.2, abs=0.02)


def test_energy_segments_split_at_a_long_pause():
    samples = np.concatenate([tone(1.0), np.zeros(round(0.6 * SAMPLE_RATE), dtype=np.float32), tone(1.0)])
    found = energy_segments(samples, SAMPLE_RATE, "Red", min_pause_s=0.3)
    assert [round(s.start_s, 1) for s in found] == [0.0, 1.6]


def test_energy_segments_of_silence_are_empty():
    assert energy_segments(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE, "Red") == []


def test_oracle_segments_sit_at_the_clip_onset_and_score_zero_against_themselves():
    clips = [Clip("Red", 1.0, tone(2.0)), Clip("Blue", 2.0, tone(2.0))]
    reference = oracle_segments(clips)
    assert [(s.speaker, round(s.start_s, 1), round(s.end_s, 1)) for s in reference] == [("Red", 1.0, 3.0), ("Blue", 2.0, 4.0)]
    assert diarization_error_rate(reference, reference).der == 0.0
