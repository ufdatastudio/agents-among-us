"""Tests for the speech clips of a recorded game."""

import numpy as np
import pytest

from core.audio.mix import SAMPLE_RATE
from core.audio.speech import (
    BASE_SPEED,
    KNEE,
    MAX_SPEED,
    MIN_SLOT_S,
    OVERRUN,
    OVERRUN_S,
    Utterance,
    fit_clip,
    place_clips,
    read_speech,
    read_voices,
    spoken_form,
    synthesize_utterances,
)
from tests.conftest import FakeVoice


def test_spoken_form_says_names_and_rooms_as_a_person_would():
    text = "Agent_3 was in UpperEngine, then O2. Agent_12 lied \U0001F600 <to>x</to>"
    assert spoken_form(text) == "Agent 3 was in Upper Engine, then O 2. Agent 12 lied to x /to"


def test_a_clip_that_fits_is_spoken_at_the_base_speed():
    samples, speed, _ = fit_clip(FakeVoice(0.5), "one two three four", "v0", slot_s=10.0)
    assert speed == BASE_SPEED
    assert len(samples) / SAMPLE_RATE == pytest.approx(4 * 0.5 / BASE_SPEED, abs=0.01)


def test_a_clip_that_is_too_long_is_spoken_faster_to_fit_its_slot():
    samples, speed, _ = fit_clip(FakeVoice(0.5), "a b c d e f g h", "v0", slot_s=4.0)
    assert BASE_SPEED < speed <= MAX_SPEED
    assert len(samples) / SAMPLE_RATE <= 4.0 + 1e-6


def test_a_statement_of_one_word_gets_at_least_the_shortest_slot():
    samples, speed, _ = fit_clip(FakeVoice(1.0), "misspoke", "v0", slot_s=0.4)
    assert speed == BASE_SPEED
    assert len(samples) / SAMPLE_RATE == pytest.approx(1.0 / BASE_SPEED, abs=0.01)
    assert len(samples) / SAMPLE_RATE <= MIN_SLOT_S


def test_a_clip_that_cannot_fit_is_cut_at_the_end_of_its_slot_with_a_fade():
    samples, speed, natural = fit_clip(FakeVoice(2.0), "a b c d e f g h", "v0", slot_s=8.0)
    assert speed == MAX_SPEED
    assert len(samples) == round(8.0 * SAMPLE_RATE)
    assert natural == pytest.approx(8 * 2.0 / MAX_SPEED, abs=0.01) and natural > 8.0
    assert samples[-1] == 0.0 and samples[0] == pytest.approx(0.5)


def test_a_clip_may_run_on_to_its_limit_before_it_is_cut():
    samples, _, natural = fit_clip(FakeVoice(2.0), "a b c d e f g h", "v0", slot_s=8.0, limit_s=9.0)
    assert len(samples) == round(9.0 * SAMPLE_RATE)
    samples, _, _ = fit_clip(FakeVoice(2.0), "a b c d e f g h", "v0", slot_s=8.0, limit_s=12.0)
    assert len(samples) == round(natural * SAMPLE_RATE)


def test_a_clip_never_runs_into_the_next_statement_of_its_author(tmp_path):
    first = Utterance(1, 0, "Agent_0", 0.0, 8.0, "a b c d e f g h")
    same = Utterance(1, 1, "Agent_0", 8.5, 12.0, "x y")
    other = Utterance(1, 2, "Agent_1", 0.0, 8.0, "a b c d e f g h")
    clips = synthesize_utterances([first, same, other], {"Agent_0": "v0", "Agent_1": "v1"}, FakeVoice(2.5), tmp_path)
    assert clips[0].duration_s == pytest.approx(8.5)
    assert clips[2].duration_s == pytest.approx(8.0 * OVERRUN + OVERRUN_S, abs=0.01)


def test_clips_are_written_with_an_index_that_reads_back(tmp_path):
    utterances = [Utterance(1, 0, "Agent_0", 1.0, 9.0, "Hello Agent_1."), Utterance(1, 1, "Agent_1", 4.0, 12.0, "I saw nothing.")]
    voices = {"Agent_0": "v0", "Agent_1": "v1"}
    written = synthesize_utterances(utterances, voices, FakeVoice(), tmp_path / "speech")
    assert [c.file for c in written] == ["r1_000_Agent_0.wav", "r1_001_Agent_1.wav"]
    assert read_voices(tmp_path / "speech") == voices
    read_voices_, clips = read_speech(tmp_path / "speech")
    assert read_voices_ == voices
    assert [c.utterance for c in clips] == utterances
    assert clips[0].samples.shape == written[0].samples.shape
    assert np.allclose(clips[0].samples, written[0].samples, atol=1e-3)


def test_an_author_without_a_voice_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        synthesize_utterances([Utterance(1, 0, "Agent_9", 0.0, 5.0, "Hi.")], {}, FakeVoice(), tmp_path)


def test_a_folder_without_an_index_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_speech(tmp_path)


def test_clips_are_placed_at_their_start_and_panned():
    one = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
    track = place_clips([(1.0, one, -1.0), (3.0, one, 1.0), (5.0, one, 0.0)], length_s=7.0, level_dbfs=-40.0)
    assert track.shape == (7 * SAMPLE_RATE, 2)
    assert track[: SAMPLE_RATE].max() == 0.0
    left, right, middle = track[round(1.5 * SAMPLE_RATE)], track[round(3.5 * SAMPLE_RATE)], track[round(5.5 * SAMPLE_RATE)]
    assert left[0] > 0 and left[1] < 1e-6
    assert right[1] > 0 and right[0] < 1e-6
    assert middle[0] == pytest.approx(middle[1]) and middle[0] < left[0]
    assert middle[0] ** 2 + middle[1] ** 2 == pytest.approx(left[0] ** 2, rel=1e-3)


def test_the_track_has_the_level_that_was_asked_for():
    one = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
    track = place_clips([(0.0, one, 0.0), (2.0, one * 0.5, 0.0)], length_s=4.0, level_dbfs=-30.0)
    active = np.abs(track).max(axis=1) > 0
    assert 20 * np.log10(np.sqrt(np.mean(track[active] ** 2))) == pytest.approx(-30.0, abs=0.01)


def test_many_voices_at_once_are_limited_below_full_scale():
    loud = np.full(SAMPLE_RATE, 0.9, dtype=np.float32)
    quiet = np.full(SAMPLE_RATE, 0.05, dtype=np.float32)
    track = place_clips([(0.0, loud, 0.0)] * 6 + [(2.0, quiet, 0.0)], length_s=4.0, level_dbfs=-6.0)
    assert KNEE < np.abs(track[round(0.5 * SAMPLE_RATE)]).max() < 1.0
    assert np.abs(track).max() < 1.0


def test_a_clip_that_runs_past_the_end_of_the_track_is_cut():
    track = place_clips([(1.5, np.ones(SAMPLE_RATE, dtype=np.float32), 0.0)], length_s=2.0)
    assert track.shape[0] == 2 * SAMPLE_RATE and track[-1].max() > 0
