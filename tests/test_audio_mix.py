"""Tests for per-reader audio mixing in the situation room."""

import numpy as np

from core.audio.mix import SAMPLE_RATE, Clip, mix_for_reader, overlap_fraction, read_wav, speakers_at, write_wav


def tone(seconds: float, level: float = 0.2) -> np.ndarray:
    return np.full(round(seconds * SAMPLE_RATE), level, dtype=np.float32)


def test_clip_reports_its_duration_and_end():
    clip = Clip("Red", 2.0, tone(1.5))
    assert clip.duration_s == 1.5
    assert clip.end_s == 3.5


def test_mix_places_each_speaker_at_its_offset():
    clips = [Clip("Red", 0.0, tone(1.0, 0.1)), Clip("Blue", 0.5, tone(1.0, 0.2))]
    mix = mix_for_reader(clips, reader="Green", until_s=2.0)
    assert mix[round(0.25 * SAMPLE_RATE)] == np.float32(0.1)
    assert abs(mix[round(0.75 * SAMPLE_RATE)] - 0.3) < 1e-6
    assert mix[round(1.25 * SAMPLE_RATE)] == np.float32(0.2)
    assert mix[round(1.75 * SAMPLE_RATE)] == 0.0


def test_mix_leaves_out_the_readers_own_voice():
    clips = [Clip("Red", 0.0, tone(1.0, 0.1)), Clip("Blue", 0.0, tone(1.0, 0.2))]
    mix = mix_for_reader(clips, reader="Red", until_s=1.0)
    assert np.allclose(mix, 0.2)


def test_mix_cuts_a_clip_still_playing_and_skips_later_ones():
    clips = [Clip("Red", 0.0, tone(4.0)), Clip("Blue", 3.0, tone(1.0))]
    mix = mix_for_reader(clips, reader="Green", until_s=2.0)
    assert len(mix) == 2 * SAMPLE_RATE
    assert np.allclose(mix, 0.2)


def test_mix_scales_down_when_the_sum_would_clip():
    clips = [Clip(name, 0.0, tone(1.0, 0.8)) for name in ("Red", "Blue", "Lime")]
    mix = mix_for_reader(clips, reader="Green", until_s=1.0)
    assert np.max(np.abs(mix)) <= 1.0


def test_gain_changes_a_speakers_level():
    clips = [Clip("Red", 0.0, tone(1.0, 0.1))]
    mix = mix_for_reader(clips, reader="Green", until_s=1.0, gains_db={"Red": 6.0206})
    assert abs(mix[0] - 0.2) < 1e-3


def test_overlap_fraction():
    assert overlap_fraction([Clip("Red", 0.0, tone(2.0))]) == 0.0
    assert overlap_fraction([Clip("Red", 0.0, tone(2.0)), Clip("Blue", 3.0, tone(1.0))]) == 0.0
    both = overlap_fraction([Clip("Red", 0.0, tone(2.0)), Clip("Blue", 1.0, tone(2.0))])
    assert abs(both - 1.0 / 3.0) < 1e-9
    assert overlap_fraction([]) == 0.0


def test_speakers_at_lists_who_is_talking():
    clips = [Clip("Red", 0.0, tone(2.0)), Clip("Blue", 1.0, tone(2.0))]
    assert speakers_at(clips, 0.5) == ["Red"]
    assert speakers_at(clips, 1.5) == ["Red", "Blue"]
    assert speakers_at(clips, 1.5, exclude="Red") == ["Blue"]
    assert speakers_at(clips, 5.0) == []


def test_wav_round_trip(tmp_path):
    samples = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    write_wav(tmp_path / "x.wav", samples)
    loaded, rate = read_wav(tmp_path / "x.wav")
    assert rate == SAMPLE_RATE
    assert np.allclose(loaded, samples, atol=1e-4)
