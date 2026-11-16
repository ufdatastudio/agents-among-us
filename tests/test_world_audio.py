"""Tests for the voices of a recorded game and the track of its video."""

import pytest

from core.audio.mix import SAMPLE_RATE
from core.audio.speech import read_speech, synthesize_utterances
from core.world_map import build_map
from scripts.world_audio import (
    build_track,
    concurrency,
    match_clips,
    seat_pans,
    utterances_of,
    voices_of,
)
from scripts.world_replay import Pacing, build_timeline, load_replay
from tests.conftest import FakeVoice


@pytest.fixture()
def replay(mini_game):
    return load_replay(mini_game)


@pytest.fixture()
def speech(replay, tmp_path):
    folder = tmp_path / "speech"
    synthesize_utterances(utterances_of(replay), {"Agent_0": "v0", "Agent_1": "v1"}, FakeVoice(), folder)
    return folder


def test_the_statements_come_in_the_order_the_video_shows_them(replay):
    found = utterances_of(replay)
    assert [(u.round_num, u.index, u.author) for u in found] == [(1, 0, "Agent_0"), (1, 1, "Agent_1")]
    assert found[0].text == "I was in Storage the whole time."


def test_every_agent_gets_its_own_voice_and_the_same_one_on_every_run(replay):
    voices = voices_of(replay)
    assert sorted(voices) == sorted(replay.roles)
    assert len(set(voices.values())) == len(voices)
    assert voices == voices_of(replay)


def test_voices_are_panned_by_seat(replay):
    timeline = build_timeline(replay, build_map(), Pacing(meeting_speed=1.0))
    pans = seat_pans(timeline)
    assert max(pans.values()) == pytest.approx(0.7) and min(pans.values()) == pytest.approx(-0.7)
    leftmost = min(timeline.seats, key=lambda name: timeline.seats[name][0])
    assert pans[leftmost] < 0


def test_each_clip_is_matched_to_the_caption_of_its_statement(replay, speech):
    timeline = build_timeline(replay, build_map(), Pacing(meeting_speed=1.0))
    pairs = match_clips(timeline, read_speech(speech)[1])
    assert [clip.utterance.author for _, clip in pairs] == ["Agent_0", "Agent_1"]
    meeting = next(p for p in timeline.phases if p.kind == "meeting")
    assert pairs[0][0] == pytest.approx(meeting.start + 1.0)
    assert pairs[1][0] == pytest.approx(meeting.start + 12.0)


def test_the_track_has_each_voice_when_its_statement_is_shown(replay, speech):
    timeline = build_timeline(replay, build_map(), Pacing(meeting_speed=1.0))
    meeting = next(p for p in timeline.phases if p.kind == "meeting")
    length = timeline.duration + 5.0
    track = build_track(timeline, speech, offset_s=5.0, length_s=length)
    assert track.shape == (round(length * SAMPLE_RATE), 2)
    first = round((5.0 + meeting.start + 1.0) * SAMPLE_RATE)
    assert abs(track[first - 1]).max() == 0
    assert abs(track[first + SAMPLE_RATE // 2]).max() > 0
    second = round((5.0 + meeting.start + 12.0) * SAMPLE_RATE)
    assert abs(track[second + SAMPLE_RATE // 2]).max() > 0
    assert abs(track[round((5.0 + meeting.start + 10.5) * SAMPLE_RATE)]).max() == 0


def test_voices_need_meetings_played_at_the_speed_they_happened(replay, speech):
    timeline = build_timeline(replay, build_map(), Pacing(meeting_speed=4.0))
    with pytest.raises(ValueError, match="speed"):
        build_track(timeline, speech, 0.0, timeline.duration)


def test_speech_of_another_game_is_refused(replay, speech, tmp_path):
    other = tmp_path / "other"
    utterances = utterances_of(replay)
    changed = [utterances[0].__class__(1, 0, "Agent_0", 1.0, 9.0, "Something else was said.")] + utterances[1:]
    synthesize_utterances(changed, {"Agent_0": "v0", "Agent_1": "v1"}, FakeVoice(), other)
    timeline = build_timeline(replay, build_map(), Pacing(meeting_speed=1.0))
    with pytest.raises(ValueError, match="another game"):
        build_track(timeline, other, 0.0, timeline.duration)


def test_concurrency_counts_the_time_that_voices_overlap(replay, tmp_path):
    utterances = utterances_of(replay)
    overlapping = [utterances[0], utterances[1].__class__(1, 1, "Agent_1", 4.0, 12.0, utterances[1].text)]
    synthesize_utterances(overlapping, {"Agent_0": "v0", "Agent_1": "v1"}, FakeVoice(), tmp_path / "s")
    shared, mean = concurrency(read_speech(tmp_path / "s")[1])
    assert 0 < shared < 1 and 1 < mean < 2
