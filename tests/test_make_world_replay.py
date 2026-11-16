"""Tests for the pictures and the video of a game on the tile map."""

import json
import shutil
import subprocess

import pytest

pytest.importorskip("matplotlib")

from core.audio.speech import synthesize_utterances
from core.world_map import build_map
from scripts.make_world_replay import (
    FINAL_H,
    FINAL_W,
    Renderer,
    find_run,
    main,
    render_video,
)
from scripts.world_audio import utterances_of
from scripts.world_replay import build_timeline, load_replay
from scripts.world_setup import describe_setup


@pytest.fixture()
def renderer(mini_game):
    world, replay = build_map(), load_replay(mini_game)
    return Renderer(replay, build_timeline(replay, world), world)


def test_a_frame_has_the_size_of_the_video(renderer):
    assert renderer.frame(0.5).size == (FINAL_W, FINAL_H)


def test_a_meeting_frame_differs_from_a_walking_frame(renderer):
    meeting = next(p for p in renderer.timeline.phases if p.kind == "meeting")
    assert renderer.frame(meeting.start + 1.0).tobytes() != renderer.frame(0.5).tobytes()


def test_every_phase_of_the_game_can_be_drawn(renderer):
    for phase in renderer.timeline.phases:
        assert renderer.frame((phase.start + phase.end) / 2).size == (FINAL_W, FINAL_H)


def test_main_writes_a_still_picture_for_each_time(mini_game, tmp_path):
    out = tmp_path / "pictures"
    main([str(mini_game), "--out", str(out), "--still", "1", "--still", "12.5", "--name", "mini"])
    assert sorted(p.name for p in out.glob("*.png")) == ["mini-t1.png", "mini-t12.5.png"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_the_video_is_written_with_ffmpeg(mini_game, tmp_path):
    path = tmp_path / "game.mp4"
    duration = render_video(mini_game, speed=4.0, path=path, fps=2, jobs=2)
    assert duration > 10
    assert path.stat().st_size > 1000


def test_the_setup_screen_has_the_size_of_the_video_and_differs_from_the_game(renderer, mini_game):
    setup = describe_setup(renderer.replay, renderer.world)
    card = renderer.intro(setup)
    assert card.size == (FINAL_W, FINAL_H)
    assert card.tobytes() != renderer.frame(0.5).tobytes()


def test_main_writes_the_setup_screen_as_a_picture(mini_game, tmp_path):
    main([str(mini_game), "--out", str(tmp_path), "--card", "--name", "mini"])
    assert [p.name for p in tmp_path.glob("*.png")] == ["mini-setup.png"]


def test_the_row_of_a_game_is_found_by_its_game_id(mini_game, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"game_id": "other_1", "model": "a"}) + "\n" + json.dumps({"game_id": "tag_interleaved-60_3", "model": "b"}) + "\n")
    replay = load_replay(mini_game)
    assert find_run(rows, replay)["model"] == "b"
    replay.game_id = "missing_1"
    assert find_run(rows, replay) is None


def test_speech_needs_the_speed_of_the_meeting(mini_game, tmp_path):
    with pytest.raises(SystemExit):
        main([str(mini_game), "--out", str(tmp_path), "--speech", str(tmp_path), "--speed", "4"])


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg is not installed")
def test_the_video_starts_with_the_setup_screen_and_has_sound(mini_game, tmp_path, fake_voice):
    replay = load_replay(mini_game)
    speech = tmp_path / "speech"
    synthesize_utterances(utterances_of(replay), {"Agent_0": "v0", "Agent_1": "v1"}, fake_voice, speech)
    setup = describe_setup(replay, build_map(), voices={"Agent_0": "af_heart", "Agent_1": "am_adam"}, speed=1.0)
    path = tmp_path / "game.mp4"
    duration = render_video(mini_game, speed=1.0, path=path, fps=2, jobs=2, setup=setup, intro_s=4.0, speech=speech)
    bare = render_video(mini_game, speed=1.0, path=tmp_path / "bare.mp4", fps=2, jobs=2)
    assert duration == pytest.approx(bare + 4.0)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True
    )
    assert sorted(probe.stdout.split()) == ["audio", "video"]
