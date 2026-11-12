"""Tests for the pictures and the video of a game on the tile map."""

import shutil

import pytest

pytest.importorskip("matplotlib")

from core.world_map import build_map
from scripts.make_world_replay import FINAL_H, FINAL_W, Renderer, main, render_video
from scripts.world_replay import build_timeline, load_replay


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
