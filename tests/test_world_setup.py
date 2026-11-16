"""Tests for the setup screen of a recorded game."""

import pytest

from core.world_map import build_map, display_name
from scripts.world_replay import load_replay
from scripts.world_setup import describe_setup, locations_text

ROOM = {
    "type": "room", "budget_s": 180.0, "tick_s": 10.0, "words_per_s": 2.5, "max_words": 40, "onset_jitter_s": 2.0,
    "quiet_close_ticks": 3, "overlap": "interleaved", "window_s": 2.0, "dropout": 0.0, "barge_in": False, "closed_at": 60.0,
}


@pytest.fixture()
def replay(mini_game):
    found = load_replay(mini_game)
    found.room = dict(ROOM)
    return found


def rows(setup, heading):
    return dict(next(s for s in setup.sections if s.heading == heading).rows)


def test_the_setup_names_the_model_the_roster_and_the_seed(replay):
    run = {"model": "org/model-x", "reasoning_effort": "low", "max_tokens": 768, "max_rounds": 10}
    setup = describe_setup(replay, build_map(), run=run, served="vLLM on a test GPU")
    models, game = rows(setup, "Models"), rows(setup, "Game")
    assert models["Agents"] == "org/model-x plays all 4 agents"
    assert models["Served by"] == "vLLM on a test GPU"
    assert models["Reasoning"] == "low effort, at most 768 tokens per reply"
    assert game["Agents"].startswith("4: 3 honest and 1 Byzantine")
    assert "seed 3" in game["Scenario"] and "10 rounds" in game["Scenario"]


def test_the_options_come_from_the_record_of_the_room(replay):
    options = rows(describe_setup(replay, build_map(), speed=4.0), "Options")
    assert options["Protocol"] == "interleaved-60"
    assert options["Meeting"].startswith("180 s of meeting time, in ticks of 10 s")
    assert "at most 40 words" in options["Statements"] and "2.5 words per second" in options["Statements"]
    assert "windows of 2 s" in options["The transcript"]
    assert options["Also"] == "no words dropped, no interruptions, the room closes after 3 quiet ticks"
    assert options["This video"] == "meetings play 4 times faster than they happened, and every walk takes 1.6 s"
    assert rows(describe_setup(replay, build_map(), speed=1.0), "Options")["This video"].startswith("meetings play at the speed they happened")


def test_a_line_with_no_source_is_left_out(replay):
    replay.room = {}
    setup = describe_setup(replay, build_map())
    assert list(rows(setup, "Options")) == ["Protocol", "This video"]
    assert "Served by" not in rows(setup, "Models") and "Reasoning" not in rows(setup, "Models")


def test_the_board_protocol_says_what_the_board_is(replay):
    replay.protocol = "board-180"
    assert "Board" in rows(describe_setup(replay, build_map()), "Options")


def test_voices_are_listed_and_the_notes_say_the_agents_did_not_hear_them(replay):
    voices = {"Agent_0": "af_heart", "Agent_1": "am_adam", "Agent_2": "af_bella", "Agent_3": "am_puck"}
    setup = describe_setup(replay, build_map(), voices=voices)
    text = rows(setup, "Models")["Voices"]
    assert "Kokoro" in text and "0 Heart, 1 Adam, 2 Bella, 3 Puck" in text
    assert any("never heard them" in note for note in setup.notes)
    assert any("no sound" in note for note in describe_setup(replay, build_map()).notes)


def test_the_setup_lists_every_room_of_the_map(replay):
    world = build_map()
    assert locations_text(world).startswith(f"{len(world.rooms)} rooms")
    assert "Cafeteria" in locations_text(world)
    assert describe_setup(replay, world).rooms == [display_name(name) for name in world.rooms]
