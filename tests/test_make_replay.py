"""Tests for the replay of a meeting."""

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from scripts.make_replay import (
    HEIGHT,
    WIDTH,
    Renderer,
    State,
    Timeline,
    Utterance,
    last_words,
    load_fonts,
    load_meeting,
    main,
    parse_still,
    read_results,
    render_video,
    seat_positions,
    wrap,
)

RESULTS = """=== Round Results Log ===

=== Round 1 ===
ELIMINATION: Agent_3.
Round 1 Votes Received: {'Agent_1': 1, 'Agent_0': 2}
EJECTION: Agent_0 was ejected.
Player Statuses: {'Agent_0': 'ejected', 'Agent_1': 'active', 'Agent_2': 'active', 'Agent_3': 'eliminated'}
"""


def write_meeting_folder(root: Path) -> Path:
    """Writes a game of 4 agents with one meeting, a victim, a Byzantine agent that is ejected and 3 statements."""
    game = root / "Game_tag_interleaved-60_3_Run0"
    votes = {"Agent_0": "Agent_1", "Agent_1": "Agent_0", "Agent_2": "Agent_0"}
    for role, names in (("Byz", ["Agent_0"]), ("Honest", ["Agent_1", "Agent_2", "Agent_3"])):
        for name in names:
            folder = game / role / name
            folder.mkdir(parents=True)
            if name in votes:
                (folder / "vote.log").write_text(f"=== Voting History ===\nRound 1: Voted for {votes[name]}\n", encoding="utf-8")
    (game / "discussion_chat.csv").write_text(
        "discussion_num,reason,agent_num,model,role,message\n"
        '1,"** MEETING CALLED by Agent_2. Body reported: Agent_3 located in Storage. Additional victims confirmed eliminated: None **",1,vllm:org/model,Honest,Agent_0 was alone.\n',
        encoding="utf-8",
    )
    (game / "roundResults.log").write_text(RESULTS, encoding="utf-8")
    (game / "room").mkdir()
    rows = [
        {"type": "room", "closed_at": 60.0, "tick_s": 10.0},
        {"type": "utterance", "author": "Agent_1", "start_s": 12.0, "end_s": 22.0, "text": "Agent_0 was alone near the body and never said why."},
        {"type": "utterance", "author": "Agent_0", "start_s": 1.0, "end_s": 9.0, "text": "I was in Storage the whole time."},
        {"type": "utterance", "author": "Agent_2", "start_s": 12.5, "end_s": 20.0, "text": "I vote Agent_0."},
    ]
    (game / "room" / "room_round1.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return game


def test_the_words_of_a_statement_appear_evenly_from_its_start_to_its_end():
    u = Utterance("Agent_1", 10.0, 20.0, "one two three four five six seven eight nine ten")
    assert u.visible(9.0) == []
    assert len(u.visible(15.0)) == 5
    assert u.visible(25.0)[-1] == "ten" and len(u.visible(25.0)) == 10
    assert u.speaking(15.0) and not u.speaking(20.0)


def test_the_timeline_maps_video_time_to_the_meeting_clock_and_the_votes():
    line = Timeline(closed_at=60.0, n_votes=4, speed=3.0, intro_s=2.0, hold_s=1.0, per_vote_s=0.5, result_s=3.0)
    assert line.duration == 2.0 + 20.0 + 1.0 + 2.0 + 3.0
    assert line.at(1.0) == State(0.0)
    assert line.at(12.0) == State(30.0)
    assert line.at(22.5) == State(60.0)
    assert line.at(23.1).votes_shown == 1 and line.at(24.2).votes_shown == 3
    assert line.at(25.5) == State(60.0, votes_shown=4, result=True)


def test_seats_start_at_the_top_and_go_clockwise():
    seats = seat_positions(["a", "b", "c", "d"], center=(100, 100), radii=(50, 40))
    assert seats["a"] == pytest.approx((100, 60))
    assert seats["b"][0] > 100 and seats["c"][1] > 100 and seats["d"][0] < 100


def test_wrapping_keeps_lines_inside_the_width_and_the_end_of_a_long_statement():
    font = load_fonts()["small"]
    lines = wrap("word " * 40, font, 120 * 2)
    assert len(lines) > 3 and all(font.getlength(line) <= 120 * 2 for line in lines)
    tail = last_words(["alpha"] * 30 + ["omega"], font, 150 * 2, 2)
    assert len(tail) == 2 and tail[0].startswith("... ") and tail[-1].endswith("omega")
    assert all(font.getlength(line) <= 150 * 2 for line in tail)


def test_a_meeting_is_read_from_the_game_folder(tmp_path):
    meeting = load_meeting(write_meeting_folder(tmp_path))
    assert meeting.roster == ["Agent_0", "Agent_1", "Agent_2", "Agent_3"]
    assert meeting.roles["Agent_0"] == "byzantine" and meeting.victims == ["Agent_3"]
    assert meeting.speakers == ["Agent_0", "Agent_1", "Agent_2"]
    assert meeting.votes == {"Agent_0": "Agent_1", "Agent_1": "Agent_0", "Agent_2": "Agent_0"}
    assert meeting.tally == {"Agent_1": 1, "Agent_0": 2} and meeting.ejected == "Agent_0"
    assert [u.author for u in meeting.utterances] == ["Agent_0", "Agent_1", "Agent_2"]
    assert meeting.reason == "Meeting called by Agent_2. Body reported: Agent_3 located in Storage."
    assert meeting.setting == "timed room, interleaved, 60 s" and meeting.seed == 3 and meeting.closed_at == 60.0


def test_a_meeting_that_does_not_exist_is_refused(tmp_path):
    game = write_meeting_folder(tmp_path)
    with pytest.raises(ValueError):
        load_meeting(game, round_num=7)
    with pytest.raises(ValueError):
        load_meeting(tmp_path)


def test_results_of_a_missing_log_are_empty(tmp_path):
    assert read_results(tmp_path / "none.log", 1)["tally"] == {}


def test_still_requests_name_a_meeting_time_or_a_stage(tmp_path):
    meeting = load_meeting(write_meeting_folder(tmp_path))
    assert parse_still("85", meeting) == ("t085", State(85.0))
    assert parse_still("votes", meeting)[1].votes_shown == 3
    assert parse_still("result", meeting)[1].result is True


def test_a_speaking_agent_is_drawn_in_the_accent_color_and_the_frame_has_the_output_size(tmp_path):
    meeting = load_meeting(write_meeting_folder(tmp_path))
    renderer = Renderer(meeting)
    quiet, talking = renderer.frame(State(0.0)), renderer.frame(State(15.0))
    assert quiet.size == talking.size == (WIDTH, HEIGHT)
    x, y = renderer.seats["Agent_1"]
    ring = (int(x), int(y) - 35)
    assert talking.getpixel(ring) != quiet.getpixel(ring)


def test_the_ejected_seat_is_dimmed_when_the_result_is_shown(tmp_path):
    renderer = Renderer(load_meeting(write_meeting_folder(tmp_path)))
    x, y = renderer.seats["Agent_0"]
    inside = (int(x) + 16, int(y) + 16)
    before = renderer.frame(State(60.0, votes_shown=3))
    after = renderer.frame(State(60.0, votes_shown=3, result=True))
    assert before.getpixel(inside) != after.getpixel(inside)


def test_the_command_line_writes_the_requested_stills(tmp_path):
    game = write_meeting_folder(tmp_path / "logs")
    main([str(game), "--out", str(tmp_path / "out"), "--no-video", "--stills", "15,result"])
    names = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert names == ["situation-room-tag_interleaved-60_3-round1-result.png", "situation-room-tag_interleaved-60_3-round1-t015.png"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_short_video_is_written(tmp_path):
    meeting = load_meeting(write_meeting_folder(tmp_path / "logs"))
    line = Timeline(meeting.closed_at, len(meeting.votes), speed=60.0, intro_s=0.2, hold_s=0.2, per_vote_s=0.2, result_s=0.4)
    render_video(Renderer(meeting), line, tmp_path / "replay.mp4", fps=5)
    assert (tmp_path / "replay.mp4").stat().st_size > 1000
