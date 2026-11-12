"""Tests for the replay of a recorded game on the tile map."""

import math

import pytest

from core.world_map import DOOR, PASSABLE, build_map
from scripts.world_replay import (
    MEETING_ROOM,
    Act,
    Pacing,
    PathFinder,
    build_timeline,
    load_replay,
    parse_acts,
    room_slots,
    seat_points,
)


@pytest.fixture(scope="module")
def world():
    return build_map()


@pytest.fixture()
def replay(mini_game):
    return load_replay(mini_game)


def tile_room(world, x: float, y: float) -> str | None:
    return world.room_at.get((math.floor(x), math.floor(y)))


def test_parse_acts_reads_the_acts_and_only_the_first_cause_of_a_round(mini_game):
    text = (mini_game / "Honest" / "Agent_1" / "action.log").read_text(encoding="utf-8")
    text += "Discussion Started because Agent_2: Body Reported: Agent_9. Additional victims confirmed eliminations: None\n"
    acts, causes = parse_acts(text)
    assert acts[1] == [Act("Cafeteria", "move", "Storage"), Act("Storage", "report", "Agent_3")]
    assert causes == {1: ("Agent_1", "Agent_3")}


def test_load_replay_reads_roles_rounds_meeting_and_winner(replay):
    assert replay.winner == "honest"
    assert replay.roles["Agent_0"] == "byzantine"
    entry = replay.rounds[0]
    assert entry.ticks() == 2
    assert (entry.reporter, entry.body, entry.victims) == ("Agent_1", "Agent_3", ["Agent_3"])
    assert entry.meeting.votes == {"Agent_0": "Agent_1", "Agent_1": "Agent_0", "Agent_2": "Agent_0"}
    assert entry.meeting.ejected == "Agent_0"
    assert [s.author for s in entry.meeting.spoken] == ["Agent_0", "Agent_1"]
    assert entry.meeting.closed_at == 60.0


def test_load_replay_refuses_a_folder_that_is_not_a_game(tmp_path):
    with pytest.raises(ValueError):
        load_replay(tmp_path)


def test_every_room_has_standing_spots_on_its_own_floor(world):
    for name, room in world.rooms.items():
        slots = room_slots(room)
        assert len(slots) >= 7, name
        assert all(tile_room(world, x, y) == name for x, y in slots)
        assert all(math.dist(a, b) >= 1.3 for i, a in enumerate(slots) for b in slots[i + 1:])


def test_a_walk_goes_through_a_door_and_over_walkable_tiles(world):
    finder = PathFinder(world)
    start, goal = room_slots(world.rooms["Reactor"])[0], room_slots(world.rooms[MEETING_ROOM])[0]
    path = finder.route(start, goal, ("Reactor", MEETING_ROOM))
    assert path[0] == start and path[-1] == goal
    kinds = []
    for a, b in zip(path, path[1:]):
        for k in range(41):
            x, y = a[0] + (b[0] - a[0]) * k / 40, a[1] + (b[1] - a[1]) * k / 40
            kinds.append(world.tile(math.floor(x), math.floor(y)))
    assert all(kind in PASSABLE for kind in kinds)
    assert DOOR in kinds
    assert sum(math.dist(a, b) for a, b in zip(path, path[1:])) >= math.dist(start, goal)


def test_seats_are_distinct_places_in_the_meeting_room(world):
    seats = seat_points([f"Agent_{n}" for n in range(10)], world)
    assert len(set(seats.values())) == 10
    assert all(tile_room(world, x, y) == MEETING_ROOM for x, y in seats.values())


def test_the_phases_follow_the_rules_of_the_engine(world, replay):
    timeline = build_timeline(replay, world)
    assert [p.kind for p in timeline.phases] == ["round", "kill", "tick", "report", "gather", "meeting", "vote", "eject", "end"]
    assert timeline.phases[-1].end == pytest.approx(timeline.duration)


def test_a_tag_leaves_a_body_where_the_victim_stood(world, replay):
    timeline = build_timeline(replay, world)
    strike = timeline.strikes[0]
    assert (strike.killer, strike.victim) == ("Agent_0", "Agent_3")
    frame = timeline.at(strike.end + 0.05)
    assert [name for name, _, _ in frame.bodies] == ["Agent_3"]
    assert "Agent_3" not in [t.name for t in frame.tokens]
    assert tile_room(world, frame.bodies[0][1], frame.bodies[0][2]) == "Storage"
    assert frame.status["Agent_3"] == ("killed", 1)


def test_the_moves_of_the_tick_of_a_report_are_dropped(world, replay):
    timeline = build_timeline(replay, world)
    report = next(p for p in timeline.phases if p.kind == "report")
    for t in (report.start + 0.05, report.end - 0.05):
        rooms = {tok.name: tile_room(world, tok.x, tok.y) for tok in timeline.at(t).tokens}
        assert rooms == {"Agent_0": "Storage", "Agent_1": "Storage", "Agent_2": "Storage"}


def test_the_moves_of_a_tick_without_a_report_are_carried_out(world, replay):
    timeline = build_timeline(replay, world)
    walk = next(p for p in timeline.phases if p.kind == "tick")
    rooms = {tok.name: tile_room(world, tok.x, tok.y) for tok in timeline.at(walk.start - 0.01).tokens}
    assert rooms["Agent_1"] == "Cafeteria" and rooms["Agent_2"] == "Admin"
    rooms = {tok.name: tile_room(world, tok.x, tok.y) for tok in timeline.at(walk.end + 0.01).tokens}
    assert rooms["Agent_1"] == "Storage" and rooms["Agent_2"] == "Storage"


def test_everyone_in_play_sits_at_a_seat_when_the_meeting_starts(world, replay):
    timeline = build_timeline(replay, world)
    meeting = next(p for p in timeline.phases if p.kind == "meeting")
    frame = timeline.at(meeting.start + 0.01)
    assert sorted(t.name for t in frame.tokens) == ["Agent_0", "Agent_1", "Agent_2"]
    for tok in frame.tokens:
        assert math.dist((tok.x, tok.y), timeline.seats[tok.name]) < 0.05
    assert frame.table is not None


def test_the_vote_and_the_ejection_follow_the_log(world, replay):
    timeline = build_timeline(replay, world)
    vote = next(p for p in timeline.phases if p.kind == "vote")
    frame = timeline.at(vote.end - 0.05)
    assert sorted(frame.arrows) == [("Agent_0", "Agent_1"), ("Agent_1", "Agent_0"), ("Agent_2", "Agent_0")]
    assert frame.tally == {"Agent_1": 1, "Agent_0": 2}
    final = timeline.at(timeline.duration - 0.05)
    assert [t.name for t in final.tokens] == ["Agent_1", "Agent_2"]
    assert final.status["Agent_0"] == ("ejected", 1)
    assert final.banner.title == "Honest agents win"
    assert [text for _, text in final.events] == [
        "Agent_0 tagged Agent_3 in Storage",
        "Agent_1 reported Agent_3's body in Storage",
        "Agent_0 was ejected, byzantine, with 2 of 3 votes",
    ]


def test_no_agent_jumps_between_two_frames(world, replay):
    timeline = build_timeline(replay, world)
    step = 0.05
    last = {t.name: (t.x, t.y) for t in timeline.at(0).tokens}
    t = step
    while t < timeline.duration:
        for tok in timeline.at(t).tokens:
            if tok.name in last:
                assert math.dist(last[tok.name], (tok.x, tok.y)) < 3.0, (tok.name, t)
            last[tok.name] = (tok.x, tok.y)
        t += step


def test_a_statement_appears_word_by_word_while_it_is_spoken(world, replay):
    pacing = Pacing()
    timeline = build_timeline(replay, world, pacing)
    meeting = next(p for p in timeline.phases if p.kind == "meeting")
    caption = next(c for c in timeline.captions if c.speaker == "Agent_0")
    assert caption.start == pytest.approx(meeting.start + 1.0 / pacing.meeting_speed)
    full = caption.text
    midway = [c for c in timeline.at((caption.start + caption.end) / 2).captions if c[0] == "Agent_0"][0]
    assert 0 < len(midway[1].split()) < len(full.split()) and not midway[2]
    after = [c for c in timeline.at(caption.end + 0.01).captions if c[0] == "Agent_0"][0]
    assert after[1] == full and after[2]


def test_a_faster_meeting_makes_a_shorter_video(world, replay):
    slow = build_timeline(replay, world, Pacing(meeting_speed=2.0)).duration
    fast = build_timeline(replay, world, Pacing(meeting_speed=8.0)).duration
    assert slow - fast == pytest.approx(60.0 / 2.0 - 60.0 / 8.0)
