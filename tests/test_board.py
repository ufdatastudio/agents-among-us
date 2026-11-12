"""Tests for the board, a reader's view of the claims of a meeting."""

import pytest

from core.adapters.scripted import ScriptedAdapter
from core.board import MAX_LINES, MAX_SUBJECTS, record_observations_from_log, record_statement, render_board, retract
from core.ledger import CLAIM, FACT, Ledger
from core.situation_room import Meeting, Room, RoomConfig
from core.timed_discussion import run_timed_discussion

ROSTER = ["Agent_2", "Agent_3", "Agent_5"]
LOG = """=== Action Log for Agent_2 ===

Round 1/10
Players Remaining: 10
Current Location: [Shields]
  -> Occupants: []
Adjacent Location(s):
    [Navigation] -> Occupants: ['Agent_3', 'Agent_4']
    [Storage] -> Occupants: None
Bodies Seen: None

Action 1 Selected: move -> Navigation
======================


Current Location: [Navigation]
  -> Occupants: ['Agent_3']
Adjacent Location(s):
    [Shields] -> Occupants: None
Bodies Seen: None
"""


def reader_ledger() -> Ledger:
    ledger = Ledger()
    record_observations_from_log(ledger, "Agent_2", LOG)
    return ledger


def test_an_observation_log_becomes_one_private_fact_for_each_tick():
    ledger = reader_ledger()
    facts = ledger.view("Agent_2", types=[FACT])
    assert [(f.round_num, f.tick, f.content["room"]) for f in facts] == [(1, 1, "Shields"), (1, 2, "Navigation")]
    assert facts[0].content["adjacent"]["Navigation"] == ["Agent_3", "Agent_4"]
    assert facts[1].content["beside"] == ["Agent_3"]
    assert ledger.view("Agent_5") == []


def test_a_statement_gives_claims_about_places_and_accusations():
    ledger = Ledger()
    records = record_statement(ledger, "Agent_5", "I saw Agent_3 in Cafeteria. Agent_3 is suspicious.", 8.0, ROSTER, 1)
    kinds = sorted((r.content["kind"], r.subject, r.content["room"]) for r in records)
    assert kinds == [("accuse", "Agent_3", None), ("seen", "Agent_3", "Cafeteria")]
    assert all(r.type == CLAIM and r.time_s == 8.0 and r.visible_to == tuple(ROSTER) for r in records)


def test_the_board_puts_what_the_reader_saw_next_to_each_claim():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_5", "I saw Agent_3 in Cafeteria.", 8.0, ROSTER, 1)
    record_statement(ledger, "Agent_3", "I was in Navigation.", 9.0, ROSTER, 1)
    board = render_board(ledger, "Agent_2", t=20.0, round_num=1)
    assert "Agent_5 said it saw Agent_3 in Cafeteria. You saw Agent_3 only in Navigation, never in Cafeteria." in board
    assert "Agent_3 said it was in Navigation. You saw Agent_3 next door (round 1, tick 1)." in board


def test_a_claim_about_an_agent_the_reader_did_not_see_says_so():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_3", "I saw Agent_5 in Storage.", 5.0, ROSTER, 1)
    assert "You did not see Agent_5." in render_board(ledger, "Agent_2", t=10.0, round_num=1)


def test_a_claim_about_the_reader_is_checked_against_where_the_reader_was():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_5", "I saw Agent_2 in Storage.", 5.0, ROSTER, 1)
    board = render_board(ledger, "Agent_2", t=10.0, round_num=1)
    assert "You\n" in board
    assert "You were in Shields and Navigation, never in Storage." in board


def test_accusations_and_vouching_are_counted_for_each_subject():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_5", "Agent_3 is suspicious.", 5.0, ROSTER, 1)
    record_statement(ledger, "Agent_2", "Agent_3 is suspicious.", 6.0, ROSTER, 1)
    board = render_board(ledger, "Agent_3", t=10.0, round_num=1)
    assert "Accused by Agent_5 and Agent_2." in board


def test_the_reader_does_not_see_its_own_claims_and_statements_still_being_spoken():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_2", "I was in Shields.", 5.0, ROSTER, 1)
    assert "" == render_board(ledger, "Agent_2", t=10.0, round_num=1)
    record_statement(ledger, "Agent_5", "I saw Agent_3 in Storage.", 30.0, ROSTER, 1)
    assert "" == render_board(ledger, "Agent_2", t=29.0, round_num=1)
    assert "Agent_5 said" in render_board(ledger, "Agent_2", t=30.0, round_num=1)


def test_a_retracted_claim_leaves_the_board_and_stays_in_the_ledger():
    ledger = reader_ledger()
    records = record_statement(ledger, "Agent_5", "I saw Agent_3 in Storage.", 30.0, ROSTER, 1)
    retract(ledger, records, ROSTER)
    assert "" == render_board(ledger, "Agent_2", t=40.0, round_num=1)
    assert any(r.id == records[0].id for r in ledger.records)


def test_claims_of_another_meeting_are_not_on_the_board():
    ledger = reader_ledger()
    record_statement(ledger, "Agent_5", "I saw Agent_3 in Storage.", 5.0, ROSTER, 1)
    assert "" == render_board(ledger, "Agent_2", t=40.0, round_num=2)


def test_the_board_stays_bounded():
    ledger = reader_ledger()
    names = [f"Agent_{n}" for n in range(20, 20 + MAX_SUBJECTS + 4)]
    for name in names:
        record_statement(ledger, "Agent_5", f"{name} is suspicious.", 5.0, ROSTER, 1)
    for room in ("Storage", "Cafeteria", "Admin", "Weapons", "Electrical", "Reactor", "Security", "MedBay"):
        record_statement(ledger, "Agent_5", f"I saw Agent_3 in {room}.", 6.0, ROSTER, 1)
    board = render_board(ledger, "Agent_2", t=10.0, round_num=1)
    subjects = [line for line in board.splitlines()[1:] if not line.startswith(" ")]
    assert len(subjects) <= MAX_SUBJECTS
    assert f"  and {8 - MAX_LINES} more." in board


def test_the_room_writes_claims_and_ends_each_percept_with_the_board():
    ledger = reader_ledger()
    config = RoomConfig(budget_s=30.0, tick_s=10.0, quiet_close_ticks=0, board=True)
    adapters = [ScriptedAdapter(n, ROSTER, {1: "I saw Agent_3 in Cafeteria."} if n == "Agent_5" else {}) for n in ROSTER]
    room = run_timed_discussion(adapters, Meeting(round_num=1), config, ledger=ledger)
    heard = adapters[0].heard
    assert heard[0].text == ""
    assert "Agent_5 said it saw Agent_3 in Cafeteria." in heard[-1].text
    assert any(r.type == CLAIM and r.author == "Agent_5" for r in ledger.records)
    assert room.ledger is ledger


def test_without_the_board_a_percept_is_unchanged():
    adapters = [ScriptedAdapter(n, ROSTER, {1: "I was in Storage."} if n == "Agent_5" else {}) for n in ROSTER]
    run_timed_discussion(adapters, Meeting(round_num=1), RoomConfig(budget_s=30.0, quiet_close_ticks=0), ledger=Ledger())
    assert all("Board of this meeting" not in p.text for a in adapters for p in a.heard)


def test_a_cut_statement_keeps_only_the_claims_that_were_spoken():
    ledger = reader_ledger()
    room = Room(ROSTER, RoomConfig(budget_s=30.0, words_per_s=1.0, max_words=40, board=True), ledger, round_num=1)
    room.append("Agent_5", "I saw Agent_3 in Storage. I saw Agent_3 in Cafeteria.", start_s=0.0, tick=1)
    full = render_board(ledger, "Agent_2", t=30.0, round_num=1)
    assert "in Cafeteria" in full
    room.close(5.0)
    cut = render_board(ledger, "Agent_2", t=30.0, round_num=1)
    assert "in Storage" in cut
    assert "in Cafeteria" not in cut


def test_the_board_needs_a_ledger():
    with pytest.raises(ValueError):
        Room(ROSTER, RoomConfig(board=True))
