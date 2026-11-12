"""Tests for the analysis of private reasoning in P2 games."""

import json

import pytest

from scripts.analyze_p2_thoughts import (
    flagged_as_liar,
    intent_profile,
    read_thoughts,
    report,
    vote_profile,
)
from test_analyze_p2_text import write_game_folder


def write_thoughts(game, records):
    (game / "thought.log").write_text("\n".join(json.dumps(r) for r in records) + "\n{broken line\n", encoding="utf-8")


VOTE_THINK = "Agent_0 is lying about Storage. Multiple agents confirmed Agent_1 was there. Agent_0 should be ejected."
ROOM_THINK = "I need to deflect suspicion onto Agent_1. My teammate Agent_2 will back me."


@pytest.fixture
def game(tmp_path):
    folder = write_game_folder(tmp_path)
    write_thoughts(
        folder,
        [
            {"round": 1, "phase": "VOTING", "agent": "Agent_1", "role": "honest", "think": "Maybe Agent_3.", "output": "Agent_3"},
            {"round": 1, "phase": "VOTING", "agent": "Agent_1", "role": "honest", "think": VOTE_THINK, "output": "Agent_0"},
            {"round": 1, "phase": "ROOM", "agent": "Agent_0", "role": "byzantine", "think": ROOM_THINK, "output": "Hello"},
            {"round": 1, "phase": "ROOM", "agent": "Agent_1", "role": "honest", "think": "Agent_0 seems fine.", "output": "Hello"},
        ],
    )
    return tmp_path


def test_the_reasoning_names_the_liars_without_the_voter():
    assert flagged_as_liar(VOTE_THINK, "Agent_1") == {"Agent_0"}
    assert flagged_as_liar("Agent_2 did not lie to Agent_0.", "Agent_1") == set()
    assert flagged_as_liar("Agent_1 contradicts Agent_4.", "Agent_1") == {"Agent_4"}


def test_a_broken_line_of_the_log_is_skipped(game):
    records = read_thoughts(next(game.glob("Game_*_Run0")) / "thought.log")
    assert len(records) == 4


def test_the_last_reasoning_before_a_vote_is_related_to_the_vote(game):
    (cell,) = vote_profile([game]).values()
    # Agent_1 voted for Agent_0, a Byzantine, after calling Agent_0 a liar and citing the crowd.
    assert cell["votes"] == 1 and cell["flags_liar"] == 1
    assert cell["votes_liar"] == 1 and cell["votes_other"] == 0
    assert cell["liar_byz"] == 1 and cell["liar_honest"] == 0
    assert cell["crowd"] == 1 and cell["crowd_hit"] == 1 and cell["no_crowd"] == 0
    assert cell["target_unnamed"] == 0


def test_room_reasoning_that_states_an_intent_to_deceive_is_counted_by_role(game):
    stats = intent_profile([game])
    assert (stats["byzantine"]["n"], stats["byzantine"]["intent"], stats["byzantine"]["teammate"]) == (1, 1, 1)
    assert (stats["honest"]["n"], stats["honest"]["intent"]) == (1, 0)


def test_the_report_has_both_tables(game):
    text = report(vote_profile([game]), intent_profile([game]))
    assert "Honest votes and the reasoning before them." in text and "Room reasoning by role." in text
    assert "| byzantine | 1 | 100% | 100% |" in text
