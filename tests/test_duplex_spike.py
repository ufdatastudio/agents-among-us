"""Tests for the pure parts of the duplex spike."""

import asyncio

import pytest

from core.duplex import Burst, DuplexAgent, StreamPrompt, run_duplex_meeting
from core.ledger import Ledger
from scripts.duplex_spike import (
    RecordingHandle,
    beat_tokens,
    intro,
    replay_setup,
    round_log,
    scenario,
    summarize,
    transcript,
    turn_format,
)
from tests.test_duplex import chatml


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert kwargs == {"enable_thinking": False}
        return chatml(messages, add_generation_prompt)


def test_the_scenario_has_one_byzantine_agent_and_leaves_out_the_victim():
    setups = scenario(6, seed=1)
    assert [s.name for s in setups] == [f"Agent_{i}" for i in range(5)]
    assert [s.role for s in setups] == ["byzantine", "honest", "honest", "honest", "honest"]
    assert "Agent_5 dead in Storage" in setups[1].memory
    assert "Tick 3: you were in Storage" in setups[0].memory
    assert scenario(6, seed=1) == setups
    assert scenario(6, seed=2) != setups


def test_a_meeting_needs_four_agents():
    with pytest.raises(ValueError):
        scenario(3, seed=1)


def test_the_intro_gives_the_memory_the_role_and_the_beat_rules():
    honest, byzantine = scenario(5, 1)[1], scenario(5, 1)[0]
    text = intro(honest, beat_s=2.0, budget_s=90.0, statement_words=30)
    assert honest.memory in text and "01:30" in text
    assert "beats of 2 seconds" in text and "at most 5 words" in text and "<silent/>" in text
    assert "NEVER confess" in intro(byzantine, 2.0, 90.0, 30) and "NEVER confess" not in text


def test_a_burst_has_tokens_for_the_words_that_fit_in_a_beat():
    assert beat_tokens(2.0) == 8
    assert beat_tokens(0.1) == 2


def test_the_turn_format_comes_from_the_tokenizer_template_with_its_arguments():
    fmt = turn_format(FakeTokenizer(), {"enable_thinking": False})
    assert fmt.cue == "<|im_end|>\n<|im_start|>assistant\n"


def test_recording_keeps_what_was_sent_and_written_and_the_run_is_summarized(scripted):
    ledger = Ledger()
    inner = scripted([Burst("hello", "length", 12, 3, 0.5), Burst(" there", "stop", 4, 2, 0.25)])
    handle = RecordingHandle(inner, "Agent_1")
    fmt = turn_format(FakeTokenizer(), {"enable_thinking": False})
    agent = DuplexAgent("Agent_1", handle, fmt, "sys", "intro", ledger.subscribe("Agent_1"), 2.0, 8)
    meeting = asyncio.run(run_duplex_meeting([agent], ledger, 4.0, 2.0))
    assert [row["burst"] for row in handle.rows] == ["hello", " there"]
    assert handle.rows[0]["finish"] == "length" and handle.rows[0]["output_tokens"] == 3
    assert inner.closed
    assert transcript(ledger) == "[00:00-00:04] Agent_1: hello there"
    summary = summarize(meeting)
    assert summary["beats"] == 2 and summary["total"]["words"] == 2 and summary["agents"]["Agent_1"]["statements"] == 1


ACTION_LOG = """=== Action Log for Agent_3 ===

Round 1/10
Current Location: [Storage]
Action 1 Selected: wait -> none
Round End.

Round 2/10
Current Location: [Shields]
"""


def test_a_round_log_is_cut_between_the_headers_of_its_round_and_the_next():
    assert round_log(ACTION_LOG, 1).splitlines()[0] == "Round 1/10"
    assert "Shields" not in round_log(ACTION_LOG, 1)
    assert round_log(ACTION_LOG, 2).endswith("[Shields]")
    assert round_log("no headers", 1) == "no headers"


def test_a_replay_gives_the_listeners_their_logs_and_the_clips_of_everyone_else_in_that_round(tmp_path):
    import json

    game, speech = tmp_path / "game", tmp_path / "speech"
    for name in ("Agent_3", "Agent_4"):
        (game / "Honest" / name).mkdir(parents=True)
        (game / "Honest" / name / "action.log").write_text(ACTION_LOG, encoding="utf-8")
    speech.mkdir()
    rows = [
        {"round_num": 1, "author": "Agent_0", "start_s": 1.0, "text": "hi", "file": "a.wav"},
        {"round_num": 1, "author": "Agent_3", "start_s": 2.0, "text": "me", "file": "b.wav"},
        {"round_num": 2, "author": "Agent_1", "start_s": 3.0, "text": "later", "file": "c.wav"},
        {"round_num": 1, "author": "Agent_4", "start_s": 4.0, "text": "you", "file": "d.wav"},
    ]
    (speech / "speech.json").write_text(json.dumps({"voices": {}, "clips": rows}), encoding="utf-8")
    setups, clips = replay_setup(game, speech, ["Agent_3", "Agent_4"], 1)
    assert [(s.name, s.role) for s in setups] == [("Agent_3", "honest"), ("Agent_4", "honest")]
    assert "Round 2" not in setups[0].memory
    assert clips == [("Agent_0", speech / "a.wav", 1.0, "hi")]
    with pytest.raises(FileNotFoundError):
        replay_setup(game, speech, ["Agent_0"], 1)
