"""Tests for the discussion modes of the game engine."""

import json

import pytest

from core.game_engine import GameEngine
from core.situation_room import RoomConfig

COMPOSITION = {
    "name": "RoomTest",
    "honest_model": ["stub-model"],
    "byzantine_model": ["stub-model"],
    "honest_count": 3,
    "byzantine_count": 1,
    "capture_thoughts": False,
}


class StubLLM:
    """Votes for Agent_0, or for Agent_1 when it is Agent_0, and says one fixed thing in a discussion."""

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1, max_tokens=160, preserve_think_tags=False):
        if "You are in a voting phase" in user_prompt:
            candidates = next(line for line in user_prompt.splitlines() if line.startswith("Candidates:"))
            return "Agent_0" if "Agent_0" in candidates else "Agent_1"
        if "timed meeting" in user_prompt and "nobody has spoken yet" not in user_prompt:
            return "<silent/>"
        return "I was in Cafeteria and Agent_0 was acting strange near the body, so I suspect Agent_0."


def make_engine(tmp_path, monkeypatch, **kwargs):
    monkeypatch.chdir(tmp_path)
    engine = GameEngine("t1", num_agents=4, num_rounds=2, **kwargs)
    engine.setup(COMPOSITION)
    for agent in engine.agents:
        agent.llm = StubLLM()
    return engine


@pytest.mark.parametrize("model", ["vllm:Qwen/Qwen2.5-7B-Instruct", "vllm:deepseek-ai/DeepSeek-R1", "vllm:meta-llama/Llama-3.1-8B-Instruct"])
def test_a_vllm_game_refuses_a_model_that_is_not_eligible(tmp_path, monkeypatch, model):
    monkeypatch.chdir(tmp_path)
    engine = GameEngine("t2", num_agents=4, num_rounds=1)
    with pytest.raises(ValueError, match="country of concern|not on the eligible list"):
        engine.setup({**COMPOSITION, "honest_model": [model]})
    assert not (tmp_path / "logs").exists()


def test_a_vllm_game_accepts_an_eligible_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = GameEngine("t3", num_agents=4, num_rounds=1)
    engine.setup({**COMPOSITION, "honest_model": ["vllm:ibm-granite/granite-3.3-8b-instruct"]})
    assert engine.agents[1].model_name == "vllm:ibm-granite/granite-3.3-8b-instruct"


def test_an_unknown_discussion_mode_is_rejected():
    with pytest.raises(ValueError, match="discussion_mode"):
        GameEngine("t0", discussion_mode="simultaneous")


def test_turn_taking_is_still_the_default(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch)
    assert engine.discussion_mode == "turns"
    engine.run_discussion_phase(1)
    log = (tmp_path / engine.logger.base_dir / "discussion.log").read_text(encoding="utf-8")
    assert log.count("I was in Cafeteria") == 8  # 4 agents, 2 messages each
    assert not (tmp_path / engine.logger.base_dir / "room").exists()
    assert engine.state.world_data["agents"]["Agent_0"]["status"] == "ejected"
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["votes_received"] == 3


def test_a_timed_meeting_logs_the_room_and_collects_the_votes(tmp_path, monkeypatch):
    config = RoomConfig(budget_s=60.0, tick_s=10.0, quiet_close_ticks=2)
    engine = make_engine(tmp_path, monkeypatch, discussion_mode="timed", room_config=config)
    engine.run_discussion_phase(1)
    base = tmp_path / engine.logger.base_dir
    log = (base / "discussion.log").read_text(encoding="utf-8")
    assert "=== Round 1 ===" in log
    # Every agent spoke into the empty room at the first tick, then stayed silent.
    assert log.count("I was in Cafeteria") == 4
    rows = [json.loads(line) for line in (base / "room" / "room_round1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "room" and rows[0]["closed_early"] is True
    assert sum(row["type"] == "utterance" for row in rows) == 4
    # The three honest agents vote for Agent_0, and Agent_0 votes for Agent_1.
    assert engine.state.world_data["agents"]["Agent_0"]["status"] == "ejected"
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["votes_received"] == 3
    assert engine.state.world_data["agents"]["Agent_1"]["stats"]["correct_votes"] == 1
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["skipped_votes"] == 0
    assert engine.state.world_data["agents"]["Agent_1"]["stats"]["votes_received"] == 1
    assert (base / "room" / "Agent_1_round1_heard.log").exists()


def test_a_board_meeting_writes_its_claims_to_the_ledger_and_shows_the_board(tmp_path, monkeypatch):
    config = RoomConfig(budget_s=60.0, tick_s=10.0, quiet_close_ticks=2, board=True)
    engine = make_engine(tmp_path, monkeypatch, discussion_mode="timed", room_config=config)
    engine.run_discussion_phase(1)
    base = tmp_path / engine.logger.base_dir
    claims = [r for r in engine.state.ledger.records if r.type == "claim"]
    assert {r.author for r in claims} == {"Agent_0", "Agent_1", "Agent_2", "Agent_3"}
    assert {r.subject for r in claims if r.content["kind"] == "accuse"} == {"Agent_0"}
    written = [json.loads(line) for line in (base / "room" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(written) == len(engine.state.ledger)
    assert "Board of this meeting" in (base / "room" / "Agent_1_round1_heard.log").read_text(encoding="utf-8")


def test_a_human_seat_falls_back_to_turn_taking(tmp_path, monkeypatch):
    engine = make_engine(tmp_path, monkeypatch, discussion_mode="timed")
    assert engine._use_timed_discussion(engine.agents) is True
    setattr(engine.agents[1], "is_human", True)
    assert engine._use_timed_discussion(engine.agents) is False
