"""Tests for the rule that every model agent votes, and for the meeting that is held again without a vote."""

import json
import re

import pytest

from agents.prompts import HONEST_DEFAULT_VOTE_INSTRUCTIONS_WITH_THOUGHTS, BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS, require_vote
from core.game_engine import GameEngine, MeetingFailed
from core.situation_room import RoomConfig
from core.voting import VOTE_TEMPERATURES, ask_until_valid, vote_feedback

COMPOSITION = {
    "name": "VoteTest",
    "honest_model": ["stub-model"],
    "byzantine_model": ["stub-model"],
    "honest_count": 3,
    "byzantine_count": 1,
    "capture_thoughts": False,
}
SAID = "I was in Cafeteria and Agent_0 was acting strange near the body, so I suspect Agent_0."


class VoteStub:
    """Votes for Agent_0 (Agent_1 when it is Agent_0) and gives one voter a bad reply for its first asks.

    Attributes:
        self_voter: Agent that votes for itself until it is told the reply is not valid, or None.
        bad_voter: Agent whose first `bad_asks` vote replies name nobody, or None.
        bad_asks: How many of its asks fail, or None for all of them.
        votes: Every vote ask as (voter, temperature, prompt).
    """

    def __init__(self, bad_voter=None, bad_asks=None, self_voter=None):
        self.self_voter = self_voter
        self.bad_voter = bad_voter
        self.bad_asks = bad_asks
        self.votes = []
        self.readings = []

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1, max_tokens=160, preserve_think_tags=False):
        if "Say which candidate the reply below votes for" in user_prompt:
            # The model that reads a vote reply finds no candidate in "I abstain."
            self.readings.append(user_prompt)
            return "NONE"
        if "You are in a voting phase" in user_prompt:
            line = next(row for row in user_prompt.splitlines() if row.startswith("Candidates:"))
            named = set(re.findall(r"Agent_\d", line))
            voter = next(name for name in ("Agent_0", "Agent_1", "Agent_2", "Agent_3") if name not in named)
            asked_before = sum(1 for who, _, _ in self.votes if who == voter)
            self.votes.append((voter, temperature, user_prompt))
            if voter == self.bad_voter and (self.bad_asks is None or asked_before < self.bad_asks):
                return "I abstain."
            if voter == self.self_voter and "not a valid vote" not in user_prompt:
                return voter
            return "Agent_0" if voter != "Agent_0" else "Agent_1"
        if "timed meeting" in user_prompt and "nobody has spoken yet" not in user_prompt:
            return "<silent/>"
        return SAID


def make_engine(tmp_path, monkeypatch, stub, **kwargs):
    monkeypatch.chdir(tmp_path)
    engine = GameEngine("v1", num_agents=4, num_rounds=2, **kwargs)
    engine.setup(COMPOSITION)
    for agent in engine.agents:
        agent.llm = stub
    return engine


def game_dir(tmp_path, engine):
    return tmp_path / engine.logger.base_dir


def test_require_vote_removes_every_offer_to_skip():
    prompt = (
        "Note: Skipping your vote is an option. Your vote is ANONYMOUS.\n"
        + HONEST_DEFAULT_VOTE_INSTRUCTIONS_WITH_THOUGHTS
        + BYZANTINE_DEFAULT_VOTE_INSTRUCTIONS
    )
    result = require_vote(prompt)
    assert "SKIP" not in result and "Skipping" not in result
    assert "You must vote for one of the candidates." in result
    assert "reply with ONLY the exact name of the agent." in result


def test_ask_until_valid_stops_at_the_first_candidate():
    temperatures = []

    def ask(temperature, feedback):
        temperatures.append(temperature)
        return "Blue" if len(temperatures) == 2 else "no idea"

    assert ask_until_valid(ask, ["Blue", "Red"], retries=2) == "Blue"
    assert temperatures == list(VOTE_TEMPERATURES[:2])


def test_ask_until_valid_gives_none_when_every_ask_fails():
    asked = []
    assert ask_until_valid(lambda temperature, feedback: asked.append(temperature), ["Blue"], retries=2) is None
    assert len(asked) == 3
    assert ask_until_valid(lambda temperature, feedback: "SKIP", ["Blue"], retries=2) is None


def test_no_vote_prompt_offers_a_skip_by_default(tmp_path, monkeypatch):
    stub = VoteStub()
    engine = make_engine(tmp_path, monkeypatch, stub)
    engine.run_discussion_phase(1)
    assert len(stub.votes) == 4
    assert all("SKIP" not in prompt for _, _, prompt in stub.votes)
    assert all("You must vote" in prompt for _, _, prompt in stub.votes)


def test_a_vote_that_names_nobody_is_asked_again_with_a_higher_temperature(tmp_path, monkeypatch):
    stub = VoteStub(bad_voter="Agent_1", bad_asks=1)
    engine = make_engine(tmp_path, monkeypatch, stub)
    engine.run_discussion_phase(1)
    asks = [temperature for voter, temperature, _ in stub.votes if voter == "Agent_1"]
    assert asks == list(VOTE_TEMPERATURES[:2])
    # The reply "I abstain." names nobody, so the model was asked to read it once.
    assert len(stub.readings) == 1
    assert not (game_dir(tmp_path, engine) / "voided_meetings.jsonl").exists()
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["votes_received"] == 3


def test_a_meeting_without_a_valid_vote_is_rewound_and_held_again(tmp_path, monkeypatch):
    # Three asks fail, which is every ask of the first attempt, and the next attempt gets a valid vote.
    stub = VoteStub(bad_voter="Agent_1", bad_asks=3)
    engine = make_engine(tmp_path, monkeypatch, stub)
    engine.run_discussion_phase(1)
    base = game_dir(tmp_path, engine)
    (line,) = (base / "voided_meetings.jsonl").read_text(encoding="utf-8").splitlines()
    voided = json.loads(line)
    assert voided["round"] == 1 and voided["attempt"] == 1 and voided["missing_voters"] == ["Agent_1"]
    assert voided["discussion"].count(SAID) == 8
    # The logs the agents read hold the second attempt only.
    discussion = (base / "discussion.log").read_text(encoding="utf-8")
    assert discussion.count("=== Round 1 ===") == 1 and discussion.count(SAID) == 8
    chat = (base / "discussion_chat.csv").read_text(encoding="utf-8")
    assert chat.count("Cafeteria") == 8
    for name in ("Agent_0", "Agent_1", "Agent_2", "Agent_3"):
        role = "Byz" if name == "Agent_0" else "Honest"
        votes = (base / role / name / "vote.log").read_text(encoding="utf-8")
        assert votes.count("Voted for") == 1
    stats = engine.state.world_data["agents"]
    assert all(agent["stats"]["skipped_votes"] == 0 for agent in stats.values())
    assert stats["Agent_0"]["stats"]["votes_received"] == 3 and stats["Agent_0"]["status"] == "ejected"


def test_a_timed_meeting_without_a_valid_vote_is_held_again(tmp_path, monkeypatch):
    stub = VoteStub(bad_voter="Agent_2", bad_asks=3)
    config = RoomConfig(budget_s=60.0, tick_s=10.0, quiet_close_ticks=2)
    engine = make_engine(tmp_path, monkeypatch, stub, discussion_mode="timed", room_config=config)
    engine.run_discussion_phase(1)
    base = game_dir(tmp_path, engine)
    assert (base / "room" / "room_round1_voided1.jsonl").exists()
    assert (base / "room" / "room_round1.jsonl").exists()
    discussion = (base / "discussion.log").read_text(encoding="utf-8")
    assert discussion.count(SAID) == 4
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["votes_received"] == 3


def test_the_game_stops_when_every_attempt_lacks_a_vote(tmp_path, monkeypatch):
    stub = VoteStub(bad_voter="Agent_1")
    engine = make_engine(tmp_path, monkeypatch, stub, max_meeting_attempts=2)
    with pytest.raises(MeetingFailed, match="Agent_1"):
        engine.run_discussion_phase(1)
    base = game_dir(tmp_path, engine)
    assert len((base / "voided_meetings.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    # Nothing of the voided meetings is left in the logs, and no vote was recorded.
    assert (base / "discussion.log").read_text(encoding="utf-8").count(SAID) == 0
    assert all(agent["status"] == "active" for agent in engine.state.world_data["agents"].values())
    assert "Voted for" not in (base / "Honest" / "Agent_2" / "vote.log").read_text(encoding="utf-8")


def test_allow_skip_keeps_the_skip_option_and_counts_it(tmp_path, monkeypatch):
    stub = VoteStub(bad_voter="Agent_1")
    engine = make_engine(tmp_path, monkeypatch, stub, allow_skip=True)
    engine.run_discussion_phase(1)
    assert all("Skipping your vote is an option" in prompt for _, _, prompt in stub.votes)
    assert engine.state.world_data["agents"]["Agent_1"]["stats"]["skipped_votes"] == 1
    assert not (game_dir(tmp_path, engine) / "voided_meetings.jsonl").exists()


@pytest.mark.parametrize("settings", [{"max_meeting_attempts": 0}, {"vote_retries": -1}])
def test_the_limits_are_checked(settings):
    with pytest.raises(ValueError, match="max_meeting_attempts"):
        GameEngine("v2", **settings)


def test_a_retry_tells_the_agent_what_was_wrong_and_who_it_may_vote_for():
    notes = []

    def ask(temperature, feedback):
        notes.append(feedback)
        return "Blue" if feedback else "Agent_5"

    assert ask_until_valid(ask, ["Blue", "Red"], retries=1, reply_of=lambda: "Agent_5") == "Blue"
    assert notes[0] is None
    assert '"Agent_5"' in notes[1] and "Blue, Red" in notes[1] and "yourself" in notes[1] and "out of the game" in notes[1]
    assert "empty" in vote_feedback("", ["Blue"])


def test_an_agent_that_votes_for_itself_is_corrected_on_the_second_ask(tmp_path, monkeypatch):
    stub = VoteStub(self_voter="Agent_2")
    engine = make_engine(tmp_path, monkeypatch, stub)
    engine.run_discussion_phase(1)
    assert not (game_dir(tmp_path, engine) / "voided_meetings.jsonl").exists()
    corrected = [prompt for voter, _, prompt in stub.votes if voter == "Agent_2" and "not a valid vote" in prompt]
    assert len(corrected) == 1 and 'Your last reply was "Agent_2"' in corrected[0]
    assert engine.state.world_data["agents"]["Agent_0"]["stats"]["votes_received"] == 3


def test_the_vote_prompt_tells_the_agent_that_it_cannot_vote_for_itself(tmp_path, monkeypatch):
    stub = VoteStub()
    engine = make_engine(tmp_path, monkeypatch, stub)
    engine.run_discussion_phase(1)
    for voter, _, prompt in stub.votes:
        assert f"You are {voter}, so you cannot vote for yourself." in prompt
    assert "You are Agent_0" not in require_vote("Note: Skipping your vote is an option. Your vote is ANONYMOUS.")
