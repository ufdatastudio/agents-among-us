"""Tests for the adapter that lets the existing agents speak and vote in the situation room."""

from pathlib import Path

import pytest

from agents.byzantine_agent import ByzantineAgent
from agents.honest_agent import HonestAgent
from agents.room_prompts import NOBODY_SPOKE
from core.adapters.game_agent import GameAgentAdapter, clean_speech
from core.situation_room import Meeting, Percept, RoomConfig, Tick
from core.timed_discussion import collect_votes, run_timed_discussion

NAMES = ["Agent_0", "Agent_1", "Agent_2"]
HEARD = "[00:01-00:07] Agent_1: I was in Electrical the whole time"


class StubLLM:
    """Stands in for the model client and records what it is asked."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1, max_tokens=160, preserve_think_tags=False):
        self.calls.append(
            {
                "model": model_name,
                "system": system_prompt,
                "prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "preserve_think_tags": preserve_think_tags,
            }
        )
        return self.respond(system_prompt, user_prompt)


def percept(text=HEARD, tick=1, t=10.0):
    return Percept(reader="Agent_0", tick=tick, t=t, cursor=t, modality="text", text=text)


@pytest.fixture
def logs(tmp_path):
    action = tmp_path / "action.log"
    action.write_text("Round 1/5\nCurrent Location: [Cafeteria]\n", encoding="utf-8")
    results = tmp_path / "results.log"
    results.write_text("=== Round Results Log ===\n", encoding="utf-8")
    return {"log_path": str(action), "results_log_path": str(results), "discussion_log_path": str(tmp_path / "d.log")}


def make_adapter(agent, logs, tmp_path, respond, config=None):
    agent.llm = StubLLM(respond)
    calls = []

    def view_factory(name, round_num):
        calls.append((name, round_num))
        return {**logs, "capture_thoughts": False, "require_think_tags": False}

    adapter = GameAgentAdapter(agent, NAMES, config or RoomConfig(), view_factory, tmp_path / "room")
    return adapter, calls


def honest(name="Agent_0"):
    return HonestAgent(name, "red", "stub-model")


def byzantine(name="Agent_0", teammates=("Agent_2",)):
    return ByzantineAgent(name, "red", list(teammates), "stub-model")


def test_clean_speech_removes_the_name_prefix_quotes_and_line_breaks():
    assert clean_speech("**Agent_0:** I was in Cafeteria\nwith Agent_1", "Agent_0") == "I was in Cafeteria with Agent_1"
    assert clean_speech('agent_0: "Not me."', "Agent_0") == "Not me."
    assert clean_speech(None, "Agent_0") == ""


def test_a_reply_shows_the_model_what_the_agent_heard(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "I doubt that, Agent_1.")
    adapter.begin_meeting(Meeting(1, "body found", "Agent_1"))
    adapter.hear(percept())
    reply = adapter.reply(Tick(1, 10.0, 170.0))
    assert reply.text == "I doubt that, Agent_1."
    (call,) = adapter.agent.llm.calls
    assert HEARD in call["prompt"]
    assert "Current Location: [Cafeteria]" in call["prompt"]
    assert "The meeting is at 00:10. It lasts 03:00 in total, so 02:50 remain." in call["prompt"]
    assert "Agent_0" in call["system"]


def test_the_view_is_built_once_per_meeting_not_per_tick(logs, tmp_path):
    adapter, calls = make_adapter(honest(), logs, tmp_path, lambda s, p: "ok")
    adapter.begin_meeting(Meeting(2, "", None))
    for tick in range(1, 4):
        adapter.hear(percept(tick=tick))
        adapter.reply(Tick(tick, tick * 10.0, 100.0))
    assert calls == [("Agent_0", 2)]


def test_silence_is_one_call_with_no_length_retry(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "<silent/>")
    adapter.begin_meeting(Meeting(1, "", None))
    adapter.hear(percept())
    reply = adapter.reply(Tick(1, 10.0, 170.0))
    assert reply.text is None
    assert len(adapter.agent.llm.calls) == 1


def test_a_failed_model_call_is_silence_not_a_statement(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "move")
    adapter.begin_meeting(Meeting(1, "", None))
    adapter.hear(percept())
    assert adapter.reply(Tick(1, 10.0, 170.0)).text is None


def test_an_address_tag_becomes_the_addressee(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "<to>Agent_1</to> Where were you?")
    adapter.begin_meeting(Meeting(1, "", None))
    adapter.hear(percept())
    reply = adapter.reply(Tick(1, 10.0, 170.0))
    assert reply.text == "Where were you?"
    assert reply.addressed_to == ("Agent_1",)


def test_a_byzantine_agent_keeps_its_system_prompt_and_teammates(logs, tmp_path):
    adapter, _ = make_adapter(byzantine(), logs, tmp_path, lambda s, p: "I was with Agent_2.")
    adapter.begin_meeting(Meeting(1, "", None))
    adapter.hear(percept())
    adapter.reply(Tick(1, 10.0, 170.0))
    (call,) = adapter.agent.llm.calls
    assert "Your Teammates: Agent_2" in call["prompt"]
    assert call["system"] == adapter.agent._system_prompt()


def test_the_vote_reads_what_the_agent_heard(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "Agent_1")
    adapter.begin_meeting(Meeting(3, "", None))
    vote = adapter.vote(percept(tick=9), ["Agent_1", "Agent_2", "SKIP"])
    assert vote.agent == "Agent_0" and vote.target == "Agent_1"
    (call,) = adapter.agent.llm.calls
    assert HEARD in call["prompt"]
    written = Path(tmp_path / "room" / "Agent_0_round3_heard.log").read_text(encoding="utf-8")
    assert written.startswith("=== Round 3 ===\n") and HEARD in written


def test_a_vote_that_names_nobody_is_a_skip(logs, tmp_path):
    adapter, _ = make_adapter(honest(), logs, tmp_path, lambda s, p: "I cannot decide.")
    adapter.begin_meeting(Meeting(1, "", None))
    assert adapter.vote(percept(), ["Agent_1", "Agent_2", "SKIP"]).target == "SKIP"


def test_three_wrapped_agents_hold_a_meeting_and_vote(logs, tmp_path):
    config = RoomConfig(budget_s=60.0, tick_s=10.0, quiet_close_ticks=2)
    spoken = {"Agent_0": "I was in Cafeteria and saw Agent_1 leave.", "Agent_1": "<silent/>", "Agent_2": "<silent/>"}
    adapters = []
    for name in NAMES:
        agent = byzantine(name, ["Agent_2"]) if name == "Agent_1" else honest(name)

        def respond(system, prompt, name=name):
            if "You are in a voting phase" in prompt:
                return "Agent_1"
            # Speak into the empty room, then stay silent.
            return spoken[name] if NOBODY_SPOKE in prompt else "<silent/>"

        adapters.append(make_adapter(agent, logs, tmp_path, respond, config)[0])

    room = run_timed_discussion(adapters, Meeting(1, "body found", "Agent_0"), config)
    assert [u.author for u in room.utterances()] == ["Agent_0"]
    assert room.closed_early
    # Agent_1 votes for itself, which counts as a skip.
    assert collect_votes(adapters, room) == {"Agent_0": "Agent_1", "Agent_1": "SKIP", "Agent_2": "Agent_1"}
    heard_by_agent_2 = (tmp_path / "room" / "Agent_2_round1_heard.log").read_text(encoding="utf-8")
    assert "Agent_0: I was in Cafeteria and saw Agent_1 leave." in heard_by_agent_2
