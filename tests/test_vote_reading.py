"""Tests for reading which candidate a voting reply chooses."""

import pytest

from agents.honest_agent import HonestAgent
from agents.vote_reading import RESOLVER_PROMPT, VoteReading, ask_model_to_read_vote, read_vote, read_vote_rules

CANDIDATES = ["Agent_1", "Agent_2", "Agent_3", "Agent_5"]


@pytest.mark.parametrize(
    "reply, target, method",
    [
        ("Agent_3", "Agent_3", "exact"),
        ('"Agent_3."', "Agent_3", "exact"),
        ("**Agent_3**", "Agent_3", "exact"),
        ("agent 3", "Agent_3", "exact"),
        ("I vote for Agent_2", "Agent_2", "marker"),
        ("VOTE: Agent_5", "Agent_5", "marker"),
        ("My vote is Agent_1 because they were alone.", "Agent_1", "marker"),
        ("Agent_2 said they were in Admin, but Agent_5 saw them in Cafeteria. I vote Agent_5.", "Agent_5", "marker"),
        ("I considered voting for Agent_2 but my vote is Agent_3", "Agent_3", "marker"),
        ("Agent_5, not Agent_2", "Agent_5", "single"),
        ("I don't vote for Agent_2, I think Agent_3 did it", "Agent_3", "single"),
        ("Agent_3 acted suspicious.", "Agent_3", "single"),
        ("Agent_2 and Agent_3 both look guilty", None, "ambiguous"),
        ("I cannot decide.", None, "none"),
        ("", None, "none"),
    ],
)
def test_the_rules_read_the_reply(reply, target, method):
    assert read_vote_rules(reply, CANDIDATES) == VoteReading(target, method)


def test_a_name_inside_a_longer_name_is_not_a_mention():
    assert read_vote_rules("Agent_10 is the one", CANDIDATES).target is None
    assert read_vote_rules("Agent_10 is the one", ["Agent_1", "Agent_10"]).target == "Agent_10"


def test_skip_is_read_only_when_it_is_a_candidate():
    assert read_vote_rules("SKIP", CANDIDATES + ["SKIP"]) == VoteReading("SKIP", "exact")
    assert read_vote_rules("SKIP", CANDIDATES).target is None


def test_the_model_reads_only_the_replies_the_rules_leave_open():
    asked = []

    def resolve(reply, names):
        asked.append(reply)
        return "Agent_3"

    assert read_vote("Agent_5", CANDIDATES, resolve).method == "exact"
    assert asked == []
    reading = read_vote("The one who claimed to be in Electrical", CANDIDATES, resolve)
    assert reading == VoteReading("Agent_3", "llm") and len(asked) == 1
    # One word, and the text of a failed call, have nothing to read.
    assert read_vote("move", CANDIDATES, resolve).target is None
    assert read_vote("SKIP (Timeout)", CANDIDATES, resolve).target is None
    assert len(asked) == 1


def test_a_model_answer_that_is_not_a_candidate_is_not_a_vote():
    assert read_vote("Agent_2 and Agent_3 both look guilty", CANDIDATES, lambda reply, names: "Agent_9").method == "ambiguous"
    assert read_vote("Agent_2 and Agent_3 both look guilty", CANDIDATES, lambda reply, names: None).target is None


class Reader:
    """A model client that answers the reading prompt and records it."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1, max_tokens=160, preserve_think_tags=False):
        self.calls.append({"prompt": user_prompt, "temperature": temperature})
        return self.answer


@pytest.mark.parametrize("answer, expected", [("Agent_3", "Agent_3"), ("<think>hmm</think>Agent_5.", "Agent_5"), ("NONE", None), ("Agent_9", None)])
def test_ask_model_to_read_vote_accepts_only_a_candidate(answer, expected):
    agent = HonestAgent("Agent_0", "red", "stub-model")
    agent.llm = Reader(answer)
    heard = "Agent_3: I was in Electrical."
    assert ask_model_to_read_vote(agent, "the one in Electrical", CANDIDATES, heard) == expected
    (call,) = agent.llm.calls
    assert call["temperature"] == 0.0
    assert "the one in Electrical" in call["prompt"] and heard in call["prompt"]
    assert RESOLVER_PROMPT.split("{")[0] in call["prompt"]


def test_a_failing_model_call_leaves_the_reply_unread():
    class Broken:
        def generate(self, *args, **kwargs):
            raise RuntimeError("server down")

    agent = HonestAgent("Agent_0", "red", "stub-model")
    agent.llm = Broken()
    assert ask_model_to_read_vote(agent, "the one in Electrical", CANDIDATES) is None
