"""Tests for the prompts an agent sees in the situation room."""

from agents.room_prompts import NOBODY_SPOKE, build_room_prompt, room_instructions

PROMPT = dict(
    recent_action_log="Current Location: [Cafeteria]",
    results_log="Round 1 Votes Received: {}",
    transcript="[00:01-00:07] Agent_1: I was in Electrical",
    now="00:30",
    budget="03:00",
    remaining="02:30",
    max_words=40,
)


def test_honest_prompt_has_the_room_rules_and_the_transcript():
    prompt = build_room_prompt(role="honest", **PROMPT)
    assert "[00:01-00:07] Agent_1: I was in Electrical" in prompt
    assert "Current Location: [Cafeteria]" in prompt
    assert "The meeting is at 00:30. It lasts 03:00 in total, so 02:30 remain." in prompt
    assert "at most 40 words" in prompt
    assert "<silent/>" in prompt
    assert "<to>Name</to>" in prompt
    assert "Move the group toward ejecting a Byzantine" in prompt


def test_the_room_replaces_the_turn_based_length_rule():
    prompt = build_room_prompt(role="honest", **PROMPT)
    assert "at least 45 words" not in prompt
    assert "80 words" not in prompt


def test_byzantine_prompt_keeps_the_deception_rules_and_names_the_teammates():
    prompt = build_room_prompt(role="byzantine", teammates=["Agent_2", "Agent_3"], **PROMPT)
    assert "NEVER confess" in prompt
    assert "Your Teammates: Agent_2, Agent_3" in prompt


def test_honest_prompt_never_shows_teammates():
    prompt = build_room_prompt(role="honest", teammates=["Agent_2"], **PROMPT)
    assert "Teammates" not in prompt


def test_an_empty_room_says_nobody_has_spoken():
    prompt = build_room_prompt(role="honest", **{**PROMPT, "transcript": "  "})
    assert NOBODY_SPOKE in prompt


def test_private_reasoning_rules_appear_only_when_thoughts_are_captured():
    assert "<think>" not in room_instructions("honest", 40, capture_thoughts=False)
    honest = room_instructions("honest", 40, capture_thoughts=True)
    assert "<think>...</think>" in honest
    assert "<silent/>" in honest.split("</think>")[-1]
    assert "lie or redirect" in room_instructions("byzantine", 40, capture_thoughts=True)
    assert "who you believe" in honest


def test_a_longer_word_limit_asks_for_a_longer_statement():
    short = room_instructions("honest", 40, False)
    longer = room_instructions("honest", 90, False)
    assert "one or two short sentences and at most 40 words" in short
    assert "a few sentences and at most 90 words" in longer
    assert "one or two short sentences" not in longer
