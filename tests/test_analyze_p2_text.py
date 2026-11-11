"""Tests for the transcript analysis of P2 games."""

import json
from pathlib import Path

import pytest

from scripts.analyze_p2_text import (
    Claim,
    Game,
    Statement,
    agent_names,
    claim_profile,
    extract_claims,
    extract_edges,
    finished_ids,
    influence_profile,
    load_games,
    jaccard,
    label_statement,
    listening_profile,
    parse_game,
    report,
    verify_claim,
)

ROOMS = ["Cafeteria", "Storage", "UpperEngine", "Navigation"]


def edges(speaker, text):
    return [(e.kind, e.target) for e in extract_edges(speaker, text, 0)]


def test_agent_names_reads_the_spellings_models_use():
    assert agent_names("Agent_3, Agent 4 and Agents_5, then agent6") == ["Agent_3", "Agent_4", "Agent_5", "Agent_6"]
    assert agent_names("Let's vote 8 and get 2 out") == ["Agent_8", "Agent_2"]
    assert agent_names("Agent_10 and Agent_1") == ["Agent_10", "Agent_1"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("I suspect Agent_5 because of the timing.", [("accuse", "Agent_5")]),
        ("I trust Agent_8, who was with me.", [("trust", "Agent_8")]),
        ("Don't trust Agent_6's shifting story.", [("accuse", "Agent_6")]),
        ("I didn't see anything suspicious from Agent_6.", []),
        ("Agent_4 was lingering near the vent, which is suspicious.", [("accuse", "Agent_4")]),
        ("Agent_9, Your alibi is weak.", [("accuse", "Agent_9")]),
        ("I found Agent_3 dead in Storage.", []),
        ("Vote 8.", [("accuse", "Agent_8")]),
        ("I suspect Agent_5. I think they look guilty.", [("accuse", "Agent_5")]),
    ],
)
def test_edges_read_who_is_accused_and_who_is_vouched_for(text, expected):
    assert edges("Agent_1", text) == expected


def test_a_speaker_does_not_accuse_itself_or_a_victim_of_the_round():
    assert edges("Agent_1", "Agent_1 is suspicious, says Agent_1.") == []
    assert edges("Agent_1", "The body of Agent_3 looks suspicious.") == []


def test_a_pronoun_after_a_victim_does_not_point_at_the_agent_named_before():
    assert edges("Agent_1", "I trust Agent_5. The body of Agent_3 looks suspicious to them.") == [("trust", "Agent_5")]


def test_labels_cover_the_kinds_of_speech_act():
    assert {"accuse", "vote_call"} <= label_statement("Agent_2 is suspicious, so vote Agent_2.")
    assert "report" in label_statement("I saw Agent_4 in Storage.")
    assert "defend" in label_statement("I didn't do it, I swear.")
    assert "question" in label_statement("Where were you?")
    assert "challenge" in label_statement("Please clarify your whereabouts.")
    assert label_statement("Hmm.") == {"other"}


def make_game(**kwargs):
    # Agent_0 was in Cafeteria and then Storage with Agent_1. Agent_2 stayed in Navigation.
    base = dict(
        game_id="t_clean-180_1", model="m", protocol="clean-180", seed=1,
        roles={"Agent_0": "byzantine", "Agent_1": "honest", "Agent_2": "honest", "Agent_3": "honest"},
        rooms={
            "Agent_0": [(1, 1, "Cafeteria"), (1, 2, "Storage")],
            "Agent_1": [(1, 1, "UpperEngine"), (1, 2, "Storage")],
            "Agent_2": [(1, 1, "Navigation"), (1, 2, "Navigation")],
        },
    )
    return Game(**{**base, **kwargs})


@pytest.mark.parametrize(
    "speaker, text, claim, verdict",
    [
        ("Agent_0", "I was in Storage the whole time.", Claim("self", "Agent_0", "Agent_0", "Storage"), True),
        ("Agent_0", "I was in Navigation the whole time.", Claim("self", "Agent_0", "Agent_0", "Navigation"), False),
        ("Agent_0", "I saw Agent_2 in Navigation.", Claim("seen", "Agent_0", "Agent_2", "Navigation"), True),
        ("Agent_0", "I saw Agent_2 in Storage.", Claim("seen", "Agent_0", "Agent_2", "Storage"), False),
        ("Agent_0", "I was in Storage with Agent_1.", Claim("with", "Agent_0", "Agent_1", "Storage"), True),
        ("Agent_0", "I was in Storage with Agent_2.", Claim("with", "Agent_0", "Agent_2", "Storage"), False),
    ],
)
def test_claims_are_checked_against_where_agents_were(speaker, text, claim, verdict):
    game = make_game()
    assert claim in extract_claims(speaker, text, ROOMS)
    assert verify_claim(game, claim, meeting=1) is verdict


def test_a_sentence_about_the_listener_or_with_a_negation_gives_no_claim_about_the_speaker():
    assert extract_claims("Agent_0", "You were in Storage with me.", ROOMS) == []
    assert extract_claims("Agent_0", "I was not in Storage.", ROOMS) == []
    claims = extract_claims("Agent_0", "I saw Agent_1 with Agent_2 in Storage.", ROOMS)
    assert {(c.kind, c.subject) for c in claims} == {("seen", "Agent_1"), ("seen", "Agent_2")}


def test_a_claim_about_a_later_round_is_not_checked_against_the_earlier_meeting():
    game = make_game(rooms={"Agent_0": [(1, 1, "Cafeteria"), (2, 1, "Storage")]})
    claim = Claim("self", "Agent_0", "Agent_0", "Storage")
    assert verify_claim(game, claim, meeting=1) is False
    assert verify_claim(game, claim, meeting=2) is True
    assert verify_claim(make_game(rooms={}), claim, meeting=1) is None


def statement(index, speaker, role, text, meeting=1, **kwargs):
    return Statement(meeting, index, speaker, role, text, **kwargs)


def test_claim_profile_counts_false_claims_by_role():
    game = make_game(
        statements=[
            statement(0, "Agent_0", "byzantine", "I was in Navigation."),
            statement(1, "Agent_1", "honest", "I was in Storage."),
        ]
    )
    cell = claim_profile([game])
    assert cell[("m", "clean", "byzantine")]["false"] == 1
    assert cell[("m", "clean", "honest")]["false"] == 0 and cell[("m", "clean", "honest")]["claims"] == 1


def test_influence_profile_places_each_candidate_by_who_accused_them():
    # Agent_0 (Byzantine) accuses Agent_2, Agent_3 (honest) accuses Agent_0, and Agent_1 votes for Agent_2.
    game = make_game(
        statements=[
            statement(0, "Agent_0", "byzantine", "I suspect Agent_2."),
            statement(1, "Agent_3", "honest", "Agent_0 is suspicious."),
        ],
        votes={1: {"Agent_1": "Agent_2"}},
        active={1: {"Agent_0", "Agent_1", "Agent_2", "Agent_3"}},
    )
    (cell,) = influence_profile([game]).values()
    assert cell["honest_votes"] == 1
    assert cell["honest|byzantine only|pairs"] == 1 and cell["honest|byzantine only|votes"] == 1
    assert cell["byzantine|honest only|pairs"] == 1 and cell["byzantine|honest only|votes"] == 0
    assert cell["honest|nobody|pairs"] == 1


def test_listening_profile_counts_answers_that_name_the_accuser():
    game = make_game(
        statements=[
            statement(0, "Agent_0", "byzantine", "Agent_2 is suspicious.", tick=1, cursor=0.0, start_s=0.5),
            statement(1, "Agent_2", "honest", "Agent_0, I was in Navigation.", tick=2, cursor=10.0, start_s=10.5),
            statement(2, "Agent_3", "honest", "I saw nothing.", tick=1, cursor=0.0, start_s=1.0),
        ],
    )
    (cell,) = listening_profile([game]).values()
    assert (cell["accused"], cell["answerable"], cell["answered"]) == (1, 1, 1)
    assert cell["timed_statements"] == 3
    # Agent_2 heard Agent_0 and Agent_3 before its cursor, and Agent_0 and Agent_3 spoke in the same tick as each other.
    assert cell["concurrent_total"] == 2


def test_jaccard():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0


def write_game_folder(root: Path) -> Path:
    game = root / "Game_tag_clean-180_1_Run0"
    for role, name, room, seen in (("Byz", "Agent_0", "Storage", "['Agent_1']"), ("Honest", "Agent_1", "Storage", "['Agent_0']")):
        folder = game / role / name
        folder.mkdir(parents=True)
        (folder / "action.log").write_text(
            f"=== Action Log ===\n\nRound 1/10\nPlayers Remaining: 2\nCurrent Location: [{room}]\n  -> Occupants: {seen}\nBodies Seen: None\n",
            encoding="utf-8",
        )
        other = "Agent_1" if name == "Agent_0" else "Agent_0"
        (folder / "vote.log").write_text(f"=== Voting History ===\nRound 1: Voted for {other}\n", encoding="utf-8")
    (game / "discussion_chat.csv").write_text(
        "discussion_num,reason,agent_num,model,role,message\n"
        '1,why,1,vllm:org/model,Honest,"Agent 0 is suspicious, I was in Storage."\n'
        "1,why,0,vllm:org/model,Byzantine,I was in Storage with Agent_1.\n",
        encoding="utf-8",
    )
    (game / "roundResults.log").write_text(
        "=== Round Results Log ===\n\n=== Round 1 ===\nRound 1 Votes Received: {'Agent_0': 1, 'Agent_1': 1}\n"
        "Player Statuses: {'Agent_0': 'active', 'Agent_1': 'active'}\n",
        encoding="utf-8",
    )
    room = game / "room"
    room.mkdir()
    rows = [
        {"type": "room"},
        {"type": "utterance", "author": "Agent_1", "tick": 1, "start_s": 0.5, "text": "Agent 0 is suspicious, I was in Storage.", "intended_words": 50, "spoken_words": 40, "cursor": 0.0},
        {"type": "utterance", "author": "Agent_0", "tick": 2, "start_s": 10.5, "text": "I was in Storage with Agent_1.", "intended_words": 7, "spoken_words": 7, "cursor": 10.0},
    ]
    (room / "room_round1.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return game


def test_parse_game_reads_roles_statements_votes_rooms_and_the_room_record(tmp_path):
    game = parse_game(write_game_folder(tmp_path))
    assert game is not None
    assert (game.model, game.protocol, game.seed) == ("vllm:org/model", "clean-180", 1)
    assert game.roles == {"Agent_0": "byzantine", "Agent_1": "honest"}
    first, second = game.statements
    # The no-break space in the first statement is read as a space, and the room record is matched by author and text.
    assert "Agent 0 is suspicious" in first.text
    assert (first.tick, first.intended, first.spoken) == (1, 50, 40) and second.tick == 2
    assert game.votes == {1: {"Agent_0": "Agent_1", "Agent_1": "Agent_0"}}
    assert game.rooms["Agent_0"] == [(1, 1, "Storage")] and game.beside["Agent_0"][(1, 1)] == {"Agent_1"}
    assert game.active[1] == {"Agent_0", "Agent_1"}


def test_the_report_has_every_table_and_parses_a_game_folder(tmp_path):
    game = parse_game(write_game_folder(tmp_path))
    text = report([game])
    for title in ("speech act", "Claims about places", "Accusations by role", "Honest votes by what", "What agents pick up"):
        assert title in text
    assert parse_game(tmp_path) is None


def test_only_the_games_that_finished_are_read(tmp_path):
    write_game_folder(tmp_path)
    results = tmp_path / "p2.jsonl"
    results.write_text(json.dumps({"game_id": "tag_clean-180_1", "result": "Byzantines Win"}) + "\n" + json.dumps({"game_id": "tag_clean-60_1", "result": None}) + "\n", encoding="utf-8")
    assert finished_ids([results]) == {"tag_clean-180_1"}
    assert [g.game_id for g in load_games([tmp_path], finished_ids([results]))] == ["tag_clean-180_1"]
    assert load_games([tmp_path], {"tag_clean-60_1"}) == []
    assert len(load_games([tmp_path])) == 1
