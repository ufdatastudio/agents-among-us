"""Tests for the P2 protocols, the game runner's helpers and the summary statistics."""

import json
from types import SimpleNamespace

import pytest

from config.protocols import NO_DISCUSSION, P2_PROTOCOLS, SEQUENTIAL, protocol_by_name, timed
from core.game_engine import GameEngine
from core.state import GameState
from scripts.run_p2 import (
    composition,
    game_command,
    load_done,
    parse_output,
    plan,
    read_discussion,
    read_redone,
    read_stats,
    read_tokens,
    scenario_name,
)
from scripts.summarize_p2 import format_table, summarize, wilson


def test_p2_has_eleven_distinct_cells_in_the_documented_order():
    names = [p.name for p in P2_PROTOCOLS]
    assert len(names) == len(set(names)) == 11
    assert names[0] == "sequential" and names[-1] == "none"
    assert names[1:10] == [f"{o}-{b}" for o in ("clean", "interleaved", "noisy") for b in (60, 180, 300)]


def test_protocol_flags_select_the_mode_budget_and_noise():
    assert SEQUENTIAL.args == ("--discussion_mode", "turns")
    assert NO_DISCUSSION.args == ("--skip_discussion",)
    clean = timed("clean", 180)
    assert "--room_budget_s" in clean.args and clean.args[clean.args.index("--room_budget_s") + 1] == "180"
    assert "--room_dropout" not in clean.args
    noisy = timed("noisy", 60)
    assert noisy.args[noisy.args.index("--room_dropout") + 1] == "0.1"
    assert noisy.args[noisy.args.index("--room_tag_policy") + 1] == "first_chunk"


def test_an_unknown_protocol_is_an_error():
    assert protocol_by_name("clean-180").name == "clean-180"
    assert protocol_by_name("board-180").args[-1] == "--room_board"
    with pytest.raises(KeyError, match="no protocol named"):
        protocol_by_name("clean-90")


def test_the_plan_visits_every_protocol_of_a_seed_before_the_next_seed():
    games = plan(P2_PROTOCOLS, range(1, 3), done=set())
    assert len(games) == 22
    assert [seed for _, seed in games] == [1] * 11 + [2] * 11
    assert [p.name for p, _ in games[:11]] == [p.name for p in P2_PROTOCOLS]


def test_the_plan_skips_finished_games():
    games = plan(P2_PROTOCOLS, range(1, 3), done={("sequential", 1), ("none", 2)})
    assert len(games) == 20
    assert ("sequential", 1) not in {(p.name, s) for p, s in games}


def test_a_game_command_carries_the_seed_and_the_protocol_flags():
    command = game_command(timed("interleaved", 300), 7, "job_interleaved-300_7", "p2_model", 10, 10)
    assert command[1:3] == ["-m", "main"]
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--room_seed") + 1] == "7"
    assert command[command.index("--game_id") + 1] == "job_interleaved-300_7"
    assert command[command.index("--room_overlap") + 1] == "interleaved"


def test_the_composition_runs_every_agent_on_the_served_model():
    comp = composition("p2_x", "ibm-granite/granite-3.3-8b-instruct")
    assert comp["honest_model"] == comp["byzantine_model"] == ["vllm:ibm-granite/granite-3.3-8b-instruct"]
    assert (comp["honest_count"], comp["byzantine_count"]) == (8, 2)
    assert comp["capture_thoughts"] is False


def test_parse_output_reads_the_result_and_the_rounds():
    text = "--- Round 1 Movement Phase ---\n--- Round 2 Movement Phase ---\nGame Over. Result: Byzantines Win\n"
    assert parse_output(text) == {"result": "Byzantines Win", "winner": "byzantine", "rounds_played": 2}
    honest = parse_output("--- Round 3 Movement Phase ---\nGame Over. Result: Honest Agents Win, Max Rounds Reached")
    assert honest["winner"] == "honest" and honest["rounds_played"] == 3
    crashed = parse_output("--- Round 1 Movement Phase ---\nTraceback")
    assert crashed["result"] is None and crashed["winner"] is None


def test_read_stats_sums_votes_by_team(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text(
        "agent_name,alignment,correct_votes,incorrect_votes,skipped_votes,ejections\n"
        "Agent_0,B,0,2,1,1\nAgent_1,H,3,1,0,0\nAgent_2,H,2,0,1,0\n",
        encoding="utf-8",
    )
    stats = read_stats(path)
    assert stats["honest_correct_votes"] == 5 and stats["honest_incorrect_votes"] == 1
    assert stats["byzantine_incorrect_votes"] == 2 and stats["byzantine_ejections"] == 1
    assert read_stats(tmp_path / "missing.csv") == {}


def test_read_discussion_counts_meetings_and_statements(tmp_path):
    path = tmp_path / "discussion.log"
    path.write_text("=== Discussion Log ===\n\n=== Round 1 ===\nAgent_1: hello\nAgent_2: hi\n=== Round 3 ===\nAgent_1: bye\n")
    assert read_discussion(path) == {"meetings": 2, "statements": 3}
    assert read_discussion(tmp_path / "none.log") == {"meetings": 0, "statements": 0}


def test_load_done_keeps_only_clean_finished_games(tmp_path):
    path = tmp_path / "out.jsonl"
    records = [
        {"protocol": "sequential", "seed": 1, "returncode": 0, "result": "Byzantines Win"},
        {"protocol": "sequential", "seed": 2, "returncode": 1, "result": None},
        {"protocol": "none", "seed": 1, "returncode": 0, "result": None},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    assert load_done(path) == {("sequential", 1)}
    assert load_done(tmp_path / "missing.jsonl") == set()


def test_wilson_interval():
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.237, abs=0.002) and high == pytest.approx(0.763, abs=0.002)
    low, high = wilson(0, 10)
    assert low == 0.0 and high == pytest.approx(0.278, abs=0.002)
    assert wilson(0, 0) == (0.0, 1.0)


def test_summarize_groups_by_model_and_protocol_and_skips_failures_in_the_rates():
    def game(protocol, winner, seconds, result="done"):
        return {
            "model": "m", "protocol": protocol, "result": result if winner else None, "winner": winner, "rounds_played": 4,
            "meetings": 2, "statements": 20, "wall_s": seconds,
            "stats": {"honest_correct_votes": 3, "honest_incorrect_votes": 1},
        }

    rows = summarize([game("none", "honest", 60), game("sequential", "byzantine", 120), game("sequential", "honest", 180), game("sequential", None, 10)])
    assert [r["protocol"] for r in rows] == ["sequential", "none"]
    sequential = rows[0]
    assert sequential["games"] == 2 and sequential["failed"] == 1
    assert sequential["honest_win_rate"] == 0.5
    assert sequential["mean_minutes"] == pytest.approx(2.5)
    assert sequential["honest_vote_accuracy"] == 0.75
    assert "| m | default | 0 | sequential | 2 | 1 | 0.50 |" in format_table(rows)


def test_read_tokens_takes_the_per_model_totals_from_the_stats_file(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text(
        "agent_name,api_input_tokens,api_output_tokens,api_calls,api_reasoning_tokens,api_reasoning_estimated_calls,"
        "api_truncated_calls,api_empty_replies,api_max_output_tokens\n"
        "Agent_0,5000,900,40,600,40,3,2,160\nAgent_1,5000,900,40,600,40,3,2,160\n",
        encoding="utf-8",
    )
    tokens = read_tokens(path)
    assert tokens["api_output_tokens"] == 900 and tokens["api_reasoning_tokens"] == 600
    assert tokens["api_truncated_calls"] == 3 and tokens["api_max_output_tokens"] == 160
    assert read_tokens(tmp_path / "missing.csv") == {}


def test_summarize_reports_reasoning_share_and_keeps_efforts_apart():
    def game(effort, reasoning):
        return {
            "model": "m", "protocol": "sequential", "result": "done", "winner": "honest", "rounds_played": 3, "meetings": 1,
            "statements": 10, "wall_s": 60, "reasoning_effort": effort, "max_tokens": 768,
            "tokens": {"api_output_tokens": 1000, "api_reasoning_tokens": reasoning, "api_calls": 50,
                       "api_truncated_calls": 5, "api_empty_replies": 2},
        }

    rows = summarize([game("low", 300), game("medium", 800)])
    assert [r["effort"] for r in rows] == ["low", "medium"]
    assert rows[0]["reasoning_share"] == 0.3 and rows[1]["reasoning_share"] == 0.8
    assert rows[0]["truncated_share"] == 0.1 and rows[0]["empty_share"] == 0.04
    assert rows[0]["tokens_per_call"] == 20 and rows[0]["token_limit"] == 768


def make_state(seed):
    agents = [SimpleNamespace(name=f"Agent_{i}", role="honest", color="x", model_name="m", action_num=0) for i in range(10)]
    state = GameState(agents, SimpleNamespace(game_id="g"), seed=seed)
    return {name: data["location"] for name, data in state.world_data["agents"].items()}


def test_a_seed_fixes_the_start_rooms():
    assert make_state(3) == make_state(3)
    assert any(make_state(3) != make_state(seed) for seed in range(4, 12))


def test_games_with_one_seed_start_alike_under_any_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    comp = composition("seedtest", "ibm-granite/granite-3.3-8b-instruct")
    starts = []
    for game_id, mode in (("a", "turns"), ("b", "timed")):
        engine = GameEngine(game_id, num_agents=10, seed=11, discussion_mode=mode)
        engine.setup(comp)
        starts.append({n: d["location"] for n, d in engine.state.world_data["agents"].items()})
    assert starts[0] == starts[1]


def test_read_redone_counts_the_rewound_meetings(tmp_path):
    path = tmp_path / "voided_meetings.jsonl"
    path.write_text('{"round": 1, "attempt": 1}\n{"round": 2, "attempt": 1}\n', encoding="utf-8")
    assert read_redone(path) == 2
    assert read_redone(tmp_path / "missing.jsonl") == 0


def test_summarize_reports_skipped_votes_and_redone_meetings():
    def game(redone, skipped):
        return {
            "model": "m", "protocol": "none", "result": "done", "winner": "honest", "rounds_played": 3, "meetings": 2,
            "statements": 0, "wall_s": 60, "redone_meetings": redone,
            "stats": {"honest_correct_votes": 1, "honest_incorrect_votes": 1, "honest_skipped_votes": skipped},
        }

    (row,) = summarize([game(0, 2), game(2, 0)])
    assert row["honest_skipped_share"] == pytest.approx(2 / 6)
    assert row["redone_per_game"] == 1.0
    table = format_table([row])
    assert "| 33% | 1.00 |" in table
    # Records written before the rule have no count, which the table leaves empty.
    old = {k: v for k, v in game(0, 0).items() if k != "redone_meetings"}
    assert summarize([old])[0]["redone_per_game"] is None


def test_the_word_limit_reaches_the_game_command_only_when_it_is_set():
    protocol = protocol_by_name("clean-60")
    assert "--room_max_words" not in game_command(protocol, 1, "g", "s", 3, 4)
    command = game_command(protocol, 1, "g", "s", 3, 4, room_max_words=90)
    assert command[command.index("--room_max_words") + 1] == "90"


def test_summaries_keep_runs_with_other_settings_apart():
    def game(**settings):
        return {
            "model": "m", "protocol": "none", "result": "done", "winner": "honest", "rounds_played": 3, "meetings": 2,
            "statements": 0, "wall_s": 60, **settings,
        }

    rows = summarize([game(), game(capture=True), game(room_max_words=90), game(room_max_words=40)])
    assert sorted((r["effort"], r["games"]) for r in rows) == [("default", 2), ("default +think", 1), ("default words 90", 1)]


def test_summarize_reports_api_errors_per_game():
    def game(errors):
        return {
            "model": "m", "protocol": "none", "result": "done", "winner": "honest", "rounds_played": 3, "meetings": 2,
            "statements": 0, "wall_s": 60, "tokens": {"api_errors": errors, "api_calls": 10},
        }

    (row,) = summarize([game(4), game(0)])
    assert row["errors_per_game"] == 2.0
    assert "| 2.0 |" in format_table([row])


def test_two_runs_of_one_model_get_different_scenarios_and_config_files():
    model = "ibm-granite/granite-3.3-8b-instruct"
    assert scenario_name(model, "v7-granite") == "p2_ibm-granite_granite-3.3-8b-instruct_v7-granite"
    assert scenario_name(model, "v7-granite") != scenario_name(model, "v7-granite-think")
