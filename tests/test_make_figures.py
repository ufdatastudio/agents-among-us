"""Tests for the numbers and figures of the doc."""

import json
from pathlib import Path

import pytest

from scripts.make_figures import (
    FIGURES,
    collect_doc,
    collect_p2,
    numbers,
    parse_table,
    plot_all,
    setting_of,
    short_model,
    tint,
)

DOC = Path("docs/situation-room.md")
DATA = Path("docs/figures/data.json")


def test_a_table_is_read_without_its_header_and_separator():
    text = "intro\n| A | B |\n|---|---|\n| 1 | 2 s |\n| 3 | 4 s |\n\nafter\n"
    assert parse_table(text, "| A | B") == [["1", "2 s"], ["3", "4 s"]]


def test_numbers_reads_ranges_and_thousands():
    assert numbers("1,217 to 2,171") == [1217.0, 2171.0]
    assert numbers("11.3 s") == [11.3]
    assert numbers("96%") == [96.0]


def test_the_doc_tables_give_the_numbers_of_the_measurements():
    data = collect_doc(DOC)
    granite = data["concurrency_l4"]["Granite 3.3 8B"]
    assert granite["at once"][10] == 6.9 and granite["one after another"] == 14.2
    assert sorted(data["concurrency_rtx"]["Gemma 4 31B"]["at once"]) == [1, 2, 5, 10, 20]
    assert len(data["reasoning"]) == 6
    high = next(r for r in data["reasoning"] if r["model"] == "gpt-oss-120b" and r["effort"] == "high")
    assert high["mean"] == [3979.0, 4096.0] and high["meeting_s"] == 343.0
    assert [r["speakers"] for r in data["audio"]] == [1, 2, 3, 5, 10]
    assert data["audio"][-1]["recall"] == 12.0


def test_settings_and_model_names():
    assert [setting_of(p) for p in ("none", "sequential", "clean-180", "noisy-60")] == ["no discussion", "turns", "timed room", "timed room"]
    assert short_model("vllm:ibm-granite/granite-3.3-8b-instruct") == "Granite 3.3 8B"
    assert short_model("openai/gpt-oss-120b") == "gpt-oss-120b"


def test_collect_p2_counts_wins_and_votes_by_setting_and_protocol(tmp_path):
    def game(protocol, winner, correct, skipped=0):
        return {
            "model": "ibm-granite/granite-3.3-8b-instruct", "protocol": protocol, "result": "done", "winner": winner,
            "stats": {"honest_correct_votes": correct, "honest_incorrect_votes": 1, "honest_skipped_votes": skipped},
        }

    failed = {"model": "ibm-granite/granite-3.3-8b-instruct", "protocol": "none", "result": None, "winner": None}
    path = tmp_path / "p2.jsonl"
    path.write_text("\n".join(json.dumps(g) for g in (game("clean-60", "honest", 2), game("clean-180", "byzantine", 1, 2), game("none", "honest", 3), failed)), encoding="utf-8")
    cells = collect_p2([path])["Granite 3.3 8B"]
    timed = cells["settings"]["timed room"]
    assert (timed["games"], timed["wins"], timed["correct"], timed["incorrect"], timed["skipped"]) == (2, 1, 3, 2, 2)
    assert 0 < timed["ci_low"] < 0.5 < timed["ci_high"] < 1
    assert cells["settings"]["no discussion"]["games"] == 1
    assert set(cells["protocols"]) == {"clean-60", "clean-180", "none"}


def test_tint_mixes_a_color_with_the_surface():
    assert tint("#2a78d6", 1.0) == "#2a78d6"
    assert tint("#2a78d6", 0.0) == "#fcfcfb"


def test_every_figure_is_drawn_from_the_committed_data(tmp_path):
    pytest.importorskip("matplotlib")
    files = plot_all(json.loads(DATA.read_text(encoding="utf-8")), tmp_path)
    assert len(files) == len(FIGURES)
    assert all(f.stat().st_size > 10_000 for f in files)
