"""Tests for the page that replays a duplex meeting."""

import asyncio
import json

import pytest

from core.duplex import Burst, DuplexAgent, run_duplex_meeting
from core.ledger import Ledger
from core.stream import StreamWriter
from scripts.duplex_spike import RecordingHandle, turn_format, write_results
from scripts.duplex_viewer import PLACEHOLDER, TEMPLATE, build_page, load_run, run_metrics
from tests.test_duplex_spike import FakeTokenizer


@pytest.fixture
def run_folder(tmp_path, scripted):
    ledger = Ledger()
    fmt = turn_format(FakeTokenizer(), {"enable_thinking": False})
    handles = {
        "Agent_1": RecordingHandle(scripted([Burst("I was in", "length", 40, 4, 0.4), Burst("Storage", "stop", 20, 2, 0.3)]), "Agent_1"),
        "Agent_2": RecordingHandle(scripted(["<silent/>", Burst("Liar </script>", "stop", 25, 3, 0.35)]), "Agent_2"),
    }
    agents = [DuplexAgent(n, h, fmt, "sys", "intro", ledger.subscribe(n), 2.0, 8) for n, h in handles.items()]
    meeting = asyncio.run(run_duplex_meeting(agents, ledger, 6.0, 2.0))
    write_results(tmp_path / "gemma-x", ledger, meeting, handles, "google/gemma-4-31B-it", {"beat": 2.0, "budget": 6.0, "agents": 2}, {"Agent_1": "honest"})
    return tmp_path / "gemma-x"


def test_a_run_folder_becomes_chunks_and_session_rows(run_folder):
    run = load_run(run_folder)
    assert (run["name"], run["model"], run["beat"], run["budget"], run["mode"]) == ("gemma-x", "google/gemma-4-31B-it", 2.0, 6.0, "live")
    assert run["agents"] == ["Agent_1", "Agent_2"] and run["live"] == ["Agent_1", "Agent_2"]
    first = [c for c in run["chunks"] if c["agent"] == "Agent_1" and c["text"]]
    assert [c["text"] for c in first] == ["I was in", " Storage"]
    assert (first[0]["start"], first[0]["end"], first[0]["stream"]) == (0.0, 2.0, "Agent_1/1")
    assert [row["finish"] for row in run["sessions"]["Agent_1"]] == ["length", "stop", "stop"]
    assert run["roles"] == {"Agent_1": "honest"}
    assert run["total"]["words"] == 6


def test_a_folder_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path)


def test_the_page_has_the_data_and_no_way_to_close_the_script_early(run_folder):
    page = build_page([load_run(run_folder)])
    assert page.startswith("<!doctype html>") and PLACEHOLDER not in page
    start = page.index("const RUNS = ") + len("const RUNS = ")
    data = page[start : page.index(";\nconst $", start)]
    assert "</script>" not in data
    runs = json.loads(data.replace("<\\/", "</"))
    assert runs[0]["name"] == "gemma-x"
    assert any("Liar </script>" in c["text"] for c in runs[0]["chunks"])


def test_a_fragment_for_the_artifact_tool_has_no_document_tags(run_folder):
    page = build_page([load_run(run_folder)], standalone=False)
    assert "<html" not in page and "<title>Duplex Replay</title>" in page


def test_a_template_without_a_place_for_the_data_is_refused(tmp_path):
    bad = tmp_path / "t.html"
    bad.write_text("<p>nothing</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        build_page([], template=bad)
    assert PLACEHOLDER in TEMPLATE.read_text(encoding="utf-8")


def test_a_replay_run_lists_recorded_speakers_with_the_live_listener(tmp_path):
    ledger = Ledger()
    StreamWriter(ledger, "Agent_0").write("hello", 0.0, 1.0)
    ledger.write_jsonl(tmp_path / "ledger.jsonl")
    (tmp_path / "stats.json").write_text(json.dumps({"model": "m", "args": {"beat": 2.0, "budget": 4.0, "replay": "x", "audio": True}}), encoding="utf-8")
    (tmp_path / "session_Agent_3.jsonl").write_text(json.dumps({"sent": "s", "burst": "b", "finish": "stop"}) + "\n", encoding="utf-8")
    run = load_run(tmp_path)
    assert (run["mode"], run["audio"], run["live"], run["agents"]) == ("replay", True, ["Agent_3"], ["Agent_0", "Agent_3"])


def test_metrics_describe_the_channel_of_a_run(run_folder):
    metrics = load_run(run_folder)["metrics"]
    assert metrics["beats"] == 3
    assert metrics["speech_beats"] == round(3 / 6, 4)
    assert metrics["cut_bursts"] == round(1 / 6, 4)
    assert metrics["statements"] == 2
    assert metrics["words_per_statement"] == 3.0
    assert metrics["repeated_statements"] == 0.0
    assert metrics["voices_at_once"] == 1.0
    assert metrics["overlap_beats"] == round(1 / 3, 4)
    assert metrics["tokens_sent"] == 85
    assert metrics["engine_s_per_beat"] == 0.175


def test_a_statement_that_repeats_earlier_words_counts_as_a_repeat_whatever_the_case_and_punctuation():
    chunk = lambda stream, text, start: {"stream": stream, "agent": "Agent_1", "text": text, "start": start, "end": start + 2.0}
    run = {
        "sessions": {}, "beat": 2.0, "budget": 8.0,
        "chunks": [chunk("a", "Vote 0!", 0.0), chunk("b", "vote 0", 2.0), chunk("c", "Vote 0. Now.", 4.0), chunk("d", "Vote 0!", 6.0)],
    }
    metrics = run_metrics(run)
    assert metrics["statements"] == 4 and metrics["repeated_statements"] == 0.5
    assert metrics["speech_beats"] is None and metrics["engine_s_per_beat"] is None
