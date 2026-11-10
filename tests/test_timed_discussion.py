"""Tests for the tick loop, the vote tick, the in-process transport and the scripted adapter."""

import json
import threading
import time

import pytest

from core.adapters.base import AgentAdapter
from core.adapters.scripted import ScriptedAdapter
from core.situation_room import Meeting, Reply, RoomConfig, Tick, Vote
from core.timed_discussion import SKIP, collect_votes, onset_delay, run_timed_discussion
from core.transport import InProcessTransport

NAMES = ["Red", "Blue", "Lime"]
MEETING = Meeting(round_num=2, reason="Red found a body", caller="Red")


def words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))


def cast(scripts: dict[str, dict[int, str]] | None = None, votes: dict[str, str] | None = None) -> list:
    scripts = scripts or {}
    votes = votes or {}
    return [ScriptedAdapter(n, NAMES, scripts.get(n, {}), votes.get(n, "SKIP")) for n in NAMES]


def config(**settings) -> RoomConfig:
    base = {"budget_s": 30.0, "tick_s": 10.0, "quiet_close_ticks": 0, "onset_jitter_s": 2.0}
    return RoomConfig(**{**base, **settings})


def test_scripted_adapter_satisfies_the_protocol():
    assert isinstance(ScriptedAdapter("Red", NAMES), AgentAdapter)


def test_everyone_speaks_in_the_same_tick_and_none_hears_it_yet():
    adapters = cast({n: {1: words(10)} for n in NAMES})
    room = run_timed_discussion(adapters, MEETING, config())
    first = [u for u in room.utterances() if u.tick == 1]
    assert sorted(u.author for u in first) == sorted(NAMES)
    assert all(0.0 <= u.start_s < 2.0 for u in first)
    assert all(a.heard[0].text == "" for a in adapters)
    assert all(a.heard[0].tick == 1 and a.heard[0].t == 0.0 for a in adapters)


def test_later_ticks_hear_what_was_said_earlier():
    adapters = cast({"Red": {1: "I found the body"}})
    run_timed_discussion(adapters, MEETING, config())
    assert "Red: I found the body" in adapters[1].heard[1].text
    assert "Red (you): I found the body" in adapters[0].heard[1].text


def test_busy_agents_are_not_asked_at_a_tick():
    adapters = cast({"Red": {1: words(40)}})
    run_timed_discussion(adapters, MEETING, config())
    assert [p.tick for p in adapters[0].heard] == [1, 3]
    assert [p.tick for p in adapters[1].heard] == [1, 2, 3]


def test_meeting_is_announced_and_ended_once():
    adapters = cast()
    run_timed_discussion(adapters, MEETING, config())
    assert all(a.meetings == [MEETING] and a.ended == 1 for a in adapters)


def test_replies_are_collected_in_parallel():
    barrier = threading.Barrier(3, timeout=5)

    class Waiting(ScriptedAdapter):
        def reply(self, tick: Tick) -> Reply:
            barrier.wait()
            return super().reply(tick)

    adapters = [Waiting(n, NAMES, {1: words(5)}) for n in NAMES]
    room = run_timed_discussion(adapters, MEETING, config(budget_s=10.0))
    assert len(room.utterances()) == 3


def test_a_slow_agent_counts_as_silent_and_the_others_still_speak():
    class Slow(ScriptedAdapter):
        def reply(self, tick: Tick) -> Reply:
            time.sleep(0.5)
            return super().reply(tick)

    adapters = [Slow("Red", NAMES, {1: words(5)}), ScriptedAdapter("Blue", NAMES, {1: words(5)}),
                ScriptedAdapter("Lime", NAMES, {})]
    room = run_timed_discussion(adapters, MEETING, config(budget_s=10.0), reply_timeout_s=0.05)
    assert [u.author for u in room.utterances()] == ["Blue"]


def test_a_failing_agent_counts_as_silent():
    class Broken(ScriptedAdapter):
        def reply(self, tick: Tick) -> Reply:
            raise RuntimeError("model crashed")

    adapters = [Broken("Red", NAMES), ScriptedAdapter("Blue", NAMES, {1: words(5)}), ScriptedAdapter("Lime", NAMES)]
    room = run_timed_discussion(adapters, MEETING, config(budget_s=10.0))
    assert [u.author for u in room.utterances()] == ["Blue"]


def test_the_budget_cuts_an_utterance_in_progress():
    adapters = cast({"Red": {2: words(40)}})
    room = run_timed_discussion(adapters, MEETING, config(budget_s=20.0))
    utterance = room.lanes["Red"].utterances[0]
    assert utterance.intended == 40
    assert len(utterance.words) < 40
    assert utterance.end_s <= 20.0 + 1e-9
    assert room.closed_at == 20.0


def test_quiet_discussion_closes_early():
    adapters = cast({n: {1: words(10)} for n in NAMES})
    room = run_timed_discussion(adapters, MEETING, config(budget_s=180.0, quiet_close_ticks=3))
    assert room.closed_early
    assert room.ticks == 4
    assert room.closed_at == 40.0


def test_the_discussion_stays_open_while_someone_is_talking():
    adapters = cast({"Red": {1: words(40)}})
    room = run_timed_discussion(
        adapters, MEETING, config(budget_s=100.0, quiet_close_ticks=3, words_per_s=0.5)
    )
    assert not room.closed_early
    assert room.ticks == 10


def test_barge_in_lets_a_speaker_cut_itself_off():
    adapters = cast({"Red": {1: words(40), 2: "No, wait."}})
    room = run_timed_discussion(adapters, MEETING, config(barge_in=True))
    first, second = room.lanes["Red"].utterances
    assert len(first.words) < 40
    assert first.end_s <= second.start_s + 1e-9
    assert [p.tick for p in adapters[0].heard] == [1, 2, 3]


def test_the_same_seed_gives_the_same_discussion_and_another_seed_a_different_one():
    scripts = {n: {1: words(12), 2: words(8)} for n in NAMES}
    first = run_timed_discussion(cast(scripts), MEETING, config(seed=4))
    again = run_timed_discussion(cast(scripts), MEETING, config(seed=4))
    other = run_timed_discussion(cast(scripts), MEETING, config(seed=5))
    assert first.ground_truth() == again.ground_truth()
    assert first.percept_log == again.percept_log
    assert [u["start_s"] for u in first.ground_truth()] != [u["start_s"] for u in other.ground_truth()]


def test_onset_delay_is_seeded_and_shorter_than_a_tick():
    settings = config(seed=1, onset_jitter_s=50.0, tick_s=10.0)
    assert onset_delay(settings, "Red", 1) == onset_delay(settings, "Red", 1)
    assert onset_delay(settings, "Red", 1) != onset_delay(settings, "Red", 2)
    assert all(0.0 <= onset_delay(settings, "Red", t) < 10.0 for t in range(1, 50))


def test_addressed_agents_are_recorded():
    adapters = cast({"Red": {1: "<to>Blue</to> Where were you?"}})
    room = run_timed_discussion(adapters, MEETING, config())
    assert room.lanes["Red"].utterances[0].addressed_to == ("Blue",)
    assert room.lanes["Red"].utterances[0].text == "Where were you?"


def test_votes_are_collected_from_the_final_state():
    adapters = cast({"Red": {1: "Blue was alone"}}, votes={"Red": "Blue", "Blue": "Red", "Lime": "Blue"})
    room = run_timed_discussion(adapters, MEETING, config())
    assert collect_votes(adapters, room) == {"Red": "Blue", "Blue": "Red", "Lime": "Blue"}
    final = adapters[1].heard[-1]
    assert final.t == room.closed_at
    assert "Red: Blue was alone" in final.text


def test_invalid_missing_and_self_votes_count_as_skips():
    class NoVote(ScriptedAdapter):
        def vote(self, percept, candidates) -> Vote:
            raise RuntimeError("no answer")

    adapters = [
        ScriptedAdapter("Red", NAMES, vote_for="Red"),
        ScriptedAdapter("Blue", NAMES, vote_for="Nobody"),
        NoVote("Lime", NAMES),
    ]
    room = run_timed_discussion(adapters, MEETING, config(budget_s=10.0))
    assert collect_votes(adapters, room) == {"Red": SKIP, "Blue": SKIP, "Lime": SKIP}


def test_the_record_can_be_written_after_a_full_meeting(tmp_path):
    adapters = cast({n: {1: words(6)} for n in NAMES}, votes={"Red": "Blue"})
    room = run_timed_discussion(adapters, MEETING, config())
    collect_votes(adapters, room)
    path = tmp_path / "room.jsonl"
    room.write_jsonl(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "room" and kinds.count("utterance") == 3
    assert kinds.count("percept") == sum(len(a.heard) for a in adapters)


def test_transport_returns_results_in_order_and_isolates_failures():
    def fail():
        raise RuntimeError("boom")

    results = InProcessTransport().collect({"a": lambda: 1, "b": fail, "c": lambda: 3})
    assert results == {"a": 1, "b": None, "c": 3}
    assert list(results) == ["a", "b", "c"]
    assert InProcessTransport().collect({}) == {}


def test_transport_gives_up_on_a_slow_job():
    results = InProcessTransport().collect({"fast": lambda: "ok", "slow": lambda: time.sleep(0.5)}, timeout_s=0.05)
    assert results["fast"] == "ok"
    assert results["slow"] is None


@pytest.mark.parametrize("budget_s, tick_s, expected_ticks", [(30.0, 10.0, 3), (25.0, 10.0, 3), (10.0, 10.0, 1)])
def test_number_of_ticks_follows_the_budget(budget_s, tick_s, expected_ticks):
    room = run_timed_discussion(cast(), MEETING, config(budget_s=budget_s, tick_s=tick_s))
    assert room.ticks == expected_ticks
    assert room.closed_at == min(expected_ticks * tick_s, budget_s)
