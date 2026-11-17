"""Tests for the typed append-only ledger."""

import asyncio
import threading

import pytest

from core.ledger import CHUNK, CLAIM, FACT, NOTE, Ledger


def test_records_get_the_next_id_and_keep_their_order():
    ledger = Ledger()
    first = ledger.append(FACT, "engine", {"room": "Storage"}, subject="Agent_1", round_num=1, tick=1)
    second = ledger.append(CLAIM, "Agent_2", {"kind": "seen"}, subject="Agent_1", round_num=1, time_s=4.0)
    assert (first.id, second.id) == (1, 2)
    assert ledger.get(2) is second
    assert len(ledger) == 2


def test_a_reader_sees_open_records_and_the_ones_that_list_it():
    ledger = Ledger()
    ledger.append(FACT, "engine", {}, visible_to=("Agent_1",))
    ledger.append(CLAIM, "Agent_2", {})
    ledger.append(FACT, "engine", {}, visible_to=("Agent_3", "Agent_4"))
    assert [r.id for r in ledger.view("Agent_1")] == [1, 2]
    assert [r.id for r in ledger.view("Agent_3")] == [2, 3]
    assert [r.id for r in ledger.view("Agent_9")] == [2]


def test_view_can_keep_some_types():
    ledger = Ledger()
    ledger.append(FACT, "engine", {})
    ledger.append(CLAIM, "Agent_2", {})
    assert [r.type for r in ledger.view("Agent_1", types=[CLAIM])] == [CLAIM]


def test_an_unknown_type_and_a_reference_to_a_later_record_are_refused():
    ledger = Ledger()
    with pytest.raises(ValueError):
        ledger.append("rumor", "Agent_1", {})
    with pytest.raises(ValueError):
        ledger.append(NOTE, "engine", {}, refs=[1])
    first = ledger.append(CLAIM, "Agent_1", {})
    assert ledger.append(NOTE, "engine", {}, refs=[first.id]).refs == (1,)


def test_the_ledger_is_written_and_read_back_with_the_same_ids(tmp_path):
    ledger = Ledger()
    ledger.append(FACT, "engine", {"room": "Storage", "beside": ["Agent_2"]}, subject="Agent_1", round_num=1, tick=1, visible_to=("Agent_1",))
    ledger.append(CLAIM, "Agent_2", {"kind": "seen"}, subject="Agent_1", round_num=1, time_s=3.5, refs=[1])
    path = tmp_path / "game" / "ledger.jsonl"
    ledger.write_jsonl(path)
    again = Ledger.read_jsonl(path)
    assert [r.to_dict() for r in again.records] == [r.to_dict() for r in ledger.records]


def test_a_chunk_is_a_record_type():
    ledger = Ledger()
    record = ledger.append(CHUNK, "Agent_1", {"stream": "Agent_1/1"}, time_s=2.0)
    assert (record.type, record.time_s) == (CHUNK, 2.0)


def test_a_subscription_returns_each_record_once_and_only_to_its_reader():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")
    ledger.append(CLAIM, "Agent_2", {})
    ledger.append(FACT, "engine", {}, visible_to=("Agent_3",))
    ledger.append(CLAIM, "Agent_3", {})
    assert [r.id for r in heard.poll()] == [1, 3]
    assert heard.poll() == []
    ledger.append(CLAIM, "Agent_2", {})
    assert [r.id for r in heard.poll()] == [4]


def test_a_subscription_can_follow_some_types_and_start_late():
    ledger = Ledger()
    ledger.append(CLAIM, "Agent_2", {})
    ledger.append(CHUNK, "Agent_2", {})
    ledger.append(CHUNK, "Agent_3", {})
    chunks = ledger.subscribe("Agent_1", types=[CHUNK], after=2)
    assert [r.id for r in chunks.poll()] == [3]


def test_a_record_is_not_heard_before_it_is_fully_spoken_and_holds_back_later_ones():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")
    ledger.append(CHUNK, "Agent_2", {}, time_s=2.0)
    ledger.append(CHUNK, "Agent_3", {}, time_s=4.0)
    ledger.append(CHUNK, "Agent_2", {}, time_s=2.5)
    assert [r.id for r in heard.poll(until_s=2.0)] == [1]
    assert heard.poll(until_s=3.0) == []
    assert [r.id for r in heard.poll(until_s=4.0)] == [2, 3]


def test_a_record_with_no_time_is_heard_at_once():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")
    ledger.append(FACT, "engine", {})
    assert len(heard.poll(until_s=0.0)) == 1


def test_wait_returns_when_a_visible_record_arrives_from_another_thread():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")

    async def run():
        timer = threading.Timer(0.05, lambda: ledger.append(CLAIM, "Agent_2", {}))
        timer.start()
        return await heard.wait(timeout_s=2.0)

    assert asyncio.run(run()) is True
    assert len(heard.poll()) == 1


def test_wait_returns_at_once_for_a_record_already_there_and_times_out_for_none():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")
    assert asyncio.run(heard.wait(timeout_s=0.05)) is False
    ledger.append(CLAIM, "Agent_2", {})
    assert asyncio.run(heard.wait(timeout_s=0.05)) is True


def test_wait_ignores_a_record_the_reader_may_not_read():
    ledger = Ledger()
    heard = ledger.subscribe("Agent_1")

    async def run():
        timer = threading.Timer(0.02, lambda: ledger.append(FACT, "engine", {}, visible_to=("Agent_3",)))
        timer.start()
        return await heard.wait(timeout_s=0.2)

    assert asyncio.run(run()) is False


def test_a_listener_hears_every_append_until_it_is_removed():
    ledger = Ledger()
    seen = []
    remove = ledger.listen(seen.append)
    ledger.append(CLAIM, "Agent_1", {})
    remove()
    ledger.append(CLAIM, "Agent_1", {})
    assert [r.id for r in seen] == [1]
