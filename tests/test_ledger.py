"""Tests for the typed append-only ledger."""

import pytest

from core.ledger import CLAIM, FACT, NOTE, Ledger


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
