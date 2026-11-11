"""Tests for the situation room's data structure and renderers."""

import json
import re

import pytest

from core.situation_room import (
    INAUDIBLE,
    Room,
    RoomConfig,
    format_time,
    parse_reply,
    render,
)

ROSTER = ["Red", "Blue", "Lime"]


def words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def make_room(**settings) -> Room:
    return Room(ROSTER, RoomConfig(**settings))


def test_config_rejects_bad_settings():
    with pytest.raises(ValueError):
        RoomConfig(overlap="loud")
    with pytest.raises(ValueError):
        RoomConfig(tag_policy="sometimes")
    with pytest.raises(ValueError):
        RoomConfig(budget_s=0)
    with pytest.raises(ValueError):
        RoomConfig(dropout=1.5)
    with pytest.raises(ValueError):
        RoomConfig(quiet_close_ticks=-1)


def test_format_time():
    assert format_time(0) == "00:00"
    assert format_time(2.4) == "00:02"
    assert format_time(75.9) == "01:15"


def test_words_are_spoken_at_an_even_pace():
    room = make_room()
    utterance = room.append("Red", words(40), start_s=0.0, tick=1)
    assert utterance.pace_s == pytest.approx(0.4)
    assert utterance.end_s == pytest.approx(16.0)
    assert utterance.heard_count(0.0) == 0
    assert utterance.heard_count(10.0) == 25
    assert utterance.heard_count(16.0) == 40
    assert utterance.heard_count(99.0) == 40


def test_room_keeps_only_max_words():
    room = make_room(max_words=40)
    utterance = room.append("Red", words(50), start_s=0.0, tick=1)
    assert len(utterance.words) == 40
    assert utterance.intended == 50


def test_known_duration_sets_the_pace():
    room = make_room()
    utterance = room.append("Red", words(10), start_s=1.0, tick=1, duration_s=5.0)
    assert utterance.pace_s == pytest.approx(0.5)
    assert utterance.end_s == pytest.approx(6.0)


def test_empty_utterance_is_rejected():
    with pytest.raises(ValueError):
        make_room().append("Red", "   ", start_s=0.0, tick=1)


def test_speaking_again_while_busy_needs_barge_in():
    room = make_room()
    room.append("Red", words(10), start_s=0.0, tick=1)
    with pytest.raises(ValueError):
        room.append("Red", words(3), start_s=2.0, tick=2)


def test_barge_in_cuts_the_earlier_utterance():
    room = make_room(barge_in=True)
    room.append("Red", words(10), start_s=0.0, tick=1)
    room.append("Red", words(3), start_s=2.0, tick=2)
    first, second = room.lanes["Red"].utterances
    assert len(first.words) == 5
    assert first.end_s == pytest.approx(2.0)
    assert second.start_s == 2.0
    assert room.free_at("Red") == pytest.approx(3.2)


def test_free_at_follows_the_last_utterance():
    room = make_room()
    assert room.free_at("Blue") == 0.0
    room.append("Blue", words(10), start_s=3.0, tick=1)
    assert room.free_at("Blue") == pytest.approx(7.0)


def test_close_cuts_unfinished_and_late_utterances():
    room = make_room()
    room.append("Red", words(40), start_s=0.0, tick=1)
    room.append("Blue", words(5), start_s=12.0, tick=2)
    room.close(10.0)
    assert len(room.lanes["Red"].utterances[0].words) == 25
    assert room.lanes["Blue"].utterances[0].words == ()
    assert room.lanes["Blue"].utterances[0].intended == 5
    assert room.closed_at == 10.0


def test_clean_render_shows_finished_and_unfinished_utterances():
    room = make_room()
    room.append("Red", "I found the body in Electrical", start_s=0.0, tick=1)
    room.append("Blue", words(30), start_s=1.0, tick=1)
    assert render(room, "Lime", 0.0) == ""
    lines = render(room, "Lime", 5.0).split("\n")
    assert lines == [
        "[00:00-00:02] Red: I found the body in Electrical",
        "[00:01-...  ] Blue: w0 w1 w2 w3 w4 w5 w6 w7 w8 w9 ...",
    ]


def test_clean_render_marks_the_readers_own_words():
    room = make_room()
    room.append("Red", "I was in Admin", start_s=0.0, tick=1)
    assert "Red (you): I was in Admin" in render(room, "Red", 5.0)
    assert "Red: I was in Admin" in render(room, "Blue", 5.0)


def test_interleaved_render_puts_simultaneous_speakers_in_the_same_window():
    room = make_room(overlap="interleaved")
    room.append("Red", words(10, "r"), start_s=0.0, tick=1)
    room.append("Blue", words(10, "b"), start_s=0.0, tick=1)
    assert render(room, "Lime", 1.9) == ""
    assert set(render(room, "Lime", 2.0).split("\n")) == {
        "[00:00] Red: r0 r1 r2 r3 r4",
        "[00:00] Blue: b0 b1 b2 b3 b4",
    }
    later = render(room, "Lime", 4.0).split("\n")
    assert len(later) == 4
    assert {line for line in later[2:]} == {"[00:02] Red: r5 r6 r7 r8 r9", "[00:02] Blue: b5 b6 b7 b8 b9"}


def test_interleaved_render_only_grows():
    room = make_room(overlap="interleaved", seed=5)
    room.append("Red", words(30, "r"), start_s=0.0, tick=1)
    room.append("Blue", words(30, "b"), start_s=0.7, tick=1)
    room.append("Lime", words(30, "l"), start_s=1.3, tick=1)
    earlier = render(room, "Red", 4.0)
    assert earlier
    assert render(room, "Red", 8.0).startswith(earlier + "\n")
    assert render(room, "Red", 12.0).startswith(render(room, "Red", 8.0) + "\n")


def test_interleaved_order_is_repeatable_and_varies_with_seed_and_reader():
    def order(seed: int, reader: str) -> list[str]:
        room = Room([f"A{i}" for i in range(8)] + ["Z"], RoomConfig(overlap="interleaved", seed=seed))
        for i in range(8):
            room.append(f"A{i}", words(5), start_s=0.0, tick=1)
        return [line.split()[1].rstrip(":") for line in render(room, reader, 2.0).split("\n")]

    assert order(1, "Z") == order(1, "Z")
    assert len({tuple(order(seed, "Z")) for seed in range(10)}) > 1
    assert len({tuple(order(1, reader)) for reader in ["Z", "A0", "A1", "A2", "A3"]}) > 1


def test_noisy_render_without_dropout_matches_interleaved_with_tags():
    quiet = make_room(overlap="noisy", dropout=0.0)
    plain = make_room(overlap="interleaved")
    for room in (quiet, plain):
        room.append("Red", words(10, "r"), start_s=0.0, tick=1)
    assert render(quiet, "Blue", 4.0) == render(plain, "Blue", 4.0)


def test_full_dropout_hides_other_speakers_but_not_the_reader():
    room = make_room(overlap="noisy", dropout=1.0)
    room.append("Red", words(10, "r"), start_s=0.0, tick=1)
    assert render(room, "Blue", 4.0).split("\n") == [f"[00:00] Red: {INAUDIBLE}", f"[00:02] Red: {INAUDIBLE}"]
    assert render(room, "Red", 4.0).split("\n")[0] == "[00:00] Red (you): r0 r1 r2 r3 r4"


def test_first_chunk_policy_tags_only_the_start_of_an_utterance():
    room = make_room(overlap="noisy", tag_policy="first_chunk")
    room.append("Red", words(10, "r"), start_s=0.0, tick=1)
    assert render(room, "Blue", 4.0).split("\n") == ["[00:00] Red: r0 r1 r2 r3 r4", "[00:02] r5 r6 r7 r8 r9"]


def test_dropout_hides_some_words_and_keeps_them_hidden_later():
    room = make_room(overlap="noisy", dropout=0.5, seed=3)
    room.append("Red", words(40, "r"), start_s=0.0, tick=1)
    earlier = render(room, "Blue", 4.0)
    assert INAUDIBLE in earlier
    assert re.search(r"\br\d+\b", earlier)
    assert render(room, "Blue", 8.0).startswith(earlier + "\n")
    assert render(room, "Blue", 4.0) == earlier
    assert render(room, "Lime", 4.0) != earlier


def test_percept_is_logged_and_carries_the_time():
    room = make_room()
    room.append("Red", "hello there", start_s=0.0, tick=1)
    percept = room.percept("Blue", 5.0, tick=2)
    assert (percept.reader, percept.tick, percept.cursor, percept.modality) == ("Blue", 2, 5.0, "text")
    assert room.percept_log == [{"reader": "Blue", "tick": 2, "t": 5.0, "text": percept.text}]


def test_parse_reply_reads_silence_and_address_tags():
    assert parse_reply("Red", 1, None, ROSTER).text is None
    assert parse_reply("Red", 1, "<silent/>", ROSTER).text is None
    assert parse_reply("Red", 1, "I will pass <SILENT />", ROSTER).text is None
    assert parse_reply("Red", 1, "   ", ROSTER).text is None
    reply = parse_reply("Red", 3, "<to>blue</to> You were in  Admin.", ROSTER)
    assert reply.text == "Blue, You were in Admin."
    assert reply.addressed_to == ("Blue",)
    assert reply.tick == 3
    many = parse_reply("Red", 1, "<to>Blue, Lime</to><to>Nobody</to> Explain.", ROSTER)
    assert many.addressed_to == ("Blue", "Lime")
    assert many.text == "Blue, Lime, Nobody, Explain."


def test_parse_reply_keeps_the_names_a_model_wraps_in_tags():
    leading = parse_reply("Red", 1, "<to>Blue</to>, I saw you in Admin.", ROSTER)
    assert leading.text == "Blue, I saw you in Admin."
    assert leading.addressed_to == ("Blue",)
    inline = parse_reply("Red", 1, "I trust <to>Lime</to>. Let us check with <to>Blue</to> in Shields.", ROSTER)
    assert inline.text == "I trust Lime. Let us check with Blue in Shields."
    assert inline.addressed_to == ("Lime", "Blue")
    assert parse_reply("Red", 1, "<to>Blue</to>", ROSTER).text is None
    assert parse_reply("Red", 1, "<to>Blue</to>", ROSTER).addressed_to == ("Blue",)


def test_statements_follow_the_observer_format():
    room = make_room()
    room.append("Red", "I found a body", start_s=0.0, tick=1)
    room.append("Blue", "I was in Admin", start_s=1.0, tick=1)
    room.append("Red", "Blue is lying", start_s=10.0, tick=2)
    assert room.statements(caller="Red") == [
        {"Agent": "Red", "Text": "I found a body", "Reported": 1, "S_Num": 1},
        {"Agent": "Blue", "Text": "I was in Admin", "Reported": 0, "S_Num": 1},
        {"Agent": "Red", "Text": "Blue is lying", "Reported": 0, "S_Num": 2},
    ]


def test_jsonl_has_the_room_the_utterances_and_the_percepts(tmp_path):
    room = make_room(seed=9)
    room.append("Red", "hello there", start_s=0.5, tick=1, addressed_to=("Blue",))
    room.percept("Blue", 5.0, tick=2)
    room.close(5.0)
    path = tmp_path / "meeting" / "room.jsonl"
    room.write_jsonl(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in rows] == ["room", "utterance", "percept"]
    assert rows[0]["agents"] == ROSTER and rows[0]["seed"] == 9 and rows[0]["closed_at"] == 5.0
    assert rows[1]["author"] == "Red" and rows[1]["addressed_to"] == ["Blue"] and rows[1]["end_s"] == 1.3
    assert rows[2]["reader"] == "Blue"
