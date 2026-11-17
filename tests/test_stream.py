"""Tests for the chunks of a stream and the append-only rendering."""

import numpy as np
import pytest

from core.audio.mix import SAMPLE_RATE, write_wav
from core.ledger import CLAIM, Ledger
from core.stream import (
    AUDIO,
    Chunk,
    StreamWriter,
    chunk_of,
    read_audio,
    render_heard,
    stream_texts,
    write_audio_stream,
    write_replay,
)


def chunks_of(ledger: Ledger) -> list[Chunk]:
    return [chunk_of(r) for r in ledger.records]


def test_a_writer_appends_numbered_chunks_that_end_when_they_are_audible():
    ledger = Ledger()
    writer = StreamWriter(ledger, "Agent_3", round_num=2)
    writer.write("I was in", 12.0, 14.0)
    writer.write(" Shields", 14.0, 16.0)
    writer.close(16.0)
    first, second, last = chunks_of(ledger)
    assert (first.stream, first.seq, first.start_s, first.end_s, first.words) == ("Agent_3/1", 0, 12.0, 14.0, "I was in")
    assert (second.seq, second.words) == (1, " Shields")
    assert (last.final, last.words, last.end_s) == (True, "", 16.0)
    assert ledger.get(1).time_s == 14.0 and ledger.get(1).round_num == 2


def test_each_stream_of_an_author_has_its_own_name():
    ledger = Ledger()
    StreamWriter(ledger, "Agent_3").write("one", 0, 1, final=True)
    second = StreamWriter(ledger, "Agent_3")
    second.write("two", 2, 3)
    other = StreamWriter(ledger, "Agent_4")
    other.write("three", 2, 3)
    assert (second.name, other.name) == ("Agent_3/2", "Agent_4/1")


def test_a_closed_stream_and_a_backwards_chunk_are_refused():
    ledger = Ledger()
    writer = StreamWriter(ledger, "Agent_3")
    with pytest.raises(ValueError):
        writer.write("late", 5.0, 4.0)
    writer.close(1.0)
    with pytest.raises(ValueError):
        writer.write("more", 1.0, 2.0)
    with pytest.raises(ValueError):
        StreamWriter(ledger, "Agent_3", "smell")


def test_a_chunk_can_be_limited_to_some_listeners():
    ledger = Ledger()
    StreamWriter(ledger, "Agent_3", visible_to=["Agent_1"]).write("psst", 0, 1)
    assert len(ledger.view("Agent_1")) == 1
    assert ledger.view("Agent_2") == []


def test_only_chunks_can_be_read_as_chunks():
    ledger = Ledger()
    record = ledger.append(CLAIM, "Agent_1", {})
    with pytest.raises(ValueError):
        chunk_of(record)


def test_chunks_of_one_author_in_a_batch_join_into_a_line_and_split_words_stay_whole():
    ledger = Ledger()
    a = StreamWriter(ledger, "Agent_3")
    b = StreamWriter(ledger, "Agent_4")
    a.write("I was in Shi", 12.0, 14.0)
    b.write("No way", 12.0, 14.0)
    a.write("elds with", 14.0, 16.0)
    heard = render_heard(chunks_of(ledger), "Agent_1")
    assert heard.text == "[00:12] Agent_3: I was in Shields with\n[00:12] Agent_4: No way"


def test_a_listener_does_not_hear_itself_and_a_final_marker_adds_nothing():
    ledger = Ledger()
    me = StreamWriter(ledger, "Agent_1")
    me.write("my words", 0, 2)
    other = StreamWriter(ledger, "Agent_2")
    other.write("hello", 0, 2)
    other.close(2.0)
    assert render_heard(chunks_of(ledger), "Agent_1").text == "[00:00] Agent_2: hello"
    assert render_heard([], "Agent_1").text == ""


def test_the_rendering_of_a_later_batch_does_not_change_the_lines_of_an_earlier_one():
    ledger = Ledger()
    writer = StreamWriter(ledger, "Agent_2")
    writer.write("first part", 0, 2)
    early = render_heard(chunks_of(ledger), "Agent_1").text
    writer.write(" second part", 2, 4)
    both = render_heard(chunks_of(ledger)[1:], "Agent_1").text
    assert early == "[00:00] Agent_2: first part"
    assert both == "[00:02] Agent_2: second part"


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, np.linspace(-0.5, 0.5, round(2.5 * SAMPLE_RATE), dtype=np.float32))
    return path


def test_a_clip_is_cut_into_audio_chunks_with_the_words_shared_out(clip):
    ledger = Ledger()
    writer = write_audio_stream(ledger, "Agent_2", clip, 10.0, 1.0, "one two three four five six")
    chunks = chunks_of(ledger)
    assert writer.closed
    assert [c.modality for c in chunks[:-1]] == [AUDIO] * 3
    assert [(c.start_s, round(c.end_s, 3)) for c in chunks[:3]] == [(10.0, 11.0), (11.0, 12.0), (12.0, 12.5)]
    assert chunks[-1].final and chunks[-1].end_s == 12.5
    assert stream_texts(ledger) == {"Agent_2/1": "one two three four five six"}


def test_a_model_that_cannot_hear_reads_the_words_of_audio_chunks(clip):
    ledger = Ledger()
    write_audio_stream(ledger, "Agent_2", clip, 10.0, 1.0, "one two three")
    heard = render_heard(chunks_of(ledger), "Agent_1")
    assert heard.text == "[00:10] Agent_2: one two three"
    assert heard.audio == ()


def test_a_model_that_hears_gets_a_placeholder_and_the_samples_for_each_audio_chunk(clip):
    ledger = Ledger()
    write_audio_stream(ledger, "Agent_2", clip, 10.0, 1.0, "one two three")
    heard = render_heard(chunks_of(ledger), "Agent_1", audio_placeholder="<audio>")
    assert heard.text == "[00:10] Agent_2: <audio> <audio> <audio>"
    assert heard.sample_rate == SAMPLE_RATE
    assert [len(a) for a in heard.audio] == [SAMPLE_RATE, SAMPLE_RATE, SAMPLE_RATE // 2]
    assert np.allclose(np.concatenate(heard.audio), read_audio({**chunks_of(ledger)[0].data, "start": 0, "count": round(2.5 * SAMPLE_RATE)}))


def test_audio_of_two_speakers_is_ordered_like_the_lines(clip):
    ledger = Ledger()
    write_audio_stream(ledger, "Agent_2", clip, 10.0, 1.0)
    write_audio_stream(ledger, "Agent_3", clip, 10.0, 2.5)
    heard = render_heard(chunks_of(ledger), "Agent_1", audio_placeholder="<a>")
    assert heard.text.splitlines() == ["[00:10] Agent_2: <a> <a> <a>", "[00:10] Agent_3: <a>"]
    assert len(heard.audio) == 4


def test_audio_with_a_different_rate_from_its_chunk_is_refused(clip):
    ledger = Ledger()
    write_audio_stream(ledger, "Agent_2", clip, 0.0)
    data = chunks_of(ledger)[0].data
    with pytest.raises(ValueError):
        read_audio({**data, "rate": 8000})


def test_a_replay_appends_overlapping_speakers_in_the_order_they_finish_each_chunk(tmp_path):
    long = tmp_path / "long.wav"
    short = tmp_path / "short.wav"
    write_wav(long, np.full(round(3.0 * SAMPLE_RATE), 0.1, dtype=np.float32))
    write_wav(short, np.full(round(1.0 * SAMPLE_RATE), 0.2, dtype=np.float32))
    ledger = Ledger()
    writers = write_replay(ledger, [("Agent_2", long, 0.0, "a b c"), ("Agent_3", short, 1.0, "d")], chunk_s=1.0)
    ends = [r.time_s for r in ledger.records]
    assert ends == sorted(ends)
    assert [w.closed for w in writers] == [True, True]
    heard = ledger.subscribe("Agent_1")
    first = heard.poll(until_s=1.0)
    assert [(r.author, r.time_s) for r in first] == [("Agent_2", 1.0)]
    second = heard.poll(until_s=2.0)
    assert sorted((r.author, r.time_s) for r in second) == [("Agent_2", 2.0), ("Agent_3", 2.0), ("Agent_3", 2.0)]
    assert stream_texts(ledger) == {"Agent_2/1": "a b c", "Agent_3/1": "d"}
