"""Tests for the sessions, the turn format and the beat runner."""

import asyncio

import pytest

from core.duplex import Burst, DuplexAgent, TurnFormat, run_duplex_meeting
from core.ledger import Ledger
from core.stream import stream_texts


def chatml(messages: list[dict], add_generation_prompt: bool) -> str:
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")


FORMAT = TurnFormat.probe(chatml)


def make_agent(name, ledger, handle, beat_s=2.0, max_tokens=8, **kwargs):
    return DuplexAgent(
        name, handle, FORMAT, f"You are {name}.", "A meeting has begun.", ledger.subscribe(name), beat_s, max_tokens, **kwargs
    )


def run(agents, ledger, budget_s=6.0, beat_s=2.0):
    return asyncio.run(run_duplex_meeting(agents, ledger, budget_s, beat_s))


def test_the_format_is_cut_out_of_the_template():
    assert FORMAT.gap == "<|im_end|>\n<|im_start|>user\n"
    assert FORMAT.cue == "<|im_end|>\n<|im_start|>assistant\n"
    assert FORMAT.opening("S", "U") == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    assert FORMAT.next_turn("hello") == "<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"


def test_a_template_that_drops_a_message_is_refused():
    with pytest.raises(ValueError):
        TurnFormat.probe(lambda messages, add: "".join(m["content"] for m in messages if m["role"] != "assistant"))


def test_the_first_prompt_has_the_system_prompt_and_the_next_ones_only_what_is_new(scripted):
    ledger = Ledger()
    a_handle = scripted(["hello there"])
    agents = [make_agent("Agent_1", ledger, a_handle), make_agent("Agent_2", ledger, scripted([]))]
    run(agents, ledger, budget_s=6.0)
    first, second, third = (p.text for p in a_handle.prompts)
    assert first.startswith("<|im_start|>system\nYou are Agent_1.") and "A meeting has begun." in first
    assert first.endswith("[00:00] ...<|im_end|>\n<|im_start|>assistant\n")
    assert second == "<|im_end|>\n<|im_start|>user\n[00:02] ...<|im_end|>\n<|im_start|>assistant\n"
    assert third.startswith("<|im_end|>\n<|im_start|>user\n[00:04]")


def test_agents_read_one_snapshot_so_two_who_speak_together_do_not_hear_each_other_until_the_next_beat(scripted):
    ledger = Ledger()
    one, two = scripted(["from one"]), scripted(["from two"])
    run([make_agent("Agent_1", ledger, one), make_agent("Agent_2", ledger, two)], ledger, budget_s=4.0)
    assert "from" not in one.prompts[0].text and "from" not in two.prompts[0].text
    assert one.prompts[1].text.count("[00:00] Agent_2: from two") == 1
    assert two.prompts[1].text.count("[00:00] Agent_1: from one") == 1
    assert "Agent_1: from one" not in one.prompts[1].text


def test_a_burst_that_hits_its_limit_goes_on_in_the_same_statement_until_the_model_stops(scripted):
    ledger = Ledger()
    handle = scripted([Burst("I was in", "length"), Burst("Shields with", "length"), Burst("Agent 2", "stop")])
    run([make_agent("Agent_1", ledger, handle)], ledger, budget_s=8.0)
    assert stream_texts(ledger) == {"Agent_1/1": "I was in Shields with Agent 2"}
    assert [r.content["final"] for r in ledger.records] == [False, False, False, True]


def test_a_burst_that_continues_a_statement_gets_a_space_unless_it_starts_with_one_or_a_closing_mark(scripted):
    ledger = Ledger()
    handle = scripted([Burst("Agent_0", "length"), Burst("was there", "length"), Burst("!", "length"), Burst(" and left", "stop")])
    run([make_agent("Agent_1", ledger, handle)], ledger, budget_s=8.0)
    assert [r.content["data"] for r in ledger.records if not r.content["final"]] == ["Agent_0", " was there", "!", " and left"]
    assert stream_texts(ledger) == {"Agent_1/1": "Agent_0 was there! and left"}


def test_a_new_statement_starts_without_a_space(scripted):
    ledger = Ledger()
    handle = scripted([Burst("one", "stop"), Burst("two", "stop")])
    run([make_agent("Agent_1", ledger, handle)], ledger, budget_s=4.0)
    assert [r.content["data"] for r in ledger.records if not r.content["final"]] == ["one", "two"]


def test_silence_ends_a_statement_and_the_next_words_start_another(scripted):
    ledger = Ledger()
    handle = scripted([Burst("one", "length"), "<silent/>", Burst("two", "length")])
    meeting = run([make_agent("Agent_1", ledger, handle)], ledger, budget_s=8.0)
    assert stream_texts(ledger) == {"Agent_1/1": "one", "Agent_1/2": "two"}
    stats = meeting.stats["Agent_1"]
    assert (stats.statements, stats.spoke, stats.silent, stats.words, stats.beats) == (2, 2, 2, 2, 4)
    assert ledger.records[-1].content["final"] is True


def test_a_statement_still_open_when_the_meeting_ends_is_closed_at_the_budget(scripted):
    ledger = Ledger()
    run([make_agent("Agent_1", ledger, scripted([Burst("on and", "length")] * 3))], ledger, budget_s=6.0)
    assert ledger.records[-1].content["final"] is True
    assert ledger.records[-1].time_s == 6.0


def test_the_limit_of_a_burst_and_the_beat_come_from_the_agent(scripted):
    ledger = Ledger()
    handle = scripted([])
    meeting = run([make_agent("Agent_1", ledger, handle, beat_s=2.0, max_tokens=5)], ledger, budget_s=4.0)
    assert handle.limits == [5, 5]
    assert meeting.beats == 2


def test_every_session_is_closed_even_when_one_fails(scripted):
    class Broken:
        closed = False

        async def send(self, prompt, max_tokens):
            raise RuntimeError("engine died")

        async def close(self):
            self.closed = True

    ledger = Ledger()
    good, bad = scripted([]), Broken()
    with pytest.raises(RuntimeError):
        run([make_agent("Agent_1", ledger, good), make_agent("Agent_2", ledger, bad)], ledger)
    assert good.closed and bad.closed


def test_statistics_add_up_what_the_engine_reports(scripted):
    ledger = Ledger()
    handle = scripted([Burst("hi", "stop", prompt_tokens=30, output_tokens=2, wall_s=0.5), Burst("<silent/>", "stop", 10, 4, 0.25)])
    meeting = run([make_agent("Agent_1", ledger, handle)], ledger, budget_s=4.0)
    stats = meeting.stats["Agent_1"]
    assert (stats.prompt_tokens, stats.output_tokens, stats.engine_s) == (40, 6, 0.75)


def test_an_agent_with_an_audio_model_sends_the_arrays_it_heard(scripted, tmp_path):
    import numpy as np

    from core.audio.mix import SAMPLE_RATE, write_wav
    from core.stream import write_audio_stream

    path = tmp_path / "clip.wav"
    write_wav(path, np.zeros(SAMPLE_RATE, dtype=np.float32) + 0.1)
    ledger = Ledger()
    write_audio_stream(ledger, "Agent_2", path, 0.0, 1.0, "hello")
    handle = scripted([])
    run([make_agent("Agent_1", ledger, handle, audio_placeholder="<audio>")], ledger, budget_s=4.0, beat_s=2.0)
    assert handle.prompts[0].audio == ()
    assert "[00:00] Agent_2: <audio>" in handle.prompts[1].text
    assert len(handle.prompts[1].audio) == 1
    assert handle.prompts[1].sample_rate == SAMPLE_RATE
