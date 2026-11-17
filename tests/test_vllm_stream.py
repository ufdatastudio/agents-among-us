"""Tests for the vLLM session, with a fake engine in place of vLLM."""

import asyncio
import sys
import types
from dataclasses import dataclass

import pytest

from core.duplex import StreamPrompt
from core.vllm_stream import VllmStreamEngine, kept_text, text_of

PIECES = {1: "▁I", 2: "▁was", 3: "▁in", 4: "▁Shields", 5: "▁with", 0: "<eos>"}


class FakeTokenizer:
    """Decodes like a SentencePiece tokenizer, which drops the space before the first token."""

    def decode(self, ids, skip_special_tokens=True):
        text = "".join(PIECES[i] for i in ids if not (skip_special_tokens and i == 0)).replace("▁", " ")
        return text[1:] if text.startswith(" ") else text

    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split())))


def test_tokens_keep_the_space_they_have_after_the_tokens_before_them():
    tokenizer = FakeTokenizer()
    assert tokenizer.decode([4]) == "Shields"
    assert text_of(tokenizer, [1, 2, 3], [4]) == " Shields"
    assert text_of(tokenizer, [], [1, 2]) == "I was"
    assert text_of(tokenizer, [1], []) == ""


def test_a_burst_cut_by_its_limit_loses_its_last_token_and_a_stopped_one_keeps_all():
    tokenizer = FakeTokenizer()
    assert kept_text(tokenizer, [], [1, 2, 3, 4], "length") == "I was in"
    assert kept_text(tokenizer, [], [1, 2, 3, 0], "stop") == "I was in"
    assert kept_text(tokenizer, [3], [4, 5], "length") == " Shields"
    assert kept_text(tokenizer, [], [], "length") == ""


@dataclass
class Piece:
    token_ids: list
    text: str
    finish_reason: str | None


@dataclass
class Output:
    outputs: list


@dataclass
class StreamingInput:
    prompt: dict
    sampling_params: object = None


class FakeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeEngine:
    """Reads chunks like AsyncLLM.generate and answers each with the next scripted burst."""

    def __init__(self, bursts, fail_at=None):
        self.bursts = list(bursts)
        self.fail_at = fail_at
        self.seen = []
        self.kwargs = None

    async def generate(self, prompt, params, request_id, **kwargs):
        self.kwargs = kwargs
        async for chunk in prompt:
            self.seen.append(chunk)
            if self.fail_at == len(self.seen):
                raise RuntimeError("engine died")
            ids, text, reason = self.bursts.pop(0)
            half = len(ids) // 2
            yield Output([Piece(ids[:half], text[: len(text) // 2], None)])
            yield Output([Piece(ids[half:], text[len(text) // 2 :], reason)])


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = FakeParams
    sampling = types.ModuleType("vllm.sampling_params")
    sampling.RequestOutputKind = types.SimpleNamespace(DELTA="delta")
    engine = types.ModuleType("vllm.engine")
    protocol = types.ModuleType("vllm.engine.protocol")
    protocol.StreamingInput = StreamingInput
    for name, module in (("vllm", vllm), ("vllm.sampling_params", sampling), ("vllm.engine", engine), ("vllm.engine.protocol", protocol)):
        monkeypatch.setitem(sys.modules, name, module)


def test_a_session_returns_one_burst_for_each_chunk_without_the_dropped_token(fake_vllm):
    engine = FakeEngine([([1, 2, 3, 4], "I was in Shields", "length"), ([5, 0], "with", "stop")])

    async def run():
        handle = VllmStreamEngine(engine, FakeTokenizer(), temperature=0.5).open("Agent_1")
        first = await handle.send(StreamPrompt("hello"), 4)
        second = await handle.send(StreamPrompt("again"), 8)
        await handle.close()
        return first, second

    first, second = asyncio.run(run())
    assert (first.text, first.finish_reason, first.output_tokens) == ("I was in", "length", 4)
    assert (second.text, second.finish_reason) == (" with", "stop")
    assert [c.prompt["prompt"] for c in engine.seen] == ["hello", "again"]
    assert [c.sampling_params.max_tokens for c in engine.seen] == [4, 8]
    assert engine.seen[0].sampling_params.temperature == 0.5 and engine.seen[0].sampling_params.output_kind == "delta"
    assert engine.kwargs == {"tokenization_kwargs": {"add_special_tokens": False}}


def test_audio_goes_to_the_engine_as_multimodal_data_with_its_rate(fake_vllm):
    import numpy as np

    engine = FakeEngine([([1], "I", "stop")])

    async def run():
        handle = VllmStreamEngine(engine, FakeTokenizer()).open("Agent_1")
        await handle.send(StreamPrompt("<audio>", (np.zeros(10, dtype=np.float32),), 16000), 4)
        await handle.close()

    asyncio.run(run())
    audio = engine.seen[0].prompt["multi_modal_data"]["audio"]
    assert len(audio) == 1 and audio[0][1] == 16000


def test_a_failure_of_the_engine_reaches_the_caller_and_later_calls_are_refused(fake_vllm):
    engine = FakeEngine([([1], "I", "stop")], fail_at=2)

    async def run():
        handle = VllmStreamEngine(engine, FakeTokenizer()).open("Agent_1")
        await handle.send(StreamPrompt("one"), 4)
        with pytest.raises(RuntimeError, match="engine died"):
            await handle.send(StreamPrompt("two"), 4)
        with pytest.raises(RuntimeError, match="failed earlier"):
            await handle.send(StreamPrompt("three"), 4)
        await handle.close()

    asyncio.run(run())


def test_sessions_get_different_request_ids(fake_vllm):
    engine = FakeEngine([])

    async def run():
        opener = VllmStreamEngine(engine, FakeTokenizer())
        a, b = opener.open("Agent_1"), opener.open("Agent_1")
        names = (a._request_id, b._request_id)
        await a.close()
        await b.close()
        return names

    first, second = asyncio.run(run())
    assert first != second
