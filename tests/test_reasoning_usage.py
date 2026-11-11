"""Tests for how reasoning tokens, cut-off replies and empty replies are counted."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from core.api_clients import APIResponse, OpenAICompatibleClient, reasoning_tokens_of
from core.llm import ModelManager
from scripts.bench_vllm_concurrency import percentile, summarize


def completion(text="", reasoning=None, reported=None, completion_tokens=200, finish="stop", extra_name="reasoning_content"):
    """Builds a chat completion shaped like the OpenAI client's."""
    message = SimpleNamespace(content=text, model_extra={})
    if reasoning is not None:
        if extra_name == "model_extra":
            message.model_extra = {"reasoning_content": reasoning}
        else:
            setattr(message, extra_name, reasoning)
    details = SimpleNamespace(reasoning_tokens=reported) if reported is not None else None
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=completion_tokens, completion_tokens_details=details)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)], usage=usage)


def test_a_reported_reasoning_count_is_trusted():
    assert reasoning_tokens_of(completion("ok", reasoning="x" * 50, reported=137), "ok", 200) == (137, False)


def test_without_a_report_the_count_is_estimated_from_the_reasoning_text():
    response = completion("a" * 100, reasoning="b" * 300)
    assert reasoning_tokens_of(response, "a" * 100, 200) == (150, True)


@pytest.mark.parametrize("name", ["reasoning_content", "reasoning", "model_extra"])
def test_the_reasoning_text_is_found_under_the_names_servers_use(name):
    response = completion("a" * 100, reasoning="b" * 100, extra_name=name)
    assert reasoning_tokens_of(response, "a" * 100, 100) == (50, True)


def test_a_reply_with_no_reasoning_counts_zero():
    assert reasoning_tokens_of(completion("a short reply"), "a short reply", 20) == (0, False)


def test_all_reasoning_and_no_reply_counts_every_token():
    response = completion("", reasoning="b" * 400, finish="length", completion_tokens=1024)
    assert reasoning_tokens_of(response, "", 1024) == (1024, True)


def make_client(response):
    client = OpenAICompatibleClient("http://127.0.0.1:1/v1", "EMPTY", "vLLM")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    return client


def test_the_client_flags_a_reply_cut_at_the_limit_and_an_empty_reply():
    cut = make_client(completion("", reasoning="b" * 400, finish="length", completion_tokens=1024)).generate("m", "s", "u", 1.0, 1024)
    assert cut.truncated and cut.empty
    assert cut.reasoning_tokens == 1024 and cut.reasoning_estimated and cut.output_tokens == 1024
    fine = make_client(completion("I was in Cafeteria.", completion_tokens=6)).generate("m", "s", "u", 1.0, 160)
    assert not fine.truncated and not fine.empty and fine.reasoning_tokens == 0 and fine.text == "I was in Cafeteria."


def test_the_totals_of_a_model_count_every_reply_from_parallel_threads():
    manager = ModelManager()
    reply = APIResponse("ok", 100, 50, reasoning_tokens=30, reasoning_estimated=True, truncated=False, empty=False)
    cut = APIResponse("", 100, 80, reasoning_tokens=80, truncated=True, empty=True)

    def record(index):
        manager._record_usage("vllm:m", cut if index % 4 == 0 else reply)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(record, range(400)))
    usage = manager.get_token_usage()["vllm:m"]
    assert usage["calls"] == 400
    assert usage["truncated_calls"] == usage["empty_replies"] == 100
    assert usage["reasoning_tokens"] == 100 * 80 + 300 * 30
    assert usage["output_tokens"] == 100 * 80 + 300 * 50
    assert usage["reasoning_estimated_calls"] == 300
    assert usage["max_output_tokens"] == 80


def test_a_response_without_reasoning_fields_still_works():
    response = APIResponse("ok", 10, 5)
    assert (response.reasoning_tokens, response.truncated, response.empty) == (0, False, False)


def test_percentile_uses_the_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 0.5) == 5
    assert percentile(values, 0.9) == 9
    assert percentile(values, 1.0) == 10
    assert percentile([7], 0.9) == 7


def test_the_benchmark_summary_reports_reasoning_share_and_cut_replies():
    results = [
        {"latency_s": 1.0, "prompt_tokens": 100, "completion_tokens": 100, "cached_tokens": 0, "empty": True,
         "truncated": True, "reasoning_tokens": 100, "reasoning_estimated": True},
        {"latency_s": 1.0, "prompt_tokens": 100, "completion_tokens": 100, "cached_tokens": 0, "empty": False,
         "truncated": False, "reasoning_tokens": 60, "reasoning_estimated": True},
    ]
    summary = summarize(2.0, results)
    assert summary["reasoning_fraction"] == 0.8
    assert summary["truncated_fraction"] == 0.5 and summary["empty_fraction"] == 0.5
    assert summary["completion_tokens_p50"] == 100 and summary["completion_tokens_p90"] == 100


def test_a_failed_call_is_counted_and_reads_as_move():
    class Failing:
        def generate(self, *args, **kwargs):
            raise RuntimeError("Error code: 400 - This model's maximum context length is 16384 tokens")

    manager = ModelManager()
    manager.api_clients["vllm"] = Failing()
    assert manager._generate_api("vllm:m", "system", "user", 0.1) == "move"
    assert manager._generate_api("vllm:m", "system", "user", 0.1) == "move"
    assert manager.get_token_usage()["vllm:m"]["api_errors"] == 2


def test_an_empty_reply_is_logged_with_its_finish_reason():
    from loguru import logger

    messages = []
    handler = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        make_client(completion("", completion_tokens=0, finish="stop")).generate("m", "s", "u", 0.1, 160)
    finally:
        logger.remove(handler)
    assert any("empty reply" in m and "finish_reason=stop" in m and "completion_tokens=0" in m for m in messages)
