"""Tests for the synthetic prompts the vLLM concurrency benchmark sends."""

from scripts.bench_vllm_concurrency import board_lines, prompt_head, user_prompt


def test_head_only_grows_from_tick_to_tick():
    for tick in range(1, 6):
        earlier = prompt_head(3, tick, 10, 0)
        later = prompt_head(3, tick + 1, 10, 0)
        assert later.startswith(earlier)
        assert len(later) > len(earlier)


def test_prompts_repeat_for_the_same_inputs():
    assert user_prompt(2, 4, 10, 7) == user_prompt(2, 4, 10, 7)


def test_agents_get_different_private_memory():
    assert prompt_head(0, 1, 10, 0) != prompt_head(1, 1, 10, 0)


def test_salt_changes_the_prompt():
    assert prompt_head(0, 3, 10, 0) != prompt_head(0, 3, 10, 1)


def test_each_tick_adds_one_line_per_speaker():
    assert len(board_lines(2, 10, 0)) == 10
