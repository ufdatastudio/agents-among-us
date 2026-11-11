"""Tests for the vllm provider settings."""

import pytest

from core.api_clients import APIResponse, get_client
from core.llm import ModelManager


def test_the_vllm_provider_needs_no_key_and_uses_the_server_url(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://127.0.0.1:23456/v1")
    monkeypatch.delenv("VLLM_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("VLLM_TIMEOUT_S", raising=False)
    client = get_client("vllm", {})
    assert str(client.client.base_url).startswith("http://127.0.0.1:23456/v1")
    assert client.extra_body is None
    assert client.timeout_s == 300.0


def test_reasoning_effort_and_timeout_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VLLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("VLLM_TIMEOUT_S", "45")
    client = get_client("vllm", {})
    assert client.extra_body == {"reasoning_effort": "low"}
    assert client.timeout_s == 45.0


def test_hosted_providers_keep_their_sixty_second_timeout():
    client = get_client("navigator", {"NAVIGATOR_TOOLKIT_API_KEY": "test-key"})
    assert client.timeout_s == 60
    assert client.extra_body is None


class FixedClient:
    """A model client that always gives one reply."""

    def __init__(self, text):
        self.text = text

    def generate(self, model_id, system_prompt, user_prompt, temperature, max_tokens=160):
        return APIResponse(self.text, 10, 5)


@pytest.mark.parametrize(
    "provider, reply, expected",
    [
        ("vllm", 'Agent_2 said "I was in Storage" but the logs say Navigation.', 'Agent_2 said "I was in Storage" but the logs say Navigation.'),
        ("vllm", 'I vote for Agent_8, the one who called it a "hard push".', 'I vote for Agent_8, the one who called it a "hard push".'),
        ("vllm", '"Agent_8"', "Agent_8"),
        ("vllm", '  "I was in Storage, and Agent_2 said "hi" to me."  ', 'I was in Storage, and Agent_2 said "hi" to me.'),
        ("navigator", 'Agent_2 said "I was in Storage" but the logs say Navigation.', "I was in Storage"),
    ],
)
def test_only_the_vllm_provider_keeps_the_quotes_inside_a_reply(provider, reply, expected):
    manager = ModelManager()
    manager.api_clients[provider] = FixedClient(reply)
    assert manager._generate_api(f"{provider}:m", "system", "user", 0.1) == expected
