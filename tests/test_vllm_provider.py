"""Tests for the vllm provider settings."""

from core.api_clients import get_client


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
