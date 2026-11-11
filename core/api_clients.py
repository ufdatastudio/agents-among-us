"""API client wrappers for Navigator, OpenAI, Anthropic and local vLLM providers."""

import os
import time
from dataclasses import dataclass

from loguru import logger


@dataclass
class APIResponse:
    """Response from an API provider.

    Attributes:
        text: The reply text.
        input_tokens: Prompt tokens.
        output_tokens: Completion tokens, including any reasoning.
        reasoning_tokens: Completion tokens spent on reasoning before the reply.
        reasoning_estimated: True when `reasoning_tokens` was estimated and not reported by the server.
        truncated: True when the reply stopped at the token limit.
        empty: True when the reply text is empty, which happens when reasoning uses the whole budget.
    """

    text: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0
    reasoning_estimated: bool = False
    truncated: bool = False
    empty: bool = False


def reasoning_tokens_of(response, text: str, output_tokens: int) -> tuple[int, bool]:
    """Counts the completion tokens a reasoning model spent on reasoning in one reply.

    A server that reports `completion_tokens_details.reasoning_tokens` is trusted. Otherwise the
    count is estimated from the reasoning text that vLLM's parser separates from the reply, by
    splitting the completion tokens in proportion to the characters of reasoning and reply.

    Args:
        response: A chat completion from an OpenAI-compatible server.
        text: The reply text, without reasoning.
        output_tokens: Completion tokens of the whole reply.

    Returns:
        The number of reasoning tokens, and whether it is an estimate.
    """
    details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
    reported = getattr(details, "reasoning_tokens", None)
    if reported:
        return int(reported), False
    message = response.choices[0].message
    extra = getattr(message, "model_extra", None) or {}
    reasoning = ""
    for name in ("reasoning_content", "reasoning"):
        reasoning = getattr(message, name, None) or extra.get(name) or ""
        if reasoning:
            break
    if not reasoning:
        return 0, False
    return round(output_tokens * len(reasoning) / (len(reasoning) + len(text))), True


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (Navigator, OpenAI)."""

    def __init__(self, base_url, api_key, provider_name, extra_body=None, timeout_s=60):
        """Creates the client.

        Args:
            base_url: Root of the OpenAI-compatible API.
            api_key: Key for the API.
            provider_name: Name used in log messages.
            extra_body: Extra JSON fields for every request, such as `reasoning_effort` for gpt-oss.
            timeout_s: Seconds to wait for one reply.
        """
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.provider_name = provider_name
        self.extra_body = extra_body
        self.timeout_s = timeout_s

    def generate(self, model_id, system_prompt, user_prompt, temperature, max_tokens=160):
        """Generate a response using the OpenAI-compatible API.

        Args:
            model_id: The model identifier (without provider prefix).
            system_prompt: System prompt for the model.
            user_prompt: User prompt for the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            APIResponse with generated text and token usage.
        """
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        last_error = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    # Increased due to agent thoughts implementation when caller requests it
                    # (original value = 160). Default remains 160 when capture_thoughts is off.
                    max_tokens=max_tokens,
                    timeout=self.timeout_s,
                    **({"extra_body": self.extra_body} if self.extra_body else {}),
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                output_tokens = usage.completion_tokens if usage else 0
                reasoning_tokens, estimated = reasoning_tokens_of(response, text, output_tokens)
                if not text.strip():
                    choice = response.choices[0]
                    logger.warning(
                        "[{}] empty reply: finish_reason={}, completion_tokens={}, prompt_tokens={}, refusal={!r}",
                        self.provider_name, choice.finish_reason, output_tokens,
                        usage.prompt_tokens if usage else 0, getattr(choice.message, "refusal", None),
                    )
                return APIResponse(
                    text=text.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    reasoning_estimated=estimated,
                    truncated=response.choices[0].finish_reason == "length",
                    empty=not text.strip(),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    "[{}] Attempt {}/3 failed: {}. Retrying in {}s...",
                    self.provider_name, attempt + 1, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[{self.provider_name}] All 3 attempts failed. Last error: {last_error}"
        )


class AnthropicClient:
    """Client for the Anthropic API."""

    def __init__(self, api_key):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, model_id, system_prompt, user_prompt, temperature, max_tokens=160):
        """Generate a response using the Anthropic API.

        Args:
            model_id: The model identifier (without provider prefix).
            system_prompt: System prompt for the model.
            user_prompt: User prompt for the model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            APIResponse with generated text and token usage.
        """
        import anthropic

        last_error = None
        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=model_id,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature,
                    # Increased due to agent thoughts implementation when caller requests it
                    # (original value = 160). Default remains 160 when capture_thoughts is off.
                    max_tokens=max_tokens,
                    timeout=60,
                )
                text = response.content[0].text if response.content else ""
                return APIResponse(
                    text=text.strip(),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )
            except (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
            ) as e:
                last_error = e
                wait = 2**attempt
                logger.warning(
                    "[Anthropic] Attempt {}/3 failed: {}. Retrying in {}s...",
                    attempt + 1, e, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[Anthropic] All 3 attempts failed. Last error: {last_error}"
        )


def get_client(provider, api_keys):
    """Factory to get an API client for a provider.

    Args:
        provider: One of 'navigator', 'openai', 'anthropic', 'vllm'.
        api_keys: Dict mapping provider names to API key strings.

    Returns:
        An API client instance. The 'vllm' provider talks to a vLLM server on the same node
        at VLLM_BASE_URL (default http://127.0.0.1:8000/v1), which needs no key. Set
        VLLM_REASONING_EFFORT (low, medium or high) to pass `reasoning_effort` to a gpt-oss
        server, and VLLM_TIMEOUT_S to change the 300 s wait for a reply.

    Raises:
        ValueError: If the required API key is missing.
    """
    if provider == "vllm":
        effort = os.environ.get("VLLM_REASONING_EFFORT")
        return OpenAICompatibleClient(
            base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key="EMPTY",
            provider_name="vLLM",
            extra_body={"reasoning_effort": effort} if effort else None,
            timeout_s=float(os.environ.get("VLLM_TIMEOUT_S", "300")),
        )

    key_map = {
        "navigator": "NAVIGATOR_TOOLKIT_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_key = key_map.get(provider)
    if not env_key:
        raise ValueError(f"Unknown API provider: {provider}")

    api_key = api_keys.get(env_key, "")
    if not api_key:
        raise ValueError(
            f"API key '{env_key}' is required for provider '{provider}' but not set."
        )

    if provider == "navigator":
        return OpenAICompatibleClient(
            base_url="https://api.ai.it.ufl.edu/v1",
            api_key=api_key,
            provider_name="Navigator",
        )
    elif provider == "openai":
        return OpenAICompatibleClient(
            base_url="https://api.openai.com/v1",
            api_key=api_key,
            provider_name="OpenAI",
        )
    elif provider == "anthropic":
        return AnthropicClient(api_key=api_key)

    raise ValueError(f"Unknown API provider: {provider}")
