"""API client wrappers for Navigator, OpenAI, and Anthropic providers."""

import time
from dataclasses import dataclass

from loguru import logger


@dataclass
class APIResponse:
    """Response from an API provider."""

    text: str
    input_tokens: int
    output_tokens: int


class OpenAICompatibleClient:
    """Client for OpenAI-compatible APIs (Navigator, OpenAI)."""

    def __init__(self, base_url, api_key, provider_name):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.provider_name = provider_name

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
                    max_tokens=max_tokens,
                    timeout=60,
                )
                text = response.choices[0].message.content or ""
                usage = response.usage
                return APIResponse(
                    text=text.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
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
        provider: One of 'navigator', 'openai', 'anthropic'.
        api_keys: Dict mapping provider names to API key strings.

    Returns:
        An API client instance.

    Raises:
        ValueError: If the required API key is missing.
    """
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
