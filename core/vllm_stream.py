"""A session with a model in vLLM that keeps its context between chunks of input.

vLLM can take a request whose prompt arrives in pieces. The first piece starts the request, later pieces are appended
to the same context and its cached keys and values, and the model writes after each piece up to the token limit of that
piece. This module wraps that in the `StreamHandle` of `core/duplex.py`: `send` appends a piece and returns the burst the
model writes after it.

The engine drops the last token it sampled when it appends the next piece, because that token has no cached keys and
values yet. The text of a burst here is decoded from its token ids without that token, so it matches the context.

vLLM is imported when an engine is built, so the rest of the code imports this module without vLLM installed.

Usage:
    engine = VllmStreamEngine(async_llm, tokenizer, temperature=0.7)
    handle = engine.open("Agent_1")
    burst = await handle.send(StreamPrompt("..."), max_tokens=8)
"""

import asyncio
import time
from typing import Any

from loguru import logger

from core.duplex import Burst, StreamPrompt

_CONTEXT_TOKENS = 8


def text_of(tokenizer: Any, context: list[int], ids: list[int]) -> str:
    """Decodes tokens with the spacing they have after the tokens that came before them.

    A tokenizer that marks spaces on the following token drops the space of the first token when it decodes the tokens
    alone, so the text is the difference between the decoding of context and tokens together and of the context.

    Args:
        tokenizer: A tokenizer with `decode`.
        context: Tokens that came before.
        ids: Tokens to decode.

    Returns:
        The text of the tokens. Special tokens give no text.
    """
    if not ids:
        return ""
    whole = tokenizer.decode(context + ids, skip_special_tokens=True)
    before = tokenizer.decode(context, skip_special_tokens=True) if context else ""
    return whole[len(before) :] if whole.startswith(before) else tokenizer.decode(ids, skip_special_tokens=True)


def kept_text(tokenizer: Any, context: list[int], ids: list[int], finish_reason: str | None) -> str:
    """Gives the text of a burst that stays in the context of the session.

    The engine drops the token it sampled last when a burst ends at its token limit, so that token has no text here.
    A burst that stopped on its own ended with a stop token, which has no text either.

    Args:
        tokenizer: A tokenizer with `decode`.
        context: Tokens that came before the burst.
        ids: Tokens of the burst.
        finish_reason: `length` or `stop`.

    Returns:
        The text.
    """
    return text_of(tokenizer, context, ids[:-1] if finish_reason == "length" else ids)


class VllmHandle:
    """One streaming request of vLLM, used as a session."""

    def __init__(self, engine: Any, tokenizer: Any, request_id: str, temperature: float, seed: int | None) -> None:
        """Starts the request. It waits for the first chunk.

        Args:
            engine: A vLLM `AsyncLLM`.
            tokenizer: Tokenizer of the model.
            request_id: Name of the request, which must be unique in the engine.
            temperature: Sampling temperature.
            seed: Seed for sampling, or None.
        """
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        self._engine = engine
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._seed = seed
        self._delta = RequestOutputKind.DELTA
        self._params_class = SamplingParams
        self._request_id = request_id
        self._inputs: asyncio.Queue = asyncio.Queue()
        self._waiting: asyncio.Future | None = None
        self._failure: BaseException | None = None
        self._started = 0.0
        self._context: list[int] = []
        self._prompt_tokens = 0
        self._task = asyncio.create_task(self._drain())

    def _params(self, max_tokens: int) -> Any:
        """Builds the sampling parameters for one chunk."""
        return self._params_class(
            temperature=self._temperature, max_tokens=max_tokens, seed=self._seed, output_kind=self._delta
        )

    async def _chunks(self):
        """Yields the chunks that `send` queues, until `close`."""
        while True:
            item = await self._inputs.get()
            if item is None:
                return
            yield item

    async def _drain(self) -> None:
        """Reads the output of the request and hands each burst to the `send` that is waiting for it."""
        ids: list[int] = []
        try:
            async for out in self._engine.generate(
                self._chunks(), self._params(1), self._request_id, tokenization_kwargs={"add_special_tokens": False}
            ):
                piece = out.outputs[0]
                ids.extend(piece.token_ids)
                if piece.finish_reason is None:
                    continue
                kept = kept_text(self._tokenizer, self._context, ids, piece.finish_reason)
                burst = Burst(
                    kept, piece.finish_reason, self._prompt_tokens, len(ids), time.monotonic() - self._started
                )
                self._context = (self._context + ids[:-1] if piece.finish_reason == "length" else self._context + ids)[
                    -_CONTEXT_TOKENS:
                ]
                ids = []
                if self._waiting is not None and not self._waiting.done():
                    self._waiting.set_result(burst)
        except Exception as error:  # noqa: BLE001 - handed to the caller that waits
            logger.error("session {} failed: {!r}", self._request_id, error)
            self._failure = error
            if self._waiting is not None and not self._waiting.done():
                self._waiting.set_exception(error)

    async def send(self, prompt: StreamPrompt, max_tokens: int) -> Burst:
        """Appends a chunk of input and returns the burst the model writes after it.

        Args:
            prompt: The new input.
            max_tokens: Most tokens the burst may have.

        Returns:
            The burst.

        Raises:
            RuntimeError: If the session failed earlier.
        """
        from vllm.engine.protocol import StreamingInput

        if self._failure is not None:
            raise RuntimeError(f"session {self._request_id} failed earlier") from self._failure
        engine_prompt: dict[str, Any] = {"prompt": prompt.text}
        if prompt.audio:
            engine_prompt["multi_modal_data"] = {"audio": [(a, prompt.sample_rate) for a in prompt.audio]}
        self._prompt_tokens = len(self._tokenizer.encode(prompt.text, add_special_tokens=False))
        self._waiting = asyncio.get_running_loop().create_future()
        self._started = time.monotonic()
        await self._inputs.put(StreamingInput(prompt=engine_prompt, sampling_params=self._params(max_tokens)))
        return await self._waiting

    async def close(self) -> None:
        """Ends the request, which frees its cached keys and values."""
        await self._inputs.put(None)
        try:
            await asyncio.wait_for(self._task, 30.0)
        except asyncio.TimeoutError:
            self._task.cancel()


class VllmStreamEngine:
    """Opens sessions on one vLLM engine."""

    def __init__(self, engine: Any, tokenizer: Any, temperature: float = 0.7, seed: int | None = None) -> None:
        """Creates the opener.

        Args:
            engine: A vLLM `AsyncLLM`.
            tokenizer: Tokenizer of the model.
            temperature: Sampling temperature of every session.
            seed: Seed for sampling, or None.
        """
        self._engine = engine
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._seed = seed
        self._opened = 0

    def open(self, name: str) -> VllmHandle:
        """Opens a session.

        Args:
            name: Agent name, used in the request id.

        Returns:
            The session. It must be used from the running event loop.
        """
        self._opened += 1
        return VllmHandle(self._engine, self._tokenizer, f"{name}-{self._opened}", self._temperature, self._seed)
