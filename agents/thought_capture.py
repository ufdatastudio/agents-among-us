"""Shared helpers for optional private thought capture on LLM turns."""

from __future__ import annotations

from agents.response_parse import split_think_and_output

# Increased due to agent thoughts implementation (original value = 160).
MAX_TOKENS_WITH_THOUGHTS = 384
MAX_TOKENS_DEFAULT = 160

THINK_TAG_RETRY_REMINDER = (
    "\n\nYou forgot the <think>...</think> tags. Respond again: put private "
    "reasoning inside <think>...</think>, then output only the final action "
    "after the closing tag."
)

# Appended when capture_thoughts is on but a custom prompt override is used.
THINK_FORMAT_APPENDIX = """

THINKING FORMAT:
1. First write private reasoning inside <think>...</think>.
2. After the closing </think> tag, output ONLY the public answer.
3. Do NOT put the final answer inside the think block.
"""


def capture_enabled(agent, world_view=None) -> bool:
    world_view = world_view or {}
    if "capture_thoughts" in world_view:
        return bool(world_view["capture_thoughts"])
    return bool(getattr(agent, "capture_thoughts", False))


def require_tags_enabled(agent, world_view=None) -> bool:
    world_view = world_view or {}
    if "require_think_tags" in world_view:
        return bool(world_view["require_think_tags"])
    return bool(getattr(agent, "require_think_tags", False))


def with_think_format_if_capturing(body: str, agent, world_view=None) -> str:
    """Append the shared think-format appendix for custom prompt overrides."""
    if capture_enabled(agent, world_view):
        return f"{body.rstrip()}{THINK_FORMAT_APPENDIX}"
    return body


def tokens_for_turn(agent, world_view=None) -> int:
    if capture_enabled(agent, world_view):
        # Increased due to agent thoughts implementation (original value = 160).
        return MAX_TOKENS_WITH_THOUGHTS
    return MAX_TOKENS_DEFAULT


def generate_with_optional_thoughts(
    agent,
    system_prompt,
    prompt,
    *,
    temperature,
    phase,
    round_num,
    tick=0,
    world_view=None,
):
    """Generate a response, optionally parsing/logging private <think> content.

    Returns:
        public_output (str): text for action matching / public logging.
    """
    world_view = world_view or {}
    max_tokens = tokens_for_turn(agent, world_view)
    raw = agent.llm.generate(
        agent.model_name,
        system_prompt,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not capture_enabled(agent, world_view):
        return raw.strip() if isinstance(raw, str) else raw

    think, public, meta = split_think_and_output(raw)

    if require_tags_enabled(agent, world_view) and not meta.get("had_tags"):
        retry_prompt = prompt + THINK_TAG_RETRY_REMINDER
        raw = agent.llm.generate(
            agent.model_name,
            system_prompt,
            retry_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        think, public, meta = split_think_and_output(raw)

    logger = getattr(agent, "logger", None)
    if logger is not None:
        logger.log_thought(
            round=round_num,
            phase=phase,
            tick=tick,
            agent=agent.name,
            role=getattr(agent, "role", ""),
            model=getattr(agent, "model_name", ""),
            think=think,
            output=public,
            had_tags=meta.get("had_tags", False),
            parse_ok=meta.get("parse_ok", False),
        )

    return public
