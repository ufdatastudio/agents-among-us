"""Shared helpers for optional private thought capture on LLM turns."""

from __future__ import annotations

from agents.response_parse import split_think_and_output, strip_think_markup

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
    # Human players never run through the LLM / thought-capture path.
    if getattr(agent, "is_human", False):
        return ""

    world_view = world_view or {}
    capturing = capture_enabled(agent, world_view)
    max_tokens = tokens_for_turn(agent, world_view)
    # Preserve tags only when we will parse them; otherwise let LLM postprocess
    # strip, then scrub again so tagged text never becomes a public action/line.
    raw = agent.llm.generate(
        agent.model_name,
        system_prompt,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        preserve_think_tags=capturing,
    )

    if not capturing:
        text = raw.strip() if isinstance(raw, str) else str(raw or "")
        return strip_think_markup(text)

    think, public, meta = split_think_and_output(raw)

    if require_tags_enabled(agent, world_view) and not meta.get("had_tags"):
        retry_prompt = prompt + THINK_TAG_RETRY_REMINDER
        raw = agent.llm.generate(
            agent.model_name,
            system_prompt,
            retry_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            preserve_think_tags=True,
        )
        think, public, meta = split_think_and_output(raw)

    # Defense-in-depth: public channels must never carry residual think markup.
    public = strip_think_markup(public)

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

    # Publish thought snapshot immediately so the live UI shows private
    # reasoning before the matching public discussion line / vote is logged.
    game_state = getattr(agent, "game_state", None)
    if game_state is not None and hasattr(game_state, "publish_latest_thought"):
        game_state.publish_latest_thought(
            {
                "round": round_num,
                "phase": phase,
                "tick": tick,
                "agent": agent.name,
                "role": getattr(agent, "role", ""),
                "model": getattr(agent, "model_name", ""),
                "think": think,
                "output": public,
                "had_tags": meta.get("had_tags", False),
                "parse_ok": meta.get("parse_ok", False),
            }
        )
        game_state.save_json()

    return public
