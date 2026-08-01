"""Split LLM responses into private <think> content and public output.

Piece 1 of thought-capture: pure utility, no agent/game wiring.
"""

from __future__ import annotations

import re
from typing import Any

# First <think>...</think> block only. DOTALL so thoughts may span lines.
_THINK_BLOCK_RE = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)
_OPEN_THINK_RE = re.compile(r"<think>", re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"</think>", re.IGNORECASE)


def split_think_and_output(raw: str) -> tuple[str, str, dict[str, Any]]:
    """Split a raw model response into private thinking and public output.

    Returns:
        (think, public_output, meta) where meta includes:
          - had_tags (bool): an opening <think> tag was present
          - parse_ok (bool): structure was usable (well-formed or clean no-tag fallback)
          - rescued_from_think (bool): public answer was pulled out of the think block
            because nothing followed the closing tag

    Never raises on malformed input.
    """
    meta: dict[str, Any] = {
        "had_tags": False,
        "parse_ok": False,
        "rescued_from_think": False,
    }

    text = _coerce_to_str(raw)
    if not text.strip():
        # Empty input: nothing to parse; treat as a clean empty public output.
        meta["parse_ok"] = True
        return "", "", meta

    match = _THINK_BLOCK_RE.search(text)
    if match is None:
        # Unclosed <think> with no closing tag: flag and keep full text as public
        # so downstream action matching still has something to work with.
        if _OPEN_THINK_RE.search(text) and not _CLOSE_THINK_RE.search(text):
            meta["had_tags"] = True
            meta["parse_ok"] = False
            return "", text.strip(), meta

        # No tags at all: entire string is the public output.
        meta["had_tags"] = False
        meta["parse_ok"] = True
        return "", text.strip(), meta

    # Honor only the first think block; everything after its close is public.
    meta["had_tags"] = True
    think = match.group(1).strip()
    public = text[match.end() :].strip()

    if public:
        meta["parse_ok"] = True
        return think, public, meta

    # Closing tag present but no answer after it — try to rescue the last
    # non-empty line of the think block as the public output.
    if think:
        rescued = _rescue_public_from_think(think)
        if rescued is not None:
            think_body, public_line = rescued
            meta["parse_ok"] = True
            meta["rescued_from_think"] = True
            return think_body, public_line, meta

    meta["parse_ok"] = False
    return think, "", meta


def _coerce_to_str(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    try:
        return str(raw)
    except Exception:
        return ""


def _rescue_public_from_think(think: str) -> tuple[str, str] | None:
    """If the answer was left inside <think>, take the last non-empty line."""
    lines = [line.strip() for line in think.splitlines() if line.strip()]
    if not lines:
        return None
    if len(lines) == 1:
        # Single-line think with no trailing public — cannot separate reasoning
        # from answer reliably; treat the whole line as public and leave think empty.
        return "", lines[0]
    public_line = lines[-1]
    think_body = "\n".join(lines[:-1]).strip()
    return think_body, public_line
