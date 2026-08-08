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
# Unclosed open tag through end-of-string (defense for public channels).
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think_markup(text: str) -> str:
    """Remove all <think> markup from text destined for public channels.

    Strips well-formed blocks, unclosed open-tag tails, and stray closers.
    Never raises.
    """
    if not text:
        return ""
    out = _THINK_BLOCK_RE.sub("", text)
    out = _UNCLOSED_THINK_RE.sub("", out)
    out = _CLOSE_THINK_RE.sub("", out)
    return out.strip()


def split_think_and_output(raw: str) -> tuple[str, str, dict[str, Any]]:
    """Split a raw model response into private thinking and public output.

    Returns:
        (think, public_output, meta) where meta includes:
          - had_tags (bool): an opening <think> tag was present
          - parse_ok (bool): structure was usable (well-formed or clean no-tag fallback)
          - rescued_from_think (bool): public answer was pulled out of the think block
            because nothing followed the closing tag

    Never raises on malformed input. Public output is always sanitized so residual
    think tags cannot leak into discussion / action / vote channels.
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
        # Unclosed <think>: keep content after the open tag private; do not
        # expose the reasoning blob as public_output.
        open_m = _OPEN_THINK_RE.search(text)
        if open_m and not _CLOSE_THINK_RE.search(text):
            meta["had_tags"] = True
            meta["parse_ok"] = False
            before = text[: open_m.start()].strip()
            think = text[open_m.end() :].strip()
            public = strip_think_markup(before)
            return think, public, meta

        # No tags at all: entire string is the public output.
        meta["had_tags"] = False
        meta["parse_ok"] = True
        return "", strip_think_markup(text), meta

    # Honor only the first think block; everything after its close is public
    # (then scrub any further think markup from that public slice).
    meta["had_tags"] = True
    think = match.group(1).strip()
    public = strip_think_markup(text[match.end() :])

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
            return think_body, strip_think_markup(public_line), meta

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
