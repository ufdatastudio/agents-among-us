"""Thought-capture composition flags (Piece 3).

Pure helpers only — no LLM / agent / game wiring.
"""


def thought_capture_flags_from_composition(composition):
    """Resolve thought-capture flags from a composition dict.

    Defaults:
      - capture_thoughts: True   (master switch for prompt/parse/log)
      - require_think_tags: mirrors capture_thoughts (retry once on missing tags)
    """
    if not isinstance(composition, dict):
        composition = {}
    capture_thoughts = bool(composition.get("capture_thoughts", True))
    # Strict tag retry is always on whenever capture is on (no separate toggle).
    require_think_tags = capture_thoughts
    return capture_thoughts, require_think_tags
