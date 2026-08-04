"""Thought-capture composition flags (Piece 3).

Pure helpers only — no LLM / agent / game wiring.
"""


def thought_capture_flags_from_composition(composition):
    """Resolve thought-capture flags from a composition dict.

    Defaults:
      - capture_thoughts: True   (master switch for prompt/parse/log)
      - require_think_tags: False  (strict mode; retry wiring comes later)
    """
    if not isinstance(composition, dict):
        composition = {}
    capture_thoughts = bool(composition.get("capture_thoughts", True))
    require_think_tags = bool(composition.get("require_think_tags", False))
    return capture_thoughts, require_think_tags
