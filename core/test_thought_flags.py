"""Unit tests for thought-capture composition flag defaults (Piece 3)."""

import unittest

from core.thought_flags import thought_capture_flags_from_composition


class TestThoughtCaptureFlags(unittest.TestCase):
    def test_defaults_when_keys_missing(self):
        capture, require = thought_capture_flags_from_composition({})
        self.assertTrue(capture)
        self.assertFalse(require)

    def test_defaults_when_composition_not_a_dict(self):
        capture, require = thought_capture_flags_from_composition(None)
        self.assertTrue(capture)
        self.assertFalse(require)

    def test_explicit_overrides(self):
        capture, require = thought_capture_flags_from_composition(
            {"capture_thoughts": False, "require_think_tags": True}
        )
        self.assertFalse(capture)
        self.assertTrue(require)

    def test_partial_override_keeps_other_default(self):
        capture, require = thought_capture_flags_from_composition(
            {"require_think_tags": True}
        )
        self.assertTrue(capture)
        self.assertTrue(require)


if __name__ == "__main__":
    unittest.main()
