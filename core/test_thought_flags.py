"""Unit tests for thought-capture composition flag defaults (Piece 3)."""

import unittest

from core.thought_flags import thought_capture_flags_from_composition


class TestThoughtCaptureFlags(unittest.TestCase):
    def test_defaults_when_keys_missing(self):
        capture, require = thought_capture_flags_from_composition({})
        self.assertTrue(capture)
        self.assertTrue(require)

    def test_defaults_when_composition_not_a_dict(self):
        capture, require = thought_capture_flags_from_composition(None)
        self.assertTrue(capture)
        self.assertTrue(require)

    def test_require_follows_capture_on(self):
        capture, require = thought_capture_flags_from_composition(
            {"capture_thoughts": True, "require_think_tags": False}
        )
        self.assertTrue(capture)
        self.assertTrue(require)

    def test_require_off_when_capture_off(self):
        capture, require = thought_capture_flags_from_composition(
            {"capture_thoughts": False, "require_think_tags": True}
        )
        self.assertFalse(capture)
        self.assertFalse(require)


if __name__ == "__main__":
    unittest.main()
