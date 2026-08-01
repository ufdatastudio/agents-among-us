"""Unit tests for split_think_and_output (Piece 1 — no game integration)."""

import unittest

from agents.response_parse import split_think_and_output


class TestSplitThinkAndOutput(unittest.TestCase):
    def test_well_formed_think_then_answer(self):
        raw = (
            "<think>\n"
            "Medbay looked empty; Cafeteria is safer.\n"
            "</think>\n"
            "Cafeteria"
        )
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "Medbay looked empty; Cafeteria is safer.")
        self.assertEqual(public, "Cafeteria")
        self.assertTrue(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])
        self.assertFalse(meta["rescued_from_think"])

    def test_missing_tags_entirely(self):
        raw = "Cafeteria"
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "")
        self.assertEqual(public, "Cafeteria")
        self.assertFalse(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])
        self.assertFalse(meta["rescued_from_think"])

    def test_plain_string_no_tags(self):
        raw = "Agent_3 is suspicious based on the logs."
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "")
        self.assertEqual(public, raw)
        self.assertFalse(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])

    def test_answer_left_inside_think_rescued(self):
        raw = (
            "<think>\n"
            "I should avoid Medbay after the body report.\n"
            "Cafeteria\n"
            "</think>"
        )
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "I should avoid Medbay after the body report.")
        self.assertEqual(public, "Cafeteria")
        self.assertTrue(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])
        self.assertTrue(meta["rescued_from_think"])

    def test_empty_string(self):
        think, public, meta = split_think_and_output("")
        self.assertEqual(think, "")
        self.assertEqual(public, "")
        self.assertFalse(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])
        self.assertFalse(meta["rescued_from_think"])

    def test_multiple_think_blocks_honors_first_only(self):
        raw = (
            "<think>first reasoning</think>\n"
            "SKIP\n"
            "<think>second should be public text</think>"
        )
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "first reasoning")
        self.assertIn("SKIP", public)
        self.assertIn("<think>second should be public text</think>", public)
        self.assertTrue(meta["had_tags"])
        self.assertTrue(meta["parse_ok"])
        self.assertFalse(meta["rescued_from_think"])

    def test_unclosed_think_tag(self):
        raw = "<think>\nI never closed this\nCafeteria"
        think, public, meta = split_think_and_output(raw)
        self.assertEqual(think, "")
        self.assertEqual(public, raw.strip())
        self.assertTrue(meta["had_tags"])
        self.assertFalse(meta["parse_ok"])
        self.assertFalse(meta["rescued_from_think"])

    def test_none_input_does_not_raise(self):
        think, public, meta = split_think_and_output(None)  # type: ignore[arg-type]
        self.assertEqual(think, "")
        self.assertEqual(public, "")
        self.assertTrue(meta["parse_ok"])


if __name__ == "__main__":
    unittest.main()
