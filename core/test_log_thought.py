"""Verify log_thought writes JSONL to thought.log and leaves public logs untouched."""

import json
import os
import shutil
import unittest
from types import SimpleNamespace

from core.logger import LogManager

TEST_GAME_ID = "_piece2_thought_logger"


class TestLogThought(unittest.TestCase):
    def setUp(self):
        agents = [
            SimpleNamespace(name="Agent_0", role="honest"),
            SimpleNamespace(name="Agent_1", role="byzantine"),
        ]
        self.logger = LogManager(TEST_GAME_ID, agents)
        self.base = self.logger.base_dir

    def tearDown(self):
        if os.path.isdir(self.base):
            shutil.rmtree(self.base)

    def _snapshot_public_logs(self):
        """Sizes of every public log that agents/observers may read."""
        paths = [
            self.logger.paths["discussion"],
            self.logger.paths["round_results"],
            self.logger.paths["discussion_chat"],
            self.logger.paths["agents"]["Agent_0"]["action"],
            self.logger.paths["agents"]["Agent_0"]["vote"],
            self.logger.paths["agents"]["Agent_1"]["action"],
            self.logger.paths["agents"]["Agent_1"]["vote"],
        ]
        return {p: os.path.getsize(p) for p in paths}

    def test_log_thought_schema_and_isolation(self):
        before = self._snapshot_public_logs()
        thought_path = self.logger.paths["thought"]
        self.assertTrue(os.path.exists(thought_path))
        self.assertEqual(os.path.getsize(thought_path), 0)

        self.logger.log_thought(
            round=1,
            phase="MOVEMENT",
            tick=0,
            agent="Agent_0",
            role="honest",
            model="test-model",
            think="Medbay looked empty; Cafeteria is safer.",
            output="Cafeteria",
            had_tags=True,
            parse_ok=True,
        )
        self.logger.log_thought(
            round=1,
            phase="VOTING",
            tick=0,
            agent="Agent_1",
            role="byzantine",
            model="test-model",
            think="Deflect onto Agent_0.\nSKIP",
            output="SKIP",
            had_tags=True,
            parse_ok=True,
        )
        # Missing-tag style entry
        self.logger.log_thought(
            round=2,
            phase="DISCUSSION",
            tick=1,
            agent="Agent_0",
            role="honest",
            model="test-model",
            think="",
            output="Agent_3 is suspicious.",
            had_tags=False,
            parse_ok=True,
        )

        with open(thought_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]

        self.assertEqual(len(lines), 3)

        first = json.loads(lines[0])
        self.assertEqual(
            set(first.keys()),
            {
                "round",
                "phase",
                "tick",
                "agent",
                "role",
                "model",
                "think",
                "output",
                "had_tags",
                "parse_ok",
            },
        )
        self.assertEqual(first["agent"], "Agent_0")
        self.assertEqual(first["phase"], "MOVEMENT")
        self.assertEqual(first["think"], "Medbay looked empty; Cafeteria is safer.")
        self.assertEqual(first["output"], "Cafeteria")
        self.assertTrue(first["had_tags"])
        self.assertTrue(first["parse_ok"])

        second = json.loads(lines[1])
        self.assertIn("\n", second["think"])  # newlines survive JSONL

        third = json.loads(lines[2])
        self.assertEqual(third["think"], "")
        self.assertFalse(third["had_tags"])

        after = self._snapshot_public_logs()
        self.assertEqual(
            before,
            after,
            "log_thought must not modify discussion/action/vote/stats logs",
        )


if __name__ == "__main__":
    unittest.main()
