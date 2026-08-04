"""Unit tests for latest-thought live-state publishing (Piece 9)."""

import os
import shutil
import unittest
from types import SimpleNamespace

from core.logger import LogManager
from core.state import GameState

TEST_GAME_ID = "_piece9_latest_thoughts"


class TestLatestThoughtsPublish(unittest.TestCase):
    def setUp(self):
        agents = [
            SimpleNamespace(name="Agent_0", role="honest", color="red", model_name="m", action_num=0),
            SimpleNamespace(name="Agent_1", role="byzantine", color="blue", model_name="m", action_num=0),
        ]
        self.logger = LogManager(TEST_GAME_ID, agents)
        self.state = GameState(agents, self.logger)
        self.base = self.logger.base_dir

    def tearDown(self):
        if os.path.isdir(self.base):
            shutil.rmtree(self.base)
        live = self.state.live_state_file
        if os.path.isfile(live):
            # Only remove if it's our test write path; live file is shared name.
            pass

    def test_publish_keeps_only_latest_per_agent(self):
        self.state.publish_latest_thought(
            {
                "round": 1,
                "phase": "MOVEMENT",
                "tick": 1,
                "agent": "Agent_0",
                "role": "honest",
                "model": "m",
                "think": "first",
                "output": "Cafeteria",
                "had_tags": True,
                "parse_ok": True,
            }
        )
        self.state.publish_latest_thought(
            {
                "round": 1,
                "phase": "MOVEMENT",
                "tick": 2,
                "agent": "Agent_0",
                "role": "honest",
                "model": "m",
                "think": "second",
                "output": "MedBay",
                "had_tags": True,
                "parse_ok": True,
            }
        )
        self.state.publish_latest_thought(
            {
                "round": 1,
                "phase": "VOTING",
                "tick": 1,
                "agent": "Agent_1",
                "role": "byzantine",
                "model": "m",
                "think": "vote plan",
                "output": "SKIP",
                "had_tags": True,
                "parse_ok": True,
            }
        )

        latest = self.state.world_data["latest_thoughts"]
        self.assertEqual(set(latest.keys()), {"Agent_0", "Agent_1"})
        self.assertEqual(latest["Agent_0"]["think"], "second")
        self.assertEqual(latest["Agent_0"]["tick"], 2)
        self.assertEqual(latest["Agent_1"]["phase"], "VOTING")
        # Bounded by agent count: never accumulates history list.
        self.assertEqual(len(latest), 2)

        history = self.state.world_data["thought_history"]
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["think"], "first")
        self.assertEqual(history[1]["think"], "second")
        self.assertEqual(history[2]["agent"], "Agent_1")
        self.assertEqual([row["seq"] for row in history], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
