"""Unit tests for movement thought-capture helper (Pieces 4–5)."""

import json
import os
import shutil
import unittest
from types import SimpleNamespace

from agents.thought_capture import (
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_WITH_THOUGHTS,
    generate_with_optional_thoughts,
    tokens_for_turn,
)
from core.logger import LogManager

TEST_GAME_ID = "_piece45_thought_capture"


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1, max_tokens=160):
        self.calls.append(
            {
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected extra LLM call")
        return self.responses.pop(0)


class TestThoughtCaptureHelper(unittest.TestCase):
    def setUp(self):
        agents = [SimpleNamespace(name="Agent_0", role="honest")]
        self.logger = LogManager(TEST_GAME_ID, agents)
        self.base = self.logger.base_dir

    def tearDown(self):
        if os.path.isdir(self.base):
            shutil.rmtree(self.base)

    def _agent(self, llm, capture=True, require=False):
        return SimpleNamespace(
            name="Agent_0",
            role="honest",
            model_name="test-model",
            llm=llm,
            logger=self.logger,
            capture_thoughts=capture,
            require_think_tags=require,
        )

    def test_tokens_follow_capture_flag(self):
        on = SimpleNamespace(capture_thoughts=True)
        off = SimpleNamespace(capture_thoughts=False)
        self.assertEqual(tokens_for_turn(on), MAX_TOKENS_WITH_THOUGHTS)
        self.assertEqual(tokens_for_turn(off), MAX_TOKENS_DEFAULT)

    def test_capture_parses_and_logs_public_only(self):
        llm = _FakeLLM(
            [
                "<think>\nMedbay is risky.\n</think>\nCafeteria",
            ]
        )
        agent = self._agent(llm, capture=True)
        public = generate_with_optional_thoughts(
            agent,
            "sys",
            "move prompt",
            temperature=0.1,
            phase="MOVEMENT",
            round_num=1,
            tick=2,
            world_view={"capture_thoughts": True, "require_think_tags": False},
        )
        self.assertEqual(public, "Cafeteria")
        self.assertEqual(llm.calls[0]["max_tokens"], MAX_TOKENS_WITH_THOUGHTS)

        with open(self.logger.paths["thought"], encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["think"], "Medbay is risky.")
        self.assertEqual(rows[0]["output"], "Cafeteria")
        self.assertEqual(rows[0]["phase"], "MOVEMENT")
        self.assertTrue(rows[0]["had_tags"])

        # Public action log must stay empty beyond LogManager headers.
        action_path = self.logger.paths["agents"]["Agent_0"]["action"]
        with open(action_path, encoding="utf-8") as f:
            action_text = f.read()
        self.assertNotIn("Medbay is risky", action_text)
        self.assertNotIn("<think>", action_text)

    def test_require_tags_retries_once(self):
        llm = _FakeLLM(
            [
                "Cafeteria",
                "<think>\nok</think>\nCafeteria",
            ]
        )
        agent = self._agent(llm, capture=True, require=True)
        public = generate_with_optional_thoughts(
            agent,
            "sys",
            "move prompt",
            temperature=0.1,
            phase="MOVEMENT",
            round_num=1,
            tick=1,
            world_view={"capture_thoughts": True, "require_think_tags": True},
        )
        self.assertEqual(public, "Cafeteria")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("forgot the <think>", llm.calls[1]["user_prompt"])

    def test_capture_off_skips_parse_and_log(self):
        llm = _FakeLLM(["Cafeteria"])
        agent = self._agent(llm, capture=False)
        public = generate_with_optional_thoughts(
            agent,
            "sys",
            "move prompt",
            temperature=0.1,
            phase="MOVEMENT",
            round_num=1,
            tick=1,
            world_view={"capture_thoughts": False},
        )
        self.assertEqual(public, "Cafeteria")
        self.assertEqual(llm.calls[0]["max_tokens"], MAX_TOKENS_DEFAULT)
        self.assertEqual(os.path.getsize(self.logger.paths["thought"]), 0)

    def test_discussion_public_never_leaks_into_discussion_log(self):
        """Observer / discussion.log must only ever see public_output."""
        secret = "PRIVATE_PLAN_VOTE_OUT_AGENT_2"
        spoken = "Agent_1 was near the body in Medbay."
        llm = _FakeLLM([f"<think>\n{secret}\n</think>\n{spoken}"])
        agent = self._agent(llm, capture=True)

        public = generate_with_optional_thoughts(
            agent,
            "sys",
            "discussion prompt",
            temperature=1.0,
            phase="DISCUSSION",
            round_num=2,
            tick=3,
            world_view={"capture_thoughts": True, "require_think_tags": False},
        )
        self.assertEqual(public, spoken)
        self.assertNotIn(secret, public)
        self.assertNotIn("<think>", public)

        # Same path game_engine uses for public discussion visibility.
        formatted_msg = f"{agent.name}: {public}"
        self.logger.write_log("discussion", None, formatted_msg)
        self.logger.log_discussion_chat(
            discussion_num=2,
            reason="body report",
            agent_name=agent.name,
            model_name=agent.model_name,
            role=agent.role,
            message=public,
        )

        with open(self.logger.paths["discussion"], encoding="utf-8") as f:
            discussion_text = f.read()
        with open(self.logger.paths["discussion_chat"], encoding="utf-8") as f:
            chat_csv = f.read()
        with open(self.logger.paths["thought"], encoding="utf-8") as f:
            thought_rows = [json.loads(line) for line in f if line.strip()]

        self.assertIn(spoken, discussion_text)
        self.assertNotIn(secret, discussion_text)
        self.assertNotIn("<think>", discussion_text)
        self.assertIn(spoken, chat_csv)
        self.assertNotIn(secret, chat_csv)

        self.assertEqual(len(thought_rows), 1)
        self.assertEqual(thought_rows[0]["phase"], "DISCUSSION")
        self.assertEqual(thought_rows[0]["think"], secret)
        self.assertEqual(thought_rows[0]["output"], spoken)

        # Observer input shape: only public Text field.
        round_statements = [
            {
                "Agent": agent.name,
                "Text": public,
                "Reported": 0,
                "S_Num": 1,
            }
        ]
        self.assertNotIn(secret, round_statements[0]["Text"])

    def test_voting_matches_public_not_think_content(self):
        # Think mentions SKIP; public vote is Agent_1 — matching must use public.
        llm = _FakeLLM(
            ["<think>\nMaybe I should SKIP to look innocent.\n</think>\nAgent_1"]
        )
        agent = self._agent(llm, capture=True)
        public = generate_with_optional_thoughts(
            agent,
            "sys",
            "vote prompt",
            temperature=0.1,
            phase="VOTING",
            round_num=3,
            tick=1,
            world_view={"capture_thoughts": True},
        )
        self.assertEqual(public, "Agent_1")
        candidates = ["Agent_1", "Agent_2", "SKIP"]
        vote = "SKIP"
        for cand in candidates:
            if cand in public:
                vote = cand
                break
        self.assertEqual(vote, "Agent_1")

        with open(self.logger.paths["thought"], encoding="utf-8") as f:
            row = json.loads(f.readline())
        self.assertEqual(row["phase"], "VOTING")
        self.assertIn("SKIP", row["think"])
        self.assertEqual(row["output"], "Agent_1")


if __name__ == "__main__":
    unittest.main()
