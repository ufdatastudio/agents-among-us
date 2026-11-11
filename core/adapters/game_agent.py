"""An adapter that lets an existing honest or Byzantine agent speak in the situation room.

The agents were written for turn-taking, and read the discussion from log files. The
adapter builds its own prompt from what the room lets the agent hear, and generates with
the agent's system prompt and model client, so the role and objective stay as they were.
For the vote it writes what the agent heard to a file in the format `vote()` already reads.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from agents.room_prompts import build_room_prompt
from agents.thought_capture import capture_enabled, generate_with_optional_thoughts
from core.situation_room import Meeting, Percept, Reply, RoomConfig, Tick, Vote, format_time, parse_reply
from core.voting import SKIP, ask_until_valid

# Phase label for the thought log. It differs from DISCUSSION on purpose, because the
# generator retries any DISCUSSION reply of fewer than 45 words, and silence is allowed here.
ROOM_PHASE = "ROOM"

# What ModelManager returns when an API call fails, a leftover from the movement phase.
FAILED_CALL = "move"


def clean_speech(raw: str | None, name: str) -> str:
    """Tidies a model's output the way the turn-based engine tidies a message.

    Args:
        raw: The agent's public output.
        name: The agent's own name, which models sometimes put at the start.

    Returns:
        One line of text without a leading name or surrounding quotes.
    """
    text = " ".join((raw or "").replace("\r", "").split())
    text = re.sub(rf"^(\*\*)?{re.escape(name)}:?(\*\*)?:?\s*", "", text, flags=re.IGNORECASE)
    return text.strip().strip('"').strip("'")


class GameAgentAdapter:
    """Speaks and votes in the room on behalf of an existing agent.

    Attributes:
        name: Agent name.
        agent: The wrapped HonestAgent or ByzantineAgent.
    """

    def __init__(
        self,
        agent: Any,
        roster: list[str],
        config: RoomConfig,
        view_factory: Callable[[str, int], dict],
        percept_dir: Path,
        vote_retries: int = 2,
    ) -> None:
        """Creates the adapter.

        Args:
            agent: The agent to wrap.
            roster: Names of every agent in the room.
            config: Settings of the discussion.
            view_factory: Builds the agent's view of the game for a round. It is called once per
                meeting from the controller's thread, because building a view changes game state.
            percept_dir: Folder for the files in which the agent's final percept is written.
            vote_retries: Asks to add when the agent's vote reply does not name a candidate.
        """
        self.name = agent.name
        self.agent = agent
        self._roster = roster
        self._config = config
        self._view_factory = view_factory
        self._percept_dir = percept_dir
        self._vote_retries = vote_retries
        self._meeting: Meeting | None = None
        self._view: dict = {}
        self._action_log = ""
        self._results_log = ""
        self._percept: Percept | None = None

    def begin_meeting(self, meeting: Meeting) -> None:
        """Reads the agent's logs once, before the first tick.

        Args:
            meeting: Round, reason and caller of the meeting.
        """
        self._meeting = meeting
        self._view = self._view_factory(self.name, meeting.round_num)
        full_action_log = self.agent._read_file(self._view["log_path"])
        self._action_log = self.agent._get_current_round_log(full_action_log, meeting.round_num)
        self._results_log = self.agent._read_file(self._view["results_log_path"])
        self._percept = None

    def hear(self, percept: Percept) -> None:
        """Keeps what the agent hears at this tick.

        Args:
            percept: The rendering of everything the agent has heard so far.
        """
        self._percept = percept

    def reply(self, tick: Tick) -> Reply:
        """Asks the agent's model what to say at a tick.

        Args:
            tick: The decision point.

        Returns:
            An utterance, or silence.
        """
        assert self._meeting is not None and self._percept is not None, "reply before begin_meeting and hear"
        view = {**self._view, "tick": tick.tick, "phase": ROOM_PHASE}
        prompt = build_room_prompt(
            role=self.agent.role,
            recent_action_log=self._action_log,
            results_log=self._results_log,
            transcript=self._percept.text,
            now=format_time(tick.t),
            budget=format_time(self._config.budget_s),
            remaining=format_time(tick.remaining_s),
            max_words=self._config.max_words,
            teammates=tuple(getattr(self.agent, "teammates", None) or ()),
            capture_thoughts=capture_enabled(self.agent, view),
        )
        raw = generate_with_optional_thoughts(
            self.agent,
            self.agent._system_prompt(),
            prompt,
            temperature=1.0,
            phase=ROOM_PHASE,
            round_num=self._meeting.round_num,
            tick=tick.tick,
            world_view=view,
        )
        speech = clean_speech(raw, self.name)
        if speech.lower() == FAILED_CALL:
            logger.warning("{} got a failed model call at tick {}, so it stays silent", self.name, tick.tick)
            speech = ""
        return parse_reply(self.name, tick.tick, speech, self._roster)

    def vote(self, percept: Percept, candidates: list[str]) -> Vote:
        """Asks the agent for its vote, showing what it heard in the room.

        Args:
            percept: Everything the agent heard.
            candidates: Names the agent may vote for, with SKIP only when skipping is allowed.

        Returns:
            The vote. Its target is SKIP when no ask gave a candidate.
        """
        assert self._meeting is not None, "vote before begin_meeting"
        if getattr(self.agent, "is_hybrid", False):
            logger.warning("{} is a hybrid agent, but the room does not prune its context", self.name)
        round_num = self._meeting.round_num
        self._percept_dir.mkdir(parents=True, exist_ok=True)
        path = self._percept_dir / f"{self.name}_round{round_num}_heard.log"
        path.write_text(f"=== Round {round_num} ===\n{percept.text}\n", encoding="utf-8")
        view = {**self._view, "discussion_log_path": str(path), "tick": percept.tick, "phase": "VOTING"}

        def ask(temperature: float, feedback: str | None = None) -> str | None:
            return self.agent.vote({**view, "vote_temperature": temperature, "vote_feedback": feedback}, candidates, round_num)

        def last_reply() -> str:
            reads = getattr(self.agent, "vote_reads", [])
            return reads[-1]["reply"] if reads else ""

        target = ask_until_valid(ask, candidates, self._vote_retries, self.name, last_reply)
        return Vote(self.name, SKIP if target is None else target)

    def end_meeting(self) -> None:
        """Drops the last percept. The meeting stays known, because the vote comes after it."""
        self._percept = None
