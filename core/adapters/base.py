"""The interface between the situation room and an agent.

The room knows nothing about models, GPUs or APIs. An adapter hears the room and
replies to it, and it keeps the agent's private state (role, action log, results log).
The messages are plain dataclasses, so an adapter can live in another process,
on another node, or behind an API.
"""

from typing import Protocol, runtime_checkable

from core.situation_room import Meeting, Percept, Reply, Tick, Vote


@runtime_checkable
class AgentAdapter(Protocol):
    """An agent as the situation room sees it."""

    name: str

    def begin_meeting(self, meeting: Meeting) -> None:
        """Tells the agent a meeting has started.

        Args:
            meeting: Round, reason and caller of the meeting.
        """

    def hear(self, percept: Percept) -> None:
        """Delivers what the agent hears at a tick.

        Args:
            percept: The rendering of everything the agent has heard so far.
        """

    def reply(self, tick: Tick) -> Reply:
        """Asks the agent what it says at a tick.

        Args:
            tick: The decision point.

        Returns:
            An utterance, or silence.
        """

    def vote(self, percept: Percept, candidates: list[str]) -> Vote:
        """Asks the agent for its vote after the discussion.

        Args:
            percept: Everything the agent heard.
            candidates: Names the agent may vote for, including SKIP.

        Returns:
            The vote.
        """

    def end_meeting(self) -> None:
        """Tells the agent the meeting is over."""
