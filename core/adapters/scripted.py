"""A deterministic adapter for tests and dry runs of the situation room."""

from collections.abc import Callable, Mapping

from core.situation_room import Meeting, Percept, Reply, Tick, Vote, parse_reply


class ScriptedAdapter:
    """Replies from a script and records everything it hears.

    Attributes:
        name: Agent name.
        heard: Every percept delivered, in order.
        meetings: Every meeting started, in order.
        ended: Number of meetings ended.
    """

    def __init__(
        self,
        name: str,
        roster: list[str],
        script: Mapping[int, str] | Callable[[Tick, list[Percept]], str | None] | None = None,
        vote_for: str = "SKIP",
    ) -> None:
        """Creates the adapter.

        Args:
            name: Agent name.
            roster: Names of every agent, used to read `<to>` tags.
            script: Raw replies by tick number, or a function of the tick and the percepts
                heard so far. A missing reply is silence.
            vote_for: The agent's vote.
        """
        self.name = name
        self._roster = roster
        self._script = script or {}
        self._vote_for = vote_for
        self.heard: list[Percept] = []
        self.meetings: list[Meeting] = []
        self.ended = 0

    def begin_meeting(self, meeting: Meeting) -> None:
        """Records the meeting."""
        self.meetings.append(meeting)

    def hear(self, percept: Percept) -> None:
        """Records the percept."""
        self.heard.append(percept)

    def reply(self, tick: Tick) -> Reply:
        """Answers from the script.

        Args:
            tick: The decision point.

        Returns:
            The scripted reply for this tick, or silence.
        """
        if callable(self._script):
            raw = self._script(tick, self.heard)
        else:
            raw = self._script.get(tick.tick)
        return parse_reply(self.name, tick.tick, raw, self._roster)

    def vote(self, percept: Percept, candidates: list[str]) -> Vote:
        """Votes for the configured target.

        Args:
            percept: Everything the agent heard.
            candidates: Names the agent may vote for.

        Returns:
            The vote.
        """
        self.heard.append(percept)
        return Vote(self.name, self._vote_for)

    def end_meeting(self) -> None:
        """Counts the end of a meeting."""
        self.ended += 1
