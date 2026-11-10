"""The tick loop of a timed discussion, and the simultaneous vote that follows it."""

import random
from collections.abc import Sequence
from functools import partial

from loguru import logger

from core.adapters.base import AgentAdapter
from core.situation_room import Meeting, Percept, Reply, Room, RoomConfig, Tick, Vote
from core.transport import InProcessTransport

SKIP = "SKIP"


def onset_delay(config: RoomConfig, agent: str, tick: int) -> float:
    """Gives the seeded delay before an agent starts speaking within a tick.

    Args:
        config: Settings of the discussion.
        agent: Agent name.
        tick: Tick number.

    Returns:
        A delay in seconds, always shorter than a tick.
    """
    limit = min(config.onset_jitter_s, config.tick_s * 0.99)
    return random.Random(f"{config.seed}|onset|{agent}|{tick}").uniform(0.0, limit)


def _turn(adapter: AgentAdapter, percept: Percept, tick: Tick) -> Reply:
    """Lets one agent hear the room and answer."""
    adapter.hear(percept)
    return adapter.reply(tick)


def run_timed_discussion(
    adapters: Sequence[AgentAdapter],
    meeting: Meeting,
    config: RoomConfig,
    transport: InProcessTransport | None = None,
    reply_timeout_s: float | None = None,
) -> Room:
    """Runs a discussion in which every agent speaks during the same timed window.

    At each tick every ready agent hears the same room state and replies in parallel.
    The controller commits the replies after all have arrived, so the order of arrival
    has no effect. An agent is ready when its last utterance has ended, or always when
    barge-in is on. The discussion ends when the budget is used up, or earlier after
    `quiet_close_ticks` ticks in which nobody spoke and nobody was still talking.

    Args:
        adapters: One adapter per agent.
        meeting: Round, reason and caller of the meeting.
        config: Settings of the discussion.
        transport: How to reach the adapters. Defaults to the in-process transport.
        reply_timeout_s: Time to wait for the replies of one tick, or None to wait for ever.

    Returns:
        The room, closed, with every utterance committed.
    """
    transport = transport or InProcessTransport()
    names = [a.name for a in adapters]
    room = Room(names, config)
    for adapter in adapters:
        adapter.begin_meeting(meeting)

    t = 0.0
    quiet = 0
    while t < config.budget_s - 1e-9:
        room.ticks += 1
        tick = Tick(room.ticks, t, config.budget_s - t)
        ready = [a for a in adapters if config.barge_in or room.free_at(a.name) <= t + 1e-9]
        jobs = {a.name: partial(_turn, a, room.percept(a.name, t, tick.tick), tick) for a in ready}
        replies = transport.collect(jobs, reply_timeout_s)
        spoke = 0
        for adapter in ready:
            reply = replies[adapter.name]
            if reply is None or reply.text is None:
                continue
            start = t + onset_delay(config, adapter.name, tick.tick)
            room.append(adapter.name, reply.text, start, tick.tick, cursor=t, addressed_to=reply.addressed_to)
            spoke += 1
        t += config.tick_s
        still_talking = any(room.free_at(name) > t + 1e-9 for name in names)
        quiet = 0 if spoke or still_talking else quiet + 1
        logger.debug("tick {} at {:.0f} s: {} spoke, {} quiet", tick.tick, tick.t, spoke, quiet)
        if config.quiet_close_ticks and quiet >= config.quiet_close_ticks and t < config.budget_s - 1e-9:
            room.closed_early = True
            break

    room.close(min(t, config.budget_s))
    for adapter in adapters:
        adapter.end_meeting()
    return room


def collect_votes(
    adapters: Sequence[AgentAdapter],
    room: Room,
    transport: InProcessTransport | None = None,
    timeout_s: float | None = None,
) -> dict[str, str]:
    """Collects every agent's vote in one simultaneous tick.

    Every agent sees the final state of the room. A missing, invalid or self-directed
    vote counts as a skip.

    Args:
        adapters: One adapter per agent.
        room: The closed room.
        transport: How to reach the adapters. Defaults to the in-process transport.
        timeout_s: Time to wait for the votes, or None to wait for ever.

    Returns:
        The target of each agent's vote.
    """
    transport = transport or InProcessTransport()
    end = room.closed_at if room.closed_at is not None else room.config.budget_s
    names = [a.name for a in adapters]
    jobs = {}
    for adapter in adapters:
        percept = room.percept(adapter.name, end, room.ticks + 1)
        candidates = [n for n in names if n != adapter.name] + [SKIP]
        jobs[adapter.name] = partial(adapter.vote, percept, candidates)
    results = transport.collect(jobs, timeout_s)
    votes = {}
    for name in names:
        vote: Vote | None = results[name]
        valid = vote is not None and vote.target in names and vote.target != name
        votes[name] = vote.target if valid else SKIP
    return votes
