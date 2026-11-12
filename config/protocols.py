"""The discussion protocols that the situation room experiments compare.

A protocol is a name and the `main.py` flags that select it. Every protocol replays the
same scenario seeds, so games differ only in how the agents talk.
"""

from dataclasses import dataclass

P2_BUDGETS_S = (60, 180, 300)
NOISY_DROPOUT = 0.1
NOISY_TAG_POLICY = "first_chunk"


@dataclass(frozen=True)
class Protocol:
    """One way of running the discussion.

    Attributes:
        name: Short label used in file names and tables.
        args: Flags for `main.py`.
    """

    name: str
    args: tuple[str, ...]


def timed(overlap: str, budget_s: int) -> Protocol:
    """Builds a timed-room protocol.

    Args:
        overlap: `clean`, `interleaved` or `noisy`.
        budget_s: Length of the discussion in room seconds.

    Returns:
        The protocol. The noisy one drops a tenth of the words and tags only each speaker's first line.
    """
    args = ["--discussion_mode", "timed", "--room_overlap", overlap, "--room_budget_s", str(budget_s)]
    if overlap == "noisy":
        args += ["--room_dropout", str(NOISY_DROPOUT), "--room_tag_policy", NOISY_TAG_POLICY]
    return Protocol(f"{overlap}-{budget_s}", tuple(args))


SEQUENTIAL = Protocol("sequential", ("--discussion_mode", "turns"))
NO_DISCUSSION = Protocol("none", ("--skip_discussion",))

# Sequential, the nine timed cells and no discussion, as in phase P2 of docs/situation-room.md.
P2_PROTOCOLS = (
    SEQUENTIAL,
    *(timed(overlap, budget) for overlap in ("clean", "interleaved", "noisy") for budget in P2_BUDGETS_S),
    NO_DISCUSSION,
)

# Protocols of the shared ledger comparison in docs/shared-ledger.md. `board-180` is the interleaved room of 180 s
# whose percepts end with the board of the meeting, so it differs from `interleaved-180` only by the board.
BOARD_180 = Protocol("board-180", (*timed("interleaved", 180).args, "--room_board"))
P6_PROTOCOLS = (BOARD_180,)
ALL_PROTOCOLS = (*P2_PROTOCOLS, *P6_PROTOCOLS)


def protocol_by_name(name: str) -> Protocol:
    """Finds a P2 or P6 protocol by its name.

    Args:
        name: For example `sequential`, `clean-180`, `none` or `board-180`.

    Returns:
        The protocol.

    Raises:
        KeyError: If no P2 protocol has the name.
    """
    for protocol in ALL_PROTOCOLS:
        if protocol.name == name:
            return protocol
    raise KeyError(f"no protocol named {name!r}, choose from {[p.name for p in ALL_PROTOCOLS]}")
