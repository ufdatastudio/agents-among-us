"""A typed, append-only record store in which each record lists the agents who may read it.

Everything an agent could know in a game is a record: what the engine showed it (a fact), what an agent said
(a claim), what it privately believes (a belief), what it promised (a commitment), how it voted (a vote) and what
it did (an action). The ledger gives every record the next id, so ids order the records, and nothing is changed
or removed after it is appended. A correction is a new record that refers to the old one.

A chunk is one piece of a stream, such as a few spoken words or a slice of audio, written while the speaker is still
talking. A reader follows the ledger with a subscription, which remembers how far the reader has read and returns only
the records that have become audible since.

Usage:
    ledger = Ledger()
    fact = ledger.append(FACT, "engine", {"room": "Storage"}, subject="Agent_3", round_num=1, tick=2, visible_to=("Agent_3",))
    ledger.view("Agent_3")  # the records Agent_3 may read
    heard = ledger.subscribe("Agent_3")
    heard.poll(until_s=12.0)  # the records Agent_3 has not read yet that were fully spoken by 12 s
"""

import asyncio
import json
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

FACT = "fact"
CLAIM = "claim"
BELIEF = "belief"
COMMITMENT = "commitment"
VOTE = "vote"
ACTION = "action"
NOTE = "note"
CHUNK = "chunk"
TYPES = (FACT, CLAIM, BELIEF, COMMITMENT, VOTE, ACTION, NOTE, CHUNK)
_EPSILON = 1e-9


@dataclass(frozen=True)
class Record:
    """One entry of the ledger.

    Attributes:
        id: Position in the ledger, from 1.
        type: One of `TYPES`.
        author: Who wrote it, which is an agent or `engine`.
        round_num: Round of the game it belongs to, or None.
        tick: Movement tick of the round for a fact, or None.
        time_s: Room time in a meeting at which a claim or chunk was fully spoken, or None.
        subject: The agent the record is about, or None.
        content: What the record says, as plain values.
        refs: Ids of earlier records that this one cites or corrects.
        visible_to: Agents who may read it, or None when every agent may.
    """

    id: int
    type: str
    author: str
    round_num: int | None
    tick: int | None
    time_s: float | None
    subject: str | None
    content: dict
    refs: tuple[int, ...] = ()
    visible_to: tuple[str, ...] | None = None

    def visible(self, reader: str) -> bool:
        """Tells whether an agent may read this record.

        Args:
            reader: Agent name.

        Returns:
            True when the record is open to everyone or lists the reader.
        """
        return self.visible_to is None or reader in self.visible_to

    def to_dict(self) -> dict:
        """Returns the record as plain values for JSON."""
        return asdict(self)


class Ledger:
    """The records of one game, in the order they were appended."""

    def __init__(self) -> None:
        """Creates an empty ledger."""
        self.records: list[Record] = []
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Record], None]] = []

    def __len__(self) -> int:
        return len(self.records)

    def append(
        self,
        type: str,
        author: str,
        content: dict,
        *,
        round_num: int | None = None,
        tick: int | None = None,
        time_s: float | None = None,
        subject: str | None = None,
        refs: Sequence[int] = (),
        visible_to: Iterable[str] | None = None,
    ) -> Record:
        """Appends a record.

        Args:
            type: One of `TYPES`.
            author: Who wrote it.
            content: What it says.
            round_num: Round of the game.
            tick: Movement tick of the round.
            time_s: Room time at which a claim was fully spoken.
            subject: The agent it is about.
            refs: Ids of earlier records it cites or corrects.
            visible_to: Agents who may read it, or None for everyone.

        Returns:
            The record, with the next id.

        Raises:
            ValueError: If the type is unknown or a reference is not an earlier record.
        """
        if type not in TYPES:
            raise ValueError(f"type must be one of {TYPES}, got {type!r}")
        with self._lock:
            new_id = len(self.records) + 1
            if any(not 1 <= ref < new_id for ref in refs):
                raise ValueError(f"refs {list(refs)} must be ids of earlier records, and the next id is {new_id}")
            record = Record(
                id=new_id, type=type, author=author, round_num=round_num, tick=tick, time_s=time_s,
                subject=subject, content=dict(content), refs=tuple(refs),
                visible_to=None if visible_to is None else tuple(visible_to),
            )
            self.records.append(record)
            listeners = list(self._listeners)
        for listener in listeners:
            listener(record)
        return record

    def listen(self, listener: Callable[[Record], None]) -> Callable[[], None]:
        """Registers a function that is called with every record after it is appended.

        The function runs in the thread that appended, outside the lock of the ledger, so it must be quick and
        must not raise.

        Args:
            listener: The function to call.

        Returns:
            A function that removes the listener.
        """
        with self._lock:
            self._listeners.append(listener)

        def remove() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return remove

    def subscribe(self, reader: str, types: Iterable[str] | None = None, after: int = 0) -> "Subscription":
        """Follows the ledger as one reader.

        Args:
            reader: Agent name.
            types: Record types to follow, or None for all.
            after: Id after which to start, so that records up to it count as read.

        Returns:
            The subscription.
        """
        return Subscription(self, reader, types, after)

    def get(self, record_id: int) -> Record:
        """Returns the record with an id."""
        return self.records[record_id - 1]

    def view(self, reader: str, types: Iterable[str] | None = None) -> list[Record]:
        """Lists the records an agent may read, in order.

        Args:
            reader: Agent name.
            types: Record types to keep, or None for all.

        Returns:
            The records.
        """
        wanted = None if types is None else set(types)
        return [r for r in list(self.records) if r.visible(reader) and (wanted is None or r.type in wanted)]

    def write_jsonl(self, path: Path) -> None:
        """Writes the ledger as JSON lines, one record per line.

        Args:
            path: Output file. Its folder is created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in list(self.records):
                handle.write(json.dumps(record.to_dict()) + "\n")

    @classmethod
    def read_jsonl(cls, path: Path) -> "Ledger":
        """Reads a ledger written by `write_jsonl`.

        Args:
            path: Input file.

        Returns:
            The ledger, with the same ids.
        """
        ledger = cls()
        for line in path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            ledger.append(
                data["type"], data["author"], data["content"], round_num=data["round_num"], tick=data["tick"],
                time_s=data["time_s"], subject=data["subject"], refs=data["refs"], visible_to=data["visible_to"],
            )
        return ledger


class Subscription:
    """The position of one reader in a ledger, which returns each record once.

    A record becomes audible at its `time_s`, so a reader that polls with a time does not see a chunk before the
    speaker has finished it. Records are returned in id order, and a record that is not audible yet holds back the
    ones after it, so the reader never sees a later record without an earlier one.
    """

    def __init__(self, ledger: Ledger, reader: str, types: Iterable[str] | None = None, after: int = 0) -> None:
        """Creates a subscription.

        Args:
            ledger: The ledger to follow.
            reader: Agent name.
            types: Record types to follow, or None for all.
            after: Id after which to start.
        """
        self.ledger = ledger
        self.reader = reader
        self._types = None if types is None else frozenset(types)
        self.cursor = after

    def _wanted(self, record: Record) -> bool:
        """Tells whether the record is one this subscription follows."""
        return record.visible(self.reader) and (self._types is None or record.type in self._types)

    def pending(self) -> bool:
        """Tells whether the ledger has a record this reader has not read, whatever its time.

        Returns:
            True when a record after the cursor is visible to the reader.
        """
        return any(self._wanted(r) for r in self.ledger.records[self.cursor :])

    def poll(self, until_s: float | None = None) -> list[Record]:
        """Returns the records the reader has not read and moves the cursor past them.

        Args:
            until_s: Room time up to which records are audible, or None for all of them. A record without a
                time is audible at once.

        Returns:
            The new records in id order.
        """
        new = []
        for record in list(self.ledger.records[self.cursor :]):
            if until_s is not None and record.time_s is not None and record.time_s > until_s + _EPSILON:
                break
            self.cursor = record.id
            if self._wanted(record):
                new.append(record)
        return new

    async def wait(self, timeout_s: float | None = None) -> bool:
        """Waits until the ledger has a record this reader has not read.

        Args:
            timeout_s: Seconds to wait, or None to wait for ever.

        Returns:
            True when there is a record to read, and False when the time ran out.
        """
        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        def wake(record: Record) -> None:
            if self._wanted(record):
                loop.call_soon_threadsafe(event.set)

        remove = self.ledger.listen(wake)
        try:
            if self.pending():
                return True
            await asyncio.wait_for(event.wait(), timeout_s)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            remove()
