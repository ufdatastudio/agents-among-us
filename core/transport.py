"""How the controller reaches its agents.

Phase 0 has only the in-process transport, which runs the agents' calls on a thread
pool. Transports for other processes and nodes would carry the same messages.
"""

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from loguru import logger


class InProcessTransport:
    """Runs one call per agent at once and collects the results."""

    def __init__(self, max_workers: int | None = None) -> None:
        """Creates the transport.

        Args:
            max_workers: Calls to run at the same time, or None to run one per agent. Use 1 for a
                backend that cannot serve calls in parallel, such as a model loaded in this process.
        """
        self._max_workers = max_workers

    def collect(self, jobs: Mapping[str, Callable[[], Any]], timeout_s: float | None = None) -> dict[str, Any]:
        """Runs every job in parallel and waits for the results.

        A job that raises, or that has not finished when the timeout passes, yields None,
        which the caller treats as silence. A job that times out is left to finish on its own.

        Args:
            jobs: A callable for each agent name.
            timeout_s: Time to wait for all jobs, or None to wait for ever.

        Returns:
            The result of each job, in the order of `jobs`.
        """
        results: dict[str, Any] = {name: None for name in jobs}
        if not jobs:
            return results
        pool = ThreadPoolExecutor(max_workers=min(len(jobs), self._max_workers or len(jobs)))
        futures = {name: pool.submit(job) for name, job in jobs.items()}
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for name, future in futures.items():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                results[name] = future.result(timeout=remaining)
            except FutureTimeout:
                logger.warning("{} did not answer within {} s", name, timeout_s)
            except Exception as error:  # noqa: BLE001 - one agent's failure must not stop the others
                logger.error("{} failed: {!r}", name, error)
        pool.shutdown(wait=False, cancel_futures=True)
        return results
