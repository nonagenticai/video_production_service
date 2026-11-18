"""Separate a recording job's QUEUE-WAIT from its true EXECUTION time.

Why this exists
---------------
``recording_time_seconds`` used to be measured from a ``start = time.time()``
taken *before* ``RecordingConcurrencyManager.acquire()``. The fleet cap is a
Redis semaphore shared by every replica, so that wait is unbounded in principle
and routinely long in practice — the stored number was therefore
``queue-wait + execution``, presented as if it were execution.

That is not cosmetic. The cap is 3 fleet-wide, so a 10-minute bucket can hold at
most 1800 slot-seconds of real execution; observed reported sums in single
buckets were 7885s (438% of physically possible) and 8538s (474%). Bounding the
08:10 bucket (12 jobs, mean reported 711s, mean true execute <=150s) puts >=79%
of the reported figure in queue-wait. Any deadline or capacity number sized from
it is sized from saturation, and a slow recording becomes indistinguishable from
a starved one.

Both halves are kept rather than one. The wait is genuinely the interesting
number for a capacity question ("were we starved?"), and the execute is the
interesting number for a performance question ("was the recorder slow?").
Discarding the wait would answer the second question by making the first
unanswerable without archaeology.

Clock choice: ``time.monotonic()``, not ``time.time()``. These are durations,
and a wall-clock step (NTP slew, container clock jump) must not be able to
produce a negative or inflated duration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class JobTiming:
    """Accumulates queue-wait and execution seconds for one recording job.

    Both fields ACCUMULATE, because ``process_job`` retries: a job that waits
    for a slot on each of three attempts has three waits and three executions,
    and the honest totals are the sums. Retry back-off is deliberately in
    neither bucket — it is not queue-wait and it is not execution.
    """

    wait_seconds: float = 0.0
    execute_seconds: float = 0.0

    def add_wait(self, seconds: float) -> None:
        """Add one slot-acquisition wait."""
        self.wait_seconds += max(0.0, seconds)

    def add_execute(self, seconds: float) -> None:
        """Add one post-acquisition execution span."""
        self.execute_seconds += max(0.0, seconds)

    @property
    def total_seconds(self) -> float:
        """Wall time attributable to this job (wait + execute)."""
        return self.wait_seconds + self.execute_seconds


class WaitClock:
    """Context manager timing a slot wait into a :class:`JobTiming`.

    Records on exit whether or not the body raised, so a
    ``RecordingQueueSaturatedError`` — the case where the wait is by definition
    the whole story — still reports how long it waited.
    """

    __slots__ = ("_started", "_timing")

    def __init__(self, timing: JobTiming | None) -> None:
        self._timing = timing
        self._started = 0.0

    def __enter__(self) -> WaitClock:
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._timing is not None:
            self._timing.add_wait(time.monotonic() - self._started)


class ExecuteClock:
    """Context manager timing an execution span into a :class:`JobTiming`.

    Must be entered only AFTER a slot has been acquired — that is the entire
    point of the class. Records on exit whether or not the body raised, so a
    failed attempt still contributes the execution it really consumed.
    """

    __slots__ = ("_started", "_timing")

    def __init__(self, timing: JobTiming | None) -> None:
        self._timing = timing
        self._started = 0.0

    def __enter__(self) -> ExecuteClock:
        self._started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._timing is not None:
            self._timing.add_execute(time.monotonic() - self._started)
