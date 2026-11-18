"""Tests for src/services/concurrency.py -- semaphore-based concurrency control."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from src.services.recording.concurrency import (
    QUEUE_SATURATED_ERROR_PREFIX,
    RecordingConcurrencyManager,
    RecordingQueueSaturatedError,
)

# ── Init ──────────────────────────────────────────────────────────


class TestInit:
    def test_default_max_concurrent(self):
        mgr = RecordingConcurrencyManager()
        assert mgr.max_concurrent == 3

    def test_custom_max_concurrent(self):
        mgr = RecordingConcurrencyManager(max_concurrent=5)
        assert mgr.max_concurrent == 5

    def test_initial_counts(self):
        mgr = RecordingConcurrencyManager()
        assert mgr.active_count == 0
        assert mgr.queued_count == 0


# ── acquire / release ────────────────────────────────────────────


class TestAcquireRelease:
    async def test_acquire_increments_active(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        await mgr.acquire()
        assert mgr.active_count == 1

    async def test_release_decrements_active(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        await mgr.acquire()
        await mgr.release()
        assert mgr.active_count == 0

    async def test_release_does_not_go_negative(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        await mgr.release()
        assert mgr.active_count == 0


# ── stats ────────────────────────────────────────────────────────


class TestStats:
    async def test_returns_correct_dict(self):
        mgr = RecordingConcurrencyManager(max_concurrent=2)
        await mgr.acquire()
        stats = mgr.stats()
        assert stats["active"] == 1
        assert stats["queued"] == 0
        assert stats["max_concurrent"] == 2
        assert stats["available_slots"] == 1


# ── available_slots ──────────────────────────────────────────────


class TestAvailableSlots:
    async def test_correct_when_partially_used(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        await mgr.acquire()
        await mgr.acquire()
        assert mgr.available_slots == 1

    async def test_zero_when_full(self):
        mgr = RecordingConcurrencyManager(max_concurrent=1)
        await mgr.acquire()
        assert mgr.available_slots == 0


# ── Semaphore blocking ───────────────────────────────────────────


class TestSemaphoreBlocking:
    async def test_blocks_at_max(self):
        mgr = RecordingConcurrencyManager(max_concurrent=1)
        await mgr.acquire()
        assert mgr.active_count == 1

        # Second acquire should not complete within the timeout
        acquired = False

        async def try_acquire():
            nonlocal acquired
            await mgr.acquire()
            acquired = True

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.05)
        assert not acquired
        assert mgr.queued_count == 1

        # Release to unblock
        await mgr.release()
        await asyncio.sleep(0.05)
        assert acquired
        assert mgr.active_count == 1
        assert mgr.queued_count == 0

        # Clean up
        await mgr.release()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ── Lock atomicity ───────────────────────────────────────────────


class TestLockAtomicity:
    async def test_concurrent_acquire_release_consistency(self):
        """Multiple concurrent acquire/release cycles should leave counts consistent."""
        mgr = RecordingConcurrencyManager(max_concurrent=5)

        async def worker():
            await mgr.acquire()
            await asyncio.sleep(0.01)
            await mgr.release()

        await asyncio.gather(*[worker() for _ in range(10)])
        assert mgr.active_count == 0
        assert mgr.queued_count == 0


# ── Per-pod sub-limit vs the fleet cap ────────────────────────────


class TestPerPodSubLimit:
    """The per-pod ceiling must be settable BELOW the fleet-wide cap.

    They used to be the same number, which coupled two unrelated decisions: any
    raise of the fleet cap to gain throughput across replicas also let ONE pod
    stack that many Chromium+ffmpeg processes inside a single 8Gi memory limit.
    """

    def test_per_pod_defaults_to_the_fleet_cap(self):
        mgr = RecordingConcurrencyManager(max_concurrent=6)
        assert mgr.per_pod_max == 6

    def test_per_pod_can_be_lower_than_the_fleet_cap(self):
        mgr = RecordingConcurrencyManager(max_concurrent=6, per_pod_max=3)
        assert mgr.max_concurrent == 6
        assert mgr.per_pod_max == 3

    def test_per_pod_is_clamped_to_the_fleet_cap(self):
        """A per-pod limit above the fleet cap can never bind, so it is a lie."""
        mgr = RecordingConcurrencyManager(max_concurrent=2, per_pod_max=9)
        assert mgr.per_pod_max == 2

    async def test_the_pod_blocks_at_its_own_limit_not_the_fleet_cap(self):
        """Without Redis there is no fleet view, so the per-pod limit is the gate."""
        mgr = RecordingConcurrencyManager(max_concurrent=6, per_pod_max=1)
        await mgr.acquire()

        acquired = False

        async def second() -> None:
            nonlocal acquired
            await mgr.acquire()
            acquired = True

        task = asyncio.create_task(second())
        await asyncio.sleep(0.05)
        assert not acquired, "per-pod sub-limit did not bind"
        assert mgr.queued_count == 1

        await mgr.release()
        await asyncio.sleep(0.05)
        assert acquired

        await mgr.release()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


# ── Bounded wait ─────────────────────────────────────────────────


class TestQueueWaitTimeout:
    """A saturated fleet must fail the job loudly, not wait past the caller.

    On 2026-09-03 `acquire()` waited unboundedly while the orchestrator's 600s
    client timeout expired, so a capacity shortfall was indistinguishable from a
    recording failure and the only trace was a reaped job labelled "stuck".
    """

    async def test_waiting_past_the_ceiling_raises_queue_saturated(self):
        mgr = RecordingConcurrencyManager(max_concurrent=1, queue_wait_timeout_seconds=0.05)
        await mgr.acquire()

        with pytest.raises(RecordingQueueSaturatedError):
            await mgr.acquire()

        await mgr.release()

    async def test_a_timed_out_waiter_does_not_leak_a_slot(self):
        """The failed waiter must hand nothing back that it never took.

        Releasing a semaphore this call did not acquire would INFLATE the pod's
        limit -- a bug that only shows up as an OOM much later.
        """
        mgr = RecordingConcurrencyManager(max_concurrent=1, queue_wait_timeout_seconds=0.05)
        await mgr.acquire()
        with contextlib.suppress(RecordingQueueSaturatedError):
            await mgr.acquire()

        assert mgr.active_count == 1
        assert mgr.queued_count == 0

        await mgr.release()
        assert mgr.active_count == 0
        # Exactly one slot is free again, not two.
        await mgr.acquire()
        with pytest.raises(RecordingQueueSaturatedError):
            await mgr.acquire()
        await mgr.release()

    async def test_the_error_is_not_a_generic_recording_failure(self):
        """Callers must be able to catch saturation specifically."""
        assert issubclass(RecordingQueueSaturatedError, RuntimeError)
        assert QUEUE_SATURATED_ERROR_PREFIX == "QUEUE_SATURATED"

    async def test_zero_keeps_the_unbounded_behaviour(self):
        """0 disables the bound, so an existing deployment can opt out."""
        mgr = RecordingConcurrencyManager(max_concurrent=1, queue_wait_timeout_seconds=0)
        await mgr.acquire()

        task = asyncio.create_task(mgr.acquire())
        await asyncio.sleep(0.1)
        assert not task.done(), "a 0 ceiling must not time out"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await mgr.release()
