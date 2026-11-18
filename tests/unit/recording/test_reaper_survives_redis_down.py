"""The orphan reaper must work with NO Redis — that is the path that needs it most.

#165 added ``_reap_orphaned_slots`` as the ONLY mechanism that reclaims a leaked
recording slot, after three leaked slots wedged production for nine hours. But it
ran solely inside ``_heartbeat_loop`` AFTER that loop's ``raw is None: continue``
bail, and ``_ensure_heartbeat()`` was itself called only under ``if raw is not
None``. So on the local-only path the reaper was dead:

  * ``RecordingConcurrencyManager`` built with no Redis client at all, and
  * ``_try_acquire_global``'s deliberate fall-open on a Redis outage,

both append ``token=None`` PLACEHOLDER slots. A placeholder holds no Redis entry,
so the TTL cannot expire it either — with the reaper off, a leaked local permit
was unreclaimable by ANY mechanism. Once all ``per_pod_max`` permits leaked, every
later ``acquire()`` blocked on the per-pod semaphore, and because
``_ensure_heartbeat()`` sat AFTER that blocking acquire, the wedge locked out its
own repair. Only a pod restart cleared it.

These tests drive ``raw is None`` explicitly. A fake that always hands back a live
redis would leave the fix unexercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

import src.services.recording.concurrency as concurrency_mod
from src.services.recording.concurrency import RecordingConcurrencyManager


def _local_only_mgr(**kw) -> RecordingConcurrencyManager:
    """A manager with NO Redis: every admission is a ``token=None`` placeholder.

    ``slot_ttl_seconds=2`` makes the heartbeat interval ``max(1.0, ttl/2)`` = 1.0s,
    so one tick lands inside a short test.
    """
    kw.setdefault("max_concurrent", 3)
    kw.setdefault("slot_ttl_seconds", 2)
    kw.setdefault("poll_interval_seconds", 0.01)
    return RecordingConcurrencyManager(redis_client=None, **kw)


@pytest.fixture(autouse=True)
def _tiny_grace(monkeypatch):
    monkeypatch.setattr(concurrency_mod, "_ORPHAN_GRACE_SECONDS", 0.01)


class TestReaperOnTheLocalOnlyPath:
    async def test_the_manager_really_is_on_the_raw_is_none_path(self):
        """Guard the premise: if this ever returns a client, the tests below are vacuous."""
        mgr = _local_only_mgr()
        assert mgr._raw() is None
        await mgr.close()

    async def test_a_leaked_local_permit_is_reclaimed_without_redis(self, caplog):
        """The core property. No Redis ⇒ no TTL ⇒ the reaper is the only hope."""
        mgr = _local_only_mgr(per_pod_max=2)

        async def leaks_its_slot() -> None:
            await mgr.acquire("job-leaked")  # returns without ever releasing

        await asyncio.create_task(leaks_its_slot())
        assert mgr.active_count == 1

        with caplog.at_level(logging.ERROR, logger="src.services.recording.concurrency"):
            await asyncio.sleep(1.3)  # one heartbeat tick

        assert mgr.active_count == 0, "a leaked local-only permit was never reclaimed"
        assert any("ORPHANED recording slot" in rec.message for rec in caplog.records), (
            "the reaper never ran on the local-only path"
        )
        await mgr.close()

    async def test_a_fully_wedged_pod_unwedges_itself_without_redis(self):
        """The outage shape: every permit leaked, so the next acquire blocks.

        Under the defect this could never return — the reaper was off (no Redis)
        and the line that starts it sat AFTER the blocking per-pod acquire, so a
        caller could not reach the repair from inside the wedge.
        """
        mgr = _local_only_mgr(per_pod_max=1)

        async def leaks_its_slot() -> None:
            await mgr.acquire("job-wedger")

        await asyncio.create_task(leaks_its_slot())
        assert mgr.active_count == 1

        # A brand-new job. It must block (the only permit is held) and then be
        # rescued by the reaper handing that permit back.
        await asyncio.wait_for(mgr.acquire("job-after-wedge"), timeout=5.0)

        await mgr.release("job-after-wedge")
        await mgr.close()

    async def test_a_live_local_recording_is_never_reaped(self):
        """The safety half: reclamation must not become a duration-based killer."""
        mgr = _local_only_mgr(per_pod_max=2)
        started = asyncio.Event()

        async def long_recording() -> None:
            await mgr.acquire("job-long")
            started.set()
            await asyncio.sleep(30)  # still going across several heartbeats

        task = asyncio.create_task(long_recording())
        await started.wait()

        await asyncio.sleep(1.3)  # a heartbeat tick passes

        assert mgr.active_count == 1, "a live local-only recording was reaped"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await mgr.close()

    async def test_the_heartbeat_starts_before_the_blocking_pod_acquire(self):
        """Ordering, asserted directly rather than only via the wedge above.

        ``_ensure_heartbeat()`` must run BEFORE ``_acquire_pod_slot``. When it ran
        after, a caller that blocked on a saturated pod never started the reaper,
        so the pod's repair was gated on the very thing the wedge prevented.
        """
        mgr = _local_only_mgr(per_pod_max=1, queue_wait_timeout_seconds=0.2)

        async def leaks_its_slot() -> None:
            await mgr.acquire("job-holder")

        await asyncio.create_task(leaks_its_slot())

        # This caller times out waiting for the (held) permit. Even so, by the time
        # it blocked the heartbeat task must already exist.
        mgr._hb_task = None  # prove the NEXT acquire is what starts it
        with contextlib.suppress(concurrency_mod.RecordingQueueSaturatedError):
            await mgr.acquire("job-blocked")

        assert mgr._hb_task is not None, (
            "the reaper was not started before the blocking per-pod acquire"
        )
        await mgr.close()
