"""Redis liveness lease for in-flight recording / compilation jobs.

A job is dispatched with ``asyncio.create_task`` on the pod that accepted the
request, so exactly one pod runs it. This lease is a cross-pod LIVENESS BEACON
(not a mutex): the owning pod ``SET``s a per-job key with a short TTL and
heartbeats it while the job runs. The age-based reapers (recording + compilation
cleanup) check :meth:`present` before failing/deleting a merely-OLD job — a job
whose lease is still live is running on some pod and must never be reaped. A
crashed pod stops heartbeating, its leases TTL-expire, and the reaper can then
safely fail the genuinely-orphaned job (dispatch reclaim). On shutdown the pod
releases its leases via :meth:`release_all` so reclaim is immediate rather than
waiting out the TTL.

Mirrors ``w2v_orchestrator.services.runs.run_lease``: drops to the raw redis
client via ``_ensure_connected``, and — critically — :meth:`present` falls
CLOSED on a Redis error (returns ``True`` → "assume alive, skip reap") so a
transient Redis blip can never cause the reaper to fail a live job across the
whole fleet at once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# redis-py exceptions don't subclass the builtins, so a mid-flight command drop
# (ConnectionError / TimeoutError / ResponseError) would slip past a builtin-only
# tuple. RedisError is their common base — catch it alongside the builtins.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RedisError)


class JobLease:
    """Per-job liveness lease with a single shared heartbeat loop."""

    def __init__(
        self,
        redis_client: Any | None,
        *,
        prefix: str,
        instance_id: str | None = None,
        ttl_seconds: int = 120,
        # float, not int: this is fed straight to asyncio.sleep(), which takes a
        # float, and tests need sub-second ticks to exercise the heartbeat inside a
        # 1s TTL. ttl_seconds stays int because it becomes Redis `SET ... ex=`.
        heartbeat_seconds: float = 45.0,
        stalled_probe: Callable[[], Awaitable[set[str]]] | None = None,
    ) -> None:
        self._redis = redis_client
        self._prefix = prefix
        self._instance_id = instance_id or uuid.uuid7().hex
        self._ttl = ttl_seconds
        self._heartbeat = heartbeat_seconds
        # Returns the set of job ids that have made NO PROGRESS for the reaper's
        # own staleness window. See _heartbeat_loop for why this exists. Optional:
        # a lease with no probe keeps the pre-2026-08-26 renew-everything behaviour.
        self._stalled_probe = stalled_probe
        self._held: set[str] = set()
        self._lock = asyncio.Lock()
        self._hb_task: asyncio.Task | None = None

    def key(self, job_id: str) -> str:
        """Redis key holding the liveness beacon for *job_id*."""
        return f"{self._prefix}:jobexec:{job_id}"

    def _raw(self) -> Any | None:
        if self._redis is None:
            return None
        ensure = getattr(self._redis, "_ensure_connected", None)
        try:
            return ensure() if callable(ensure) else self._redis
        except Exception as exc:  # noqa: BLE001 -- any connection failure → no lease
            logger.warning("Redis unavailable for job lease %s (%s)", self._prefix, exc)
            return None

    async def acquire(self, job_id: str) -> None:
        """Register *job_id* as live on this pod and set/refresh its lease."""
        async with self._lock:
            self._held.add(job_id)
        self._ensure_heartbeat()
        raw = self._raw()
        if raw is None:
            return
        try:
            await raw.set(self.key(job_id), self._instance_id, ex=self._ttl)
        except _REDIS_ERRORS as exc:
            # Fall open: dispatch proceeds. The reaper falls closed on its own
            # Redis error, so a missing lease under an outage is not catastrophic.
            logger.warning("Job lease acquire failed for %s (%s)", job_id, exc)

    async def present(self, job_id: str) -> bool:
        """True if a live lease exists for *job_id* (held by any pod).

        Falls CLOSED (returns True → "assume alive, skip reap") on a Redis error,
        and returns False when Redis is entirely absent (no liveness info → the
        reaper degrades to pure age-based behaviour).
        """
        raw = self._raw()
        if raw is None:
            return False
        try:
            return await raw.get(self.key(job_id)) is not None
        except _REDIS_ERRORS as exc:
            # Fall CLOSED on a mid-flight blip: assume alive → skip the reap, so a
            # transient Redis error can never mass-fail live jobs across the fleet.
            logger.warning(
                "Job lease presence check failed for %s (%s); assuming alive", job_id, exc
            )
            return True

    async def release(self, job_id: str) -> None:
        """Drop this pod's lease for *job_id* (job reached a terminal state)."""
        async with self._lock:
            self._held.discard(job_id)
        raw = self._raw()
        if raw is None:
            return
        with contextlib.suppress(*_REDIS_ERRORS):
            await raw.delete(self.key(job_id))

    def _ensure_heartbeat(self) -> None:
        if self._hb_task is None or self._hb_task.done():
            self._hb_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"job-lease-hb:{self._prefix}"
            )

    async def _heartbeat_loop(self) -> None:
        """Renew held leases — but NOT for jobs that have stopped making progress.

        This loop used to renew every held lease unconditionally, and that turned
        the reaper into a no-op against the exact failure it exists to catch.

        Measured 2026-08-25 on video-production-service: 18 recording jobs wedged
        on pods that were ALIVE (they did not crash). ``stale_recording_jobs``
        selected all 18 correctly — it keys on ``updated_at``, which
        ``record_progress`` bumps, so those rows were genuinely progress-dead. The
        reaper then skipped every one of them because ``present()`` was True, and
        it was True because THIS loop kept renewing their leases forever. A blind
        signal overrode a correct one. The 18 held slots in a fleet-wide semaphore
        capped at 3, so the service stopped dispatching entirely while still
        answering status polls, and every downstream stage spun to its wall budget.

        The lease's real job is "another pod picked this up, do not race it". Once a
        job has been progress-dead for the reaper's whole window, a live lease is
        not evidence of work — it is this loop asserting something false. So stop
        renewing it: the key TTL-expires within ``ttl_seconds`` and the EXISTING
        reaper then reclaims it with no change to cleanup_service at all.

        ⚠️ The probe MUST measure the same thing the reaper does. It is backed by
        ``stale_recording_jobs`` — the identical resolver, at the identical
        ``stale_job_timeout_seconds`` — precisely so the two can never disagree. A
        heartbeat that gave up earlier than the reaper's window would mass-expire
        healthy long jobs; one that gave up later would restore this bug.

        A probe FAILURE renews as before (fails open): losing the ability to
        measure progress must not become a fleet-wide reap.
        """
        while True:
            await asyncio.sleep(self._heartbeat)
            raw = self._raw()
            if raw is None:
                continue
            async with self._lock:
                job_ids = list(self._held)
            stalled: set[str] = set()
            if self._stalled_probe is not None and job_ids:
                try:
                    stalled = await self._stalled_probe()
                except Exception as exc:  # noqa: BLE001 -- any probe failure → renew as before
                    logger.warning(
                        "Job lease progress probe failed for %s (%s); renewing all held leases",
                        self._prefix,
                        exc,
                    )
                    stalled = set()
            for job_id in job_ids:
                if job_id in stalled:
                    # Log at WARNING, not INFO. The sibling reaper logs its own skip
                    # reason at INFO, and at LOG_LEVEL=WARNING that produced
                    # "Found 18 / Reaped 0" with nothing in between — which reads
                    # like a reaper that ran and found nothing to do.
                    logger.warning(
                        "Job lease NOT renewed for %s: no progress within the reaper's "
                        "staleness window; letting the lease expire so it can be reclaimed",
                        job_id,
                    )
                    async with self._lock:
                        self._held.discard(job_id)
                    continue
                with contextlib.suppress(*_REDIS_ERRORS):
                    await raw.set(self.key(job_id), self._instance_id, ex=self._ttl)

    async def release_all(self) -> None:
        """Shutdown: cancel the heartbeat and release every lease this pod holds."""
        if self._hb_task is not None:
            self._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        async with self._lock:
            job_ids = list(self._held)
            self._held.clear()
        raw = self._raw()
        if raw is None:
            return
        for job_id in job_ids:
            with contextlib.suppress(*_REDIS_ERRORS):
                await raw.delete(self.key(job_id))
