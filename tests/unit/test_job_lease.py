"""Tests for the per-job Redis liveness lease (src/services/job_lease.py)."""

from __future__ import annotations

import asyncio
import contextlib

import fakeredis
import fakeredis.aioredis

from src.services.job_lease import JobLease


def _shared_pair() -> tuple:
    server = fakeredis.FakeServer()
    r1 = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    r2 = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return r1, r2


class TestLeaseVisibility:
    async def test_present_across_pods_then_released(self):
        """A lease taken by one pod is visible to another pod, and clears on release."""
        r1, r2 = _shared_pair()
        owner = JobLease(r1, prefix="recording", instance_id="pod-a", ttl_seconds=60)
        observer = JobLease(r2, prefix="recording", instance_id="pod-b", ttl_seconds=60)

        assert await observer.present("job-1") is False
        await owner.acquire("job-1")
        assert await observer.present("job-1") is True

        await owner.release("job-1")
        assert await observer.present("job-1") is False
        await owner.release_all()

    async def test_ttl_expiry_clears_a_crashed_pods_lease(self):
        """A crashed pod stops heartbeating → its lease TTL-expires → present False."""
        r1, r2 = _shared_pair()
        # heartbeat far in the future so it never refreshes within the test window.
        owner = JobLease(r1, prefix="c", instance_id="a", ttl_seconds=1, heartbeat_seconds=1000)
        observer = JobLease(r2, prefix="c", instance_id="b")

        await owner.acquire("job-x")
        assert await observer.present("job-x") is True

        if owner._hb_task is not None:
            owner._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await owner._hb_task

        await asyncio.sleep(1.2)
        assert await observer.present("job-x") is False

    async def test_release_all_clears_every_held_lease(self):
        r1, _ = _shared_pair()
        lease = JobLease(r1, prefix="recording", instance_id="a", ttl_seconds=60)
        await lease.acquire("j1")
        await lease.acquire("j2")
        await lease.release_all()
        assert await lease.present("j1") is False
        assert await lease.present("j2") is False


class TestLeaseDegraded:
    async def test_no_redis_present_is_false(self):
        """With no Redis there is no liveness info → present() is False (reaper then
        degrades to pure age-based behaviour)."""
        lease = JobLease(None, prefix="recording", instance_id="a")
        await lease.acquire("j1")  # no-op registration, no crash
        assert await lease.present("j1") is False
        await lease.release("j1")
        await lease.release_all()

    async def test_present_mid_flight_error_falls_closed(self):
        """A redis-py ConnectionError raised MID-COMMAND (not a builtin subclass)
        must NOT abort the reaper sweep — present() falls closed (assume alive =
        skip the reap), the conservative value that avoids a wrong reap."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        class _FailGet:
            async def get(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

        lease = JobLease(_FailGet(), prefix="recording", instance_id="a")
        assert await lease.present("j-any") is True

    async def test_acquire_mid_flight_error_does_not_raise(self):
        """A mid-flight Redis error on acquire falls open (dispatch proceeds)."""
        from redis.exceptions import TimeoutError as RedisTimeoutError

        class _FailSet:
            async def set(self, *a, **k):
                raise RedisTimeoutError("dropped mid-flight")

            async def delete(self, *a, **k):
                raise RedisTimeoutError("dropped mid-flight")

        lease = JobLease(_FailSet(), prefix="recording", instance_id="a", heartbeat_seconds=1000)
        await lease.acquire("j1")  # must not raise
        await lease.release_all()  # cleanup path also tolerates the blip


class TestHeartbeatStopsVouchingForDeadJobs:
    """The heartbeat must not keep a lease alive for a job that stopped progressing.

    Regression cover for 2026-08-25 on video-production-service: 18 recording jobs
    wedged on pods that were ALIVE. `stale_recording_jobs` selected all 18 correctly
    (it keys on `updated_at`, which record_progress bumps), but the reaper skipped
    every one because `present()` was True — and it was True only because the
    heartbeat renewed their leases unconditionally. Those 18 held slots in a
    fleet-wide semaphore capped at 3, so the service stopped dispatching while still
    answering status polls, and downstream stages spun to their wall budgets.
    """

    async def test_a_stalled_job_lease_is_left_to_expire(self):
        r1, r2 = _shared_pair()
        owner = JobLease(
            r1,
            prefix="recording",
            instance_id="pod-a",
            ttl_seconds=1,
            heartbeat_seconds=0.05,
            stalled_probe=lambda: _immediately({"stuck"}),
        )
        observer = JobLease(r2, prefix="recording", instance_id="pod-b")

        await owner.acquire("stuck")
        assert await observer.present("stuck") is True

        # Let several heartbeat ticks pass; none should renew, so the 1s TTL wins.
        await asyncio.sleep(1.4)
        assert await observer.present("stuck") is False, (
            "the heartbeat kept renewing a progress-dead job — this is the original bug"
        )
        with contextlib.suppress(Exception):
            await owner.release_all()

    async def test_a_progressing_job_is_still_renewed(self):
        """The negative control. Without this the fix could just never renew anything."""
        r1, r2 = _shared_pair()
        owner = JobLease(
            r1,
            prefix="recording",
            instance_id="pod-a",
            ttl_seconds=1,
            heartbeat_seconds=0.05,
            stalled_probe=lambda: _immediately(set()),  # nothing is stalled
        )
        observer = JobLease(r2, prefix="recording", instance_id="pod-b")

        await owner.acquire("healthy")
        await asyncio.sleep(1.4)  # longer than the TTL: only renewal keeps it alive
        assert await observer.present("healthy") is True, (
            "a live, progressing job lost its lease — this would mass-reap healthy work"
        )
        with contextlib.suppress(Exception):
            await owner.release_all()

    async def test_a_probe_failure_falls_open_and_renews(self):
        """Losing the ability to MEASURE progress must not become a fleet-wide reap."""
        r1, r2 = _shared_pair()

        async def _boom() -> set[str]:
            raise RuntimeError("database unreachable")

        owner = JobLease(
            r1,
            prefix="recording",
            instance_id="pod-a",
            ttl_seconds=1,
            heartbeat_seconds=0.05,
            stalled_probe=_boom,
        )
        observer = JobLease(r2, prefix="recording", instance_id="pod-b")

        await owner.acquire("job-y")
        await asyncio.sleep(1.4)
        assert await observer.present("job-y") is True, (
            "a probe error expired a live lease — must fail open, not closed"
        )
        with contextlib.suppress(Exception):
            await owner.release_all()


async def _immediately(value: set[str]) -> set[str]:
    return value
