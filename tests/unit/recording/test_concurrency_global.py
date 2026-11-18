"""Fleet-wide (Redis-backed) recording concurrency tests.

Two ``RecordingConcurrencyManager`` instances sharing one Redis server model two
replicas. The global cap must hold ACROSS them (not N x cap), and a crashed
holder's slot must auto-release via its TTL.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import fakeredis
import fakeredis.aioredis

import src.services.recording.concurrency as concurrency_mod
from src.services.recording.concurrency import RecordingConcurrencyManager


def _shared_pair() -> tuple:
    """Two async fakeredis clients backed by ONE server (== two pods, one Redis)."""
    server = fakeredis.FakeServer()
    r1 = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    r2 = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
    return r1, r2


async def _acquire_within(mgr: RecordingConcurrencyManager, deadline: float) -> bool:
    """True if *mgr* acquires a slot within *deadline* seconds."""
    task = asyncio.create_task(mgr.acquire())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=deadline)
        return True
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return False


class TestGlobalCapAcrossPods:
    async def test_two_pods_share_one_global_cap(self):
        """cap=2 across 2 pods → total concurrent slots is 2, NOT 2 per pod."""
        r1, r2 = _shared_pair()
        key = "test:recording:slots:a"
        m1 = RecordingConcurrencyManager(
            max_concurrent=2, redis_client=r1, slot_key=key, poll_interval_seconds=0.01
        )
        m2 = RecordingConcurrencyManager(
            max_concurrent=2, redis_client=r2, slot_key=key, poll_interval_seconds=0.01
        )

        # Pod 1 fills BOTH global slots.
        await m1.acquire()
        await m1.acquire()
        assert await r1.zcard(key) == 2

        # Pod 2 must NOT be able to take a third slot (were the cap per-pod, it
        # would happily run 2 more → 4 total).
        acquired = False

        async def try_third() -> None:
            nonlocal acquired
            await m2.acquire()
            acquired = True

        task = asyncio.create_task(try_third())
        await asyncio.sleep(0.1)
        assert not acquired
        assert m2.active_count == 0

        # Free one global slot → pod 2's pending acquire now completes.
        await m1.release()
        await asyncio.sleep(0.1)
        assert acquired
        assert m2.active_count == 1
        # Exactly two slots held fleet-wide (pod1 one + pod2 one).
        assert len(m1._held) + len(m2._held) == 2
        assert await r1.zcard(key) == 2

        await m1.release()
        await m2.release()
        await m1.close()
        await m2.close()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_crashed_holder_slot_auto_releases_via_ttl(self):
        """A crashed pod that never heartbeats leaks nothing: its slot TTL-expires
        and another pod can then acquire."""
        r1, r2 = _shared_pair()
        key = "test:recording:slots:ttl"
        dead = RecordingConcurrencyManager(
            max_concurrent=1,
            redis_client=r1,
            slot_key=key,
            slot_ttl_seconds=1,
            poll_interval_seconds=0.01,
        )
        alive = RecordingConcurrencyManager(
            max_concurrent=1,
            redis_client=r2,
            slot_key=key,
            slot_ttl_seconds=1,
            poll_interval_seconds=0.01,
        )

        await dead.acquire()
        assert await r1.zcard(key) == 1

        # Simulate a HARD crash: stop the heartbeat so the slot is never refreshed
        # and never explicitly released.
        if dead._hb_task is not None:
            dead._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await dead._hb_task

        # While the (soon-stale) slot is still within TTL, the other pod is blocked.
        assert await _acquire_within(alive, 0.15) is False

        # After the TTL lapses the crashed holder's slot is pruned on the next
        # acquire attempt, so the live pod gets in.
        await asyncio.sleep(1.1)
        assert await _acquire_within(alive, 1.0) is True
        assert alive.active_count == 1

        await alive.release()
        await alive.close()


class TestLocalFallback:
    async def test_no_redis_is_per_pod_only(self):
        """Without Redis the limiter is the classic per-pod semaphore (unchanged)."""
        m = RecordingConcurrencyManager(max_concurrent=1)
        await m.acquire()
        assert m.active_count == 1
        assert await _acquire_within(m, 0.05) is False
        await m.release()
        await m.close()

    async def test_mid_flight_redis_error_falls_open(self):
        """A redis-py ConnectionError raised MID-COMMAND (does not subclass the
        builtins) must not wedge acquire() — it falls open to the per-pod path."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        class _MidFlightFail:
            async def zremrangebyscore(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

            async def zadd(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

            async def zrank(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

            async def zrem(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

        m = RecordingConcurrencyManager(
            max_concurrent=1, redis_client=_MidFlightFail(), poll_interval_seconds=0.01
        )
        # Must complete (fall open) rather than raise or hang.
        await asyncio.wait_for(m.acquire(), timeout=1.0)
        assert m.active_count == 1
        await m.release()
        await m.close()


class _FlakyRedis:
    """fakeredis proxy whose commands can be switched to raise on demand.

    Models a TRANSIENT blip rather than an outage: the same client is healthy
    before and after, which is exactly the shape that made the token bug below
    permanent instead of self-healing.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.fail = False

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)

        async def _call(*args, **kwargs):
            if self.fail:
                from redis.exceptions import ConnectionError as RedisConnectionError

                raise RedisConnectionError("transient blip")
            return await attr(*args, **kwargs)

        return _call


class TestFallOpenAdmissionOwnsNoSlot:
    async def test_fall_open_release_does_not_free_a_running_jobs_slot(self):
        """The bug: a blip-admitted job's release ZREM'd a RUNNING job's token.

        `_held` was a plain FIFO of real tokens, but a fall-open admission appended
        nothing — so `release()`'s `popleft()` handed it somebody else's live token.
        One transient blip therefore freed a slot that was still in use, permanently
        over-admitting the fleet (and raising the odds of the next blip).
        """
        server = fakeredis.FakeServer()
        inner = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        flaky = _FlakyRedis(inner)
        key = "test:recording:slots:fallopen"
        m = RecordingConcurrencyManager(
            max_concurrent=2, redis_client=flaky, slot_key=key, poll_interval_seconds=0.01
        )

        # Job A is admitted normally and holds a REAL Redis token.
        await m.acquire()
        assert await inner.zcard(key) == 1

        # Job B is admitted through a transient blip — it holds no Redis slot.
        flaky.fail = True
        await m.acquire()
        flaky.fail = False
        assert m.active_count == 2
        assert await inner.zcard(key) == 1  # still only A's token

        # B finishes first. Its release must remove NOTHING: A is still recording.
        await m.release()
        assert await inner.zcard(key) == 1, "B's release consumed A's live slot"

        # A finishes; now the slot is genuinely free.
        await m.release()
        assert await inner.zcard(key) == 0
        await m.close()

    async def test_heartbeat_never_writes_a_placeholder_into_redis(self):
        """A placeholder must never reach ZADD — it would become a real occupant.

        This is the assertion that bites: the heartbeat refreshes every held entry,
        and a `None` in that list is serialised as the member "None", CREATING a
        ZSET entry that no job owns and no release ever removes. The fleet would
        then run one slot short forever, per pod, per blip.
        """
        server = fakeredis.FakeServer()
        inner = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        flaky = _FlakyRedis(inner)
        key = "test:recording:slots:placeholder"
        m = RecordingConcurrencyManager(
            max_concurrent=2,
            redis_client=flaky,
            slot_key=key,
            poll_interval_seconds=0.01,
            slot_ttl_seconds=2,  # → heartbeat interval 1.0s
        )
        await m.acquire()  # a REAL token, whose deadline the heartbeat must refresh
        flaky.fail = True
        await m.acquire()  # a blip admission, holding no token
        flaky.fail = False
        token = (await inner.zrange(key, 0, -1))[0]
        before = await inner.zscore(key, token)  # ty: ignore[no-matching-overload]  # FakeRedis vs redis>=8 protocol-typed overloads

        await asyncio.sleep(1.3)  # let one heartbeat cycle run

        # Two failure modes, one cycle. Passing `None` to ZADD either materialises a
        # phantom member "None" that no job owns and no release removes (the fleet
        # then runs a slot short forever), or raises a DataError that kills the
        # heartbeat task outright — after which the REAL token stops being refreshed
        # and TTL-expires out from under a running job. Assert against both.
        assert await inner.zcard(key) == 1, "the heartbeat materialised a phantom slot"
        assert await inner.zscore(key, token) > before, "the heartbeat stopped refreshing"  # ty: ignore[no-matching-overload]

        await m.release()
        await m.release()
        await m.close()
        assert await inner.zcard(key) == 0


class TestStallBlindSpotIsAtLeastVisible:
    async def test_held_without_progress_is_reported_even_with_an_empty_queue(self):
        """The probe cannot see a wedge under trickle traffic; /health can.

        `stalled` stays False here on purpose — nothing is queued, and this signal
        cannot tell a wedge from one legitimately long recording, so wiring it to
        the kubelet would restart healthy pods mid-render. It is reported, not
        judged.
        """
        m = RecordingConcurrencyManager(max_concurrent=1, poll_interval_seconds=0.01)
        idle = m.stall_status(1800.0)
        assert idle["held_without_progress_seconds"] is None  # nothing held

        await m.acquire()
        await asyncio.sleep(0.05)
        held = m.stall_status(1800.0)
        assert held["stalled"] is False  # empty queue → not judged stalled
        assert held["held_without_progress_seconds"] is not None
        assert held["held_without_progress_seconds"] >= 0.05

        await m.release()
        assert m.stall_status(1800.0)["held_without_progress_seconds"] is None
        await m.close()


class TestTheBlindSpotIsAlertable:
    """The gauge and the WARNING are the whole intervention — and neither restarts."""

    async def test_gauge_is_published_on_every_stall_check(self, monkeypatch):
        published: list[float] = []
        import src.middleware.metrics as metrics

        monkeypatch.setattr(metrics, "set_held_without_progress_seconds", published.append)

        m = RecordingConcurrencyManager(max_concurrent=1, poll_interval_seconds=0.01)
        m.stall_status(1800.0)
        assert published == [0.0], "an idle manager must still publish, or the series has holes"

        await m.acquire()
        await asyncio.sleep(0.05)
        m.stall_status(1800.0)
        assert len(published) == 2
        assert published[1] >= 0.05

        await m.release()
        m.stall_status(1800.0)
        assert published[2] == 0.0
        await m.close()

    async def test_a_long_hold_warns_but_never_reports_stalled(self, caplog):
        """The whole point: loud, and still alive.

        An empty queue means `stalled` stays False no matter how long the hold — a
        long legitimate recording reads identically to a wedge here, and a liveness
        probe that cannot tell them apart kills healthy pods mid-render.
        """
        m = RecordingConcurrencyManager(max_concurrent=1, poll_interval_seconds=0.01)
        await m.acquire()
        m._last_progress -= 5000.0  # simulate a long hold with nothing admitted/released

        with caplog.at_level(logging.WARNING, logger="src.services.recording.concurrency"):
            status = m.stall_status(1800.0)

        assert status["stalled"] is False, "an empty queue must never fail the liveness probe"
        held = status["held_without_progress_seconds"]
        assert held is not None, "a held slot must report a duration, not None"
        assert held > 1800.0
        assert any("NO admission or release" in r.message for r in caplog.records)
        await m.release()
        await m.close()

    async def test_a_hold_under_the_threshold_is_silent(self, caplog):
        m = RecordingConcurrencyManager(max_concurrent=1, poll_interval_seconds=0.01)
        await m.acquire()
        with caplog.at_level(logging.WARNING, logger="src.services.recording.concurrency"):
            m.stall_status(1800.0)
        assert not [r for r in caplog.records if "NO admission or release" in r.message]
        await m.release()
        await m.close()

    async def test_the_warning_is_throttled(self, caplog):
        """The liveness probe calls this on its own cadence; unthrottled it floods."""
        m = RecordingConcurrencyManager(max_concurrent=1, poll_interval_seconds=0.01)
        await m.acquire()
        m._last_progress -= 5000.0
        with caplog.at_level(logging.WARNING, logger="src.services.recording.concurrency"):
            for _ in range(5):
                m.stall_status(1800.0)
        hits = [r for r in caplog.records if "NO admission or release" in r.message]
        assert len(hits) == 1, f"expected one throttled line, got {len(hits)}"
        await m.release()
        await m.close()


class TestASlotNamesItsOwnerAndItsHolder:
    """Owner + task identity: what makes a leak decidable instead of merely timed."""

    async def test_release_frees_the_callers_own_slot_under_out_of_order_completion(self):
        """The exact form of what #159 could only mitigate.

        Two normally-admitted jobs, the SECOND finishing first. Without an owner the
        release could only guess, and the guess took the first job's live token —
        freeing a slot that was still recording. With an owner there is no guess.
        """
        server = fakeredis.FakeServer()
        r = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        key = "test:recording:slots:owned"
        m = RecordingConcurrencyManager(
            max_concurrent=2, redis_client=r, slot_key=key, poll_interval_seconds=0.01
        )
        await m.acquire("job-A")
        token_a = next(slot.token for slot in m._held if slot.owner == "job-A")
        await m.acquire("job-B")
        assert await r.zcard(key) == 2

        await m.release("job-B")

        assert await r.zrange(key, 0, -1) == [token_a], "B's release took A's live token"
        assert m.active_count == 1
        await m.release("job-A")
        assert await r.zcard(key) == 0
        await m.close()

    async def test_an_unpaired_release_changes_nothing(self):
        """A release for a slot we do not hold must not inflate the per-pod permit.

        The old code released the semaphore regardless, raising the permit count
        above `per_pod_max` — more Chromium+ffmpeg than the memory envelope
        survives. `_prepare_retry` documents this as the phantom-release hazard.
        """
        m = RecordingConcurrencyManager(max_concurrent=1, per_pod_max=1, poll_interval_seconds=0.01)
        await m.acquire("job-real")

        await m.release("job-ghost")  # never acquired
        assert m.active_count == 1, "an unpaired release must not decrement active"

        # The cap must still bind: with the only permit held, a second acquire waits.
        assert await _acquire_within(m, 0.05) is False
        await m.release("job-real")
        await m.close()

    async def test_local_only_acquire_and_release_still_pair(self):
        """No Redis at all — every admission must still be recorded and reclaimable.

        Regression pin: `_held` was populated only on the Redis path, so with no
        Redis every release was unpaired and (once unpaired became a correct
        no-op) the permit was never handed back — the pod wedged on its first job.
        """
        m = RecordingConcurrencyManager(max_concurrent=1, per_pod_max=1, poll_interval_seconds=0.01)
        await m.acquire("job-1")
        assert m.active_count == 1
        await m.release("job-1")
        assert m.active_count == 0
        # The permit came back: a second job can run.
        assert await _acquire_within(m, 0.2) is True
        await m.release()
        await m.close()


class TestAnOrphanedSlotIsReclaimed:
    """The wedge, closed without a threshold: a slot held by a finished task."""

    @staticmethod
    def _mgr(key: str, r) -> RecordingConcurrencyManager:
        return RecordingConcurrencyManager(
            max_concurrent=2,
            redis_client=r,
            slot_key=key,
            poll_interval_seconds=0.01,
            slot_ttl_seconds=2,  # → heartbeat every 1.0s
        )

    async def test_a_slot_whose_owning_task_is_gone_is_reclaimed(self, monkeypatch, caplog):
        """This is what the TTL could never do.

        The heartbeat renews every held slot unconditionally, so a leaked lease was
        refreshed forever — three of them wedged recording for nine hours. A slot
        whose owner has finished is held by NOBODY, which is decidable now.
        """
        monkeypatch.setattr(concurrency_mod, "_ORPHAN_GRACE_SECONDS", 0.01)
        server = fakeredis.FakeServer()
        r = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        key = "test:recording:slots:orphan"
        m = self._mgr(key, r)

        async def leaks_its_slot() -> None:
            await m.acquire("job-leaked")  # returns without ever releasing

        await asyncio.create_task(leaks_its_slot())
        assert await r.zcard(key) == 1
        assert m.active_count == 1

        with caplog.at_level(logging.ERROR, logger="src.services.recording.concurrency"):
            await asyncio.sleep(1.3)  # one heartbeat tick

        assert await r.zcard(key) == 0, "the leaked fleet slot was never reclaimed"
        assert m.active_count == 0
        assert any("ORPHANED recording slot" in rec.message for rec in caplog.records)
        # And the per-pod permit came back, so the pod is not wedged.
        assert await _acquire_within(m, 0.3) is True
        await m.release()
        await m.close()

    async def test_a_live_long_running_holder_is_never_reclaimed(self, monkeypatch):
        """The safety property — the whole reason this is a task check, not a timer.

        A legitimately long recording sits far past any grace with no admissions or
        releases anywhere. A duration-based rule would reap it and kill the render;
        an owner-liveness rule cannot, because its task is still running.
        """
        monkeypatch.setattr(concurrency_mod, "_ORPHAN_GRACE_SECONDS", 0.01)
        server = fakeredis.FakeServer()
        r = fakeredis.aioredis.FakeRedis(server=server, decode_responses=True)
        key = "test:recording:slots:longjob"
        m = self._mgr(key, r)

        started = asyncio.Event()

        async def long_recording() -> None:
            await m.acquire("job-long")
            started.set()
            await asyncio.sleep(30)  # still going across several heartbeats

        task = asyncio.create_task(long_recording())
        await started.wait()

        await asyncio.sleep(1.3)  # a heartbeat tick passes

        assert await r.zcard(key) == 1, "a live recording's slot was reaped"
        assert m.active_count == 1

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await m.close()
