"""Concurrency control for recording subprocesses.

FLEET-WIDE (multi-replica) recording concurrency cap.

Recording subprocesses (Chromium + ffmpeg) are memory-heavy, so the ceiling
matters. A plain per-pod ``asyncio.Semaphore`` only caps ONE replica — at N
replicas the real ceiling is ``N x max_concurrent`` → ``N x RAM`` → OOM. This
limiter enforces a SINGLE global cap across every replica via a Redis
sorted-set "distributed semaphore" with TTL-based auto-release: each held slot
is a member scored with an expiry deadline, so a crashed holder's slot frees
itself once its score lapses (no leaked slots). A per-pod ``asyncio.Semaphore``
sub-limit is kept as well, so one replica can never exceed the global ceiling
locally either.

Degrades gracefully: with no Redis client (or a Redis outage) it behaves
exactly like the old per-pod semaphore — the ``acquire()`` / ``release()``
interface and every counter/stat are preserved, so the fallback is transparent.

The admit uses the race-free "add-self, check-rank, remove-if-over" pattern on
the raw redis client (``ZADD`` / ``ZRANK`` / ``ZREMRANGEBYSCORE``); no Lua/eval
is needed, so it runs unchanged on fakeredis in tests. Following
``w2v_orchestrator.services.runs.run_lease``, it drops to the raw client via
``_ensure_connected`` and falls OPEN on a Redis error (proceed local-only rather
than wedge every recording).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import Any, TypedDict

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# Grace before a held slot whose owning task has finished is treated as orphaned.
# This is a RACE window, not a wedge threshold: `release()` runs in the owner's own
# `finally`, so between "task done" and "slot released" there are microseconds of
# legitimate overlap. Five seconds is orders of magnitude above that and orders of
# magnitude below any recording. It is deliberately NOT the duration threshold this
# module refuses to invent (see `stall_status`) — that one needs a measured
# distribution because it tries to guess whether work is progressing. This one needs
# no measurement at all, because a finished task is not doing work.
_ORPHAN_GRACE_SECONDS = 5.0


@dataclass
class _HeldSlot:
    """One admitted job's claim on a slot, with the identity to reclaim it.

    ``token`` is the Redis sorted-set member, or ``None`` for a job admitted through
    the fall-open path or with no Redis at all (neither holds a fleet slot).

    ``owner`` is the job id. Without it ``release()`` could only guess which entry to
    consume, and the guess was wrong in a way that freed a RUNNING job's slot.

    ``task`` is a weak reference to the asyncio task that acquired the slot, and it
    is what makes a leak DECIDABLE. A slot held by a live task is work in progress,
    however long it runs; a slot held by a task that has finished or been
    garbage-collected is held by nobody. That distinction needs no threshold and no
    traffic measurement — which is why this exists rather than a timer.
    """

    token: str | None
    owner: str | None
    acquired_at: float
    task: weakref.ref[asyncio.Task[Any]] | None = field(default=None, repr=False)

    def is_orphaned(self, now: float) -> bool:
        """True when no live task is still holding this slot."""
        if now - self.acquired_at < _ORPHAN_GRACE_SECONDS:
            return False
        if self.task is None:
            return False  # acquired outside a task (sync context): cannot judge
        task = self.task()
        return task is None or task.done()


# redis-py's own exceptions (ConnectionError / TimeoutError / ResponseError) do NOT
# subclass the builtin OSError/ConnectionError/TimeoutError, so a MID-FLIGHT command
# failure would slip past a builtin-only tuple. RedisError is their common base;
# catch it alongside the builtins everywhere a raw Redis command is issued.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RedisError)


class RecordingQueueSaturatedError(RuntimeError):
    """Raised when `acquire()` gives up waiting for a recording slot.

    DELIBERATELY its own type, and it carries its own textual marker, because the
    single most expensive property of the 2026-09-03 collapse was that saturation
    was INDISTINGUISHABLE from a recording failure: jobs waited unboundedly, the
    caller timed out at 600s, and the only trace was a reaped job whose error said
    it was "stuck". A capacity problem must not be diagnosable only as a recorder
    bug -- so callers can catch this exactly, and the message a job is failed with
    starts with `QUEUE_SATURATED`.
    """


# Prefix on the job's stored error_message. Grep-able, and never emitted by any
# recorder path -- so `error_message LIKE 'QUEUE_SATURATED%'` cleanly separates
# "the fleet had no capacity" from "the recording itself broke".
QUEUE_SATURATED_ERROR_PREFIX = "QUEUE_SATURATED"


class ExecutorStallStatus(TypedDict):
    """Structured report from :pymeth:`RecordingConcurrencyManager.stall_status`."""

    stalled: bool
    queued: int
    active: int
    available_slots: int
    seconds_since_progress: float | None
    threshold_seconds: float
    # How long slots have been held with ZERO admissions or releases, reported
    # regardless of whether anything is queued. `stalled` deliberately does not
    # read this: it cannot tell a wedged executor from one legitimately long
    # recording, and a liveness probe that confuses those restarts healthy pods
    # mid-render. It exists so the one state the probe is blind to — a wedged pod
    # under traffic too thin to keep the queue non-empty — is at least VISIBLE on
    # /health instead of being indistinguishable from idle. None when nothing is
    # held. See `stall_status` for why this is measured but not judged.
    held_without_progress_seconds: float | None


class RecordingConcurrencyManager:
    """Fleet-wide (Redis-backed) recording slot limiter with per-pod sub-limit."""

    def __init__(
        self,
        max_concurrent: int = 3,
        *,
        per_pod_max: int | None = None,
        queue_wait_timeout_seconds: float = 0.0,
        redis_client: Any | None = None,
        instance_id: str | None = None,
        slot_key: str = "video_production:recording:global_slots",
        slot_ttl_seconds: int = 900,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        self._max_concurrent = max_concurrent
        # The per-pod sub-limit is now SEPARATE from the fleet cap. It used to be
        # the same number, which quietly coupled the two: raising the fleet cap to
        # get throughput across replicas also let a SINGLE pod stack that many
        # Chromium+ffmpeg processes inside one 8Gi memory limit, which is how this
        # pod has been OOM-killed before (at 2Gi, then again at 4Gi). Clamped to
        # the fleet cap because a per-pod limit above it can never bind.
        self._per_pod_max = min(max_concurrent, per_pod_max or max_concurrent)
        self._semaphore = asyncio.Semaphore(self._per_pod_max)
        # <= 0 means the pre-2026-09-04 unbounded wait.
        self._queue_wait_timeout = queue_wait_timeout_seconds
        # Throttle for the "still waiting" log so a saturated fleet reports itself
        # without one line per 0.5s poll per waiter.
        self._wait_log_interval = 30.0
        self._last_wait_log = 0.0
        self._active_count = 0
        self._queued_count = 0
        self._lock = asyncio.Lock()
        self._redis = redis_client
        self._instance_id = instance_id or uuid.uuid7().hex
        self._slot_key = slot_key
        self._slot_ttl = slot_ttl_seconds
        self._poll = poll_interval_seconds
        # One :class:`_HeldSlot` per ADMITTED job, on EVERY admission path, in
        # admission order. Each names its owner and the task holding it, so a
        # release removes the caller's OWN slot and a leak is decidable rather than
        # merely suspected.
        self._held: deque[_HeldSlot] = deque()
        self._hb_task: asyncio.Task | None = None
        # -- Executor stall detection (see `stall_status`) ------------------
        # Monotonic clock of the last FORWARD PROGRESS: a slot admitted or a
        # slot released. Not "a job completed" -- a heartbeat tied to completed
        # jobs cannot tell an idle service from a wedged one.
        self._last_progress = time.monotonic()
        # Monotonic clock of the moment the wait queue last went 0 -> non-empty,
        # or None while nothing is waiting. Waiting work is what makes a stall
        # *observable*; with an empty queue the executor is idle, not broken.
        self._queue_nonempty_since: float | None = None
        # Throttle for the held-without-progress WARNING below. The liveness probe
        # calls `stall_status` on its own cadence, so an unthrottled line here
        # would repeat every probe interval for the whole duration of a wedge.
        self._last_hold_warn = 0.0

    # -- Redis plumbing -----------------------------------------------------

    def _raw(self) -> Any | None:
        """Return the raw redis client, or None when Redis is absent/unavailable.

        Mirrors ``run_lease.raw_redis``: drop through the w2v_common ``RedisClient``
        wrapper (whose ``_ensure_connected`` raises on an outage) to the native
        client, whose ``ZADD``/``ZRANK`` are used directly. A raw fakeredis passed
        by tests has no ``_ensure_connected`` and is returned as-is.
        """
        if self._redis is None:
            return None
        ensure = getattr(self._redis, "_ensure_connected", None)
        try:
            return ensure() if callable(ensure) else self._redis
        except Exception as exc:  # noqa: BLE001 -- any connection failure → local-only
            logger.warning("Redis unavailable for global concurrency (%s); local-only", exc)
            return None

    def _new_slot(self, token: str | None, owner: str | None) -> _HeldSlot:
        """Build a held-slot record, capturing the task that is acquiring it.

        The task reference is taken here rather than passed in, so it is captured for
        EVERY caller — including ones that pass no ``owner``. The leak check must not
        depend on a caller remembering to identify itself.
        """
        task: weakref.ref[asyncio.Task[Any]] | None = None
        try:
            # Narrow BEFORE taking the ref. `current_task()` is `Task | None`, and
            # `weakref.ref` of that is `ref[Task | None]`, which is not assignable
            # to `ref[Task] | None` because `weakref.ref` is invariant in its
            # parameter. Splitting the None case out fixes the type at the
            # declaration rather than suppressing the checker over it.
            current = asyncio.current_task()
        except RuntimeError:  # no running loop
            current = None
        if current is not None:
            task = weakref.ref(current)
        return _HeldSlot(token=token, owner=owner, acquired_at=time.monotonic(), task=task)

    async def _try_acquire_global(self, raw: Any, owner: str | None = None) -> bool:
        """Attempt to admit one global slot. True if admitted (token retained)."""
        now = time.time()
        token = f"{self._instance_id}:{uuid.uuid7().hex}"
        try:
            # Auto-release crashed/expired holders, then admit via add-self/check-rank.
            await raw.zremrangebyscore(self._slot_key, "-inf", now)
            await raw.zadd(self._slot_key, {token: now + self._slot_ttl})
            rank = await raw.zrank(self._slot_key, token)
            if rank is not None and rank < self._max_concurrent:
                async with self._lock:
                    self._held.append(self._new_slot(token, owner))
                return True
            await raw.zrem(self._slot_key, token)
            return False
        except _REDIS_ERRORS as exc:
            # Fall OPEN on a mid-flight Redis blip (same as a full outage): don't
            # wedge recordings — the per-pod semaphore still caps this replica.
            # Record a PLACEHOLDER so `_held` still has one entry for this admission:
            # this job holds no Redis slot, and its eventual release must therefore
            # remove nothing rather than consuming a running job's token.
            logger.warning("Global slot acquire failed (%s); proceeding local-only", exc)
            async with self._lock:
                self._held.append(self._new_slot(None, owner))
            return True

    # -- Public acquire/release --------------------------------------------

    def _publish_queue_depth(self) -> None:
        """Export the wait-queue depth. Called on every change, never per poll.

        `stats()` has always carried `queued`, and `/health` has always returned
        it — but nothing logged or exported it, so during the 2026-09-03 collapse
        the queue grew into the hundreds with no signal anywhere except the
        *symptom* (wall times climbing 15.4s → 1212.3s). A depth that exists only
        inside a health payload nobody polls is not observability.
        """
        try:
            from src.middleware.metrics import set_queued_jobs

            set_queued_jobs(self._queued_count)
        except Exception:
            logger.debug("Could not publish recording queue depth", exc_info=True)

    def _publish_hold_gauge(self, held: float | None) -> None:
        """Export held-without-progress seconds; 0 when nothing is held.

        Exported rather than only returned, because the shape this measures is
        invisible to the liveness probe (see `stall_status`) and therefore
        invisible to anything that only reacts to a failing probe. A gauge is
        also what makes the threshold question ANSWERABLE: nobody can pick a safe
        restart threshold for this without first seeing the distribution of
        legitimate hold times under real traffic, and nothing was recording it.
        """
        try:
            from src.middleware.metrics import set_held_without_progress_seconds

            set_held_without_progress_seconds(held or 0.0)
        except Exception:
            logger.debug("Could not publish held-without-progress gauge", exc_info=True)

    def _warn_long_hold(self, held: float, threshold: float) -> None:
        """Throttled WARNING for slots held far past any plausible progress gap.

        Reuses the caller's stall threshold rather than inventing a second number:
        "held this long with nothing admitted or released" is the same duration the
        probe already treats as wedge-worthy when work is queued. The difference is
        that this fires with an EMPTY queue too — the case the probe cannot see —
        and it only logs. It must never escalate to a restart on its own: a long
        legitimate recording produces an identical reading, and a liveness probe
        that cannot tell them apart kills healthy pods mid-render.
        """
        now = time.monotonic()
        if now - self._last_hold_warn < self._wait_log_interval:
            return
        self._last_hold_warn = now
        logger.warning(
            "Recording slots held %.0fs with NO admission or release (threshold %.0fs, "
            "%d active, %d queued). With an empty queue the liveness probe reports "
            "healthy by design, so this line is the only signal for a wedge under "
            "thin traffic — check whether a job is genuinely still recording.",
            held,
            threshold,
            self._active_count,
            self._queued_count,
        )

    def _log_wait(self, waited: float) -> None:
        """Throttled WARNING while a job is stuck behind a full fleet.

        Throttled to `_wait_log_interval`, NOT emitted per poll: the poll runs
        every 0.5s per waiter, so an unthrottled line here would be a log flood
        exactly when the service is least able to afford one.
        """
        now = time.monotonic()
        if now - self._last_wait_log < self._wait_log_interval:
            return
        self._last_wait_log = now
        logger.warning(
            "Recording queue saturated: %d waiting, %d active, fleet cap %d "
            "(this waiter has waited %.1fs)",
            self._queued_count,
            self._active_count,
            self._max_concurrent,
            waited,
        )

    async def _acquire_pod_slot(self, deadline: float | None) -> None:
        """Take the per-pod sub-limit, honouring the queue-wait deadline.

        Bounded here as well as around the Redis poll: with the fleet cap at 6 and
        the per-pod cap at 3, a burst aimed at one replica blocks HERE first, and
        an unbounded wait at this step would silently defeat the whole bound.
        """
        if deadline is None:
            await self._semaphore.acquire()
            return
        remaining = deadline - time.monotonic()
        msg = (
            f"no per-pod recording slot within {self._queue_wait_timeout:.0f}s "
            f"(per-pod cap {self._per_pod_max})"
        )
        if remaining <= 0:
            raise RecordingQueueSaturatedError(msg)
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=remaining)
        except TimeoutError as exc:
            raise RecordingQueueSaturatedError(msg) from exc

    async def acquire(self, owner: str | None = None) -> None:
        """Acquire a recording slot, waiting if the fleet is at capacity.

        Pass ``owner`` (the job id) so the matching ``release(owner)`` reclaims THIS
        job's slot instead of guessing. It is optional only so callers with no
        natural id keep working; the orphan check does not depend on it, because a
        caller that forgets to identify itself must not thereby become
        unreclaimable.

        Raises:
            RecordingQueueSaturatedError: when `queue_wait_timeout_seconds` is set
                (> 0) and elapses first. An UNBOUNDED wait here is what turned a
                capacity shortfall into a silent client-side timeout with the job
                still nominally alive; failing loudly and attributably, with time
                still on the caller's clock, beats waiting past their deadline.
        """
        async with self._lock:
            if self._queued_count == 0:
                # Start the stall clock fresh on the 0 -> 1 transition. Without
                # this, a service idle overnight would look "stalled" the
                # instant its first job arrived, because `_last_progress` would
                # still hold last night's timestamp.
                self._queue_nonempty_since = time.monotonic()
            self._queued_count += 1
        self._publish_queue_depth()
        started = time.monotonic()
        deadline = started + self._queue_wait_timeout if self._queue_wait_timeout > 0 else None
        secured = False
        pod_slot = False
        try:
            # Started BEFORE the blocking per-pod acquire, and NOT gated on Redis.
            # Both properties are load-bearing, and both were wrong:
            #
            #   * Gated on `raw is not None`, the maintenance loop never started on
            #     the local-only path -- which is the exact path that appends
            #     `token=None` placeholders (`_try_acquire_global`'s fall-open and
            #     the no-Redis branch below). A placeholder holds no Redis entry, so
            #     no TTL can expire it either: with the reaper off, a leaked local
            #     permit was unreclaimable by ANY mechanism.
            #   * Placed after `_acquire_pod_slot`, the start was unreachable once
            #     the pod was already wedged: every caller blocked on the semaphore
            #     and none of them ever reached the line that starts the reaper. The
            #     wedge locked out its own repair.
            #
            # Starting it here costs one idle task per manager on a pod that never
            # sees Redis, and buys the local path the reclamation it could not
            # otherwise have. Reclaiming a leak must never depend on the thing that
            # is down.
            self._ensure_heartbeat()
            await self._acquire_pod_slot(deadline)  # per-pod sub-limit
            pod_slot = True
            raw = self._raw()
            if raw is not None:
                # Poll for a free fleet-wide slot. An asyncio.Event can't work here:
                # the slot is freed by a DIFFERENT pod (or by TTL expiry in Redis),
                # so there is nothing in-process to wait on — Redis polling is the
                # only cross-replica signal.
                while not await self._try_acquire_global(raw, owner):
                    self._log_wait(time.monotonic() - started)
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RecordingQueueSaturatedError(
                            f"no fleet recording slot within {self._queue_wait_timeout:.0f}s "
                            f"(fleet cap {self._max_concurrent}, {self._queued_count} waiting)"
                        )
                    await asyncio.sleep(self._poll)
            else:
                # No Redis: this pod is capped by the per-pod semaphore alone and
                # there is no fleet token to hold. Record the admission ANYWAY.
                # `_held` must carry one entry per admitted job on EVERY path, not
                # just the Redis one — `release(owner)` finds the caller's slot here,
                # and `_reap_orphaned_slots` can only reclaim what it can see.
                # Leaving this path empty made every local-only release unpaired,
                # and (once an unpaired release became correctly a no-op) the permit
                # was never handed back and the pod wedged on its first job. Caught
                # by the existing suite, not by inspection.
                async with self._lock:
                    self._held.append(self._new_slot(None, owner))
            secured = True
        finally:
            async with self._lock:
                self._queued_count -= 1
                if self._queued_count == 0:
                    self._queue_nonempty_since = None
                if secured:
                    self._active_count += 1
                    self._last_progress = time.monotonic()
            # Cancelled/failed while waiting for a global slot: hand back the
            # per-pod slot we already took so it isn't leaked (would otherwise
            # wedge this replica's sub-limit down by one). Guarded on `pod_slot`
            # now that the per-pod acquire can itself time out — releasing a
            # semaphore this call never took would INFLATE the pod's limit.
            if not secured and pod_slot:
                self._semaphore.release()
            self._publish_queue_depth()

        # ONE line per admitted job that actually had to wait — not per poll, and
        # nothing at all on an uncontended fleet. This is the number that explains
        # a slow render: reported wall time minus this is the real work.
        waited = time.monotonic() - started
        if waited >= 1.0:
            logger.info(
                "Recording slot acquired after %.1fs queued (%d still waiting, %d active/%d)",
                waited,
                self._queued_count,
                self._active_count,
                self._max_concurrent,
            )

    def _take_held_locked(self, owner: str | None = None) -> tuple[bool, str | None]:
        """Consume this caller's held entry; return ``(matched, token)``. Caller holds `_lock`.

        With an ``owner`` this is EXACT: the job's own slot is removed and its own
        token returned, which is the only form correct under out-of-order release.
        The heuristic below is the fallback for owner-less callers, and it is a
        mitigation rather than a fix — the reasoning is kept because it explains the
        direction of failure that remains for them.

        Releases arrive out of order — a short job admitted after a long one
        finishes first — and this deque carries no per-job identity, so it cannot
        know WHICH entry the releasing job owns. Real tokens are fungible (any ZREM
        frees exactly one slot), so among them order is irrelevant. A placeholder is
        not fungible with a real token: it represents an admission that holds no
        Redis slot at all.

        Popping blindly from the left therefore had a direction of failure. A
        blip-admitted job releasing while a normally-admitted job was still
        recording took the RUNNING job's live token, freeing its fleet slot
        mid-recording — a permanent, self-amplifying cap violation from one
        transient Redis error.

        Draining placeholders first inverts that. The worst case becomes holding an
        already-finished job's token until the placeholder-holder releases: a
        transient UNDER-admission that resolves itself, instead of an over-admission
        that does not.
        """
        if owner is not None:
            for index, slot in enumerate(self._held):
                if slot.owner == owner:
                    del self._held[index]
                    return True, slot.token
            # An owner with no slot means the release is unpaired — a double
            # release, or one for a job whose slot was already reclaimed as
            # orphaned. Consuming SOMEBODY ELSE'S entry here is exactly the bug this
            # parameter exists to remove, so take nothing and say so.
            logger.warning(
                "release(owner=%s) found no held slot for that owner; releasing NOTHING "
                "(double release, or the slot was already reclaimed as orphaned)",
                owner,
            )
            return False, None
        for index, slot in enumerate(self._held):
            if slot.token is None:
                del self._held[index]
                return True, None
        if self._held:
            return True, self._held.popleft().token
        return False, None

    async def release(self, owner: str | None = None) -> None:
        """Release a recording slot (global + per-pod).

        Pass the same ``owner`` given to :pymeth:`acquire` for an exact match.
        """
        async with self._lock:
            # token=None with matched=True is an admission that owns no Redis slot
            # (fall-open, or no Redis): nothing to ZREM, but a real permit to return.
            matched, token = self._take_held_locked(owner)
            if matched:
                if self._active_count > 0:
                    self._active_count -= 1
                self._last_progress = time.monotonic()
        if not matched:
            # UNPAIRED release — this caller holds nothing. Returning here is the
            # point of tracking owners: the old code released the per-pod semaphore
            # anyway, raising its permit count above `per_pod_max` and admitting more
            # Chromium+ffmpeg processes than the memory envelope survives (the
            # phantom-release hazard `_prepare_retry` documents). A release for a
            # slot we do not hold must change nothing at all.
            return
        if token is not None:
            raw = self._raw()
            if raw is not None:
                try:
                    await raw.zrem(self._slot_key, token)
                except _REDIS_ERRORS as exc:
                    logger.warning("Global slot release failed (%s); will TTL-expire", exc)
        self._semaphore.release()
        self._publish_queue_depth()

    # -- Heartbeat / shutdown ----------------------------------------------

    def _ensure_heartbeat(self) -> None:
        """Lazily start the background loop that refreshes held slot deadlines."""
        if self._hb_task is None or self._hb_task.done():
            self._hb_task = asyncio.create_task(self._heartbeat_loop(), name="rec-concurrency-hb")

    async def _heartbeat_loop(self) -> None:
        """Reclaim orphaned slots, then refresh the deadline of every held global
        slot so long recordings keep their slot (only a genuinely crashed pod stops
        refreshing → TTL frees it).

        Reaping runs FIRST and unconditionally, ahead of the ``raw is None`` bail.
        It used to run after it, which made the reaper dead on precisely the path
        that needs it most: with no Redis there is no TTL either, so a local-only
        leak had no second mechanism behind it. Reaping requires nothing from Redis
        -- it decides on a weakref to the owning task and hands the permit back to
        the local semaphore -- so gating it on a live client was gating a local
        repair on a remote dependency.
        """
        interval = max(1.0, self._slot_ttl / 2)
        while True:
            await asyncio.sleep(interval)
            await self._reap_orphaned_slots()
            raw = self._raw()
            if raw is None:
                continue
            async with self._lock:
                # Slots with no token hold no Redis slot — nothing to refresh.
                tokens = [s.token for s in self._held if s.token is not None]
            if not tokens:
                continue
            deadline = time.time() + self._slot_ttl
            try:
                await raw.zadd(self._slot_key, {t: deadline for t in tokens})
            except _REDIS_ERRORS as exc:
                logger.warning("Global slot heartbeat failed: %s", exc)

    async def _reap_orphaned_slots(self) -> None:
        """Reclaim slots whose owning task is gone. Runs on every heartbeat tick.

        This is the half of the wedge that instrumentation cannot fix. A slot whose
        owner finished without releasing is held by NOBODY, and until now nothing
        could reclaim it: the TTL cannot, because the heartbeat immediately below
        renews every held slot unconditionally — so a leaked lease was refreshed
        forever, which is precisely how three leaked slots wedged recording for nine
        hours with no ERROR line.

        The check needs no threshold and no measured traffic distribution, which is
        what makes it safe. A long legitimate recording holds its slot from a task
        that is still running, so it can never be reaped no matter how long it takes;
        a finished task is not doing work no matter how briefly it ran. That is a
        fact about the process, not a guess about duration.

        Reaping is loud: a leak here is a real defect somewhere (a missing
        ``finally``, a task killed without unwinding) and reclaiming it silently
        would hide the cause while papering over the symptom.
        """
        now = time.monotonic()
        reaped: list[_HeldSlot] = []
        async with self._lock:
            for slot in list(self._held):
                if slot.is_orphaned(now):
                    self._held.remove(slot)
                    reaped.append(slot)
                    if self._active_count > 0:
                        self._active_count -= 1
            if reaped:
                self._last_progress = now
        if not reaped:
            return
        for slot in reaped:
            logger.error(
                "Reclaimed an ORPHANED recording slot: owner=%s held %.0fs, owning task is "
                "gone without releasing. The slot is now free, but the missing release is "
                "a defect — find the path that skipped it.",
                slot.owner or "<unidentified>",
                now - slot.acquired_at,
            )
            self._semaphore.release()
        tokens = [s.token for s in reaped if s.token is not None]
        if tokens:
            raw = self._raw()
            if raw is not None:
                with contextlib.suppress(*_REDIS_ERRORS):
                    await raw.zrem(self._slot_key, *tokens)
        self._publish_queue_depth()

    async def close(self) -> None:
        """Shutdown: cancel the heartbeat and release every held global slot so a
        graceful drain frees fleet capacity immediately (rather than by TTL)."""
        if self._hb_task is not None:
            self._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._hb_task
            self._hb_task = None
        async with self._lock:
            tokens = [s.token for s in self._held if s.token is not None]
            self._held.clear()
        if not tokens:
            return
        raw = self._raw()
        if raw is not None:
            with contextlib.suppress(*_REDIS_ERRORS):
                await raw.zrem(self._slot_key, *tokens)

    # -- Executor stall detection ------------------------------------------

    def stall_status(self, threshold_seconds: float) -> ExecutorStallStatus:
        """Report whether the recording executor has WEDGED.

        The distinction this makes -- and the whole reason it exists -- is
        "idle, loop alive" vs "loop dead". A service with nothing queued is not
        unhealthy no matter how long it has been quiet, so an empty wait queue
        reports ``stalled=False`` unconditionally. Only work that is *waiting*
        makes a lack of progress observable.

        "Progress" is a slot ADMITTED or a slot RELEASED, not a job completed:
        a heartbeat tied to completed jobs goes stale during one legitimately
        long recording, and a heartbeat tied to the HTTP API stays fresh while
        the executor is dead -- which is exactly how a 9-hour outage read as
        healthy on ``/health/ready``.

        The clock also restarts on the queue's 0 -> non-empty transition, so an
        overnight idle period cannot make the first job of the morning look
        like a stall.

        ⚠️ KNOWN BLIND SPOT, measured 2026-09-04 rather than argued. The reset at
        the top of this method is gated on the queue being EMPTY, so the probe
        fires exactly as designed under sustained traffic (a new request queues
        before the previous one finishes timing out, `_queued_count` never reaches
        0, and `since` grows unbounded). Under TRICKLE traffic it cannot: one
        waiter at a time, each timing out with an empty queue between, resets the
        clock on every exit. So a wedged pod still receiving thin traffic fails
        every request and never tells the kubelet. `held_without_progress_seconds`
        exists to make that state visible; it is deliberately NOT wired into
        `stalled`, because it cannot distinguish a wedge from one legitimately long
        recording, and a liveness probe that confuses those kills healthy pods
        mid-render. Closing it for real is a restart-policy decision that wants
        production traffic shape behind it, not a reading of this file.

        It is reported on **`/health/live`** (not `/health`, which carries `stats()`
        only), exported as the `recording_slot_held_without_progress_seconds` gauge,
        and logged as a throttled WARNING past `threshold_seconds`.

        ⚠️ Those signals are for a SLOW executor, not a leaked one. The leak case — a
        slot held by a job that is gone — is no longer a measurement problem and no
        longer needs this probe at all: `_reap_orphaned_slots` reclaims it on the
        next heartbeat by asking whether the owning task is still alive, which is
        decidable without any threshold. What is left here is genuinely a judgement
        call about a LIVE but slow holder, and that is the one that still wants a
        measured distribution before anything restarts a pod.

        Args:
            threshold_seconds: How long the queue may be non-empty with zero
                admissions or releases before the executor is called wedged.
                Must exceed the longest legitimate slot hold.

        Returns:
            An :pyclass:`ExecutorStallStatus`.
        """
        held_without_progress = (
            time.monotonic() - self._last_progress if self._active_count > 0 else None
        )
        self._publish_hold_gauge(held_without_progress)
        if held_without_progress is not None and held_without_progress > threshold_seconds:
            self._warn_long_hold(held_without_progress, threshold_seconds)
        queue_started = self._queue_nonempty_since
        if queue_started is None or self._queued_count == 0:
            return {
                "stalled": False,
                "queued": self._queued_count,
                "active": self._active_count,
                "available_slots": self.available_slots,
                "seconds_since_progress": None,
                "threshold_seconds": threshold_seconds,
                "held_without_progress_seconds": held_without_progress,
            }
        # Measure from whichever happened LATER: the last forward progress, or
        # the moment work started waiting.
        since = time.monotonic() - max(self._last_progress, queue_started)
        return {
            "stalled": since > threshold_seconds,
            "queued": self._queued_count,
            "active": self._active_count,
            "available_slots": self.available_slots,
            "seconds_since_progress": since,
            "threshold_seconds": threshold_seconds,
            "held_without_progress_seconds": held_without_progress,
        }

    # -- Metrics -----------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of currently active recording slots on this pod."""
        return self._active_count

    @property
    def queued_count(self) -> int:
        """Number of recordings waiting for a slot on this pod."""
        return self._queued_count

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent recordings allowed (fleet-wide)."""
        return self._max_concurrent

    @property
    def per_pod_max(self) -> int:
        """Maximum concurrent recordings allowed on THIS pod (never above the fleet cap)."""
        return self._per_pod_max

    @property
    def available_slots(self) -> int:
        """Number of free recording slots from this pod's local view."""
        return max(0, self._max_concurrent - self._active_count)

    def stats(self) -> dict:
        """Return a snapshot of concurrency metrics."""
        return {
            "active": self._active_count,
            "queued": self._queued_count,
            "max_concurrent": self._max_concurrent,
            "per_pod_max": self._per_pod_max,
            "queue_wait_timeout_seconds": self._queue_wait_timeout,
            "available_slots": self.available_slots,
        }
