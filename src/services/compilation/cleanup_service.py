"""Background cleanup service for expired compilation jobs.

⛔ Tenant provenance is DELIBERATELY left to the library's tenant-change drop
-----------------------------------------------------------------------------
Every ``set_identity_context`` call in this module sits inside a
``for … tenant_id … in rows:`` loop over a cross-tenant scan, and rebinds a
DIFFERENT tenant on each iteration. It must stay the bare three-argument form.

Since w2v-common 0.67.7, omitting ``tenant_source`` does NOT unconditionally
clear it. The parameter defaults to a ``KEEP_TENANT_SOURCE`` sentinel, so the
bare call means "keep what is ambient, unless the tenant changes": same tenant
preserves provenance, a DIFFERENT tenant drops it to ``None``. That is precisely
what this loop needs — each row rebinds another tenant, so the drop fires on
every iteration and the reaper fails closed automatically, with no per-row
bookkeeping of its own.

Adding ``tenant_source=`` here would be an ESCALATION, not a fix. An explicit
value ALWAYS wins, so it would bypass the tenant-change drop and vouch whichever
tenant ran last (or a stale ambient value — this loop is driven by ``_run_loop``,
not by a handler) for another tenant's row. That is exactly what WS-4 exists to
stop. Contrast the request-dispatched ``_bg`` entrypoints in ``src/api/routes/``
(see their module docstrings): they use the SAME bare call, and there it
preserves, because they rebind the tenant they inherited.

Failing closed costs nothing here: ``build_outbound_identity`` reports ``None``
as ``'unrecorded'`` and refuses every metered destination, and a reaper only
deletes storage and flips rows to a terminal state — it never spends a tenant's
credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import text
from w2v_common.auth.identity_context import set_identity_context
from w2v_common.database.session import get_db_session

from src.config import get_settings
from src.services.job_lease import JobLease
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# Transaction-scoped advisory-lock keys (pgbouncer-safe) so a single replica runs
# each sweep. Distinct from the recording service's keys.
_COMPILATION_EXPIRED_LOCK = 0x7602_0011
_COMPILATION_STALE_LOCK = 0x7602_0012
# The BATCH (aggregate) sweeps get their own keys. Sharing a key with the job
# sweeps would silently serialise two independent sweeps behind one lock and let a
# slow job scan starve the batch reclaim entirely.
_COMPILATION_BATCH_STALE_LOCK = 0x7602_0013
_COMPILATION_BATCH_EXPIRED_LOCK = 0x7602_0014


# The reaper's own stale-job query, hoisted so the lease heartbeat and the reaper
# ask the identical question. They MUST NOT diverge: a probe that gave up earlier
# than the reaper's window would mass-expire healthy long compiles, and one that
# gave up later would restore the bug this exists to close.
_STALE_COMPILATION_QUERY = (
    "SELECT job_id, tenant_id, status FROM stale_compilation_jobs(:age, :pending_age, :lim)"
)
_STALE_COMPILATION_LIMIT = 100

# The BATCH (aggregate) sweeps. ``compilation_batches`` had no resolver, no reaper
# and no prune at all before 2026-09-06, so a batch whose owning
# ``BatchService.process_batch`` task died -- pod restart, OOM, or any exception the
# route's handler failed to catch -- stayed 'processing' (or 'pending', if the pod
# died before the task ran) FOREVER, with the client polling batch status until its
# own deadline and no webhook ever firing.
_STALE_COMPILATION_BATCH_QUERY = (
    "SELECT batch_id, tenant_id, status FROM stale_compilation_batches(:age, :lim)"
)
_EXPIRED_COMPILATION_BATCH_QUERY = (
    "SELECT batch_id, tenant_id FROM expired_compilation_batches(:cb, :fb, :lim)"
)
_COMPILATION_BATCH_LIMIT = 100


async def stalled_compilation_job_ids() -> set[str]:
    """Compilation job ids with NO progress inside the reaper's staleness window.

    The compilation sibling of ``recording.cleanup_service.stalled_recording_job_ids``,
    and it was missing. ``JobLease._heartbeat_loop`` only skips renewing a lease when
    a ``stalled_probe`` names it, so the compilation lease -- constructed WITHOUT one
    -- renewed every held lease every ``lease_heartbeat_seconds``, forever, for as
    long as the pod stayed alive. ``_reap_stale_jobs`` then skipped every row it had
    correctly selected, because ``present()`` was True and it was True only because
    that loop kept asserting it.

    The consequence is a job that can never reach a terminal state: a compile wedged
    on a LIVE pod (a stalled MinIO read on the uncancellable ``retrieve_uri_sync``
    worker thread, or an ffmpeg step inside ``asyncio.to_thread`` that never returns)
    stays ``processing`` permanently -- never failed, never webhooked, never billed
    as an error, with the caller polling until its own deadline. This is the exact
    failure measured on the RECORDING lease on 2026-08-25 (18 wedged jobs, reaper
    skipped all 18); the fix was applied to one of the two identical call sites.

    Raises on failure -- ``JobLease`` treats a probe error as "renew as before", so a
    database blip degrades to the old behaviour instead of mass-expiring leases.
    """
    settings = get_settings()
    async with get_db_session() as db:
        rows = (
            await db.execute(
                text(_STALE_COMPILATION_QUERY),
                {
                    "age": settings.storage.stale_job_timeout_seconds,
                    "pending_age": settings.storage.compilation_pending_stale_seconds,
                    "lim": _STALE_COMPILATION_LIMIT,
                },
            )
        ).all()
    return {row[0] for row in rows}


class CleanupService:
    """Periodically cleans up expired compilation jobs and their storage files."""

    def __init__(self, job_lease: JobLease | None = None) -> None:
        self._task: asyncio.Task | None = None
        self._job_lease = job_lease
        settings = get_settings()
        self._enabled = settings.cleanup.enabled
        self._interval = settings.cleanup.interval_seconds
        self._completed_ttl = settings.cleanup.completed_ttl_days
        self._failed_ttl = settings.cleanup.failed_ttl_days
        self._stale_timeout = settings.storage.stale_job_timeout_seconds
        # SEPARATE, far longer clock for the pre-dispatch 'pending' state. It is a
        # DIFFERENT measurement, not a tunable variant of the one above: `_stale_timeout`
        # bounds SILENCE on a running job, this one bounds QUEUE WAIT on a job that has
        # not started. Pointing both at `stale_job_timeout_seconds` would fail out live
        # queued work — see the field docstring in src/config.py.
        self._pending_stale_timeout = settings.storage.compilation_pending_stale_seconds
        # A THIRD clock, for the aggregate row -- and a DIFFERENT KIND of clock. The
        # two above measure silence/queue-wait; this one is a WALL-CLOCK CEILING on a
        # whole batch, because process_batch never commits between the 'processing'
        # flip and the end, so from any other connection `updated_at` is frozen at
        # batch start however much progress is being made. See _reap_stale_batches
        # and the field docstring in src/config.py.
        self._batch_stale_timeout = settings.storage.compilation_batch_stale_seconds
        self._storage: StorageService | None = None

    async def start(self) -> None:
        """Start the background cleanup loop."""
        if not self._enabled:
            logger.info("Cleanup service disabled")
            return
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "Cleanup service started (interval=%ds, completed_ttl=%dd, failed_ttl=%dd)",
            self._interval,
            self._completed_ttl,
            self._failed_ttl,
        )

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            logger.info("Cleanup service stopped")

    async def _run_loop(self) -> None:
        """Main loop: run a cleanup cycle, then wait out the interval."""
        # Work FIRST, sleep after. A sleep-first loop never completes a single
        # cycle in a pod that restarts more often than the interval (3600s),
        # which is the normal case during a rollout -- so the reap never runs.
        while True:
            try:
                await self._cleanup_cycle()
                await self._reap_stale_jobs()
                await self._reap_stale_batches()
                await self._cleanup_expired_batches()
            except asyncio.CancelledError:
                break
            except Exception:
                # Belt-and-braces, matching src/services/recording/cleanup_service.py.
                # Each cycle already guards itself, but this loop OWNS the asyncio
                # task: anything that escapes a cycle kills the task outright, and
                # because `self._task` holds a reference for the pod's lifetime the
                # task is never GC'd, so "Task exception was never retrieved" is
                # never emitted either. The failure is TOTALLY silent. A curated
                # tuple here encodes "failures I thought of" -- which is exactly how
                # a storage/SQLAlchemy error out of `delete_job_dir` silently killed
                # the compilation reaper with it. Nothing may escape.
                logger.exception("Cleanup loop cycle failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _cleanup_cycle(self) -> None:
        """Find and delete expired completed/failed jobs.

        Phase D RLS: the expired-job scan is cross-tenant (no tenant context at boot),
        so it runs through the ``expired_compilation_jobs`` SECURITY DEFINER read
        resolver (owner bypasses ENABLE). Each job's storage is deleted, then the DB
        row is removed in its own tenant-bound session via the existing ``delete_jobs``.

        Wraps its whole body in ``except Exception``, exactly as the recording
        sibling's ``_cleanup_expired`` does. This is the level the guard belongs
        at -- not a widened tuple in ``_run_loop`` -- because it keeps the two
        sweeps INDEPENDENT: a cleanup failure must not also skip
        ``_reap_stale_jobs`` in the same cycle, which a loop-level catch alone
        would do. ``_reap_stale_jobs`` already guards itself the same way.
        """
        try:
            from src.services.compilation.job_manager import CompilationJobManager

            now = datetime.now(UTC)
            if self._storage is None:
                self._storage = StorageService()
            storage = self._storage

            completed_before = now - timedelta(days=self._completed_ttl)
            failed_before = now - timedelta(days=self._failed_ttl)
            async with get_db_session() as db:
                # Single replica per sweep (advisory xact-lock held for the scan).
                locked = (
                    await db.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": _COMPILATION_EXPIRED_LOCK},
                    )
                ).scalar()
                if not locked:
                    logger.debug("compilation cleanup: another replica holds the lock; skipping")
                    return
                rows = (
                    await db.execute(
                        text(
                            "SELECT job_id, tenant_id FROM expired_compilation_jobs(:cb, :fb, :lim)"
                        ),
                        {"cb": completed_before, "fb": failed_before, "lim": 200},
                    )
                ).all()
            if not rows:
                return

            deleted = 0
            for job_id, tenant_id in rows:
                try:
                    await storage.delete_job_dir(job_id)
                except Exception:
                    # Storage deletion is best-effort: the DB row must still go, or
                    # the job is immortal. The backend may be MinIO, so the error
                    # class is whatever its SDK raises -- not one of three builtins.
                    logger.exception("Failed to delete storage for job %s", job_id)
                try:
                    # ⛔ tenant_source stays absent BY DESIGN — this loop rebinds a
                    # DIFFERENT tenant per row, so the library's tenant-change drop
                    # clears provenance to None on every iteration and the reaper
                    # fails closed by itself. Adding tenant_source= here would be an
                    # ESCALATION, not a fix: an explicit value always wins and would
                    # bypass that drop. See the module docstring.
                    set_identity_context(tenant_id, None, None)
                    async with get_db_session() as db2:
                        await CompilationJobManager.delete_jobs(db2, [job_id])
                    deleted += 1
                except Exception:
                    logger.exception("Failed to delete compilation job %s", job_id)

            logger.info("Cleanup: deleted %d expired compilation jobs", deleted)
        except Exception:
            logger.exception("Compilation cleanup cycle failed")

    async def _reap_stale_jobs(self) -> None:
        """Fail compilation jobs stuck in processing/queued past the age threshold.

        Multi-replica-safe, mirroring the recording reaper: an advisory xact-lock
        makes a single pod sweep, and a job whose Redis liveness lease is still
        present (running on another pod) is SKIPPED. Only a genuinely orphaned job
        — a crashed pod's job whose lease TTL-expired — is failed, giving the
        crashed-pod compilation dispatch a safe reclaim (fail + client-retryable)
        instead of hanging in ``processing`` forever. Cross-tenant scan via the
        ``stale_compilation_jobs`` SECURITY DEFINER resolver.
        """
        from src.services.compilation.job_manager import (
            CompilationJobManager,
            JobStateConflictError,
        )

        try:
            async with get_db_session() as db:
                locked = (
                    await db.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": _COMPILATION_STALE_LOCK},
                    )
                ).scalar()
                if not locked:
                    logger.debug("compilation reaper: another replica holds the lock; skipping")
                    return
                rows = (
                    await db.execute(
                        text(_STALE_COMPILATION_QUERY),
                        {
                            "age": self._stale_timeout,
                            "pending_age": self._pending_stale_timeout,
                            "lim": _STALE_COMPILATION_LIMIT,
                        },
                    )
                ).all()
            if not rows:
                return

            logger.warning(
                "Found %d stale compilation job(s); checking liveness before reaping", len(rows)
            )
            reaped = 0
            for job_id, tenant_id, status in rows:
                if self._job_lease is not None and await self._job_lease.present(job_id):
                    logger.info(
                        "compilation reaper: job %s has a live lease; skipping (other pod)", job_id
                    )
                    continue
                try:
                    # Name the window that actually fired. Reporting the processing
                    # window on a pending row would misstate the deadline by 36x and
                    # send the next reader looking at the wrong knob.
                    window = (
                        self._stale_timeout
                        if status == "processing"
                        else self._pending_stale_timeout
                    )
                    error_msg = f"Job reaped: stuck in '{status}' for over {window}s"
                    # ⛔ tenant_source stays absent BY DESIGN — this loop rebinds a
                    # DIFFERENT tenant per row, so the library's tenant-change drop
                    # clears provenance to None on every iteration and the reaper
                    # fails closed by itself. Adding tenant_source= here would be an
                    # ESCALATION, not a fix: an explicit value always wins and would
                    # bypass that drop. See the module docstring.
                    set_identity_context(tenant_id, None, None)
                    async with get_db_session() as db2:
                        # mark_failed commits internally.
                        await CompilationJobManager.mark_failed(db2, job_id, error_msg)
                    reaped += 1
                except JobStateConflictError:
                    logger.info("compilation reaper: job %s already terminal; skipping", job_id)
                except Exception:
                    logger.exception("Failed to reap stale compilation job %s", job_id)
            logger.warning("Reaped %d stale compilation job(s)", reaped)
        except Exception:
            logger.exception("Compilation stale reaper cycle failed")

    async def _reap_stale_batches(self) -> None:
        """Fail compilation BATCHES stuck in pending/processing past the age threshold.

        The aggregate sibling of ``_reap_stale_jobs``, and it was missing entirely: the
        only writer that ever advances a ``compilation_batches`` row is the
        request-spawned ``BatchService.process_batch`` task, so a pod restart, an OOM,
        or any exception its caller's handler fails to catch leaves the row immortal --
        client polling forever, no webhook, failure never accounted.

        ⚠️ THE WINDOW IS A WALL-CLOCK CEILING, NOT A NO-PROGRESS WINDOW.

        ``process_batch`` opens ONE transaction for the batch's whole life and does not
        commit again between the ``'processing'`` flip and the final status write: every
        per-job progress write is a bare ``flush()``. So from any OTHER connection --
        including this reaper's -- ``updated_at`` is frozen at batch start no matter how
        much progress the batch is making. Measured live: the resolver returns a batch
        that is demonstrably progressing. Do not read this clock as "silence".

        What keeps that from failing live work is the ROW LOCK, not the clock. Once the
        first child job flushes, the owning transaction holds this row until the batch
        ends, and ``mark_batch_failed`` uses ``SKIP LOCKED`` -- so an in-flight batch is
        passed over for the real reason (a live transaction owns it), and only a batch
        with no owner holding the row is ever reaped. The residual exposure is a batch
        whose FIRST child job has not finished within the window; that one is reaped
        mid-flight, and ``process_batch`` strips the reaper's error entry if it later
        finishes anyway.

        Closing that residue properly needs a COMMITTED heartbeat on the batch row (a
        short autonomous transaction per progress tick), which is a real change to
        ``process_batch``'s transaction shape and is deliberately NOT done here. This is
        not a no-progress reaper and must not be described as one.

        There is no liveness lease to consult either (a batch is not a job and holds
        none). The transition is still guarded: ``mark_batch_failed`` re-checks the
        status under the lock it does get, so a batch that legitimately completed
        between the scan and the write is left alone.

        Multi-replica-safe via its own advisory xact-lock; cross-tenant scan via the
        ``stale_compilation_batches`` SECURITY DEFINER resolver, then a tenant-bound
        mutation per row.
        """
        from src.services.compilation.batch_service import BatchService

        try:
            async with get_db_session() as db:
                locked = (
                    await db.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": _COMPILATION_BATCH_STALE_LOCK},
                    )
                ).scalar()
                if not locked:
                    logger.debug("batch reaper: another replica holds the lock; skipping")
                    return
                rows = (
                    await db.execute(
                        text(_STALE_COMPILATION_BATCH_QUERY),
                        {"age": self._batch_stale_timeout, "lim": _COMPILATION_BATCH_LIMIT},
                    )
                ).all()
            if not rows:
                return

            logger.warning("Found %d stale compilation batch(es); reaping", len(rows))
            reaped = 0
            for batch_id, tenant_id, status in rows:
                try:
                    reason = (
                        f"Batch reaped: stuck in '{status}' for over "
                        f"{self._batch_stale_timeout}s with no batch-level progress; "
                        "the owning process is gone"
                    )
                    # ⛔ tenant_source stays absent BY DESIGN — this loop rebinds a
                    # DIFFERENT tenant per row, so the library's tenant-change drop
                    # clears provenance to None on every iteration and the reaper
                    # fails closed by itself. See the module docstring.
                    set_identity_context(tenant_id, None, None)
                    async with get_db_session() as db2:
                        # mark_batch_failed commits internally and names what actually
                        # happened, so `reaped` counts real reclaims only and every skip
                        # is logged with its real reason rather than a guess.
                        outcome = await BatchService.mark_batch_failed(db2, batch_id, reason)
                        if outcome == "reaped":
                            reaped += 1
                        elif outcome == "locked":
                            # A live transaction still owns this row (the owning
                            # process_batch holds it from its first child flush to the
                            # end). NOT an orphan; retried next sweep. The reaper does
                            # not wait for it -- see mark_batch_failed.
                            logger.info(
                                "batch reaper: batch %s is locked by a live owner; skipping",
                                batch_id,
                            )
                        else:
                            logger.info("batch reaper: batch %s is %s; skipping", batch_id, outcome)
                except Exception:
                    logger.exception("Failed to reap stale compilation batch %s", batch_id)
            logger.warning("Reaped %d stale compilation batch(es)", reaped)
        except Exception:
            logger.exception("Compilation batch stale reaper cycle failed")

    async def _cleanup_expired_batches(self) -> None:
        """Delete terminal compilation BATCH rows past their TTL -- the prune backstop.

        The reaper above stops a batch being immortal in a NON-terminal state; without
        this, every batch row ever created still accumulates forever in a terminal one.
        Batches own no storage of their own (the child jobs' artifacts are pruned by
        ``_cleanup_cycle``), so this is a pure row delete.

        Guarded as its own sweep, like ``_cleanup_cycle``: a prune failure must not
        also skip the reclaim above in the same cycle.
        """
        from src.services.compilation.batch_service import BatchService

        try:
            now = datetime.now(UTC)
            completed_before = now - timedelta(days=self._completed_ttl)
            failed_before = now - timedelta(days=self._failed_ttl)
            async with get_db_session() as db:
                locked = (
                    await db.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": _COMPILATION_BATCH_EXPIRED_LOCK},
                    )
                ).scalar()
                if not locked:
                    logger.debug("batch prune: another replica holds the lock; skipping")
                    return
                rows = (
                    await db.execute(
                        text(_EXPIRED_COMPILATION_BATCH_QUERY),
                        {
                            "cb": completed_before,
                            "fb": failed_before,
                            "lim": _COMPILATION_BATCH_LIMIT,
                        },
                    )
                ).all()
            if not rows:
                return

            deleted = 0
            for batch_id, tenant_id in rows:
                try:
                    # ⛔ tenant_source stays absent BY DESIGN — see the module docstring.
                    set_identity_context(tenant_id, None, None)
                    async with get_db_session() as db2:
                        deleted += await BatchService.delete_batches(db2, [batch_id])
                except Exception:
                    logger.exception("Failed to delete expired compilation batch %s", batch_id)
            logger.info("Cleanup: deleted %d expired compilation batches", deleted)
        except Exception:
            logger.exception("Compilation batch prune cycle failed")
