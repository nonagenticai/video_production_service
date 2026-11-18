"""Background job cleanup service for expired recordings.

Runs a periodic async loop that finds expired jobs, deletes their
stored video files, and marks them as expired in the database.
Also reaps stale jobs stuck in processing/queued status.

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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from w2v_common.auth.identity_context import set_identity_context
from w2v_common.database.session import get_db_session

from src.config import get_settings
from src.services.job_lease import JobLease
from src.services.recording.job_manager import JobStateConflictError, RecordingJobManager
from src.services.storage_service import StorageService
from src.utils.error_formatting import format_error

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

# Postgres advisory-lock keys (transaction-scoped → pgbouncer-safe). Only one
# replica runs each sweep at a time; the rest no-op. Distinct per sweep so the
# expired-cleanup and stale-reaper don't serialise against each other.
_RECORDING_EXPIRED_LOCK = 0x7602_0001
_RECORDING_STALE_LOCK = 0x7602_0002


# The ONE definition of "this recording job has stopped making progress".
#
# `stale_recording_jobs` is SECURITY DEFINER and keys on `updated_at`, which
# `RecordingJobManager.record_progress` bumps — so it already means "no progress
# for p_max_age_s", cross-tenant, without needing a bound tenant GUC.
#
# It is a module constant because TWO callers must agree on it exactly: the reaper
# below, and the JobLease heartbeat probe (`stalled_recording_job_ids`). On
# 2026-08-25 they did not agree — the heartbeat had no notion of progress at all
# and renewed leases for 18 progress-dead jobs, which made the reaper skip every
# one of them. If these two ever read different windows the bug comes back, so
# they read the same string.
_STALE_RECORDING_QUERY = (
    "SELECT job_id, tenant_id, status FROM stale_recording_jobs(:age, :queued_age, :lim)"
)

# Bound on rows pulled per cycle. Shared for the same reason as the query.
_STALE_RECORDING_LIMIT = 100

# "age" param for the lease-aware orphan sweep below (see `_reap_stale_jobs`).
# Pinned far beyond any real job's age so that sweep's call to
# `stale_recording_jobs` can NEVER re-select a 'processing' row -- the
# unmodified, module-level `_stale_params` call remains the SOLE authority for
# that branch. ~31.7 years; Postgres INTERVAL handles it without overflow.
_LEASE_ORPHAN_AGE_SENTINEL = 10**9


def _stale_params(settings: Settings) -> dict[str, int]:
    """The ONE parameter binding for `_STALE_RECORDING_QUERY`.

    Shared for the same reason the query string is: the reaper and the JobLease
    heartbeat probe must ask the identical question. Two windows now, so there are
    two ways for them to drift instead of one.

    `age` is the no-progress window and applies ONLY to `status='processing'`.
    `queued_age` applies to the PRE-DISPATCH statuses -- `'pending'` and
    `'queued'` -- and is much longer, because a job waiting for a fleet-wide slot
    has not failed at anything: nothing bumps its `updated_at` while it waits, so
    the old shared window measured queue wait and reaped 276 of 282 jobs before
    they ever ran (2026-09-03). `'pending'` is in that branch because it was in
    NEITHER before -- jobs are INSERTed at `'pending'` and `mark_queued` moves
    them on, so an owner that died in between left a row no clock could reach.
    """
    storage = settings.storage
    return {
        "age": storage.stale_job_timeout_seconds,
        "queued_age": storage.queued_job_timeout_seconds,
        "lim": _STALE_RECORDING_LIMIT,
    }


async def stalled_recording_job_ids() -> set[str]:
    """Job ids with NO progress inside the reaper's own staleness window.

    Wired into `JobLease(stalled_probe=...)` so the heartbeat stops renewing a
    lease the reaper is about to want. Deliberately reuses the reaper's query
    rather than asking the same question a second way.

    Raises on failure — JobLease treats a probe error as "renew as before", so a
    database blip degrades to the old behaviour instead of mass-expiring leases.
    """
    settings = get_settings()
    async with get_db_session() as db:
        rows = (await db.execute(text(_STALE_RECORDING_QUERY), _stale_params(settings))).all()
    return {row[0] for row in rows}


class CleanupService:
    """Periodically clean up expired recording jobs and their files.

    Attributes:
        _storage: StorageService for file deletion.
        _interval: Seconds between cleanup cycles.
        _stale_timeout: Seconds before a stuck job is reaped.
        _task: Background asyncio task handle.
    """

    def __init__(self, storage: StorageService, job_lease: JobLease | None = None) -> None:
        self._storage = storage
        self._job_lease = job_lease
        settings = get_settings()
        self._interval = settings.storage.cleanup_interval_seconds
        self._stale_reap_interval = settings.storage.stale_reap_interval_seconds
        self._stale_timeout = settings.storage.stale_job_timeout_seconds
        self._queued_timeout = settings.storage.queued_job_timeout_seconds
        # Only read (and only ever bound as a SQL param) when a lease is
        # actually configured -- see `_reap_stale_jobs`'s second sweep.
        self._queued_orphan_timeout = settings.storage.queued_orphan_timeout_seconds
        self._stale_params = _stale_params(settings)
        self._task: asyncio.Task | None = None
        self._stale_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the cleanup background tasks (expired-job sweep + stale-reap sweep)."""
        self._task = asyncio.create_task(self._run_loop())
        self._stale_task = asyncio.create_task(self._run_stale_loop())
        logger.info(
            "CleanupService started (interval=%ds, stale_reap_interval=%ds)",
            self._interval,
            self._stale_reap_interval,
        )

    async def stop(self) -> None:
        """Stop both cleanup background tasks."""
        for attr in ("_task", "_stale_task"):
            task: asyncio.Task | None = getattr(self, attr)
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                setattr(self, attr, None)
        logger.info("CleanupService stopped")

    async def _run_loop(self) -> None:
        """Main cleanup loop: the expired-job sweep, on its (hourly) interval.

        The stale-reap sweep used to run here too, but sharing this interval
        meant a job stale at +stale_job_timeout_seconds (600s) sat un-reaped
        until the next hourly tick -- measured up to +3596s late. It now runs
        on its own, much shorter cadence: see `_run_stale_loop`.
        """
        # Work FIRST, sleep after. A sleep-first loop never completes a single
        # cycle in a pod that restarts more often than the interval (3600s),
        # which is the normal case during a rollout -- so the reap never runs.
        while True:
            try:
                await self._cleanup_expired()
                await self._reap_stale_jobs()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup loop")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _run_stale_loop(self) -> None:
        """Stale-reap loop, on its OWN (much shorter) cadence than `_run_loop`.

        `_reap_stale_jobs` is cheap -- one advisory-lock probe plus a couple of
        small indexed SELECTs -- unlike `_cleanup_expired`'s real storage
        deletes, so it can safely run far more often than the hourly
        `_run_loop`. Measured: without this, a job stale at
        +stale_job_timeout_seconds (600s) was reaped by the next HOURLY tick,
        up to +3596s late -- ~6x the window it was meant to enforce, silently
        holding a fleet concurrency slot the whole time. `_reap_stale_jobs`
        is still ALSO invoked from `_run_loop` every cycle, so this is purely
        additive: no change to what gets reaped, only to how soon.

        Work-first for the same reason `_run_loop` is (see its comment): a
        pod that restarts more often than the interval must still complete
        one cycle.
        """
        while True:
            try:
                await self._reap_stale_jobs()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in stale-reap loop")
            try:
                await asyncio.sleep(self._stale_reap_interval)
            except asyncio.CancelledError:
                break

    async def _cleanup_expired(self) -> None:
        """Find and clean up expired jobs.

        Phase D RLS: the expired-job scan is cross-tenant (no tenant context at boot),
        so it runs through the ``expired_recording_jobs`` SECURITY DEFINER read resolver
        (owner bypasses ENABLE). Each job is then mutated in its own tenant-bound
        session, reusing the existing ``mark_expired``.
        """
        try:
            limit = get_settings().job.expired_batch_limit
            async with get_db_session() as db:
                # One replica per sweep: the advisory xact-lock is held for the
                # cross-tenant scan; a losing pod no-ops this cycle.
                locked = (
                    await db.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": _RECORDING_EXPIRED_LOCK},
                    )
                ).scalar()
                if not locked:
                    logger.debug("expired-cleanup: another replica holds the lock; skipping")
                    return
                rows = (
                    await db.execute(
                        text("SELECT job_id, tenant_id FROM expired_recording_jobs(:lim)"),
                        {"lim": limit},
                    )
                ).all()
            if not rows:
                return

            logger.info("Found %d expired jobs to clean up", len(rows))
            cleaned = 0
            for job_id, tenant_id in rows:
                try:
                    await self._storage.cleanup_job(job_id)
                    # ⛔ tenant_source stays absent BY DESIGN — this loop rebinds a
                    # DIFFERENT tenant per row, so the library's tenant-change drop
                    # clears provenance to None on every iteration and the reaper
                    # fails closed by itself. Adding tenant_source= here would be an
                    # ESCALATION, not a fix: an explicit value always wins and would
                    # bypass that drop. See the module docstring.
                    set_identity_context(tenant_id, None, None)
                    async with get_db_session() as db2:
                        await RecordingJobManager(db2).mark_expired(job_id)
                    cleaned += 1
                except Exception:
                    logger.exception("Failed to clean up job %s", job_id)
            logger.info("Cleaned up %d expired jobs", cleaned)
        except Exception:
            logger.exception("Cleanup cycle failed")

    async def _select_stale_and_orphan_rows(self) -> tuple[Sequence[Any], dict[str, int]]:
        """Run both selection queries (see `_reap_stale_jobs`) under one
        advisory-lock-holding session, dedupe by job id, and return the
        combined row list plus a map of job id -> the window that fired for
        any row found ONLY by the accelerated orphan sweep.

        Split out of `_reap_stale_jobs` purely to keep that method's
        complexity down; no behavioural seam here.
        """
        async with get_db_session() as db:
            locked = (
                await db.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)"),
                    {"k": _RECORDING_STALE_LOCK},
                )
            ).scalar()
            if not locked:
                logger.debug("stale reaper: another replica holds the lock; skipping")
                return [], {}
            rows = (await db.execute(text(_STALE_RECORDING_QUERY), self._stale_params)).all()
            orphan_rows: Sequence[Any] = ()
            if self._job_lease is not None:
                orphan_params = {
                    "age": _LEASE_ORPHAN_AGE_SENTINEL,
                    "queued_age": self._queued_orphan_timeout,
                    "lim": _STALE_RECORDING_LIMIT,
                }
                orphan_rows = (await db.execute(text(_STALE_RECORDING_QUERY), orphan_params)).all()

        seen_ids = {row[0] for row in rows}
        # Maps a job id reaped ONLY via the accelerated orphan sweep to the
        # window that actually applied, so the persisted error message never
        # misattributes the reap to `queued_job_timeout_seconds`.
        orphan_window_by_id: dict[str, int] = {}
        combined = list(rows)
        for row in orphan_rows:
            if row[0] in seen_ids:
                continue
            combined.append(row)
            seen_ids.add(row[0])
            orphan_window_by_id[row[0]] = self._queued_orphan_timeout
        return combined, orphan_window_by_id

    async def _reap_stale_jobs(self) -> None:
        """Find and fail jobs stuck in processing/queued/pending for too long.

        Multi-replica-safe: (1) an advisory xact-lock ensures a single pod runs
        the sweep; (2) a job whose Redis liveness lease is still present is
        running on another pod and is SKIPPED — only a genuinely orphaned job
        (crashed pod, lease TTL-expired) past the age threshold is failed, which
        is the crashed-pod dispatch-reclaim path. Cross-tenant scan via the
        ``stale_recording_jobs`` SECURITY DEFINER resolver; each reap runs in its
        own tenant-bound session via ``mark_failed``.

        Runs TWO selection queries against the same resolver, both inside the
        one advisory-lock-holding session:

        1. The unmodified, original sweep (``self._stale_params`` — 600s for
           ``processing``, ``queued_job_timeout_seconds`` [7200s] for
           ``pending``/``queued``). Unchanged behaviour, unchanged windows —
           the sole authority when no ``JobLease`` is configured, since
           liveness cannot be confirmed at all in that case.
        2. An ADDITIONAL sweep, only when a ``JobLease`` IS configured, using
           ``queued_orphan_timeout_seconds`` (450s, far tighter) for
           ``pending``/``queued`` and a sentinel ``age`` that can never match
           ``processing``. A row this sweep selects whose lease is ALSO
           absent cannot be a live pod's job still within its normal queue
           wait (see the setting's own docstring for why) — it can only be a
           pod that died before ``RecordingConcurrencyManager.acquire``'s own
           300s ``queue_wait_timeout_seconds`` could fail it out on its own.
           This is what turns "an orphaned queued job sits for up to 7200s"
           into "sits for at most ~queued_orphan_timeout_seconds +
           stale_reap_interval_seconds" — recovery via PROMPT FAILURE, reusing
           the exact same lease-presence check and ``mark_failed`` path as
           sweep 1, not a new locking scheme. Re-dispatch was considered and
           rejected: nothing in this service currently re-queues a failed job
           onto another pod, and inventing that here would be a materially
           larger change than this defect calls for.

        Rows found by both sweeps are deduped by job id (sweep 1 wins).
        """
        try:
            combined, orphan_window_by_id = await self._select_stale_and_orphan_rows()
            if not combined:
                return

            logger.warning(
                "Found %d stale job(s) (%d via the lease-aware orphan sweep); checking "
                "liveness before reaping",
                len(combined),
                len(orphan_window_by_id),
            )
            reaped = 0
            for job_id, tenant_id, status in combined:
                if self._job_lease is not None and await self._job_lease.present(job_id):
                    logger.info(
                        "stale reaper: job %s has a live lease (running on another pod); skipping",
                        job_id,
                    )
                    continue
                try:
                    # Name the window that actually fired. A pre-dispatch job
                    # ('pending'/'queued') and a processing job are reaped on
                    # different clocks now, and an error message quoting the wrong
                    # one sends the next reader to the wrong setting. Keyed on
                    # 'processing' rather than on the queued statuses so that adding
                    # a further pre-dispatch status cannot silently mislabel it.
                    window = orphan_window_by_id.get(job_id) or (
                        self._stale_timeout if status == "processing" else self._queued_timeout
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
                        await RecordingJobManager(db2).mark_failed(job_id, error_msg)
                    reaped += 1
                except JobStateConflictError:
                    # Raced with the owning pod's own terminal transition — safe.
                    logger.info("stale reaper: job %s already terminal; skipping", job_id)
                except Exception as exc:
                    logger.exception("Failed to reap stale job %s: %s", job_id, format_error(exc))
            logger.warning("Reaped %d stale job(s)", reaped)
        except Exception:
            logger.exception("Stale job reaper cycle failed")
