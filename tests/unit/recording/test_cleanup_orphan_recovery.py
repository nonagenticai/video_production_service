"""Defect 2 (reaper cadence) and Defect 3 (orphaned `queued` job recovery).

Defect 2: `_reap_stale_jobs` used to run only on the hourly
`cleanup_interval_seconds` cadence shared with the (expensive) expired-job
storage sweep. A job stale at +stale_job_timeout_seconds sat un-reaped for up
to +3596s -- ~6x the window it was meant to enforce. `_run_stale_loop` now
runs the SAME `_reap_stale_jobs` on its own, much shorter cadence.

Defect 3: a `queued`/`pending` job whose owning pod died before
`RecordingConcurrencyManager.acquire`'s own 300s `queue_wait_timeout_seconds`
loop could fail it out could previously only be reclaimed by the 7200s
`queued_job_timeout_seconds` fallback. `_reap_stale_jobs` now runs an
ADDITIONAL, lease-aware sweep (`queued_orphan_timeout_seconds`, 450s) whenever
a `JobLease` is configured, reusing the exact same lease-presence check and
`mark_failed` path as the original sweep -- prompt failure, not re-dispatch,
and no new locking scheme.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from src.database.models import RecordingJob
from tests.conftest import TENANT_A
from tests.unit.recording.test_cleanup_reaper_lease import _lease, _make_service
from tests.unit.recording.test_cleanup_service import _make_cleanup_service, _seed


async def _status_of(db, job_id: str) -> RecordingJob | None:
    from sqlalchemy import select

    return (
        await db.execute(select(RecordingJob).where(RecordingJob.job_id == job_id))
    ).scalar_one_or_none()


# ── Defect 2: reaper cadence ─────────────────────────────────────────


class TestStaleReapCadence:
    async def test_stale_job_reaped_within_the_fast_interval_not_the_hourly_one(self, db_session):
        """A `processing` job that goes stale AFTER `start()` has already run
        its first cycle must still be reaped inside the FAST
        `stale_reap_interval_seconds` cadence, not wait for the slow (here:
        3600s) `cleanup_interval_seconds` tick that drives `_cleanup_expired`.

        The job is seeded FRESH and only aged (via a direct UPDATE) after the
        first cycle of both loops has already run and found nothing -- both
        `_run_loop` and `_run_stale_loop` do work BEFORE their first sleep
        (see `test_cleanup_loops_work_first.py`), so a job seeded already-stale
        would be reaped by `_run_loop`'s own first cycle regardless of this
        fix, which would make this test pass for the wrong reason.

        Mutation-proof: reverting `CleanupService.start()` to launch only
        `_run_loop` (the pre-fix shape, one combined hourly loop) leaves this
        job 'processing' after the test's short second wait, since
        `_run_loop`'s SECOND cycle does not fire until its 3600s interval
        elapses.
        """
        await _seed(db_session, job_id="cadence-j", tenant_id=TENANT_A, status="processing")

        svc = _make_cleanup_service(interval=3600, stale_timeout=600, stale_reap_interval=0.05)
        await svc.start()
        try:
            # Let both loops complete their work-first FIRST cycle, finding
            # nothing (the job is still fresh) -- this is what makes the
            # later reap attributable to a SECOND cycle, not the first.
            await asyncio.sleep(0.1)

            from sqlalchemy import update

            from src.database.models import RecordingJob

            await db_session.execute(
                update(RecordingJob)
                .where(RecordingJob.job_id == "cadence-j")
                .values(updated_at=datetime.now(UTC) - timedelta(seconds=700))
            )
            await db_session.commit()

            await asyncio.sleep(0.3)
        finally:
            await svc.stop()

        db_session.expire_all()
        row = await _status_of(db_session, "cadence-j")
        assert row is not None and row.status == "failed"

    async def test_a_fresh_processing_job_survives_the_fast_loop(self, db_session):
        """Control: the fast loop must not reap anything merely by running
        often -- a job well inside its window stays untouched no matter how
        many extra sweeps happen."""
        await _seed(db_session, job_id="cadence-fresh", tenant_id=TENANT_A, status="processing")

        svc = _make_cleanup_service(interval=3600, stale_timeout=600, stale_reap_interval=0.05)
        await svc.start()
        try:
            await asyncio.sleep(0.3)
        finally:
            await svc.stop()

        db_session.expire_all()
        row = await _status_of(db_session, "cadence-fresh")
        assert row is not None and row.status == "processing"


# ── Defect 3: orphaned `queued`/`pending` job recovery ──────────────


class TestQueuedOrphanRecovery:
    async def test_a_dead_pods_queued_job_is_failed_promptly(self, db_session):
        """450s (`queued_orphan_timeout_seconds`) is far past a live pod's own
        300s queue-wait SLA but far short of the 7200s fallback -- exactly
        the gap Defect 3 closes. No lease was ever acquired for this job id,
        simulating a pod that died mid-wait.
        """
        abandoned = datetime.now(UTC) - timedelta(seconds=500)
        await _seed(
            db_session,
            job_id="queued-dead",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=abandoned,
        )

        lease = _lease()  # never acquired -> present() is False
        svc = _make_service(lease, queued_orphan_timeout=450)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        row = await _status_of(db_session, "queued-dead")
        assert row is not None
        assert row.status == "failed"
        assert "reaped" in (row.error_message or "").lower()
        # Names the ACCELERATED window (450s), proving the new orphan sweep
        # fired -- not the original 7200s fallback, which this row is nowhere
        # near (500s < 7200s).
        assert "over 450s" in (row.error_message or "")

    async def test_a_live_pods_queued_job_is_not_reaped_even_past_the_orphan_window(
        self, db_session
    ):
        """A live pod's lease is held for the WHOLE `process_job` body
        (including the queued wait), so it must protect the row even once
        the row is older than `queued_orphan_timeout_seconds`."""
        waiting = datetime.now(UTC) - timedelta(seconds=500)
        await _seed(
            db_session,
            job_id="queued-alive",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=waiting,
        )

        lease = _lease()
        await lease.acquire("queued-alive")
        svc = _make_service(lease, queued_orphan_timeout=450)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        row = await _status_of(db_session, "queued-alive")
        assert row is not None and row.status == "queued"
        await lease.release_all()

    async def test_no_lease_configured_leaves_the_wide_fallback_window_unchanged(self, db_session):
        """Defect-3 fix must not weaken the no-lease (Redis absent) fallback:
        with `job_lease=None` a queued job past the 450s orphan threshold but
        still well inside `queued_job_timeout_seconds` (7200s) is left
        alone -- exactly today's behaviour, unchanged. Liveness cannot be
        confirmed at all without a lease, so only the wide age-only window
        may act.
        """
        waiting = datetime.now(UTC) - timedelta(seconds=500)
        await _seed(
            db_session,
            job_id="queued-no-lease",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=waiting,
        )

        svc = _make_service(None, queued_orphan_timeout=450)  # job_lease=None

        await svc._reap_stale_jobs()

        db_session.expire_all()
        row = await _status_of(db_session, "queued-no-lease")
        assert row is not None and row.status == "queued"

    async def test_a_dead_pods_pending_job_is_also_recovered(self, db_session):
        """`pending` shares the orphan sweep with `queued` (both are
        pre-dispatch statuses covering the same failure -- see
        `queued_job_timeout_seconds`'s own docstring)."""
        abandoned = datetime.now(UTC) - timedelta(seconds=500)
        await _seed(
            db_session,
            job_id="pending-dead",
            tenant_id=TENANT_A,
            status="pending",
            updated_at=abandoned,
        )

        lease = _lease()
        svc = _make_service(lease, queued_orphan_timeout=450)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        row = await _status_of(db_session, "pending-dead")
        assert row is not None and row.status == "failed"

    async def test_second_sweep_never_reaches_a_stale_processing_job_twice(self, db_session):
        """The sentinel `age` on the orphan sweep must never re-select a
        'processing' row -- the original sweep (`self._stale_params`) stays
        the sole authority for that branch. A dedupe-by-id would mask this if
        the sentinel were wrong, so this asserts the row is reaped exactly
        once (mark_failed raises JobStateConflictError on a second hit,
        which would surface as an ERROR log, not a second reap -- the status
        assertion alone proves no double-processing broke anything)."""
        silent = datetime.now(UTC) - timedelta(seconds=700)
        await _seed(
            db_session,
            job_id="processing-dead",
            tenant_id=TENANT_A,
            status="processing",
            updated_at=silent,
        )

        lease = _lease()
        svc = _make_service(lease, queued_orphan_timeout=450)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        row = await _status_of(db_session, "processing-dead")
        assert row is not None and row.status == "failed"
        assert "over 600s" in (row.error_message or "")
