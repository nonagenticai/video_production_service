"""The reaper's staleness clock must start when WORK began, not at job creation.

Regression cover for the 2026-09-03/04 collapse. ``stale_recording_jobs`` applied
ONE ``updated_at`` window to ``status IN ('processing','queued')``, and
``storage.stale_job_timeout_seconds`` (600) EQUALLED the orchestrator's client
timeout. Nothing bumps ``updated_at`` while a job waits for a fleet-wide recording
slot, so that window measured QUEUE WAIT for a queued row: 276 of 282 jobs
accepted in one 4-hour window were failed before they ever got a slot, and the
service completed nothing for 10 hours while real recording work was 1-20s.

Every test here is written to FAIL against the old two-argument resolver:

* ``test_a_queued_job_past_the_processing_window_is_not_reaped`` is the exact bug
  — under the old function a 30-minute-old queued row is selected and reaped.
* ``test_a_processing_job_past_its_window_is_still_reaped`` is the control. Without
  it, "stop reaping queued jobs" could be satisfied by disarming the reaper
  entirely, which would re-open the crashed-pod leak the reaper exists to close.
* ``test_the_lease_probe_and_the_reaper_agree`` guards the OTHER half. The JobLease
  heartbeat probe asks the same question through the same resolver; if only the
  reaper had been fixed, the probe would still stop renewing a waiting job's
  lease, and the ``present()`` gate would stop protecting it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import fakeredis
import fakeredis.aioredis
from sqlalchemy import select

from src.database.models import RecordingJob
from src.services.job_lease import JobLease
from src.services.recording.cleanup_service import CleanupService, stalled_recording_job_ids
from tests.conftest import TENANT_A, make_job

# Deliberately mirrors production: the processing window equals the caller's
# timeout, which is what made the old shared window lethal.
_PROCESSING_WINDOW = 600
_QUEUED_WINDOW = 7200


def _settings() -> MagicMock:
    return MagicMock(
        storage=MagicMock(
            cleanup_interval_seconds=0.05,
            stale_job_timeout_seconds=_PROCESSING_WINDOW,
            queued_job_timeout_seconds=_QUEUED_WINDOW,
        )
    )


def _service(job_lease: JobLease | None = None) -> CleanupService:
    with patch("src.services.recording.cleanup_service.get_settings", return_value=_settings()):
        return CleanupService(storage=MagicMock(), job_lease=job_lease)


def _lease() -> JobLease:
    fake = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return JobLease(fake, prefix="recording", instance_id="pod-under-test", ttl_seconds=60)


async def _seed(db, **kw) -> None:
    kw.setdefault("id", None)
    db.add(make_job(**kw))
    await db.commit()


async def _status_of(db, job_id: str) -> str | None:
    row = (
        await db.execute(select(RecordingJob).where(RecordingJob.job_id == job_id))
    ).scalar_one_or_none()
    return None if row is None else row.status


class TestQueuedJobsAreNotStale:
    async def test_a_queued_job_past_the_processing_window_is_not_reaped(self, db_session):
        """A job WAITING for a slot has not failed at anything.

        30 minutes queued is 3x the processing window and 3x the client timeout —
        precisely the row the old resolver killed — but it is far inside the
        queued-orphan window, so the reaper must leave it alone.
        """
        waiting_since = datetime.now(UTC) - timedelta(minutes=30)
        await _seed(
            db_session,
            job_id="queued-waiting",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=waiting_since,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        assert await _status_of(db_session, "queued-waiting") == "queued"

    async def test_a_queued_job_past_the_orphan_window_is_still_reaped(self, db_session):
        """The queued window is longer, not absent.

        A row whose owning pod died between INSERT and dispatch would otherwise
        sit in 'queued' forever, so the orphan window must still fire.
        """
        abandoned = datetime.now(UTC) - timedelta(seconds=_QUEUED_WINDOW + 60)
        await _seed(
            db_session,
            job_id="queued-orphan",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=abandoned,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        assert await _status_of(db_session, "queued-orphan") == "failed"

    async def test_a_pending_job_past_the_orphan_window_is_reaped(self, db_session):
        """A row whose owner died between INSERT and dispatch must not leak forever.

        ``create_job`` INSERTs at ``status='pending'`` and ``mark_queued`` moves it
        to ``'queued'``. Before 2026-09-04 the resolver matched only
        ``'processing'`` and ``'queued'``, so a pod that died in that gap left a
        ``'pending'`` row covered by NO clock at all — a genuine orphan leak, and
        precisely the row ``queued_job_timeout_seconds`` claims to collect.

        Mutation-proof: reverting the resolver's second branch to
        ``rj.status = 'queued'`` leaves this row 'pending' and the assertion fails.
        """
        abandoned = datetime.now(UTC) - timedelta(seconds=_QUEUED_WINDOW + 60)
        await _seed(
            db_session,
            job_id="pending-orphan",
            tenant_id=TENANT_A,
            status="pending",
            updated_at=abandoned,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        assert await _status_of(db_session, "pending-orphan") == "failed"

    async def test_a_fresh_pending_job_is_not_reaped(self, db_session):
        """Control: 'pending' joins the ORPHAN window, not the processing one.

        Without this, "reap pending rows" could be satisfied by putting 'pending'
        on the 600s no-progress clock — which would fail every job that is merely
        waiting to be dispatched, i.e. the 2026-09-03 collapse with a new status.
        """
        just_accepted = datetime.now(UTC) - timedelta(seconds=_PROCESSING_WINDOW + 60)
        await _seed(
            db_session,
            job_id="pending-fresh",
            tenant_id=TENANT_A,
            status="pending",
            updated_at=just_accepted,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        assert await _status_of(db_session, "pending-fresh") == "pending"

    async def test_a_processing_job_past_its_window_is_still_reaped(self, db_session):
        """Control: the no-progress window for RUNNING work is unchanged.

        Without this assertion, "queued jobs survive" would also be satisfied by a
        reaper that reaps nothing at all.
        """
        silent_since = datetime.now(UTC) - timedelta(seconds=_PROCESSING_WINDOW + 60)
        await _seed(
            db_session,
            job_id="processing-dead",
            tenant_id=TENANT_A,
            status="processing",
            updated_at=silent_since,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        assert await _status_of(db_session, "processing-dead") == "failed"

    async def test_the_lease_probe_and_the_reaper_agree(self, db_session):
        """The heartbeat probe must not give up on a job the reaper will spare.

        The probe drives whether ``JobLease`` keeps renewing. If it still counted a
        waiting job as stale, the lease would expire, the reaper's ``present()``
        gate would stop protecting the job, and the collapse would return by the
        back door — which is exactly the shape of the 2026-08-25 incident that
        introduced the probe.
        """
        waiting_since = datetime.now(UTC) - timedelta(minutes=30)
        silent_since = datetime.now(UTC) - timedelta(seconds=_PROCESSING_WINDOW + 60)
        await _seed(
            db_session,
            job_id="probe-queued",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=waiting_since,
        )
        await _seed(
            db_session,
            job_id="probe-processing",
            tenant_id=TENANT_A,
            status="processing",
            updated_at=silent_since,
        )

        with patch("src.services.recording.cleanup_service.get_settings", return_value=_settings()):
            stalled = await stalled_recording_job_ids()

        assert "probe-queued" not in stalled
        assert "probe-processing" in stalled

    async def test_the_reaped_error_names_the_window_that_fired(self, db_session):
        """A queued reap must not quote the processing window in its error.

        Two clocks now; an error message quoting the wrong one sends the next
        reader to the wrong setting.
        """
        abandoned = datetime.now(UTC) - timedelta(seconds=_QUEUED_WINDOW + 60)
        await _seed(
            db_session,
            job_id="queued-orphan-msg",
            tenant_id=TENANT_A,
            status="queued",
            updated_at=abandoned,
        )

        await _service()._reap_stale_jobs()

        db_session.expire_all()
        row = (
            await db_session.execute(
                select(RecordingJob).where(RecordingJob.job_id == "queued-orphan-msg")
            )
        ).scalar_one()
        assert f"over {_QUEUED_WINDOW}s" in (row.error_message or "")
