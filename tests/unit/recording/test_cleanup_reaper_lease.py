"""Lease-gated recording stale-reaper tests (real Postgres + fakeredis).

The age-based reaper must SKIP a stale job whose Redis liveness lease is still
present (it is running on another pod) and only reap a genuinely orphaned one
(crashed pod, lease gone) — exactly once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import fakeredis
import fakeredis.aioredis
from sqlalchemy import select

from src.database.models import RecordingJob
from src.services.job_lease import JobLease
from src.services.recording.cleanup_service import CleanupService
from tests.conftest import TENANT_A, make_job


async def _status_of(db, job_id):
    return (
        await db.execute(select(RecordingJob).where(RecordingJob.job_id == job_id))
    ).scalar_one_or_none()


async def _seed(db, **kw):
    kw.setdefault("id", None)
    db.add(make_job(**kw))
    await db.commit()


def _make_service(job_lease, stale_timeout=600, queued_timeout=7200, queued_orphan_timeout=450):
    """``queued_orphan_timeout_seconds`` must be a real int whenever ``job_lease``
    is not None: `_reap_stale_jobs` binds it as a SECOND query's parameter
    (the lease-aware orphan sweep) any time a lease is configured, regardless
    of what status the seeded rows carry -- a MagicMock stand-in fails to
    adapt in psycopg exactly like the omissions `test_cleanup_service.py`
    documents.
    """
    with patch("src.services.recording.cleanup_service.get_settings") as mock_gs:
        mock_gs.return_value = MagicMock(
            storage=MagicMock(
                cleanup_interval_seconds=0.05,
                stale_reap_interval_seconds=0.02,
                stale_job_timeout_seconds=stale_timeout,
                queued_job_timeout_seconds=queued_timeout,
                queued_orphan_timeout_seconds=queued_orphan_timeout,
            )
        )
        return CleanupService(storage=MagicMock(), job_lease=job_lease)


def _lease() -> JobLease:
    fake = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return JobLease(fake, prefix="recording", instance_id="pod-under-test", ttl_seconds=60)


class TestReaperLeaseGate:
    async def test_skips_stale_job_with_live_lease(self, db_session):
        """A stale processing job that still holds a live lease (running on another
        pod) is NOT reaped."""
        old = datetime.now(UTC) - timedelta(hours=1)
        await _seed(
            db_session, job_id="live-j", tenant_id=TENANT_A, status="processing", updated_at=old
        )

        lease = _lease()
        await lease.acquire("live-j")  # simulate the owning pod's live lease
        svc = _make_service(lease)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        # Still processing — the live lease protected it.
        assert (await _status_of(db_session, "live-j")).status == "processing"
        await lease.release_all()

    async def test_reaps_orphan_without_lease_exactly_once(self, db_session):
        """A stale processing job with NO live lease (crashed pod) is failed once."""
        old = datetime.now(UTC) - timedelta(hours=1)
        await _seed(
            db_session, job_id="orphan-j", tenant_id=TENANT_A, status="processing", updated_at=old
        )

        lease = _lease()  # no acquire → present() is False
        svc = _make_service(lease)

        await svc._reap_stale_jobs()
        db_session.expire_all()
        reaped = await _status_of(db_session, "orphan-j")
        assert reaped.status == "failed"
        assert "reaped" in (reaped.error_message or "").lower()

        # A second sweep must be a safe no-op (job already terminal).
        await svc._reap_stale_jobs()
        db_session.expire_all()
        assert (await _status_of(db_session, "orphan-j")).status == "failed"
        await lease.release_all()
