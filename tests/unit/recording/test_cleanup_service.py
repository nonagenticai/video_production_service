"""Tests for src/services/cleanup_service.py -- CleanupService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select

from src.database.models import RecordingJob
from src.services.recording.cleanup_service import CleanupService
from tests.conftest import TENANT_A, TENANT_B, make_job


async def _status_of(db, job_id):
    return (
        await db.execute(select(RecordingJob).where(RecordingJob.job_id == job_id))
    ).scalar_one_or_none()


async def _seed(db, **kw):
    # id=None → let the DB autoincrement (make_job defaults id=1, which collides
    # when seeding multiple rows in one test).
    kw.setdefault("id", None)
    db.add(make_job(**kw))
    await db.commit()


# ── Helpers ────────────────────────────────────────────────────────


def _make_cleanup_service(
    storage=None,
    interval=0.05,
    stale_timeout=600,
    queued_timeout=7200,
    stale_reap_interval=0.02,
    queued_orphan_timeout=450,
):
    """Build a CleanupService with mocked settings.

    ``queued_job_timeout_seconds`` is REQUIRED, not optional: the reaper binds it
    as a query parameter, and a MagicMock stand-in fails to adapt in psycopg. The
    resulting ProgrammingError is swallowed by ``_reap_stale_jobs``'s own
    ``except Exception``, so the omission reads as "reaped nothing" rather than as
    an error -- which is how this fixture would silently stop testing anything.
    ``stale_reap_interval_seconds`` is required for the same reason once
    ``start()`` is exercised: ``_run_stale_loop`` feeds it straight to
    ``asyncio.sleep``, which raises TypeError on a MagicMock -- and that failure
    sits OUTSIDE `_reap_stale_jobs`'s own try/except, so it is not swallowed.
    """
    storage = storage or AsyncMock()
    with patch("src.services.recording.cleanup_service.get_settings") as mock_gs:
        mock_settings = MagicMock(
            storage=MagicMock(
                cleanup_interval_seconds=interval,
                stale_reap_interval_seconds=stale_reap_interval,
                stale_job_timeout_seconds=stale_timeout,
                queued_job_timeout_seconds=queued_timeout,
                queued_orphan_timeout_seconds=queued_orphan_timeout,
            )
        )
        mock_gs.return_value = mock_settings
        return CleanupService(storage=storage)


# ── start / stop ─────────────────────────────────────────────────


class TestStartStop:
    async def test_start_creates_task(self):
        svc = _make_cleanup_service()
        await svc.start()
        assert svc._task is not None
        assert not svc._task.done()
        await svc.stop()

    async def test_stop_cancels_task(self):
        svc = _make_cleanup_service()
        await svc.start()
        task = svc._task
        await svc.stop()
        assert svc._task is None
        assert task.cancelled() or task.done()

    async def test_stop_without_start_is_noop(self):
        svc = _make_cleanup_service()
        await svc.stop()  # Should not raise
        assert svc._task is None


# ── _cleanup_expired ─────────────────────────────────────────────


class TestCleanupExpired:
    async def test_cleans_up_expired_jobs(self, db_session):
        """Real DB: the cross-tenant resolver finds the expired job; mark_expired flags it."""
        past = datetime.now(UTC) - timedelta(days=1)
        await _seed(
            db_session,
            job_id="expired-j",
            tenant_id=TENANT_A,
            status="completed",
            expired=False,
            expires_at=past,
        )
        storage = AsyncMock()
        svc = _make_cleanup_service(storage=storage)

        await svc._cleanup_expired()

        storage.cleanup_job.assert_awaited_once_with("expired-j")
        db_session.expire_all()
        job = await _status_of(db_session, "expired-j")
        assert job is not None and job.expired is True

    async def test_handles_individual_job_cleanup_error(self, db_session):
        """If one job's storage cleanup fails, others are still marked expired."""
        past = datetime.now(UTC) - timedelta(days=1)
        await _seed(
            db_session,
            job_id="j1",
            tenant_id=TENANT_A,
            status="completed",
            expired=False,
            expires_at=past,
        )
        await _seed(
            db_session,
            job_id="j2",
            tenant_id=TENANT_B,
            status="completed",
            expired=False,
            expires_at=past,
        )
        storage = AsyncMock()
        # First job's storage cleanup raises, second succeeds.
        storage.cleanup_job = AsyncMock(
            side_effect=lambda jid: (
                (_ for _ in ()).throw(RuntimeError("disk")) if jid == "j1" else None
            )
        )
        svc = _make_cleanup_service(storage=storage)

        await svc._cleanup_expired()

        assert storage.cleanup_job.await_count == 2
        db_session.expire_all()
        # j1 storage failed before mark → not flagged; j2 flagged expired.
        assert (await _status_of(db_session, "j1")).expired is False
        assert (await _status_of(db_session, "j2")).expired is True

    async def test_no_expired_jobs_returns_early(self, db_session):
        # No expired jobs seeded → nothing to clean.
        storage = AsyncMock()
        svc = _make_cleanup_service(storage=storage)
        await svc._cleanup_expired()
        storage.cleanup_job.assert_not_awaited()

    @patch("src.services.recording.cleanup_service.get_db_session")
    async def test_handles_db_session_error(self, mock_get_db):
        """If the DB session fails, the cleanup cycle should not crash."""
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db connect failed"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_db.return_value = mock_ctx

        storage = AsyncMock()
        svc = _make_cleanup_service(storage=storage)

        # Should not raise (just log)
        await svc._cleanup_expired()
        storage.cleanup_job.assert_not_awaited()


# ── _reap_stale_jobs ─────────────────────────────────────────────


class TestReapStaleJobs:
    async def test_reaps_stale_jobs(self, db_session):
        """Real DB: the cross-tenant resolver finds the stale job; mark_failed reaps it."""
        old = datetime.now(UTC) - timedelta(hours=1)
        await _seed(
            db_session, job_id="stale-j", tenant_id=TENANT_A, status="processing", updated_at=old
        )
        # A fresh processing job must NOT be reaped.
        await _seed(db_session, job_id="fresh-j", tenant_id=TENANT_A, status="processing")

        svc = _make_cleanup_service(stale_timeout=600)
        await svc._reap_stale_jobs()

        db_session.expire_all()
        stale = await _status_of(db_session, "stale-j")
        assert stale.status == "failed"
        assert "reaped" in (stale.error_message or "").lower()
        assert (await _status_of(db_session, "fresh-j")).status == "processing"

    async def test_no_stale_jobs_returns_early(self, db_session):
        # No stale jobs seeded → nothing reaped (a fresh job is left alone).
        await _seed(db_session, job_id="fresh-j", tenant_id=TENANT_A, status="processing")
        svc = _make_cleanup_service(stale_timeout=600)
        await svc._reap_stale_jobs()
        db_session.expire_all()
        assert (await _status_of(db_session, "fresh-j")).status == "processing"

    @patch("src.services.recording.cleanup_service.get_db_session")
    async def test_handles_db_session_error(self, mock_get_db):
        """If the DB session fails, the reaper should not crash."""
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=RuntimeError("db connect failed"))
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_get_db.return_value = mock_ctx

        svc = _make_cleanup_service()
        await svc._reap_stale_jobs()  # Should not raise


# ── _run_loop ────────────────────────────────────────────────────


class TestRunLoop:
    async def test_loop_runs_cleanup_and_stops_on_cancel(self):
        """Verify the loop invokes _cleanup_expired and stops gracefully."""
        svc = _make_cleanup_service(interval=0.02)
        cleanup_calls = 0

        async def fake_cleanup():
            nonlocal cleanup_calls
            cleanup_calls += 1

        svc._cleanup_expired = fake_cleanup
        svc._reap_stale_jobs = AsyncMock()

        await svc.start()
        await asyncio.sleep(0.1)  # Let a few iterations run
        await svc.stop()

        assert cleanup_calls >= 1

    async def test_loop_survives_cleanup_exception(self):
        """Loop should continue even if _cleanup_expired raises."""
        svc = _make_cleanup_service(interval=0.02)
        call_count = 0

        async def failing_cleanup():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")

        svc._cleanup_expired = failing_cleanup
        svc._reap_stale_jobs = AsyncMock()

        await svc.start()
        await asyncio.sleep(0.1)
        await svc.stop()

        # Should have been called multiple times despite first error
        assert call_count >= 2
