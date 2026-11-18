"""D1: a batch that is reaped and then finishes must not keep the reaper's error.

The batch stale window is a WALL-CLOCK CEILING, not a no-progress window:
``process_batch`` opens one transaction for the batch's whole life and never commits
between the 'processing' flip and the final status write, so from any other
connection ``updated_at`` is frozen at batch start however much progress is made.
The row lock (plus ``SKIP LOCKED`` in the reaper) protects a batch once its first
child job has flushed — but a batch whose first child has not finished inside the
window IS reapable mid-flight.

When that batch then finishes successfully, the owning task writes ``status`` but
would not write ``errors`` at all (it never reassigned the attribute), so the
reaper's "the owning process is gone" entry survived FOREVER on a batch that
succeeded. Asserted here on the resulting ROW STATE.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from sqlalchemy import select

from src.database.models import CompilationBatch
from src.services.compilation.batch_service import BatchService
from src.services.compilation.cleanup_service import CleanupService
from tests.conftest import TENANT_A


async def _seed(db, batch_id, *, errors, minutes_old=600):
    ts = datetime.now(UTC) - timedelta(minutes=minutes_old)
    db.add(
        CompilationBatch(
            batch_id=batch_id,
            tenant_id=TENANT_A,
            status="processing",
            total_jobs=0,
            completed_jobs=0,
            failed_jobs=0,
            job_ids=[],
            errors=errors,
            created_at=ts,
            updated_at=ts,
        )
    )
    await db.commit()


async def _row(db, batch_id):
    db.expire_all()
    return (
        await db.execute(select(CompilationBatch).where(CompilationBatch.batch_id == batch_id))
    ).scalar_one_or_none()


def _cleanup_service():
    with patch("src.services.compilation.cleanup_service.get_settings") as mock_gs:
        mock_gs.return_value = MagicMock(
            cleanup=MagicMock(
                enabled=True, interval_seconds=3600, completed_ttl_days=30, failed_ttl_days=7
            ),
            storage=MagicMock(
                stale_job_timeout_seconds=600,
                compilation_pending_stale_seconds=21600,
                compilation_batch_stale_seconds=3600,
            ),
        )
        return CleanupService()


def _batch_service(db_session_factory):
    return BatchService(MagicMock(), db_session_factory)


class TestReapedBatchThatFinishesAnyway:
    async def test_successful_finish_strips_the_reapers_error_entry(
        self, db_session, db_session_factory
    ):
        """End to end: reaped mid-flight, then the owner finishes successfully."""
        bid = f"batch-revived-{uuid.uuid4()}"
        await _seed(db_session, bid, errors=[])

        await _cleanup_service()._reap_stale_batches()
        reaped = await _row(db_session, bid)
        assert reaped.status == "failed"
        assert any(e.get("source") == "reaper" for e in reaped.errors)

        # The owning task knows nothing about the reap and finishes normally.
        await _batch_service(db_session_factory).process_batch(db_session, bid, TENANT_A)

        row = await _row(db_session, bid)
        assert row.status == "completed"
        assert row.errors == []

    async def test_real_per_job_errors_survive_the_strip(self, db_session, db_session_factory):
        """Only the reaper's own entries go — a child job's failure is evidence."""
        bid = f"batch-mixed-{uuid.uuid4()}"
        await _seed(
            db_session,
            bid,
            errors=[
                {"job_id": "j1", "error": "ffmpeg exploded"},
                {"error": "Batch reaped: stuck in 'processing'", "source": "reaper"},
            ],
        )

        await _batch_service(db_session_factory).process_batch(db_session, bid, TENANT_A)

        row = await _row(db_session, bid)
        assert row.errors == [{"job_id": "j1", "error": "ffmpeg exploded"}]
