"""``compilation_batches`` reaper + prune tests (real Postgres).

Before 2026-09-06 the aggregate batch row had NO resolver, NO reaper and NO prune:
the only writer that ever advances it is the request-spawned
``BatchService.process_batch`` task, so a pod restart / OOM / uncaught exception left
the row at 'processing' (or 'pending') FOREVER. A client polling batch status waited
forever and the failure was never accounted.

Every assertion below is on the resulting ROW STATE — status, recorded reason,
presence of the row — never on a mock's call count.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from sqlalchemy import select

from src.database.models import CompilationBatch
from src.services.compilation.cleanup_service import CleanupService
from tests.conftest import TENANT_A

_BATCH_STALE = 3600


async def _seed(db, batch_id, *, status="processing", minutes_old=600, errors=None):
    ts = datetime.now(UTC) - timedelta(minutes=minutes_old)
    db.add(
        CompilationBatch(
            batch_id=batch_id,
            tenant_id=TENANT_A,
            status=status,
            total_jobs=2,
            completed_jobs=0,
            failed_jobs=0,
            job_ids=[],
            errors=errors if errors is not None else [],
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


def _make_service(*, batch_stale=_BATCH_STALE, completed_ttl=30, failed_ttl=7):
    with patch("src.services.compilation.cleanup_service.get_settings") as mock_gs:
        mock_gs.return_value = MagicMock(
            cleanup=MagicMock(
                enabled=True,
                interval_seconds=3600,
                completed_ttl_days=completed_ttl,
                failed_ttl_days=failed_ttl,
            ),
            storage=MagicMock(
                stale_job_timeout_seconds=600,
                compilation_pending_stale_seconds=21600,
                compilation_batch_stale_seconds=batch_stale,
            ),
        )
        return CleanupService()


class TestBatchReaper:
    async def test_reaps_orphaned_processing_batch(self, db_session):
        """The measured defect: an abandoned 'processing' batch is immortal."""
        bid = f"batch-proc-{uuid.uuid4()}"
        await _seed(db_session, bid, status="processing", minutes_old=600)

        await _make_service()._reap_stale_batches()

        row = await _row(db_session, bid)
        assert row.status == "failed"
        assert row.completed_at is not None
        assert "reaped" in str(row.errors).lower()

    async def test_reaps_orphaned_pending_batch(self, db_session):
        """A pod that died BEFORE process_batch ran leaves the row at 'pending'."""
        bid = f"batch-pend-{uuid.uuid4()}"
        await _seed(db_session, bid, status="pending", minutes_old=600)

        await _make_service()._reap_stale_batches()

        row = await _row(db_session, bid)
        assert row.status == "failed"
        assert "pending" in str(row.errors).lower()

    async def test_preserves_existing_errors(self, db_session):
        """The reason is APPENDED — a partially failed batch keeps its job errors."""
        bid = f"batch-err-{uuid.uuid4()}"
        await _seed(
            db_session,
            bid,
            status="processing",
            minutes_old=600,
            errors=[{"job_id": "j1", "error": "boom"}],
        )

        await _make_service()._reap_stale_batches()

        row = await _row(db_session, bid)
        assert row.status == "failed"
        assert len(row.errors) == 2
        assert row.errors[0]["error"] == "boom"

    async def test_leaves_fresh_batch_alone(self, db_session):
        """A batch inside its window is live work, not an orphan."""
        bid = f"batch-fresh-{uuid.uuid4()}"
        await _seed(db_session, bid, status="processing", minutes_old=1)

        await _make_service()._reap_stale_batches()

        assert (await _row(db_session, bid)).status == "processing"

    async def test_never_overwrites_a_terminal_batch(self, db_session):
        """A completed batch must not be rewritten to 'failed' by the reaper."""
        bid = f"batch-done-{uuid.uuid4()}"
        await _seed(db_session, bid, status="completed", minutes_old=600)

        await _make_service()._reap_stale_batches()

        assert (await _row(db_session, bid)).status == "completed"


class TestBatchPrune:
    async def test_prunes_expired_terminal_batch(self, db_session):
        bid = f"batch-old-{uuid.uuid4()}"
        await _seed(db_session, bid, status="completed", minutes_old=60 * 24 * 40)

        await _make_service()._cleanup_expired_batches()

        assert await _row(db_session, bid) is None

    async def test_keeps_terminal_batch_inside_ttl(self, db_session):
        bid = f"batch-recent-{uuid.uuid4()}"
        await _seed(db_session, bid, status="completed", minutes_old=60)

        await _make_service()._cleanup_expired_batches()

        assert await _row(db_session, bid) is not None

    async def test_never_prunes_a_non_terminal_batch(self, db_session):
        """An ancient 'processing' row is the reaper's business, never the pruner's:
        deleting it would destroy the evidence instead of accounting the failure."""
        bid = f"batch-live-{uuid.uuid4()}"
        await _seed(db_session, bid, status="processing", minutes_old=60 * 24 * 40)

        await _make_service()._cleanup_expired_batches()

        assert (await _row(db_session, bid)).status == "processing"


class TestBatchReaperDoesNotBlockOnALockedRow:
    """D2: a locked batch row must not stall the sweep.

    ``process_batch`` opens ONE transaction for the batch's whole life -- every
    per-job progress write is a bare ``flush()`` -- so from the first child job
    onward the OWNING transaction holds this row's lock for hours, legitimately.
    ``_reap_stale_batches`` is awaited inline in ``_run_loop``, so a plain
    ``FOR UPDATE`` here would block ``_cleanup_cycle`` and ``_reap_stale_jobs``
    behind it indefinitely: a fix for one immortal row wedging the sweeper that
    reaps all the others.

    The deadline below is the assertion. With ``SKIP LOCKED`` the sweep returns in
    well under a second; without it, it waits on the lock until this test's holder
    releases -- i.e. forever from the sweep's point of view.
    """

    async def test_sweep_returns_promptly_and_leaves_the_locked_row_alone(
        self, db_session, db_session_factory
    ):
        bid = f"batch-locked-{uuid.uuid4()}"
        await _seed(db_session, bid, status="processing", minutes_old=600)

        async with db_session_factory() as owner:
            # Hold the row lock exactly as a live process_batch does: acquired, not
            # committed, held for the duration.
            held = (
                await owner.execute(
                    select(CompilationBatch)
                    .where(CompilationBatch.batch_id == bid)
                    .with_for_update()
                )
            ).scalar_one()
            assert held.status == "processing"

            await asyncio.wait_for(_make_service()._reap_stale_batches(), timeout=15)

            # The live owner still holds it, so the batch was passed over, not failed.
            assert (await _row(db_session, bid)).status == "processing"
            await owner.rollback()

    async def test_a_locked_row_does_not_stop_an_unlocked_one_being_reaped(
        self, db_session, db_session_factory
    ):
        """The sweep must skip the locked row and still do its job on the rest."""
        locked_bid = f"batch-locked2-{uuid.uuid4()}"
        free_bid = f"batch-free-{uuid.uuid4()}"
        await _seed(db_session, locked_bid, status="processing", minutes_old=600)
        await _seed(db_session, free_bid, status="processing", minutes_old=600)

        async with db_session_factory() as owner:
            await owner.execute(
                select(CompilationBatch)
                .where(CompilationBatch.batch_id == locked_bid)
                .with_for_update()
            )

            await asyncio.wait_for(_make_service()._reap_stale_batches(), timeout=15)

            assert (await _row(db_session, locked_bid)).status == "processing"
            assert (await _row(db_session, free_bid)).status == "failed"
            await owner.rollback()
