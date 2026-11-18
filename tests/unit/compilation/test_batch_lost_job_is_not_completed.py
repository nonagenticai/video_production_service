"""A batch that silently LOST jobs must never report ``completed``.

``_run_job`` accounts for a job in exactly two places: ``completed_jobs`` on the
success path and ``failed_jobs`` in its ``except Exception`` handler.
``asyncio.CancelledError`` is a ``BaseException``, so it escapes that handler,
is swallowed by ``asyncio.gather(..., return_exceptions=True)``, and increments
NEITHER counter.

The old terminal-status decision read ``failed_jobs`` alone, so
``completed_jobs=2, failed_jobs=0, total=5`` became ``status="completed"`` with
``completed_at`` stamped and ``errors=[]`` — served verbatim by
``GET /api/compile/batches/{id}``. The orchestrator stops polling and advances
while three videos were never produced.

These tests exercise the REAL ``asyncio.gather`` path: every fake ``process_job``
awaits ``asyncio.sleep(0)`` before it does anything, so the tasks genuinely
suspend and interleave — a fake that never yields could not reproduce this.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.compilation.batch_service import BatchService
from tests.conftest import TENANT_U1


def _make_db_session_factory():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_session), mock_session


def _make_batch(batch_id="batch-lost", job_ids=None):
    batch = MagicMock()
    batch.batch_id = batch_id
    batch.status = "pending"
    batch.job_ids = list(job_ids or [])
    batch.completed_jobs = 0
    batch.failed_jobs = 0
    batch.errors = []
    batch.updated_at = None
    batch.completed_at = None
    return batch


def _compiler_with(outcomes: dict[str, type[BaseException] | None]):
    """Compiler whose ``process_job`` really suspends before its outcome."""
    compiler = AsyncMock()

    async def _process_job(_db, job_id, _tenant_id):
        # Real suspension point: the tasks interleave inside gather().
        await asyncio.sleep(0)
        exc = outcomes[job_id]
        if exc is not None:
            raise exc(f"{job_id} lost")
        return

    compiler.process_job = AsyncMock(side_effect=_process_job)
    return compiler


class TestLostJobAccounting:
    async def test_cancelled_job_does_not_yield_completed(self):
        """2 of 5 jobs done, 3 lost to CancelledError -> NOT 'completed'."""
        job_ids = ["j1", "j2", "j3", "j4", "j5"]
        compiler = _compiler_with(
            {
                "j1": None,
                "j2": None,
                "j3": asyncio.CancelledError,
                "j4": asyncio.CancelledError,
                "j5": asyncio.CancelledError,
            }
        )
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=job_ids)
        svc = BatchService(compiler, factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(AsyncMock(), batch_id="batch-lost", tenant_id=TENANT_U1)

        assert batch.completed_jobs == 2
        assert batch.status != "completed", (
            "a batch that produced 2 of 5 videos was reported finished-clean"
        )
        assert batch.status == "partial"

    async def test_lost_jobs_are_named_in_errors(self):
        """Every lost job must leave an error entry naming it."""
        job_ids = ["j1", "j2", "j3"]
        compiler = _compiler_with(
            {"j1": None, "j2": asyncio.CancelledError, "j3": asyncio.CancelledError}
        )
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=job_ids)
        svc = BatchService(compiler, factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(AsyncMock(), batch_id="batch-lost", tenant_id=TENANT_U1)

        named = {e.get("job_id") for e in batch.errors}
        assert {"j2", "j3"} <= named, batch.errors
        assert all(e.get("source") == "lost" for e in batch.errors if e.get("job_id") != "j1")

    async def test_every_job_lost_sets_failed(self):
        """Zero jobs produced -> 'failed', not 'completed'."""
        job_ids = ["j1", "j2"]
        compiler = _compiler_with(dict.fromkeys(job_ids, asyncio.CancelledError))
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=job_ids)
        svc = BatchService(compiler, factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(AsyncMock(), batch_id="batch-lost", tenant_id=TENANT_U1)

        assert batch.completed_jobs == 0
        assert batch.status == "failed"

    async def test_real_cancellation_of_the_batch_still_propagates(self):
        """Accounting for lost jobs must NOT swallow a genuine cancellation."""
        started = asyncio.Event()

        compiler = AsyncMock()

        async def _process_job(_db, _job_id, _tenant_id):
            started.set()
            await asyncio.sleep(3600)

        compiler.process_job = AsyncMock(side_effect=_process_job)
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=["j1", "j2"])
        svc = BatchService(compiler, factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            task = asyncio.create_task(
                svc.process_batch(AsyncMock(), batch_id="batch-lost", tenant_id=TENANT_U1)
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert batch.completed_at is None


class TestDoubleCountedJobAccounting:
    """A job counted in BOTH counters must never resolve to ``completed``.

    ``_run_job`` increments ``completed_jobs`` and only THEN flushes. If that
    flush raises, the ``except Exception`` handler increments ``failed_jobs`` for
    the very same job, so ``completed_jobs + failed_jobs > total_jobs`` and
    ``completed_jobs`` no longer proves a video was produced. Deriving the status
    from ``completed_jobs >= total_jobs`` alone reports ``completed`` for a job
    whose success was never durably written — the exact optimistic terminal
    status this module exists to stop. ``origin/main`` reported ``failed`` here.
    """

    @staticmethod
    def _db_failing_flush_on(nth: int):
        """AsyncSession double whose ``nth`` ``flush()`` (1-based) raises."""
        db = AsyncMock()
        calls = {"n": 0}

        async def _flush(*_a, **_kw):
            calls["n"] += 1
            if calls["n"] == nth:
                raise RuntimeError("progress flush lost the connection")

        db.flush = AsyncMock(side_effect=_flush)
        return db

    async def test_completed_then_failed_flush_is_not_completed(self):
        compiler = _compiler_with({"j1": None})
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=["j1"])
        svc = BatchService(compiler, factory)
        # flush #1 is the 'processing' flip; #2 is the success-path progress write.
        db = self._db_failing_flush_on(2)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-lost", tenant_id=TENANT_U1)

        assert batch.completed_jobs == 1
        assert batch.failed_jobs == 1
        assert batch.status != "completed", (
            "one job was counted as BOTH completed and failed, yet the batch "
            "reported finished-clean"
        )
        assert batch.status == "failed"

    async def test_double_count_is_named_in_errors(self):
        compiler = _compiler_with({"j1": None})
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=["j1"])
        svc = BatchService(compiler, factory)
        db = self._db_failing_flush_on(2)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-lost", tenant_id=TENANT_U1)

        anomalies = [e for e in batch.errors if e.get("source") == "lost"]
        assert anomalies, batch.errors
        assert any("counted twice" in e["error"] for e in anomalies), batch.errors

    async def test_partial_batch_with_a_double_count_is_not_completed(self):
        """3 jobs: j1 clean, j2 double-counted, j3 clean -> at most 'partial'."""
        compiler = _compiler_with({"j1": None, "j2": None, "j3": None})
        factory, _ = _make_db_session_factory()
        batch = _make_batch(job_ids=["j1", "j2", "j3"])
        svc = BatchService(compiler, factory)
        # #1 processing flip, #2 first success write, #3 second success write fails.
        db = self._db_failing_flush_on(3)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-lost", tenant_id=TENANT_U1)

        assert batch.completed_jobs == 3
        assert batch.failed_jobs == 1
        assert batch.status == "partial", batch.status
