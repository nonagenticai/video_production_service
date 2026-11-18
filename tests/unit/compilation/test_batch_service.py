"""Unit tests for batch_service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.compilation.batch_service import BatchService
from tests.conftest import TENANT_U1, TENANT_U3, TENANT_U4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_session_factory():
    """Return a MagicMock that behaves as an async-context-manager factory."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return MagicMock(return_value=mock_session), mock_session


def _make_batch(batch_id="batch-1", job_ids=None, status="pending"):
    """Return a MagicMock CompilationBatch with the required attributes."""
    batch = MagicMock()
    batch.batch_id = batch_id
    batch.status = status
    batch.job_ids = job_ids or []
    batch.completed_jobs = 0
    batch.failed_jobs = 0
    batch.errors = []
    batch.updated_at = None
    batch.completed_at = None
    return batch


# ---------------------------------------------------------------------------
# TestCreateBatch
# ---------------------------------------------------------------------------


class TestCreateBatch:
    @patch("src.services.compilation.batch_service.uuid.uuid7", return_value="batch-abc")
    @patch("src.services.compilation.batch_service.CompilationBatch")
    async def test_creates_batch_and_returns_batch_id(self, mock_batch_cls, _mock_uuid):
        """create_batch should create a batch record, jobs, and return the batch_id."""
        compiler_service = AsyncMock()
        compiler_service.create_job = AsyncMock(side_effect=["job-1", "job-2"])
        db_session_factory, _ = _make_db_session_factory()
        db = AsyncMock()
        db.add = MagicMock()

        mock_batch_instance = MagicMock()
        mock_batch_instance.job_ids = []
        mock_batch_cls.return_value = mock_batch_instance

        svc = BatchService(compiler_service, db_session_factory)
        configs = [{"preset": "hd"}, {"preset": "sd"}]
        result = await svc.create_batch(db, tenant_id=TENANT_U4, configs=configs)

        assert result == "batch-abc"
        assert db.add.call_count == 1
        assert db.flush.call_count == 2
        assert compiler_service.create_job.call_count == 2

    @patch("src.services.compilation.batch_service.uuid.uuid7", return_value="batch-xyz")
    @patch("src.services.compilation.batch_service.CompilationBatch")
    async def test_job_ids_stored_on_batch(self, mock_batch_cls, _mock_uuid):
        """The batch record should hold all job_ids returned by create_job."""
        compiler_service = AsyncMock()
        compiler_service.create_job = AsyncMock(side_effect=["j-1", "j-2", "j-3"])
        db_session_factory, _ = _make_db_session_factory()
        db = AsyncMock()
        db.add = MagicMock()

        mock_batch_instance = MagicMock()
        mock_batch_instance.job_ids = []
        mock_batch_cls.return_value = mock_batch_instance

        svc = BatchService(compiler_service, db_session_factory)
        configs = [{"a": 1}, {"b": 2}, {"c": 3}]
        await svc.create_batch(db, tenant_id=TENANT_U3, configs=configs)

        assert mock_batch_instance.job_ids == ["j-1", "j-2", "j-3"]


# ---------------------------------------------------------------------------
# TestProcessBatch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    async def test_all_jobs_succeed_sets_completed(self):
        """When every job succeeds the final batch status should be 'completed'."""
        compiler_service = AsyncMock()
        db_session_factory, _mock_session = _make_db_session_factory()
        db = AsyncMock()

        batch = _make_batch(job_ids=["j1", "j2"])
        svc = BatchService(compiler_service, db_session_factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-1", tenant_id=TENANT_U1)

        assert batch.status == "completed"
        assert batch.completed_jobs == 2
        assert batch.failed_jobs == 0
        assert batch.completed_at is not None

    async def test_all_jobs_fail_sets_failed(self):
        """When every job raises the final batch status should be 'failed'."""
        compiler_service = AsyncMock()
        compiler_service.process_job = AsyncMock(side_effect=RuntimeError("boom"))
        db_session_factory, _mock_session = _make_db_session_factory()
        db = AsyncMock()

        batch = _make_batch(job_ids=["j1", "j2"])
        svc = BatchService(compiler_service, db_session_factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-1", tenant_id=TENANT_U1)

        assert batch.status == "failed"
        assert batch.failed_jobs == 2
        assert batch.completed_jobs == 0
        assert len(batch.errors) == 2

    async def test_partial_failure_sets_partial(self):
        """When some jobs succeed and others fail the status should be 'partial'."""
        compiler_service = AsyncMock()
        # First call succeeds, second raises
        compiler_service.process_job = AsyncMock(side_effect=[None, RuntimeError("oops")])
        db_session_factory, _mock_session = _make_db_session_factory()
        db = AsyncMock()

        batch = _make_batch(job_ids=["j1", "j2"])
        svc = BatchService(compiler_service, db_session_factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=batch)):
            await svc.process_batch(db, batch_id="batch-1", tenant_id=TENANT_U1)

        assert batch.status == "partial"
        assert batch.completed_jobs == 1
        assert batch.failed_jobs == 1

    async def test_not_found_raises_value_error(self):
        """process_batch should raise ValueError when _get_batch returns None."""
        compiler_service = AsyncMock()
        db_session_factory, _ = _make_db_session_factory()
        db = AsyncMock()

        svc = BatchService(compiler_service, db_session_factory)

        with patch.object(svc, "_get_batch", new=AsyncMock(return_value=None)):
            with pytest.raises(ValueError, match="Batch not found"):
                await svc.process_batch(db, batch_id="missing", tenant_id=TENANT_U1)


# ---------------------------------------------------------------------------
# TestGetBatchStatus
# ---------------------------------------------------------------------------


class TestGetBatchStatus:
    async def test_found_returns_batch(self):
        """get_batch_status returns the batch when found in the DB."""
        fake_batch = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_batch

        db = AsyncMock()
        db.execute.return_value = mock_result

        result = await BatchService.get_batch_status(db, batch_id="b-1")
        assert result is fake_batch

    async def test_not_found_returns_none(self):
        """get_batch_status returns None when no batch matches."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = AsyncMock()
        db.execute.return_value = mock_result

        result = await BatchService.get_batch_status(db, batch_id="nope")
        assert result is None
