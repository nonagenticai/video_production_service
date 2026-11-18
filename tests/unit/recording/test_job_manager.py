"""Unit tests for RecordingJobManager — mock-only create/get paths.

State-transition coverage lives in ``tests/unit/test_job_managers_real_db.py``.
This file targets ``create_job`` and ``get_job`` which shape ORM objects
or query results without atomic UPDATE semantics, plus ``record_usage``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

from src.database.models import RecordingJob
from src.services.recording.job_manager import RecordingJobManager as JobManager
from tests.conftest import TENANT_U1


def _mock_db():
    db = AsyncMock()
    db.flush = AsyncMock()
    db.add = MagicMock()
    return db


class TestCreateJob:
    async def test_creates_job_and_flushes(self):
        db = _mock_db()
        jm = JobManager(db)
        job = await jm.create_job(tenant_id=TENANT_U1, part_id="p1", scripts_dir="/s")
        assert isinstance(job, RecordingJob)
        assert job.tenant_id == TENANT_U1
        assert job.part_id == "p1"
        assert job.status == "pending"
        assert job.retry_count == 0
        assert job.progress == 0.0
        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    async def test_default_part_type(self):
        db = _mock_db()
        jm = JobManager(db)
        job = await jm.create_job(tenant_id=TENANT_U1, part_id="p1")
        assert job.part_type == "browser_recording"

    async def test_custom_kwargs(self):
        db = _mock_db()
        jm = JobManager(db)
        job = await jm.create_job(
            tenant_id=TENANT_U1, part_id="p1", batch_id="b1", priority=5, max_retries=4
        )
        assert job.batch_id == "b1"
        assert job.priority == 5
        assert job.max_retries == 4


class TestGetJob:
    async def test_get_job_returns_scalar(self):
        mock_job = MagicMock(spec=RecordingJob)
        mock_job.job_id = "test-job-1"
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_job
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)

        jm = JobManager(db)
        got = await jm.get_job("test-job-1", tenant_id=TENANT_U1)
        assert got is mock_job

    async def test_get_job_missing_returns_none(self):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)

        jm = JobManager(db)
        got = await jm.get_job("nonexistent", tenant_id=TENANT_U1)
        assert got is None


def _usage_params(db):
    """Compile the INSERT ``record_usage`` handed to the session and return its bound params.

    ``record_usage`` never returns anything and never touches an ORM object — the
    only observable behaviour is the statement itself. These are BILLING numbers,
    so what matters is which quantities land in which columns, not that
    ``execute`` was called once. Asserting the compiled params is the closest a
    mock-only test can get to reading the persisted row.
    """
    db.execute.assert_awaited_once()
    return db.execute.await_args.args[0].compile(dialect=postgresql.dialect()).params


class TestRecordUsage:
    """record_usage uses sqlalchemy.dialects.postgresql.insert → mock-only."""

    async def test_record_usage_basic_bills_the_supplied_quantities(self):
        db = AsyncMock()
        jm = JobManager(db)
        await jm.record_usage(
            tenant_id=TENANT_U1,
            recordings_completed=1,
            duration_seconds=42.0,
            file_size_bytes=5_000_000,
        )
        params = _usage_params(db)
        assert params["tenant_id"] == TENANT_U1
        assert params["date"] == datetime.now(UTC).strftime("%Y-%m-%d")
        # One attempt is always counted; the completion is what the caller passed.
        assert params["recordings_started"] == 1
        assert params["recordings_completed"] == 1
        assert params["recordings_failed"] == 0
        assert params["total_duration_seconds"] == 42.0
        assert params["total_file_size_bytes"] == 5_000_000

    async def test_record_usage_is_error_bills_a_failure_not_a_completion(self):
        db = AsyncMock()
        jm = JobManager(db)
        await jm.record_usage(tenant_id=TENANT_U1, is_error=True)
        params = _usage_params(db)
        assert params["recordings_failed"] == 1
        assert params["recordings_completed"] == 0
        # A failed recording still consumed an attempt.
        assert params["recordings_started"] == 1
        assert params["total_duration_seconds"] == 0.0
        assert params["total_file_size_bytes"] == 0

    async def test_record_usage_upserts_by_incrementing_not_overwriting(self):
        """A second call on the same (tenant, date) must ADD to the existing row.

        Without ``on_conflict_do_update`` every recording after the first of a
        day would either raise on the unique constraint or reset the counters —
        both of which silently corrupt the daily bill.
        """
        db = AsyncMock()
        jm = JobManager(db)
        await jm.record_usage(tenant_id=TENANT_U1, recordings_completed=1, duration_seconds=7.5)
        db.execute.assert_awaited_once()
        sql = str(db.execute.await_args.args[0].compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT ON CONSTRAINT uq_recording_usage_tenant_date DO UPDATE" in sql
        assert "recordings_started = (recording_usage.recordings_started + " in sql
        assert (
            "total_duration_seconds = (recording_usage.total_duration_seconds "
            "+ excluded.total_duration_seconds)"
        ) in sql
