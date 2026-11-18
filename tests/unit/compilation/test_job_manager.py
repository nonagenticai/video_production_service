"""Unit tests for CompilationJobManager — mock-only create/get/list paths.

State-transition coverage lives in ``tests/unit/test_job_managers_real_db.py``.
This file targets ``create_job``, ``get_job``, ``count_active_jobs`` and
``record_usage``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from sqlalchemy.dialects import postgresql

from src.services.compilation.job_manager import CompilationJobManager as JobManager
from tests.conftest import TENANT_U1, TENANT_U4


def _mock_db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


def _mock_job(**kwargs):
    job = MagicMock()
    job.job_id = kwargs.get("job_id", "test-job-1")
    job.tenant_id = kwargs.get("tenant_id", TENANT_U4)
    job.status = kwargs.get("status", "pending")
    job.created_at = kwargs.get("created_at", datetime.now(UTC))
    return job


class TestJobManagerCreateJob:
    async def test_create_job_returns_compilation_job(self):
        db = _mock_db()
        db.flush = AsyncMock()

        job = await JobManager.create_job(
            db,
            tenant_id=TENANT_U4,
            shot_list={"parts": []},
            recordings=[],
        )

        assert job.tenant_id == TENANT_U4
        assert job.status == "pending"
        assert job.shot_list == {"parts": []}
        db.add.assert_called_once()
        db.flush.assert_awaited_once()

    async def test_create_job_with_all_params(self):
        db = _mock_db()
        db.flush = AsyncMock()

        job = await JobManager.create_job(
            db,
            tenant_id=TENANT_U1,
            shot_list={"parts": [{"id": 1}]},
            recordings=[{"path": "/v.mp4"}],
            narration_audio_dir="/audio",
            clips_dir="/clips",
            transition_type="fade",
            crossfade_duration=1.5,
            target_width=1920,
            target_height=1080,
            crf=23,
            theme_config={"bg": "#000"},
            edit_decisions=[{"t": 0}],
            preset="hd",
            webhook_url="https://hook.example.com",
            webhook_secret="secret",
            filter_config={"saturation": 1.2},
            watermark_config={"path": "/wm.png"},
            subtitles={"lang": "en"},
        )

        assert job.preset == "hd"
        assert job.webhook_url == "https://hook.example.com"
        assert job.subtitles == {"lang": "en"}


class TestJobManagerGetJob:
    async def test_get_job_returns_scalar(self):
        mock_job = _mock_job()
        result = MagicMock()
        result.scalar_one_or_none.return_value = mock_job
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)

        got = await JobManager.get_job(db, "test-job-1")
        assert got is mock_job

    async def test_get_job_missing_returns_none(self):
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)

        got = await JobManager.get_job(db, "no-such-id")
        assert got is None


class TestJobManagerCountActiveJobs:
    async def test_count_active_jobs(self):
        db = _mock_db()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        db.execute.return_value = mock_result

        count = await JobManager.count_active_jobs(db, TENANT_U4)
        assert count == 5

    async def test_count_active_jobs_none_returns_zero(self):
        db = _mock_db()
        mock_result = MagicMock()
        mock_result.scalar.return_value = None
        db.execute.return_value = mock_result

        count = await JobManager.count_active_jobs(db, TENANT_U4)
        assert count == 0


def _usage_params(db):
    """Compile the INSERT ``record_usage`` handed to the session; return its bound params.

    ``record_usage`` returns None and mutates no ORM object, so the statement is
    the only observable. These are BILLING numbers — the property worth pinning
    is which quantity lands in which column (and in which ON CONFLICT
    increment), not that ``execute`` happened to be awaited once.
    """
    db.execute.assert_awaited_once()
    return db.execute.await_args.args[0].compile(dialect=postgresql.dialect()).params


class TestJobManagerRecordUsage:
    """record_usage uses sqlalchemy.dialects.postgresql.insert → mock-only."""

    async def test_record_usage_basic_bills_the_supplied_quantities(self):
        db = AsyncMock()
        await JobManager.record_usage(
            db,
            tenant_id=TENANT_U1,
            compilations_completed=1,
            output_duration=42.0,
            output_size=5_000_000,
        )
        params = _usage_params(db)
        assert params["tenant_id"] == TENANT_U1
        assert params["date"] == datetime.now(UTC).strftime("%Y-%m-%d")
        assert params["compilations_started"] == 1
        assert params["compilations_completed"] == 1
        assert params["compilations_failed"] == 0
        assert params["total_output_duration_seconds"] == 42.0
        assert params["total_output_size_bytes"] == 5_000_000
        # The ON CONFLICT branch must bill the SAME quantities as the INSERT
        # branch — a mismatch bills a repeat compilation differently from a
        # first one on the same day.
        assert params["inc_completed"] == 1
        assert params["inc_duration"] == 42.0
        assert params["inc_size"] == 5_000_000

    async def test_record_usage_is_error_bills_a_failure_not_a_completion(self):
        db = AsyncMock()
        await JobManager.record_usage(db, tenant_id=TENANT_U1, is_error=True)
        params = _usage_params(db)
        assert params["compilations_failed"] == 1
        assert params["inc_failed"] == 1
        assert params["compilations_completed"] == 0
        # A failed compilation still consumed an attempt.
        assert params["compilations_started"] == 1
        assert params["total_output_duration_seconds"] == 0.0
        assert params["total_output_size_bytes"] == 0

    async def test_record_usage_nonfinite_output_duration_clamped(self):
        """``inf``/``nan`` must reach the DB as 0.0, not as a poisoned total.

        ffprobe returns ``N/A`` for some containers and the caller turns that
        into ``inf``. Postgres accepts ``inf`` in a double column, so an
        unclamped value permanently wedges that tenant's daily duration total at
        infinity — every later increment is a no-op.
        """
        db = AsyncMock()
        await JobManager.record_usage(
            db,
            tenant_id=TENANT_U1,
            compilations_completed=1,
            output_duration=float("inf"),
        )
        params = _usage_params(db)
        assert params["total_output_duration_seconds"] == 0.0
        assert params["inc_duration"] == 0.0

    async def test_record_usage_nan_output_duration_clamped(self):
        db = AsyncMock()
        await JobManager.record_usage(db, tenant_id=TENANT_U1, output_duration=float("nan"))
        params = _usage_params(db)
        assert params["total_output_duration_seconds"] == 0.0
        assert params["inc_duration"] == 0.0

    async def test_record_usage_defaults_bill_only_the_attempt(self):
        """The zero-argument call must count a started compilation and nothing else."""
        db = AsyncMock()
        await JobManager.record_usage(db, tenant_id=TENANT_U1)
        params = _usage_params(db)
        assert params["compilations_started"] == 1
        assert params["compilations_completed"] == 0
        assert params["compilations_failed"] == 0
        assert params["total_output_duration_seconds"] == 0.0
        assert params["total_output_size_bytes"] == 0

    async def test_record_usage_upserts_by_incrementing_not_overwriting(self):
        """A second compilation on the same (tenant, date) must ADD to the row."""
        db = AsyncMock()
        await JobManager.record_usage(db, tenant_id=TENANT_U1, compilations_completed=1)
        db.execute.assert_awaited_once()
        sql = str(db.execute.await_args.args[0].compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT ON CONSTRAINT uq_compilation_usage_tenant_date DO UPDATE" in sql
        assert "compilations_started = compilation_usage.compilations_started + 1" in sql
        assert "total_output_size_bytes = compilation_usage.total_output_size_bytes + " in sql
