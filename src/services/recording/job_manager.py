"""Job lifecycle management for recording jobs.

Provides CRUD operations, status transitions, batch management,
expiry handling, and daily usage aggregation for RecordingJob rows.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings
from src.database.models import RecordingJob, RecordingUsage

logger = logging.getLogger(__name__)


class JobStateConflictError(RuntimeError):
    """Raised when an atomic job-state transition loses a race.

    A caller should treat this as "another worker already transitioned
    this job" — log and bail cleanly, do not retry.
    """


class RecordingJobManager:
    """Instance-based job lifecycle manager wrapping an async DB session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def commit(self) -> None:
        """Commit the wrapped session."""
        await self._db.commit()

    async def rollback(self) -> None:
        """Roll back the wrapped session."""
        await self._db.rollback()

    async def create_job(
        self,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        part_type: str | None = None,
        scripts_dir: str | None = None,
        script_source: str | None = None,
        recording_config: dict | None = None,
        auth_config: dict | None = None,
        webhook_url: str | None = None,
        batch_id: str | None = None,
        priority: int = 0,
        batch_order: int | None = None,
        max_retries: int = 2,
    ) -> RecordingJob:
        """Create a new recording job and flush it to the session.

        Args:
            tenant_id: Owning tenant identifier.
            part_id: Part to record.
            part_type: Type of recording (default: browser_recording).
            scripts_dir: Path to Playwright scripts on disk.
            script_source: Optional inline content (e.g. HTML for content recording).
            recording_config: Per-request recording overrides.
            auth_config: Auth env vars for the subprocess.
            webhook_url: URL to POST on completion/failure.
            batch_id: Batch grouping identifier.
            priority: Scheduling priority (higher is more important).
            batch_order: Sequence index within a batch.
            max_retries: Maximum retry attempts.

        Returns:
            The newly created RecordingJob instance.
        """
        logger.debug("Creating job: tenant=%s part_id=%s batch_id=%s", tenant_id, part_id, batch_id)
        job = RecordingJob(
            job_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            part_id=part_id,
            part_type=part_type or "browser_recording",
            scripts_dir=scripts_dir,
            script_source=script_source,
            recording_config=recording_config,
            auth_config=auth_config,
            status="pending",
            webhook_url=webhook_url,
            batch_id=batch_id,
            priority=priority,
            batch_order=batch_order,
            max_retries=max_retries,
            retry_count=0,
            progress=0.0,
        )
        self._db.add(job)
        await self._db.flush()
        return job

    async def get_job(self, job_id: str, *, tenant_id: uuid.UUID) -> RecordingJob | None:
        """Fetch a single recording job owned by ``tenant_id``, or None.

        The tenant filter mirrors :meth:`list_jobs`. Postgres RLS already scopes
        this query, but the app layer must agree with the DB rather than lean on
        RLS as the only guard — callers must supply the tenant from the auth
        context, never from a caller-controlled parameter.
        """
        result = await self._db.execute(
            select(RecordingJob).where(
                RecordingJob.job_id == job_id,
                RecordingJob.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        *,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        status: str | None = None,
        sort_order: str = "desc",
    ) -> tuple[list[RecordingJob], int]:
        """List jobs with pagination, returns (jobs, total_count)."""
        query = select(RecordingJob).where(RecordingJob.tenant_id == tenant_id)
        count_query = (
            select(func.count())
            .select_from(RecordingJob)
            .where(RecordingJob.tenant_id == tenant_id)
        )

        if status:
            query = query.where(RecordingJob.status == status)
            count_query = count_query.where(RecordingJob.status == status)

        if sort_order == "asc":
            query = query.order_by(RecordingJob.created_at.asc())
        else:
            query = query.order_by(RecordingJob.created_at.desc())

        query = query.offset(offset).limit(limit)

        result = await self._db.execute(query)
        jobs = list(result.scalars().all())

        count_result = await self._db.execute(count_query)
        total = count_result.scalar() or 0

        return jobs, total

    async def mark_processing(self, job_id: str) -> None:
        """Atomically transition a pending/queued job to 'processing'.

        Raises :class:`JobStateConflictError` if the job is not currently
        pending or queued (e.g., already running, completed, or cancelled).
        """
        logger.debug("Job %s -> processing", job_id)
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(
                    RecordingJob.job_id == job_id,
                    RecordingJob.status.in_(["pending", "queued"]),
                )
                .values(status="processing", updated_at=datetime.now(UTC))
            ),
        )
        await self._db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is not pending/queued — cannot mark_processing"
            )

    async def mark_queued(self, job_id: str) -> None:
        """Atomically transition a pending job to 'queued' (awaiting concurrency slot).

        Raises :class:`JobStateConflictError` if the job is not currently pending.
        """
        logger.debug("Job %s -> queued", job_id)
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(RecordingJob.job_id == job_id, RecordingJob.status == "pending")
                .values(status="queued", updated_at=datetime.now(UTC))
            ),
        )
        await self._db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is not pending — cannot mark_queued"
            )

    async def mark_completed(
        self,
        job_id: str,
        *,
        video_path: str | None = None,
        duration_seconds: float | None = None,
        file_size_bytes: int | None = None,
        width: int | None = None,
        height: int | None = None,
        capture_width: int | None = None,
        capture_height: int | None = None,
        max_frame_seconds: float | None = None,
        has_audio: bool = False,
        recording_time_seconds: float | None = None,
        queue_wait_seconds: float | None = None,
        artifact_kind: str | None = None,
        retention_hours: int = 168,
    ) -> None:
        """Atomically transition a processing job to 'completed' with result metadata.

        Raises :class:`JobStateConflictError` if the job is not currently processing.
        """
        logger.debug("Job %s -> completed (video=%s)", job_id, video_path)
        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "status": "completed",
            "video_path": video_path,
            "duration_seconds": duration_seconds,
            "file_size_bytes": file_size_bytes,
            "width": width,
            "height": height,
            # The raster the recorder actually captured at, BEFORE the downscale
            # to width x height. NULL from any recorder that does not supersample.
            "capture_width": capture_width,
            "capture_height": capture_height,
            # Slowest single frame's own capture time in AnimationRecorder's
            # frame-step loop. NULL from any recorder that does not frame-step.
            "max_frame_seconds": max_frame_seconds,
            "has_audio": has_audio,
            "recording_time_seconds": recording_time_seconds,
            "queue_wait_seconds": queue_wait_seconds,
            # What the stored artifact IS, as declared by the producer. Persisted
            # so `GET /jobs/{id}/video` can dispatch on the same fact
            # `_verify_artifact` did, instead of guessing from the file suffix.
            "artifact_kind": artifact_kind,
            "progress": 1.0,
            "progress_stage": "completed",
            "updated_at": now,
            "completed_at": now,
            "expires_at": now + timedelta(hours=retention_hours),
        }
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(RecordingJob.job_id == job_id, RecordingJob.status == "processing")
                .values(**values)
            ),
        )
        await self._db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is not processing — cannot mark_completed"
            )

    async def mark_failed(self, job_id: str, error: str, *, retention_hours: int = 168) -> None:
        """Atomically transition a non-terminal job to 'failed' with error details.

        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        settings = get_settings()
        max_err = settings.job.max_error_length
        log_trunc = settings.job.log_error_truncate_length
        logger.debug("Job %s -> failed: %s", job_id, error[:log_trunc])
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(
                    RecordingJob.job_id == job_id,
                    RecordingJob.status.in_(["pending", "queued", "processing"]),
                )
                .values(
                    status="failed",
                    error_message=error[:max_err],
                    last_error=error[:max_err],
                    updated_at=now,
                    completed_at=now,
                    expires_at=now + timedelta(hours=retention_hours),
                )
            ),
        )
        await self._db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is already terminal — cannot mark_failed"
            )

    async def mark_cancelled(self, job_id: str) -> None:
        """Atomically transition a non-terminal job to 'cancelled'.

        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        logger.debug("Job %s -> cancelled", job_id)
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(
                    RecordingJob.job_id == job_id,
                    RecordingJob.status.in_(["pending", "queued", "processing"]),
                )
                .values(status="cancelled", updated_at=now, completed_at=now)
            ),
        )
        await self._db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is already terminal — cannot mark_cancelled"
            )

    async def update_progress(self, job_id: str, progress: float, stage: str) -> None:
        """Update the progress percentage and stage label for an in-flight job.

        Silent no-op if the job is not processing/queued — progress bumps are
        advisory and shouldn't interfere with terminal transitions.
        """
        await self._db.execute(
            update(RecordingJob)
            .where(
                RecordingJob.job_id == job_id,
                RecordingJob.status.in_(["pending", "queued", "processing"]),
            )
            .values(progress=progress, progress_stage=stage, updated_at=datetime.now(UTC))
        )
        await self._db.commit()

    async def update_retry_count(self, job_id: str, retry_count: int) -> None:
        """Update the retry counter for a job (atomic, any non-terminal state)."""
        await self._db.execute(
            update(RecordingJob)
            .where(
                RecordingJob.job_id == job_id,
                RecordingJob.status.in_(["pending", "queued", "processing"]),
            )
            .values(retry_count=retry_count, updated_at=datetime.now(UTC))
        )
        await self._db.commit()

    async def store_last_error(self, job_id: str, error: str) -> None:
        """Persist the most recent error message without changing job status."""
        max_err = get_settings().job.max_error_length
        await self._db.execute(
            update(RecordingJob)
            .where(RecordingJob.job_id == job_id)
            .values(last_error=error[:max_err], updated_at=datetime.now(UTC))
        )
        await self._db.commit()

    async def get_batch_jobs(self, batch_id: str, tenant_id: uuid.UUID) -> list[RecordingJob]:
        """Fetch all jobs belonging to a batch, ordered by batch_order."""
        result = await self._db.execute(
            select(RecordingJob)
            .where(RecordingJob.batch_id == batch_id, RecordingJob.tenant_id == tenant_id)
            .order_by(RecordingJob.batch_order.asc())
        )
        return list(result.scalars().all())

    async def cancel_batch_remaining(self, batch_id: str) -> int:
        """Atomically cancel all pending/queued jobs in a batch.

        Returns the number of jobs actually cancelled. Uses a single bulk
        UPDATE so the transition is race-free with concurrent workers.
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await self._db.execute(
                update(RecordingJob)
                .where(
                    RecordingJob.batch_id == batch_id,
                    RecordingJob.status.in_(["pending", "queued"]),
                )
                .values(status="cancelled", updated_at=now, completed_at=now)
            ),
        )
        await self._db.commit()
        return int(result.rowcount or 0)

    async def get_expired_jobs(self, limit: int | None = None) -> list[RecordingJob]:
        """Find completed/failed jobs past their expiry time."""
        if limit is None:
            limit = get_settings().job.expired_batch_limit
        now = datetime.now(UTC)
        result = await self._db.execute(
            select(RecordingJob)
            .where(
                RecordingJob.expires_at <= now,
                RecordingJob.expired.is_(False),
                RecordingJob.status.in_(["completed", "failed", "cancelled"]),
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_expired(self, job_id: str) -> None:
        """Flag a job as expired after its files have been cleaned up.

        Advisory update — no state guard; the cleanup service only calls this
        for terminal jobs anyway.
        """
        logger.debug("Job %s -> expired", job_id)
        await self._db.execute(
            update(RecordingJob)
            .where(RecordingJob.job_id == job_id)
            .values(expired=True, updated_at=datetime.now(UTC))
        )
        await self._db.commit()

    async def get_stale_jobs(
        self, max_age_seconds: int = 600, limit: int = 100
    ) -> list[RecordingJob]:
        """Find jobs stuck in processing/queued status longer than max_age_seconds."""
        cutoff = datetime.now(UTC) - timedelta(seconds=max_age_seconds)
        result = await self._db.execute(
            select(RecordingJob)
            .where(
                RecordingJob.status.in_(["processing", "queued"]),
                RecordingJob.updated_at <= cutoff,
            )
            .limit(limit)
        )
        return list(result.scalars().all())

    async def record_usage(
        self,
        *,
        tenant_id: uuid.UUID,
        recordings_completed: int = 0,
        is_error: bool = False,
        duration_seconds: float = 0.0,
        file_size_bytes: int = 0,
    ) -> None:
        """Upsert a daily usage row for the given tenant (insert or increment)."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        insert_stmt = pg_insert(RecordingUsage).values(
            tenant_id=tenant_id,
            date=today,
            recordings_started=1,
            recordings_completed=recordings_completed,
            recordings_failed=1 if is_error else 0,
            total_duration_seconds=duration_seconds,
            total_file_size_bytes=file_size_bytes,
        )
        stmt = insert_stmt.on_conflict_do_update(
            constraint="uq_recording_usage_tenant_date",
            set_={
                "recordings_started": (RecordingUsage.recordings_started + 1),
                "recordings_completed": (
                    RecordingUsage.recordings_completed + insert_stmt.excluded.recordings_completed
                ),
                "recordings_failed": (
                    RecordingUsage.recordings_failed + insert_stmt.excluded.recordings_failed
                ),
                "total_duration_seconds": (
                    RecordingUsage.total_duration_seconds
                    + insert_stmt.excluded.total_duration_seconds
                ),
                "total_file_size_bytes": (
                    RecordingUsage.total_file_size_bytes
                    + insert_stmt.excluded.total_file_size_bytes
                ),
            },
        )
        await self._db.execute(stmt)
