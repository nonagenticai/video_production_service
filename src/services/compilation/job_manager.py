"""Job lifecycle management for compilation jobs."""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.compilation import DailyUsageItem
from src.database.models import CompilationJob, CompilationUsage


class JobStateConflictError(RuntimeError):
    """Raised when an atomic job-state transition loses a race.

    A caller should treat this as "another worker already transitioned
    this job" — log and bail cleanly, do not retry.
    """


class CompilationJobManager:
    """Static methods for compilation job CRUD and usage tracking."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        shot_list: dict | None = None,
        recordings: list[dict] | None = None,
        narration_audio_dir: str | None = None,
        clips_dir: str | None = None,
        transition_type: str | None = None,
        crossfade_duration: float | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        crf: int | None = None,
        theme_config: dict | None = None,
        edit_decisions: list[dict] | None = None,
        preset: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        filter_config: dict | None = None,
        watermark_config: dict | None = None,
        subtitles: dict | None = None,
    ) -> CompilationJob:
        """Insert a new compilation job into the database and return it."""
        job = CompilationJob(
            job_id=str(uuid.uuid7()),
            tenant_id=tenant_id,
            shot_list=shot_list,
            recordings=recordings,
            narration_audio_dir=narration_audio_dir,
            clips_dir=clips_dir,
            preset=preset,
            transition_type=transition_type,
            crossfade_duration=crossfade_duration,
            target_width=target_width,
            target_height=target_height,
            crf=crf,
            theme_config=theme_config,
            edit_decisions=edit_decisions,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            filter_config=filter_config,
            watermark_config=watermark_config,
            subtitles=subtitles,
            status="pending",
        )
        db.add(job)
        await db.flush()
        return job

    @staticmethod
    async def get_job(db: AsyncSession, job_id: str) -> CompilationJob | None:
        """Retrieve a single compilation job by its job_id."""
        result = await db.execute(select(CompilationJob).where(CompilationJob.job_id == job_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a pending job to 'processing'.

        Raises :class:`JobStateConflictError` if the job is not currently pending.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(CompilationJob)
                .where(CompilationJob.job_id == job_id, CompilationJob.status == "pending")
                .values(status="processing", updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Compilation job {job_id!r} is not pending — cannot mark_processing"
            )

    @staticmethod
    async def record_progress(
        db: AsyncSession,
        job_id: str,
        *,
        progress: float | None = None,
        stage: str | None = None,
    ) -> bool:
        """Bump ``updated_at`` (and optionally progress/stage) for a RUNNING job.

        The compilation sibling of ``recording.job_manager.update_progress``, and
        it was missing entirely -- which is what made the stale reaper measure the
        wrong thing. ``storage.stale_job_timeout_seconds`` documents itself as
        "seconds without PROGRESS", and ``stale_compilation_jobs`` correctly asks
        ``updated_at <= now() - interval``; but on this domain nothing ever wrote
        ``updated_at`` between ``mark_processing`` and a terminal transition, so
        the predicate degenerated to "started more than 600s ago". Every compile
        past the p50 of ~7.6 minutes was reaped as ``failed`` while still running.

        Scoped to ``status = 'processing'`` so it can never resurrect the clock on
        a job that has already been cancelled or reaped -- a heartbeat that could
        write to a terminal row would let a doomed job argue itself back to life.

        Returns:
            True if the row was still processing and was bumped; False if it was
            not (cancelled, already reaped, or gone). The caller uses the False to
            STOP heartbeating rather than to retry.
        """
        values: dict[str, datetime | float | str] = {"updated_at": datetime.now(UTC)}
        if progress is not None:
            values["progress"] = progress
        if stage is not None:
            values["current_stage"] = stage
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(CompilationJob)
                .where(
                    CompilationJob.job_id == job_id,
                    CompilationJob.status == "processing",
                )
                .values(**values)
            ),
        )
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        output_path: str | None = None,
        total_duration_seconds: float | None = None,
        total_parts: int | None = None,
        file_size_bytes: int | None = None,
        chapter_timestamps: list[dict] | None = None,
        compilation_time_seconds: float | None = None,
        thumbnail_path: str | None = None,
        compilation_report: dict | None = None,
        quality_metrics: dict | None = None,
    ) -> None:
        """Atomically transition a processing job to 'completed' with output metadata.

        Raises :class:`JobStateConflictError` if the job is not currently processing.
        """
        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "status": "completed",
            "output_path": output_path,
            "total_duration_seconds": total_duration_seconds,
            "total_parts": total_parts,
            "file_size_bytes": file_size_bytes,
            "chapter_timestamps": chapter_timestamps,
            "compilation_time_seconds": compilation_time_seconds,
            "thumbnail_path": thumbnail_path,
            "updated_at": now,
            "completed_at": now,
        }
        if compilation_report is not None:
            values["compilation_report"] = compilation_report
        if quality_metrics is not None:
            values["quality_metrics"] = quality_metrics

        result = cast(
            CursorResult[Any],
            await db.execute(
                update(CompilationJob)
                .where(
                    CompilationJob.job_id == job_id,
                    CompilationJob.status == "processing",
                )
                .values(**values)
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Compilation job {job_id!r} is not processing — cannot mark_completed"
            )

    @staticmethod
    async def mark_failed(db: AsyncSession, job_id: str, error: str) -> None:
        """Atomically transition a non-terminal job to 'failed'.

        Raises :class:`JobStateConflictError` if the job is already terminal.
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(CompilationJob)
                .where(
                    CompilationJob.job_id == job_id,
                    CompilationJob.status.in_(["pending", "processing"]),
                )
                .values(
                    status="failed",
                    error_message=error[:4000],
                    updated_at=now,
                    completed_at=now,
                )
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"Compilation job {job_id!r} is already terminal — cannot mark_failed"
            )

    @staticmethod
    async def record_usage(
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        compilations_completed: int = 0,
        is_error: bool = False,
        output_duration: float = 0.0,
        output_size: int = 0,
    ) -> None:
        """Upsert daily usage counters for a tenant using ON CONFLICT."""
        # Sanitise float values: clamp non-finite values to 0
        if not math.isfinite(output_duration):
            output_duration = 0.0

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        failed_inc = 1 if is_error else 0
        stmt = (
            pg_insert(CompilationUsage)
            .values(
                tenant_id=tenant_id,
                date=today,
                compilations_started=1,
                compilations_completed=compilations_completed,
                compilations_failed=failed_inc,
                total_output_duration_seconds=output_duration,
                total_output_size_bytes=output_size,
            )
            .on_conflict_do_update(
                constraint="uq_compilation_usage_tenant_date",
                set_={
                    "compilations_started": text("compilation_usage.compilations_started + 1"),
                    "compilations_completed": text(
                        "compilation_usage.compilations_completed + :inc_completed"
                    ).bindparams(inc_completed=compilations_completed),
                    "compilations_failed": text(
                        "compilation_usage.compilations_failed + :inc_failed"
                    ).bindparams(inc_failed=failed_inc),
                    "total_output_duration_seconds": text(
                        "compilation_usage.total_output_duration_seconds + :inc_duration"
                    ).bindparams(inc_duration=output_duration),
                    "total_output_size_bytes": text(
                        "compilation_usage.total_output_size_bytes + :inc_size"
                    ).bindparams(inc_size=output_size),
                },
            )
        )
        await db.execute(stmt)

    @staticmethod
    async def find_expired_jobs(
        db: AsyncSession,
        status: str,
        before: datetime,
        limit: int = 100,
    ) -> list[CompilationJob]:
        """Find jobs with given status completed/updated before cutoff."""
        result = await db.execute(
            select(CompilationJob)
            .where(CompilationJob.status == status)
            .where(CompilationJob.updated_at < before)
            .order_by(CompilationJob.updated_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_jobs(db: AsyncSession, job_ids: list[str]) -> int:
        """Delete jobs by their job_ids. Returns count deleted."""
        if not job_ids:
            return 0
        result = await db.execute(delete(CompilationJob).where(CompilationJob.job_id.in_(job_ids)))
        cursor_result = cast(CursorResult[Any], result)
        return cursor_result.rowcount or 0

    @staticmethod
    async def list_jobs(
        db: AsyncSession,
        tenant_id: uuid.UUID,
        *,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[CompilationJob], int]:
        """List jobs for a tenant with optional filtering and pagination."""
        from sqlalchemy import func

        query = select(CompilationJob).where(CompilationJob.tenant_id == tenant_id)
        count_query = (
            select(func.count())
            .select_from(CompilationJob)
            .where(CompilationJob.tenant_id == tenant_id)
        )
        if status:
            query = query.where(CompilationJob.status == status)
            count_query = count_query.where(CompilationJob.status == status)
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        # Paginate
        offset = (page - 1) * page_size
        query = query.order_by(CompilationJob.created_at.desc()).offset(offset).limit(page_size)
        result = await db.execute(query)
        jobs = list(result.scalars().all())
        return jobs, total

    @staticmethod
    async def mark_cancelled(db: AsyncSession, job_id: str) -> bool:
        """Atomically transition a pending/processing job to 'cancelled'.

        Returns True if the transition succeeded, False if the job was
        already terminal (completed, failed, or cancelled).
        """
        now = datetime.now(UTC)
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(CompilationJob)
                .where(
                    CompilationJob.job_id == job_id,
                    CompilationJob.status.in_(["pending", "processing"]),
                )
                .values(status="cancelled", updated_at=now)
            ),
        )
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def find_jobs_by_criteria(
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID | None = None,
        job_ids: list[str] | None = None,
        status_filter: list[str] | None = None,
        min_age_hours: float | None = None,
        limit: int = 100,
    ) -> list[CompilationJob]:
        """Find jobs matching criteria (WP-6: on-demand cleanup)."""
        from datetime import timedelta

        query = select(CompilationJob)
        if tenant_id:
            query = query.where(CompilationJob.tenant_id == tenant_id)
        if job_ids:
            query = query.where(CompilationJob.job_id.in_(job_ids))
        if status_filter:
            query = query.where(CompilationJob.status.in_(status_filter))
        if min_age_hours is not None:
            cutoff = datetime.now(UTC) - timedelta(hours=min_age_hours)
            query = query.where(CompilationJob.created_at < cutoff)
        query = query.order_by(CompilationJob.created_at.asc()).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def count_active_jobs(
        db: AsyncSession,
        tenant_id: uuid.UUID,
    ) -> int:
        """Count active (pending/processing) jobs for a tenant (WP-6)."""
        from sqlalchemy import func

        result = await db.execute(
            select(func.count())
            .select_from(CompilationJob)
            .where(
                CompilationJob.tenant_id == tenant_id,
                CompilationJob.status.in_(["pending", "processing"]),
            )
        )
        return result.scalar() or 0

    @staticmethod
    async def get_usage_summary(
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict:
        """Get aggregated usage stats for a tenant."""
        from sqlalchemy import func

        query = select(
            func.coalesce(func.sum(CompilationUsage.compilations_started), 0).label(
                "compilations_started"
            ),
            func.coalesce(func.sum(CompilationUsage.compilations_completed), 0).label(
                "compilations_completed"
            ),
            func.coalesce(func.sum(CompilationUsage.compilations_failed), 0).label(
                "compilations_failed"
            ),
            func.coalesce(
                func.sum(CompilationUsage.total_output_duration_seconds),
                0.0,
            ).label("total_output_duration_seconds"),
            func.coalesce(func.sum(CompilationUsage.total_output_size_bytes), 0).label(
                "total_output_size_bytes"
            ),
        ).where(CompilationUsage.tenant_id == tenant_id)
        if date_from:
            query = query.where(CompilationUsage.date >= date_from)
        if date_to:
            query = query.where(CompilationUsage.date <= date_to)
        result = await db.execute(query)
        row = result.one()
        return {
            "tenant_id": tenant_id,
            "compilations_started": row.compilations_started,
            "compilations_completed": row.compilations_completed,
            "compilations_failed": row.compilations_failed,
            "total_output_duration_seconds": float(row.total_output_duration_seconds),
            "total_output_size_bytes": row.total_output_size_bytes,
        }

    @staticmethod
    async def get_usage_daily(
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[DailyUsageItem]:
        """Get daily usage breakdown for a tenant."""
        query = select(CompilationUsage).where(CompilationUsage.tenant_id == tenant_id)
        if date_from:
            query = query.where(CompilationUsage.date >= date_from)
        if date_to:
            query = query.where(CompilationUsage.date <= date_to)
        query = query.order_by(CompilationUsage.date.asc())
        result = await db.execute(query)
        rows = result.scalars().all()
        return [
            DailyUsageItem(
                date=row.date,
                compilations_started=row.compilations_started or 0,
                compilations_completed=row.compilations_completed or 0,
                compilations_failed=row.compilations_failed or 0,
                total_output_duration_seconds=float(row.total_output_duration_seconds or 0.0),
                total_output_size_bytes=row.total_output_size_bytes or 0,
            )
            for row in rows
        ]
