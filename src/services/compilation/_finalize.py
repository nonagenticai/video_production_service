"""Finalize stage helpers (mixin)."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.exceptions import ArtifactNotStoredError
from src.services.compilation._compiler_context import _JobContext

if TYPE_CHECKING:
    from src.services.ffmpeg import FFmpegOps as _FFmpegOpsType
    from src.services.storage_service import StorageService


def _ffmpeg() -> type[_FFmpegOpsType]:
    """Resolve FFmpegOps from the compiler_service module (late-bound)."""
    from src.services.compilation import compiler_service

    return compiler_service.FFmpegOps


logger = logging.getLogger(__name__)


class _FinalizeMixin:
    """Mixin providing finalize-stage logic: chapters, thumbnails, validation, persist."""

    _storage: StorageService

    if TYPE_CHECKING:
        # Provided by sibling mixins on the composed CompilerService class.
        def _compute_chapter_timestamps(
            self,
            segments: list[Path],
            parts_by_id: dict,
            transition_type: str = "cut",
            crossfade_dur: float = 0.0,
        ) -> list[dict]: ...

    @staticmethod
    def _embed_chapter_markers(ctx: _JobContext, chapter_timestamps: list[dict]) -> None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        if not (chapter_timestamps and output_path.exists()):
            return
        try:
            chaptered_path = work_dir / "demo_chaptered.mp4"
            _ffmpeg().embed_chapters(output_path, chaptered_path, chapter_timestamps)
            if chaptered_path.exists():
                shutil.move(str(chaptered_path), str(output_path))
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            logger.warning("Chapter marker embedding failed for job %s", ctx.job_id, exc_info=True)
            ctx.record_degradation("chapter_markers", "Chapter marker embedding failed", exc)

    @staticmethod
    def _generate_thumbnail(ctx: _JobContext) -> str | None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        if not output_path.exists():
            return None
        try:
            thumb_path = work_dir / "thumbnail.jpg"
            _ffmpeg().extract_thumbnail(
                output_path, thumb_path, timestamp=1.0, width=640, height=360
            )
            if thumb_path.exists():
                return str(thumb_path)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            logger.warning("Auto-thumbnail generation failed for job %s", ctx.job_id, exc_info=True)
            ctx.record_degradation("thumbnail", "Thumbnail generation failed", exc)
        return None

    @staticmethod
    def _validate_quality(ctx: _JobContext) -> dict | None:
        output_path = ctx.output_path
        if not output_path.exists():
            return None
        try:
            from src.services.compilation.quality_validator import QualityValidator

            validator = QualityValidator(target_width=ctx.target_w, target_height=ctx.target_h)
            qr = validator.validate(output_path)
            return {
                "valid": qr.valid,
                "issues": qr.issues,
                "file_size_bytes": qr.file_size_bytes,
                "duration_seconds": qr.duration_seconds,
                "resolution": qr.resolution,
                "audio_peak_db": qr.audio_peak_db,
                "encoding_valid": qr.encoding_valid,
            }
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            logger.warning("Quality validation failed for job %s", ctx.job_id, exc_info=True)
            ctx.record_degradation("quality_validation", "Quality validation failed", exc)
            return None

    def _finalize_sync(
        self, ctx: _JobContext
    ) -> tuple[float, int, list[dict], str | None, dict | None]:
        output_path = ctx.output_path
        total_duration = _ffmpeg().probe_duration(output_path) if output_path.exists() else 0.0
        file_size = output_path.stat().st_size if output_path.exists() else 0
        chapter_timestamps = self._compute_chapter_timestamps(
            ctx.segments,
            ctx.parts_by_id,
            transition_type=ctx.transition_type,
            crossfade_dur=ctx.crossfade_dur,
        )
        self._embed_chapter_markers(ctx, chapter_timestamps)
        thumbnail_path = self._generate_thumbnail(ctx)
        quality_report_dict = self._validate_quality(ctx)
        return total_duration, file_size, chapter_timestamps, thumbnail_path, quality_report_dict

    async def _upload_final_artifacts(
        self, ctx: _JobContext, thumbnail_path: str | None
    ) -> tuple[str, str | None, str | None]:
        """Persist the compiled video (and thumbnail) to durable storage.

        Raises:
            ArtifactNotStoredError: If the compiled video is missing on disk, or
                could not be uploaded. There is deliberately **no** fallback to
                the pod-local ``work_dir`` path here: no consumer can read it.
                ``GET /api/compile/jobs/{id}/video`` goes through
                ``StorageService.retrieve``, where the minio backend treats the
                value as an object key (miss) and the local backend rejects it as
                outside ``storage.local_path``. Both 404, so returning it would
                only buy a ``completed`` job and a billed
                ``compilations_completed`` unit for a video nobody can fetch.
        """
        output_path = ctx.output_path
        job_id = ctx.job_id
        settings = ctx.settings
        stored_thumbnail_path = thumbnail_path
        presigned_url: str | None = None
        if not output_path.exists():
            raise ArtifactNotStoredError(
                f"Compiled video for job {job_id} is missing at {output_path}; "
                "nothing was uploaded to storage"
            )
        try:
            stored_output_path = await self._storage.store_file(
                job_id, output_path, settings.compilation.output_filename
            )
        except Exception as exc:
            # Broad on purpose. The intent here is "the upload failed", and a
            # curated tuple only encodes the failures whoever wrote it happened
            # to imagine -- a backend-specific error (e.g. minio's ``S3Error``,
            # which is none of OSError/RuntimeError/ValueError/TimeoutError)
            # would sail past it, past ``process_job``'s identically-shaped
            # handler, and strand the job in ``processing``. Translating every
            # non-``BaseException`` failure into ``ArtifactNotStoredError``
            # guarantees the job reaches ``mark_failed`` +
            # ``record_usage(is_error=True)``. ``asyncio.CancelledError`` is a
            # ``BaseException`` and still propagates untouched.
            raise ArtifactNotStoredError(
                f"Failed to upload compiled video to storage for job {job_id}: {exc}",
                original_error=exc,
            ) from exc
        logger.info("Stored compiled video for job %s: %s", job_id, stored_output_path)
        try:
            presigned_url = await self._storage.presign_url(stored_output_path)
        except Exception as exc:
            # Degraded but honest: the video IS in storage and the /video route
            # can serve it. Only the convenience URL in the compilation report
            # is missing, so this must not fail an otherwise-deliverable job.
            logger.warning("Failed to presign compiled video for job %s", job_id, exc_info=True)
            ctx.record_degradation("presigned_url", "Could not presign the compiled video URL", exc)
        if thumbnail_path:
            try:
                stored_thumbnail_path = await self._storage.store_file(
                    job_id, Path(thumbnail_path), "thumbnail.jpg"
                )
            except Exception as exc:  # noqa: BLE001 -- thumbnail is not the billed deliverable
                # The thumbnail is not the billed deliverable, so a failure here
                # degrades rather than fails. Broadened from the old curated
                # tuple for the same reason as above: an unlisted backend error
                # would otherwise escape and hard-fail a perfectly good video.
                logger.warning("Failed to upload thumbnail to storage for job %s", job_id)
                ctx.record_degradation(
                    "thumbnail_upload", "Thumbnail upload to storage failed", exc
                )
        return stored_output_path, stored_thumbnail_path, presigned_url

    async def _finalize(
        self, ctx: _JobContext, db: AsyncSession, progress_callback: Any | None
    ) -> None:
        """Compute results, embed chapters, generate thumbnail, validate, persist."""
        from src.services.compilation.job_manager import (
            CompilationJobManager,
            JobStateConflictError,
        )

        job_id = ctx.job_id
        work_dir = ctx.work_dir
        settings = ctx.settings

        (
            total_duration,
            file_size,
            chapter_timestamps,
            thumbnail_path,
            quality_report_dict,
        ) = await asyncio.to_thread(self._finalize_sync, ctx)
        elapsed = time.time() - ctx.start_time

        (
            stored_output_path,
            stored_thumbnail_path,
            presigned_url,
        ) = await self._upload_final_artifacts(ctx, thumbnail_path)

        # Build compilation report
        audio_details = ctx.audio_mixing_details
        compilation_report: dict[str, Any] = {
            "job_id": job_id,
            "compilation_time_seconds": round(elapsed, 3),
            "stages": ctx.stage_timings,
            "segments_processed": len(ctx.segments),
            "segments_excluded": ctx.segments_excluded,
            "transitions_used": ctx.transitions_used,
            "audio_mixing_details": audio_details,
            "input_duration_seconds": round(ctx.input_duration, 3),
            "output_duration_seconds": round(total_duration, 3),
        }
        if presigned_url:
            compilation_report["presigned_url"] = presigned_url
        # Derive the report status from ctx.degradations rather than from
        # subtitles alone, which is all the original code keyed on.
        #
        # ⚠️ COVERAGE IS PARTIAL, AND "success" IS THEREFORE NOT A GUARANTEE.
        # It means "no COVERED site recorded a drop". Covered today:
        #
        #   narration download, narration loudnorm, background music,
        #   watermark (guard + failure), colour correction, background image
        #   (download + guard), logo guard, motion title card, AI-audio strip,
        #   per-segment narration, chapter markers, thumbnail (generation +
        #   upload), quality validation, presign, crossfade fallback, subtitles.
        #
        # KNOWN NOT COVERED -- these still ship as "success":
        #   * title_card_generator.py swallows three drops of its own
        #     (_build_background -> silent gradient fallback; generate_image ->
        #     logo dropped; generate_video -> static instead of animated). So a
        #     background image that DOWNLOADS but fails to render is reported
        #     clean, even though the guard-refusal path for the same feature is
        #     reported. That asymmetry is real and unfixed.
        #   * any drop shaped as an early return, a logger.info, or a
        #     contextlib.suppress -- see tests/unit/test_degradation_coverage.py,
        #     which measures only one of those shapes.
        #
        # Do not upgrade this comment to a completeness claim without a guard
        # that actually enforces it; the guard we have does not.
        degradations = list(ctx.degradations)
        if ctx.subtitle_error:
            compilation_report["subtitle_error"] = ctx.subtitle_error
            degradations.append(
                {"feature": "subtitles", "reason": "Subtitles could not be applied"}
            )
        if degradations:
            compilation_report["status"] = "partial_success"
            compilation_report["degradations"] = degradations
        else:
            compilation_report["status"] = "success"

        if progress_callback:
            await progress_callback.complete()

        try:
            await CompilationJobManager.mark_completed(
                db,
                job_id,
                output_path=stored_output_path,
                total_duration_seconds=total_duration,
                total_parts=len(ctx.segments),
                file_size_bytes=file_size,
                chapter_timestamps=chapter_timestamps,
                compilation_time_seconds=elapsed,
                thumbnail_path=stored_thumbnail_path,
                compilation_report=compilation_report,
                quality_metrics=quality_report_dict,
            )
        except JobStateConflictError:
            # Somebody else already wrote a terminal state for this job -- a
            # cancel, or the stale reaper failing it. `output_path` did NOT
            # persist, so there is no artifact to announce. Leaving `ctx.completed`
            # False is what stops `process_job` firing a "completed" webhook for
            # work whose output was thrown away.
            logger.warning(
                "Compilation job %s was no longer 'processing' when its result arrived "
                "(cancelled, or reaped as stale): the output path was NOT persisted and "
                "no completion will be announced",
                job_id,
            )
            return
        ctx.completed = True

        await CompilationJobManager.record_usage(
            db,
            tenant_id=ctx.tenant_id,
            compilations_completed=1,
            output_duration=total_duration,
            output_size=file_size,
        )

        logger.info(
            "Job %s completed: duration=%.1fs, parts=%d, size=%d bytes, elapsed=%.1fs",
            job_id,
            total_duration,
            len(ctx.segments),
            file_size,
            elapsed,
        )

        # Cleanup intermediates
        if settings.compilation.cleanup_intermediates:
            for subdir in ("edited", "title_cards", "normalized"):
                d = work_dir / subdir
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
