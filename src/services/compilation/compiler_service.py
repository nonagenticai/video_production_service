"""Core video compilation orchestration service.

Split into mixin modules (P3.2):

- :mod:`src.services.compilation._compiler_context` -- ``_JobContext`` dataclass
  + module-level webhook task tracker.
- :mod:`src.services.compilation._segment_builders` -- per-segment build,
  edit, normalize, narration mixing.
- :mod:`src.services.compilation._assembly` -- video assembly + chapter
  timestamp computation.
- :mod:`src.services.compilation._effects` -- color correction, watermark,
  audio mixing, subtitles.
- :mod:`src.services.compilation._finalize` -- chapter embedding, thumbnails,
  quality validation, persistence.

Public symbols (``CompilerService``, ``_JobContext``) are re-exported from this
module so existing imports continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.database.session import get_db_session

from src.services.compilation._assembly import _AssemblyMixin
from src.services.compilation._compiler_context import _JobContext, _webhook_tasks
from src.services.compilation._effects import _EffectsMixin
from src.services.compilation._finalize import _FinalizeMixin
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from src.services.compilation.job_manager import CompilationJobManager
from src.services.compilation.title_card_generator import TitleCardGenerator
from src.services.compilation.video_editor import VideoEditor
from src.services.ffmpeg import FFmpegOps
from src.services.storage_service import StorageService
from src.utils.error_formatting import format_error

logger = logging.getLogger(__name__)


async def _download_narration_audio(ctx: _JobContext, url: str) -> None:
    """Fetch the tenant-supplied narration track into the job work dir.

    Sets ``ctx.narration_audio_dir`` to the local directory on success and to
    ``None`` on any failure — the compile then proceeds without narration rather
    than aborting. That fail-soft contract is deliberate, and it is also what
    makes the size ceiling load-bearing: ``ResponseTooLargeError`` subclasses
    ``ValueError``, so an over-cap track lands in the handler below and produces
    a SILENT video with the job still reported successful. Hence the explicit
    :data:`~src.utils.url_guard.ARTIFACT_AUDIO_MAX_BYTES` rather than
    w2v_common's 64 MiB default, which is only ~70 minutes of 128 kbps mp3.

    ``safe_get`` also validates the target and every redirect hop: ``url`` is
    tenant-controlled, so this is an SSRF sink as well as a size sink.
    """
    from w2v_common.net import safe_get

    from src.utils.url_guard import ARTIFACT_AUDIO_MAX_BYTES, artifact_allow_hosts

    narr_dir = ctx.work_dir / "narration_audio"
    narr_dir.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resp = await safe_get(
                http_client,
                url,
                allow_hosts=artifact_allow_hosts(),
                max_fetch_bytes=ARTIFACT_AUDIO_MAX_BYTES,
            )
            resp.raise_for_status()
            narr_file = narr_dir / "narration_full.mp3"
            narr_file.write_bytes(resp.content)
            logger.info("Downloaded narration audio (%d bytes) to %s", len(resp.content), narr_file)
        ctx.narration_audio_dir = narr_dir
    except (httpx.HTTPError, OSError, RuntimeError, ValueError, TimeoutError) as exc:
        logger.warning("Failed to download narration audio from %s: %s", url, exc)
        ctx.narration_audio_dir = None
        # The single worst drop on this path: the whole video ships SILENT.
        # The docstring above has always said so ("produces a SILENT video with
        # the job still reported successful") -- this is what makes it true in
        # the report instead of only in a comment. The URL is tenant-supplied
        # and stays in the log line above; it must not reach the report.
        ctx.record_degradation("narration", "Narration audio could not be downloaded", exc)


__all__ = [
    "CompilerService",
    "FFmpegOps",
    "TitleCardGenerator",
    "VideoEditor",
    "_JobContext",
    "_webhook_tasks",
]


class CompilerService(
    _SegmentBuildersMixin,
    _AssemblyMixin,
    _EffectsMixin,
    _FinalizeMixin,
):
    """Orchestrates video compilation from recordings to final video."""

    def __init__(
        self,
        storage: StorageService,
        ws_manager: Any | None = None,
        job_lease: Any | None = None,
    ) -> None:
        self._storage = storage
        self._ws_manager = ws_manager
        self._job_lease = job_lease
        self._active_tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _heartbeat_interval(stale_timeout_seconds: float) -> float:
        """Beat several times inside the reaper's window, never near its edge.

        A quarter of the window gives at least three writes before the row could
        age out, so an ordinary scheduling delay or a slow beat cannot by itself
        trip the reaper. Clamped to [5s, 30s]: the floor keeps a small configured
        window from turning into a write storm, the ceiling keeps a large one from
        leaving the row untouched for minutes at a time.
        """
        return max(5.0, min(30.0, stale_timeout_seconds / 4))

    async def _progress_heartbeat(self, ctx: _JobContext, interval: float) -> None:
        """Write ``updated_at`` for as long as -- and only while -- work advances.

        This is the "silence, not age" half of the reaper fix. It wakes on a timer
        but writes on a COUNTER: if ``ctx.progress_ticks`` has not moved since the
        last beat, it writes NOTHING and the row ages exactly as the reaper
        intends. So:

          * a healthy compile past 600s keeps bumping ``updated_at`` and is never
            reaped, however long it legitimately runs;
          * a wedged compile (a stalled MinIO read, an ffmpeg step inside
            ``asyncio.to_thread`` that never returns) stops ticking while its
            event loop stays perfectly responsive -- so the beats keep firing,
            find no movement, write nothing, and the reaper collects it on time.

        A plain timer-driven heartbeat would have given the first property and
        destroyed the second, which is the leaked-lease-renewed-forever shape that
        hid the nine-hour recording wedge. Raising the 600s threshold instead
        would have traded a fast wrong answer for a slow one and left wedged
        compiles unreaped entirely.

        Uses its OWN session: ``process_job``'s ``AsyncSession`` is in use by the
        compile itself and is not safe for concurrent statements. The task is
        started with ``create_task`` inside the job's context, so it inherits the
        job's identity ContextVars and the Phase-D RLS bind applies to its session
        (same tenant re-seeded ⇒ provenance preserved, per w2v-common 0.67.7).
        """
        last_tick = -1
        while True:
            await asyncio.sleep(interval)
            tick = ctx.progress_ticks
            if tick == last_tick:
                continue  # SILENT: let the row age. This is the whole point.
            last_tick = tick
            try:
                async with get_db_session() as db:
                    still_running = await CompilationJobManager.record_progress(
                        db,
                        ctx.job_id,
                        progress=ctx.progress_percent,
                        stage=ctx.progress_stage,
                    )
            except Exception as exc:  # noqa: BLE001 -- a blip must not kill the compile
                logger.warning(
                    "Compilation progress heartbeat failed for job %s (%s); "
                    "the row will age toward the reaper until a beat succeeds",
                    ctx.job_id,
                    exc,
                )
                continue
            if not still_running:
                # The row left 'processing' underneath us (cancelled, or reaped).
                # Stop beating: continuing would be a heartbeat arguing with a
                # terminal state, and it can no longer write anything anyway.
                logger.info(
                    "Compilation job %s is no longer 'processing'; stopping progress heartbeat",
                    ctx.job_id,
                )
                return

    @contextlib.asynccontextmanager
    async def _job_liveness(self, ctx: _JobContext, job_id: str) -> AsyncGenerator[None]:
        """Hold a Redis liveness lease while a compilation job runs, so the
        lease-gated stale reaper never fails a job that is still executing on
        another pod (and can safely reclaim a crashed pod's job once its lease
        TTL-expires), AND heartbeat ``updated_at`` while the job makes progress so
        the DB-side reaper measures silence rather than elapsed time.

        Both halves are needed. The lease covers "is a pod still holding this
        job"; the progress heartbeat covers "is that pod still getting anywhere".
        A job can pass the first and fail the second -- that is precisely a wedge.
        """
        if self._job_lease is not None:
            await self._job_lease.acquire(job_id)
        interval = self._heartbeat_interval(ctx.settings.storage.stale_job_timeout_seconds)
        hb = asyncio.create_task(
            self._progress_heartbeat(ctx, interval), name=f"compile-progress-hb-{job_id}"
        )
        try:
            yield
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb
            if self._job_lease is not None:
                await self._job_lease.release(job_id)

    def register_task(self, job_id: str, task: asyncio.Task) -> None:
        """Track an active background compilation task by job ID."""
        self._active_tasks[job_id] = task

    def unregister_task(self, job_id: str) -> None:
        """Remove a completed or cancelled task from tracking."""
        self._active_tasks.pop(job_id, None)

    async def cancel_job(self, db: AsyncSession, job_id: str) -> bool:
        """Cancel a running job."""
        # Cancel asyncio task if active
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
        self.unregister_task(job_id)
        # Mark in DB
        cancelled = await CompilationJobManager.mark_cancelled(db, job_id)
        if cancelled:
            await db.commit()
        return cancelled

    async def create_job(
        self,
        db: AsyncSession,
        *,
        tenant_id: uuid.UUID,
        shot_list: dict,
        recordings: list[dict],
        narration_audio_dir: str | None = None,
        clips_dir: str | None = None,
        transition_type: str | None = None,
        crossfade_duration: float | None = None,
        xfade_transition: str | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        crf: int | None = None,
        theme_name: str | None = None,
        theme_config: dict | None = None,
        edit_decisions: list[dict] | None = None,
        preset: str | None = None,
        webhook_url: str | None = None,
        webhook_secret: str | None = None,
        color_correction: dict | None = None,
        watermark: dict | None = None,
        subtitles: dict | None = None,
    ) -> str:
        """Create a new compilation job record in the database."""
        # Store theme_name inside theme_config so it persists with the job
        merged_config = dict(theme_config) if theme_config else {}
        if theme_name:
            merged_config["theme_name"] = theme_name

        # Build filter_config from color correction + watermark + xfade_transition
        filter_config = None
        if color_correction or watermark or xfade_transition:
            filter_config = {}
            if color_correction:
                filter_config["color_correction"] = color_correction
            if watermark:
                filter_config["watermark"] = watermark
            if xfade_transition:
                filter_config["xfade_transition"] = xfade_transition

        job = await CompilationJobManager.create_job(
            db,
            tenant_id=tenant_id,
            shot_list=shot_list,
            recordings=recordings,
            narration_audio_dir=narration_audio_dir,
            clips_dir=clips_dir,
            transition_type=transition_type,
            crossfade_duration=crossfade_duration,
            target_width=target_width,
            target_height=target_height,
            crf=crf,
            theme_config=merged_config or None,
            edit_decisions=edit_decisions,
            preset=preset,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret,
            filter_config=filter_config,
            subtitles=subtitles,
        )
        return job.job_id

    async def _handle_process_job_exception(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: uuid.UUID,
        job: Any,
        exc: Exception,
        progress_callback: Any | None,
    ) -> None:
        from src.services.compilation.job_manager import JobStateConflictError

        if hasattr(exc, "command"):
            logger.error("Compilation job %s FFmpeg command: %s", job_id, exc.command)
        logger.error("Compilation job %s failed: %s", job_id, exc, exc_info=True)
        if progress_callback:
            await progress_callback.error(format_error(exc))
        try:
            await CompilationJobManager.mark_failed(db, job_id, format_error(exc))
        except JobStateConflictError:
            logger.info(
                "Compilation job %s already terminal; skipping mark_failed on exception",
                job_id,
            )
        try:
            await CompilationJobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
        except Exception:
            # Best-effort: a usage-counter write failure here must NOT replace
            # the real compilation error we are already handling (`exc`, raised
            # by the caller right after this method returns). Cost of this
            # degraded path: this job's failed-usage counter under-counts by
            # one for the day. Nothing reconciles that drift -- there is no
            # usage-counter repair job in this service.
            logger.exception(
                "Compilation job %s: record_usage failed while handling %s -- usage counter "
                "will under-count",
                job_id,
                type(exc).__name__,
            )
        await db.commit()
        if job.webhook_url:
            task = asyncio.create_task(
                self._fire_webhook(
                    job.webhook_url,
                    job.webhook_secret,
                    job_id,
                    "failed",
                    {"error": format_error(exc)},
                )
            )
            _webhook_tasks.add(task)
            task.add_done_callback(_webhook_tasks.discard)

    async def process_job(
        self,
        db: AsyncSession,
        job_id: str,
        tenant_id: uuid.UUID,
        progress_callback: Any | None = None,
    ) -> None:
        """Process a pending compilation job.

        Delegates to stage methods: _initialize, _process_segments,
        _assemble_video, _apply_effects, _finalize.
        """
        from src.config import get_settings
        from src.services.compilation.job_manager import JobStateConflictError

        logger.info("Processing job %s for tenant %s", job_id, tenant_id)
        job = await CompilationJobManager.get_job(db, job_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        try:
            await CompilationJobManager.mark_processing(db, job_id)
        except JobStateConflictError:
            logger.info(
                "Compilation job %s no longer pending (cancelled or claimed) -- exiting",
                job_id,
            )
            return

        ctx = _JobContext(
            job_id=job_id,
            tenant_id=tenant_id,
            job=job,
            settings=get_settings(),
            start_time=time.time(),
        )

        async with self._job_liveness(ctx, job_id):
            try:
                await self._initialize(ctx, progress_callback)

                await self._process_segments(ctx, progress_callback)

                if not ctx.segments:
                    try:
                        await CompilationJobManager.mark_failed(
                            db,
                            job_id,
                            "No segments to compile: all segments were empty or excluded",
                        )
                    except JobStateConflictError:
                        logger.info(
                            "Compilation job %s already terminal; skipping no-segments mark_failed",
                            job_id,
                        )
                    await CompilationJobManager.record_usage(db, tenant_id=tenant_id, is_error=True)
                    await db.commit()
                    return

                await asyncio.to_thread(self._assemble_video, ctx)

                if progress_callback:
                    await progress_callback.report("assembling", 85.0, "Assembling final video")

                await self._apply_effects(ctx, progress_callback)

                await self._finalize(ctx, db, progress_callback)

            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                await self._handle_process_job_exception(
                    db, job_id, tenant_id, job, exc, progress_callback
                )
                raise

            await db.commit()

            # WP-2: Fire webhook on completion/failure (non-blocking).
            #
            # Gated on `ctx.completed`, which is True only if `mark_completed`
            # actually wrote the row. It used to be unconditional, and `_finalize`
            # swallows a `JobStateConflictError` and returns normally -- so a job
            # the stale reaper had already failed still emitted "completed",
            # carrying an `output_path` that was never persisted. A consumer
            # trusting that webhook got a completion with no artifact.
            webhook_url = job.webhook_url
            webhook_secret = job.webhook_secret
            if webhook_url:
                if ctx.completed:
                    event = "completed"
                    payload: dict[str, str | float] = {
                        "output_path": str(ctx.output_path),
                        "duration": ctx.input_duration,
                    }
                else:
                    event = "failed"
                    payload = {
                        "error": (
                            "the job left 'processing' before its result could be stored "
                            "(cancelled, or reaped as stale); no output was persisted"
                        )
                    }
                task = asyncio.create_task(
                    self._fire_webhook(webhook_url, webhook_secret, job_id, event, payload)
                )
                _webhook_tasks.add(task)
                task.add_done_callback(_webhook_tasks.discard)

    @staticmethod
    def _confined_local_dir(job_id: str, value: str | None, *, field: str) -> Path | None:
        """Confine a tenant-supplied local *directory* to the storage root.

        ``narration_audio_dir`` and ``clips_dir`` are both bare on-disk
        directories that the compiler then joins filenames onto and reads —
        ``{dir}/narration_full.mp3``, ``{dir}/{part_id}.mp3``,
        ``{dir}/{part_id}.mp4`` — and ``narration_audio_dir`` is additionally
        ``iterdir()``-ed straight into the job log. Unconfined, either is an
        arbitrary read plus a directory-listing disclosure.

        This is the *same* boundary and the *same* helper the file-valued asset
        refs use (``resolve_local_asset_path``); a directory is just a path, and
        a second confinement mechanism is the thing to avoid. Refusal is
        fail-soft — the compile continues without narration / AI clips, exactly
        as it already does when the directory simply is not there.

        Args:
            job_id: For the log line only.
            value: The raw tenant string, or ``None``/empty when not supplied.
            field: Field name for the log line.

        Returns:
            The resolved, confined directory, or ``None`` when absent or refused.
        """
        from src.utils.url_guard import resolve_local_asset_path

        if not value:
            return None
        resolved = resolve_local_asset_path(value)
        if resolved is None:
            logger.warning("Job %s: %s refused by the local-path guard", job_id, field)
        return resolved

    async def _initialize(self, ctx: _JobContext, progress_callback: Any | None) -> None:
        """Set up work directories, resolve presets/themes, build lookups."""
        stage_t = time.time()
        ctx.work_dir = self._storage.get_job_dir(ctx.job_id)
        ctx.edited_dir = ctx.work_dir / "edited"
        ctx.title_cards_dir = ctx.work_dir / "title_cards"
        ctx.normalized_dir = ctx.work_dir / "normalized"
        ctx.edited_dir.mkdir(parents=True, exist_ok=True)
        ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
        ctx.normalized_dir.mkdir(parents=True, exist_ok=True)
        ctx.record_stage("initialize", stage_t)

        if progress_callback:
            await progress_callback.report("initializing", 5.0, "Setting up work directories")

        from src.services.compilation.presets import resolve_preset_config

        job = ctx.job
        settings = ctx.settings
        preset_config = resolve_preset_config(
            job.preset,
            target_width=job.target_width,
            target_height=job.target_height,
            crf=job.crf,
            fps=job.fps,
            transition_type=job.transition_type,
            crossfade_duration=job.crossfade_duration,
        )
        ctx.target_w = preset_config.get("target_width") or settings.compilation.target_width
        ctx.target_h = preset_config.get("target_height") or settings.compilation.target_height
        ctx.target_fps = preset_config.get("fps") or settings.compilation.target_fps
        ctx.cfr_mode = settings.compilation.cfr_mode
        ctx.crf = preset_config.get("crf") or settings.compilation.crf
        ctx.transition_type = (
            preset_config.get("transition_type") or settings.compilation.transition_type
        )
        ctx.crossfade_dur = (
            preset_config.get("crossfade_duration") or settings.compilation.crossfade_duration
        )

        shot_list = job.shot_list or {"parts": [], "recording_order": []}
        ctx.parts_by_id = {p["id"]: p for p in shot_list.get("parts", [])}
        ctx.recording_order = shot_list.get("recording_order", [])

        for r in job.recordings or []:
            if not r.get("success") or not r.get("video_path"):
                continue
            ctx.recording_map[str(r["part_id"])] = r

        for ed in job.edit_decisions or []:
            ctx.edit_decisions[str(ed["part_id"])] = ed

        from src.services.compilation.themes import resolve_theme_config

        raw_theme = job.theme_config or {}
        theme_name = raw_theme.get("theme_name")
        theme_config = resolve_theme_config(theme_name, raw_theme)
        ctx.theme_config = theme_config
        ctx.title_gen = TitleCardGenerator(theme_name=theme_name, theme_config=theme_config)
        ctx.editor = VideoEditor(output_dir=ctx.edited_dir, crf=ctx.crf)

        narration_dir_value = job.narration_audio_dir
        if narration_dir_value and (
            narration_dir_value.startswith("http://") or narration_dir_value.startswith("https://")
        ):
            # narration_audio_dir is a URL -- download the audio to a local directory.
            # SSRF sink on a tenant-controlled value: ``safe_get`` validates the
            # target and re-validates every redirect hop, trusting only the
            # configured MinIO artifact hosts. A guard rejection raises
            # ``SSRFError`` (a ``ValueError``), which the existing except tuple
            # already catches, so the fail-soft "carry on without narration"
            # behaviour below is preserved unchanged.
            #
            # That fail-soft path is exactly why ``max_fetch_bytes`` is stated
            # rather than inherited. ``ResponseTooLargeError`` is a
            # ``ValueError`` too, so w2v_common 0.55.0's 64 MiB default would be
            # caught right here and produce a SILENT video — no narration, no
            # error, just a warning line. 64 MiB is only ~70 minutes of 128 kbps
            # mp3, which a long-form narration passes, so this site takes
            # ARTIFACT_AUDIO_MAX_BYTES (256 MiB) instead.
            #
            # The fetch lives in the module-level ``_download_narration_audio``
            # so this sink can be driven on its own. It used to be inline in
            # ``_initialize``, which meant the only way to reach it was to stand
            # up the whole preparation stage (storage dirs, presets, themes,
            # editor) — expensive enough that the size behaviour of this sink
            # had no test at all.
            await _download_narration_audio(ctx, narration_dir_value)
        else:
            # LFI sink, and NOT the same case as the branch above: there the
            # directory is one this service just created under the job work dir,
            # so it is internally produced and must not be pushed through the
            # tenant guard. Here the value is the raw tenant string.
            ctx.narration_audio_dir = self._confined_local_dir(
                ctx.job_id, narration_dir_value, field="narration_audio_dir"
            )

        if ctx.narration_audio_dir:
            logger.info(
                "Job %s: narration_audio_dir=%s exists=%s",
                ctx.job_id,
                ctx.narration_audio_dir,
                ctx.narration_audio_dir.exists(),
            )
            if ctx.narration_audio_dir.exists():
                logger.info(
                    "Job %s: narration files: %s",
                    ctx.job_id,
                    list(ctx.narration_audio_dir.iterdir()),
                )
        else:
            logger.info("Job %s: no narration_audio_dir provided", ctx.job_id)

        # Same guard, same reason: ``{clips_dir}/{part_id}.mp4`` is read off the
        # pod and normalized into the output video. ``part_id`` cannot climb back
        # out of the confined directory because ``ShotListPartInput.id`` and
        # ``RecordingInput.part_id`` are restricted to a single filename
        # component at admission (``validate_path_component``).
        ctx.clips_dir = self._confined_local_dir(ctx.job_id, job.clips_dir, field="clips_dir")

        ctx.input_duration = sum(
            r.get("duration_seconds", 0.0) for r in (job.recordings or []) if r.get("success")
        )

    async def _fire_webhook(
        self,
        url: str,
        secret: str | None,
        job_id: str,
        event: str,
        data: dict,
    ) -> None:
        """Fire a webhook notification (WP-2)."""
        try:
            from src.services.compilation.webhook_service import fire_webhook

            await fire_webhook(url, secret, event, job_id, data)
        except OSError, RuntimeError, ValueError, TimeoutError:
            logger.warning("Webhook delivery failed for job %s", job_id)
