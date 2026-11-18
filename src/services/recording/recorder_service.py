"""Core recording orchestration service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import get_settings
from src.exceptions import ArtifactNotStoredError, RecordingError, VideoProductionError
from src.middleware.metrics import (
    record_duration_metric,
    record_file_size_metric,
    record_job_metric,
    set_active_jobs,
)
from src.services.ffmpeg import FFmpegOps
from src.services.recording.animation_recorder import AnimationRecorder
from src.services.recording.artifacts import (
    ARTIFACT_KIND_SCREENS_ZIP,
    artifact_kind_of,
    screens_zip_defect,
)
from src.services.recording.concurrency import (
    QUEUE_SATURATED_ERROR_PREFIX,
    RecordingQueueSaturatedError,
)
from src.services.recording.content_recorder import ContentRecorder
from src.services.recording.job_manager import JobStateConflictError, RecordingJobManager
from src.services.recording.playwright_recorder import PlaywrightRecorder
from src.services.recording.timing import ExecuteClock, JobTiming, WaitClock
from src.services.recording.url_recorder import UrlRecorder
from src.services.storage_service import StorageService
from src.utils.error_formatting import format_error

if TYPE_CHECKING:
    from src.database.models import RecordingJob
    from src.services.job_lease import JobLease
    from src.services.recording.concurrency import RecordingConcurrencyManager
    from src.services.recording.event_publisher import EventPublisher
    from src.services.recording.webhook_service import WebhookService

logger = logging.getLogger(__name__)

_webhook_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget webhook tasks


class RecorderService:
    """Orchestrates Playwright recording, retry logic, storage, and notifications.

    Coordinates the PlaywrightRecorder, StorageService, concurrency manager,
    webhook delivery, and event publishing for the full lifecycle of a
    recording job.
    """

    def __init__(
        self,
        storage: StorageService,
        recorder: PlaywrightRecorder,
        content_recorder: ContentRecorder | None = None,
        url_recorder: UrlRecorder | None = None,
        animation_recorder: AnimationRecorder | None = None,
        concurrency_manager: RecordingConcurrencyManager | None = None,
        webhook_service: WebhookService | None = None,
        event_publisher: EventPublisher | None = None,
        job_lease: JobLease | None = None,
    ) -> None:
        self._storage = storage
        self._recorder = recorder
        self._content_recorder = content_recorder
        self._url_recorder = url_recorder
        self._animation_recorder = animation_recorder
        self._concurrency = concurrency_manager
        self._webhook = webhook_service
        self._event_publisher = event_publisher
        self._job_lease = job_lease
        settings = get_settings()
        self._max_retries = settings.recording.max_retries
        self._retry_base_delay = settings.recording.retry_base_delay_s
        self._retryable_patterns = settings.recording.retryable_error_patterns
        self._retention_hours = settings.storage.retention_hours

    @contextlib.asynccontextmanager
    async def _job_liveness(self, job_id: str) -> AsyncGenerator[None]:
        """Hold a Redis liveness lease for the duration a job is processed.

        The lease-gated stale reaper (``cleanup_service``) skips any job whose
        lease is present, so a long-running job on this pod is never failed by
        another replica's reaper; on a crash the lease TTL-expires and the reaper
        can then safely reclaim (fail) the orphan.
        """
        if self._job_lease is not None:
            await self._job_lease.acquire(job_id)
        try:
            yield
        finally:
            if self._job_lease is not None:
                await self._job_lease.release(job_id)

    async def _acquire_slot(
        self,
        jm: RecordingJobManager,
        job_id: str,
        tenant_id: uuid.UUID,
        webhook_url: str | None,
        timing: JobTiming | None = None,
    ) -> None:
        """Mark the job queued and wait for a recording slot.

        The single place the queue wait happens, so saturation has exactly one
        outcome everywhere. If the wait times out, the job is failed HERE with the
        ``QUEUE_SATURATED`` marker and the error is re-raised. Letting it merely
        propagate would leave the row in ``queued`` forever — which is the state
        the (now much longer) queued-orphan window is specifically NOT meant to be
        the answer to.
        """
        if not self._concurrency:
            return
        await jm.mark_queued(job_id)
        try:
            # The ONLY place a job blocks on the fleet semaphore, so it is the
            # only place the wait can be measured. Timed here rather than by each
            # caller so no caller can forget, and so the wait can never drift back
            # into ``recording_time_seconds``.
            with WaitClock(timing):
                await self._concurrency.acquire(job_id)
        except RecordingQueueSaturatedError as exc:
            await self._handle_failure(
                jm,
                job_id,
                tenant_id,
                webhook_url,
                f"{QUEUE_SATURATED_ERROR_PREFIX}: {exc}",
            )
            raise
        set_active_jobs(self._concurrency.active_count)

    async def create_job(
        self,
        jm: RecordingJobManager,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        part_type: str | None = None,
        scripts_dir: str | None = None,
        recording_config: dict | None = None,
        auth_config: dict | None = None,
        webhook_url: str | None = None,
        batch_id: str | None = None,
        priority: int = 0,
        batch_order: int | None = None,
    ) -> str:
        """Create an async recording job, returns job_id."""
        max_retries = self._max_retries
        if recording_config and recording_config.get("max_retries") is not None:
            max_retries = recording_config["max_retries"]

        job = await jm.create_job(
            tenant_id=tenant_id,
            part_id=part_id,
            part_type=part_type,
            scripts_dir=scripts_dir,
            recording_config=recording_config,
            auth_config=auth_config,
            webhook_url=webhook_url,
            batch_id=batch_id,
            priority=priority,
            batch_order=batch_order,
            max_retries=max_retries,
        )
        return job.job_id

    async def _prepare_retry(
        self,
        jm: RecordingJobManager,
        job_id: str,
        attempt: int,
        max_retries: int,
        last_error: str | None,
    ) -> None:
        """Sleep with exponential backoff and update job state for a retry attempt.

        The concurrency slot is NOT touched here. The previous attempt's
        ``_attempt_recording`` already released its slot in its ``finally``, and
        the next attempt re-acquires a fresh one — ``_attempt_recording`` owns
        the whole acquire/release lifecycle. Releasing here would be a *phantom*
        release: the job holds no slot at this point, so ``Semaphore.release()``
        raises the local permit count above ``max_concurrent`` (over-admission
        past the OOM cap) and ``_held.popleft()`` pops a **different** running
        job's live global token off the FIFO, ``ZREM``-ing it early and freeing
        a stranger's slot. The job then re-acquires in both ``_prepare_retry``
        and ``_attempt_recording``, leaking one permit/token per retry.
        """
        delay = self._retry_base_delay * (2 ** (attempt - 1))
        logger.warning(
            "Retrying job %s (attempt %d/%d) after %.1fs delay — last_error: %s",
            job_id,
            attempt + 1,
            max_retries + 1,
            delay,
            (last_error or "")[: get_settings().job.log_error_truncate_length],
        )
        await asyncio.sleep(delay)
        await jm.update_retry_count(job_id, attempt)
        await jm.update_progress(job_id, 0.1, "retrying")
        await jm.commit()

    def _is_exception_retryable(self, exc: Exception, error_msg: str) -> bool:
        """Return True when *exc* is transient and worth retrying."""
        if isinstance(exc, VideoProductionError) and exc.retryable:
            return True
        return self._is_retryable(error_msg)

    async def _run_recording_attempt(
        self,
        jm: RecordingJobManager,
        job: RecordingJob,
        job_id: str,
        tenant_id: uuid.UUID,
        webhook_url: str | None,
        attempt: int,
        max_retries: int,
        timing: JobTiming,
    ) -> tuple[bool, str | None]:
        """Run a single recording attempt.

        Returns ``(done, last_error)``. When *done* is True the caller should
        stop iterating: success was achieved or a fatal error was raised.
        On a retryable failure we return ``(False, error_msg)`` so the caller
        can prepare another attempt.
        """
        try:
            result = await self._attempt_recording(jm, job, job_id, timing)

            if not result.get("success"):
                error_msg = result.get("error", "Recording failed")
                if attempt < max_retries and self._is_retryable(error_msg):
                    await jm.store_last_error(job_id, error_msg)
                    await jm.commit()
                    return False, error_msg
                raise RuntimeError(error_msg)

            await self._handle_success(
                jm, job_id, tenant_id, webhook_url, job.part_id, result, timing
            )
            return True, None

        except RecordingQueueSaturatedError:
            # Already terminal: _acquire_slot failed the job with the
            # QUEUE_SATURATED marker. Must NOT fall into the generic handler
            # below — a second mark_failed on a terminal row raises
            # JobStateConflictError and would replace an accurate capacity
            # diagnosis with a confusing one. Nor is it retryable in-process:
            # a retry re-queues into the same full fleet.
            raise
        except Exception as exc:
            last_error = format_error(exc)
            if attempt < max_retries and self._is_exception_retryable(exc, last_error):
                await jm.store_last_error(job_id, last_error)
                await jm.commit()
                return False, last_error

            await self._handle_failure(jm, job_id, tenant_id, webhook_url, last_error)
            raise

    async def process_job(self, jm: RecordingJobManager, job_id: str, tenant_id: uuid.UUID) -> None:
        """Process a pending recording job with retry support.

        Args:
            jm: RecordingJobManager instance wrapping the DB session.
            job_id: UUID of the job to process.
            tenant_id: Owning tenant identifier.

        Raises:
            ValueError: If the job_id does not exist.
            RuntimeError: If recording fails after all retries.
        """
        logger.info("Processing job %s for tenant %s", job_id, tenant_id)
        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        max_retries = job.max_retries or self._max_retries
        webhook_url = job.webhook_url

        # Note: the actual pending → processing transition happens inside
        # _attempt_recording (after acquiring a concurrency slot if configured).
        # Here we only publish the initial progress marker.
        async with self._job_liveness(job_id):
            await jm.update_progress(job_id, 0.1, "initializing")

            if self._event_publisher:
                await self._event_publisher.publish_status(tenant_id, job_id, "processing")

            # Accumulates across attempts. NOT a wall-clock start: the wall span
            # from here also contains every slot wait and every retry back-off.
            timing = JobTiming()
            last_error: str | None = None

            for attempt in range(max_retries + 1):
                if attempt > 0:
                    await self._prepare_retry(jm, job_id, attempt, max_retries, last_error)

                done, last_error = await self._run_recording_attempt(
                    jm, job, job_id, tenant_id, webhook_url, attempt, max_retries, timing
                )
                if done:
                    return

    async def _attempt_recording(
        self,
        jm: RecordingJobManager,
        job: RecordingJob,
        job_id: str,
        timing: JobTiming | None = None,
    ) -> dict:
        """Acquire a concurrency slot, run the recorder, and release the slot.

        Returns:
            The result dict produced by :pymethod:`PlaywrightRecorder.record_part`.
        """
        await self._acquire_slot(jm, job_id, job.tenant_id, job.webhook_url, timing)

        # Clock starts AFTER the slot is held. The wait that preceded it is
        # already recorded separately by _acquire_slot, and must never be folded
        # back into the execution figure.
        with ExecuteClock(timing):
            try:
                # INSIDE the try: the slot is already held, so ANY failure here --
                # notably JobStateConflictError when the reaper has already failed
                # a job that waited out its staleness window -- must still run the
                # `finally` release. With this call outside the try, three such
                # conflicts leaked all three fleet slots and wedged recording for
                # 9 hours with zero ERROR lines after the third.
                await jm.mark_processing(job_id)

                async def progress_cb(progress: float, stage: str) -> None:
                    await jm.update_progress(job_id, progress, stage)

                result = await self._recorder.record_part(
                    part_id=job.part_id,
                    scripts_dir=job.scripts_dir or "",
                    recording_config=job.recording_config,
                    auth_config=job.auth_config,
                    progress_callback=progress_cb,
                )
            finally:
                if self._concurrency:
                    await self._concurrency.release(job_id)
                    set_active_jobs(self._concurrency.active_count)

        return result

    async def _verify_artifact(self, job_id: str, source_path: Path, result: dict) -> None:
        """Run the integrity check that belongs to this artifact's KIND.

        Dispatches on the ``artifact_kind`` the PRODUCER declared, never on what
        the file looks like. "This is not a video" is a fact only the recorder
        knows; "this video failed to probe" is a fact only ffprobe knows. They
        are different, and collapsing them -- e.g. by catching a probe failure
        and continuing -- would let a corrupt recording through as a
        zero-duration success again.

        Raises:
            RecordingError: If the artifact fails its kind's check.
        """
        kind = artifact_kind_of(result)

        if kind == ARTIFACT_KIND_SCREENS_ZIP:
            screen_ids = result.get("screen_ids") or ()
            defect = await asyncio.to_thread(screens_zip_defect, source_path, screen_ids)
            if defect is not None:
                raise RecordingError(
                    f"Recording job {job_id} produced an unusable screens archive "
                    f"(path={source_path}): {defect}; treating as a capture failure "
                    f"rather than completing/billing an empty screen set"
                )
            return

        probe = await asyncio.to_thread(FFmpegOps.probe_all, source_path)
        if not probe.probed:
            raise RecordingError(
                f"Recording job {job_id} produced a video that ffprobe could "
                f"not read (path={source_path}); treating as a recording "
                f"failure rather than completing/billing a zero-duration result"
            )

    async def _handle_success(
        self,
        jm: RecordingJobManager,
        job_id: str,
        tenant_id: uuid.UUID,
        webhook_url: str | None,
        part_id: str,
        result: dict,
        timing: JobTiming,
    ) -> None:
        """Store the video, mark the job completed, record usage, and notify.

        Raises:
            ArtifactNotStoredError: If the recorder reported success but produced
                no retrievable artifact -- no ``video_path`` key, or a path that
                does not exist on disk. Completing such a job would set
                ``video_path=None`` on the row and still bill a
                ``recordings_completed`` unit plus its duration and bytes, for a
                recording that can never be downloaded. Raised **before** any
                ``mark_completed`` / ``record_usage`` write so the caller's
                failure path owns the job's terminal state.
            RecordingError: If the artifact fails the integrity check for its
                KIND (see ``src/services/recording/artifacts.py``). Raised
                **before** ``store_video`` and before ``mark_completed`` /
                ``record_usage``, so a known-bad artifact is never uploaded,
                completed or billed.

                For ``artifact_kind == "video"`` (the default, and what every
                recorder that declares nothing gets) the check is the recorder's
                own probe: ``FFmpegOps.probe_all`` could not read the file at
                all -- ffprobe crashed, timed out, or returned unparseable
                output (``ProbeResult.probed is False``). Every recorder folds
                that failure into ``duration=0.0``, which is indistinguishable
                from a genuinely empty file and would otherwise be stored as
                ``completed`` and billed. Re-probing here is cheap (same
                resolved-path+mtime cache key the recorder already populated on
                success, so ffprobe only actually re-runs in the failure case
                that is about to fail the job anyway).

                For ``artifact_kind == "screens_zip"`` the artifact is an
                ARCHIVE of per-screen stills/clips, not a video, and ffprobe
                cannot read one by construction -- probing it would veto every
                screens job. It gets its own equally strict check instead
                (``screens_zip_defect``): non-empty file, genuinely a zip, every
                entry CRC-clean, and a non-empty ``<id>.png``/``<id>.mp4`` for
                every screen the recorder reported capturing. The exemption is
                from ffprobe, NOT from verification: a corrupt or under-filled
                zip fails exactly as hard as an unprobeable video.

        Note ``store_video`` itself stays unguarded: a genuine upload failure
        already raises and already fails the job correctly.
        """
        await jm.update_progress(job_id, 0.9, "storing_video")
        await jm.commit()

        video_path_str = result.get("video_path")
        stored_path = None
        if video_path_str:
            source_path = Path(video_path_str)
            if source_path.exists():
                await self._verify_artifact(job_id, source_path, result)
                stored_path = await self._storage.store_video(job_id, source_path, part_id)

        if stored_path is None:
            raise ArtifactNotStoredError(
                f"Recording job {job_id} reported success but produced no retrievable "
                f"video (video_path={video_path_str!r}); refusing to complete or meter it"
            )

        await jm.mark_completed(
            job_id,
            video_path=stored_path,
            duration_seconds=result.get("duration_seconds"),
            file_size_bytes=result.get("file_size_bytes"),
            width=result.get("width"),
            height=result.get("height"),
            # The EFFECTIVE capture raster, as the recorder resolved it (see
            # AnimationRecorder._resolve_capture_scale). Persisted for the same
            # reason artifact_kind is: it is a fact only the PRODUCER knows, the
            # supersampling factor is stepped down silently under the raster
            # budget, and a consumer sizing source assets to the capture must
            # not have to re-derive it from a private constant. Recorders that
            # do not supersample report nothing and the columns stay NULL.
            capture_width=result.get("capture_width"),
            capture_height=result.get("capture_height"),
            # Slowest single frame's own capture time, as AnimationRecorder
            # measured it -- see Settings.animation_render_stall_window_seconds.
            # NULL from any recorder that does not frame-step.
            max_frame_seconds=result.get("max_frame_seconds"),
            has_audio=result.get("has_audio", False),
            # EXECUTION only. This used to be the wall span measured from BEFORE
            # the fleet semaphore was acquired, i.e. queue-wait + execution --
            # see src/services/recording/timing.py for the measurement showing
            # >=79% of the reported figure was in fact wait.
            recording_time_seconds=timing.execute_seconds,
            queue_wait_seconds=timing.wait_seconds,
            # The SAME declared fact `_verify_artifact` dispatched on, now
            # persisted so the download route can dispatch on it too instead of
            # guessing the media type from the stored suffix.
            artifact_kind=artifact_kind_of(result),
            retention_hours=self._retention_hours,
        )

        await jm.record_usage(
            tenant_id=tenant_id,
            recordings_completed=1,
            duration_seconds=result.get("duration_seconds", 0.0),
            file_size_bytes=result.get("file_size_bytes", 0),
        )
        await jm.commit()

        record_job_metric("completed")
        record_duration_metric(result.get("duration_seconds", 0.0))
        record_file_size_metric(result.get("file_size_bytes", 0))

        if self._event_publisher:
            await self._event_publisher.publish_status(tenant_id, job_id, "completed")

        if webhook_url and self._webhook:
            task = asyncio.create_task(
                self._webhook.notify(
                    webhook_url,
                    job_id,
                    "completed",
                    result={
                        "duration_seconds": result.get("duration_seconds"),
                        "file_size_bytes": result.get("file_size_bytes"),
                    },
                )
            )
            _webhook_tasks.add(task)
            task.add_done_callback(_webhook_tasks.discard)

        # Logged as two SEPARATE numbers, always -- on a saturated fleet and an
        # idle one alike. One combined number is what made a starved job
        # indistinguishable from a slow one.
        logger.info(
            "Job %s completed: execute=%.1fs queue_wait=%.1fs (video=%.1fs, %d bytes)",
            job_id,
            timing.execute_seconds,
            timing.wait_seconds,
            result.get("duration_seconds", 0.0),
            result.get("file_size_bytes", 0),
        )

    async def _handle_failure(
        self,
        jm: RecordingJobManager,
        job_id: str,
        tenant_id: uuid.UUID,
        webhook_url: str | None,
        error: str,
    ) -> None:
        """Mark the job as failed, record usage, and notify."""
        logger.error("Recording job %s failed: %s", job_id, error)
        try:
            await jm.mark_failed(job_id, error, retention_hours=self._retention_hours)
        except JobStateConflictError:
            logger.info("Recording job %s already terminal; skipping mark_failed", job_id)
        try:
            await jm.record_usage(tenant_id=tenant_id, is_error=True)
        except Exception:
            # Best-effort: a usage-counter write failure here must NOT replace
            # the real recording error already being handled -- the caller's
            # bare `raise` right after this method returns depends on that.
            # Cost of this degraded path: this job's failed-usage counter
            # under-counts by one for the day, and (since it's swallowed
            # before jm.commit()) the mark_failed status write above may also
            # fail to persist if the two share one DB transaction. Nothing in
            # this service reconciles that drift -- there is no usage-counter
            # or stuck-job repair sweep here.
            logger.exception(
                "Recording job %s: record_usage failed while handling failure -- usage "
                "counter will under-count",
                job_id,
            )
        await jm.commit()

        record_job_metric("failed")

        if self._event_publisher:
            await self._event_publisher.publish_status(tenant_id, job_id, "failed")

        if webhook_url and self._webhook:
            task = asyncio.create_task(
                self._webhook.notify(
                    webhook_url,
                    job_id,
                    "failed",
                    error=error,
                )
            )
            _webhook_tasks.add(task)
            task.add_done_callback(_webhook_tasks.discard)

    @staticmethod
    def _cleanup_staging(result: dict | None) -> None:
        """Best-effort removal of the per-job staging dir created by a recorder.

        The content/animation/url recorders stage every temp artifact (frames,
        webm, trimmed clips, zips, the encoded output) under one ``staging_dir``
        they return in their result dict. This fires after the output has been
        stored (success) or the job marked failed — on every path — so the
        ``/tmp`` staging dirs never accumulate. Idempotent and exception-safe.
        """
        if not result:
            return
        staging = result.get("staging_dir")
        if staging:
            shutil.rmtree(staging, ignore_errors=True)

    async def _finalize_result(
        self,
        jm: RecordingJobManager,
        job_id: str,
        tenant_id: uuid.UUID,
        webhook_url: str | None,
        part_id: str,
        result: dict,
        timing: JobTiming,
        default_error: str,
    ) -> None:
        """Route a recorder result to success/failure, then always clean staging."""
        try:
            if not result.get("success"):
                error_msg = result.get("error", default_error)
                await self._handle_failure(jm, job_id, tenant_id, webhook_url, error_msg)
                raise RuntimeError(error_msg)
            try:
                await self._handle_success(
                    jm, job_id, tenant_id, webhook_url, part_id, result, timing
                )
            except (ArtifactNotStoredError, RecordingError) as exc:
                # A recorder that claims success but hands back nothing storable
                # (ArtifactNotStoredError) or an unprobeable file
                # (RecordingError -- see _handle_success) is a failure, and must
                # be routed to the same terminal path as an explicit
                # ``success: False``. ``_run_recording_attempt`` does this for
                # the retrying job paths via its own ``except Exception``; the
                # content/animation/url/screens paths come through here
                # instead, so they need it too -- otherwise this raise would
                # escape with the job left in ``processing``.
                await self._handle_failure(jm, job_id, tenant_id, webhook_url, format_error(exc))
                raise
        finally:
            self._cleanup_staging(result)

    async def create_content_job(
        self,
        jm: RecordingJobManager,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        html_content: str,
        recording_config: dict | None = None,
        webhook_url: str | None = None,
    ) -> str:
        """Create a content recording job, storing HTML in script_source."""
        job = await jm.create_job(
            tenant_id=tenant_id,
            part_id=part_id,
            part_type="html_content",
            scripts_dir=None,
            script_source=html_content,
            recording_config=recording_config,
            webhook_url=webhook_url,
            max_retries=0,
        )
        return job.job_id

    async def process_content_job(
        self, jm: RecordingJobManager, job_id: str, tenant_id: uuid.UUID
    ) -> None:
        """Process a content recording job using ContentRecorder."""
        if self._content_recorder is None:
            raise RuntimeError("ContentRecorder not available")

        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        webhook_url = job.webhook_url
        html_content = job.script_source
        if not html_content:
            await self._handle_failure(jm, job_id, tenant_id, webhook_url, "No HTML content")
            raise RuntimeError("No HTML content stored for content recording job")

        async with self._job_liveness(job_id):
            await jm.update_progress(job_id, 0.1, "initializing")

            if self._event_publisher:
                await self._event_publisher.publish_status(tenant_id, job_id, "processing")

            timing = JobTiming()

            # Extract scroll_config from recording_config JSONB
            rec_config = dict(job.recording_config or {})
            scroll_config = rec_config.pop("scroll_config", None)

            async def progress_cb(progress: float, stage: str) -> None:
                await jm.update_progress(job_id, progress, stage)

            await self._acquire_slot(jm, job_id, tenant_id, webhook_url, timing)

            # Clock starts AFTER the slot is held. The wait that preceded it is
            # already recorded separately by _acquire_slot, and must never be
            # folded back into the execution figure.
            with ExecuteClock(timing):
                try:
                    # See _attempt_recording: mark_processing MUST be inside the try
                    # so a JobStateConflictError cannot leak the held slot.
                    await jm.mark_processing(job_id)
                    result = await self._content_recorder.record_content(
                        job_id=job_id,
                        html_content=html_content,
                        recording_config=rec_config or None,
                        scroll_config=scroll_config,
                        progress_callback=progress_cb,
                    )
                finally:
                    if self._concurrency:
                        await self._concurrency.release(job_id)
                        # Re-publish the gauge on release, exactly as the acquire
                        # side does. Omitting it here left `active_recording_jobs`
                        # pinned at its high-water mark forever on an idle pod --
                        # an operator reading saturation that does not exist, which
                        # is the fail-open direction for an observability signal
                        # and plausibly corroborated a false "capacity" theory
                        # during the 2026-09-03 outage investigation.
                        set_active_jobs(self._concurrency.active_count)

            await self._finalize_result(
                jm,
                job_id,
                tenant_id,
                webhook_url,
                job.part_id,
                result,
                timing,
                "Content recording failed",
            )

    async def create_animation_job(
        self,
        jm: RecordingJobManager,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        html_content: str,
        recording_config: dict | None = None,
        webhook_url: str | None = None,
    ) -> str:
        """Create a motion-graphics animation job.

        The template HTML is stored in ``script_source`` (mirroring the content
        path); the animation parameters (``params`` / ``fps`` / ``duration_ms`` /
        ``output_format``) ride inside ``recording_config`` JSONB under the
        ``animation_config`` key, alongside any viewport/quality overrides.
        """
        job = await jm.create_job(
            tenant_id=tenant_id,
            part_id=part_id,
            part_type="animation",
            scripts_dir=None,
            script_source=html_content,
            recording_config=recording_config,
            webhook_url=webhook_url,
            max_retries=0,
        )
        return job.job_id

    async def process_animation_job(
        self, jm: RecordingJobManager, job_id: str, tenant_id: uuid.UUID
    ) -> None:
        """Process a motion-graphics animation job using AnimationRecorder."""
        if self._animation_recorder is None:
            raise RuntimeError("AnimationRecorder not available")

        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        webhook_url = job.webhook_url
        html_content = job.script_source
        if not html_content:
            await self._handle_failure(jm, job_id, tenant_id, webhook_url, "No template HTML")
            raise RuntimeError("No template HTML stored for animation job")

        async with self._job_liveness(job_id):
            await jm.update_progress(job_id, 0.1, "initializing")
            if self._event_publisher:
                await self._event_publisher.publish_status(tenant_id, job_id, "processing")

            timing = JobTiming()
            rec_config = dict(job.recording_config or {})
            animation_config = rec_config.pop("animation_config", None)

            async def progress_cb(progress: float, stage: str) -> None:
                await jm.update_progress(job_id, progress, stage)

            await self._acquire_slot(jm, job_id, tenant_id, webhook_url, timing)

            # Clock starts AFTER the slot is held. The wait that preceded it is
            # already recorded separately by _acquire_slot, and must never be
            # folded back into the execution figure.
            with ExecuteClock(timing):
                try:
                    # See _attempt_recording: mark_processing MUST be inside the try
                    # so a JobStateConflictError cannot leak the held slot.
                    await jm.mark_processing(job_id)
                    result = await self._animation_recorder.record_animation(
                        job_id=job_id,
                        html_content=html_content,
                        animation_config=animation_config,
                        recording_config=rec_config or None,
                        progress_callback=progress_cb,
                    )
                finally:
                    if self._concurrency:
                        await self._concurrency.release(job_id)
                        # Re-publish the gauge (see the content path above).
                        set_active_jobs(self._concurrency.active_count)

            await self._finalize_result(
                jm,
                job_id,
                tenant_id,
                webhook_url,
                job.part_id,
                result,
                timing,
                "Animation recording failed",
            )

    async def create_url_job(
        self,
        jm: RecordingJobManager,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        url: str,
        auth_info: dict | None = None,
        recording_config: dict | None = None,
        scroll_config: dict | None = None,
        webhook_url: str | None = None,
    ) -> str:
        """Create a URL recording job.

        The target URL is stored in ``script_source`` (mirroring the content
        path's use of that column for inline HTML); ``scroll_config`` is stashed
        inside ``recording_config`` JSONB and ``auth_info`` in ``auth_config``.
        """
        merged_config = dict(recording_config or {})
        if scroll_config:
            merged_config["scroll_config"] = scroll_config

        job = await jm.create_job(
            tenant_id=tenant_id,
            part_id=part_id,
            part_type="url",
            scripts_dir=None,
            script_source=url,
            recording_config=merged_config or None,
            auth_config=auth_info or None,
            webhook_url=webhook_url,
            max_retries=0,
        )
        return job.job_id

    async def process_url_job(
        self, jm: RecordingJobManager, job_id: str, tenant_id: uuid.UUID
    ) -> None:
        """Process a URL recording job using UrlRecorder."""
        if self._url_recorder is None:
            raise RuntimeError("UrlRecorder not available")

        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        webhook_url = job.webhook_url
        url = job.script_source
        if not url:
            await self._handle_failure(jm, job_id, tenant_id, webhook_url, "No URL")
            raise RuntimeError("No URL stored for URL recording job")

        async with self._job_liveness(job_id):
            await jm.update_progress(job_id, 0.1, "initializing")

            if self._event_publisher:
                await self._event_publisher.publish_status(tenant_id, job_id, "processing")

            timing = JobTiming()

            rec_config = dict(job.recording_config or {})
            scroll_config = rec_config.pop("scroll_config", None)

            async def progress_cb(progress: float, stage: str) -> None:
                await jm.update_progress(job_id, progress, stage)

            await self._acquire_slot(jm, job_id, tenant_id, webhook_url, timing)

            # Clock starts AFTER the slot is held. The wait that preceded it is
            # already recorded separately by _acquire_slot, and must never be
            # folded back into the execution figure.
            with ExecuteClock(timing):
                try:
                    # See _attempt_recording: mark_processing MUST be inside the try
                    # so a JobStateConflictError cannot leak the held slot.
                    await jm.mark_processing(job_id)
                    result = await self._url_recorder.record_url(
                        job_id=job_id,
                        url=url,
                        auth_info=job.auth_config,
                        recording_config=rec_config or None,
                        scroll_config=scroll_config,
                        progress_callback=progress_cb,
                    )
                except Exception as exc:
                    # A recorder that RAISES instead of returning ``success: False``
                    # — notably ``SiteLoginFailedError``, a sign-in that did not
                    # work — must take the SAME terminal path as any other
                    # recording failure: mark_failed, record_usage(is_error=True),
                    # the failed metric, the SSE event and the webhook. Letting it
                    # escape untouched left only the route's
                    # ``_mark_job_failed_safe``, so the failure never reached
                    # metrics/SSE/webhook.
                    await self._handle_failure(
                        jm, job_id, tenant_id, webhook_url, format_error(exc)
                    )
                    raise
                finally:
                    if self._concurrency:
                        await self._concurrency.release(job_id)
                        # Re-publish the gauge (see the content path above).
                        set_active_jobs(self._concurrency.active_count)

            await self._finalize_result(
                jm,
                job_id,
                tenant_id,
                webhook_url,
                job.part_id,
                result,
                timing,
                "URL recording failed",
            )

    async def create_screens_job(
        self,
        jm: RecordingJobManager,
        *,
        tenant_id: uuid.UUID,
        part_id: str,
        url: str,
        steps: list[dict],
        auth_info: dict | None = None,
        recording_config: dict | None = None,
        webhook_url: str | None = None,
    ) -> str:
        """Create a per-screen capture job (one screenshot per screen, one zip).

        Mirrors :meth:`create_url_job`: target URL → ``script_source``, the
        ordered capture steps → ``recording_config`` JSONB under
        ``screen_steps``, auth → ``auth_config``.
        """
        merged_config = dict(recording_config or {})
        merged_config["screen_steps"] = steps
        job = await jm.create_job(
            tenant_id=tenant_id,
            part_id=part_id,
            part_type="screens",
            scripts_dir=None,
            script_source=url,
            recording_config=merged_config or None,
            auth_config=auth_info or None,
            webhook_url=webhook_url,
            max_retries=0,
        )
        return job.job_id

    async def process_screens_job(
        self, jm: RecordingJobManager, job_id: str, tenant_id: uuid.UUID
    ) -> None:
        """Process a per-screen capture job using UrlRecorder.capture_screens."""
        if self._url_recorder is None:
            raise RuntimeError("UrlRecorder not available")

        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        webhook_url = job.webhook_url
        url = job.script_source
        if not url:
            await self._handle_failure(jm, job_id, tenant_id, webhook_url, "No URL")
            raise RuntimeError("No URL stored for screen capture job")

        async with self._job_liveness(job_id):
            await jm.update_progress(job_id, 0.1, "initializing")

            if self._event_publisher:
                await self._event_publisher.publish_status(tenant_id, job_id, "processing")

            timing = JobTiming()

            rec_config = dict(job.recording_config or {})
            steps = rec_config.pop("screen_steps", None) or []
            record_clips = bool(rec_config.pop("record_clips", False))

            async def progress_cb(progress: float, stage: str) -> None:
                await jm.update_progress(job_id, progress, stage)

            await self._acquire_slot(jm, job_id, tenant_id, webhook_url, timing)

            # Clip mode records a short interaction clip + still per screen; the
            # default mode takes one still per screen. Both return the same result
            # shape (one zip in video_path), so the success path below is shared.
            capture = (
                self._url_recorder.capture_screen_clips
                if record_clips
                else self._url_recorder.capture_screens
            )
            # Clock starts AFTER the slot is held. The wait that preceded it is
            # already recorded separately by _acquire_slot, and must never be
            # folded back into the execution figure.
            with ExecuteClock(timing):
                try:
                    # See _attempt_recording: mark_processing MUST be inside the try
                    # so a JobStateConflictError cannot leak the held slot.
                    await jm.mark_processing(job_id)
                    result = await capture(
                        job_id=job_id,
                        url=url,
                        steps=steps,
                        auth_info=job.auth_config,
                        recording_config=rec_config or None,
                        progress_callback=progress_cb,
                    )
                except Exception as exc:
                    # See process_url_job: a recorder that RAISES (a sign-in that
                    # did not work is the live case) must take the same terminal
                    # path as one that returns ``success: False`` — otherwise the
                    # failure reaches neither metrics, SSE nor the webhook.
                    await self._handle_failure(
                        jm, job_id, tenant_id, webhook_url, format_error(exc)
                    )
                    raise
                finally:
                    if self._concurrency:
                        await self._concurrency.release(job_id)
                        # Re-publish the gauge (see the content path above).
                        set_active_jobs(self._concurrency.active_count)

            await self._finalize_result(
                jm,
                job_id,
                tenant_id,
                webhook_url,
                job.part_id,
                result,
                timing,
                "Screen capture failed",
            )

    async def cancel_job(
        self, jm: RecordingJobManager, job_id: str, *, tenant_id: uuid.UUID
    ) -> None:
        """Cancel a running or pending job owned by ``tenant_id``.

        Only cancels if the job is still in a cancellable state (pending,
        queued, or processing).  Jobs that have already completed or failed
        are left unchanged.  ``tenant_id`` must come from the caller's auth
        context; a job belonging to another tenant is treated as not found.
        """
        job = await jm.get_job(job_id, tenant_id=tenant_id)
        if job is None:
            return

        # Only cancel jobs that are still in a cancellable state
        if job.status not in ("pending", "queued", "processing"):
            logger.info(
                "Job %s already in terminal state '%s', skipping cancellation",
                job_id,
                job.status,
            )
            return

        # Try to kill the subprocess if running
        if job.status == "processing":
            await self._recorder.cancel_recording(job.part_id)

        try:
            await jm.mark_cancelled(job_id)
        except JobStateConflictError:
            logger.info("Job %s became terminal while cancelling — nothing to do", job_id)
            return
        record_job_metric("cancelled")

        # Clean up stored files
        await self._storage.cleanup_job(job_id)

        if self._event_publisher:
            # ``tenant_id`` is a uuid column and ``publish_status`` now takes the
            # UUID; it renders the ``recording:jobs:{tenant_id}`` channel itself,
            # so the canonical dashed spelling stays the one on the wire.
            await self._event_publisher.publish_status(job.tenant_id, job_id, "cancelled")

    def _is_retryable(self, error_message: str) -> bool:
        """Check if an error is transient and worth retrying."""
        error_lower = error_message.lower()
        return any(pattern.lower() in error_lower for pattern in self._retryable_patterns)
