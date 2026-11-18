"""Tests for src/services/recording/recorder_service.py — RecorderService orchestration.

Rewritten as part of P3.4 (test sweep) to remove SUT-internal ``mock.patch`` sites
in favour of Protocol-conforming fakes and real-behaviour assertions.

Mocking strategy (post-rewrite):

* JobManager — replaced with ``_FakeJobManager`` that records every call AND
  mutates a real ``RecordingJob`` instance, so tests assert *observable state*
  rather than "this method was called".
* Recorder / storage / concurrency / webhook / event publisher — replaced with
  small Protocol-shaped fakes that capture inputs and let tests inspect them.
* Prometheus metrics (``record_job_metric`` etc.) — NOT patched. Tests read
  the live ``prometheus_client.REGISTRY`` and assert counter deltas. This means
  if a metric label or function signature drifts, the test catches it.
* ``get_settings`` — still patched at the module boundary (config is a true
  system boundary, not SUT internal).
"""

from __future__ import annotations

import asyncio
import tempfile
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from src.database.models import RecordingJob
from src.exceptions import ArtifactNotStoredError, RecordingError, SiteLoginFailedError
from src.services.ffmpeg.base import ProbeResult
from src.services.recording.artifacts import ARTIFACT_KIND_SCREENS_ZIP
from src.services.recording.concurrency import (
    QUEUE_SATURATED_ERROR_PREFIX,
    RecordingConcurrencyManager,
    RecordingQueueSaturatedError,
)
from src.services.recording.event_publisher import EventPublisher
from src.services.recording.job_manager import JobStateConflictError, RecordingJobManager
from src.services.recording.playwright_recorder import PlaywrightRecorder
from src.services.recording.recorder_service import RecorderService
from src.services.recording.webhook_service import WebhookService
from src.services.storage_service import StorageService
from tests.conftest import TENANT_U1, make_job

# ── Fakes (Protocol-conforming) ─────────────────────────────────────


class _FakeJobManager:
    """Real fake for ``RecordingJobManager`` that mutates a stored job in-place.

    Implements the subset of the manager interface used by ``RecorderService``.
    Records every call so tests can assert ordering and inputs, but the
    primary observable is the mutated ``RecordingJob`` itself.
    """

    def __init__(
        self, job: RecordingJob | None, *, conflict_on_mark_processing: bool = False
    ) -> None:
        self._job = job
        # When True, ``mark_processing`` raises ``JobStateConflictError`` — the
        # production shape when the reaper has already failed a job that waited
        # out its staleness window.
        self._conflict_on_mark_processing = conflict_on_mark_processing
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.commit_count = 0
        self.rollback_count = 0
        self.usage_records: list[dict[str, Any]] = []
        self.last_errors: list[str] = []
        self.progress_updates: list[tuple[float, str]] = []
        self.retry_count_updates: list[int] = []

    async def create_job(self, **kwargs: Any) -> RecordingJob:
        self.calls.append(("create_job", kwargs))
        # In production the manager builds a new row; the fake re-uses
        # whatever job the test seeded so callers can inspect its fields.
        if self._job is None:
            self._job = make_job(job_id=kwargs.get("part_id", "auto"))
        # Record max_retries on the job for assertions
        if "max_retries" in kwargs:
            self._job.max_retries = kwargs["max_retries"]
        return self._job

    async def get_job(self, job_id: str, *, tenant_id: str) -> RecordingJob | None:
        self.calls.append(("get_job", {"job_id": job_id, "tenant_id": tenant_id}))
        return self._job

    async def mark_processing(self, job_id: str) -> None:
        self.calls.append(("mark_processing", {"job_id": job_id}))
        if self._conflict_on_mark_processing:
            raise JobStateConflictError(
                f"Recording job {job_id!r} is not pending/queued -- cannot mark_processing"
            )
        if self._job is not None:
            self._job.status = "processing"

    async def mark_queued(self, job_id: str) -> None:
        self.calls.append(("mark_queued", {"job_id": job_id}))
        if self._job is not None:
            self._job.status = "queued"

    async def mark_completed(self, job_id: str, **kwargs: Any) -> None:
        self.calls.append(("mark_completed", {"job_id": job_id, **kwargs}))
        if self._job is not None:
            self._job.status = "completed"

    async def mark_failed(self, job_id: str, error: str, **kwargs: Any) -> None:
        self.calls.append(("mark_failed", {"job_id": job_id, "error": error, **kwargs}))
        if self._job is not None:
            self._job.status = "failed"
            self._job.error_message = error

    async def mark_cancelled(self, job_id: str) -> None:
        self.calls.append(("mark_cancelled", {"job_id": job_id}))
        if self._job is not None:
            self._job.status = "cancelled"

    async def update_progress(self, job_id: str, progress: float, stage: str) -> None:
        self.calls.append(
            ("update_progress", {"job_id": job_id, "progress": progress, "stage": stage})
        )
        self.progress_updates.append((progress, stage))
        if self._job is not None:
            self._job.progress = progress

    async def update_retry_count(self, job_id: str, retry_count: int) -> None:
        self.calls.append(("update_retry_count", {"job_id": job_id, "retry_count": retry_count}))
        self.retry_count_updates.append(retry_count)
        if self._job is not None:
            self._job.retry_count = retry_count

    async def store_last_error(self, job_id: str, error: str) -> None:
        self.calls.append(("store_last_error", {"job_id": job_id, "error": error}))
        self.last_errors.append(error)

    async def record_usage(self, **kwargs: Any) -> None:
        self.calls.append(("record_usage", kwargs))
        self.usage_records.append(kwargs)

    async def commit(self) -> None:
        self.commit_count += 1
        self.calls.append(("commit", {}))

    async def rollback(self) -> None:
        self.rollback_count += 1
        self.calls.append(("rollback", {}))


@dataclass
class _FakeRecorder:
    """Real fake for ``PlaywrightRecorder`` — returns scripted results."""

    results: list[dict[str, Any]] = field(default_factory=list)
    cancel_returns: bool = True
    record_calls: list[dict[str, Any]] = field(default_factory=list)
    cancel_calls: list[str] = field(default_factory=list)

    async def record_part(
        self,
        *,
        part_id: str,
        scripts_dir: str,
        recording_config: Any | None = None,
        auth_config: Any | None = None,
        progress_callback: Callable[[float, str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        self.record_calls.append(
            {
                "part_id": part_id,
                "scripts_dir": scripts_dir,
                "recording_config": recording_config,
                "auth_config": auth_config,
            }
        )
        # Drive the progress callback so we exercise it too.
        if progress_callback is not None:
            await progress_callback(0.5, "recording")
        if not self.results:
            raise RuntimeError("FakeRecorder ran out of scripted results")
        return self.results.pop(0)

    async def cancel_recording(self, part_id: str) -> bool:
        self.cancel_calls.append(part_id)
        return self.cancel_returns


@dataclass
class _FakeStorage:
    """Real fake for ``StorageService``."""

    stored: list[tuple[str, str, str]] = field(default_factory=list)
    cleaned: list[str] = field(default_factory=list)
    store_return: str = "/stored/video.mp4"

    async def store_video(self, job_id: str, source_path: Any, part_id: str) -> str:
        self.stored.append((job_id, str(source_path), part_id))
        return self.store_return

    async def cleanup_job(self, job_id: str) -> None:
        self.cleaned.append(job_id)


@dataclass
class _FakeConcurrency:
    """Real fake for ``RecordingConcurrencyManager``."""

    active_count: int = 0
    acquire_count: int = 0
    release_count: int = 0

    async def acquire(self, owner: str | None = None) -> None:
        self.acquire_count += 1
        self.active_count += 1

    async def release(self, owner: str | None = None) -> None:
        self.release_count += 1
        self.active_count = max(0, self.active_count - 1)


@dataclass
class _FakeWebhook:
    """Real fake for ``WebhookService.notify``."""

    notifications: list[dict[str, Any]] = field(default_factory=list)

    async def notify(self, url: str, job_id: str, status: str, **payload: Any) -> None:
        self.notifications.append(
            {"url": url, "job_id": job_id, "status": status, "payload": payload}
        )


@dataclass
class _FakeEventPublisher:
    """Real fake for ``EventPublisher.publish_status``."""

    events: list[tuple[str, str, str]] = field(default_factory=list)

    async def publish_status(self, tenant_id: str, job_id: str, status: str) -> None:
        self.events.append((tenant_id, job_id, status))


# ── Helpers ────────────────────────────────────────────────────────


def _mock_settings() -> MagicMock:
    """Return a MagicMock settings with recording/storage defaults."""
    s = MagicMock()
    s.recording.max_retries = 2
    s.recording.retry_base_delay_s = 0.01  # fast retries in tests
    s.recording.retryable_error_patterns = [
        "timed out",
        "browser disconnected",
        "target closed",
    ]
    s.recording.output_format = "mp4"
    s.recording.default_quality_preset = "standard"
    s.storage.retention_hours = 168
    s.job.log_error_truncate_length = 200
    return s


def _make_service(
    *,
    storage: _FakeStorage | None = None,
    recorder: _FakeRecorder | None = None,
    concurrency_manager: _FakeConcurrency | RecordingConcurrencyManager | None = None,
    webhook_service: _FakeWebhook | None = None,
    event_publisher: _FakeEventPublisher | None = None,
) -> RecorderService:
    """Build a RecorderService wired up with fakes (config patched at boundary)."""
    storage = storage if storage is not None else _FakeStorage()
    recorder = recorder if recorder is not None else _FakeRecorder()
    with patch(
        "src.services.recording.recorder_service.get_settings", return_value=_mock_settings()
    ):
        # The fakes are structural duck-types of the production collaborators.
        # We use ``cast`` so the type-checker sees the intended hand-off
        # rather than silently suppressing a diagnostic.
        return RecorderService(
            storage=cast(StorageService, storage),
            recorder=cast(PlaywrightRecorder, recorder),
            concurrency_manager=cast(RecordingConcurrencyManager | None, concurrency_manager),
            webhook_service=cast(WebhookService | None, webhook_service),
            event_publisher=cast(EventPublisher | None, event_publisher),
        )


def _make_jm(
    job: RecordingJob | None, *, conflict_on_mark_processing: bool = False
) -> tuple[RecordingJobManager, _FakeJobManager]:
    """Return ``(typed_manager, fake)``.

    The first element is the same fake re-typed as ``RecordingJobManager`` so
    the type-checker accepts hand-offs to ``RecorderService``; the second is
    the same instance kept untyped so tests can read the recorded calls.
    """
    fake = _FakeJobManager(job, conflict_on_mark_processing=conflict_on_mark_processing)
    return cast(RecordingJobManager, fake), fake


# The production recorders hand back a path to a file that really exists, and
# ``_handle_success`` now refuses to complete (or bill) a job whose artifact is
# not retrievable. So the happy-path helper must return a real path: before this
# it defaulted to ``video_path=None``, which meant every "success" test in this
# module was in fact exercising -- and pinning as correct -- the
# completed-but-unretrievable bug.
_ARTIFACT_DIR = tempfile.TemporaryDirectory(prefix="vprec-tests-")
_DEFAULT_VIDEO_PATH = Path(_ARTIFACT_DIR.name) / "recorded.webm"
_DEFAULT_VIDEO_PATH.write_bytes(b"fake recorded video")


def _success_result(
    duration: float = 5.0,
    size: int = 50000,
    video_path: str | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "video_path": video_path if video_path is not None else str(_DEFAULT_VIDEO_PATH),
        "duration_seconds": duration,
        "file_size_bytes": size,
        "width": 1920,
        "height": 1080,
        "has_audio": False,
    }


def _write_screens_zip(path: Path, screen_ids: tuple[str, ...] = ("home",)) -> Path:
    """Write a REAL zip shaped like ``UrlRecorder._zip_screens`` output.

    The production screens artifact is an archive of ``<capture_id>.png``
    entries. Fixtures that write ``b"zip-bytes"`` to a file named ``.zip`` do
    not exercise the archive integrity check at all -- they are indistinguishable
    from any other opaque blob -- so every screens success fixture here writes a
    genuine zip.
    """
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for screen_id in screen_ids:
            zf.writestr(f"{screen_id}.png", b"\x89PNG\r\n\x1a\n" + b"pixels" * 8)
    return path


def _screens_result(
    zip_path: Path,
    screen_ids: tuple[str, ...] = ("home",),
    size: int = 4242,
) -> dict[str, Any]:
    """A screens-capture result in the shape the real ``UrlRecorder`` returns.

    Crucially it carries ``artifact_kind="screens_zip"`` and ``screen_ids`` --
    the two fields the success path dispatches on. Without them a test would
    silently take the VIDEO branch and prove nothing about screens.
    """
    result = _success_result(duration=0.0, size=size, video_path=str(zip_path))
    result["artifact_kind"] = ARTIFACT_KIND_SCREENS_ZIP
    result["screen_ids"] = list(screen_ids)
    return result


def _counter_value(metric_name: str, **labels: str) -> float:
    """Read the current value of a labelled prometheus counter (or 0.0 if absent)."""
    for sample in REGISTRY.collect():
        for s in sample.samples:
            if s.name == metric_name and all(s.labels.get(k) == v for k, v in labels.items()):
                return s.value
    return 0.0


def _histogram_count(metric_name: str) -> float:
    """Read the current observation count of a histogram."""
    return _counter_value(f"{metric_name}_count")


def _gauge_value(metric_name: str) -> float:
    """Read the current value of a gauge."""
    for sample in REGISTRY.collect():
        for s in sample.samples:
            if s.name == metric_name:
                return s.value
    return 0.0


@pytest.fixture(autouse=True)
def _default_successful_probe():
    """Default every test in this module to a successful re-probe.

    ``_handle_success`` re-probes the stored artifact via ``FFmpegOps.probe_all``
    to guard against billing an unprobeable recording (see
    ``TestNoBillingOnUnprobedRecording`` below). The success-path fixtures in
    this module (``_DEFAULT_VIDEO_PATH`` etc.) write throwaway byte content,
    not a real container ffprobe can parse, so without this default every
    pre-existing success-path test would trip the new guard for the wrong
    reason. Tests that specifically exercise the guard override this with
    their own nested ``patch(...)``.
    """
    ok_probe = ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=True)
    with patch(
        "src.services.recording.recorder_service.FFmpegOps.probe_all",
        return_value=ok_probe,
    ):
        yield


# ── create_job ─────────────────────────────────────────────────────


class TestCreateJob:
    async def test_creates_job_returns_job_id(self) -> None:
        service = _make_service()
        mock_job = make_job(job_id="j-123")
        jm, fake = _make_jm(mock_job)

        result = await service.create_job(
            jm,
            tenant_id=TENANT_U1,
            part_id="p1",
            scripts_dir="/scripts",
        )

        # Observable: the returned id matches the seeded job and the manager
        # was asked to persist a row with the right tenant/part.
        assert result == "j-123"
        create_calls = [c for c in fake.calls if c[0] == "create_job"]
        assert len(create_calls) == 1
        assert create_calls[0][1]["tenant_id"] == TENANT_U1
        assert create_calls[0][1]["part_id"] == "p1"
        assert create_calls[0][1]["scripts_dir"] == "/scripts"

    async def test_uses_custom_max_retries_from_config(self) -> None:
        service = _make_service()
        jm, fake = _make_jm(make_job(job_id="j-456"))

        await service.create_job(
            jm,
            tenant_id=TENANT_U1,
            part_id="p1",
            recording_config={"max_retries": 5},
        )

        # Observable: the recording_config override actually flows through to
        # the persisted row.
        create_call = next(c for c in fake.calls if c[0] == "create_job")
        assert create_call[1]["max_retries"] == 5
        # The mutated job should reflect the new max_retries (state assertion).
        assert fake._job is not None
        assert fake._job.max_retries == 5

    async def test_defaults_max_retries_from_settings(self) -> None:
        service = _make_service()
        jm, fake = _make_jm(make_job(job_id="j-789"))

        await service.create_job(jm, tenant_id=TENANT_U1, part_id="p1")

        create_call = next(c for c in fake.calls if c[0] == "create_job")
        assert create_call[1]["max_retries"] == 2  # from _mock_settings


# ── process_job ────────────────────────────────────────────────────


class TestProcessJob:
    async def test_successful_recording(self) -> None:
        before_completed = _counter_value("recording_jobs_total", status="completed")
        before_dur_count = _histogram_count("recording_duration_seconds")
        before_size_count = _histogram_count("recording_file_size_bytes")

        job = make_job(job_id="j-ok", status="pending", max_retries=2)
        jm, fake = _make_jm(job)
        recorder = _FakeRecorder(results=[_success_result(duration=5.0, size=50000)])
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)

        await service.process_job(jm, "j-ok", TENANT_U1)

        # Observable state on the job itself.
        assert job.status == "completed"
        # The recorder was driven with the right inputs.
        assert recorder.record_calls[0]["part_id"] == job.part_id
        # JobManager committed at least once and persisted completion.
        assert fake.commit_count >= 1
        assert any(call[0] == "mark_completed" for call in fake.calls)
        # Real prometheus state — counter and histograms ticked.
        after_completed = _counter_value("recording_jobs_total", status="completed")
        assert after_completed - before_completed == pytest.approx(1.0)
        assert _histogram_count("recording_duration_seconds") - before_dur_count == pytest.approx(
            1.0
        )
        assert _histogram_count("recording_file_size_bytes") - before_size_count == pytest.approx(
            1.0
        )

    async def test_job_not_found_raises(self) -> None:
        jm, fake = _make_jm(None)
        service = _make_service()

        with pytest.raises(ValueError, match="Job not found"):
            await service.process_job(jm, "j-missing", TENANT_U1)

        # No status mutations occurred.
        assert all(call[0] != "mark_processing" for call in fake.calls)

    async def test_non_retryable_failure_raises(self) -> None:
        before_failed = _counter_value("recording_jobs_total", status="failed")

        job = make_job(job_id="j-fail", status="pending", max_retries=2)
        jm, fake = _make_jm(job)
        recorder = _FakeRecorder(results=[{"success": False, "error": "Permission denied"}])
        service = _make_service(recorder=recorder)

        with pytest.raises(RuntimeError, match="Permission denied"):
            await service.process_job(jm, "j-fail", TENANT_U1)

        # Observable: job moved to failed exactly once and stored the error.
        assert job.status == "failed"
        mark_failed_calls = [c for c in fake.calls if c[0] == "mark_failed"]
        assert len(mark_failed_calls) == 1
        # The recorder service wraps the underlying RuntimeError via
        # ``format_error`` before persisting; assert the original message is
        # preserved within the wrapped form.
        assert "Permission denied" in mark_failed_calls[0][1]["error"]
        # Real prometheus counter ticked exactly once.
        after_failed = _counter_value("recording_jobs_total", status="failed")
        assert after_failed - before_failed == pytest.approx(1.0)
        # Non-retryable: the recorder should have been called only once.
        assert len(recorder.record_calls) == 1

    async def test_record_usage_failure_does_not_mask_real_error(self) -> None:
        """A record_usage() blowup inside ``_handle_failure`` must not replace
        the real recording error.

        Regression test: ``_handle_failure`` used to call
        ``jm.record_usage`` unguarded. If it raised, that new exception
        silently replaced the real recording failure on the bare ``raise``
        in ``_run_recording_attempt`` -- every failed recording job's actual
        cause was erased from the propagated exception.
        """

        class _BoomJobManager(_FakeJobManager):
            async def record_usage(self, **kwargs: Any) -> None:
                self.calls.append(("record_usage", kwargs))
                raise ConnectionError("usage-db pool exhausted")

        job = make_job(job_id="j-fail", status="pending", max_retries=2)
        fake = _BoomJobManager(job)
        jm = cast(RecordingJobManager, fake)
        recorder = _FakeRecorder(results=[{"success": False, "error": "Permission denied"}])
        service = _make_service(recorder=recorder)

        with pytest.raises(RuntimeError, match="Permission denied") as exc_info:
            await service.process_job(jm, "j-fail", TENANT_U1)

        # The original recording error propagated, not the ConnectionError
        # from the usage-counter write.
        assert isinstance(exc_info.value, RuntimeError)
        assert not isinstance(exc_info.value, ConnectionError)

        # Second-order: mark_failed still ran and commit() still happened
        # despite record_usage blowing up.
        mark_failed_calls = [c for c in fake.calls if c[0] == "mark_failed"]
        assert len(mark_failed_calls) == 1
        assert job.status == "failed"
        assert fake.commit_count >= 1

    async def test_retryable_error_retries_then_succeeds(self) -> None:
        job = make_job(job_id="j-retry", status="pending", max_retries=2)
        jm, fake = _make_jm(job)
        recorder = _FakeRecorder(
            results=[
                {"success": False, "error": "Recording timed out"},
                _success_result(duration=3.0, size=30000),
            ]
        )
        service = _make_service(recorder=recorder)

        await service.process_job(jm, "j-retry", TENANT_U1)

        # Observable: two attempts ran, the first error was stored, retry
        # bookkeeping advanced, and final state is completed.
        assert len(recorder.record_calls) == 2
        assert fake.last_errors == ["Recording timed out"]
        assert fake.retry_count_updates == [1]
        assert job.status == "completed"

    async def test_event_publisher_called_on_success(self) -> None:
        job = make_job(job_id="j-pub", status="pending", max_retries=0)
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder(results=[_success_result(duration=1.0, size=1000)])
        events = _FakeEventPublisher()
        service = _make_service(recorder=recorder, event_publisher=events)

        await service.process_job(jm, "j-pub", TENANT_U1)

        # Observable: the publisher saw exactly the (processing, completed) pair.
        statuses = [evt[2] for evt in events.events]
        assert statuses == ["processing", "completed"]
        assert all(evt[0] == TENANT_U1 and evt[1] == "j-pub" for evt in events.events)

    async def test_event_publisher_called_on_failure(self) -> None:
        job = make_job(job_id="j-fpub", status="pending", max_retries=0)
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder(results=[{"success": False, "error": "fatal error"}])
        events = _FakeEventPublisher()
        service = _make_service(recorder=recorder, event_publisher=events)

        with pytest.raises(RuntimeError):
            await service.process_job(jm, "j-fpub", TENANT_U1)

        statuses = [evt[2] for evt in events.events]
        assert statuses == ["processing", "failed"]

    async def test_concurrency_manager_acquire_release(self) -> None:
        before_active = _gauge_value("active_recording_jobs")
        job = make_job(job_id="j-conc", status="pending", max_retries=0)
        jm, fake = _make_jm(job)
        recorder = _FakeRecorder(results=[_success_result(duration=2.0, size=20000)])
        conc = _FakeConcurrency()
        service = _make_service(recorder=recorder, concurrency_manager=conc)

        await service.process_job(jm, "j-conc", TENANT_U1)

        # Observable: every acquire was matched by a release, and the gauge
        # ended at the same baseline (active_count returned to 0).
        assert conc.acquire_count == 1
        assert conc.release_count == 1
        assert conc.active_count == 0
        # The gauge was driven to reflect active_count and ended at the baseline.
        assert _gauge_value("active_recording_jobs") == pytest.approx(before_active)
        # And we definitely went through the queued state.
        assert any(call[0] == "mark_queued" for call in fake.calls)

    async def test_retry_does_not_leak_a_concurrency_slot(self) -> None:
        """A retried job must own exactly one acquire/release per attempt.

        Regression: ``_prepare_retry`` used to ``release()`` then ``acquire()``
        the concurrency manager, but the previous attempt's ``_attempt_recording``
        had already released its slot in its ``finally`` — so the release ran
        while the job held nothing (a phantom release that, in prod, over-counts
        the local semaphore past ``max_concurrent`` and pops a *different* running
        job's live global token). The job then acquired twice (in ``_prepare_retry``
        and ``_attempt_recording``) but released once per attempt, leaking one
        permit/token per retry. With ``_FakeConcurrency`` flooring ``active_count``
        at 0, the buggy path ends at ``active_count == 1`` and 3 acquires; the fix
        keeps it perfectly balanced at one acquire/release per attempt."""
        job = make_job(job_id="j-retry-conc", status="pending", max_retries=2)
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder(
            results=[
                {"success": False, "error": "Recording timed out"},
                _success_result(duration=3.0, size=30000),
            ]
        )
        conc = _FakeConcurrency()
        service = _make_service(recorder=recorder, concurrency_manager=conc)

        await service.process_job(jm, "j-retry-conc", TENANT_U1)

        # Two attempts ran; each acquired and released exactly once — no phantom
        # release, no leaked slot.
        assert len(recorder.record_calls) == 2
        assert job.status == "completed"
        assert conc.acquire_count == 2
        assert conc.release_count == 2
        assert conc.active_count == 0

    async def test_mark_processing_conflict_does_not_leak_the_slot(self) -> None:
        """A ``mark_processing`` failure AFTER the slot is held must still release it.

        ROOT CAUSE of the 2026-09-04 outage. ``mark_processing`` sat between
        ``acquire()`` and the ``try:`` whose ``finally`` releases the slot. When
        the reaper had already failed a job that waited out its 600s staleness
        window, ``mark_processing`` raised ``JobStateConflictError`` and the held
        slot was never released. Three such conflicts consumed all three
        fleet-wide slots; the Redis heartbeat then kept refreshing the leaked
        tokens forever, defeating TTL auto-release, and every subsequent job sat
        in ``queued`` until the reaper failed it -- 303 failed / 0 completed in
        one day, with the pod reporting Ready and zero ERROR lines.

        Measured live before the restart: ``active=3, queued=472,
        available_slots=0``.
        """
        job = make_job(job_id="j-conflict", status="pending", max_retries=0)
        jm, _fake = _make_jm(job, conflict_on_mark_processing=True)

        recorder = _FakeRecorder(results=[_success_result(duration=1.0, size=1000)])
        conc = _FakeConcurrency()
        service = _make_service(recorder=recorder, concurrency_manager=conc)

        with pytest.raises(JobStateConflictError):
            await service.process_job(jm, "j-conflict", TENANT_U1)

        # The slot was taken and MUST have been handed back.
        assert conc.acquire_count == 1
        assert conc.release_count == 1, "mark_processing conflict leaked a concurrency slot"
        assert conc.active_count == 0
        # And the recorder never ran -- the conflict aborted before capture.
        assert recorder.record_calls == []

    async def test_webhook_on_success(self) -> None:
        job = make_job(
            job_id="j-wh",
            status="pending",
            max_retries=0,
            webhook_url="http://example.com/hook",
        )
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder(results=[_success_result(duration=1.0, size=1000)])
        webhook = _FakeWebhook()
        service = _make_service(recorder=recorder, webhook_service=webhook)

        await service.process_job(jm, "j-wh", TENANT_U1)

        # Webhook delivery is fire-and-forget — let the background task run.
        for _ in range(20):
            await asyncio.sleep(0)
            if webhook.notifications:
                break

        assert len(webhook.notifications) == 1
        notif = webhook.notifications[0]
        assert notif["url"] == "http://example.com/hook"
        assert notif["job_id"] == "j-wh"
        assert notif["status"] == "completed"
        assert notif["payload"]["result"]["duration_seconds"] == 1.0
        assert notif["payload"]["result"]["file_size_bytes"] == 1000


# ── cancel_job ────────────────────────────────────────────────────


class TestCancelJob:
    async def test_cancel_processing_job(self) -> None:
        before_cancelled = _counter_value("recording_jobs_total", status="cancelled")

        job = make_job(job_id="j-cancel", status="processing")
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder(cancel_returns=True)
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)

        await service.cancel_job(jm, "j-cancel", tenant_id=TENANT_U1)

        # Observable: subprocess was killed, status moved to cancelled,
        # storage was cleaned up, and the cancelled counter ticked.
        assert recorder.cancel_calls == [job.part_id]
        assert job.status == "cancelled"
        assert storage.cleaned == ["j-cancel"]
        assert _counter_value(
            "recording_jobs_total", status="cancelled"
        ) - before_cancelled == pytest.approx(1.0)

    async def test_cancel_pending_job_skips_subprocess_kill(self) -> None:
        job = make_job(job_id="j-pending", status="pending")
        jm, _fake = _make_jm(job)
        recorder = _FakeRecorder()
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)

        await service.cancel_job(jm, "j-pending", tenant_id=TENANT_U1)

        # Observable: no subprocess kill (pending has no live process), but
        # the row still moved to cancelled and storage was cleaned.
        assert recorder.cancel_calls == []
        assert job.status == "cancelled"
        assert storage.cleaned == ["j-pending"]

    async def test_cancel_nonexistent_job_returns(self) -> None:
        jm, fake = _make_jm(None)
        service = _make_service()

        await service.cancel_job(jm, "no-such-job", tenant_id=TENANT_U1)

        # Observable: nothing was mutated.
        assert all(call[0] != "mark_cancelled" for call in fake.calls)

    async def test_cancel_publishes_event(self) -> None:
        job = make_job(job_id="j-evt", status="pending", tenant_id=TENANT_U1)
        jm, _fake = _make_jm(job)
        events = _FakeEventPublisher()
        service = _make_service(event_publisher=events)

        await service.cancel_job(jm, "j-evt", tenant_id=TENANT_U1)

        assert events.events == [(TENANT_U1, "j-evt", "cancelled")]

    async def test_cancel_terminal_job_is_noop(self) -> None:
        """Already-completed jobs must not be re-cancelled (state-conflict guard)."""
        job = make_job(job_id="j-done", status="completed")
        jm, _fake = _make_jm(job)
        events = _FakeEventPublisher()
        recorder = _FakeRecorder()
        service = _make_service(recorder=recorder, event_publisher=events)

        await service.cancel_job(jm, "j-done", tenant_id=TENANT_U1)

        # Observable: status preserved, no events, no recorder calls.
        assert job.status == "completed"
        assert events.events == []
        assert recorder.cancel_calls == []


# ── _is_retryable ────────────────────────────────────────────────


class TestIsRetryable:
    def test_retryable_pattern_matches(self) -> None:
        service = _make_service()
        assert service._is_retryable("Recording timed out after 90s") is True
        assert service._is_retryable("browser disconnected unexpectedly") is True
        assert service._is_retryable("Target closed") is True

    def test_non_retryable_returns_false(self) -> None:
        service = _make_service()
        assert service._is_retryable("Permission denied") is False
        assert service._is_retryable("Script file not found") is False

    def test_case_insensitive(self) -> None:
        service = _make_service()
        assert service._is_retryable("TIMED OUT") is True
        assert service._is_retryable("Browser Disconnected") is True


# ── screen capture jobs ─────────────────────────────────────────────


class _FakeUrlRecorder:
    """Minimal url-recorder fake exposing both screen-capture entry points."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []
        self.clip_calls: list[dict[str, Any]] = []

    async def capture_screens(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self._result

    async def capture_screen_clips(self, **kwargs: Any) -> dict[str, Any]:
        self.clip_calls.append(kwargs)
        return self._result


class TestScreensJob:
    async def test_create_screens_job_stores_steps_and_type(self) -> None:
        service = _make_service()
        jm, fake = _make_jm(make_job(job_id="s-1"))

        steps = [{"selector": ".nav", "capture_id": "home"}]
        result = await service.create_screens_job(
            jm,
            tenant_id=TENANT_U1,
            part_id="demo",
            url="https://x.com/demo",
            steps=steps,
        )

        assert result == "s-1"
        create_call = next(c for c in fake.calls if c[0] == "create_job")
        assert create_call[1]["part_type"] == "screens"
        assert create_call[1]["script_source"] == "https://x.com/demo"
        assert create_call[1]["recording_config"]["screen_steps"] == steps

    async def test_process_screens_job_success_stores_zip(self, tmp_path) -> None:
        zip_path = _write_screens_zip(tmp_path / "screens.zip")
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        url_rec = _FakeUrlRecorder(_screens_result(zip_path))
        service._url_recorder = cast(Any, url_rec)

        job = make_job(job_id="s-2")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        jm, _fake = _make_jm(job)

        await service.process_screens_job(jm, "s-2", TENANT_U1)

        # capture_screens was driven with the stored url + steps (steps popped
        # out of recording_config).
        assert len(url_rec.calls) == 1
        assert url_rec.calls[0]["url"] == "https://x.com/demo"
        assert url_rec.calls[0]["steps"] == [{"selector": ".nav", "capture_id": "home"}]
        assert "screen_steps" not in (url_rec.calls[0]["recording_config"] or {})
        assert job.status == "completed"

    async def test_process_screens_job_clip_mode_uses_clip_recorder(self, tmp_path) -> None:
        zip_path = _write_screens_zip(tmp_path / "screens.zip")
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        url_rec = _FakeUrlRecorder(_screens_result(zip_path))
        service._url_recorder = cast(Any, url_rec)

        job = make_job(job_id="s-3")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {
            "screen_steps": [{"selector": ".nav", "capture_id": "home"}],
            "record_clips": True,
            "clip_dwell_ms": 4000,
        }
        jm, _fake = _make_jm(job)

        await service.process_screens_job(jm, "s-3", TENANT_U1)

        # Clip mode routes to capture_screen_clips, not capture_screens; the
        # record_clips flag is popped, clip_dwell_ms rides along in config.
        assert len(url_rec.clip_calls) == 1
        assert len(url_rec.calls) == 0
        assert "record_clips" not in (url_rec.clip_calls[0]["recording_config"] or {})
        assert url_rec.clip_calls[0]["recording_config"]["clip_dwell_ms"] == 4000
        assert job.status == "completed"

    async def test_process_screens_job_cleans_staging_dir_on_success(self, tmp_path) -> None:
        # The recorder returns its per-job staging dir; the service must rmtree it
        # AFTER the zip has been stored, so /tmp never accumulates.
        staging = tmp_path / "vprec_s4"
        staging.mkdir()
        zip_path = _write_screens_zip(staging / "screens.zip")
        result = _screens_result(zip_path, size=9)
        result["staging_dir"] = str(staging)

        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(Any, _FakeUrlRecorder(result))

        job = make_job(job_id="s-4")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        jm, _fake = _make_jm(job)

        await service.process_screens_job(jm, "s-4", TENANT_U1)

        assert job.status == "completed"
        assert storage.stored, "the zip should have been stored before cleanup"
        assert not staging.exists(), "staging dir must be removed after storing"

    async def test_process_screens_job_cleans_staging_dir_on_failure(self, tmp_path) -> None:
        # Even when the recorder reports failure, its staging dir is cleaned.
        staging = tmp_path / "vprec_s5"
        staging.mkdir()
        (staging / "partial.webm").write_bytes(b"x")
        result = {"success": False, "error": "boom", "staging_dir": str(staging)}

        service = _make_service()
        service._url_recorder = cast(Any, _FakeUrlRecorder(result))

        job = make_job(job_id="s-5")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        jm, _fake = _make_jm(job)

        with pytest.raises(RuntimeError):
            await service.process_screens_job(jm, "s-5", TENANT_U1)

        assert not staging.exists(), "staging dir must be removed on the failure path too"


class _SignInRaisingUrlRecorder:
    """Url-recorder fake whose sign-in did not work — it RAISES, like the real one."""

    async def record_url(self, **kwargs: Any) -> dict[str, Any]:
        raise SiteLoginFailedError(_SIGN_IN_ERROR)

    async def capture_screens(self, **kwargs: Any) -> dict[str, Any]:
        raise SiteLoginFailedError(_SIGN_IN_ERROR)

    async def capture_screen_clips(self, **kwargs: Any) -> dict[str, Any]:
        raise SiteLoginFailedError(_SIGN_IN_ERROR)


_SIGN_IN_ERROR = (
    "site_login_failed: the page still asks for a password after signing in "
    "— the saved cookie session was not accepted"
)


class TestSignInFailureIsFinalized:
    """A sign-in that did not work must fail the job like ANY other recording failure.

    Finding: ``SiteLoginFailedError`` escaped before ``_finalize_result``, so the
    only thing that ran was the route's ``_mark_job_failed_safe`` — no
    ``record_usage(is_error=True)``, no failed metric, no SSE event and no
    webhook. The URL path had the same hole.
    """

    @pytest.mark.parametrize("path", ["screens", "clips", "url"])
    async def test_a_sign_in_failure_takes_the_normal_failure_path(self, path: str) -> None:
        events = _FakeEventPublisher()
        webhook = _FakeWebhook()
        service = _make_service(event_publisher=events, webhook_service=webhook)
        service._url_recorder = cast(Any, _SignInRaisingUrlRecorder())

        job = make_job(job_id="sf-1", webhook_url="http://example.com/hook")
        job.part_type = "url" if path == "url" else "screens"
        job.script_source = "https://x.com/demo"
        rec_config: dict[str, Any] = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        if path == "clips":
            rec_config["record_clips"] = True
        job.recording_config = rec_config
        jm, fake = _make_jm(job)

        before_failed = _counter_value("recording_jobs_total", status="failed")
        with pytest.raises(SiteLoginFailedError):
            if path == "url":
                await service.process_url_job(jm, "sf-1", TENANT_U1)
            else:
                await service.process_screens_job(jm, "sf-1", TENANT_U1)

        # The job is failed in the DB with the code the orchestrator matches.
        assert job.status == "failed"
        assert job.error_message is not None
        assert "site_login_failed" in job.error_message
        # ... and the failure was billed as an ERROR, counted, published and sent.
        assert any(record.get("is_error") for record in fake.usage_records), (
            "record_usage(is_error=True) must run — otherwise only the route's "
            "_mark_job_failed_safe ran and the failure is invisible to usage"
        )
        assert _counter_value("recording_jobs_total", status="failed") == before_failed + 1
        assert [event[2] for event in events.events] == ["processing", "failed"]
        for _ in range(20):
            await asyncio.sleep(0)
            if webhook.notifications:
                break
        assert [notif["status"] for notif in webhook.notifications] == ["failed"]
        assert webhook.notifications[0]["job_id"] == "sf-1"


# ── Billing safety: never meter an artifact that does not exist ────


class TestNoBillingWithoutStoredArtifact:
    """A recorder can report ``success: True`` and still hand back nothing.

    Before the fix ``_handle_success`` tested for exactly that -- ``video_path``
    absent, or a path that is not on disk -- and then ran ``mark_completed`` and
    ``record_usage(recordings_completed=1, ...)`` unconditionally anyway. The
    tenant was billed a recording, plus its duration and bytes, for a row whose
    ``video_path`` is ``NULL``. Asserting the absence of the billing write is the
    point of these tests; asserting only the status would miss half the bug.
    """

    @staticmethod
    def _assert_not_billed(fake: _FakeJobManager) -> None:
        completed = [u for u in fake.usage_records if u.get("recordings_completed")]
        assert not completed, f"job was metered as a completed recording: {completed}"
        assert not any(c[0] == "mark_completed" for c in fake.calls), (
            "job was marked completed without a retrievable artifact"
        )
        assert fake.usage_records == [{"tenant_id": TENANT_U1, "is_error": True}], (
            f"expected exactly one error-usage record, got {fake.usage_records}"
        )

    async def test_absent_video_path_fails_job_and_is_not_billed(self) -> None:
        result = _success_result(duration=5.0, size=50000)
        result["video_path"] = None
        recorder = _FakeRecorder(results=[result])
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="nb-1", max_retries=0)
        jm, fake = _make_jm(job)

        with pytest.raises(ArtifactNotStoredError, match="produced no retrievable video"):
            await service.process_job(jm, "nb-1", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored, "nothing should have been uploaded"
        self._assert_not_billed(fake)

    async def test_nonexistent_video_path_fails_job_and_is_not_billed(self, tmp_path) -> None:
        missing = tmp_path / "never-written.webm"
        recorder = _FakeRecorder(
            results=[_success_result(duration=5.0, size=50000, video_path=str(missing))]
        )
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="nb-2", max_retries=0)
        jm, fake = _make_jm(job)

        with pytest.raises(ArtifactNotStoredError, match="produced no retrievable video"):
            await service.process_job(jm, "nb-2", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored
        self._assert_not_billed(fake)

    async def test_screens_job_without_artifact_fails_and_is_not_billed(self) -> None:
        """The content/url/screens paths reach ``_handle_success`` via
        ``_finalize_result``, which has no ``except Exception`` of its own -- so
        without explicit routing the raise would escape with the job stranded in
        ``processing`` rather than failed."""
        result = _success_result(duration=0.0, size=4242)
        result["video_path"] = None
        service = _make_service()
        service._url_recorder = cast(Any, _FakeUrlRecorder(result))

        job = make_job(job_id="nb-3")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        jm, fake = _make_jm(job)

        with pytest.raises(ArtifactNotStoredError):
            await service.process_screens_job(jm, "nb-3", TENANT_U1)

        assert job.status == "failed"
        self._assert_not_billed(fake)


# ── Slot-wait must NOT be billed as execution time ──────────────────


@dataclass
class _SlowConcurrency:
    """Concurrency fake whose ``acquire`` really blocks for ``wait_seconds``.

    The wait is a REAL ``asyncio.sleep``, not a monkeypatched clock, and it is
    deliberately much larger than the execution it wraps. A fake that admits
    instantly makes the separability assertion structurally vacuous: with a
    ~0s wait, ``execute`` and ``wait + execute`` are the same number and the
    test cannot fail no matter where the clock starts.
    """

    wait_seconds: float = 0.0
    active_count: int = 0
    acquire_count: int = 0
    release_count: int = 0

    async def acquire(self, owner: str | None = None) -> None:
        self.acquire_count += 1
        await asyncio.sleep(self.wait_seconds)
        self.active_count += 1

    async def release(self, owner: str | None = None) -> None:
        self.release_count += 1
        self.active_count = max(0, self.active_count - 1)


@dataclass
class _SlowRecorder:
    """``record_part`` fake that really spends ``execute_seconds`` recording.

    ``results`` is consumed one entry per call, with the last entry repeating,
    so a retry scenario is scripted as data rather than by reassigning the
    method (which is an implicit shadow ty rejects, and hides the arity).
    """

    execute_seconds: float = 0.0
    results: list[dict[str, Any]] = field(default_factory=lambda: [_success_result()])
    call_count: int = 0

    async def record_part(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self.execute_seconds)
        self.call_count += 1
        return self.results[min(self.call_count, len(self.results)) - 1]

    async def cancel_recording(self, part_id: str) -> bool:
        return True


class _SlowUrlRecorder:
    """Url-recorder fake that really spends ``execute_seconds`` capturing."""

    def __init__(self, result: dict[str, Any], execute_seconds: float) -> None:
        self._result = result
        self._execute = execute_seconds

    async def record_url(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self._execute)
        return self._result

    async def capture_screens(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self._execute)
        return self._result

    async def capture_screen_clips(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self._execute)
        return self._result


class _SlowContentRecorder:
    def __init__(self, result: dict[str, Any], execute_seconds: float) -> None:
        self._result = result
        self._execute = execute_seconds

    async def record_content(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self._execute)
        return self._result


class _SlowAnimationRecorder:
    def __init__(self, result: dict[str, Any], execute_seconds: float) -> None:
        self._result = result
        self._execute = execute_seconds

    async def record_animation(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self._execute)
        return self._result


# A job that waits N for a slot and executes M must report M, not N+M. The wait
# is ~6x the execution so the contaminated value (N+M) and the correct one (M)
# are separated far beyond any scheduling jitter: a clock started before
# ``acquire()`` lands near 0.35s, one started after lands near 0.05s.
_SLOT_WAIT_S = 0.30
_EXECUTE_S = 0.05


def _completed_kwargs(fake: Any) -> dict[str, Any]:
    """The kwargs the service passed to ``mark_completed``."""
    calls = [kw for name, kw in fake.calls if name == "mark_completed"]
    assert len(calls) == 1, f"expected exactly one mark_completed, got {len(calls)}"
    return calls[0]


def _assert_separated(fake: Any, *, attempts: int = 1) -> None:
    """Assert execution and queue-wait were stored as two distinct numbers."""
    kw = _completed_kwargs(fake)
    execute = kw["recording_time_seconds"]
    wait = kw["queue_wait_seconds"]

    # Execution: at least the recorder's own sleep, and strictly BELOW the
    # slot wait. The second half is the whole test -- a clock started before
    # `acquire()` reports >= wait + execute and fails here.
    assert execute >= _EXECUTE_S * attempts
    assert execute < _SLOT_WAIT_S, (
        f"recording_time_seconds={execute:.3f}s is at or above the {_SLOT_WAIT_S}s "
        f"slot wait -- queue-wait is still being billed as execution time"
    )
    # Queue-wait: recorded, and recorded as the wait it really was.
    assert wait >= _SLOT_WAIT_S * attempts
    # And the two are genuinely separate numbers, not the same span twice.
    assert wait > execute


class TestQueueWaitIsNotExecutionTime:
    """``recording_time_seconds`` must mean execution, with the wait kept apart.

    Regression cover for the metric that reported 438% and 474% of the
    physically possible execution in a single 10-minute bucket, because the
    clock at all five entry points started before the fleet semaphore.
    """

    async def test_process_job_reports_execution_not_queue_wait(self) -> None:
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        service._recorder = cast(Any, _SlowRecorder(execute_seconds=_EXECUTE_S))
        jm, fake = _make_jm(make_job(job_id="qw-1"))

        await service.process_job(jm, "qw-1", TENANT_U1)

        assert conc.acquire_count == 1
        _assert_separated(fake)

    async def test_retries_accumulate_each_attempts_wait_and_execution(self) -> None:
        """Two attempts means two slot waits -- and neither belongs in execution.

        The retrying path passes ONE timing across attempts, so this also pins
        that the accumulation stays in the right buckets rather than collapsing
        back into a single wall span.
        """
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        recorder = _SlowRecorder(
            execute_seconds=_EXECUTE_S,
            results=[
                {"success": False, "error": "browser disconnected"},
                _success_result(),
            ],
        )
        service._recorder = cast(Any, recorder)
        jm, fake = _make_jm(make_job(job_id="qw-2"))

        await service.process_job(jm, "qw-2", TENANT_U1)

        assert recorder.call_count == 2
        assert conc.acquire_count == 2
        _assert_separated(fake, attempts=2)

    async def test_content_job_reports_execution_not_queue_wait(self) -> None:
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        service._content_recorder = cast(Any, _SlowContentRecorder(_success_result(), _EXECUTE_S))
        job = make_job(job_id="qw-3")
        job.script_source = "<html></html>"
        jm, fake = _make_jm(job)

        await service.process_content_job(jm, "qw-3", TENANT_U1)

        _assert_separated(fake)

    async def test_animation_job_reports_execution_not_queue_wait(self) -> None:
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        service._animation_recorder = cast(
            Any, _SlowAnimationRecorder(_success_result(), _EXECUTE_S)
        )
        job = make_job(job_id="qw-4")
        job.script_source = "<html></html>"
        jm, fake = _make_jm(job)

        await service.process_animation_job(jm, "qw-4", TENANT_U1)

        _assert_separated(fake)

    async def test_url_job_reports_execution_not_queue_wait(self) -> None:
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        service._url_recorder = cast(Any, _SlowUrlRecorder(_success_result(), _EXECUTE_S))
        job = make_job(job_id="qw-5")
        job.script_source = "https://x.com/demo"
        jm, fake = _make_jm(job)

        await service.process_url_job(jm, "qw-5", TENANT_U1)

        _assert_separated(fake)

    async def test_screens_job_reports_execution_not_queue_wait(self) -> None:
        zip_path = Path(_ARTIFACT_DIR.name) / "screens-qw.zip"
        zip_path.write_bytes(b"PK\x03\x04fake")
        conc = _SlowConcurrency(wait_seconds=_SLOT_WAIT_S)
        service = _make_service(concurrency_manager=cast(Any, conc))
        service._url_recorder = cast(
            Any,
            _SlowUrlRecorder(
                _success_result(duration=0.0, size=4242, video_path=str(zip_path)),
                _EXECUTE_S,
            ),
        )
        job = make_job(job_id="qw-6")
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        jm, fake = _make_jm(job)

        await service.process_screens_job(jm, "qw-6", TENANT_U1)

        _assert_separated(fake)


# ── Billing safety: an unprobeable file is not a zero-duration success ──


class TestNoBillingOnUnprobedRecording:
    """A recorder can report ``success: True`` with a file on disk that ffprobe
    itself cannot read (corrupt / header-less mid-flush, per
    ``FFmpegOps.probe_all``'s except branch). That branch folds the failure into
    ``ProbeResult(duration=0.0, probed=False)``, indistinguishable from a
    genuinely empty file unless the caller checks ``probed``. ``_handle_success``
    re-probes the stored artifact and must refuse to complete/bill when
    ``probed`` is False -- and must still complete/bill a genuinely 0-second
    file, where ``probed`` is True.
    """

    @staticmethod
    def _assert_not_billed(fake: _FakeJobManager) -> None:
        completed = [u for u in fake.usage_records if u.get("recordings_completed")]
        assert not completed, f"job was metered as a completed recording: {completed}"
        assert not any(c[0] == "mark_completed" for c in fake.calls), (
            "job was marked completed off an unprobed (unverifiable) artifact"
        )

    async def test_unprobed_recording_is_not_billed(self) -> None:
        # RecordingError is retryable, and ``process_job``'s ``job.max_retries or
        # self._max_retries`` falls back to the settings default (2, per
        # ``_mock_settings``) for a job created with max_retries=0 -- so this
        # attempts 3 times. Every attempt is scripted to succeed at the
        # Playwright layer but fail the (globally mocked) probe, so all three
        # are consumed and the job ends up terminally failed.
        recorder = _FakeRecorder(results=[_success_result(duration=0.0, size=1234)] * 3)
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="up-1", max_retries=0)
        jm, fake = _make_jm(job)

        failed_probe = ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=False)
        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            return_value=failed_probe,
        ):
            with pytest.raises(RecordingError, match="ffprobe could not read"):
                await service.process_job(jm, "up-1", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored, "an unprobed file must never be uploaded/stored"
        self._assert_not_billed(fake)

    async def test_genuinely_zero_second_recording_is_still_billed(self) -> None:
        """Control case: probe SUCCEEDS (probed=True) with duration 0.0 -- a
        real empty recording. This must complete and bill normally, proving the
        guard above discriminates on ``probed``, not on ``duration == 0``."""
        recorder = _FakeRecorder(results=[_success_result(duration=0.0, size=1234)])
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="up-2", max_retries=0)
        jm, fake = _make_jm(job)

        real_zero_probe = ProbeResult(
            duration=0.0, width=1920, height=1080, has_audio=False, probed=True
        )
        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            return_value=real_zero_probe,
        ):
            await service.process_job(jm, "up-2", TENANT_U1)

        assert job.status == "completed"
        assert storage.stored, "a genuinely 0s but successfully-probed file must still be stored"
        completed = [u for u in fake.usage_records if u.get("recordings_completed")]
        assert completed, "a genuinely 0s recording must still be billed"


# ── The probe veto is per-ARTIFACT-KIND, not universal ─────────────


class TestNonVideoArtifactsAreVerifiedNotProbed:
    """A screens job's ``video_path`` is a ZIP of per-screen stills/clips, not a
    video. ffprobe cannot read an archive, so an unconditional ``probe_all``
    veto in ``_handle_success`` failed EVERY screens-mode capture in production
    (``RecordingError: ... ffprobe could not read (path=.../screens.zip)``).

    The fix is NOT "swallow the probe failure" -- that would restore the very
    defect the veto exists to stop. The producer declares ``artifact_kind`` and
    the success path runs the check that belongs to that kind: ffprobe for a
    video, archive integrity for a zip. Both remain vetoes.
    """

    @staticmethod
    def _assert_not_billed(fake: _FakeJobManager) -> None:
        completed = [u for u in fake.usage_records if u.get("recordings_completed")]
        assert not completed, f"job was metered as a completed recording: {completed}"
        assert not any(c[0] == "mark_completed" for c in fake.calls), (
            "job was marked completed off an unverified artifact"
        )

    @staticmethod
    def _screens_job(job_id: str) -> RecordingJob:
        job = make_job(job_id=job_id, max_retries=0)
        job.part_type = "screens"
        job.script_source = "https://x.com/demo"
        job.recording_config = {"screen_steps": [{"selector": ".nav", "capture_id": "home"}]}
        return job

    async def test_valid_screens_zip_completes_and_is_never_probed(self, tmp_path) -> None:
        """A sound screens.zip must complete AND ffprobe must not be consulted.

        Asserting the probe is not called is what makes this test about the
        EXEMPTION rather than about a lenient mock: if the artifact kind were
        ignored and the code fell through to the video branch, the probe would
        fire and this test reds even though a permissive probe mock would have
        let it pass.
        """
        zip_path = _write_screens_zip(tmp_path / "screens.zip", ("home", "pricing"))
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(
            Any, _FakeUrlRecorder(_screens_result(zip_path, ("home", "pricing")))
        )
        job = self._screens_job("z-ok")
        jm, fake = _make_jm(job)

        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            side_effect=AssertionError("ffprobe must never be run against a zip artifact"),
        ) as probe:
            await service.process_screens_job(jm, "z-ok", TENANT_U1)

        assert probe.call_count == 0
        assert job.status == "completed"
        assert storage.stored, "a sound screens archive must be stored"
        assert [u for u in fake.usage_records if u.get("recordings_completed")], (
            "a sound screens capture must still be billed"
        )

    async def test_corrupt_screens_zip_fails_and_is_not_billed(self, tmp_path) -> None:
        """Honest failure: bytes that are not an archive at all."""
        zip_path = tmp_path / "screens.zip"
        zip_path.write_bytes(b"PK\x03\x04 this is not really a zip at all")
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(Any, _FakeUrlRecorder(_screens_result(zip_path)))
        job = self._screens_job("z-corrupt")
        jm, fake = _make_jm(job)

        with pytest.raises(RecordingError, match="unusable screens archive"):
            await service.process_screens_job(jm, "z-corrupt", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored, "a corrupt archive must never be uploaded"
        self._assert_not_billed(fake)

    async def test_empty_screens_zip_fails_and_is_not_billed(self, tmp_path) -> None:
        """Honest failure: a structurally valid zip with no entries in it."""
        zip_path = tmp_path / "screens.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(Any, _FakeUrlRecorder(_screens_result(zip_path, ())))
        job = self._screens_job("z-empty")
        jm, fake = _make_jm(job)

        with pytest.raises(RecordingError, match="contains no entries"):
            await service.process_screens_job(jm, "z-empty", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored
        self._assert_not_billed(fake)

    async def test_screens_zip_missing_a_reported_screen_fails(self, tmp_path) -> None:
        """Honest failure: the archive does not deliver the screens it claims.

        A valid, non-empty zip is not enough -- the recorder reported two
        screens and only one is in the archive, so the caller would silently
        render a promo with a missing screen.
        """
        zip_path = _write_screens_zip(tmp_path / "screens.zip", ("home",))
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(
            Any, _FakeUrlRecorder(_screens_result(zip_path, ("home", "pricing")))
        )
        job = self._screens_job("z-partial")
        jm, fake = _make_jm(job)

        with pytest.raises(RecordingError, match="missing a non-empty"):
            await service.process_screens_job(jm, "z-partial", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored
        self._assert_not_billed(fake)

    async def test_clip_mode_zip_is_exempt_too(self, tmp_path) -> None:
        """The OTHER non-video mode. ``capture_screen_clips`` returns the same
        zip shape (``<id>.mp4`` + ``<id>.png``); a one-mode fix that left clip
        mode probing a zip would leave it broken in exactly the same way.
        """
        zip_path = tmp_path / "screens.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("home.png", b"\x89PNG\r\n\x1a\n" + b"pixels" * 8)
            zf.writestr("home.mp4", b"\x00\x00\x00\x18ftypmp42" + b"moov" * 8)
        storage = _FakeStorage()
        service = _make_service(storage=storage)
        service._url_recorder = cast(Any, _FakeUrlRecorder(_screens_result(zip_path)))
        job = self._screens_job("z-clip")
        job.recording_config = {
            "screen_steps": [{"selector": ".nav", "capture_id": "home"}],
            "record_clips": True,
        }
        jm, _fake = _make_jm(job)

        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            side_effect=AssertionError("ffprobe must never be run against a zip artifact"),
        ):
            await service.process_screens_job(jm, "z-clip", TENANT_U1)

        assert job.status == "completed"
        assert storage.stored

    # ── #157 regression guard: the VIDEO branch must be unchanged ──

    async def test_video_artifact_with_failed_probe_still_fails(self) -> None:
        """The behaviour PR #157 shipped, asserted explicitly here so the zip
        exemption can never be widened into "probe failures are tolerated".

        A recorder that declares no ``artifact_kind`` is a video recorder, and an
        unprobeable video is still a failed recording -- not stored, not
        completed, not billed.
        """
        recorder = _FakeRecorder(results=[_success_result(duration=0.0, size=1234)] * 3)
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="v-bad", max_retries=0)
        jm, fake = _make_jm(job)

        failed_probe = ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=False)
        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            return_value=failed_probe,
        ):
            with pytest.raises(RecordingError, match="ffprobe could not read"):
                await service.process_job(jm, "v-bad", TENANT_U1)

        assert job.status == "failed"
        assert not storage.stored
        self._assert_not_billed(fake)

    async def test_video_artifact_that_probes_still_succeeds(self) -> None:
        """Control for the guard above: the video branch still completes and
        bills when ffprobe CAN read the file."""
        recorder = _FakeRecorder(results=[_success_result(duration=7.5, size=1234)])
        storage = _FakeStorage()
        service = _make_service(storage=storage, recorder=recorder)
        job = make_job(job_id="v-ok", max_retries=0)
        jm, fake = _make_jm(job)

        good_probe = ProbeResult(
            duration=7.5, width=1920, height=1080, has_audio=False, probed=True
        )
        with patch(
            "src.services.recording.recorder_service.FFmpegOps.probe_all",
            return_value=good_probe,
        ):
            await service.process_job(jm, "v-ok", TENANT_U1)

        assert job.status == "completed"
        assert storage.stored
        assert [u for u in fake.usage_records if u.get("recordings_completed")]


# ── QUEUE_SATURATED wire contract ───────────────────────────────────


class TestQueueSaturatedContract:
    """The stored failure text of a saturated job is a CROSS-SERVICE contract.

    Consumer: ``w2v_orchestrator/src/services/stages/shared/ken_burns.py`` matches
    the literal text ``QUEUE_SATURATED`` in a failed job's ``error`` to decide the
    recording is retryable (no capacity) rather than broken. Renaming the prefix,
    or changing how ``_acquire_slot`` builds the message, silently turns every
    saturation into a hard stage failure over there — with this repo's CI green.
    Change both sides together.
    """

    def test_prefix_literal_is_pinned(self):
        assert QUEUE_SATURATED_ERROR_PREFIX == "QUEUE_SATURATED"

    async def test_saturated_job_is_stored_with_the_literal_marker(self):
        mgr = RecordingConcurrencyManager(max_concurrent=1, queue_wait_timeout_seconds=0.05)
        await mgr.acquire("holder")
        job = make_job(job_id="j-saturated", status="pending")
        jm, fake = _make_jm(job)
        service = _make_service(concurrency_manager=mgr)

        try:
            with pytest.raises(RecordingQueueSaturatedError):
                await service._acquire_slot(jm, "j-saturated", TENANT_U1, None)
        finally:
            await mgr.release("holder")

        assert job.status == "failed"
        # Literal on purpose (not f"{QUEUE_SATURATED_ERROR_PREFIX}: "): the
        # orchestrator matches the text, not this repo's constant.
        assert job.error_message is not None
        assert job.error_message.startswith("QUEUE_SATURATED: ")
        stored = next(c for c in fake.calls if c[0] == "mark_failed")[1]["error"]
        assert stored.startswith("QUEUE_SATURATED: ")
