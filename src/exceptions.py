"""Custom exception hierarchy for video_production_service."""

from __future__ import annotations


class VideoProductionError(Exception):
    """Base exception for the video production service."""

    retryable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        original_error: Exception | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.message = message
        self.original_error = original_error
        if retryable is not None:
            self.retryable = retryable
        super().__init__(message)


# ── Recording ─────────────────────────────────────────────────────


class RecordingError(VideoProductionError):
    retryable: bool = True


class PlaywrightTimeoutError(RecordingError):
    retryable: bool = True


class PlaywrightBrowserError(RecordingError):
    retryable: bool = True


class ScriptNotFoundError(RecordingError):
    retryable: bool = False


class SiteLoginFailedError(RecordingError):
    """The recording's sign-in did not work. Not retryable: the same login fails again.

    ``str(exc)`` starts with ``site_login_failed``; the job's stored error is
    ``format_error(exc)``, i.e. ``SiteLoginFailedError: site_login_failed: …``. That
    code is what the orchestrator's ``site_capture`` matches to fail the stage with a
    "check the login" message instead of rendering a video of the sign-in page. It
    never carries a secret.
    """

    retryable: bool = False


# ── FFmpeg ────────────────────────────────────────────────────────


class FFmpegError(VideoProductionError):
    retryable: bool = False


class FFmpegConversionError(FFmpegError):
    retryable: bool = False


class FFmpegProbeError(FFmpegError):
    retryable: bool = False


# ── Storage ───────────────────────────────────────────────────────


class StorageError(VideoProductionError):
    retryable: bool = False


class LocalStorageError(StorageError):
    retryable: bool = False


class MinIOStorageError(StorageError):
    retryable: bool = True


class ArtifactNotStoredError(StorageError, RuntimeError):
    """A job's deliverable never reached durable storage.

    Raised at the point a stage is about to declare success, when the artifact
    the tenant is about to be billed for is not retrievable through
    :class:`~src.services.storage_service.StorageService`. Marking such a job
    ``completed`` bills a ``compilations_completed`` / ``recordings_completed``
    unit for an object that every read path 404s on, so the job must take the
    failure path instead.

    Deliberately a ``RuntimeError`` **as well as** a ``StorageError``: both
    orchestrators funnel stage errors through an
    ``except (OSError, RuntimeError, ValueError, TimeoutError)`` handler that
    performs ``mark_failed`` + ``record_usage(is_error=True)``. Inheriting only
    from ``VideoProductionError`` (a plain ``Exception``) would slip past those
    handlers and strand the job in ``processing`` until the reaper -- trading a
    mis-billed job for a stuck one.

    ``retryable = False``: the artifact is absent because an upload backend
    rejected it or a recorder violated its result contract. Neither is
    recovered by immediately repeating the same expensive encode/browser run.
    """

    retryable: bool = False


# ── Configuration ─────────────────────────────────────────────────


class ConfigurationError(VideoProductionError):
    retryable: bool = False


class InvalidPresetError(ConfigurationError):
    retryable: bool = False


class InvalidPathError(ConfigurationError):
    retryable: bool = False


# ── Jobs ──────────────────────────────────────────────────────────


class JobError(VideoProductionError):
    retryable: bool = False


class JobNotFoundError(JobError):
    retryable: bool = False


class JobStateError(JobError):
    retryable: bool = False


# ── Batch ─────────────────────────────────────────────────────────


class BatchProcessingError(VideoProductionError):
    retryable: bool = False


class BatchCancelledError(BatchProcessingError):
    retryable: bool = False


# ── Compilation ──────────────────────────────────────────────────


class CompilationError(VideoProductionError):
    retryable: bool = False


# ── Infrastructure ────────────────────────────────────────────────


class InfrastructureError(VideoProductionError):
    retryable: bool = True


class DatabaseConnectionError(InfrastructureError):
    retryable: bool = True


class RedisConnectionError(InfrastructureError):
    retryable: bool = True


# Backward-compat aliases for code migrated from video_recorder_service
VideoRecorderError = VideoProductionError
