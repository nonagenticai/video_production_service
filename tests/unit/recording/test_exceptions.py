"""Tests for src/exceptions.py -- exception hierarchy and retryable flags."""

from __future__ import annotations

from src.exceptions import (
    BatchCancelledError,
    BatchProcessingError,
    ConfigurationError,
    DatabaseConnectionError,
    FFmpegConversionError,
    FFmpegError,
    FFmpegProbeError,
    InfrastructureError,
    InvalidPathError,
    InvalidPresetError,
    JobError,
    JobNotFoundError,
    JobStateError,
    LocalStorageError,
    MinIOStorageError,
    PlaywrightBrowserError,
    PlaywrightTimeoutError,
    RecordingError,
    RedisConnectionError,
    ScriptNotFoundError,
    StorageError,
    VideoRecorderError,
)

# ── Hierarchy ──────────────────────────────────────────────────────


class TestHierarchy:
    def test_base_is_exception(self):
        assert issubclass(VideoRecorderError, Exception)

    def test_recording_chain(self):
        assert issubclass(RecordingError, VideoRecorderError)
        assert issubclass(PlaywrightTimeoutError, RecordingError)
        assert issubclass(PlaywrightBrowserError, RecordingError)
        assert issubclass(ScriptNotFoundError, RecordingError)

    def test_ffmpeg_chain(self):
        assert issubclass(FFmpegError, VideoRecorderError)
        assert issubclass(FFmpegConversionError, FFmpegError)
        assert issubclass(FFmpegProbeError, FFmpegError)

    def test_storage_chain(self):
        assert issubclass(StorageError, VideoRecorderError)
        assert issubclass(LocalStorageError, StorageError)
        assert issubclass(MinIOStorageError, StorageError)

    def test_config_chain(self):
        assert issubclass(ConfigurationError, VideoRecorderError)
        assert issubclass(InvalidPresetError, ConfigurationError)
        assert issubclass(InvalidPathError, ConfigurationError)

    def test_job_chain(self):
        assert issubclass(JobError, VideoRecorderError)
        assert issubclass(JobNotFoundError, JobError)
        assert issubclass(JobStateError, JobError)

    def test_batch_chain(self):
        assert issubclass(BatchProcessingError, VideoRecorderError)
        assert issubclass(BatchCancelledError, BatchProcessingError)

    def test_infra_chain(self):
        assert issubclass(InfrastructureError, VideoRecorderError)
        assert issubclass(DatabaseConnectionError, InfrastructureError)
        assert issubclass(RedisConnectionError, InfrastructureError)


# ── Retryable flags ────────────────────────────────────────────────


class TestRetryableFlags:
    def test_base_not_retryable(self):
        assert VideoRecorderError("x").retryable is False

    def test_recording_retryable(self):
        assert RecordingError("x").retryable is True
        assert PlaywrightTimeoutError("x").retryable is True
        assert PlaywrightBrowserError("x").retryable is True

    def test_script_not_found_not_retryable(self):
        assert ScriptNotFoundError("x").retryable is False

    def test_ffmpeg_not_retryable(self):
        assert FFmpegError("x").retryable is False
        assert FFmpegConversionError("x").retryable is False
        assert FFmpegProbeError("x").retryable is False

    def test_storage_mixed(self):
        assert StorageError("x").retryable is False
        assert LocalStorageError("x").retryable is False
        assert MinIOStorageError("x").retryable is True

    def test_config_not_retryable(self):
        assert ConfigurationError("x").retryable is False
        assert InvalidPresetError("x").retryable is False

    def test_job_not_retryable(self):
        assert JobNotFoundError("x").retryable is False
        assert JobStateError("x").retryable is False

    def test_batch_cancelled_not_retryable(self):
        assert BatchCancelledError("x").retryable is False

    def test_infra_retryable(self):
        assert InfrastructureError("x").retryable is True
        assert DatabaseConnectionError("x").retryable is True
        assert RedisConnectionError("x").retryable is True


# ── Constructor ────────────────────────────────────────────────────


class TestConstructor:
    def test_message_and_str(self):
        exc = VideoRecorderError("boom")
        assert exc.message == "boom"
        assert str(exc) == "boom"

    def test_original_error(self):
        orig = RuntimeError("root cause")
        exc = RecordingError("wrapper", original_error=orig)
        assert exc.original_error is orig

    def test_retryable_override(self):
        exc = FFmpegError("x", retryable=True)
        assert exc.retryable is True

    def test_chaining(self):
        orig = OSError("disk full")
        exc = StorageError("store failed", original_error=orig)
        assert exc.original_error is orig
        assert exc.message == "store failed"

    def test_default_original_error_is_none(self):
        exc = VideoRecorderError("x")
        assert exc.original_error is None
