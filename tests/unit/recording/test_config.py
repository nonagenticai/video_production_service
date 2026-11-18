"""Tests for src/config.py -- settings classes, validators, helpers."""

from __future__ import annotations

import pytest

from src.config import (
    APISettings,
    AuthSettings,
    DatabaseSettings,
    FFmpegSettings,
    JobSettings,
    LoggingSettings,
    RecordingSettings,
    RedisSettings,
    Settings,
    StorageSettings,
    WebhookSettings,
    clear_settings_cache,
    get_settings,
)

# ── DatabaseSettings ────────────────────────────────────────────────


class TestDatabaseSettings:
    def test_defaults(self):
        s = DatabaseSettings(url="postgresql+psycopg://test:test@localhost:5432/test")
        assert "postgresql+psycopg" in s.url
        assert s.pool_size == 10
        assert s.max_overflow == 20
        assert s.pool_timeout == 30
        assert s.pool_recycle == 3600
        assert s.echo is False

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("DATABASE_POOL_SIZE", "42")
        s = DatabaseSettings(url="postgresql+psycopg://test:test@localhost:5432/test")
        assert s.pool_size == 42


# ── AuthSettings ────────────────────────────────────────────────────


class TestAuthSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        s = AuthSettings()
        assert s.enabled is True
        assert s.ssl_verify is True
        # w2v-common 0.67.0 DELETED ``default_tenant_id``. This assertion used to
        # pin the nil-uuid sentinel it returned; asserting its absence is the same
        # test inverted — if a fallback tenant is ever reintroduced on
        # ``AuthSettings``, this goes red instead of silently re-enabling
        # substitution across every call site that reads settings.
        assert not hasattr(s, "default_tenant_id")
        assert s.cache_maxsize == 1000
        assert s.cache_ttl == 300
        assert s.introspect_timeout == 2.0

    def test_cache_env_override(self, monkeypatch):
        monkeypatch.delenv("AUTH_ENABLED", raising=False)
        monkeypatch.setenv("AUTH_CACHE_MAXSIZE", "500")
        monkeypatch.setenv("AUTH_CACHE_TTL", "60")
        monkeypatch.setenv("AUTH_INTROSPECT_TIMEOUT", "5.0")
        s = AuthSettings()
        assert s.cache_maxsize == 500
        assert s.cache_ttl == 60
        assert s.introspect_timeout == 5.0


# ── RedisSettings ───────────────────────────────────────────────────


class TestRedisSettings:
    def test_defaults(self):
        s = RedisSettings(url="redis://localhost:6379/0")
        assert "redis://" in s.url
        assert s.max_connections == 10
        assert s.decode_responses is True


# ── APISettings ───────────────────────────────────────────────────


class TestAPISettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("API_DEBUG", raising=False)
        s = APISettings()
        assert isinstance(s.cors_origins, list)
        assert s.port == 6308
        assert s.debug is False
        assert s.max_page_size == 100

    def test_max_page_size_override(self, monkeypatch):
        monkeypatch.delenv("API_DEBUG", raising=False)
        monkeypatch.setenv("API_MAX_PAGE_SIZE", "50")
        s = APISettings()
        assert s.max_page_size == 50

    def test_cors_origins_from_list(self):
        s = APISettings(cors_origins=["http://a.com", "http://b.com"])
        assert s.cors_origins == ["http://a.com", "http://b.com"]

    def test_cors_origins_single(self):
        s = APISettings(cors_origins=["http://a.com"])
        assert s.cors_origins == ["http://a.com"]

    def test_cors_origins_empty_list(self):
        s = APISettings(cors_origins=[])
        assert s.cors_origins == []


# ── StorageSettings ────────────────────────────────────────────────


class TestStorageSettings:
    def test_retention_hours_default(self):
        s = StorageSettings()
        assert s.retention_hours == 168

    def test_cleanup_interval_default(self):
        s = StorageSettings()
        assert s.cleanup_interval_seconds == 3600

    def test_stale_job_timeout_default(self):
        s = StorageSettings()
        assert s.stale_job_timeout_seconds == 600

    def test_stale_job_timeout_override(self, monkeypatch):
        monkeypatch.setenv("STORAGE_STALE_JOB_TIMEOUT_SECONDS", "300")
        s = StorageSettings()
        assert s.stale_job_timeout_seconds == 300


# ── RecordingSettings ──────────────────────────────────────────────


class TestRecordingSettings:
    def test_defaults(self):
        s = RecordingSettings()
        assert s.headless is True
        # Fleet-wide 6 / per-pod 3 are two independent decisions now: 6 is
        # reachable at the HPA's guaranteed floor of 2 replicas, and 3 is the
        # per-pod concurrency the 8Gi memory limit is known to survive.
        assert s.max_concurrent_recordings == 6
        assert s.max_concurrent_recordings_per_pod == 3
        # A bounded queue wait, strictly below the orchestrator's 600s client
        # timeout, so saturation fails loudly instead of into a client timeout.
        assert s.queue_wait_timeout_seconds == 300.0
        assert s.queue_wait_timeout_seconds < 600
        assert s.max_retries == 2
        assert s.retry_base_delay_s == 5.0
        assert "timed out" in s.retryable_error_patterns
        assert s.output_format == "mp4"
        assert s.default_quality_preset == "standard"
        assert s.error_output_max_length == 500
        assert s.cancel_wait_timeout_s == 5.0

    def test_output_format_validation(self):
        s = RecordingSettings(output_format="webm")
        assert s.output_format == "webm"
        s2 = RecordingSettings(output_format="MP4")
        assert s2.output_format == "mp4"

    def test_output_format_invalid(self):
        with pytest.raises(ValueError, match="Invalid output_format"):
            RecordingSettings(output_format="avi")

    def test_quality_preset_validation(self):
        s = RecordingSettings(default_quality_preset="preview")
        assert s.default_quality_preset == "preview"
        s2 = RecordingSettings(default_quality_preset="4k")
        assert s2.default_quality_preset == "4k"

    def test_quality_preset_invalid(self):
        with pytest.raises(ValueError, match="Invalid quality preset"):
            RecordingSettings(default_quality_preset="ultra")


# ── FFmpegSettings ─────────────────────────────────────────────────


class TestFFmpegSettings:
    def test_defaults(self):
        s = FFmpegSettings()
        assert s.timeout == 300
        assert s.preset == "medium"
        assert s.ffprobe_timeout == 10
        assert s.default_crf == 23
        assert s.error_truncate_length == 500

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FFMPEG_DEFAULT_CRF", "18")
        monkeypatch.setenv("FFMPEG_ERROR_TRUNCATE_LENGTH", "200")
        s = FFmpegSettings()
        assert s.default_crf == 18
        assert s.error_truncate_length == 200


# ── LoggingSettings / validate_level ───────────────────────────────


class TestLoggingSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        s = LoggingSettings()
        assert s.level == "INFO"

    def test_validate_level_valid(self):
        assert LoggingSettings.validate_level("debug") == "DEBUG"
        assert LoggingSettings.validate_level("warning") == "WARNING"

    def test_validate_level_invalid(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingSettings.validate_level("VERBOSE")


# ── Settings (top-level) ───────────────────────────────────────────


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        s = Settings()
        assert s.app_name == "Video Production Service"
        assert s.environment == "development"

    def test_get_database_url(self):
        s = Settings()
        assert s.get_database_url() == s.database.url

    def test_get_redis_url(self):
        s = Settings()
        assert s.get_redis_url() == s.redis.url

    def test_is_production(self):
        s = Settings(environment="production")
        assert s.is_production() is True
        assert s.is_development() is False

    def test_is_development(self):
        s = Settings(environment="development")
        assert s.is_development() is True
        assert s.is_production() is False


# ── get_settings / clear_settings_cache ────────────────────────────


# ── JobSettings ───────────────────────────────────────────────────


class TestJobSettings:
    def test_defaults(self):
        s = JobSettings()
        assert s.max_error_length == 4000
        assert s.log_error_truncate_length == 120
        assert s.expired_batch_limit == 100

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("JOB_MAX_ERROR_LENGTH", "2000")
        monkeypatch.setenv("JOB_LOG_ERROR_TRUNCATE_LENGTH", "80")
        monkeypatch.setenv("JOB_EXPIRED_BATCH_LIMIT", "50")
        s = JobSettings()
        assert s.max_error_length == 2000
        assert s.log_error_truncate_length == 80
        assert s.expired_batch_limit == 50


# ── WebhookSettings ──────────────────────────────────────────────


class TestWebhookSettings:
    def test_defaults(self):
        s = WebhookSettings()
        assert s.max_retries == 3
        assert s.backoff_base == 2.0
        assert s.timeout == 10.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_MAX_RETRIES", "5")
        monkeypatch.setenv("WEBHOOK_BACKOFF_BASE", "3.0")
        monkeypatch.setenv("WEBHOOK_TIMEOUT", "15.0")
        s = WebhookSettings()
        assert s.max_retries == 5
        assert s.backoff_base == 3.0
        assert s.timeout == 15.0


# ── Settings (top-level) includes sub-configs ─────────────────────


class TestSettingsSubConfigs:
    def test_has_job_settings(self):
        s = Settings()
        assert hasattr(s, "job")
        assert s.job.max_error_length == 4000

    def test_has_webhook_settings(self):
        s = Settings()
        assert hasattr(s, "webhook")
        assert s.webhook.max_retries == 3


# ── get_settings / clear_settings_cache ────────────────────────────


class TestGetSettings:
    def test_caching(self):
        clear_settings_cache()
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_clear_cache(self):
        clear_settings_cache()
        get_settings()
        clear_settings_cache()
        b = get_settings()
        # After clearing, a new instance is created (may or may not be same object
        # depending on env, but the function should not raise).
        assert isinstance(b, Settings)
