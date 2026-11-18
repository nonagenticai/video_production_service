"""Unit tests for configuration module."""

from __future__ import annotations

import pytest

from src.config import (
    APISettings,
    AuthSettings,
    BackgroundMusicSettings,
    CleanupSettings,
    CompilationSettings,
    DatabaseSettings,
    FFmpegSettings,
    LoggingSettings,
    OTelSettings,
    RateLimitSettings,
    RedisSettings,
    Settings,
    StorageSettings,
    ThemeSettings,
    WatermarkSettings,
    WebhookSettings,
    clear_settings_cache,
    get_settings,
)


class TestDatabaseSettings:
    def test_default_values(self):
        s = DatabaseSettings(url="postgresql+psycopg://test:test@localhost:5432/test")
        assert s.pool_size == 10
        assert s.max_overflow == 20
        assert s.pool_timeout == 30
        assert s.pool_recycle == 3600
        assert s.echo is False
        assert "postgresql" in s.url

    def test_url_required(self, monkeypatch):
        """DATABASE_URL is required env (no localhost fallback)."""
        from pydantic import ValidationError

        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValidationError):
            DatabaseSettings()  # ty: ignore[missing-argument]  # why: deliberately omits required url to verify ValidationError


class TestAuthSettings:
    def test_default_values(self):
        s = AuthSettings()
        # Note: conftest sets AUTH_ENABLED=false, so enabled may be False in test context
        assert isinstance(s.enabled, bool)
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


class TestRedisSettings:
    def test_default_values(self):
        s = RedisSettings(url="redis://localhost:6379/0")
        assert "redis://" in s.url
        assert s.max_connections == 10
        assert s.socket_timeout == 5
        assert s.socket_connect_timeout == 5
        assert s.decode_responses is True


class TestAPISettings:
    def test_default_values(self):
        s = APISettings()
        assert s.host == "0.0.0.0"
        assert s.port == 6308
        # conftest sets API_DEBUG=true, so debug may be True in test context
        assert isinstance(s.debug, bool)
        assert s.reload is False
        assert s.cors_allow_credentials is True

    def test_cors_origins_from_list(self):
        s = APISettings(cors_origins=["a", "b", "c"])
        assert s.cors_origins == ["a", "b", "c"]

    def test_cors_origins_empty_list(self):
        s = APISettings(cors_origins=[])
        assert s.cors_origins == []

    def test_cors_origins_single(self):
        s = APISettings(cors_origins=["http://localhost"])
        assert s.cors_origins == ["http://localhost"]


class TestStorageSettings:
    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("STORAGE_LOCAL_PATH", raising=False)
        s = StorageSettings()
        assert s.backend == "local"
        assert s.local_path == "/app/data"
        assert s.minio_secure is False


class TestFFmpegSettings:
    def test_default_values(self):
        s = FFmpegSettings()
        assert s.timeout == 300
        assert s.preset == "medium"
        assert s.ffprobe_timeout == 10


class TestCompilationSettings:
    def test_default_values(self):
        s = CompilationSettings()
        assert s.output_filename == "demo.mp4"
        assert s.crf == 20
        assert s.target_width == 1920
        assert s.target_height == 1080
        assert s.target_fps == 30
        assert s.transition_type == "smart"
        assert s.crossfade_duration == 0.5
        assert s.cleanup_intermediates is True


class TestThemeSettings:
    def test_default_values(self):
        s = ThemeSettings()
        assert s.primary_color.startswith("#")
        assert s.background_color.startswith("#")
        assert s.text_color.startswith("#")


class TestBackgroundMusicSettings:
    def test_default_values(self):
        s = BackgroundMusicSettings()
        assert s.path == ""
        assert s.volume == 0.1
        assert s.fade_in_s == 2.0
        assert s.fade_out_s == 3.0


class TestCleanupSettings:
    def test_default_values(self):
        s = CleanupSettings()
        assert s.enabled is True
        assert s.interval_seconds == 3600
        assert s.completed_ttl_days == 30
        assert s.failed_ttl_days == 7


class TestWebhookSettings:
    def test_default_values(self):
        s = WebhookSettings()
        assert s.max_retries == 3
        assert s.backoff_base == 2.0
        assert s.timeout == 10.0
        assert s.enabled is True


class TestWatermarkSettings:
    def test_default_values(self):
        s = WatermarkSettings()
        assert s.image_path == ""
        assert s.position == "bottom-right"
        assert s.opacity == 0.7
        assert s.scale == 0.15


class TestRateLimitSettings:
    def test_default_values(self):
        s = RateLimitSettings()
        assert s.enabled is True
        assert s.max_concurrent_jobs == 3
        assert s.daily_quota == 100
        assert s.requests_per_minute == 30


class TestOTelSettings:
    def test_default_values(self):
        s = OTelSettings()
        assert s.enabled is False
        assert s.service_name == "video-production-service"
        # OTel is opt-in; endpoint blank by default (was hardcoded localhost:4317).
        assert s.otlp_endpoint == ""


class TestLoggingSettings:
    def test_default_values(self):
        s = LoggingSettings()
        # conftest sets LOG_LEVEL=WARNING via os.environ.setdefault
        assert s.level in ("INFO", "WARNING")
        assert s.json_logs is False

    def test_validate_level_uppercase(self):
        result = LoggingSettings.validate_level("debug")
        assert result == "DEBUG"

    def test_validate_level_invalid(self):
        import pytest

        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingSettings.validate_level("TRACE")


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.app_name == "Video Production Service"
        assert s.app_version == "0.1.0"
        # conftest sets ENVIRONMENT=test via os.environ.setdefault
        assert s.environment in ("development", "test")

    def test_sub_configs_created(self):
        s = Settings()
        assert isinstance(s.database, DatabaseSettings)
        assert isinstance(s.auth, AuthSettings)
        assert isinstance(s.redis, RedisSettings)
        assert isinstance(s.api, APISettings)
        assert isinstance(s.storage, StorageSettings)
        assert isinstance(s.ffmpeg, FFmpegSettings)
        assert isinstance(s.compilation, CompilationSettings)
        assert isinstance(s.theme, ThemeSettings)
        assert isinstance(s.background_music, BackgroundMusicSettings)
        assert isinstance(s.logging, LoggingSettings)
        assert isinstance(s.cleanup, CleanupSettings)
        assert isinstance(s.webhook, WebhookSettings)
        assert isinstance(s.watermark, WatermarkSettings)
        assert isinstance(s.rate_limit, RateLimitSettings)
        assert isinstance(s.otel, OTelSettings)

    def test_get_database_url(self):
        s = Settings()
        assert s.get_database_url() == s.database.url

    def test_get_redis_url(self):
        s = Settings()
        assert s.get_redis_url() == s.redis.url

    def test_is_production(self):
        s = Settings()
        assert s.is_production() is False

    def test_is_development(self):
        # Note: conftest sets ENVIRONMENT=test, so default may not be "development"
        s = Settings(environment="development")
        assert s.is_development() is True

    def test_is_production_true(self):
        s = Settings(environment="production")
        assert s.is_production() is True
        assert s.is_development() is False


class TestGetSettings:
    def test_get_settings_returns_settings(self):
        clear_settings_cache()
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_cached(self):
        clear_settings_cache()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_clear_settings_cache(self):
        clear_settings_cache()
        get_settings()
        clear_settings_cache()
        s2 = get_settings()
        # After clearing cache, a new instance is created
        assert isinstance(s2, Settings)
