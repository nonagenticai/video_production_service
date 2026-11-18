"""Unit tests for the ASGI application module."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_engine():
    """Create a mock engine with a properly configured begin() async context manager."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()

    @asynccontextmanager
    async def _begin():
        yield mock_conn

    mock_engine.begin = _begin
    return mock_engine, mock_conn


# ---------------------------------------------------------------------------
# HealthCheckLoggingFilter tests
# ---------------------------------------------------------------------------


class TestHealthCheckLoggingFilter:
    def test_filters_health_endpoint(self):
        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health HTTP/1.1" 200 OK',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

    def test_filters_health_ready_endpoint(self):
        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/ready HTTP/1.1" 200 OK',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

    def test_filters_health_live_endpoint(self):
        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='GET /health/live HTTP/1.1" 200 OK',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is False

    def test_passes_non_health_requests(self):
        from src.asgi import HealthCheckLoggingFilter

        f = HealthCheckLoggingFilter()
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg='POST /api/compile HTTP/1.1" 200 OK',
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is True


# ---------------------------------------------------------------------------
# Root endpoint test
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    async def test_root_returns_service_info(self):
        from src.asgi import root

        result = await root()
        assert result["name"] is not None
        assert result["status"] == "running"
        assert "endpoints" in result
        assert "version" in result


# ---------------------------------------------------------------------------
# Global exception handler test
# ---------------------------------------------------------------------------
# Lifespan tests
# ---------------------------------------------------------------------------


class TestLifespan:
    async def test_lifespan_startup_and_shutdown(self):
        """Test the startup/shutdown hook flow."""
        from src.asgi import _shutdown, _startup

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_engine, _mock_conn = _make_mock_engine()

        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)

        mock_storage = AsyncMock()
        mock_cleanup = AsyncMock()

        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=mock_redis_client),
            patch("src.asgi.get_redis_client", return_value=mock_redis_client),
            patch("src.asgi.close_redis", new_callable=AsyncMock),
            patch("src.asgi.close_engine", new_callable=AsyncMock),
            patch("src.database.models.Base"),
            patch("src.services.ffmpeg.ops.FFmpegOps.configure_from_settings"),
            patch("src.services.storage_service.StorageService") as mock_storage_cls,
            patch("src.api.ws.ConnectionManager"),
            patch("src.services.compilation.compiler_service.CompilerService"),
            patch("src.services.compilation.cleanup_service.CleanupService") as mock_cleanup_cls,
        ):
            mock_storage_cls.return_value = mock_storage
            mock_cleanup_cls.return_value = mock_cleanup
            mock_cleanup.start = AsyncMock()
            mock_cleanup.stop = AsyncMock()

            await _startup(mock_app)
            # Verify startup happened
            assert hasattr(mock_app.state, "db_engine")

            await _shutdown(mock_app)
            # Verify shutdown happened
            mock_cleanup.stop.assert_awaited_once()

    async def test_lifespan_redis_failure_continues(self):
        """If Redis fails to connect, startup should continue without it."""
        from src.asgi import _startup

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_engine, _ = _make_mock_engine()

        mock_storage = AsyncMock()
        mock_cleanup = AsyncMock()

        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch(
                "src.asgi.init_redis",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Redis down"),
            ),
            patch("src.asgi.close_redis", new_callable=AsyncMock),
            patch("src.asgi.close_engine", new_callable=AsyncMock),
            patch("src.database.models.Base"),
            patch("src.services.ffmpeg.ops.FFmpegOps.configure_from_settings"),
            patch("src.services.storage_service.StorageService") as mock_storage_cls,
            patch("src.api.ws.ConnectionManager"),
            patch("src.services.compilation.compiler_service.CompilerService"),
            patch("src.services.compilation.cleanup_service.CleanupService") as mock_cleanup_cls,
        ):
            mock_storage_cls.return_value = mock_storage
            mock_cleanup_cls.return_value = mock_cleanup
            mock_cleanup.start = AsyncMock()
            mock_cleanup.stop = AsyncMock()

            # Should not raise even though Redis fails
            await _startup(mock_app)

    async def test_lifespan_db_failure_raises(self):
        """If database fails to connect, startup should raise."""
        from src.asgi import _startup

        mock_app = MagicMock()

        with (
            patch(
                "src.asgi.get_engine",
                new_callable=AsyncMock,
                side_effect=ConnectionError("DB down"),
            ),
            pytest.raises(ConnectionError),
        ):
            await _startup(mock_app)

    async def test_lifespan_shutdown_handles_errors(self):
        """Errors during shutdown should be caught, not raised."""
        from src.asgi import _shutdown, _startup

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_engine, _ = _make_mock_engine()

        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)

        mock_storage = AsyncMock()
        mock_cleanup = AsyncMock()

        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=mock_redis_client),
            patch("src.asgi.get_redis_client", return_value=mock_redis_client),
            patch(
                "src.asgi.close_redis",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Redis close error"),
            ),
            patch(
                "src.asgi.close_engine",
                new_callable=AsyncMock,
                side_effect=RuntimeError("DB close error"),
            ),
            patch("src.database.models.Base"),
            patch("src.services.ffmpeg.ops.FFmpegOps.configure_from_settings"),
            patch("src.services.storage_service.StorageService") as mock_storage_cls,
            patch("src.api.ws.ConnectionManager"),
            patch("src.services.compilation.compiler_service.CompilerService"),
            patch("src.services.compilation.cleanup_service.CleanupService") as mock_cleanup_cls,
        ):
            mock_storage_cls.return_value = mock_storage
            mock_cleanup_cls.return_value = mock_cleanup
            mock_cleanup.start = AsyncMock()
            mock_cleanup.stop = AsyncMock(side_effect=RuntimeError("cleanup error"))

            # Should not raise even though shutdown components fail
            await _startup(mock_app)
            await _shutdown(mock_app)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_calls_uvicorn_run(self):
        import uvicorn

        with patch.object(uvicorn, "run") as mock_run:
            from src.asgi import main

            main()
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == "src.asgi:app"


# ---------------------------------------------------------------------------
# The startup "Database connected" line must never carry the DSN password
# ---------------------------------------------------------------------------
#
# REGRESSION: ``_init_database`` logged ``settings.database.url`` raw -- the
# full postgresql+psycopg://<user>:<password>@host:port/db DSN -- at INFO on the
# connect path, so the live app password landed in every pod's stdout and in
# anything shipping those logs, indefinitely.
#
# This drives the REAL ``_startup`` rather than the redactor, because the defect
# is at the CALL SITE: a test that only exercises
# ``w2v_common.observability.database_endpoint`` stays green for the entire
# duration of the leak. That helper has its own unit tests in w2v_common; this
# one exists to fail if anyone hands the raw url back to the logger here.

# Wholly synthetic and shares no prefix with any real credential -- this string
# is committed, printed by pytest on failure, and ends up in CI logs, i.e.
# exactly the places this change exists to keep secrets out of.
_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_DSN = (
    f"postgresql+psycopg://video_prod_app:{_FAKE_PASSWORD}"
    "@pgbouncer.pgbouncer:6432/video_production_service"
)


async def test_startup_db_log_does_not_leak_the_dsn_password(monkeypatch: pytest.MonkeyPatch):
    """The emitted startup log record must not contain the DSN password."""
    import io

    from src import asgi

    monkeypatch.setattr(asgi.settings.database, "url", _FAKE_DSN)

    # A handler attached directly to the module logger, NOT ``caplog``. This
    # service reconfigures logging at import/startup and re-points the root
    # handlers, so caplog's root handler can observe nothing and every assertion
    # below would pass vacuously.
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    asgi.logger.addHandler(handler)
    # ``setLevel``, NOT ``monkeypatch.setattr(logger, "level", ...)``. Earlier
    # tests in this file leave the ROOT logger at WARNING, and ``Logger`` caches
    # the resulting ``isEnabledFor(INFO) -> False`` in ``logger._cache``. Only
    # ``setLevel`` clears that cache; assigning ``.level`` directly leaves the
    # stale ``False`` in place, ``logger.info`` never runs, and the capture is
    # empty -- which is exactly what the vacuity guard below exists to catch.
    previous_level = asgi.logger.level
    asgi.logger.setLevel(logging.INFO)

    mock_app = MagicMock()
    mock_app.state = MagicMock()

    mock_engine, _ = _make_mock_engine()

    mock_redis_client = AsyncMock()
    mock_redis_client.ping = AsyncMock(return_value=True)

    mock_storage = AsyncMock()
    mock_cleanup = AsyncMock()

    try:
        with (
            patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=mock_engine),
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=mock_redis_client),
            patch("src.asgi.get_redis_client", return_value=mock_redis_client),
            patch("src.database.models.Base"),
            patch("src.services.ffmpeg.ops.FFmpegOps.configure_from_settings"),
            # Everything below the DB block either shells out to ffmpeg or opens
            # its OWN connection and burns ~60s on a retry ladder before failing
            # soft; none of it is relevant to this assertion.
            patch("src.asgi._probe_vp9_alpha", new_callable=AsyncMock),
            patch("src.asgi.ArtifactClient", return_value=MagicMock()),
            patch("src.services.storage_service.StorageService") as mock_storage_cls,
            patch("src.api.ws.ConnectionManager"),
            patch("src.services.compilation.compiler_service.CompilerService"),
            patch("src.services.compilation.cleanup_service.CleanupService") as mock_cleanup_cls,
        ):
            mock_storage_cls.return_value = mock_storage
            mock_cleanup_cls.return_value = mock_cleanup
            mock_cleanup.start = AsyncMock()
            mock_cleanup.stop = AsyncMock()

            await asgi._startup(mock_app)
    finally:
        asgi.logger.removeHandler(handler)
        asgi.logger.setLevel(previous_level)

    captured = buffer.getvalue()
    # Guard against a vacuous pass: if the line never emitted, the assertions
    # below would hold for the wrong reason.
    assert "Database connected" in captured, (
        "the 'Database connected' line never emitted; the assertions would be vacuous"
    )
    assert _FAKE_PASSWORD not in captured, "DSN password leaked into the startup log"
    assert "video_prod_app" not in captured, "DSN username leaked into the startup log"
    # Still useful, or the next person deletes the log line instead of keeping it.
    assert "pgbouncer.pgbouncer:6432/video_production_service" in captured


# ── The startup "Redis connected" line must never carry the Redis password ──
#
# REGRESSION: `_init_redis` logged `settings.redis.url` raw -- the full
# redis://[user]:<password>@host:port/db URL -- at INFO on the connect path, so
# the live Redis password landed in this pod's stdout and in anything shipping
# those logs, on every pod start, indefinitely.
#
# The test drives the REAL `_init_redis` rather than the redactor, because the
# defect was at the CALL SITE: a test that only exercises
# `w2v_common.observability.redis_endpoint` passes for the whole duration of the
# leak. That helper has its own unit tests in w2v_common; this one exists to
# fail if anyone hands the raw url back to the logger here.

# Wholly synthetic and shares no prefix with any real credential -- this string
# is committed, printed by pytest on failure, and ends up in CI logs, i.e.
# exactly the places this change exists to keep secrets out of.
_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_REDIS_URL = f"redis://vps:{_FAKE_PASSWORD}@redis.redis:6379/5"


class TestStartupRedisLogRedaction:
    async def test_init_redis_log_does_not_leak_the_redis_password(self) -> None:
        """The emitted startup log record must not contain the Redis password."""
        import io

        import src.asgi as asgi_mod

        mock_settings = MagicMock()
        mock_settings.redis.url = _FAKE_REDIS_URL

        # A handler attached directly to the module logger, NOT ``caplog``. This
        # service re-points the root handlers during startup, so caplog's root
        # handler can observe nothing and every assertion below would pass
        # vacuously.
        buffer = io.StringIO()
        handler = logging.StreamHandler(buffer)
        handler.setLevel(logging.INFO)
        asgi_mod.logger.addHandler(handler)
        # ``setLevel`` and NOT ``monkeypatch.setattr(logger, "level", ...)``:
        # ``logging.Logger`` caches ``isEnabledFor`` and only ``setLevel()``
        # clears that cache. A stale cache entry means the record never reaches
        # the handler and the negative assertions pass on EMPTY output.
        original_level = asgi_mod.logger.level
        asgi_mod.logger.setLevel(logging.INFO)

        original_settings = asgi_mod.settings
        asgi_mod.settings = mock_settings

        mock_app = MagicMock()
        mock_app.state = MagicMock()

        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock(return_value=True)

        try:
            with (
                patch(
                    "src.asgi.init_redis",
                    new_callable=AsyncMock,
                    return_value=mock_redis_client,
                ),
                patch("src.asgi.get_redis_client", return_value=mock_redis_client),
            ):
                await asgi_mod._init_redis(mock_app)
        finally:
            asgi_mod.settings = original_settings
            asgi_mod.logger.setLevel(original_level)
            asgi_mod.logger.removeHandler(handler)

        captured = buffer.getvalue()
        # Guard against a vacuous pass: if the line never emitted, the
        # assertions below would hold for the wrong reason.
        assert "Redis connected" in captured, (
            "the 'Redis connected' line never emitted; the assertions would be vacuous"
        )
        assert _FAKE_PASSWORD not in captured, "Redis password leaked into the startup log"
        assert "vps:" not in captured, "Redis username leaked into the startup log"
        # Still useful, or the next person deletes the log line instead of keeping it.
        assert "redis.redis:6379" in captured
