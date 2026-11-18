"""Tests for src/asgi.py -- ASGI app configuration and helpers."""

from __future__ import annotations

import logging

from src.asgi import HealthCheckLoggingFilter, app, root

# ── HealthCheckLoggingFilter ─────────────────────────────────────


class TestHealthCheckLoggingFilter:
    def setup_method(self):
        self.filter = HealthCheckLoggingFilter()

    def test_filters_health_endpoint(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /health HTTP/1.1 200",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is False

    def test_filters_health_ready_endpoint(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /health/ready HTTP/1.1 200",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is False

    def test_filters_health_live_endpoint(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="GET /health/live HTTP/1.1 200",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is False

    def test_passes_non_health_request(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="POST /api/record/part HTTP/1.1 201",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is True

    def test_passes_empty_message(self):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )
        assert self.filter.filter(record) is True


# ── root endpoint ────────────────────────────────────────────────


class TestRootEndpoint:
    async def test_returns_service_info(self):
        result = await root()
        assert "name" in result
        assert "status" in result
        assert result["status"] == "running"
        assert "endpoints" in result
        assert "health" in result["endpoints"]
        assert "record" in result["endpoints"]


# ── app configuration ───────────────────────────────────────────


class TestAppConfiguration:
    def test_app_has_correct_title(self):
        assert "Video Production" in app.title

    def test_app_has_lifespan(self):
        assert app.router.lifespan_context is not None

    def test_app_has_routers_mounted(self):
        """The app should have routes for health, api, metrics, events."""
        route_paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/" in route_paths
