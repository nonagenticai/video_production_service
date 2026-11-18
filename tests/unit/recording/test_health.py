"""Tests for src/api/health.py -- health check endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.health import router
from src.config import get_settings
from src.services.recording.concurrency import ExecutorStallStatus, RecordingConcurrencyManager

# ── Helpers ────────────────────────────────────────────────────────


def _make_app(storage_service=None, concurrency_manager=None):
    app = FastAPI()
    app.include_router(router)
    if storage_service is not None:
        app.state.storage_service = storage_service
    if concurrency_manager is not None:
        app.state.concurrency_manager = concurrency_manager
    return app


class _FakeEngine:
    """Fake async engine whose connect() yields a mock connection."""

    def __init__(self, healthy: bool = True):
        self._healthy = healthy

    @asynccontextmanager
    async def connect(self):
        if not self._healthy:
            raise ConnectionError("db down")
        conn = AsyncMock()
        conn.execute = AsyncMock()
        yield conn


# ── /health ───────────────────────────────────────────────────────


class TestHealthCheck:
    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    def test_db_unhealthy(self, mock_engine, mock_redis):
        mock_engine.side_effect = ConnectionError("db down")

        redis_client = AsyncMock()
        redis_client.ping = AsyncMock()
        mock_redis.return_value = redis_client

        storage = AsyncMock()
        storage.health_check.return_value = True
        app = _make_app(storage_service=storage)
        client = TestClient(app)
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "unhealthy"

    @patch("src.api.health.get_redis_client")
    @patch("src.api.health.get_engine")
    def test_storage_unhealthy(self, mock_engine, mock_redis):
        mock_engine.return_value = _FakeEngine(healthy=True)

        redis_client = AsyncMock()
        redis_client.ping = AsyncMock()
        mock_redis.return_value = redis_client

        storage = AsyncMock()
        storage.health_check.return_value = False
        app = _make_app(storage_service=storage)
        client = TestClient(app)
        resp = client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"


# ── /health/live ──────────────────────────────────────────────────


class TestHealthLive:
    def test_always_ok(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stalled_executor_fails_liveness(self):
        """A wedged recording executor must FAIL liveness so k8s restarts the pod.

        Regression for the 2026-09-04 outage: three leaked fleet slots left every
        subsequent job stuck in ``queued`` for 9 hours while ``/health/ready``
        reported healthy (it only checked the database), so the kubelet never
        restarted the pod and nothing alerted.
        """
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        # Simulate a stalled heartbeat: three jobs waiting, and the last slot
        # admission/release is far older than the threshold.
        mgr._queued_count = 3
        mgr._active_count = 3
        mgr._queue_nonempty_since = time.monotonic() - 7200.0
        mgr._last_progress = time.monotonic() - 7200.0

        app = _make_app(concurrency_manager=mgr)
        client = TestClient(app)
        resp = client.get("/health/live")

        assert resp.status_code == 503, resp.text
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["executor"]["stalled"] is True
        assert body["executor"]["queued"] == 3
        assert body["executor"]["seconds_since_progress"] > 1800.0

    def test_idle_but_alive_executor_stays_live(self):
        """An idle executor is NOT unhealthy -- this is the anti-flap guarantee.

        The heartbeat is stale by two hours here, exactly as in the stalled case;
        the ONLY difference is that nothing is queued. A probe that fired on a
        stale heartbeat alone would restart every quiet pod in the fleet on a
        loop.
        """
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        mgr._last_progress = time.monotonic() - 7200.0
        assert mgr.queued_count == 0

        app = _make_app(concurrency_manager=mgr)
        client = TestClient(app)
        resp = client.get("/health/live")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["executor"]["stalled"] is False
        assert body["executor"]["seconds_since_progress"] is None

    def test_first_job_after_long_idle_does_not_trip(self):
        """The stall clock restarts on the queue's 0 -> non-empty transition.

        Without that restart, a service quiet overnight would fail liveness the
        instant its first job of the day was enqueued -- because the heartbeat
        would still hold last night's timestamp.
        """

        async def scenario() -> ExecutorStallStatus:
            mgr = RecordingConcurrencyManager(max_concurrent=3)
            for _ in range(3):  # exhaust the pod slots so the next one queues
                await mgr.acquire()
            # Rewind the heartbeat: the three slots were taken long ago.
            mgr._last_progress = time.monotonic() - 7200.0
            waiter = asyncio.create_task(mgr.acquire())
            await asyncio.sleep(0)  # let it park in the queue
            try:
                return mgr.stall_status(1800.0)
            finally:
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await waiter

        status = asyncio.run(scenario())
        assert status["queued"] == 1, status
        # Clock measured from the enqueue moment, not the two-hour-old heartbeat.
        assert status["stalled"] is False, status
        since = status["seconds_since_progress"]
        assert since is not None and since < 5.0, status

    def test_healthz_alias_reports_the_same_stall(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        mgr._queued_count = 1
        mgr._queue_nonempty_since = time.monotonic() - 7200.0
        mgr._last_progress = time.monotonic() - 7200.0

        client = TestClient(_make_app(concurrency_manager=mgr))
        assert client.get("/healthz").status_code == 503

    def test_check_disabled_when_threshold_is_zero(self):
        mgr = RecordingConcurrencyManager(max_concurrent=3)
        mgr._queued_count = 3
        mgr._queue_nonempty_since = time.monotonic() - 7200.0
        mgr._last_progress = time.monotonic() - 7200.0

        settings = get_settings()
        original = settings.recording.executor_stall_threshold_seconds
        settings.recording.executor_stall_threshold_seconds = 0.0
        try:
            client = TestClient(_make_app(concurrency_manager=mgr))
            resp = client.get("/health/live")
            assert resp.status_code == 200
            assert "executor" not in resp.json()
        finally:
            settings.recording.executor_stall_threshold_seconds = original


# ── /health/ready ─────────────────────────────────────────────────


class TestHealthReady:
    @patch("src.api.health.get_engine")
    def test_db_down_503(self, mock_engine):
        mock_engine.side_effect = ConnectionError("connection refused")

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        assert resp.json()["ready"] is False
        assert resp.json()["status"] == "unhealthy"
