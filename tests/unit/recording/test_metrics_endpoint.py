"""Tests for src/api/metrics.py -- Prometheus metrics endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.metrics import router


def _make_client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestMetricsEndpoint:
    def test_returns_prometheus_metrics(self):
        """When prometheus_client is installed, returns metrics content."""
        client = _make_client()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # prometheus_client IS installed in dev dependencies (via pytest-cov)
        assert (
            "text/plain" in resp.headers["content-type"]
            or "openmetrics" in resp.headers["content-type"]
        )

    def test_returns_501_without_prometheus(self):
        """When prometheus_client is not available, returns 501."""
        client = _make_client()

        with patch.dict("sys.modules", {"prometheus_client": None}):
            resp = client.get("/metrics")
            assert resp.status_code == 501
            assert "not installed" in resp.text
