"""Tests for src/middleware/metrics.py -- MetricsMiddleware and metric helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.middleware.metrics import (
    MetricsMiddleware,
    record_duration_metric,
    record_file_size_metric,
    record_job_metric,
    set_active_jobs,
)

# ── MetricsMiddleware dispatch ──────────────────────────────────


class TestMetricsMiddleware:
    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("src.middleware.metrics.http_request_duration_seconds")
    @patch("src.middleware.metrics.http_requests_total")
    async def test_records_metrics_when_prometheus_available(
        self, mock_requests_total, mock_duration
    ):
        mw = MetricsMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/api/record/part"
        request.method = "POST"

        mock_response = MagicMock(status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        response = await mw.dispatch(request, call_next)
        assert response.status_code == 200
        call_next.assert_awaited_once()
        mock_requests_total.labels.assert_called_once()
        mock_duration.labels.assert_called_once()

    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", False)
    async def test_passthrough_when_prometheus_unavailable(self):
        mw = MetricsMiddleware(app=MagicMock())
        request = MagicMock()
        request.url.path = "/test"
        request.method = "GET"

        mock_response = MagicMock(status_code=200)
        call_next = AsyncMock(return_value=mock_response)

        response = await mw.dispatch(request, call_next)
        assert response is mock_response
        call_next.assert_awaited_once()


# ── record_job_metric ────────────────────────────────────────────


class TestRecordJobMetric:
    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("src.middleware.metrics.recording_jobs_total")
    def test_records_when_available(self, mock_counter):
        record_job_metric("completed")
        mock_counter.labels.assert_called_once_with(status="completed")
        mock_counter.labels.return_value.inc.assert_called_once()

    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", False)
    def test_noop_when_unavailable(self):
        # Should not raise
        record_job_metric("completed")


# ── record_duration_metric ──────────────────────────────────────


class TestRecordDurationMetric:
    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("src.middleware.metrics.recording_duration_seconds")
    def test_records_when_available(self, mock_histogram):
        record_duration_metric(10.5)
        mock_histogram.observe.assert_called_once_with(10.5)

    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", False)
    def test_noop_when_unavailable(self):
        record_duration_metric(10.5)


# ── record_file_size_metric ─────────────────────────────────────


class TestRecordFileSizeMetric:
    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("src.middleware.metrics.recording_file_size_bytes")
    def test_records_when_available(self, mock_histogram):
        record_file_size_metric(50000)
        mock_histogram.observe.assert_called_once_with(50000)

    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", False)
    def test_noop_when_unavailable(self):
        record_file_size_metric(50000)


# ── set_active_jobs ──────────────────────────────────────────────


class TestSetActiveJobs:
    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", True)
    @patch("src.middleware.metrics.active_recording_jobs")
    def test_sets_gauge_when_available(self, mock_gauge):
        set_active_jobs(3)
        mock_gauge.set.assert_called_once_with(3)

    @patch("src.middleware.metrics.PROMETHEUS_AVAILABLE", False)
    def test_noop_when_unavailable(self):
        set_active_jobs(3)
