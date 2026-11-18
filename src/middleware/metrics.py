"""Prometheus metrics middleware and instrumentation."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from prometheus_client import Counter, Gauge, Histogram

    http_requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    http_request_duration_seconds = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    recording_jobs_total = Counter(
        "recording_jobs_total",
        "Total recording jobs by status",
        ["status"],
    )
    recording_duration_seconds = Histogram(
        "recording_duration_seconds",
        "Recording duration in seconds",
        buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    )
    recording_file_size_bytes = Histogram(
        "recording_file_size_bytes",
        "Recording file size in bytes",
        buckets=(10000, 100000, 1000000, 10000000, 50000000, 100000000),
    )
    active_recording_jobs = Gauge(
        "active_recording_jobs",
        "Number of currently active recording jobs",
    )
    # The counterpart to active_recording_jobs, and the one that was missing.
    # Active alone cannot distinguish "3 running, nothing waiting" (healthy, at
    # cap) from "3 running, 200 waiting" (the 2026-09-03 collapse). Both read
    # active=3. Depth is what makes the difference visible.
    queued_recording_jobs = Gauge(
        "queued_recording_jobs",
        "Number of recording jobs waiting for a concurrency slot on this pod",
    )

    # Seconds that recording slots have been held with ZERO admissions or
    # releases, published on every stall check. 0 when nothing is held.
    #
    # This is the ONLY instrument for the one wedge shape `/health/live` cannot
    # catch: with the queue empty (trickle traffic — each waiter times out before
    # the next arrives) `stall_status` reports healthy by design, so a pod whose
    # executor is wedged fails every request and never trips the liveness probe.
    # A gauge is deliberately all this does — see `stall_status` for why the
    # signal must NOT restart pods until its distribution has been measured under
    # real traffic, and this gauge is what makes measuring it possible.
    recording_slot_held_without_progress_seconds = Gauge(
        "recording_slot_held_without_progress_seconds",
        "Seconds recording slots have been held with no slot admitted or released (0 = nothing held)",
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect HTTP request metrics for Prometheus."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Record request count and duration metrics per endpoint.

        Args:
            request: Incoming HTTP request.
            call_next: Downstream handler.

        Returns:
            Response from downstream.
        """
        if not PROMETHEUS_AVAILABLE:
            return await call_next(request)

        start_time = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start_time

        # Normalize endpoint path (replace UUIDs with placeholder)
        path = request.url.path
        method = request.method

        http_requests_total.labels(
            method=method,
            endpoint=path,
            status_code=response.status_code,
        ).inc()
        http_request_duration_seconds.labels(
            method=method,
            endpoint=path,
        ).observe(duration)

        return response


def record_job_metric(status: str) -> None:
    """Record a job completion metric."""
    if PROMETHEUS_AVAILABLE:
        recording_jobs_total.labels(status=status).inc()


def record_duration_metric(duration: float) -> None:
    """Record recording duration."""
    if PROMETHEUS_AVAILABLE:
        recording_duration_seconds.observe(duration)


def record_file_size_metric(size: int) -> None:
    """Record recording file size."""
    if PROMETHEUS_AVAILABLE:
        recording_file_size_bytes.observe(size)


def set_active_jobs(count: int) -> None:
    """Set active recording jobs gauge."""
    if PROMETHEUS_AVAILABLE:
        active_recording_jobs.set(count)


def set_queued_jobs(count: int) -> None:
    """Set the queued (waiting-for-a-slot) recording jobs gauge."""
    if PROMETHEUS_AVAILABLE:
        queued_recording_jobs.set(count)


def set_held_without_progress_seconds(seconds: float) -> None:
    """Publish how long slots have been held with no admission or release."""
    if PROMETHEUS_AVAILABLE:
        recording_slot_held_without_progress_seconds.set(seconds)
