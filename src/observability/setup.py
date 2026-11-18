"""OpenTelemetry instrumentation setup."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_observability(
    app: Any = None,
    service_name: str = "video-production-service",
    otlp_endpoint: str = "http://localhost:4317",
) -> None:
    """Initialize OpenTelemetry tracing and metrics.

    Gracefully no-ops if opentelemetry packages are not installed.
    """
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
        )
    except ImportError:
        logger.info("OpenTelemetry packages not installed, skipping instrumentation")
        return

    resource = Resource.create({"service.name": service_name})

    # Tracing
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Metrics
    metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=30000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # Custom meters
    meter = metrics.get_meter(service_name)
    meter.create_histogram("compilation_duration_seconds", description="Compilation duration")
    meter.create_up_down_counter("jobs_in_progress", description="Currently processing jobs")
    meter.create_histogram("ffmpeg_op_duration_seconds", description="FFmpeg operation duration")

    # FastAPI auto-instrumentation
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import (
                FastAPIInstrumentor,
            )

            FastAPIInstrumentor.instrument_app(app)
            logger.info("FastAPI auto-instrumentation enabled")
        except ImportError:
            logger.debug("FastAPI OTel instrumentor not available")

    logger.info("OpenTelemetry initialized (endpoint=%s)", otlp_endpoint)


def get_tracer(name: str = "video-production-service") -> Any:
    """Get a tracer, returning a no-op tracer if OTel is not initialized."""
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return None


def get_meter(name: str = "video-production-service") -> Any:
    """Get a meter, returning None if OTel is not initialized."""
    try:
        from opentelemetry import metrics

        return metrics.get_meter(name)
    except ImportError:
        return None
