"""OpenTelemetry observability module."""

from src.observability.setup import get_meter, get_tracer, setup_observability

__all__ = ["get_meter", "get_tracer", "setup_observability"]
