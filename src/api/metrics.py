"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter(tags=["metrics"])


@router.get("/metrics", operation_id="prometheus_metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return Response(
            content="# prometheus_client not installed\n",
            media_type="text/plain",
            status_code=501,
        )
