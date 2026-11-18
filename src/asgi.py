"""
ASGI application for video_production_service.

Merged FastAPI REST API combining recording and compilation capabilities.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from redis.exceptions import RedisError
from w2v_common.asgi import create_app
from w2v_common.cache.redis_client import (
    RedisClient,
    close_redis,
    get_redis_client,
    init_redis,
)
from w2v_common.clients import ArtifactClient
from w2v_common.database.connection import close_engine, get_engine
from w2v_common.observability import database_endpoint, redis_endpoint
from w2v_common.observability.request_context import RequestIdLogFilter

from src.config import get_settings

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================================
# Health Check Logging Filter
# ============================================================================


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress access logs for health-check endpoints to reduce noise."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for log records that mention health-check paths."""
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckLoggingFilter())


# ============================================================================
# Lifespan
# ============================================================================


async def _init_database(app: FastAPI) -> None:
    """Initialize database engine and create tables."""
    # Phase D RLS: install the after_begin GUC re-bind listener before any session is
    # opened, so the multi-commit recording/compilation pipelines keep
    # app.current_tenant bound across each commit under enforce.
    from src.database.rls_events import install_rls_tenant_rebind

    install_rls_tenant_rebind()
    try:
        db_engine = await get_engine()
        app.state.db_engine = db_engine
        logger.info("Database connected: %s", database_endpoint(settings.database.url))
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        raise

    # Schema is owned by alembic -- the gitops Deployment runs
    # ``alembic upgrade head`` in an initContainer before this app
    # starts (see gitops-production
    # clusters/strato/services/w2v/deployments.yaml). Do NOT call
    # ``Base.metadata.create_all`` here: it bypasses migration history
    # and can drift the live schema away from what alembic expects.
    logger.info("Database connected (schema managed by alembic)")


async def _init_redis(app: FastAPI) -> None:
    """Initialize Redis with graceful fallback when unavailable."""
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        # w2v_common IdempotencyMiddleware reads its store from ``app.state.redis``
        # (not ``redis_client``); unset it falls open and dedups nothing.
        app.state.redis = app.state.redis_client
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
    except (ImportError, OSError, RuntimeError, ValueError, RedisError) as e:
        # Recording/compilation services keep their per-pod fallback (None).
        app.state.redis_client = None
        app.state.redis = _recoverable_idempotency_store()
        if app.state.redis is None:
            logger.warning(
                "Redis client could not be built (%s) -- caching disabled and idempotency "
                "dedupe unavailable until this pod is restarted; duplicate recordings may "
                "be paid twice",
                e,
                exc_info=True,
            )
        else:
            logger.warning(
                "Redis unreachable at startup (%s) -- caching disabled; idempotency dedupe "
                "unavailable until Redis is reachable; duplicate recordings may be paid twice",
                e,
                exc_info=True,
            )


def _recoverable_idempotency_store() -> RedisClient | None:
    """Return the Redis handle the idempotency middleware can retry on, if one exists.

    A startup connect that fails must not disable dedupe for the life of the pod.
    ``RedisClient.connect()`` builds its pool and client BEFORE the verifying PING,
    so after a failed PING the handle is still usable: ``_ensure_connected()``
    returns the client and redis-py's pool opens a fresh connection per command.
    Handing that handle to the middleware means every request retries Redis --
    falling open while it is down (``get``/``set_nx`` swallow ``RedisError``),
    deduping again the moment it is back.

    ``None`` only when no client exists at all (missing/malformed ``REDIS_URL``):
    that is a config fault no retry can heal.
    """
    try:
        client = get_redis_client()
    except RuntimeError, ValueError:
        return None
    return client if client.is_available else None


async def _init_recording_services(app: FastAPI) -> None:
    """Wire up recording-domain services on app.state."""
    from src.services.job_lease import JobLease
    from src.services.recording.animation_recorder import AnimationRecorder
    from src.services.recording.concurrency import RecordingConcurrencyManager
    from src.services.recording.content_recorder import ContentRecorder
    from src.services.recording.playwright_recorder import PlaywrightRecorder
    from src.services.recording.recorder_service import RecorderService
    from src.services.recording.url_recorder import UrlRecorder
    from src.services.recording.webhook_service import WebhookService

    redis_client = app.state.redis_client
    instance_id = app.state.instance_id

    # Fleet-wide recording concurrency cap (Redis-backed; per-pod fallback if no Redis).
    concurrency_manager = RecordingConcurrencyManager(
        max_concurrent=settings.recording.max_concurrent_recordings,
        per_pod_max=settings.recording.max_concurrent_recordings_per_pod,
        queue_wait_timeout_seconds=settings.recording.queue_wait_timeout_seconds,
        redis_client=redis_client if settings.recording.global_concurrency_enabled else None,
        instance_id=instance_id,
        slot_key=settings.recording.global_slot_redis_key,
        slot_ttl_seconds=settings.recording.global_slot_ttl_seconds,
        poll_interval_seconds=settings.recording.global_slot_poll_seconds,
    )
    app.state.concurrency_manager = concurrency_manager
    logger.info(
        "Concurrency manager: fleet max=%d, per-pod max=%d, queue wait ceiling=%.0fs "
        "(fleet-wide=%s)",
        settings.recording.max_concurrent_recordings,
        settings.recording.max_concurrent_recordings_per_pod,
        settings.recording.queue_wait_timeout_seconds,
        bool(redis_client) and settings.recording.global_concurrency_enabled,
    )

    # Per-job Redis liveness lease (crashed-pod reclaim + reaper skip of live jobs).
    # stalled_probe is what stops this lease vouching for a job that has stopped
    # working. Without it the heartbeat renews every held lease forever, the reaper
    # skips every job it correctly identified as stale, and the fleet-wide recording
    # semaphore starves. See JobLease._heartbeat_loop for the measured incident.
    from src.services.recording.cleanup_service import stalled_recording_job_ids

    recording_lease = JobLease(
        redis_client,
        prefix="recording",
        instance_id=instance_id,
        ttl_seconds=settings.job.lease_ttl_seconds,
        heartbeat_seconds=settings.job.lease_heartbeat_seconds,
        stalled_probe=stalled_recording_job_ids,
    )
    app.state.recording_job_lease = recording_lease

    event_publisher = None
    if redis_client:
        from src.services.recording.event_publisher import EventPublisher

        event_publisher = EventPublisher(redis_client)
        logger.info("Event publisher initialized")

    webhook_service = WebhookService()

    app.state.recorder_service = RecorderService(
        storage=app.state.storage_service,
        recorder=PlaywrightRecorder(),
        content_recorder=ContentRecorder(),
        url_recorder=UrlRecorder(),
        animation_recorder=AnimationRecorder(),
        concurrency_manager=concurrency_manager,
        webhook_service=webhook_service,
        event_publisher=event_publisher,
        job_lease=recording_lease,
    )
    logger.info(
        "RecorderService initialized (viewport=%dx%d)",
        settings.recording.viewport_width,
        settings.recording.viewport_height,
    )


async def _init_compilation_services(app: FastAPI) -> None:
    """Wire up compilation-domain services on app.state."""
    from src.api.ws import ConnectionManager
    from src.services.compilation.compiler_service import CompilerService
    from src.services.job_lease import JobLease

    redis_client = app.state.redis_client
    instance_id = app.state.instance_id

    # Redis-backed so progress reaches a client on ANY replica (not just the pod
    # running the job); degrades to in-memory (pod-affine) when Redis is absent.
    ws_manager = ConnectionManager(redis_client=redis_client)
    app.state.ws_manager = ws_manager

    # stalled_probe is NOT optional here, for the same reason it is not optional on
    # the recording lease 50 lines above: without it JobLease._heartbeat_loop renews
    # every held lease forever, and `_reap_stale_jobs` skips every row it correctly
    # selected because `present()` is True. A compile wedged on a LIVE pod then stays
    # in `processing` permanently. This lease shipped without the probe while its
    # identical twin had it.
    from src.services.compilation.cleanup_service import stalled_compilation_job_ids

    compilation_lease = JobLease(
        redis_client,
        prefix="compilation",
        instance_id=instance_id,
        ttl_seconds=settings.job.lease_ttl_seconds,
        heartbeat_seconds=settings.job.lease_heartbeat_seconds,
        stalled_probe=stalled_compilation_job_ids,
    )
    app.state.compilation_job_lease = compilation_lease

    app.state.compiler_service = CompilerService(
        storage=app.state.storage_service,
        ws_manager=ws_manager,
        job_lease=compilation_lease,
    )
    logger.info(
        "CompilerService initialized (target=%dx%d, crf=%d)",
        settings.compilation.target_width,
        settings.compilation.target_height,
        settings.compilation.crf,
    )


async def _init_cleanup_services(app: FastAPI) -> None:
    """Start recording and compilation cleanup services."""
    from src.services.compilation.cleanup_service import (
        CleanupService as CompilationCleanupService,
    )
    from src.services.recording.cleanup_service import CleanupService as RecordingCleanupService

    recording_cleanup = RecordingCleanupService(
        app.state.storage_service, job_lease=app.state.recording_job_lease
    )
    await recording_cleanup.start()
    app.state.recording_cleanup_service = recording_cleanup

    compilation_cleanup = CompilationCleanupService(job_lease=app.state.compilation_job_lease)
    await compilation_cleanup.start()
    app.state.compilation_cleanup_service = compilation_cleanup


async def _probe_vp9_alpha(app: FastAPI) -> None:
    """Detect once, at startup, whether this ffmpeg build preserves VP9 alpha.

    Deliberately NON-fatal, and that is a blast-radius decision rather than
    leniency. Transparent overlay rendering is one product path among many: a
    replica that cannot do it still serves every recording, every compilation
    and every opaque animation correctly. Raising here would crash-loop the
    whole Deployment and convert one degraded feature into a total outage —
    and it would do so on a dependency (the ffmpeg binary baked into the
    image) that no amount of restarting can fix.

    The loudness moves to two places instead: this ERROR log at startup (see
    alpha_probe for the full message), and a hard refusal on the request path,
    so a transparent job on an incapable replica fails with a stated reason
    rather than returning a clip whose alpha was silently discarded.
    """
    from src.services.ffmpeg.alpha_probe import probe_vp9_alpha_support

    try:
        # Sync ffmpeg + ffprobe subprocesses; keep them off the event loop.
        supported = await asyncio.to_thread(probe_vp9_alpha_support)
    except (OSError, RuntimeError, ValueError) as e:
        # A probe that cannot run is treated exactly like a probe that failed:
        # unknown capability must never read as "available".
        supported = False
        logger.error("VP9 alpha capability probe could not run: %s", e, exc_info=True)

    app.state.vp9_alpha_supported = supported
    if supported:
        logger.info("VP9 alpha capability: OK -- transparent overlay renders available")
    else:
        logger.warning(
            "VP9 alpha capability: UNAVAILABLE -- transparent overlay renders will be "
            "REFUSED on this replica (all other recording/compilation paths unaffected)"
        )


def _init_observability(app: FastAPI) -> None:
    """Initialize OpenTelemetry if enabled. No-op otherwise."""
    if not settings.otel.enabled:
        return
    try:
        from src.observability import setup_observability

        setup_observability(
            app=app,
            service_name=settings.otel.service_name,
            otlp_endpoint=settings.otel.otlp_endpoint,
        )
        logger.info("OpenTelemetry instrumentation enabled")
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Failed to initialize OpenTelemetry: %s", e)


async def _release_fleet_state(app: FastAPI) -> None:
    """Release fleet-wide recording slots + per-job liveness leases on graceful
    drain so a sibling/restart reclaims capacity immediately (not by TTL)."""
    concurrency_manager = getattr(app.state, "concurrency_manager", None)
    if concurrency_manager is not None:
        try:
            await concurrency_manager.close()
        except (OSError, RuntimeError, ValueError) as e:
            logger.error("Error closing concurrency manager: %s", e, exc_info=True)
    for attr, label in (
        ("recording_job_lease", "recording job lease"),
        ("compilation_job_lease", "compilation job lease"),
    ):
        lease = getattr(app.state, attr, None)
        if lease is not None:
            try:
                await lease.release_all()
            except (OSError, RuntimeError, ValueError) as e:
                logger.error("Error releasing %s: %s", label, e, exc_info=True)


async def _shutdown_services(app: FastAPI) -> None:
    """Stop all background services and close DB/Redis connections."""
    for attr, label in (
        ("recording_cleanup_service", "recording cleanup service"),
        ("compilation_cleanup_service", "compilation cleanup service"),
    ):
        try:
            svc = getattr(app.state, attr, None)
            if svc is not None:
                await svc.stop()
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error("Error stopping %s: %s", label, e, exc_info=True)

    await _release_fleet_state(app)

    try:
        artifact_client = getattr(app.state, "artifact_client", None)
        if artifact_client is not None:
            await artifact_client.aclose()
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.error("Error closing ArtifactClient: %s", e, exc_info=True)

    for closer, label in ((close_redis, "Redis"), (close_engine, "database")):
        try:
            await closer()
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error("Error closing %s: %s", label, e, exc_info=True)


async def _startup(app: FastAPI) -> None:
    """Init DB, Redis, storage, recording + compilation services (create_app on_startup)."""
    from src.services.ffmpeg import FFmpegOps
    from src.services.storage_service import StorageService

    logger.info("=" * 70)
    logger.info("Starting Video Production Service")
    logger.info("=" * 70)

    # Stable-ish per-replica identity for Redis leases/slots (pod name in k8s).
    app.state.instance_id = os.environ.get("HOSTNAME") or f"vps-{uuid.uuid7().hex[:12]}"
    logger.info("Replica instance_id=%s", app.state.instance_id)

    await _init_database(app)
    await _init_redis(app)

    storage = StorageService()
    await storage.initialize()
    app.state.storage_service = storage

    FFmpegOps.configure_from_settings()
    # After configure_from_settings: the probe drives a real ffmpeg encode
    # through FFmpegOps.run and must inherit the configured timeout/gate.
    await _probe_vp9_alpha(app)

    await _init_recording_services(app)
    await _init_compilation_services(app)
    await _init_cleanup_services(app)
    _init_observability(app)

    artifact_client = ArtifactClient()
    app.state.artifact_client = artifact_client
    logger.info("ArtifactClient initialized: %s", artifact_client.base_url)

    logger.info("=" * 70)
    logger.info("Video Production Service started successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s:%d", settings.api.host, settings.api.port)
    logger.info("=" * 70)


async def _shutdown(app: FastAPI) -> None:
    """Stop background services and close DB/Redis (create_app on_shutdown)."""
    logger.info("Shutting down Video Production Service")
    await _shutdown_services(app)
    logger.info("Video Production Service shutdown complete")


# ============================================================================
# Application
# ============================================================================

# Routers + service-specific middleware are imported here so create_app receives
# them via routers= / extra_middleware=.
from src.api.analytics import router as analytics_router  # noqa: E402
from src.api.events import router as events_router  # noqa: E402
from src.api.health import router as health_router  # noqa: E402
from src.api.metrics import router as metrics_router  # noqa: E402
from src.api.routes import (  # noqa: E402
    compilation_router,
    compilation_ws_router,
    recording_router,
)
from src.middleware.metrics import MetricsMiddleware  # noqa: E402
from src.middleware.rate_limiter import RateLimitMiddleware  # noqa: E402

app = create_app(
    settings,
    routers=[
        recording_router,
        compilation_router,
        # WS-gated sibling of compilation_router: /api/compile/jobs/{job_id}/ws
        # carries LICENSE_GATE_WS (1008/1013), not the HTTP LICENSE_GATE, which
        # is unmountable on a WebSocket route. See src/api/routes/compilation.py.
        compilation_ws_router,
        health_router,
        metrics_router,
        events_router,
        analytics_router,
    ],
    title=settings.app_name,
    description="Video production microservice combining browser recording and FFmpeg compilation",
    version=settings.app_version,
    on_startup=[_startup],
    on_shutdown=[_shutdown],
    # Service-specific middleware, appended OUTSIDE the canonical stack. These are
    # the local w2v-specific implementations (not the factory's shared metrics):
    # Metrics (outermost) times the whole request; the rate limiter guards ingress.
    extra_middleware=[(MetricsMiddleware, {}), (RateLimitMiddleware, {})],
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route
    # that forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's
    # default `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {"name": "record", "description": "Browser recording jobs (Playwright)."},
        {
            "name": "compile",
            "description": "Video compilation, batching, presets, themes (FFmpeg).",
        },
        {"name": "themes", "description": "Compilation theme catalogue."},
        {"name": "presets", "description": "Compilation preset catalogue (built-in + custom)."},
        {"name": "templates", "description": "Reusable compilation templates."},
        {"name": "admin", "description": "On-demand admin/maintenance operations."},
        {"name": "analytics", "description": "Usage analytics for the calling tenant."},
        {"name": "events", "description": "Server-Sent Events for job progress."},
        {"name": "metrics", "description": "Prometheus metrics endpoint."},
    ],
)

# Bind request_id onto every log record (factory sets the ContextVar via
# RequestIdMiddleware; this filter surfaces it in the log formatter).
logging.getLogger().addFilter(RequestIdLogFilter())


# ============================================================================
# Root & Health Endpoints
# ============================================================================


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and available endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Video production microservice combining browser recording and FFmpeg compilation",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "record": "/api/record",
            "record_part": "/api/record/part",
            "record_content": "/api/record/content",
            "record_url": "/api/record/url",
            "record_batch": "/api/record/batch",
            "jobs": "/api/record/jobs",
            "compile": "/api/compile",
            "title_card": "/api/compile/title-card",
            "edit_plan": "/api/compile/edit-plan",
            "video": "/api/compile/{id}/video",
            "metrics": "/metrics",
            "events": "/events/{tenant_id}",
            "analytics": "/api/analytics",
        },
        "status": "running",
        "environment": settings.environment,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
