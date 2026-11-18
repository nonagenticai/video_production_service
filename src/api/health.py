"""Health check endpoints for video_production_service."""

from __future__ import annotations

import logging
from typing import Any, TypedDict, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from w2v_common.cache.redis_client import get_redis_client
from w2v_common.database.connection import get_engine

from src.config import get_settings
from src.services.recording.concurrency import ExecutorStallStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Why these tuples name library bases explicitly: a health check that cannot
# describe a dependency's failure is worse than no health check. SQLAlchemy
# raises ``SQLAlchemyError(Exception)`` and redis-py raises
# ``RedisError(Exception)``; NEITHER derives from ``OSError``, and redis-py's
# ``ConnectionError``/``TimeoutError`` are distinct classes that merely share
# their names with the builtins. A tuple of builtins alone therefore never
# fires, so a dead dependency escapes as a 500 instead of being reported
# "unhealthy" -- and on ``/health/ready`` (a k8s probe) a 500 is not the 503
# that tells the kubelet to stop routing traffic here.
_DEPENDENCY_ERRORS = (OSError, RuntimeError, SQLAlchemyError, RedisError)


async def _check_database() -> str:
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "healthy"
    except _DEPENDENCY_ERRORS as e:
        return f"unhealthy: {e}"


async def _check_redis() -> str:
    """Probe Redis for the health report.

    ``RedisError`` is required in the tuple: redis-py raises its own
    ``ConnectionError``/``TimeoutError`` off ``RedisError(Exception)``, which does
    not derive from ``OSError``. Without it, ``ping()`` against a down Redis
    escaped this handler and 500'd the whole ``/health`` aggregate — reporting the
    service as unreachable rather than reporting Redis as unhealthy, which is
    exactly the distinction the endpoint exists to make.

    Returns:
        ``"healthy"``, ``"not configured"``, or ``"unhealthy: <reason>"``.
    """
    try:
        redis_client = get_redis_client()
        if not redis_client:
            return "not configured"
        # ``RedisClient.ping`` swallows ``RedisError`` and returns False, so a
        # down Redis never raises here -- the return value IS the signal.
        if not await redis_client.ping():
            return "unhealthy: ping failed"
        return "healthy"
    except (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError) as e:
        return f"unhealthy: {e}"


async def _check_storage(request: Request) -> str:
    try:
        storage = getattr(request.app.state, "storage_service", None)
        if not storage:
            return "not initialized"
        if not await storage.health_check():
            return "unhealthy"
        return "healthy"
    except _DEPENDENCY_ERRORS as e:
        return f"unhealthy: {e}"


@router.get("/health", operation_id="health_check", response_model=None)
async def health_check(request: Request) -> dict[str, Any] | JSONResponse:
    """Combined health check for all components."""
    try:
        db_status = await _check_database()
        redis_status = await _check_redis()
        storage_status = await _check_storage(request)

        overall = "healthy"
        if db_status != "healthy":
            overall = "unhealthy"
        elif redis_status != "healthy" or storage_status != "healthy":
            overall = "degraded"

        result: dict[str, Any] = {
            "status": overall,
            "components": {
                "database": db_status,
                "redis": redis_status,
                "storage": storage_status,
                "api": "healthy",
            },
        }

        # Include concurrency stats if available
        concurrency = getattr(request.app.state, "concurrency_manager", None)
        if concurrency:
            result["concurrency"] = concurrency.stats()

        return result
    except _DEPENDENCY_ERRORS as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


class LivenessReport(TypedDict, total=False):
    """Body of ``/health/live``.

    ``executor`` carries the stall report when the check actually RAN.
    ``executor_check`` is present exactly when it did not, and says WHY:

    ``"disabled"``                  deliberately off (threshold <= 0).
    ``"manager_missing"``           no ``concurrency_manager`` on ``app.state``.
    ``"stall_status_unavailable"``  the manager has no callable ``stall_status``.
    ``"stall_status_error"``        ``stall_status()`` raised.

    The last three are BROKEN: the wedge detector is not running and liveness has
    degraded to a bare process-alive check. Before this field they were
    byte-identical to ``"disabled"``, so a startup-ordering bug or a rename could
    silently delete the only signal that catches a wedged recording executor —
    exactly the 9-hour outage this probe exists for — while ``/healthz`` stayed
    green forever. The probe still returns 200 in every one of those cases: a
    crashloop over a probe bug is worse than a blind probe. The remedy is an
    alert on ``executor_check`` not being ``"disabled"``, not a restart.
    """

    status: str
    executor: ExecutorStallStatus
    executor_check: str


def _executor_stall(request: Request) -> tuple[ExecutorStallStatus | None, str | None]:
    """Return ``(stall_report, unavailable_reason)`` — exactly one is non-None.

    Kept deliberately total: a liveness probe that raises is a probe that
    restarts the pod for its own bug, so anything unexpected here reports
    "not checked" rather than "wedged". But "not checked" must be VISIBLE and
    say which kind it is, so a broken probe cannot masquerade as a disabled one.
    """
    threshold = get_settings().recording.executor_stall_threshold_seconds
    if threshold <= 0:
        return None, "disabled"
    concurrency = getattr(request.app.state, "concurrency_manager", None)
    if concurrency is None:
        logger.error(
            "Executor stall check UNAVAILABLE: no concurrency_manager on app.state; "
            "liveness is now only a process-alive check and cannot detect a wedged executor"
        )
        return None, "manager_missing"
    status = getattr(concurrency, "stall_status", None)
    if not callable(status):
        logger.error(
            "Executor stall check UNAVAILABLE: %s has no callable stall_status(); "
            "liveness is now only a process-alive check and cannot detect a wedged executor",
            type(concurrency).__name__,
        )
        return None, "stall_status_unavailable"
    try:
        return cast(ExecutorStallStatus, status(threshold)), None
    except Exception:  # never restart a pod over a probe bug
        logger.exception("Executor stall check FAILED; the wedge detector is not running")
        return None, "stall_status_error"


@router.get("/health/live", operation_id="health_live", response_model=None)
async def health_live(request: Request) -> LivenessReport | JSONResponse:
    """Kubernetes liveness probe -- process alive AND recording executor moving.

    Why LIVENESS and not readiness: the failure this exists to catch is a
    WEDGED recording executor (slots held by a job that will never finish, so
    every subsequent job sits in ``queued`` until the reaper fails it). Pulling
    the pod out of the Service, which is all readiness does, would take the
    HTTP API down too and leave the wedge in place -- nothing else picks the
    work up. A restart is the actual remedy: it drops the leaked slots and the
    executor resumes. So the signal belongs on the probe that restarts.

    Why it cannot flap: :pymeth:`RecordingConcurrencyManager.stall_status`
    reports healthy whenever the wait queue is empty, so an idle service -- for
    any length of time -- stays alive. Only work that is demonstrably WAITING
    while no slot is admitted or released for longer than
    ``executor_stall_threshold_seconds`` fails the probe.
    """
    stall, unavailable = _executor_stall(request)
    if stall is not None and stall["stalled"]:
        logger.error(
            "Liveness FAILED: recording executor stalled -- %d job(s) queued, "
            "no slot admitted or released for %.0fs (threshold %.0fs)",
            stall["queued"],
            stall["seconds_since_progress"] or 0.0,
            stall["threshold_seconds"],
        )
        return JSONResponse(status_code=503, content={"status": "unhealthy", "executor": stall})
    result: LivenessReport = {"status": "ok"}
    if stall is not None:
        result["executor"] = stall
    else:
        # Still 200 — a crashloop over a probe bug is worse than a blind probe —
        # but the body now says the detector did not run, and which kind of
        # not-run it was. "disabled" is a choice; anything else is a defect.
        result["executor_check"] = unavailable or "unavailable"
    return result


@router.get("/healthz", operation_id="health.live", response_model=None)
async def healthz(request: Request) -> LivenessReport | JSONResponse:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await health_live(request)


@router.get("/health/ready", operation_id="health_ready", response_model=None)
async def health_ready() -> dict[str, Any] | JSONResponse:
    """Kubernetes readiness probe -- checks database connectivity."""
    try:
        engine = await get_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"status": "healthy", "ready": True, "checks": {"database": {"status": "up"}}}
    except _DEPENDENCY_ERRORS as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ready": False,
                "checks": {"database": {"status": "down", "error": str(e)}},
            },
        )


@router.get("/readyz", operation_id="health.ready", response_model=None)
async def readyz() -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready()
