"""Job listing / status / result / video / thumbnail / report / WebSocket routes.

Handler bodies live here; ``compilation.py`` re-exports them.  All variable
lookups (``_get_storage``, ``FFmpegOps``) are resolved through the
``compilation`` module via the late-bound :func:`_cm` accessor so
``mock.patch("src.api.routes.compilation.<name>")`` continues to work.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import WebSocketException
from fastapi.responses import Response
from redis.exceptions import RedisError
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth.identity_context import set_identity_context
from w2v_common.database.session import get_db

from src.api.schemas.compilation import (
    ChapterTimestampResponse,
    CompilationReportResponse,
    CompilationResultResponse,
    DegradationResponse,
    JobCancelResponse,
    JobListItem,
    JobListResponse,
    JobStatusResponse,
    QualityIssue,
    QualityMetricsResponse,
    StageReportResponse,
    ThumbnailRequest,
    ThumbnailResponse,
    VideoUrlResponse,
)
from src.api.ws import ConnectionManager, compilation_job_channel
from src.services.compilation.job_manager import CompilationJobManager

logger = logging.getLogger(__name__)


def _cm() -> Any:
    """Late-bound accessor for the compilation module."""
    from src.api.routes import compilation

    return compilation


def _build_compilation_report(job: Any, job_id: str) -> CompilationReportResponse | None:
    """Build a CompilationReportResponse from the persisted compilation_report blob."""
    if not job.compilation_report:
        return None
    rpt = job.compilation_report
    return CompilationReportResponse(
        job_id=rpt.get("job_id", job_id),
        compilation_time_seconds=rpt.get("compilation_time_seconds", 0),
        stages=[StageReportResponse(**s) for s in rpt.get("stages", [])],
        segments_processed=rpt.get("segments_processed", 0),
        segments_excluded=rpt.get("segments_excluded", 0),
        transitions_used=rpt.get("transitions_used", 0),
        audio_mixing_details=rpt.get("audio_mixing_details"),
        input_duration_seconds=rpt.get("input_duration_seconds", 0),
        output_duration_seconds=rpt.get("output_duration_seconds", 0),
        # Omitting these is what made a dropped feature invisible: the row is
        # 'completed', `error` is None, and the caller had no field to read.
        #
        # Absence defaults to "unknown", NOT "success". Only the producer knows
        # whether anything was swallowed, and a report blob without a status key
        # never carried that claim -- so asserting "success" here would invent a
        # guarantee on behalf of a producer that never made one.
        #
        # `subtitle_error` is deliberately NOT mapped: it is stored in the blob
        # but carries a resolved server path / raw ffmpeg text, and surfacing it
        # would create the path-existence oracle _apply_subtitles refuses to
        # return. The subtitle drop reaches the caller via `degradations`.
        status=rpt.get("status", "unknown"),
        degradations=[DegradationResponse(**d) for d in rpt.get("degradations", [])],
    )


def _build_quality_metrics(job: Any) -> QualityMetricsResponse | None:
    """Build a QualityMetricsResponse from the persisted quality_metrics blob."""
    if not job.quality_metrics:
        return None
    qm = job.quality_metrics
    return QualityMetricsResponse(
        valid=qm.get("valid", True),
        issues=[
            QualityIssue(
                check=issue.get("check", ""),
                severity=issue.get("severity", "info"),
                message=issue.get("message", ""),
            )
            for issue in qm.get("issues", [])
        ],
        file_size_bytes=qm.get("file_size_bytes", 0),
        duration_seconds=qm.get("duration_seconds", 0),
        resolution=qm.get("resolution", ""),
        audio_peak_db=qm.get("audio_peak_db"),
        encoding_valid=qm.get("encoding_valid", True),
    )


async def list_jobs(
    request: Request,
    db: AsyncSession = Depends(get_db),
    status: str | None = None,
    page: int = 1,
    page_size: int = 20,
) -> JobListResponse:
    """List compilation jobs for the current tenant."""
    tenant_id = _cm()._get_tenant_id(request)
    logger.debug("Listing jobs: tenant=%s, status=%s, page=%d", tenant_id, status, page)
    jobs, total = await CompilationJobManager.list_jobs(
        db,
        tenant_id,
        status=status,
        page=page,
        page_size=page_size,
    )
    return JobListResponse(
        jobs=[
            JobListItem(
                job_id=j.job_id,
                status=j.status,
                created_at=(j.created_at.isoformat() if j.created_at else None),
                completed_at=(j.completed_at.isoformat() if j.completed_at else None),
                preset=j.preset,
                error=j.error_message,
            )
            for j in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


async def cancel_job(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JobCancelResponse:
    """Cancel a running compilation job."""
    logger.info("Cancel request received: job_id=%s", job_id)
    service = _cm()._get_compiler_service(request)
    cancelled = await service.cancel_job(db, job_id)
    if not cancelled:
        job = await CompilationJobManager.get_job(db, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be cancelled (status: {job.status})",
        )
    logger.info("Job cancelled: job_id=%s", job_id)
    return JobCancelResponse(
        job_id=job_id,
        status="cancelled",
        message="Job cancelled successfully",
    )


async def get_job_status(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> JobStatusResponse:
    """Get the current status of a compilation job."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result_url = f"/api/compile/jobs/{job_id}/result" if job.status == "completed" else None

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        result_url=result_url,
    )


async def get_job_result(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CompilationResultResponse:
    """Get the full result of a completed compilation job."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )

    chapters = None
    if job.chapter_timestamps:
        chapters = [ChapterTimestampResponse(**ch) for ch in job.chapter_timestamps]

    report = _build_compilation_report(job, job.job_id)
    quality = _build_quality_metrics(job)

    # presigned_url is intentionally None — browsers cannot reach
    # minio.test:8389. Consumers download via the stream-through endpoint
    # /api/compile/jobs/{job_id}/video instead.
    return CompilationResultResponse(
        job_id=job.job_id,
        status=job.status,
        output_path=job.output_path,
        total_duration_seconds=job.total_duration_seconds,
        total_parts=job.total_parts,
        file_size_bytes=job.file_size_bytes,
        chapter_timestamps=chapters,
        compilation_time_seconds=job.compilation_time_seconds,
        presigned_url=None,
        error=job.error_message,
        compilation_report=report,
        quality_metrics=quality,
    )


async def get_job_video(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the compiled video file."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.output_path:
        raise HTTPException(status_code=404, detail="No video available")

    storage = _cm()._get_storage(request)
    data = await storage.retrieve(job.output_path)
    if data is None:
        # Compatibility fallback: some jobs persist a local output path while
        # running with MinIO backend. Serve the local file if it still exists.
        from pathlib import Path

        output_file = Path(job.output_path)
        if output_file.exists() and output_file.is_file():
            data = output_file.read_bytes()
    if data is None:
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="demo.mp4"'},
    )


async def get_job_video_url(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> VideoUrlResponse:
    """Return the stream-through download URL for the compiled video.

    url is always None — browsers cannot reach MinIO's internal endpoint
    (minio.test:8389), so consumers must follow fallback_download_url,
    which streams the bytes through this service.
    """
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.output_path:
        raise HTTPException(status_code=404, detail="No video available")

    return VideoUrlResponse(
        url=None,
        expires_in=0,
        fallback_download_url=f"/api/compile/jobs/{job_id}/video",
    )


async def get_job_thumbnail(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Download the thumbnail for a compiled video."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.thumbnail_path:
        raise HTTPException(status_code=404, detail="No thumbnail available")
    storage = _cm()._get_storage(request)
    data = await storage.retrieve(job.thumbnail_path)
    if data is None:
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'inline; filename="{job_id}_thumb.jpg"',
        },
    )


async def generate_thumbnail(
    job_id: str,
    body: ThumbnailRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ThumbnailResponse:
    """Generate a thumbnail at a specific timestamp."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if not job.output_path:
        raise HTTPException(status_code=404, detail="No video available")

    from pathlib import Path as PathLib

    video_path = PathLib(job.output_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    storage = _cm()._get_storage(request)
    thumb_dir = storage.get_job_dir(job_id)
    thumb_path = thumb_dir / f"thumbnail_{body.timestamp:.1f}s.jpg"

    _cm().FFmpegOps.extract_thumbnail(
        video_path,
        thumb_path,
        timestamp=body.timestamp,
        width=body.width,
        height=body.height,
    )

    job.thumbnail_path = str(thumb_path)
    await db.commit()

    return ThumbnailResponse(
        path=str(thumb_path),
        width=body.width,
        height=body.height,
        timestamp=body.timestamp,
    )


#: Close code for "this job is not yours, or there is no such job" — RFC 6455
#: 1008 "policy violation", the SAME code ``require_license_ws`` closes a refused
#: handshake with, so the socket has ONE refusal shape and not two.
WS_POLICY_VIOLATION = 1008

#: The single reason string for BOTH "no such job" and "not your job".
#:
#: ⚠️ ONE REASON, DELIBERATELY. A distinct ``forbidden`` vs ``not_found`` pair is
#: an enumeration oracle on this route: ``job_id`` is an opaque handle, so a
#: caller who could tell the two apart could walk the id space and learn which
#: jobs exist across every tenant in the fleet — from an unprivileged but
#: perfectly entitled licence. Collapsed to one answer, a foreign job is
#: indistinguishable from a job that never existed.
#:
#: It says ``job_not_found`` rather than ``forbidden`` because that is ALREADY
#: this service's chosen non-oracle for the same resource: under
#: ``TENANT_RLS_MODE=enforce`` a foreign job is invisible to
#: ``CompilationJobManager.get_job``, so every HTTP sibling route
#: (``get_job_status``, ``get_job_result``, ``get_job_video`` …) answers
#: ``404 Job not found`` for it. The socket now gives the same caller the same
#: answer in the same vocabulary instead of inventing a second one.
#:
#: (``/events/{tenant_id}`` in ``src/api/events.py`` chose 403 for the opposite
#: reason and that is not a contradiction: there the subject is a TENANT id,
#: which is not a secret and which the caller already knows.)
WS_JOB_NOT_FOUND_REASON = "job_not_found: No such compilation job for this tenant."


async def _authorize_job_for_socket(websocket: WebSocket, job_id: str) -> None:
    """Refuse the handshake unless ``job_id`` belongs to the socket's tenant.

    AUTHENTICATION IS NOT AUTHORIZATION, and this is the second half. The
    handshake now proves WHO is calling (``src/api/ws_identity.py``) and the
    licence gate proves they MAY use the service — but neither says a word about
    WHICH job. Without this the socket subscribed to ``compilation:job:{job_id}``
    for any id at all, and **Redis pub/sub has no RLS backstop**, so any entitled
    tenant could stream another tenant's compilation progress by naming their job
    id. Shipping the authentication without this would have closed the front door
    and left that window open on a socket we had just announced as secured.

    THE OWNERSHIP RULE IS REUSED, NOT REWRITTEN. The HTTP sibling routes scope a
    job to a tenant in exactly one way: they call
    :meth:`CompilationJobManager.get_job`, which carries NO tenant predicate of
    its own, and let postgres RLS decide — under ``TENANT_RLS_MODE=enforce`` the
    ``app.current_tenant`` GUC is bound from the identity ContextVar by
    ``get_db_session`` (and re-bound per transaction by
    ``src/database/rls_events.py``), so a foreign row simply is not there and the
    route answers 404. This function does the same call through the same session
    helper, which is why there is still only one ownership rule in the service; a
    hand-written ``job.tenant_id == caller`` comparison here would be a second
    one, free to drift from the policy the database actually enforces.

    ``set_identity_context`` is what makes that binding happen. Nothing else
    populates it on a ``websocket`` scope — ``AuthMiddleware`` is a
    ``BaseHTTPMiddleware`` and never runs here — so without it the GUC is unset,
    RLS fail-closes, and EVERY socket would be refused. The tenant it is seeded
    with is the introspected one the gate just accepted, never caller-supplied
    text. Contextvars are per-task and each connection is its own task, so this
    does not leak between sockets.

    COST. One indexed lookup on ``compilation_jobs.job_id``, once per handshake —
    not per event. That is the same query a single ``GET /jobs/{id}/status``
    already runs, and a socket that is about to live for minutes can afford it.

    Raises:
        WebSocketException: 1008 when the job is absent OR owned by someone else
            — one indistinguishable answer, see :data:`WS_JOB_NOT_FOUND_REASON`.
            Starlette turns it into a close BEFORE ``accept()``, matching the
            licence gate, so a refused peer never reaches the subscribe. It is
            also what stops a handshake for a nonexistent job from sitting open
            forever waiting for events that can never be published.
    """
    tenant_id = getattr(websocket.state, "tenant_id", None)
    if not tenant_id:
        # Unreachable while the gate is armed (it closes a tenant-less socket
        # first) but reachable with W2V_LICENSE_GATE_ENABLED unset. Refusing here
        # would make the dormant flag an outage; the pre-existing exposure is the
        # licence flag's to close, not this check's, and it is the same posture
        # every other route takes while dormant.
        return
    set_identity_context(tenant_id, None, getattr(websocket.state, "subject_id", None))
    async with _cm().get_db_session() as db:
        job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        logger.warning(
            "Refused WebSocket subscription to a job the caller does not own: tenant=%s job=%s",
            tenant_id,
            job_id,
        )
        raise WebSocketException(code=WS_POLICY_VIOLATION, reason=WS_JOB_NOT_FOUND_REASON)


async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time job progress updates.

    Multi-replica-safe: when Redis is available the socket subscribes to the
    per-job channel ``compilation:job:{job_id}`` (mirroring the SSE path in
    ``api/events.py``), so it receives progress no matter which replica runs the
    compilation. Without Redis it falls back to the pod-affine in-memory
    ``ConnectionManager`` (single-replica behaviour).

    The router's dependencies have already settled WHO (``stamp_ws_identity``)
    and WHETHER (``LICENSE_GATE_WS``); :func:`_authorize_job_for_socket` settles
    WHICH JOB, and it runs BEFORE either transport subscribes to anything.
    """
    await _authorize_job_for_socket(websocket, job_id)

    redis_client = getattr(websocket.app.state, "redis_client", None)
    if redis_client is not None:
        await _job_progress_ws_redis(websocket, job_id, redis_client)
        return

    manager: ConnectionManager | None = getattr(websocket.app.state, "ws_manager", None)
    if manager is None:
        await websocket.close(code=1011, reason="WebSocket not configured")
        return
    await manager.connect(job_id, websocket)
    try:
        while True:
            # Keep connection alive; handle client pings
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except TimeoutError:
                # Send ping to keep alive
                try:
                    await websocket.send_json({"event": "ping"})
                except OSError, ValueError, RuntimeError, TimeoutError:
                    break
    except WebSocketDisconnect:
        pass
    except OSError, ValueError, RuntimeError, TimeoutError:
        logger.debug("WebSocket error for job %s", job_id)
    finally:
        await manager.disconnect(job_id, websocket)


async def _job_progress_ws_redis(websocket: WebSocket, job_id: str, redis_client: Any) -> None:
    """Stream a compilation job's progress to the socket from Redis pub/sub.

    Any replica that runs the job publishes to ``compilation:job:{job_id}`` via
    ``ConnectionManager.broadcast``; this handler relays those messages to the
    connected client, so delivery is independent of which pod owns the job.
    """
    await websocket.accept()
    pubsub = None
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(compilation_job_channel(job_id))
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
            if message and message.get("type") == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    await websocket.send_text(data)
                except WebSocketDisconnect, OSError, ValueError, RuntimeError:
                    break
            else:
                # Idle keepalive; also surfaces a client disconnect via send failure.
                try:
                    await websocket.send_json({"event": "ping"})
                except WebSocketDisconnect, OSError, ValueError, RuntimeError:
                    break
    except WebSocketDisconnect:
        pass
    except OSError, ConnectionError, TimeoutError, RuntimeError, RedisError:
        # RedisError covers a mid-flight get_message/subscribe drop (redis-py
        # exceptions don't subclass the builtins) — degrade the socket gracefully.
        logger.debug("WebSocket redis stream error for job %s", job_id)
    finally:
        if pubsub is not None:
            try:
                await pubsub.unsubscribe()
                await pubsub.aclose()
            except OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, RedisError:
                pass


async def get_compilation_report(
    job_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CompilationReportResponse:
    """Get the compilation report for a completed job."""
    job = await CompilationJobManager.get_job(db, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    report = _build_compilation_report(job, job.job_id)
    if report is None:
        raise HTTPException(status_code=404, detail="No compilation report available")
    return report


def register(router: APIRouter) -> None:
    """Register all job-management routes onto the supplied router."""
    router.add_api_route(
        "/jobs",
        list_jobs,
        methods=["GET"],
        operation_id="list_compilation_jobs",
        response_model=JobListResponse,
    )
    router.add_api_route(
        "/jobs/{job_id}",
        cancel_job,
        methods=["DELETE"],
        operation_id="cancel_compilation_job",
        response_model=JobCancelResponse,
        responses={
            404: {"description": "Job not found"},
            409: {"description": "Job cannot be cancelled"},
        },
    )
    router.add_api_route(
        "/jobs/{job_id}",
        get_job_status,
        methods=["GET"],
        operation_id="get_compilation_job_status",
        response_model=JobStatusResponse,
        responses={404: {"description": "Job not found"}},
    )
    router.add_api_route(
        "/jobs/{job_id}/result",
        get_job_result,
        methods=["GET"],
        operation_id="get_compilation_result",
        response_model=CompilationResultResponse,
        responses={
            404: {"description": "Job not found"},
            409: {"description": "Job not completed yet"},
        },
    )
    router.add_api_route(
        "/jobs/{job_id}/video",
        get_job_video,
        methods=["GET"],
        operation_id="download_compiled_video",
        responses={404: {"description": "Job or video not found"}},
    )
    router.add_api_route(
        "/jobs/{job_id}/video-url",
        get_job_video_url,
        methods=["GET"],
        operation_id="get_compiled_video_url",
        response_model=VideoUrlResponse,
        responses={404: {"description": "Job or video not found"}},
    )
    router.add_api_route(
        "/jobs/{job_id}/thumbnail",
        get_job_thumbnail,
        methods=["GET"],
        operation_id="download_compilation_thumbnail",
        responses={404: {"description": "Job or thumbnail not found"}},
    )
    router.add_api_route(
        "/jobs/{job_id}/thumbnail",
        generate_thumbnail,
        methods=["POST"],
        operation_id="generate_compilation_thumbnail",
        response_model=ThumbnailResponse,
        responses={
            404: {"description": "Job or video not found"},
            422: {"description": "Validation error"},
        },
    )
    router.add_api_route(
        "/jobs/{job_id}/report",
        get_compilation_report,
        methods=["GET"],
        operation_id="get_compilation_report",
        response_model=CompilationReportResponse,
        responses={404: {"description": "Job or report not found"}},
    )


def register_ws(router: APIRouter) -> None:
    """Register the job-progress WebSocket onto the WS-gated router.

    Separate from :func:`register` on purpose. The supplied router must carry
    BOTH, in this order: ``stamp_ws_identity`` (which introspects the handshake
    licence token and stamps ``websocket.state.tenant_id``) and then
    ``LICENSE_GATE_WS`` — not the HTTP ``LICENSE_GATE``, which is unmountable on
    a WebSocket route. Registering this handler on a router that carries only the
    gate closes 1008 ``license_none`` for every caller, entitled or not. See the
    long comment above ``router_ws`` in ``src/api/routes/compilation.py``.
    """
    router.add_api_websocket_route("/jobs/{job_id}/ws", job_progress_ws)
