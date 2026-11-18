"""API routes for video recording.

Provides REST endpoints for recording creation, job status polling,
video retrieval, batch operations, and job cancellation.

Tenant provenance in the ``_bg`` entrypoints
--------------------------------------------
Every ``_process_*_job_bg`` below is dispatched with ``asyncio.create_task``
from a route handler, and each seeds ``set_identity_context`` so the Phase-D
RLS bind (``after_begin`` re-binds every transaction) applies to the fresh
session it opens. That seeding call OMITS ``tenant_source`` deliberately.

``asyncio.create_task`` COPIES the caller's context, so ``AuthMiddleware``'s
provenance is ALREADY PRESENT inside these tasks. Since w2v-common 0.67.7 the
``tenant_source`` parameter defaults to a ``KEEP_TENANT_SOURCE`` sentinel, which
makes "omitted" distinguishable from "explicitly None"::

    omitted, SAME tenant as ambient   → provenance PRESERVED
    omitted, DIFFERENT tenant         → provenance DROPPED to None (safety net)
    tenant_source=None   (explicit)   → cleared
    tenant_source=<value> (explicit)  → ALWAYS wins

These entrypoints rebind the SAME tenant they inherited, so the bare call keeps
the request's provenance and metered hops stay vouched. Without that preserve,
``build_outbound_identity`` would report ``None`` as ``'unrecorded'``::

    refusing to delegate tenant <t> to metered destination <d>:
    provenance is 'unrecorded', which is not vouched

Measured 2026-08-28: that is why modality-gateway refused panel image
generation in the w2v_frontend E2E post-promote matrix (the same defect, fixed
in ai_content_generation_service #109).

Do NOT "harden" this by passing ``tenant_source=get_tenant_source_from_context()``:
an explicit value ALWAYS wins, so it bypasses the different-tenant drop and would
carry one tenant's proof onto another tenant's rebind. And do NOT delete the
seeding call — it exists for the RLS bind, and removing it breaks multi-commit
jobs.

The same bare call is what a reaper/worker loop that iterates rows for DIFFERENT
tenants wants — see ``src/services/recording/cleanup_service.py``, where the
tenant-change drop clears provenance per row.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from w2v_common.auth import LICENSE_GATE
from w2v_common.auth.identity_context import set_identity_context
from w2v_common.clients import ArtifactClient
from w2v_common.database.session import get_db_session

from src.api.dependencies import get_recording_job_manager
from src.api.schemas.recording import (
    AsyncJobResponse,
    BatchJobResponse,
    BatchStatusResponse,
    CaptureScreensRequest,
    JobStatusResponse,
    PaginatedJobsResponse,
    RecordAnimationRequest,
    RecordBatchFromArtifactsRequest,
    RecordBatchRequest,
    RecordContentRequest,
    RecordingResultResponse,
    RecordPartRequest,
    RecordUrlRequest,
)
from src.api.tenant import require_request_tenant
from src.config import get_settings
from src.exceptions import VideoRecorderError
from src.services.recording.artifact_fetcher import ArtifactBatchError, ArtifactFetcher
from src.services.recording.artifacts import artifact_media_type
from src.services.recording.job_manager import RecordingJobManager as JobManager
from src.services.recording.recorder_service import RecorderService
from src.services.recording.templates import (
    OVERLAY_SAFE_TEMPLATES,
    TemplateNotFoundError,
    get_template,
)
from src.services.storage_service import StorageService
from src.utils.error_formatting import format_error

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[None]] = set()  # prevent GC of fire-and-forget tasks

# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Until now this service authenticated the licence token for tenant IDENTITY
# (AuthMiddleware answers "who are you") and never asked "may you" — a tenant
# whose licence is expired or revoked still carries a syntactically valid token
# and was served every recording route below. LICENSE_GATE closes that:
# none/expired/revoked -> 402 license_<state>, an unestablishable verdict -> 503
# license_unavailable + Retry-After, active/expiring_soon -> pass.
#
# Every route under /api/record is gated: job creation (part/content/url/batch/
# animation), job status/result/video/thumbnail reads, and cancellation are all
# billable, tenant-scoped usage of this service — exactly what the owner's rule
# ("any usage of w2v ... is only allowed based on a license, measured and
# billed") is about. There is no probe or peer-callback route under this prefix.
#
# DORMANT ON MERGE: require_license is a no-op unless W2V_LICENSE_GATE_ENABLED
# is true, read per-call from os.environ. It defaults to False, so this change
# alters no behaviour until gitops arms the flag. Preserve that default.
router = APIRouter(prefix="/api/record", tags=["record"], dependencies=list(LICENSE_GATE))


def _task_done_callback(task: asyncio.Task[None]) -> None:
    """Log unhandled exceptions from fire-and-forget background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Background task %s failed: %s", task.get_name(), exc, exc_info=exc)


def _get_tenant_id(request: Request) -> uuid.UUID:
    """Return the caller's tenant UUID, or raise 401. No "anonymous" fallback."""
    return require_request_tenant(request)


def _get_recorder_service(request: Request) -> RecorderService:
    """Retrieve the RecorderService singleton from app state."""
    return request.app.state.recorder_service


def _get_storage(request: Request) -> StorageService:
    """Retrieve the StorageService singleton from app state."""
    return request.app.state.storage_service


def get_artifact_client(request: Request | None = None) -> ArtifactClient:
    """Return the per-process ArtifactClient (override-point for tests).

    When a real request is supplied the client is read from app.state so the
    single httpx.AsyncClient is reused across requests.  The fallback (no
    request, or state not yet populated) constructs a fresh one — used in
    unit tests that patch this function directly.
    """
    if request is not None:
        client = getattr(request.app.state, "artifact_client", None)
        if client is not None:
            return client  # type: ignore[return-value]
    return ArtifactClient()


# ============================================================================
# Batch dispatch helper (shared by /batch and /batch_from_artifacts)
# ============================================================================


async def _dispatch_batch_job(
    *,
    jm: JobManager,
    tenant_id: uuid.UUID,
    service: RecorderService,
    part_ids: list[str],
    scripts_dir: str,
    recording_config: dict | None,
    auth_config: dict | None,
    webhook_url: str | None,
    execution_mode: str,
    failure_policy: str,
    priority: int,
) -> tuple[str, dict[str, str]]:
    """Create per-part jobs, commit, and spawn background recorder tasks.

    Returns (batch_id, {part_id: job_id}).
    """
    batch_id = str(uuid.uuid7())
    job_ids: dict[str, str] = {}

    for idx, part_id in enumerate(part_ids):
        job_id = await service.create_job(
            jm,
            tenant_id=tenant_id,
            part_id=part_id,
            scripts_dir=scripts_dir,
            recording_config=recording_config,
            auth_config=auth_config,
            webhook_url=webhook_url,
            batch_id=batch_id,
            priority=priority,
            batch_order=idx,
        )
        job_ids[part_id] = job_id

    await jm.commit()

    if execution_mode == "sequential":
        task = asyncio.create_task(
            _process_batch_sequential(service, list(job_ids.values()), tenant_id, failure_policy)
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
        task.add_done_callback(_task_done_callback)
    else:
        for job_id in job_ids.values():
            # Plain create_task snapshots the current context → bg task inherits
            # identity + correlation ContextVars; do NOT pass a blank context=
            # (see record_part for the full contract).
            task = asyncio.create_task(_process_job_bg(service, job_id, tenant_id))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            task.add_done_callback(_task_done_callback)

    return batch_id, job_ids


# ============================================================================
# Recording Endpoints
# ============================================================================


@router.post(
    "/part",
    operation_id="record_video_part",
    responses={
        422: {"description": "Validation error"},
    },
)
async def record_part(
    body: RecordPartRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> AsyncJobResponse:
    """Record a single part (async).

    Args:
        body: Recording request with part_id, scripts_dir, and optional config.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        AsyncJobResponse with job_id and poll URL.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)
    logger.info("POST /part received: part_id=%s tenant=%s", body.part_id, tenant_id)

    job_id = await service.create_job(
        jm,
        tenant_id=tenant_id,
        part_id=body.part_id,
        part_type=body.part_type,
        scripts_dir=body.scripts_dir,
        recording_config=body.recording_config.model_dump() if body.recording_config else None,
        auth_config=body.auth_config.model_dump() if body.auth_config else None,
        webhook_url=body.webhook_url,
    )
    await jm.commit()

    # Background task creates its own DB session — do NOT pass request-scoped jm.
    #
    # The detached background task inherits this request's identity + cost-correlation
    # ContextVars (tenant/license/subject/correlation_id/stage, including the resolved
    # X-Effective-Tenant-Id) because a plain asyncio.create_task(coro) snapshots the
    # current contextvars context at creation time. That snapshot is what makes the
    # task's WM/IR cost events attribute to the run's tenant with the right
    # correlation_id. Do NOT move this work to a thread/process pool or
    # run_in_executor, and do NOT pass a blank context= — either would drop the
    # context and silently NULL the correlation. (Regression test:
    # tests/unit/recording/test_routes.py::TestRecordPartCorrelationPropagation.)
    task = asyncio.create_task(_process_job_bg(service, job_id, tenant_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_task_done_callback)

    logger.info("POST /part completed: job_id=%s part_id=%s", job_id, body.part_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/record/jobs/{job_id}",
    )


@router.post(
    "/content",
    operation_id="record_html_content",
    responses={
        422: {"description": "Validation error"},
    },
)
async def record_content(
    body: RecordContentRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> AsyncJobResponse:
    """Record scrolling HTML content as a video (async).

    Serves the HTML via a temporary server, records Playwright scrolling
    through it, and produces a video file.

    Args:
        body: Content recording request with HTML, part_id, and optional config.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        AsyncJobResponse with job_id and poll URL.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)
    logger.info("POST /content received: part_id=%s tenant=%s", body.part_id, tenant_id)

    scroll_config_dict = body.scroll_config.model_dump() if body.scroll_config else None
    recording_config_dict = body.recording_config.model_dump() if body.recording_config else None

    # Store scroll config inside recording_config JSONB
    merged_config = recording_config_dict or {}
    if scroll_config_dict:
        merged_config["scroll_config"] = scroll_config_dict

    job_id = await service.create_content_job(
        jm,
        tenant_id=tenant_id,
        part_id=body.part_id,
        html_content=body.html_content,
        recording_config=merged_config or None,
        webhook_url=body.webhook_url,
    )
    await jm.commit()

    # Plain create_task snapshots the current context → bg task inherits identity +
    # correlation ContextVars; do NOT pass a blank context= (see record_part above).
    task = asyncio.create_task(_process_content_job_bg(service, job_id, tenant_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_task_done_callback)

    logger.info("POST /content completed: job_id=%s part_id=%s", job_id, body.part_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/record/jobs/{job_id}",
    )


async def _process_content_job_bg(
    service: RecorderService, job_id: str, tenant_id: uuid.UUID
) -> None:
    """Background task for content recording jobs."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await service.process_content_job(jm, job_id, tenant_id)
    except VideoRecorderError as exc:
        logger.error("Content recording job %s failed: %s", job_id, format_error(exc))
        await _mark_job_failed_safe(job_id, format_error(exc))
    except Exception as exc:
        logger.exception("Content recording job %s failed unexpectedly", job_id)
        await _mark_job_failed_safe(job_id, format_error(exc))


@router.post(
    "/animation",
    operation_id="record_animation",
    responses={
        400: {"description": "Unknown template_id, or transparent=true on a full-frame template"},
        422: {"description": "Validation error"},
    },
)
async def record_animation(
    body: RecordAnimationRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> AsyncJobResponse:
    """Render a branded motion-graphics template to MP4 / WebM / GIF (async).

    Loads the template (by ``template_id`` from the library, or inline ``html``),
    injects ``params`` as ``window.PARAMS``, and frame-steps the timeline for a
    deterministic capture. Poll ``poll_url`` then download via
    ``/api/record/jobs/{job_id}/video``.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)
    logger.info(
        "POST /animation received: part_id=%s template=%s tenant=%s",
        body.part_id,
        body.template_id,
        tenant_id,
    )

    if body.template_id:
        try:
            html_content = get_template(body.template_id)
        except TemplateNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # A full-frame card template paints its own opaque stage, so alpha
        # capture would produce a clip that is transparent everywhere except
        # the part the caller can see. Refuse instead of shipping that.
        if body.transparent and body.template_id not in OVERLAY_SAFE_TEMPLATES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"template '{body.template_id}' is a full-frame card and paints its own "
                    f"opaque background, so transparent=true cannot produce a usable overlay. "
                    f"Overlay-capable templates: {', '.join(sorted(OVERLAY_SAFE_TEMPLATES))} "
                    f"(or supply inline 'html' with your own transparent backdrop)."
                ),
            )
    else:
        # _require_source guarantees html is set when template_id is absent.
        html_content = body.html or ""

    # Animation params + viewport overrides ride inside recording_config JSONB.
    recording_config: dict = {
        "animation_config": {
            "params": body.params,
            "fps": body.fps,
            "duration_ms": body.duration_ms,
            # Already coerced to "webm" by the schema when transparent is set
            # with no explicit format (see _resolve_transparent_container).
            "output_format": body.output_format,
            "transparent": body.transparent,
        }
    }
    if body.width:
        recording_config["viewport_width"] = body.width
    if body.height:
        recording_config["viewport_height"] = body.height
    # Same top-level key the other routes land via RecordingConfigInput.model_dump();
    # AnimationRecorder._resolve_config reads its CRF from here. Absent ⇒ default preset.
    if body.quality_preset is not None:
        recording_config["quality_preset"] = body.quality_preset

    job_id = await service.create_animation_job(
        jm,
        tenant_id=tenant_id,
        part_id=body.part_id,
        html_content=html_content,
        recording_config=recording_config,
        webhook_url=body.webhook_url,
    )
    await jm.commit()

    task = asyncio.create_task(_process_animation_job_bg(service, job_id, tenant_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_task_done_callback)

    logger.info("POST /animation completed: job_id=%s part_id=%s", job_id, body.part_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/record/jobs/{job_id}",
    )


async def _process_animation_job_bg(
    service: RecorderService, job_id: str, tenant_id: uuid.UUID
) -> None:
    """Background task for motion-graphics animation jobs."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await service.process_animation_job(jm, job_id, tenant_id)
    except VideoRecorderError as exc:
        logger.error("Animation job %s failed: %s", job_id, format_error(exc))
        await _mark_job_failed_safe(job_id, format_error(exc))
    except Exception as exc:
        logger.exception("Animation job %s failed unexpectedly", job_id)
        await _mark_job_failed_safe(job_id, format_error(exc))


@router.post(
    "/url",
    operation_id="record_url",
    responses={
        422: {"description": "Validation error"},
    },
)
async def record_url(
    body: RecordUrlRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> AsyncJobResponse:
    """Record a live URL (e.g. an authenticated SPA) as a scrolling video (async).

    Navigates a headless Chromium to the URL — authenticating first via
    ``auth_info`` when supplied — auto-scrolls, and captures the viewport.

    Args:
        body: URL recording request with url, part_id, and optional auth/config.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        AsyncJobResponse with job_id and poll URL.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)
    logger.info("POST /url received: part_id=%s tenant=%s", body.part_id, tenant_id)

    job_id = await service.create_url_job(
        jm,
        tenant_id=tenant_id,
        part_id=body.part_id,
        url=body.url,
        auth_info=body.auth_info.model_dump(exclude_none=True) if body.auth_info else None,
        recording_config=body.recording_config.model_dump() if body.recording_config else None,
        scroll_config=body.scroll_config.model_dump() if body.scroll_config else None,
        webhook_url=body.webhook_url,
    )
    await jm.commit()

    task = asyncio.create_task(_process_url_job_bg(service, job_id, tenant_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_task_done_callback)

    logger.info("POST /url completed: job_id=%s part_id=%s", job_id, body.part_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/record/jobs/{job_id}",
    )


async def _process_url_job_bg(service: RecorderService, job_id: str, tenant_id: uuid.UUID) -> None:
    """Background task for URL recording jobs."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await service.process_url_job(jm, job_id, tenant_id)
    except VideoRecorderError as exc:
        logger.error("URL recording job %s failed: %s", job_id, format_error(exc))
        await _mark_job_failed_safe(job_id, format_error(exc))
    except Exception as exc:
        logger.exception("URL recording job %s failed unexpectedly", job_id)
        await _mark_job_failed_safe(job_id, format_error(exc))


@router.post(
    "/capture/screens",
    operation_id="capture_screens",
    responses={
        422: {"description": "Validation error"},
    },
)
async def capture_screens(
    body: CaptureScreensRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> AsyncJobResponse:
    """Capture one screenshot per demo screen in a single navigation pass (async).

    Produces a single self-describing ``.zip`` of ``<capture_id>.png`` files,
    downloaded via the shared ``GET /api/record/jobs/{job_id}/video`` endpoint.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)
    logger.info(
        "POST /capture/screens received: part_id=%s tenant=%s steps=%d",
        body.part_id,
        tenant_id,
        len(body.steps),
    )

    rec_config = body.recording_config.model_dump() if body.recording_config else {}
    if body.record_clips:
        # Stash the clip-mode flag + dwell in recording_config JSONB so
        # process_screens_job can branch to the clip recorder. _resolve_config
        # ignores unknown keys, so clip_dwell_ms rides along harmlessly.
        rec_config["record_clips"] = True
        rec_config["clip_dwell_ms"] = body.clip_dwell_ms

    job_id = await service.create_screens_job(
        jm,
        tenant_id=tenant_id,
        part_id=body.part_id,
        url=body.url,
        steps=[s.model_dump() for s in body.steps],
        auth_info=body.auth_info.model_dump(exclude_none=True) if body.auth_info else None,
        recording_config=rec_config or None,
    )
    await jm.commit()

    task = asyncio.create_task(_process_screens_job_bg(service, job_id, tenant_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_task_done_callback)

    logger.info("POST /capture/screens completed: job_id=%s part_id=%s", job_id, body.part_id)
    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/record/jobs/{job_id}",
    )


async def _process_screens_job_bg(
    service: RecorderService, job_id: str, tenant_id: uuid.UUID
) -> None:
    """Background task for per-screen capture jobs."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await service.process_screens_job(jm, job_id, tenant_id)
    except VideoRecorderError as exc:
        logger.error("Screen capture job %s failed: %s", job_id, format_error(exc))
        await _mark_job_failed_safe(job_id, format_error(exc))
    except Exception as exc:
        logger.exception("Screen capture job %s failed unexpectedly", job_id)
        await _mark_job_failed_safe(job_id, format_error(exc))


@router.post(
    "/batch",
    operation_id="record_video_batch",
    responses={
        422: {"description": "Validation error"},
    },
)
async def record_batch(
    body: RecordBatchRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> BatchJobResponse:
    """Record multiple parts (always async).

    Args:
        body: Batch request with part_ids, execution mode, and failure policy.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        BatchJobResponse with batch_id and mapping of part_id to job_id.
    """
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)

    logger.info(
        "POST /batch received: parts=%d mode=%s tenant=%s",
        len(body.part_ids),
        body.execution_mode,
        tenant_id,
    )

    batch_id, job_ids = await _dispatch_batch_job(
        jm=jm,
        tenant_id=tenant_id,
        service=service,
        part_ids=list(body.part_ids),
        scripts_dir=body.scripts_dir,
        recording_config=body.recording_config.model_dump() if body.recording_config else None,
        auth_config=body.auth_config.model_dump() if body.auth_config else None,
        webhook_url=body.webhook_url,
        execution_mode=body.execution_mode,
        failure_policy=body.failure_policy,
        priority=body.priority,
    )

    logger.info(
        "POST /batch completed: batch_id=%s jobs=%d",
        batch_id,
        len(job_ids),
    )
    return BatchJobResponse(batch_id=batch_id, job_ids=job_ids, status="pending")


async def _mark_job_failed_safe(job_id: str, error: str) -> None:
    """Persist a failure status for *job_id*, swallowing DB errors."""
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await jm.mark_failed(job_id, error)
            await jm.commit()
    except Exception:
        logger.exception("Failed to mark job %s as failed in DB", job_id)


async def _cancel_remaining_jobs(job_ids: list[str], from_index: int) -> None:
    """Cancel all jobs in *job_ids* starting at *from_index*."""
    for remaining_id in job_ids[from_index:]:
        try:
            async with get_db_session() as db:
                jm = JobManager(db)
                await jm.mark_cancelled(remaining_id)
                await jm.commit()
        except Exception:
            logger.exception("Failed to cancel remaining job %s", remaining_id)


async def _process_job_bg(service: RecorderService, job_id: str, tenant_id: uuid.UUID) -> None:
    """Background task that creates its own DB session."""
    # Phase D RLS: bind the tenant for this bg task's sessions (after_begin re-binds each txn).
    # tenant_source is OMITTED on purpose — see the module docstring. Omission keeps the
    # provenance this task inherited (create_task copies the request's context) because the
    # tenant is unchanged, and would drop it to None if the tenant ever differed. Passing
    # tenant_source= explicitly always wins and would bypass that drop.
    set_identity_context(tenant_id, None, None)
    try:
        async with get_db_session() as db:
            jm = JobManager(db)
            await service.process_job(jm, job_id, tenant_id)
    except VideoRecorderError as exc:
        logger.error("Background recording job %s failed: %s", job_id, format_error(exc))
        await _mark_job_failed_safe(job_id, format_error(exc))
    except Exception as exc:
        logger.exception("Background recording job %s failed unexpectedly", job_id)
        await _mark_job_failed_safe(job_id, format_error(exc))


async def _process_batch_sequential(
    service: RecorderService,
    job_ids: list[str],
    tenant_id: uuid.UUID,
    failure_policy: str = "continue",
) -> None:
    """Process batch jobs sequentially."""
    for idx, job_id in enumerate(job_ids):
        try:
            async with get_db_session() as db:
                jm = JobManager(db)
                await service.process_job(jm, job_id, tenant_id)
        except VideoRecorderError as exc:
            logger.error("Sequential batch job %s failed: %s", job_id, format_error(exc))
            await _mark_job_failed_safe(job_id, format_error(exc))
            if failure_policy == "fail_fast":
                await _cancel_remaining_jobs(job_ids, idx + 1)
                break
        except Exception as exc:
            logger.exception("Sequential batch job %s failed unexpectedly", job_id)
            await _mark_job_failed_safe(job_id, format_error(exc))
            if failure_policy == "fail_fast":
                await _cancel_remaining_jobs(job_ids, idx + 1)
                break


# ============================================================================
# Job Status Endpoints
# ============================================================================


@router.get("/jobs", operation_id="list_recording_jobs")
async def list_jobs(
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = None,
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
) -> PaginatedJobsResponse:
    """List jobs with pagination.

    Args:
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).
        offset: Pagination offset.
        limit: Maximum results per page (capped at 100).
        status: Optional status filter.
        sort_order: Sort direction ('asc' or 'desc').

    Returns:
        PaginatedJobsResponse with matching jobs and total count.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug(
        "GET /jobs: tenant=%s offset=%d limit=%d status=%s", tenant_id, offset, limit, status
    )
    jobs, total = await jm.list_jobs(
        tenant_id=tenant_id,
        offset=offset,
        limit=min(limit, get_settings().api.max_page_size),
        status=status,
        sort_order=sort_order,
    )
    return PaginatedJobsResponse(
        jobs=[
            JobStatusResponse(
                job_id=j.job_id,
                status=j.status,
                part_id=j.part_id,
                created_at=j.created_at.isoformat() if j.created_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
                error=j.error_message,
                result_url=f"/api/record/jobs/{j.job_id}/result"
                if j.status == "completed"
                else None,
                video_path=j.video_path if j.status == "completed" else None,
                progress=j.progress if hasattr(j, "progress") else None,
                progress_stage=j.progress_stage if hasattr(j, "progress_stage") else None,
                retry_count=j.retry_count if hasattr(j, "retry_count") else None,
                max_retries=j.max_retries if hasattr(j, "max_retries") else None,
                duration_seconds=j.duration_seconds if hasattr(j, "duration_seconds") else None,
                width=j.width if hasattr(j, "width") else None,
                height=j.height if hasattr(j, "height") else None,
                capture_width=j.capture_width if hasattr(j, "capture_width") else None,
                capture_height=j.capture_height if hasattr(j, "capture_height") else None,
                file_size_bytes=j.file_size_bytes if hasattr(j, "file_size_bytes") else None,
                has_audio=j.has_audio if hasattr(j, "has_audio") else None,
            )
            for j in jobs
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get(
    "/jobs/{job_id}",
    operation_id="get_recording_job_status",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_status(
    job_id: str,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> JobStatusResponse:
    """Get the current status of a recording job.

    Args:
        job_id: UUID of the recording job.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        JobStatusResponse with current job state.

    Raises:
        HTTPException: 404 if job not found or owned by another tenant.
    """
    logger.debug("GET /jobs/%s", job_id)
    # 404 (not 403) for another tenant's job — no enumeration oracle.
    job = await jm.get_job(job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    result_url = f"/api/record/jobs/{job_id}/result" if job.status == "completed" else None

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        part_id=job.part_id,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error=job.error_message,
        result_url=result_url,
        video_path=job.video_path if job.status == "completed" else None,
        progress=job.progress if hasattr(job, "progress") else None,
        progress_stage=job.progress_stage if hasattr(job, "progress_stage") else None,
        retry_count=job.retry_count if hasattr(job, "retry_count") else None,
        max_retries=job.max_retries if hasattr(job, "max_retries") else None,
        duration_seconds=job.duration_seconds if hasattr(job, "duration_seconds") else None,
        width=job.width if hasattr(job, "width") else None,
        height=job.height if hasattr(job, "height") else None,
        capture_width=job.capture_width if hasattr(job, "capture_width") else None,
        capture_height=job.capture_height if hasattr(job, "capture_height") else None,
        file_size_bytes=job.file_size_bytes if hasattr(job, "file_size_bytes") else None,
        has_audio=job.has_audio if hasattr(job, "has_audio") else None,
    )


@router.get(
    "/jobs/{job_id}/result",
    operation_id="get_recording_result",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job not completed yet"},
    },
)
async def get_job_result(
    job_id: str,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> RecordingResultResponse:
    """Get the recording result metadata for a completed job.

    Args:
        job_id: UUID of the recording job.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        RecordingResultResponse with video metadata.

    Raises:
        HTTPException: 404 if job not found or owned by another tenant,
            409 if not yet completed.
    """
    logger.debug("GET /jobs/%s/result", job_id)
    # 404 (not 403) for another tenant's job — no enumeration oracle.
    job = await jm.get_job(job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status: {job.status})",
            headers={"Retry-After": "30"},
        )

    return RecordingResultResponse(
        job_id=job.job_id,
        part_id=job.part_id,
        status=job.status,
        video_path=job.video_path,
        duration_seconds=job.duration_seconds,
        file_size_bytes=job.file_size_bytes,
        width=job.width,
        height=job.height,
        # The raster the recorder really captured at, as the PRODUCER resolved
        # it -- so a caller can size source assets to it without hardcoding
        # AnimationRecorder._CAPTURE_SCALE, and can see a silent clamp-down.
        capture_width=job.capture_width,
        capture_height=job.capture_height,
        has_audio=bool(job.has_audio),
        recording_time_seconds=job.recording_time_seconds,
        queue_wait_seconds=job.queue_wait_seconds,
        max_frame_seconds=job.max_frame_seconds if hasattr(job, "max_frame_seconds") else None,
        error=job.error_message,
    )


@router.get(
    "/jobs/{job_id}/video",
    operation_id="download_recording_video",
    responses={
        404: {"description": "Job or video not found"},
        410: {"description": "Video has expired"},
    },
)
async def get_job_video(
    job_id: str,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> Response:
    """Download the recorded video file.

    For local storage, returns a streaming FileResponse with range-request support.
    For MinIO storage, streams the bytes through this service (no presigned-URL
    redirect — browsers can't reach `minio.test:8389` directly, and the org
    pattern is stream-through, matching artifact_service:download_artifact).

    Args:
        job_id: UUID of the recording job.
        request: FastAPI request.
        jm: JobManager instance (injected).

    Returns:
        Video file response (FileResponse for local, Response with bytes for MinIO).

    Raises:
        HTTPException: 404 if job/video not found, 410 if video expired.
    """
    logger.info("GET /jobs/%s/video", job_id)
    t0 = time.monotonic()
    # 404 (not 403) for another tenant's job — no enumeration oracle.
    job = await jm.get_job(job_id, tenant_id=_get_tenant_id(request))
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # Check expiry
    if hasattr(job, "expired") and job.expired:
        raise HTTPException(status_code=410, detail="Video has expired and been cleaned up")

    if not job.video_path:
        raise HTTPException(status_code=404, detail="No video available")

    storage = _get_storage(request)

    # Media type comes from the KIND the producer declared and the row persisted,
    # not from a suffix table. `create_screens_job` (2 of the 5 producers) stores a
    # zip of per-screen stills/clips under `video_path` with its `.zip` suffix
    # intact, and the old suffix-only map had no "zip" key -- so every screens
    # download was served as `Content-Type: video/webm`. A NULL `artifact_kind`
    # (any row written before the column existed, or a producer that declares
    # nothing) resolves to the strict video case, exactly as before.
    _ext = job.video_path.rsplit(".", 1)[-1] if "." in job.video_path else "webm"
    media_type = artifact_media_type(getattr(job, "artifact_kind", None), _ext)
    filename = f"{job.part_id}.{_ext}"

    # Local backend: stream via FileResponse (range-request support).
    local_path = storage.get_local_path(job.video_path)
    if local_path is not None:
        logger.info(
            "GET /jobs/%s/video: local file served in %.3fs",
            job_id,
            time.monotonic() - t0,
        )
        return FileResponse(
            path=str(local_path),
            media_type=media_type,
            filename=filename,
        )

    # MinIO (or any backend without a local path): retrieve bytes and respond.
    data = await storage.retrieve(job.video_path)
    if data is None:
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    logger.info(
        "GET /jobs/%s/video: streamed from minio (%d bytes) in %.3fs",
        job_id,
        len(data),
        time.monotonic() - t0,
    )
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancel_recording_job",
    responses={
        404: {"description": "Job not found"},
        409: {"description": "Job already in terminal state"},
    },
)
async def cancel_job(
    job_id: str,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> JobStatusResponse:
    """Cancel a running or pending job.

    Args:
        job_id: UUID of the recording job.
        request: FastAPI request.
        jm: JobManager instance (injected).

    Returns:
        JobStatusResponse with cancelled status.

    Raises:
        HTTPException: 404 if job not found, 409 if already in terminal state.
    """
    logger.info("POST /jobs/%s/cancel", job_id)
    service = _get_recorder_service(request)
    tenant_id = _get_tenant_id(request)
    # 404 (not 403) for another tenant's job — no enumeration oracle.
    job = await jm.get_job(job_id, tenant_id=tenant_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail=f"Job already in terminal state: {job.status}")

    await service.cancel_job(jm, job_id, tenant_id=tenant_id)
    await jm.commit()

    logger.info("POST /jobs/%s/cancel completed", job_id)
    return JobStatusResponse(
        job_id=job.job_id,
        status="cancelled",
        part_id=job.part_id,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get(
    "/batch/{batch_id}",
    operation_id="get_recording_batch_status",
    responses={
        404: {"description": "Batch not found"},
    },
)
async def get_batch_status(
    batch_id: str,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> BatchStatusResponse:
    """Get aggregated status for a batch.

    Args:
        batch_id: UUID of the batch.
        request: FastAPI request (provides tenant context).
        jm: JobManager instance (injected).

    Returns:
        BatchStatusResponse with per-job statuses and aggregate counts.

    Raises:
        HTTPException: 404 if batch not found.
    """
    tenant_id = _get_tenant_id(request)
    logger.debug("GET /batch/%s: tenant=%s", batch_id, tenant_id)
    jobs = await jm.get_batch_jobs(batch_id, tenant_id)
    if not jobs:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    status_counts: dict[str, int] = {}
    job_responses = []
    for j in jobs:
        status = str(j.status)
        status_counts[status] = status_counts.get(status, 0) + 1
        job_responses.append(
            JobStatusResponse(
                job_id=j.job_id,
                status=j.status,
                part_id=j.part_id,
                created_at=j.created_at.isoformat() if j.created_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
                error=j.error_message,
                result_url=f"/api/record/jobs/{j.job_id}/result"
                if j.status == "completed"
                else None,
                video_path=j.video_path if j.status == "completed" else None,
                progress=j.progress if hasattr(j, "progress") else None,
                progress_stage=j.progress_stage if hasattr(j, "progress_stage") else None,
                retry_count=j.retry_count if hasattr(j, "retry_count") else None,
                max_retries=j.max_retries if hasattr(j, "max_retries") else None,
                duration_seconds=j.duration_seconds if hasattr(j, "duration_seconds") else None,
                width=j.width if hasattr(j, "width") else None,
                height=j.height if hasattr(j, "height") else None,
                capture_width=j.capture_width if hasattr(j, "capture_width") else None,
                capture_height=j.capture_height if hasattr(j, "capture_height") else None,
                file_size_bytes=j.file_size_bytes if hasattr(j, "file_size_bytes") else None,
                has_audio=j.has_audio if hasattr(j, "has_audio") else None,
            )
        )

    total = len(jobs)
    completed = status_counts.get("completed", 0)
    failed = status_counts.get("failed", 0)
    cancelled = status_counts.get("cancelled", 0)

    if completed == total:
        overall_status = "completed"
    elif failed + cancelled == total:
        overall_status = "failed"
    elif completed + failed + cancelled == total:
        overall_status = "completed_with_errors"
    else:
        overall_status = "processing"

    return BatchStatusResponse(
        batch_id=batch_id,
        total=total,
        completed=completed,
        failed=failed,
        cancelled=cancelled,
        pending=status_counts.get("pending", 0),
        processing=status_counts.get("processing", 0),
        queued=status_counts.get("queued", 0),
        status=overall_status,
        jobs=job_responses,
    )


@router.post(
    "/batch_from_artifacts",
    operation_id="record_batch_from_artifacts",
    responses={
        400: {"description": "Empty parts list or unfetchable artifact"},
        422: {"description": "Validation error"},
    },
)
async def record_batch_from_artifacts(
    body: RecordBatchFromArtifactsRequest,
    request: Request,
    jm: JobManager = Depends(get_recording_job_manager),
) -> BatchJobResponse:
    """Record a batch sourced from artifact_service.

    Downloads each (part_id, artifact_id) into
    ``${RECORDING_SCRIPTS_BASE_DIR}/<job_id>/parts/<part_id>.demo.spec.ts``
    then dispatches to the existing batch recorder.  The job_id used as the
    on-disk subdir is freshly generated here — it has no relation to the
    pipeline's job_id, which lives only as the artifact-service project_id.
    """
    settings = get_settings()
    tenant_id = _get_tenant_id(request)
    service = _get_recorder_service(request)

    on_disk_job_id = uuid.uuid7()
    artifact_client = get_artifact_client(request)
    fetcher = ArtifactFetcher(
        artifact_client=artifact_client,
        scripts_base_dir=Path(settings.recording.scripts_base_dir),
    )
    try:
        scripts_dir = await fetcher.materialize(
            job_id=on_disk_job_id,
            project_id=body.project_id,
            parts=[(p.part_id, p.artifact_id) for p in body.parts],
        )
    except httpx.HTTPStatusError as exc:
        # Artifact-service returned a 4xx/5xx (typically 404 for an unknown
        # artifact_id under the given project_id). Surface as 400 with a
        # human-readable cause instead of leaking a 500 stack to the caller.
        status = exc.response.status_code
        raise HTTPException(
            status_code=400,
            detail=(
                f"artifact_service returned {status} fetching scripts under "
                f"project_id={body.project_id}; verify (project_id, artifact_id) "
                f"pairs exist."
            ),
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"artifact_service unreachable: {exc}",
        ) from exc
    except ArtifactBatchError as exc:
        # Mixed-result batch: at least one part failed mid-flight. If every
        # failure was an artifact-service 4xx/5xx the right answer is still 400
        # (caller's input was wrong); any non-httpx failure means infra trouble.
        only_status_errors = all(isinstance(e, httpx.HTTPStatusError) for _, e in exc.failed)
        failed_ids = [pid for pid, _ in exc.failed]
        if only_status_errors:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"artifact_service returned errors for {len(failed_ids)}/"
                    f"{len(failed_ids) + len(exc.succeeded)} parts under "
                    f"project_id={body.project_id}; failed part_ids={failed_ids}."
                ),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=(f"batch materialize failed for {len(failed_ids)} parts: {exc}"),
        ) from exc

    batch_id, job_ids = await _dispatch_batch_job(
        jm=jm,
        tenant_id=tenant_id,
        service=service,
        part_ids=[p.part_id for p in body.parts],
        scripts_dir=str(scripts_dir),
        recording_config=(body.recording_config.model_dump() if body.recording_config else None),
        auth_config=body.auth_config.model_dump() if body.auth_config else None,
        webhook_url=body.webhook_url,
        execution_mode=body.execution_mode,
        failure_policy=body.failure_policy,
        priority=body.priority,
    )

    logger.info(
        "POST /batch_from_artifacts: batch_id=%s parts=%d scripts_dir=%s",
        batch_id,
        len(body.parts),
        scripts_dir,
    )
    return BatchJobResponse(
        batch_id=batch_id,
        job_ids=job_ids,
        status="pending",
        scripts_dir=str(scripts_dir),
        poll_url=f"/api/record/batch/{batch_id}",
    )
