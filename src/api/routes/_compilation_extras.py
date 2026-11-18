"""Title card / edit plan / findings / batch / audio / preview / cleanup routes.

Handler bodies live here; ``compilation.py`` re-exports them so tests can
``mock.patch`` symbols on ``src.api.routes.compilation`` and have the patch
take effect.  All variable lookups (``_cm()._get_storage``, ``_cm().FFmpegOps``,
``tempfile``, ``asyncio``, ``_cm().get_db_session``, ``_cm()._process_job_bg``,
``_cm()._process_batch_bg``) are resolved through the ``compilation`` module via the
late-bound :func:`_cm` accessor below.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.database.session import get_db

from src.api.schemas.compilation import (
    AsyncJobResponse,
    AudioExportRequest,
    AudioExportResponse,
    BatchCompileRequest,
    BatchStatusResponse,
    CleanupRequest,
    CleanupResponse,
    EditPlanRequest,
    EditPlanResponse,
    FindingsVideoRequest,
    PreviewRequest,
    PreviewResponse,
    TitleCardRequest,
    TitleCardResponse,
)
from src.services.compilation.job_manager import CompilationJobManager

logger = logging.getLogger(__name__)


def _cm() -> Any:
    """Late-bound accessor for the compilation module.

    Tests patch ``src.api.routes.compilation.<name>``; resolving lookups
    through the module ensures helpers see the same patched values.
    """
    from src.api.routes import compilation

    return compilation


async def generate_title_card(
    body: TitleCardRequest,
    request: Request,
) -> TitleCardResponse:
    """Generate a single title card video (sync)."""
    logger.info("Title card request received: title=%r", body.title[:50])

    from pathlib import Path

    from src.services.compilation.title_card_generator import TitleCardGenerator

    with _cm().tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "title_card.mp4"
        theme_name = body.theme.theme_name if body.theme else None
        theme_overrides = body.theme.model_dump(exclude={"theme_name"}) if body.theme else None
        gen = TitleCardGenerator(
            theme_name=theme_name,
            theme_config=theme_overrides,
        )
        gen.generate_video(
            title=body.title,
            subtitle=body.subtitle,
            chapter=body.chapter,
            output_path=output_path,
            duration=body.duration,
            width=body.width,
            height=body.height,
        )

        storage = _cm()._get_storage(request)
        import uuid

        temp_job_id = f"titlecard-{uuid.uuid7()}"
        stored_path = await storage.store_bytes(
            temp_job_id,
            output_path.read_bytes(),
            "title_card.mp4",
        )

        logger.info("Title card generated: path=%s", stored_path)

        return TitleCardResponse(
            video_path=stored_path,
            duration=body.duration,
            width=body.width,
            height=body.height,
        )


async def generate_edit_plan(
    body: EditPlanRequest,
    request: Request,
) -> EditPlanResponse:
    """Generate an AI edit plan for recordings."""
    logger.info(
        "Edit plan request received: recordings=%d, target_duration=%s",
        len(body.recordings),
        body.target_duration_minutes,
    )

    from src.services.compilation.edit_plan_service import EditPlanService

    service = EditPlanService()
    result = await service.generate_edit_plan(
        recordings=[r.model_dump() for r in body.recordings],
        shot_list=body.shot_list.model_dump(),
        target_duration_minutes=body.target_duration_minutes,
        llm_config=body.llm_config,
    )
    logger.info("Edit plan generated: decisions=%d", len(result.decisions))
    return result


async def on_demand_cleanup(
    body: CleanupRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CleanupResponse:
    """Perform on-demand job cleanup with selective filters."""
    tenant_id = _cm()._get_tenant_id(request)
    logger.info(
        "Cleanup request received: tenant=%s, dry_run=%s, status_filter=%s",
        tenant_id,
        body.dry_run,
        body.status_filter,
    )
    storage = _cm()._get_storage(request)

    jobs = await CompilationJobManager.find_jobs_by_criteria(
        db,
        tenant_id=tenant_id,
        job_ids=body.job_ids,
        status_filter=body.status_filter,
        min_age_hours=body.min_age_hours,
    )
    job_ids = [j.job_id for j in jobs]

    if body.dry_run:
        return CleanupResponse(deleted_count=len(job_ids), job_ids=job_ids, dry_run=True)

    for job in jobs:
        try:
            await storage.delete_job_dir(job.job_id)
        except OSError, ValueError, RuntimeError, TimeoutError:
            logger.warning("Failed to delete storage for job %s", job.job_id)

    deleted = await CompilationJobManager.delete_jobs(db, job_ids)
    await db.commit()

    logger.info("Cleanup completed: deleted=%d jobs", deleted)
    return CleanupResponse(deleted_count=deleted, job_ids=job_ids, dry_run=False)


async def create_findings_video(
    body: FindingsVideoRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncJobResponse:
    """Generate a video from findings/review data."""
    logger.info("Findings video request received: findings=%d", len(body.findings))

    from src.services.compilation.findings_video_service import FindingsVideoService

    tenant_id = _cm()._get_tenant_id(request)
    service = _cm()._get_compiler_service(request)

    fvs = FindingsVideoService()
    shot_list = fvs.build_shot_list(
        findings=[f.model_dump() for f in body.findings],
        title=body.title,
        subtitle=body.subtitle,
    )

    theme_name = body.theme_name
    if not theme_name and body.theme:
        theme_name = body.theme.theme_name

    job_id = await service.create_job(
        db,
        tenant_id=tenant_id,
        shot_list=shot_list,
        recordings=[],
        preset=body.preset,
        theme_name=theme_name,
        theme_config=body.theme.model_dump(exclude={"theme_name"}) if body.theme else None,
    )
    await db.commit()

    task = _cm().asyncio.create_task(_cm()._process_job_bg(service, job_id, tenant_id))
    service.register_task(job_id, task)

    logger.info("Findings video job created: job_id=%s", job_id)

    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/compile/jobs/{job_id}",
    )


async def batch_compile(
    body: BatchCompileRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> BatchStatusResponse:
    """Submit multiple compilation jobs as a batch."""
    logger.info("Batch compile request received: configs=%d", len(body.configs))

    from src.services.compilation.batch_service import BatchService

    tenant_id = _cm()._get_tenant_id(request)
    service = _cm()._get_compiler_service(request)
    batch_svc = BatchService(compiler_service=service, db_session_factory=_cm().get_db_session)

    configs = [c.model_dump() for c in body.configs]
    batch_id = await batch_svc.create_batch(
        db,
        tenant_id=tenant_id,
        configs=configs,
        max_concurrent=body.max_concurrent,
    )
    await db.commit()

    # Process batch in background -- must NOT use request-scoped db
    task = _cm().asyncio.create_task(
        _cm()._process_batch_bg(batch_svc, batch_id, tenant_id, body.max_concurrent)
    )
    _cm()._background_tasks.add(task)
    task.add_done_callback(_cm()._background_tasks.discard)

    batch = await BatchService.get_batch_status(db, batch_id)
    return BatchStatusResponse(
        batch_id=batch_id,
        status=batch.status if batch else "pending",
        total_jobs=batch.total_jobs if batch else len(configs),
        completed_jobs=0,
        failed_jobs=0,
        job_ids=(batch.job_ids or []) if batch else [],
    )


async def get_batch_status(
    batch_id: str,
    db: AsyncSession = Depends(get_db),
) -> BatchStatusResponse:
    """Get the status of a batch compilation."""
    from src.services.compilation.batch_service import BatchService

    batch = await BatchService.get_batch_status(db, batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        status=batch.status,
        total_jobs=batch.total_jobs,
        completed_jobs=batch.completed_jobs,
        failed_jobs=batch.failed_jobs,
        job_ids=batch.job_ids or [],
        errors=batch.errors,
        created_at=batch.created_at.isoformat() if batch.created_at else None,
        completed_at=batch.completed_at.isoformat() if batch.completed_at else None,
    )


async def audio_export(
    body: AudioExportRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AudioExportResponse:
    """Export audio from a compiled video."""
    logger.info("Audio export request received: job_id=%s, format=%s", body.job_id, body.format)
    job = await CompilationJobManager.get_job(db, body.job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {body.job_id}")
    if not job.output_path:
        raise HTTPException(status_code=404, detail="No video available")

    from pathlib import Path as PathLib

    video_path = PathLib(job.output_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    storage = _cm()._get_storage(request)
    job_dir = storage.get_job_dir(body.job_id)
    output_path = job_dir / f"audio_export.{body.format}"

    _cm().FFmpegOps.extract_audio(
        video_path,
        output_path,
        audio_format=body.format,
        bitrate=body.bitrate,
    )

    duration = _cm().FFmpegOps.probe_duration(output_path) if output_path.exists() else 0.0
    file_size = output_path.stat().st_size if output_path.exists() else 0

    logger.info("Audio export completed: job_id=%s, path=%s", body.job_id, output_path)

    return AudioExportResponse(
        path=str(output_path),
        format=body.format,
        duration_seconds=duration,
        file_size_bytes=file_size,
    )


async def generate_preview(
    body: PreviewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> PreviewResponse:
    """Generate a low-quality preview of a compiled video."""
    logger.info("Preview request received: job_id=%s", body.job_id)
    if body.job_id:
        job = await CompilationJobManager.get_job(db, body.job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {body.job_id}")
        if not job.output_path:
            raise HTTPException(status_code=404, detail="No video available")

        from pathlib import Path as PathLib

        video_path = PathLib(job.output_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")

        storage = _cm()._get_storage(request)
        job_dir = storage.get_job_dir(body.job_id)
        preview_path = job_dir / "preview.mp4"

        _cm().FFmpegOps.generate_preview(
            video_path,
            preview_path,
            duration=body.duration_seconds,
            width=body.width,
            height=body.height,
        )

        duration = _cm().FFmpegOps.probe_duration(preview_path) if preview_path.exists() else 0.0
        file_size = preview_path.stat().st_size if preview_path.exists() else 0

        return PreviewResponse(
            path=str(preview_path),
            duration_seconds=duration,
            width=body.width,
            height=body.height,
            file_size_bytes=file_size,
        )

    raise HTTPException(status_code=400, detail="Either job_id or recordings must be provided")


def register(router: APIRouter) -> None:
    """Register the misc/extras routes onto the supplied router."""
    router.add_api_route(
        "/title-card",
        generate_title_card,
        methods=["POST"],
        operation_id="generate_compilation_title_card",
        response_model=TitleCardResponse,
        responses={422: {"description": "Validation error"}},
    )
    router.add_api_route(
        "/edit-plan",
        generate_edit_plan,
        methods=["POST"],
        operation_id="generate_compilation_edit_plan",
        response_model=EditPlanResponse,
        responses={422: {"description": "Validation error"}},
    )
    router.add_api_route(
        "/admin/cleanup",
        on_demand_cleanup,
        methods=["POST"],
        tags=["admin"],
        operation_id="run_compilation_cleanup",
        response_model=CleanupResponse,
        responses={422: {"description": "Validation error"}},
    )
    router.add_api_route(
        "/findings-video",
        create_findings_video,
        methods=["POST"],
        tags=["compile"],
        operation_id="create_findings_video",
        response_model=AsyncJobResponse,
        responses={422: {"description": "Validation error"}},
    )
    router.add_api_route(
        "/batch",
        batch_compile,
        methods=["POST"],
        tags=["compile"],
        operation_id="start_compilation_batch",
        response_model=BatchStatusResponse,
        responses={422: {"description": "Validation error"}},
    )
    router.add_api_route(
        "/batch/{batch_id}",
        get_batch_status,
        methods=["GET"],
        tags=["compile"],
        operation_id="get_compilation_batch_status",
        response_model=BatchStatusResponse,
        responses={404: {"description": "Batch not found"}},
    )
    router.add_api_route(
        "/audio-export",
        audio_export,
        methods=["POST"],
        tags=["compile"],
        operation_id="export_compilation_audio",
        response_model=AudioExportResponse,
        responses={
            404: {"description": "Job or video not found"},
            422: {"description": "Validation error"},
        },
    )
    router.add_api_route(
        "/preview",
        generate_preview,
        methods=["POST"],
        tags=["compile"],
        operation_id="generate_compilation_preview",
        response_model=PreviewResponse,
        responses={
            400: {"description": "Missing job_id or recordings"},
            404: {"description": "Job or video not found"},
            422: {"description": "Validation error"},
        },
    )
