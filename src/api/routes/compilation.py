"""API routes for video compilation.

P3.3 split:

- :mod:`src.api.routes._compilation_helpers` -- shared deps + bg-task wrappers.
- :mod:`src.api.routes._compilation_presets` -- theme/preset/template handlers.
- :mod:`src.api.routes._compilation_jobs` -- list/cancel/status/result/video/
  thumbnail/report + WebSocket handlers.
- :mod:`src.api.routes._compilation_extras` -- title-card/edit-plan/findings/
  batch/audio/preview/cleanup handlers.

This module owns the ``router`` and the core compile-job creation endpoint.
Handler functions from helper modules are re-exported here so the historical
import surface ``from src.api.routes.compilation import <handler>`` continues
to work unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE, LICENSE_GATE_WS
from w2v_common.database.session import get_db, get_db_session

from src.api.routes._compilation_extras import (
    audio_export,
    batch_compile,
    create_findings_video,
    generate_edit_plan,
    generate_preview,
    generate_title_card,
    get_batch_status,
    on_demand_cleanup,
)
from src.api.routes._compilation_extras import register as _register_extras
from src.api.routes._compilation_helpers import (
    _background_tasks,
    _get_compiler_service,
    _get_storage,
    _get_tenant_id,
    _process_batch_bg,
    _process_job_bg,
)
from src.api.routes._compilation_jobs import (
    cancel_job,
    generate_thumbnail,
    get_compilation_report,
    get_job_result,
    get_job_status,
    get_job_thumbnail,
    get_job_video,
    get_job_video_url,
    job_progress_ws,
    list_jobs,
)
from src.api.routes._compilation_jobs import register as _register_jobs
from src.api.routes._compilation_jobs import register_ws as _register_jobs_ws
from src.api.routes._compilation_presets import (
    create_custom_preset,
    create_template,
    delete_custom_preset,
    delete_template,
    get_preset,
    get_theme,
    list_custom_presets,
    list_presets,
    list_templates,
    list_themes,
)
from src.api.routes._compilation_presets import register as _register_presets
from src.api.schemas.compilation import (
    AsyncJobResponse,
    CompileRequest,
)
from src.api.ws_identity import stamp_ws_identity
from src.services.compilation.compiler_service import CompilerService

# module's namespace.
from src.services.compilation.job_manager import CompilationJobManager  # noqa: F401
from src.services.ffmpeg import FFmpegOps

logger = logging.getLogger(__name__)


__all__ = [
    "CompilerService",
    "FFmpegOps",
    "_background_tasks",
    "_get_compiler_service",
    "_get_storage",
    "_get_tenant_id",
    "_process_batch_bg",
    "_process_job_bg",
    "asyncio",
    "audio_export",
    "batch_compile",
    "cancel_job",
    "compile_video",
    "create_custom_preset",
    "create_findings_video",
    "create_template",
    "delete_custom_preset",
    "delete_template",
    "generate_edit_plan",
    "generate_preview",
    "generate_thumbnail",
    "generate_title_card",
    "get_batch_status",
    "get_compilation_report",
    "get_db_session",
    "get_job_result",
    "get_job_status",
    "get_job_thumbnail",
    "get_job_video",
    "get_job_video_url",
    "get_preset",
    "get_theme",
    "job_progress_ws",
    "list_custom_presets",
    "list_jobs",
    "list_presets",
    "list_templates",
    "list_themes",
    "on_demand_cleanup",
    "router",
    "router_ws",
    "tempfile",
]


# ENTITLEMENT GATE (w2v_common.auth.entitlement, >=0.75.0), mounted router-wide.
# Same rationale as src/api/routes/recording.py: AuthMiddleware only answers
# "who are you", never "may you", so a tenant whose licence is expired or
# revoked still carried a syntactically valid token onto every compile route.
# LICENSE_GATE closes that: none/expired/revoked -> 402 license_<state>, an
# unestablishable verdict -> 503 license_unavailable + Retry-After,
# active/expiring_soon -> pass.
#
# Every route under /api/compile is gated (compile-job creation, batch/list/
# status/result/video/thumbnail/report reads, cancellation, title-card/edit-
# plan/preview/audio generation, theme/preset/template catalogue and CRUD, and
# the on-demand cleanup admin op): all of it is billable, tenant-scoped usage
# of this service, which is exactly what the owner's rule ("any usage of w2v
# ... is only allowed based on a license, measured and billed") is about.
# There is no probe or peer-callback route under this prefix.
#
# DORMANT ON MERGE: require_license is a no-op unless W2V_LICENSE_GATE_ENABLED
# is true, read per-call from os.environ. It defaults to False, so this change
# alters no behaviour until gitops arms the flag. Preserve that default.
router = APIRouter(prefix="/api/compile", tags=["compile"], dependencies=list(LICENSE_GATE))


# THE WEBSOCKET ROUTE GETS ITS OWN ROUTER, AND THIS IS NOT COSMETIC.
#
# `/api/compile/jobs/{job_id}/ws` used to be registered on `router` above, so it
# inherited the router-wide HTTP `LICENSE_GATE`. `require_license` is annotated
# `(request: Request)`; on a WebSocket route FastAPI hands the dependency a
# `WebSocket`, the `Request`-annotated parameter is therefore never filled, and
# the call raises
#     TypeError: require_license() missing 1 required positional argument: 'request'
# on EVERY handshake — armed AND dormant. Measured on origin/main, not inferred:
# see `tests/unit/test_ws_license_gate.py::test_the_http_gate_is_unmountable_on_a_websocket_route`.
# So the socket was not "gated": it was a 500 for everyone, entitled or not.
#
# There is no way to keep it on `router` and fix it. FastAPI composes a route's
# dependencies as `router.dependencies + route.dependencies`, and `include_router`
# does the same — so a router-level HTTP gate lands on a WS child no matter how
# the child is nested. A second router carrying `LICENSE_GATE_WS` instead is the
# only mount that expresses "this path is gated, by the WS mechanism".
#
# ⚠️ AND A GATE ALONE IS NOT ENOUGH — IT NEEDS AN IDENTITY STAMPED FIRST.
# `AuthMiddleware` is a `BaseHTTPMiddleware`; Starlette runs those on `http`
# scope ONLY, so nothing in the fleet auth stack stamps `websocket.state.tenant_id`
# on a handshake. `LICENSE_GATE_WS` mounted on its own therefore resolves NO
# tenant, derives `none`, and closes 1008 `license_none` for EVERY caller —
# including a fully entitled one. That is not "fail closed", it is a total
# outage of the socket, and this service is armed in production. So the router
# carries TWO dependencies, in this order and only this order:
#
#   1. `stamp_ws_identity` — reads the licence token off the handshake,
#      INTROSPECTS it, and stamps the tenant/subject the licences service
#      asserts. Never a value the caller typed into a header. It never closes.
#   2. `LICENSE_GATE_WS` — reads that stamped state and renders the verdict.
#
# FastAPI solves a route's dependencies sequentially in list order, and writes to
# `websocket.state` are visible to every later dependency and to the handler
# (measured, not assumed — see `tests/unit/test_ws_license_gate.py`). "Resolve ->
# stamp -> gate" is expressed by this list's ORDER and by nothing else; reversing
# it restores the outage silently, with every test still green except the
# entitled-caller one.
#
# The refusal path is deliberately the GATE's, not the identity step's: an
# unresolvable handshake simply arrives unstamped and gets the same 1008
# `license_<state>` close an unentitled tenant gets, and it still connects while
# `W2V_LICENSE_GATE_ENABLED` is unset. An identity step that closed on its own
# would invent a second refusal shape for the same condition AND quietly revoke
# the flag's dormancy.
#
# This mirrors `tts_service/src/api/ws.py` (PR #116, merged and running), which
# resolves identity and only then calls `require_license_ws`; the difference is
# purely where the call sits (a preceding dependency here, an inline call there),
# because this route's gate is router-mounted. See `src/api/ws_identity.py` for
# the header precedence, the JWT-shape exclusion, and the lanes refused.
#
# What was rejected, so the next auditor does not re-derive it:
#  - Reading a bare `x-tenant-id: <uuid>` header as a tenant: that is a
#    header-spoofable identity lane the HTTP gate does not have. A gate that
#    trusts a forgeable tenant is worse than one that closes
#    (`w2v_common/auth/entitlement.py`). Only the `Bearer <token>` CREDENTIAL
#    shape of that header is read, and only to be introspected.
#  - A `?license=<token>` query parameter: a credential in the query string lands
#    in access logs and proxy history, and this socket has ZERO browser consumers
#    (the frontend's progress transports are SSE and polling), so the one
#    argument for it does not apply here. `tts_service` refuses it for the same
#    reason. A short-lived single-use ticket is the answer if that ever changes.
#  - Deriving the tenant from the job row (`CompilationJob.tenant_id`): it
#    answers "is the JOB OWNER entitled", never "may YOU" — the caller stays
#    unauthenticated — and it is circular under `TENANT_RLS_MODE=enforce`, where
#    reading that row needs the `app.current_tenant` GUC that the row was going
#    to supply (`src/database/rls_events.py`).
router_ws = APIRouter(
    prefix="/api/compile",
    tags=["compile"],
    dependencies=[Depends(stamp_ws_identity), *LICENSE_GATE_WS],
)


# ============================================================================
# Compilation Endpoint (core)
# ============================================================================


@router.post(
    "",
    operation_id="start_video_compilation",
    responses={
        400: {"description": "Template not found"},
        422: {"description": "Validation error"},
    },
)
async def compile_video(
    body: CompileRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> AsyncJobResponse:
    """Compile recordings into a final video (async)."""
    tenant_id = _get_tenant_id(request)
    logger.info(
        "Compile request received: tenant=%s, recordings=%d, preset=%s",
        tenant_id,
        len(body.recordings),
        body.preset,
    )
    service = _get_compiler_service(request)

    # WP-5: Resolve template and apply defaults before explicit overrides
    if body.template:
        from src.services.compilation.template_service import TemplateService

        tpl_cfg = await TemplateService.resolve_template(db, body.template, tenant_id)
        if tpl_cfg is None:
            raise HTTPException(
                status_code=400,
                detail=f"Template not found: {body.template}",
            )
        # Apply template values as defaults -- explicit request fields take precedence
        if tpl_cfg.get("preset_name") and body.preset is None:
            body.preset = tpl_cfg["preset_name"]
        if tpl_cfg.get("theme_name") and body.theme_name is None:
            body.theme_name = tpl_cfg["theme_name"]
        if tpl_cfg.get("transition_type") and body.transition_type is None:
            body.transition_type = tpl_cfg["transition_type"]
        if tpl_cfg.get("crossfade_duration") is not None and body.crossfade_duration is None:
            body.crossfade_duration = tpl_cfg["crossfade_duration"]

    # Resolve theme_name: prefer top-level, fallback to theme.theme_name
    theme_name = body.theme_name
    if not theme_name and body.theme:
        theme_name = body.theme.theme_name

    job_id = await service.create_job(
        db,
        tenant_id=tenant_id,
        shot_list=body.shot_list.model_dump(),
        recordings=[r.model_dump() for r in body.recordings],
        narration_audio_dir=body.narration_audio_dir,
        clips_dir=body.clips_dir,
        transition_type=body.transition_type,
        crossfade_duration=body.crossfade_duration,
        xfade_transition=body.xfade_transition,
        target_width=body.target_width,
        target_height=body.target_height,
        crf=body.crf,
        theme_name=theme_name,
        theme_config=body.theme.model_dump(exclude={"theme_name"}) if body.theme else None,
        edit_decisions=[ed.model_dump() for ed in body.edit_decisions]
        if body.edit_decisions
        else None,
        preset=body.preset,
        webhook_url=body.webhook_url,
        webhook_secret=body.webhook_secret,
        color_correction=body.color_correction.model_dump() if body.color_correction else None,
        watermark=body.watermark.model_dump() if body.watermark else None,
        subtitles=body.subtitles.model_dump() if body.subtitles else None,
    )
    await db.commit()

    task = asyncio.create_task(_process_job_bg(service, job_id, tenant_id))
    service.register_task(job_id, task)

    logger.info("Compile job created: job_id=%s, tenant=%s", job_id, tenant_id)

    return AsyncJobResponse(
        job_id=job_id,
        status="pending",
        poll_url=f"/api/compile/jobs/{job_id}",
    )


# ============================================================================
# Register routes from helper modules
# ============================================================================

_register_jobs(router)
_register_presets(router)
_register_extras(router)
_register_jobs_ws(router_ws)
