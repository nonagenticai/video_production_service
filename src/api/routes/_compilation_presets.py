"""Theme + preset + template route registration.

Defines handler functions and registers them on a shared ``APIRouter`` passed
in from ``compilation.py`` via :func:`register`. Handlers are kept at module
scope so tests can import them via either path.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.database.session import get_db

from src.api.roles import require_tenant_admin
from src.api.routes._compilation_helpers import _get_tenant_id
from src.api.schemas.compilation import (
    CustomPresetCreateRequest,
    CustomPresetListResponse,
    CustomPresetResponse,
    PresetListResponse,
    PresetResponse,
    TemplateCreateRequest,
    TemplateListResponse,
    TemplateResponse,
    ThemeListResponse,
    ThemePresetResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Theme handlers
# ---------------------------------------------------------------------------


async def list_themes() -> ThemeListResponse:
    """List all available theme presets."""
    from src.services.compilation.themes import list_themes as _list_themes

    presets = _list_themes()
    return ThemeListResponse(
        themes=[
            ThemePresetResponse(
                name=t.name,
                primary_color=t.primary_color,
                background_color=t.background_color,
                text_color=t.text_color,
                accent_color=t.accent_color,
                subtitle_color=t.subtitle_color,
                gradient_start=t.gradient_start,
                gradient_end=t.gradient_end,
            )
            for t in presets
        ]
    )


async def get_theme(name: str) -> ThemePresetResponse:
    """Get a single theme preset by name."""
    from src.services.compilation.themes import THEME_PRESETS

    if name not in THEME_PRESETS:
        raise HTTPException(status_code=404, detail=f"Theme not found: {name}")
    t = THEME_PRESETS[name]
    return ThemePresetResponse(
        name=t.name,
        primary_color=t.primary_color,
        background_color=t.background_color,
        text_color=t.text_color,
        accent_color=t.accent_color,
        subtitle_color=t.subtitle_color,
        gradient_start=t.gradient_start,
        gradient_end=t.gradient_end,
    )


# ---------------------------------------------------------------------------
# Preset handlers
# ---------------------------------------------------------------------------


async def list_presets() -> PresetListResponse:
    """List all available compilation presets."""
    from src.services.compilation.presets import list_presets as _list_presets

    presets = _list_presets()
    return PresetListResponse(
        presets=[
            PresetResponse(
                name=p.name,
                description=p.description,
                target_width=p.target_width,
                target_height=p.target_height,
                crf=p.crf,
                fps=p.fps,
                transition_type=p.transition_type,
                crossfade_duration=p.crossfade_duration,
            )
            for p in presets
        ]
    )


async def list_custom_presets(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CustomPresetListResponse:
    """List custom presets for the current tenant."""
    from sqlalchemy import select

    from src.database.models import CustomPreset

    tenant_id = _get_tenant_id(request)
    result = await db.execute(select(CustomPreset).where(CustomPreset.tenant_id == tenant_id))
    presets = result.scalars().all()
    return CustomPresetListResponse(
        presets=[
            CustomPresetResponse(
                id=p.preset_id,
                name=p.name,
                description=p.description or "",
                extends=p.extends,
                target_width=p.target_width,
                target_height=p.target_height,
                crf=p.crf,
                fps=p.fps,
                transition_type=p.transition_type,
                crossfade_duration=p.crossfade_duration,
                created_at=p.created_at.isoformat() if p.created_at else None,
            )
            for p in presets
        ]
    )


async def get_preset(name: str) -> PresetResponse:
    """Get a single compilation preset by name."""
    from src.services.compilation.presets import get_preset as _get_preset

    preset = _get_preset(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Preset not found: {name}")
    return PresetResponse(
        name=preset.name,
        description=preset.description,
        target_width=preset.target_width,
        target_height=preset.target_height,
        crf=preset.crf,
        fps=preset.fps,
        transition_type=preset.transition_type,
        crossfade_duration=preset.crossfade_duration,
    )


async def create_custom_preset(
    body: CustomPresetCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> CustomPresetResponse:
    """Create a custom compilation preset."""
    from src.services.compilation.presets import get_preset

    tenant_id = _get_tenant_id(request)

    # Validate extends reference
    if body.extends and get_preset(body.extends) is None:
        raise HTTPException(status_code=400, detail=f"Unknown base preset: {body.extends}")

    # Validate CRF range
    if body.crf is not None and not (0 <= body.crf <= 51):
        raise HTTPException(status_code=400, detail="CRF must be between 0 and 51")

    import uuid

    from src.database.models import CustomPreset

    preset = CustomPreset(
        preset_id=str(uuid.uuid7()),
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        extends=body.extends,
        target_width=body.target_width,
        target_height=body.target_height,
        crf=body.crf,
        fps=body.fps,
        transition_type=body.transition_type,
        crossfade_duration=body.crossfade_duration,
    )
    db.add(preset)
    await db.commit()

    return CustomPresetResponse(
        id=preset.preset_id,
        name=preset.name,
        description=preset.description or "",
        extends=preset.extends,
        target_width=preset.target_width,
        target_height=preset.target_height,
        crf=preset.crf,
        fps=preset.fps,
        transition_type=preset.transition_type,
        crossfade_duration=preset.crossfade_duration,
    )


async def delete_custom_preset(
    preset_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a custom preset."""
    from sqlalchemy import delete as sa_delete
    from sqlalchemy import select

    from src.database.models import CustomPreset

    tenant_id = _get_tenant_id(request)
    result = await db.execute(
        select(CustomPreset).where(
            CustomPreset.preset_id == preset_id,
            CustomPreset.tenant_id == tenant_id,
        )
    )
    preset = result.scalar_one_or_none()
    if not preset:
        raise HTTPException(status_code=404, detail="Custom preset not found")

    await db.execute(
        sa_delete(CustomPreset).where(
            CustomPreset.preset_id == preset_id,
            CustomPreset.tenant_id == tenant_id,
        )
    )
    await db.commit()
    return {"deleted": True, "preset_id": preset_id}


# ---------------------------------------------------------------------------
# Template handlers
# ---------------------------------------------------------------------------


async def create_template(
    body: TemplateCreateRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TemplateResponse:
    """Create a compilation template."""
    from src.services.compilation.template_service import TemplateService

    tenant_id = _get_tenant_id(request)
    template = await TemplateService.create_template(
        db,
        tenant_id=tenant_id,
        name=body.name,
        description=body.description,
        preset_name=body.preset_name,
        theme_name=body.theme_name,
        theme_config=body.theme_config,
        music_config=body.music_config,
        title_card_config=body.title_card_config,
        transition_type=body.transition_type,
        crossfade_duration=body.crossfade_duration,
    )
    await db.commit()
    return TemplateResponse(
        id=template.template_id,
        name=template.name,
        description=template.description or "",
        preset_name=template.preset_name,
        theme_name=template.theme_name,
        theme_config=template.theme_config,
        music_config=template.music_config,
        title_card_config=template.title_card_config,
        transition_type=template.transition_type,
        crossfade_duration=template.crossfade_duration,
    )


async def list_templates(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> TemplateListResponse:
    """List all available templates (built-in + custom)."""
    from src.services.compilation.template_service import TemplateService

    tenant_id = _get_tenant_id(request)
    templates = await TemplateService.list_templates(db, tenant_id)
    return TemplateListResponse(
        templates=[
            TemplateResponse(
                id=t.template_id,
                name=t.name,
                description=t.description or "",
                preset_name=t.preset_name,
                theme_name=t.theme_name,
                theme_config=t.theme_config,
                music_config=t.music_config,
                title_card_config=t.title_card_config,
                transition_type=t.transition_type,
                crossfade_duration=t.crossfade_duration,
                is_builtin=t.is_builtin == "true",
                created_at=t.created_at.isoformat() if t.created_at else None,
            )
            for t in templates
        ]
    )


async def delete_template(
    template_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Delete a compilation template."""
    from src.services.compilation.template_service import TemplateService

    tenant_id = _get_tenant_id(request)
    deleted = await TemplateService.delete_template(db, template_id, tenant_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found or is built-in")
    await db.commit()
    return {"deleted": True, "template_id": template_id}


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


#: Brand-kit writes (custom presets, templates) are a tenant-admin surface.
#: Reads stay open to every member of the tenant. The tenant a write lands in is
#: always the caller's own (``_get_tenant_id``), never a request field.
_ADMIN_ONLY = [Depends(require_tenant_admin)]


def register(router: APIRouter) -> None:
    """Register theme/preset/template routes onto the supplied router."""
    router.add_api_route(
        "/themes",
        list_themes,
        methods=["GET"],
        tags=["themes"],
        operation_id="list_compilation_themes",
        response_model=ThemeListResponse,
    )
    router.add_api_route(
        "/themes/{name}",
        get_theme,
        methods=["GET"],
        tags=["themes"],
        operation_id="get_compilation_theme",
        response_model=ThemePresetResponse,
        responses={404: {"description": "Theme not found"}},
    )
    router.add_api_route(
        "/presets",
        list_presets,
        methods=["GET"],
        tags=["presets"],
        operation_id="list_compilation_presets",
        response_model=PresetListResponse,
    )
    router.add_api_route(
        "/presets/custom",
        list_custom_presets,
        methods=["GET"],
        tags=["presets"],
        operation_id="list_custom_compilation_presets",
        response_model=CustomPresetListResponse,
    )
    router.add_api_route(
        "/presets/{name}",
        get_preset,
        methods=["GET"],
        tags=["presets"],
        operation_id="get_compilation_preset",
        response_model=PresetResponse,
        responses={404: {"description": "Preset not found"}},
    )
    router.add_api_route(
        "/presets/custom",
        create_custom_preset,
        methods=["POST"],
        tags=["presets"],
        operation_id="create_custom_compilation_preset",
        dependencies=_ADMIN_ONLY,
        response_model=CustomPresetResponse,
        responses={
            400: {"description": "Invalid base preset or CRF value"},
            403: {"description": "tenant_admin role required"},
            422: {"description": "Validation error"},
        },
    )
    router.add_api_route(
        "/presets/custom/{preset_id}",
        delete_custom_preset,
        methods=["DELETE"],
        tags=["presets"],
        operation_id="delete_custom_compilation_preset",
        dependencies=_ADMIN_ONLY,
        responses={
            403: {"description": "tenant_admin role required"},
            404: {"description": "Custom preset not found"},
        },
    )
    router.add_api_route(
        "/templates",
        create_template,
        methods=["POST"],
        tags=["templates"],
        operation_id="create_compilation_template",
        dependencies=_ADMIN_ONLY,
        response_model=TemplateResponse,
        responses={
            403: {"description": "tenant_admin role required"},
            422: {"description": "Validation error"},
        },
    )
    router.add_api_route(
        "/templates",
        list_templates,
        methods=["GET"],
        tags=["templates"],
        operation_id="list_compilation_templates",
        response_model=TemplateListResponse,
    )
    router.add_api_route(
        "/templates/{template_id}",
        delete_template,
        methods=["DELETE"],
        tags=["templates"],
        operation_id="delete_compilation_template",
        dependencies=_ADMIN_ONLY,
        responses={
            403: {"description": "tenant_admin role required"},
            404: {"description": "Template not found or is built-in"},
        },
    )
