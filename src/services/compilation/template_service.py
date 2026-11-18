"""Compilation template management service.

Manages complete "recipes" combining preset + theme + music + title card config
into reusable compilation templates.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CompilationTemplate
from src.services.compilation.presets import get_preset

logger = logging.getLogger(__name__)

BUILTIN_TEMPLATES: dict[str, dict[str, Any]] = {
    "corporate-demo": {
        "name": "corporate-demo",
        "description": "Professional corporate product demo",
        "preset_name": "standard",
        "theme_name": "corporate",
        "transition_type": "smart",
        "crossfade_duration": 0.5,
        "title_card_config": {"duration": 5.0},
    },
    "product-walkthrough": {
        "name": "product-walkthrough",
        "description": "Step-by-step product walkthrough",
        "preset_name": "high-quality",
        "theme_name": "default",
        "transition_type": "fade",
        "crossfade_duration": 0.7,
        "title_card_config": {"duration": 4.0},
    },
    "tutorial": {
        "name": "tutorial",
        "description": "Educational tutorial format",
        "preset_name": "standard",
        "theme_name": "light",
        "transition_type": "cut",
        "crossfade_duration": 0.0,
        "title_card_config": {"duration": 3.0},
    },
    "marketing": {
        "name": "marketing",
        "description": "Marketing highlight reel",
        "preset_name": "high-quality",
        "theme_name": "creative",
        "transition_type": "fade",
        "crossfade_duration": 1.0,
        "title_card_config": {"duration": 4.0},
    },
}


class TemplateService:
    """Manages compilation templates -- complete recipes for video output."""

    @staticmethod
    async def create_template(
        db: AsyncSession,
        tenant_id: uuid.UUID,
        name: str,
        description: str | None = None,
        preset_name: str | None = None,
        theme_name: str | None = None,
        theme_config: dict | None = None,
        music_config: dict | None = None,
        title_card_config: dict | None = None,
        transition_type: str | None = None,
        crossfade_duration: float | None = None,
    ) -> CompilationTemplate:
        """Create a new compilation template in the database.

        Validates that *preset_name* refers to a known preset when provided.
        """
        if preset_name is not None and get_preset(preset_name) is None:
            raise ValueError(
                f"Unknown preset: {preset_name!r}. Provide a valid built-in preset name."
            )

        template = CompilationTemplate(
            tenant_id=tenant_id,
            name=name,
            description=description,
            preset_name=preset_name,
            theme_name=theme_name,
            theme_config=theme_config,
            music_config=music_config,
            title_card_config=title_card_config,
            transition_type=transition_type,
            crossfade_duration=crossfade_duration,
            is_builtin="false",
        )
        db.add(template)
        await db.flush()
        logger.info(
            "Created template %s (%s) for tenant %s",
            template.template_id,
            name,
            tenant_id,
        )
        return template

    @staticmethod
    async def get_template(
        db: AsyncSession,
        template_id: str,
    ) -> CompilationTemplate | None:
        """Retrieve a single template by its template_id."""
        stmt = select(CompilationTemplate).where(CompilationTemplate.template_id == template_id)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list_templates(
        db: AsyncSession,
        tenant_id: uuid.UUID,
    ) -> list[CompilationTemplate]:
        """Return all templates visible to a tenant.

        This includes the tenant's own templates plus built-in templates
        that are materialised on-the-fly so callers always see a complete
        list without requiring seed data in the database.
        """
        stmt = select(CompilationTemplate).where(CompilationTemplate.tenant_id == tenant_id)
        result = await db.execute(stmt)
        tenant_templates = list(result.scalars().all())

        # Determine which built-in names the tenant already has stored
        existing_names = {t.name for t in tenant_templates}

        # Materialise any missing built-in entries so the caller gets a
        # uniform list of CompilationTemplate objects.
        for builtin_name, cfg in BUILTIN_TEMPLATES.items():
            if builtin_name in existing_names:
                continue
            now = datetime.now(UTC)
            virtual = CompilationTemplate(
                template_id=f"builtin-{builtin_name}",
                tenant_id=tenant_id,
                name=cfg["name"],
                description=cfg.get("description"),
                preset_name=cfg.get("preset_name"),
                theme_name=cfg.get("theme_name"),
                theme_config=cfg.get("theme_config"),
                music_config=cfg.get("music_config"),
                title_card_config=cfg.get("title_card_config"),
                transition_type=cfg.get("transition_type"),
                crossfade_duration=cfg.get("crossfade_duration"),
                is_builtin="true",
                created_at=now,
                updated_at=now,
            )
            tenant_templates.append(virtual)

        return tenant_templates

    @staticmethod
    async def delete_template(
        db: AsyncSession,
        template_id: str,
        tenant_id: uuid.UUID,
    ) -> bool:
        """Delete a tenant-owned template.

        Built-in templates cannot be deleted.  Returns ``True`` when a
        row was actually removed.
        """
        # Prevent deletion of virtual built-in ids
        if template_id.startswith("builtin-"):
            logger.warning("Attempted to delete built-in template %s", template_id)
            return False

        # Ensure the template belongs to the tenant and is not built-in
        stmt = select(CompilationTemplate).where(
            CompilationTemplate.template_id == template_id,
            CompilationTemplate.tenant_id == tenant_id,
        )
        result = await db.execute(stmt)
        template = result.scalar_one_or_none()

        if template is None:
            return False

        if template.is_builtin == "true":
            logger.warning("Attempted to delete built-in template %s", template_id)
            return False

        del_stmt = delete(CompilationTemplate).where(
            CompilationTemplate.template_id == template_id,
            CompilationTemplate.tenant_id == tenant_id,
        )
        await db.execute(del_stmt)
        await db.flush()
        logger.info("Deleted template %s for tenant %s", template_id, tenant_id)
        return True

    @staticmethod
    async def resolve_template(
        db: AsyncSession,
        template_name: str,
        tenant_id: uuid.UUID,
    ) -> dict | None:
        """Resolve a template by name into a config dict.

        Lookup order:
        1. Built-in templates (by name).
        2. Tenant-owned templates in the database.

        Returns a flat dict with all configuration fields that can be
        merged directly into a ``CompileRequest``, or ``None`` when no
        template matches the given name.
        """
        # 1. Check built-in templates first
        if template_name in BUILTIN_TEMPLATES:
            cfg = BUILTIN_TEMPLATES[template_name]
            logger.debug(
                "Resolved built-in template %r for tenant %s",
                template_name,
                tenant_id,
            )
            return {
                "preset_name": cfg.get("preset_name"),
                "theme_name": cfg.get("theme_name"),
                "theme_config": cfg.get("theme_config"),
                "music_config": cfg.get("music_config"),
                "title_card_config": cfg.get("title_card_config"),
                "transition_type": cfg.get("transition_type"),
                "crossfade_duration": cfg.get("crossfade_duration"),
            }

        # 2. Fall back to tenant DB templates
        stmt = select(CompilationTemplate).where(
            CompilationTemplate.tenant_id == tenant_id,
            CompilationTemplate.name == template_name,
        )
        result = await db.execute(stmt)
        template = result.scalar_one_or_none()

        if template is None:
            logger.debug(
                "Template %r not found for tenant %s",
                template_name,
                tenant_id,
            )
            return None

        logger.debug(
            "Resolved DB template %r (%s) for tenant %s",
            template_name,
            template.template_id,
            tenant_id,
        )
        return {
            "preset_name": template.preset_name,
            "theme_name": template.theme_name,
            "theme_config": template.theme_config,
            "music_config": template.music_config,
            "title_card_config": template.title_card_config,
            "transition_type": template.transition_type,
            "crossfade_duration": template.crossfade_duration,
        }
