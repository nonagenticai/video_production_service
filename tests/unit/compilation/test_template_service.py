"""Unit tests for compilation template service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import CompilationTemplate
from src.services.compilation.template_service import BUILTIN_TEMPLATES, TemplateService
from tests.conftest import TENANT_U2

# ---------------------------------------------------------------------------
# TestBuiltinTemplates
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    """Tests for the BUILTIN_TEMPLATES constant."""

    def test_has_expected_keys(self):
        """BUILTIN_TEMPLATES contains exactly the four expected entries."""
        expected = {"corporate-demo", "product-walkthrough", "tutorial", "marketing"}
        assert set(BUILTIN_TEMPLATES.keys()) == expected
        assert len(BUILTIN_TEMPLATES) == 4


# ---------------------------------------------------------------------------
# TestCreateTemplate
# ---------------------------------------------------------------------------


class TestCreateTemplate:
    """Tests for TemplateService.create_template."""

    @patch("src.services.compilation.template_service.get_preset")
    async def test_valid_preset_creates_template(self, mock_get_preset):
        """A valid preset name results in a template being added and flushed."""
        mock_get_preset.return_value = MagicMock()  # preset exists
        mock_db = AsyncMock(spec=AsyncSession)

        template = await TemplateService.create_template(
            db=mock_db,
            tenant_id=TENANT_U2,
            name="my-template",
            preset_name="standard",
            description="A test template",
        )

        assert isinstance(template, CompilationTemplate)
        assert template.tenant_id == TENANT_U2
        assert template.name == "my-template"
        assert template.preset_name == "standard"
        assert template.is_builtin == "false"
        mock_db.add.assert_called_once_with(template)
        mock_db.flush.assert_awaited_once()

    @patch("src.services.compilation.template_service.get_preset")
    async def test_invalid_preset_raises_value_error(self, mock_get_preset):
        """An unknown preset name raises ValueError."""
        mock_get_preset.return_value = None
        mock_db = AsyncMock(spec=AsyncSession)

        with pytest.raises(ValueError, match="Unknown preset"):
            await TemplateService.create_template(
                db=mock_db,
                tenant_id=TENANT_U2,
                name="bad-template",
                preset_name="nonexistent",
            )


# ---------------------------------------------------------------------------
# TestGetTemplate
# ---------------------------------------------------------------------------


class TestGetTemplate:
    """Tests for TemplateService.get_template."""

    async def test_found_returns_template(self):
        """When a template is found, it is returned."""
        fake_template = CompilationTemplate(template_id="tpl-1", tenant_id=TENANT_U2, name="found")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_template
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.get_template(mock_db, "tpl-1")

        assert result is fake_template
        mock_db.execute.assert_awaited_once()

    async def test_not_found_returns_none(self):
        """When no template matches, None is returned."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.get_template(mock_db, "nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# TestListTemplates
# ---------------------------------------------------------------------------


class TestListTemplates:
    """Tests for TemplateService.list_templates."""

    async def test_returns_tenant_plus_builtin_templates(self):
        """Result includes tenant-owned templates plus materialised builtins."""
        tenant_template = CompilationTemplate(
            template_id="tpl-custom", tenant_id=TENANT_U2, name="custom-one"
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [tenant_template]
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        templates = await TemplateService.list_templates(mock_db, TENANT_U2)

        names = {t.name for t in templates}
        # Should contain the tenant template plus all 4 builtins
        assert "custom-one" in names
        for builtin_name in BUILTIN_TEMPLATES:
            assert builtin_name in names
        assert len(templates) == 1 + len(BUILTIN_TEMPLATES)

    async def test_no_duplication_when_tenant_has_builtin_name(self):
        """If a tenant already has a template with a builtin name, it is not duplicated."""
        # Tenant has overridden "tutorial"
        tenant_template = CompilationTemplate(
            template_id="tpl-override", tenant_id=TENANT_U2, name="tutorial"
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [tenant_template]
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        templates = await TemplateService.list_templates(mock_db, TENANT_U2)

        tutorial_count = sum(1 for t in templates if t.name == "tutorial")
        assert tutorial_count == 1
        # Total = 1 tenant template + 3 remaining builtins (tutorial already present)
        assert len(templates) == 1 + (len(BUILTIN_TEMPLATES) - 1)


# ---------------------------------------------------------------------------
# TestDeleteTemplate
# ---------------------------------------------------------------------------


class TestDeleteTemplate:
    """Tests for TemplateService.delete_template."""

    async def test_deletes_tenant_template_successfully(self):
        """A normal tenant-owned template is deleted and True is returned."""
        fake_template = CompilationTemplate(
            template_id="tpl-del", tenant_id=TENANT_U2, name="to-delete", is_builtin="false"
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_template
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.delete_template(mock_db, "tpl-del", TENANT_U2)

        assert result is True
        # db.execute is called for the select, then the delete, then flush
        assert mock_db.execute.await_count == 2
        mock_db.flush.assert_awaited_once()

    async def test_refuses_builtin_prefix(self):
        """Template IDs starting with 'builtin-' are refused and return False."""
        mock_db = AsyncMock(spec=AsyncSession)

        result = await TemplateService.delete_template(mock_db, "builtin-tutorial", TENANT_U2)

        assert result is False
        mock_db.execute.assert_not_awaited()

    async def test_refuses_is_builtin_true(self):
        """Templates with is_builtin='true' are refused even with a normal ID."""
        fake_template = CompilationTemplate(
            template_id="tpl-sneaky", tenant_id=TENANT_U2, name="sneaky", is_builtin="true"
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_template
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.delete_template(mock_db, "tpl-sneaky", TENANT_U2)

        assert result is False

    async def test_not_found_returns_false(self):
        """When no template matches the given ID and tenant, False is returned."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.delete_template(mock_db, "nonexistent", TENANT_U2)

        assert result is False


# ---------------------------------------------------------------------------
# TestResolveTemplate
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    """Tests for TemplateService.resolve_template."""

    async def test_resolves_builtin_returns_config_dict(self):
        """A built-in template name resolves to its config dict."""
        mock_db = AsyncMock(spec=AsyncSession)

        result = await TemplateService.resolve_template(mock_db, "corporate-demo", TENANT_U2)

        assert result is not None
        assert result["preset_name"] == "standard"
        assert result["theme_name"] == "corporate"
        assert result["transition_type"] == "smart"
        # Should not have queried the DB
        mock_db.execute.assert_not_awaited()

    async def test_resolves_db_template_returns_config_dict(self):
        """A tenant-owned DB template resolves to its config dict."""
        fake_template = CompilationTemplate(
            template_id="tpl-db",
            tenant_id=TENANT_U2,
            name="my-custom",
            preset_name="high-quality",
            theme_name="dark",
            theme_config={"bg": "#000"},
            music_config=None,
            title_card_config={"duration": 3.0},
            transition_type="fade",
            crossfade_duration=0.8,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fake_template
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.resolve_template(mock_db, "my-custom", TENANT_U2)

        assert result is not None
        assert result["preset_name"] == "high-quality"
        assert result["theme_name"] == "dark"
        assert result["transition_type"] == "fade"
        assert result["crossfade_duration"] == 0.8

    async def test_unknown_returns_none(self):
        """An unknown template name that is not builtin or in DB returns None."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute.return_value = mock_result

        result = await TemplateService.resolve_template(mock_db, "does-not-exist", TENANT_U2)

        assert result is None
