"""Unit tests for cleanup service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# Patch at the module where it's used, not where it's defined
_PATCH_TARGET = "src.services.compilation.cleanup_service.get_settings"


class TestCleanupServiceConfig:
    @patch(_PATCH_TARGET)
    async def test_start_disabled_does_not_create_task(self, mock_settings):
        mock_settings.return_value = MagicMock(
            cleanup=MagicMock(
                enabled=False,
                interval_seconds=60,
                completed_ttl_days=30,
                failed_ttl_days=7,
            )
        )
        from src.services.compilation.cleanup_service import CleanupService

        svc = CleanupService()
        await svc.start()
        assert svc._task is None
