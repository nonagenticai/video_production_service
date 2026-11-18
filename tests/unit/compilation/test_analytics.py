"""Unit tests for analytics query methods and API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import HTTPException

from tests.conftest import TENANT_U5

# ---------------------------------------------------------------------------
# Analytics endpoint tests
# ---------------------------------------------------------------------------


class TestAnalyticsGetTenantId:
    def test_returns_tenant_from_state(self):
        from src.api.analytics import _get_tenant_id

        request = MagicMock()
        # The auth middleware stashes ``str(tenant_id)``; the helper parses it.
        request.state.tenant_id = str(TENANT_U5)
        assert _get_tenant_id(request) == TENANT_U5

    def test_refuses_when_missing_instead_of_returning_anonymous(self):
        """This test used to assert ``_get_tenant_id(...) == "anonymous"``.

        w2v-common 0.67.0 removed every tenant fallback, and ``tenant_id`` is a
        ``uuid`` column, so the substituted string could only have become a
        DataError at the driver — or, worse, one shared bucket every
        unattributed caller can read. The refusal IS the behaviour now, so the
        assertion is inverted rather than dropped.
        """
        from src.api.analytics import _get_tenant_id

        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        with pytest.raises(HTTPException) as exc:
            _get_tenant_id(request)
        assert exc.value.status_code == 401

    def test_refuses_the_nil_uuid_by_value(self):
        """A ``None``-only guard would let the removed nil sentinel straight back in."""
        from src.api.analytics import _get_tenant_id

        for spelling in (
            "00000000-0000-0000-0000-000000000000",
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
        ):
            request = MagicMock()
            request.state.tenant_id = spelling
            with pytest.raises(HTTPException) as exc:
                _get_tenant_id(request)
            assert exc.value.status_code == 401, spelling


class TestGetUsageSummary:
    async def test_returns_summary(self):
        from src.api.analytics import get_usage_summary

        mock_summary = {
            "tenant_id": "019de842-c92a-7e37-8fdd-eb0d8c5396f9",
            "compilations_started": 10,
            "compilations_completed": 8,
            "compilations_failed": 2,
            "total_output_duration_seconds": 600.0,
            "total_output_size_bytes": 5000000,
        }

        request = MagicMock()
        request.state.tenant_id = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
        db = AsyncMock()

        with patch(
            "src.api.analytics.CompilationJobManager.get_usage_summary",
            new_callable=AsyncMock,
            return_value=mock_summary,
        ):
            result = await get_usage_summary(request, db)

        assert result.tenant_id == UUID("019de842-c92a-7e37-8fdd-eb0d8c5396f9")
        assert result.compilations_started == 10


class TestGetUsageDaily:
    async def test_returns_daily_items(self):
        from src.api.analytics import get_usage_daily
        from src.api.schemas.compilation import DailyUsageItem

        mock_daily = [
            DailyUsageItem(
                date="2026-02-01",
                compilations_started=5,
                compilations_completed=4,
                compilations_failed=1,
                total_output_duration_seconds=300.0,
                total_output_size_bytes=2500000,
            )
        ]

        request = MagicMock()
        request.state.tenant_id = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"
        db = AsyncMock()

        with patch(
            "src.api.analytics.CompilationJobManager.get_usage_daily",
            new_callable=AsyncMock,
            return_value=mock_daily,
        ):
            result = await get_usage_daily(request, db)

        assert len(result.days) == 1


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestUsageSummaryResponse:
    def test_schema_defaults(self):
        from src.api.schemas.compilation import UsageSummaryResponse

        resp = UsageSummaryResponse(tenant_id=UUID("019de842-c92a-7e37-8fdd-eb0d8c5396f9"))
        assert resp.compilations_started == 0
        assert resp.total_output_size_bytes == 0
        assert resp.compilations_completed == 0
        assert resp.compilations_failed == 0
        assert resp.total_output_duration_seconds == 0.0

    def test_schema_with_values(self):
        from src.api.schemas.compilation import UsageSummaryResponse

        resp = UsageSummaryResponse(
            tenant_id=UUID("019de842-c92a-7e37-8fdd-eb0d8c5396f9"),
            compilations_started=10,
            compilations_completed=8,
            compilations_failed=2,
            total_output_duration_seconds=3600.0,
            total_output_size_bytes=1024000,
        )
        assert resp.tenant_id == UUID("019de842-c92a-7e37-8fdd-eb0d8c5396f9")
        assert resp.compilations_started == 10
        assert resp.total_output_size_bytes == 1024000


class TestDailyUsageResponse:
    def test_schema_with_items(self):
        from src.api.schemas.compilation import DailyUsageItem, DailyUsageResponse

        item = DailyUsageItem(date="2026-01-01", compilations_started=5)
        resp = DailyUsageResponse(days=[item])
        assert len(resp.days) == 1
        assert resp.days[0].compilations_started == 5

    def test_daily_item_defaults(self):
        from src.api.schemas.compilation import DailyUsageItem

        item = DailyUsageItem(date="2026-02-23")
        assert item.compilations_started == 0
        assert item.compilations_completed == 0
        assert item.compilations_failed == 0
        assert item.total_output_duration_seconds == 0.0
        assert item.total_output_size_bytes == 0

    def test_empty_days_list(self):
        from src.api.schemas.compilation import DailyUsageResponse

        resp = DailyUsageResponse(days=[])
        assert len(resp.days) == 0
