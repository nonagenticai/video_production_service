"""Usage analytics API endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession
from w2v_common.auth import LICENSE_GATE
from w2v_common.database.session import get_db

from src.api.schemas import DailyUsageResponse, UsageSummaryResponse
from src.api.tenant import require_request_tenant
from src.services.compilation.job_manager import CompilationJobManager

logger = logging.getLogger(__name__)

# ENTITLEMENT GATE — same decision as src/api/events.py, recorded here so the
# next auditor does not have to re-derive it from the absence of a comment.
#
# This router shipped bare with no exemption note. It is not a probe, not a
# peer-callback and not an operator surface: both routes are tenant-scoped
# aggregate reads over THIS tenant's own compilation jobs, i.e. exactly the
# billable, tenant-scoped usage the licence gate exists to price. There is no
# reason for it to answer a tenant whose licence /api/compile refuses.
#
# Callers checked before gating, not assumed: no frontend, no orchestrator
# client, no generated openapi client, no Grafana dashboard, no Prometheus
# scrape (this service has no ServiceMonitor; scraping is `/metrics`, a
# different router) and no k8s probe references `/api/analytics/*`. The only
# hits are `api_integration_tester`'s local dev harness on localhost:6308, whose
# two analytics steps are already `continue_on_failure: true`. Gating cannot
# regress a working caller in any case: `require_request_tenant` below already
# 401s anything without a tenant identity, so no licence-less observability
# client could ever have read these.
router = APIRouter(prefix="/api/analytics", tags=["analytics"], dependencies=list(LICENSE_GATE))


def _get_tenant_id(request: Request) -> uuid.UUID:
    """Return the caller's tenant UUID, or raise 401. No "anonymous" fallback."""
    return require_request_tenant(request)


@router.get("/usage", operation_id="get_production_usage_summary")
async def get_usage_summary(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
) -> UsageSummaryResponse:
    """Get aggregated usage statistics for the current tenant."""
    tenant_id = _get_tenant_id(request)
    logger.debug("Usage summary request: tenant=%s, from=%s, to=%s", tenant_id, date_from, date_to)
    summary = await CompilationJobManager.get_usage_summary(
        db,
        tenant_id=tenant_id,
        date_from=date_from,
        date_to=date_to,
    )
    return UsageSummaryResponse(**summary)


@router.get("/usage/daily", operation_id="get_production_usage_daily")
async def get_usage_daily(
    request: Request,
    db: AsyncSession = Depends(get_db),
    date_from: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="End date (YYYY-MM-DD)"),
) -> DailyUsageResponse:
    """Get daily usage breakdown for the current tenant."""
    tenant_id = _get_tenant_id(request)
    logger.debug("Daily usage request: tenant=%s, from=%s, to=%s", tenant_id, date_from, date_to)
    daily_items = await CompilationJobManager.get_usage_daily(
        db,
        tenant_id=tenant_id,
        date_from=date_from,
        date_to=date_to,
    )
    return DailyUsageResponse(days=daily_items)
