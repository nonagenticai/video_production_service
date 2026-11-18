"""Portal-realm role guards for tenant-admin surfaces.

Roles: ``super_admin`` > ``tenant_admin`` > ``member``. ``super_admin`` is a
Keycloak composite OF ``tenant_admin`` (the JWT normally lists both); the guard
treats it as implying ``tenant_admin`` so a composite-only token still passes.
This mirrors ``w2v_orchestrator/src/api/routes/_common.py`` — w2v-common ships
no role guard, only the stamp: ``AuthMiddleware`` writes ``request.state.roles``
from the JWT ``realm_access.roles`` claim, and writes ``[]`` on the licence /
mesh / trusted-network paths. A caller with no roles is therefore a member as
far as these guards are concerned — fail closed, never "no roles = allowed".

The guard does NOT pick the tenant a write lands in: that is always the
caller's own resolved tenant (``require_request_tenant``), and every write and
delete query is scoped by it. So a ``tenant_admin`` of tenant A can never touch
tenant B's rows — B's row is simply not found (404).
"""

from __future__ import annotations

import uuid

from fastapi import HTTPException, Request, status

from src.api.tenant import require_request_tenant

SUPER_ADMIN = "super_admin"
TENANT_ADMIN = "tenant_admin"
_ADMIN_ROLES = frozenset({SUPER_ADMIN, TENANT_ADMIN})


def request_roles(request: Request) -> set[str]:
    """Realm roles stamped by ``AuthMiddleware`` (``[]`` when absent)."""
    return set(getattr(request.state, "roles", None) or [])


def require_tenant_admin(request: Request) -> uuid.UUID:
    """FastAPI dependency: caller must be ``tenant_admin`` or ``super_admin``.

    401 first (no tenant identity at all), then 403 (authenticated, not admin) —
    so an anonymous caller is never told the route exists behind a role.
    Returns the caller's tenant for routes that want it.
    """
    tenant_id = require_request_tenant(request)
    if not (_ADMIN_ROLES & request_roles(request)):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "tenant_admin role required")
    return tenant_id
