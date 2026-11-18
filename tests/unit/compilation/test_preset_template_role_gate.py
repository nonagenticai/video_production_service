"""Brand-kit writes (custom presets + templates) are tenant-admin only.

Driven through the REAL ``/api/compile`` router over HTTP against REAL postgres,
because the gate is a route dependency: calling the handler functions directly
(as ``test_routes_extended.py`` does) bypasses it and would pass on ungated code.

Identity is stamped exactly the way ``AuthMiddleware`` stamps it —
``request.state.tenant_id`` (str) and ``request.state.roles`` (list) — from two
test-only headers. ``roles=[]`` is what the licence / mesh path produces, i.e. a
plain member.

Every refusal is also checked AT THE DATABASE: a 403 that still wrote the row
would be a refusal in name only.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.database.session import get_db

from src.api.routes.compilation import router as compilation_router
from src.database.models import CompilationTemplate, CustomPreset
from tests.conftest import TENANT_A, TENANT_B

_TENANT_HDR = "x-test-tenant"
_ROLES_HDR = "x-test-roles"

PRESET_BODY = {"name": "brand-1080", "description": "", "target_width": 1920, "crf": 20}
TEMPLATE_BODY = {"name": "brand-intro", "description": "", "theme_config": {}}


class _StampIdentity(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
        tenant = request.headers.get(_TENANT_HDR)
        if tenant:
            request.state.tenant_id = tenant
        roles = request.headers.get(_ROLES_HDR, "")
        request.state.roles = [r for r in roles.split(",") if r]
        return await call_next(request)


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The licence gate is orthogonal to the role gate; keep it in its shipped
    # (dormant) state so a 403 here can only come from the role check.
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture
def app(db_session_factory: async_sessionmaker[AsyncSession]) -> FastAPI:
    application = FastAPI()
    application.include_router(compilation_router)
    application.add_middleware(_StampIdentity)

    async def _db() -> AsyncIterator[AsyncSession]:
        async with db_session_factory() as session:
            yield session

    application.dependency_overrides[get_db] = _db
    return application


def _hdrs(tenant: uuid.UUID | None, *roles: str) -> dict[str, str]:
    h = {_ROLES_HDR: ",".join(roles)}
    if tenant is not None:
        h[_TENANT_HDR] = str(tenant)
    return h


async def _call(app: FastAPI, method: str, path: str, headers: dict[str, str], **kw: Any) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.request(method, path, headers=headers, **kw)


async def _seed(factory: async_sessionmaker[AsyncSession], tenant: uuid.UUID) -> tuple[str, str]:
    preset_id, template_id = str(uuid.uuid7()), str(uuid.uuid7())
    async with factory() as s:
        s.add(CustomPreset(preset_id=preset_id, tenant_id=tenant, name="seeded"))
        s.add(CompilationTemplate(template_id=template_id, tenant_id=tenant, name="seeded"))
        await s.commit()
    return preset_id, template_id


async def _count(factory: async_sessionmaker[AsyncSession], model: Any, **where: Any) -> int:
    async with factory() as s:
        stmt = select(func.count()).select_from(model).filter_by(**where)
        return int((await s.execute(stmt)).scalar_one())


# --------------------------------------------------------------------------- member


@pytest.mark.parametrize(
    ("path", "body", "model"),
    [
        ("/api/compile/presets/custom", PRESET_BODY, CustomPreset),
        ("/api/compile/templates", TEMPLATE_BODY, CompilationTemplate),
    ],
)
async def test_member_cannot_create(app, db_session_factory, path, body, model) -> None:
    resp = await _call(app, "POST", path, _hdrs(TENANT_A, "member"), json=body)
    assert resp.status_code == 403, resp.text
    assert await _count(db_session_factory, model, tenant_id=TENANT_A) == 0


async def test_rolesless_licence_caller_cannot_create(app, db_session_factory) -> None:
    """The licence/mesh path stamps roles=[] — that is a member, not an admin."""
    resp = await _call(
        app, "POST", "/api/compile/presets/custom", _hdrs(TENANT_A), json=PRESET_BODY
    )
    assert resp.status_code == 403, resp.text
    assert await _count(db_session_factory, CustomPreset) == 0


async def test_member_cannot_delete_own_tenant_rows(app, db_session_factory) -> None:
    preset_id, template_id = await _seed(db_session_factory, TENANT_A)
    hdrs = _hdrs(TENANT_A, "member")

    r1 = await _call(app, "DELETE", f"/api/compile/presets/custom/{preset_id}", hdrs)
    r2 = await _call(app, "DELETE", f"/api/compile/templates/{template_id}", hdrs)

    assert (r1.status_code, r2.status_code) == (403, 403), (r1.text, r2.text)
    assert await _count(db_session_factory, CustomPreset, preset_id=preset_id) == 1
    assert await _count(db_session_factory, CompilationTemplate, template_id=template_id) == 1


async def test_member_can_read_own_tenant(app, db_session_factory) -> None:
    preset_id, template_id = await _seed(db_session_factory, TENANT_A)
    await _seed(db_session_factory, TENANT_B)  # must not leak into A's lists
    hdrs = _hdrs(TENANT_A, "member")

    presets = await _call(app, "GET", "/api/compile/presets/custom", hdrs)
    templates = await _call(app, "GET", "/api/compile/templates", hdrs)

    assert presets.status_code == 200, presets.text
    assert [p["id"] for p in presets.json()["presets"]] == [preset_id]
    assert templates.status_code == 200, templates.text
    custom = [t["id"] for t in templates.json()["templates"] if not t["is_builtin"]]
    assert custom == [template_id]


async def test_unauthenticated_write_is_401_not_403(app) -> None:
    resp = await _call(app, "POST", "/api/compile/templates", _hdrs(None), json=TEMPLATE_BODY)
    assert resp.status_code == 401, resp.text


# --------------------------------------------------------------------------- admins


@pytest.mark.parametrize("role", ["tenant_admin", "super_admin"])
async def test_admin_create_and_delete_own_tenant(app, db_session_factory, role) -> None:
    hdrs = _hdrs(TENANT_A, role)

    p = await _call(app, "POST", "/api/compile/presets/custom", hdrs, json=PRESET_BODY)
    t = await _call(app, "POST", "/api/compile/templates", hdrs, json=TEMPLATE_BODY)
    assert (p.status_code, t.status_code) == (200, 200), (p.text, t.text)
    preset_id, template_id = p.json()["id"], t.json()["id"]
    # The write lands in the CALLER's tenant.
    assert (
        await _count(db_session_factory, CustomPreset, preset_id=preset_id, tenant_id=TENANT_A) == 1
    )
    assert (
        await _count(
            db_session_factory, CompilationTemplate, template_id=template_id, tenant_id=TENANT_A
        )
        == 1
    )

    dp = await _call(app, "DELETE", f"/api/compile/presets/custom/{preset_id}", hdrs)
    dt = await _call(app, "DELETE", f"/api/compile/templates/{template_id}", hdrs)
    assert (dp.status_code, dt.status_code) == (200, 200), (dp.text, dt.text)
    assert await _count(db_session_factory, CustomPreset, preset_id=preset_id) == 0
    assert await _count(db_session_factory, CompilationTemplate, template_id=template_id) == 0


async def test_tenant_admin_cannot_delete_other_tenants_rows(app, db_session_factory) -> None:
    preset_id, template_id = await _seed(db_session_factory, TENANT_A)
    hdrs = _hdrs(TENANT_B, "tenant_admin")

    r1 = await _call(app, "DELETE", f"/api/compile/presets/custom/{preset_id}", hdrs)
    r2 = await _call(app, "DELETE", f"/api/compile/templates/{template_id}", hdrs)

    assert (r1.status_code, r2.status_code) == (404, 404), (r1.text, r2.text)
    assert await _count(db_session_factory, CustomPreset, preset_id=preset_id) == 1
    assert await _count(db_session_factory, CompilationTemplate, template_id=template_id) == 1
