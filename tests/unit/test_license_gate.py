"""The entitlement gate on /api/record and /api/compile actually refuses an
unlicensed tenant.

Until w2v-common 0.75.0 this service authenticated the licence token for
tenant IDENTITY and stopped there: ``AuthMiddleware`` answers "who are you",
never "may you". A tenant whose licence was ``expired`` or ``revoked`` still
presented a syntactically valid token, so every recording and compilation
route served them -- Playwright recordings, FFmpeg compiles and every other
billable operation. ``LICENSE_GATE`` is mounted router-wide on both
``src/api/routes/recording.py`` and ``src/api/routes/compilation.py`` to
close that.

Every test here drives a REAL HTTP request through the REAL ``router``
objects (and the REAL health router), because the interesting question is
not whether ``require_license`` works -- w2v_common owns that -- but whether
THIS service's routes actually carry it. This FastAPI build includes
sub-routers lazily (``_IncludedRouter``), so a parent router's
``dependencies=`` are applied when a request is matched, not baked into the
route objects at import time: static introspection of ``route.dependant``
would report the compilation sub-routes (jobs/presets/extras registered via
``register(router)``) as ungated and would be a *false negative*. A request
is the only honest probe.

The paired 200s are what stop this file from passing vacuously: if the gate
were mounted wrongly (or a fixture were broken) the "entitled" and
"flag off" cases would not be 200s.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.dependencies import get_recording_job_manager
from src.api.health import router as health_router
from src.api.routes.compilation import router as compilation_router
from src.api.routes.recording import router as recording_router
from tests.conftest import TENANT_U8, bind_test_tenant

SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route on the /api/record router: its
#: only dependency is ``get_recording_job_manager`` (overridden below with a
#: fake), so a non-200 here can only be the gate.
RECORD_GATED_PATH = "/api/record/jobs"

#: The cheapest route on the /api/compile router: a pure catalogue lookup with
#: NO dependencies at all (no DB, no job manager) -- so a non-200 here can
#: only be the gate.
COMPILE_GATED_PATH = "/api/compile/themes"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness
#: probe CrashLoops the pod, and an unlicensed tenant cannot be billed by
#: killing our own deployment.
PROBE_PATH = "/health/live"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=str(TENANT_U8), status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=str(TENANT_U8),
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=str(TENANT_U8),
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()
    application.include_router(health_router)
    application.include_router(recording_router)
    application.include_router(compilation_router)
    bind_test_tenant(application, TENANT_U8)

    # bind_test_tenant only stamps tenant_id (the way AuthMiddleware itself
    # does). The gate's inventory-fallback path (none/expired/revoked
    # refinement) additionally needs subject_id, or it silently sees no
    # subject, fetches an empty inventory, and derives 'none' regardless of
    # what the fake licenses client would have said -- see
    # ``entitlement._resolve_uncached``'s ``if subject_id else []`` branch.
    class _StampSubject(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampSubject)

    class _FakeJobManager:
        async def list_jobs(self, **_kwargs: Any) -> tuple[list[Any], int]:
            return [], 0

    application.dependency_overrides[get_recording_job_manager] = lambda: _FakeJobManager()
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused, on BOTH gated routers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gated_path", [RECORD_GATED_PATH, COMPILE_GATED_PATH])
@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code, gated_path):
    """402 with the state-specific code -- the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(app, gated_path)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


@pytest.mark.parametrize("gated_path", [RECORD_GATED_PATH, COMPILE_GATED_PATH])
async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch, gated_path):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(app, gated_path)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gated_path", [RECORD_GATED_PATH, COMPILE_GATED_PATH])
async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch, gated_path):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, gated_path)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served -- deliberately -- so the rollout is a flag flip, not a
    deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, RECORD_GATED_PATH)

    assert response.status_code == 200, response.text


async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, COMPILE_GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


async def test_probe_is_served_for_a_revoked_tenant(app, monkeypatch):
    """A gated probe path CrashLoops the pod. /health/live must stay 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, PROBE_PATH)

    assert response.status_code == 200, response.text


async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(RECORD_GATED_PATH)
    assert not is_license_exempt_path(COMPILE_GATED_PATH)
