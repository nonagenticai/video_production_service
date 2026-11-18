"""``/events/{tenant_id}`` and ``/api/analytics/*`` are licensed, and bounded.

THE DEFECT THIS FILE PINS. Both routers were included BARE in ``src/asgi.py``
with no exemption comment, while every deliberate exemption in this estate
carries one — so nothing distinguished "decided" from "forgotten". Cross-tenant
access was already closed (``tests/unit/test_events_tenant_guard.py``), so this
was never a data leak. What was open was RESOURCE: an SSE subscription pins a
Redis pubsub for as long as the client holds it, the rate limiter covered only
``/api/record`` and ``/api/compile``, and there was no cap on a stream's
lifetime. A tenant whose licence was revoked kept streaming, unmetered, forever.

THREE THINGS CLOSE IT, and all three are asserted here:

1. ``LICENSE_GATE`` on both routers — an unentitled tenant cannot OPEN a stream
   or read usage (and, critically, is refused before the pubsub is subscribed).
2. ``APISettings.sse_max_stream_seconds`` — an open stream now dies, so a
   revocation that lands AFTER the connect is enforced within one window when the
   client reconnects THROUGH the gate. The gate alone does not do this: it runs
   once, at connect.
3. ``RateLimitMiddleware.RATE_LIMITED_PREFIXES`` extended to ``/events`` and
   ``/api/analytics`` — with (2), concurrency is bounded by (rate x lifetime)
   rather than by nothing.

Every assertion drives a real request through the real routers: this FastAPI
build includes sub-routers lazily, so ``route.dependant.dependencies`` reports a
live-gated route as ungated and static introspection is a false negative.

WHY THE REDIS DOUBLE FAILS, AND WHY THE FRAME READS ARE BOUNDED. ``httpx``'s
``ASGITransport`` buffers the whole response body, and ``Request.is_disconnected()``
never flips under it — so an SSE route that is NOT refused streams keepalives
forever and a naive test HANGS instead of failing. That was measured: deleting
``LICENSE_GATE`` or the lifetime cap and re-running this file produced a 20s
pytest-timeout dump, not a red assertion, and a CI hang is a worse signal than a
red test. Two things fix it, and both are load-bearing rather than cosmetic: the
gate tests give ``_ensure_connected`` a ``RedisError`` side effect, so an ungated
request terminates immediately with a 200 error-frame body (a fast, unambiguous
red) while a gated one never reaches it; and the lifetime tests read a BOUNDED
number of frames, so an uncapped generator fails on the bound instead of running
out the clock.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from redis.exceptions import RedisError
from starlette.responses import StreamingResponse
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError
from w2v_common.database.session import get_db

from src.api.analytics import router as analytics_router
from src.api.events import router as events_router
from src.api.events import stream_events
from src.config import clear_settings_cache
from src.middleware.rate_limiter import RateLimitMiddleware
from src.services.compilation.job_manager import CompilationJobManager
from tests.conftest import TENANT_U8

pytestmark = pytest.mark.timeout(20)

TENANT = str(TENANT_U8)
SUBJECT = "user-under-test"
EVENTS_PATH = f"/events/{TENANT}"
ANALYTICS_PATH = "/api/analytics/usage"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    conflict = False


class _FakeLicenses:
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


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TENANT,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TENANT, status="revoked")]


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


def _redis_double(*, fail_fast: bool = False) -> tuple[MagicMock, MagicMock]:
    """A redis-client double; the pubsub is returned so a test can assert that
    ``subscribe`` was never reached — a gate that refuses AFTER opening the
    channel would pass a status-code-only test while still pinning the pubsub.

    ``fail_fast`` makes ``_ensure_connected`` raise, which is how the gate tests
    keep a REGRESSION fast: with the gate gone the handler runs, hits the raise,
    emits one error frame and ends, so the test reports a wrong status code in
    milliseconds instead of hanging on the keepalive loop. It cannot mask a
    working gate — a gated request never calls ``_ensure_connected`` at all, which
    is asserted directly.
    """
    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.get_message = AsyncMock(return_value=None)
    pubsub.unsubscribe = AsyncMock()
    pubsub.reset = AsyncMock()

    connection = MagicMock()
    connection.pubsub = MagicMock(return_value=pubsub)

    redis_client = MagicMock()
    redis_client._ensure_connected = MagicMock(
        side_effect=RedisError("fail fast") if fail_fast else None,
        return_value=connection,
    )
    return redis_client, pubsub


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


def _make_app(redis_client: Any, *, rate_limited: bool = False) -> FastAPI:
    """The REAL events + analytics routers behind an AuthMiddleware stand-in."""
    app = FastAPI()

    @app.middleware("http")
    async def _stamp(request: Request, call_next: Any) -> Any:
        request.state.tenant_id = TENANT
        request.state.subject_id = SUBJECT
        return await call_next(request)

    app.include_router(events_router)
    app.include_router(analytics_router)
    app.state.redis_client = redis_client
    if rate_limited:
        app.add_middleware(RateLimitMiddleware)

    async def _no_db() -> Any:
        yield None

    app.dependency_overrides[get_db] = _no_db
    return app


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant reaches neither the stream nor the usage numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"), [([], "license_none"), (_revoked(), "license_revoked")]
)
async def test_armed_gate_402s_the_sse_stream_and_never_subscribes(monkeypatch, inventory, code):
    """402 BEFORE the pubsub is opened. The status code alone is not the
    property under test: a gate that refused after subscribing would leave the
    resource pinned, which is the whole defect."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))
    redis_client, pubsub = _redis_double(fail_fast=True)

    response = await _get(_make_app(redis_client), EVENTS_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code
    redis_client._ensure_connected.assert_not_called()
    pubsub.subscribe.assert_not_awaited()


async def test_armed_gate_402s_analytics(monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(_make_app(None), ANALYTICS_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


async def test_armed_gate_503s_when_entitlement_cannot_be_established(monkeypatch):
    """An upstream failure is not an allow. Same shape as /api/compile."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))
    redis_client, pubsub = _redis_double(fail_fast=True)

    response = await _get(_make_app(redis_client), EVENTS_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]
    pubsub.subscribe.assert_not_awaited()


# ---------------------------------------------------------------------------
# The non-vacuity pair: an entitled tenant is still served
# ---------------------------------------------------------------------------


async def test_armed_gate_serves_analytics_to_an_entitled_tenant(monkeypatch):
    """Without this, a gate that refused everything would pass this file."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    async def _summary(_db: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"tenant_id": uuid.UUID(TENANT), "compilations_started": 3}

    monkeypatch.setattr(CompilationJobManager, "get_usage_summary", _summary)

    response = await _get(_make_app(None), ANALYTICS_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["compilations_started"] == 3


async def test_gate_is_dormant_while_the_flag_is_unset(monkeypatch):
    """Merging this changes nothing until gitops arms the flag — the same
    rollout contract /api/record and /api/compile already carry."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    async def _summary(_db: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"tenant_id": uuid.UUID(TENANT)}

    monkeypatch.setattr(CompilationJobManager, "get_usage_summary", _summary)

    response = await _get(_make_app(None), ANALYTICS_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# The resource ceiling the gate alone does NOT provide
# ---------------------------------------------------------------------------


def _direct_request(redis_client: Any) -> Request:
    """A real ``Request`` on the caller's own tenant.

    Driven through the handler rather than ASGITransport because that transport
    buffers the whole body and ``Request.is_disconnected()`` never flips under
    it — a lifetime test through the transport would HANG rather than assert.
    """
    from types import SimpleNamespace

    app = SimpleNamespace(state=SimpleNamespace(redis_client=redis_client))

    async def _receive() -> dict[str, Any]:
        # ``Request.is_disconnected()`` awaits the receive channel inside an
        # already-cancelled scope, so this is never actually consumed — but
        # without it the loop's first disconnect probe raises
        # "Receive channel has not been made available" and the generator would
        # exit through the error branch instead of the branch under test.
        return {"type": "http.request"}

    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": EVENTS_PATH,
            "headers": [],
            "app": app,
            "state": {},
        },
        receive=_receive,
    )
    request.state.tenant_id = TENANT
    return request


async def test_sse_stream_dies_at_the_lifetime_cap_and_releases_the_pubsub(monkeypatch):
    """A stream is no longer unbounded: it ends itself and unsubscribes.

    ``sse_max_stream_seconds`` is set to 0 so the deadline is already past on the
    first loop iteration. The client is told WHY (``stream_expired``) rather than
    being dropped silently, because a silent drop is indistinguishable from a
    network fault and clients retry differently for the two.
    """
    monkeypatch.setenv("API_SSE_MAX_STREAM_SECONDS", "0.0001")
    clear_settings_cache()
    try:
        redis_client, pubsub = _redis_double()

        response = await stream_events(TENANT, _direct_request(redis_client))

        assert isinstance(response, StreamingResponse)
        body = cast(AsyncGenerator[str], response.body_iterator)
        # BOUNDED, not ``[f async for f in body]``: an uncapped generator never
        # ends, and draining it would hang the suite instead of failing it.
        frames: list[str] = []
        ended = False
        for _ in range(4):
            try:
                frames.append(await anext(body))
            except StopAsyncIteration:
                ended = True
                break
        if not ended:
            await body.aclose()
    finally:
        clear_settings_cache()

    assert ended, f"stream did not terminate at the cap; got {frames}"
    assert frames[0] == ": connected\n\n"
    assert "stream_expired" in frames[-1], frames
    # The generator RAN OUT rather than being closed by the test, which means the
    # ``finally`` released the Redis pubsub. That release is the resource the cap
    # exists to reclaim.
    pubsub.unsubscribe.assert_awaited()


async def test_an_uncapped_stream_would_not_terminate(monkeypatch):
    """Control for the test above: at the shipped default the generator does NOT
    end on its own, which is exactly why the cap is load-bearing and not
    decoration."""
    clear_settings_cache()
    try:
        redis_client, pubsub = _redis_double()
        response = await stream_events(TENANT, _direct_request(redis_client))
        body = cast(AsyncGenerator[str], response.body_iterator)
        assert await anext(body) == ": connected\n\n"
        assert await anext(body) == ": keepalive\n\n"
        await body.aclose()
    finally:
        clear_settings_cache()
    pubsub.subscribe.assert_awaited_once()


async def test_the_rate_limiter_now_covers_events_and_analytics(monkeypatch):
    """Both prefixes are metered. Before this they were the only tenant-scoped
    surfaces in the service with no ingress ceiling at all."""
    assert "/events" in RateLimitMiddleware.RATE_LIMITED_PREFIXES
    assert "/api/analytics" in RateLimitMiddleware.RATE_LIMITED_PREFIXES

    monkeypatch.setenv("API_RATE_LIMIT_REQUESTS", "1")
    clear_settings_cache()
    try:
        # A redis double whose pipeline reports the tenant is already at the cap.
        pipeline = MagicMock()
        pipeline.zremrangebyscore = MagicMock()
        pipeline.zcard = MagicMock()
        pipeline.zadd = MagicMock()
        pipeline.expire = MagicMock()
        pipeline.execute = AsyncMock(return_value=[0, 1, 1, True])
        connection = MagicMock()
        connection.pipeline = MagicMock(return_value=pipeline)
        redis_client = MagicMock()
        redis_client._ensure_connected = MagicMock(return_value=connection)

        app = _make_app(redis_client, rate_limited=True)
        for path in (EVENTS_PATH, ANALYTICS_PATH):
            response = await _get(app, path)
            assert response.status_code == 429, f"{path}: {response.text}"
            assert response.headers["Retry-After"]
    finally:
        clear_settings_cache()
