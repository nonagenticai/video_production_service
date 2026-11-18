"""Cross-tenant guard on the SSE stream (``GET /events/{tenant_id}``).

``stream_production_events`` took ``tenant_id`` from the path and passed it
straight into the Redis pub/sub subscribe, never consulting the caller's
identity. Any authenticated caller naming another tenant's id got a live feed of
that tenant's job events. Redis is not Postgres: there is no RLS backstop behind
this route, and every id synthesises a valid subscription, so nothing downstream
turned the leak into a 404.

The assertions here are deliberately about the *subscribe*, not only the status
code. A guard that 403s after the generator has already opened the channel would
pass a status-code-only test while still leaking, so the negative test asserts
that neither ``_ensure_connected`` nor ``pubsub.subscribe`` was ever reached.

Every test that drives the route through ``httpx.ASGITransport`` carries an
explicit ``@pytest.mark.timeout``. ASGITransport buffers the whole response body,
and the SSE generator's keepalive branch loops on ``asyncio.sleep(1)`` while
``Request.is_disconnected()`` never flips under that transport — so deleting the
guard would make these tests HANG rather than fail. A CI hang is a worse signal
than a red test; the marks make a regression fail fast and loudly. (There is also
a global ``timeout`` in ``[tool.pytest.ini_options]``; these marks are tighter.)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse

from src.api.events import router as events_router
from src.api.events import stream_events

# Real uuids: tenant is a ``uuid.UUID`` since w2v-common 0.67.0 and the route
# parses BOTH sides before comparing, so a slug can no longer be a tenant at all.
TENANT_A = "019de900-0000-7000-8000-0000000000a7"
TENANT_OTHER = "019de900-0000-7000-8000-0000000000b7"
TENANT_B = "tenant-b"


def _redis_double() -> tuple[MagicMock, MagicMock]:
    """Build a redis-client double and return ``(client, pubsub)``.

    The pubsub is returned separately so a test can assert on ``subscribe``
    without reaching through three layers of mock plumbing.
    """
    pubsub = MagicMock()
    pubsub.subscribe = AsyncMock()
    pubsub.get_message = AsyncMock(return_value=None)
    pubsub.unsubscribe = AsyncMock()
    pubsub.reset = AsyncMock()

    connection = MagicMock()
    connection.pubsub = MagicMock(return_value=pubsub)

    redis_client = MagicMock()
    redis_client._ensure_connected = MagicMock(return_value=connection)
    return redis_client, pubsub


def _make_app(caller_tenant: str | None, redis_client: Any) -> FastAPI:
    """Minimal app mounting the events router behind a tenant-stamping middleware.

    Stands in for ``AuthMiddleware``, which is what sets ``request.state.tenant_id``
    in production. ``caller_tenant=None`` stamps nothing — that is the shape of a
    request that reaches the handler with AUTH_ENABLED=false, or via
    ``auth_public_endpoints``, and it is what the fail-closed test exercises.
    """
    app = FastAPI()

    @app.middleware("http")
    async def _stamp_tenant(request: Request, call_next: Any) -> Any:
        if caller_tenant is not None:
            request.state.tenant_id = caller_tenant
        return await call_next(request)

    app.include_router(events_router)
    app.state.redis_client = redis_client
    return app


def _direct_request(caller_tenant: str, redis_client: Any) -> Request:
    """A real ``Request`` carrying the caller's tenant and an app with Redis."""
    app = SimpleNamespace(state=SimpleNamespace(redis_client=redis_client))
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": f"/events/{caller_tenant}",
            "headers": [],
            "app": app,
            "state": {},
        }
    )
    request.state.tenant_id = caller_tenant
    return request


@pytest.mark.timeout(15)
async def test_cross_tenant_stream_is_403_and_never_subscribes() -> None:
    """A caller on tenant A asking for tenant B's stream is refused before Redis.

    403 rather than 404 (matching ai_content_generation_service PR #91): the
    tenant id is not a secret, the subscription is simply not the caller's.
    """
    redis_client, pubsub = _redis_double()
    transport = httpx.ASGITransport(app=_make_app(TENANT_A, redis_client))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/events/{TENANT_B}")

    assert response.status_code == 403
    # The leak is the subscription, not the status line: prove the channel was
    # never opened, so a guard that 403s *after* streaming cannot pass this.
    redis_client._ensure_connected.assert_not_called()
    pubsub.subscribe.assert_not_awaited()


@pytest.mark.timeout(15)
async def test_own_tenant_stream_still_subscribes_and_streams() -> None:
    """Positive control: the caller's own tenant still gets a live stream.

    Guards against the fix degenerating into "deny everything". Driven through
    the handler directly so the test can read the first SSE frame and close the
    generator instead of hanging on the keepalive loop.
    """
    redis_client, pubsub = _redis_double()
    request = _direct_request(TENANT_A, redis_client)

    response = await stream_events(TENANT_A, request)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    # ``body_iterator`` is typed ``AsyncIterable``; the SSE handler always yields
    # an async generator, and the test needs ``aclose()`` to avoid hanging on the
    # keepalive loop.
    body = cast(AsyncGenerator[str], response.body_iterator)
    try:
        assert await anext(body) == ": connected\n\n"
    finally:
        await body.aclose()

    pubsub.subscribe.assert_awaited_once_with(f"recording:jobs:{TENANT_A}")


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "requested",
    [
        TENANT_OTHER,  # a real, different tenant
        f"{TENANT_A}x",  # one char too long
        TENANT_A[:-1],  # truncated
        TENANT_A.replace("-", "")[:-2],  # undashed AND truncated
        "anonymous",  # the removed sentinel spelling
        "00000000-0000-0000-0000-000000000000",  # the removed nil sentinel
    ],
)
async def test_near_miss_tenant_ids_are_refused(requested: str) -> None:
    """Near-miss path parameters never reach a subscription.

    The comparison is on the PARSED identity now, not on the two strings, so the
    old case/trailing-whitespace variants (``TENANT_A.upper()``, ``f"{TENANT_A} "``)
    are deliberately gone from this list: uppercase hex and surrounding whitespace
    are the SAME uuid, and refusing them would be refusing the caller's own
    tenant. What must still be refused — a different tenant, an unparseable
    near-miss, and either removed sentinel — is what is asserted, with the nil
    uuid added because a ``None``-only guard would have let it through.
    """
    redis_client, pubsub = _redis_double()
    transport = httpx.ASGITransport(app=_make_app(TENANT_A, redis_client))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/events/{requested}")

    assert response.status_code == 403
    pubsub.subscribe.assert_not_awaited()


@pytest.mark.timeout(15)
@pytest.mark.parametrize("requested", ["anonymous", TENANT_A])
async def test_unresolved_caller_tenant_is_403_and_never_subscribes(requested: str) -> None:
    """No ``request.state.tenant_id`` at all ⇒ 403, including for ``/events/anonymous``.

    ``_get_tenant_id`` falls back to the literal "anonymous" when the request has
    no tenant on ``state`` — which is exactly what happens with
    ``AUTH_ENABLED=false`` (``AuthSettings.enabled`` is a plain env-overridable
    field) or if this route were added to ``auth_public_endpoints``. A plain
    equality check would then match ``GET /events/anonymous`` for EVERY caller and
    hand out a live feed, and that channel is real: ``RecordingEventPublisher``
    publishes to ``recording:jobs:{tenant_id}`` using the same fallback. The guard
    must therefore reject the sentinel rather than compare it.
    """
    redis_client, pubsub = _redis_double()
    transport = httpx.ASGITransport(app=_make_app(None, redis_client))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/events/{requested}")

    assert response.status_code == 403
    redis_client._ensure_connected.assert_not_called()
    pubsub.subscribe.assert_not_awaited()


@pytest.mark.timeout(15)
@pytest.mark.parametrize("caller", ["", "   ", "anonymous"])
async def test_sentinel_or_blank_caller_tenant_cannot_match_itself(caller: str) -> None:
    """Even when the sentinel/blank is stamped explicitly, it is not an identity."""
    redis_client, pubsub = _redis_double()
    transport = httpx.ASGITransport(app=_make_app(caller, redis_client))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/events/{caller or 'anonymous'}")

    assert response.status_code == 403
    pubsub.subscribe.assert_not_awaited()
