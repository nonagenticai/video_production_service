"""SSE endpoint for real-time job status streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from redis.exceptions import RedisError
from starlette.responses import StreamingResponse
from w2v_common.auth import LICENSE_GATE

from src.api.tenant import parse_request_tenant, require_request_tenant
from src.config import get_settings

logger = logging.getLogger(__name__)

# ENTITLEMENT GATE — WHY THIS ROUTER IS GATED LIKE /api/record AND /api/compile.
#
# This router shipped BARE, with no exemption comment, while every deliberate
# exemption in this estate carries one. That absence was the defect: an auditor
# could not tell "decided" from "forgotten". It was forgotten.
#
# Cross-tenant access was already closed (the 403 in `stream_events` below), so
# this was never a data leak. The open hole was RESOURCE: an SSE subscription is
# a long-lived connection pinning a Redis pubsub, and a tenant whose licence is
# expired or revoked could open as many as it liked, indefinitely — refused on
# every billable /api/compile call while still consuming the streaming capacity
# those calls pay for. `LICENSE_GATE` makes the subscription cost what the rest
# of the service costs: none/expired/revoked -> 402, unestablishable -> 503.
#
# Who calls this, checked before gating rather than assumed: NOBODY in
# production. A fleet-wide search (w2v_frontend, w2v_orchestrator,
# bdd_playwright_service, every generated openapi client, gitops-production
# manifests, Prometheus rules, Grafana dashboards, n8n workflows) found zero
# consumers of `/events/{tenant_id}`; the frontend's SSE is
# `/api/runs/{id}/events` on the ORCHESTRATOR. The k8s probes are `/health/live`
# and `/health/ready` and are exempt by path. The only hits are
# `api_integration_tester`'s local dev harness (localhost:6308, three steps, all
# already `continue_on_failure: true`) and this repo's own tests. So there is no
# operator console or dashboard to break — and there could not have been one:
# `require_request_tenant` below already 401/403s a caller with no tenant
# identity, so any caller succeeding today ALREADY carries one.
#
# This is option (a)+(b) together, and (a) alone was rejected: the gate runs once,
# at connect. A stream opened while entitled outlives a revocation forever. The
# `sse_max_stream_seconds` cap below is what forces re-entry through the gate.
router = APIRouter(tags=["events"], dependencies=list(LICENSE_GATE))

# redis-py's ``ConnectionError``/``TimeoutError`` hang off ``RedisError(Exception)``,
# not off the builtins of the same name, so a builtin-only tuple never fires on a
# real Redis fault. Without ``RedisError`` the SSE generator raised mid-stream
# instead of emitting the ``data: {"error": ...}`` frame it promises, and the
# ``finally`` cleanup re-raised over the original error. Same shape as
# ``_REDIS_PUBLISH_ERRORS`` in ``src/api/ws.py``.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError)

# The caller tenant used to come from ``_get_tenant_id``, which fell back to the
# literal "anonymous" when ``request.state.tenant_id`` was absent — so with
# AUTH_ENABLED=false, or if this route were ever added to
# ``auth_public_endpoints``, EVERY caller resolved to "anonymous" and
# ``GET /events/anonymous`` satisfied ``tenant_id == caller``. w2v-common 0.67.0
# removed every such fallback fleet-wide and this service followed
# (``src/api/tenant.py``): an unresolved tenant now RAISES. This route keeps its
# own 403 shape rather than letting the helper's 401 escape, because the answer a
# subscriber gets is "that subscription is not yours to open" either way, and the
# 403 contract is what the SSE clients and the guard tests are written against.


async def _sse_pubsub_loop(pubsub: Any, request: Request) -> AsyncGenerator[str]:
    """Relay one tenant's job events until the client leaves or the cap expires.

    The deadline is the resource ceiling, not a nicety. Without it the only exit
    was ``request.is_disconnected()``, which a half-open TCP connection never
    reports, so a stream — and the Redis pubsub it pins — lived until the pod
    restarted. On expiry the client is told so explicitly and the generator
    returns, which runs the ``finally`` in :func:`_sse_event_generator` and
    releases the pubsub; a client that still wants events reconnects, and the
    reconnection passes ``LICENSE_GATE`` again with a FRESH entitlement verdict.
    """
    deadline = time.monotonic() + get_settings().api.sse_max_stream_seconds
    yield ": connected\n\n"
    while True:
        if await request.is_disconnected():
            break
        if time.monotonic() >= deadline:
            logger.info("SSE stream hit its lifetime cap; closing so the client reconnects")
            yield f"data: {json.dumps({'event': 'stream_expired', 'reason': 'max_stream_seconds'})}\n\n"
            break
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
        if message and message["type"] == "message":
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            yield f"data: {data}\n\n"
        else:
            yield ": keepalive\n\n"
            await asyncio.sleep(1)


async def _sse_event_generator(
    redis_client: Any, tenant_id: uuid.UUID, request: Request
) -> AsyncGenerator[str]:
    pubsub = None
    try:
        client = redis_client._ensure_connected()
        pubsub = client.pubsub()
        channel = f"recording:jobs:{tenant_id}"
        await pubsub.subscribe(channel)
        async for chunk in _sse_pubsub_loop(pubsub, request):
            yield chunk
    except asyncio.CancelledError:
        pass
    except _REDIS_ERRORS as e:
        logger.error("SSE stream error: %s", e)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        if pubsub:
            try:
                await pubsub.unsubscribe()
                await pubsub.reset()
            except _REDIS_ERRORS as exc:
                logger.warning("Failed to clean up Redis pubsub: %s", exc)


@router.get("/events/{tenant_id}", operation_id="stream_production_events")
async def stream_events(tenant_id: str, request: Request) -> StreamingResponse:
    """Server-Sent Events stream for job status updates.

    The ``{tenant_id}`` path parameter must equal the caller's own tenant.

    Before this guard the parameter was passed straight into the Redis pub/sub
    subscribe, so any authenticated caller naming another tenant's id received a
    live feed of that tenant's job events. Redis has no RLS backstop, so nothing
    downstream caught it.

    On the path parameter, what is actually known: it is a published contract in
    THIS repo — ``README.md``, the ``/api`` endpoint map in ``src/asgi.py`` and
    the ``stream_production_events`` operation id all carry it. No consumer of
    ``/events/{tenant_id}`` was found anywhere in the w2v tree; the frontend's SSE
    is ``/api/projects/{id}/events`` on the orchestrator, not this route. A
    browser ``EventSource`` cannot set ``Authorization`` or ``X-Tenant-Id``
    headers at all, so no browser client could ever have authenticated here. The
    parameter is therefore kept only because removing a published path is a
    separate API decision, not because a caller is known to depend on it. It is a
    redundant echo of the caller's identity, validated on every request.

    Super-admin "browse as tenant" does NOT survive this check over a Keycloak
    JWT. ``_apply_effective_tenant`` (``w2v_common/auth/middleware.py:292``) is
    invoked from exactly one site (``:758``), inside the license-token branch, and
    only for subjects listed in ``AUTH_DELEGATING_SUBJECTS``. The Keycloak-JWT
    branch (``:660-698``) sets ``request.state.tenant_id`` straight from the
    ``tenant_id`` claim (falling back to ``sub``) and never looks at
    ``X-Active-Tenant-Id``. Consequence: a super-admin browsing as tenant B over a
    Keycloak JWT still carries their OWN tenant in ``request.state.tenant_id``,
    so ``GET /events/<tenant-B>`` returns 403 for them. That is the intended
    trade: this route has no admin-impersonation path, and the fix for an admin
    who needs one is an explicit elevation check, not a looser comparison.

    403, not 404: the tenant id is not a secret, the subscription simply is not
    the caller's to open. Same rationale as ai_content_generation_service PR #91.

    Args:
        tenant_id: Tenant whose events to subscribe to. Must be the caller's own.
        request: FastAPI request (used for disconnect detection and Redis access).

    Returns:
        StreamingResponse with SSE-formatted event data.

    Raises:
        HTTPException: 403 if the caller has no resolved tenant, or if
            ``tenant_id`` is not the caller's tenant.
    """
    denied = HTTPException(
        status_code=403,
        detail="Not authorized to stream events for this tenant",
    )
    try:
        caller_tenant = require_request_tenant(request)
    except HTTPException as exc:
        # Fail CLOSED. An unresolved identity must never be matchable by a path
        # parameter, whatever the path parameter says.
        logger.warning(
            "Rejected SSE subscription with no resolved caller tenant: requested=%s",
            tenant_id,
        )
        raise denied from exc
    try:
        requested_tenant = parse_request_tenant(tenant_id, source="events path parameter")
    except ValueError as exc:
        # "anonymous", "", a slug, or the nil uuid in the path: not a tenant, so
        # there is nothing it could legitimately equal.
        logger.warning("Rejected SSE subscription for a non-tenant path parameter")
        raise denied from exc
    if requested_tenant != caller_tenant:
        logger.warning(
            "Rejected cross-tenant SSE subscription: caller=%s requested=%s",
            caller_tenant,
            requested_tenant,
        )
        raise denied

    logger.info("SSE stream opened: tenant=%s", requested_tenant)
    redis_client = getattr(request.app.state, "redis_client", None)
    if not redis_client:
        logger.warning("SSE stream unavailable: Redis not connected (tenant=%s)", requested_tenant)
        return StreamingResponse(
            iter(['data: {"error": "Redis not available"}\n\n']),
            media_type="text/event-stream",
        )

    return StreamingResponse(
        _sse_event_generator(redis_client, requested_tenant, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
