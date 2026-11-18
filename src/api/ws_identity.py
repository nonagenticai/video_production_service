"""Handshake identity for the compile-progress WebSocket.

WHY THIS MODULE EXISTS AT ALL — the outage it prevents.

``AuthMiddleware`` is a ``BaseHTTPMiddleware`` subclass
(``w2v_common/auth/middleware.py``), and Starlette's
``BaseHTTPMiddleware.__call__`` returns early for any scope whose ``type`` is
not ``"http"``. ``dispatch`` therefore NEVER runs on a ``websocket`` scope, so
nothing in the fleet auth stack stamps ``websocket.state.tenant_id`` on a
handshake. Mounting ``LICENSE_GATE_WS`` on a WS route without first stamping an
identity is consequently not "fail closed": the gate resolves NO tenant, derives
``none`` and closes **1008 ``license_none`` for EVERY caller, including a fully
entitled one**. That is a total outage of the socket, not an enforcement.

So identity has to be resolved here, from the handshake itself, BEFORE the gate
runs. The shape mirrors the merged, proven implementation in
``tts_service/src/api/ws.py`` (``_handshake_license_token`` / ``_resolve_identity``,
PR #116) rather than inventing a second dialect: same header precedence, same
JWT-shape exclusion, same fail-closed introspection, same subject fallback. Two
hand-written spellings of "who is on this socket" is exactly how a socket starts
admitting a caller the HTTP API refuses.

⛔ WHAT IS DELIBERATELY NOT AN IDENTITY HERE.

* **A raw ``X-Tenant-Id: <uuid>`` header.** A bare (non-``Bearer``) value is
  accepted from NOBODY. It is unauthenticated caller-supplied text, so trusting
  it would mint a header-spoofable identity lane that the HTTP gate does not
  have — any client could name any tenant and be gated as that tenant. Only the
  ``X-Tenant-Id: Bearer <token>`` *credential* shape is read, and only as a
  token to be introspected; the tenant that comes back is the INTROSPECTED one,
  never the header text. See ``w2v_common/auth/entitlement.py`` on the same
  point: "a gate that trusts a forgeable tenant is worse than one that closes".
* **A ``?license=<token>`` query parameter.** Not offered, mirroring
  ``tts_service``, whose docstring states the reason: a query-string token lands
  in access logs and proxy history. The usual counter-argument — that browsers
  cannot set headers on the ``WebSocket`` constructor, so a query token is the
  only browser-reachable lane — does not apply to THIS socket: a fleet-wide
  search finds zero browser consumers of ``/api/compile/jobs/{job_id}/ws`` (the
  frontend's progress transports are SSE and polling). Adding a logged-credential
  lane for a consumer that does not exist is pure downside. If a browser
  consumer is ever genuinely needed, mint a short-lived single-use ticket from
  an authenticated REST call — do not relax this to a query parameter.
* **``DEFAULT_TENANT_ID`` / any fallback.** There is none; w2v-common 0.67.0
  deleted every tenant fallback, and on a pub/sub channel with no RLS backstop a
  default tenant is a cross-tenant read.

WHAT HAPPENS WHEN RESOLUTION FAILS. Nothing, here. This module stamps or it does
not; it never closes the socket itself. An unresolved handshake simply reaches
``require_license_ws`` with no tenant, which derives ``none`` and closes 1008
``license_none`` — **the gate's own refusal path**, carrying the same
``license_<state>`` vocabulary the HTTP path puts in its 402 body. Raising from
here instead would produce a second, divergent refusal shape for the same
condition, and would also close the socket while the gate is DISARMED, silently
turning a dormant flag into an outage of its own.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from collections.abc import Mapping

from fastapi import WebSocket

# Identity goes through this service's own re-export seam rather than reaching
# into w2v_common directly — ``src.auth.middleware`` is the single place this
# repo binds to the shared auth implementation.
from src.auth.middleware import _introspect_license, _looks_like_jwt

logger = logging.getLogger(__name__)

__all__ = ["resolve_ws_identity", "stamp_ws_identity"]


def _canonical_tenant(value: object) -> str | None:
    """Return ``value`` as canonical lowercase-dashed uuid text, else ``None``.

    Since migration ``004`` a compilation job's ``tenant_id`` is a postgres
    ``uuid``, which postgres renders back in exactly one spelling, and the gate
    stringifies whatever it is handed. Folding here means an uppercase, braced or
    ``urn:uuid:`` spelling of the same tenant produces the SAME licence-cache key
    as the HTTP path, instead of a second cache entry and a second upstream call.

    A non-uuid (a legacy slug, a leaked ``Bearer …`` value, a ``MagicMock`` from a
    test fake) returns ``None``, and an unresolved tenant is refused by the gate.
    The nil uuid is rejected by value: it is a removed sentinel that is still
    *registered* in ``licenses.tenants``, so it would pass an introspection and
    then mis-attribute the socket to a shared bucket (see
    ``src/api/tenant.py::NilTenantIdError``).
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = _uuid.UUID(value.strip())
    except ValueError:
        return None
    if parsed.int == 0:
        return None
    return str(parsed)


def _state_tenant_id(websocket: WebSocket) -> str | None:
    """Return an already-stamped ``websocket.state.tenant_id``, else ``None``.

    Only a real, non-empty, uuid-shaped string counts. This is the seam an
    ASGI-level authenticator (or a test) uses to supply identity without a token;
    a ``BaseHTTPMiddleware`` can never populate it, which is the whole premise of
    this module.
    """
    state = getattr(websocket, "state", None)
    if state is None:
        return None
    return _canonical_tenant(getattr(state, "tenant_id", None))


def _state_subject_id(websocket: WebSocket) -> str | None:
    """Return a non-empty already-stamped ``websocket.state.subject_id``, else ``None``.

    The ``isinstance`` guard is load-bearing rather than defensive noise: a test
    fake whose ``state`` is a ``MagicMock`` answers every attribute with a truthy
    ``MagicMock``, and without the guard that object's ``str()`` would be sent
    upstream as a subject id.
    """
    state = getattr(websocket, "state", None)
    if state is None:
        return None
    subject = getattr(state, "subject_id", None)
    if isinstance(subject, str) and subject.strip():
        return subject.strip()
    return None


def _bearer(headers: Mapping[str, str], name: str) -> str | None:
    """Return the ``Bearer`` value of header ``name`` (lowercase), else ``None``.

    A header that is present but NOT ``Bearer``-prefixed yields ``None`` — that is
    the anti-spoofing rule for ``X-Tenant-Id`` in one line: a bare tenant uuid in
    that header is not a credential and is never read as one.

    Real Starlette ``Headers`` are case-insensitive; the unit fakes pass a plain
    ``dict``, so both spellings are probed and the two behave identically.
    """
    raw = headers.get(name) or headers.get(name.title())
    if raw and raw.lower().startswith("bearer "):
        return raw[7:].strip() or None
    return None


def _handshake_license_token(websocket: WebSocket) -> str | None:
    """Extract an opaque licence token from this handshake, canonical rule.

    ⚠️ MIRRORS ``AuthMiddleware._extract_license_token`` and must not drift from
    it. It cannot simply delegate: that method takes a ``Request`` and is reached
    only from ``dispatch``, which never runs on a ``websocket`` scope.

    TWO shapes are accepted, in this order:

    1. ``X-Tenant-Id: Bearer <token>`` — w2v's own legacy convention. DEPRECATED
       but load-bearing, and checked FIRST on purpose: ``BaseServiceClient`` puts
       the CALLER's delegated token here while its constructor-baked
       SERVICE-ACCOUNT token sits on ``Authorization``. Preferring
       ``Authorization`` would silently re-attribute every delegated call from
       the customer's tenant to the service account's tenant.
    2. ``Authorization: Bearer <opaque>`` — the PLATFORM convention, and the only
       shape the ``license-authz`` ext-authz gate can read.

       Reached ONLY when the bearer is not JWT-shaped. A JWT on ``Authorization``
       is an upstream caller's own identity token and must NEVER be submitted to
       licence introspection (the 2026-04-26 prod 401s). ``_looks_like_jwt`` is
       imported from w2v_common rather than re-implemented so that test cannot
       drift either.
    """
    headers = getattr(websocket, "headers", None) or {}
    token = _bearer(headers, "x-tenant-id")
    if token:
        return token
    bearer = _bearer(headers, "authorization")
    if bearer and not _looks_like_jwt(bearer):
        return bearer
    return None


async def resolve_ws_identity(websocket: WebSocket) -> tuple[str | None, str | None]:
    """Resolve ``(tenant_id, subject_id)`` for this handshake; ``(None, None)`` ⇒ unknown.

    Resolution order:

    1. ``websocket.state.tenant_id`` — set by an ASGI-level authenticator if one is
       ever put in front of this route (or by a test).
    2. An opaque licence token from the handshake headers, **introspected**. The
       tenant returned is the one the licences service asserts for that token,
       never a value the caller typed into a header.

    Neither ⇒ ``(None, None)``, and the caller leaves the socket unstamped so the
    entitlement gate refuses it on its own terms.

    The SUBJECT rides along because the gate wants the same pair ``AuthMiddleware``
    stamps on an HTTP request, and it is only knowable on the same introspection
    call that yields the tenant — resolving it separately would introspect twice.
    Its fallback to the tenant id mirrors ``w2v_common.auth.middleware`` exactly (a
    tenant-scoped service-account token legitimately carries no subject), so the
    socket and the API derive the SAME licence state for the same caller.
    """
    tenant_id = _state_tenant_id(websocket)
    if tenant_id is not None:
        return tenant_id, _state_subject_id(websocket) or tenant_id

    token = _handshake_license_token(websocket)
    if not token:
        return None, None
    try:
        result = await _introspect_license(token)
    except Exception:
        # FAIL CLOSED on ANY introspection error. The breadth is the point, not
        # laziness: this function's contract is "no verified tenant ⇒ no
        # identity", and enumerating exception types would let an unforeseen one
        # (a misconfigured LICENSE_INTROSPECT_URL raising httpx.InvalidURL, a
        # TypeError from a changed w2v_common signature, an AttributeError in the
        # cache layer) propagate out of the dependency, making the refusal shape
        # depend on the ASGI server's unhandled-exception policy rather than on
        # the licence gate.
        #
        # `except Exception` does NOT catch asyncio.CancelledError (a
        # BaseException since 3.8), so shutdown and task-cancellation semantics
        # are untouched. The token is never logged — `logger.exception` renders
        # the traceback only, and `_introspect_license` takes care never to put
        # the token in a message. Ruff's BLE001 exempts a handler that calls
        # `logger.exception`; keep that call if this block is ever edited.
        logger.exception("WebSocket handshake: license introspection raised")
        return None, None
    if not result or not result.get("active"):
        return None, None
    # An active licence MUST carry a tenant (the fleet three-piece auth
    # invariant). An empty or non-uuid one would re-open the exact hole this
    # function exists to close, so it resolves to no identity at all.
    tenant_id = _canonical_tenant(result.get("tenant_id"))
    if tenant_id is None:
        return None, None
    subject = result.get("subject_id")
    subject_id = subject.strip() if isinstance(subject, str) and subject.strip() else tenant_id
    return tenant_id, subject_id


async def stamp_ws_identity(websocket: WebSocket) -> None:
    """FastAPI dependency: stamp the introspected identity onto ``websocket.state``.

    ⚠️ MOUNT ORDER IS THE WHOLE POINT. This must be listed BEFORE
    ``LICENSE_GATE_WS`` in the router's ``dependencies``; FastAPI solves a route's
    dependencies sequentially in list order and writes to ``websocket.state`` are
    visible to every later dependency, so "resolve → stamp → gate" is expressed by
    the ordering of that list and by nothing else. Reversed, the gate reads an
    empty state, derives ``none`` and closes every socket.

    It NEVER raises and NEVER closes. Refusal is the gate's job, so that a
    handshake with no identity produces the same 1008 ``license_<state>`` close an
    unentitled one does, and so that the socket still connects while
    ``W2V_LICENSE_GATE_ENABLED`` is unset — the flag's dormancy is a property of
    the gate, and an identity step that closed on its own would quietly revoke it.
    """
    tenant_id, subject_id = await resolve_ws_identity(websocket)
    if tenant_id is None:
        logger.debug("WebSocket handshake carried no verifiable licence identity")
        return
    websocket.state.tenant_id = tenant_id
    websocket.state.subject_id = subject_id
