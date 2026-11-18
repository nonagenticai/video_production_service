"""The entitlement gate on ``/api/compile/jobs/{job_id}/ws`` is the WS one, and it works.

THE DEFECT THIS FILE PINS. The WebSocket route was registered on the same
``APIRouter`` as every ``/api/compile`` HTTP route, so it inherited that
router's ``LICENSE_GATE``. ``require_license`` is annotated ``(request:
Request)``; FastAPI fills a ``Request``-annotated parameter only when the
connection IS a ``Request``, and on a WebSocket route it is a ``WebSocket``. The
parameter was therefore never filled and the dependency raised
``TypeError: require_license() missing 1 required positional argument: 'request'``
on EVERY handshake — with the flag armed AND with it off. The route was not
"gated"; it was a 500 for everyone.
:func:`test_the_http_gate_is_unmountable_on_a_websocket_route` reproduces that
from first principles so the claim is measured here, not quoted from an audit.

WHY THE ASSERTIONS DRIVE A REAL HANDSHAKE. This FastAPI build includes
sub-routers lazily (``_IncludedRouter``), so a parent router's ``dependencies=``
are applied when a connection is matched, not baked into the route objects at
import time. Walking ``route.dependant.dependencies`` reports a live-gated route
as ungated — a false negative. A handshake is the only honest probe, so every
test below opens one.

⚠️ THE GATE ALONE WOULD BE AN OUTAGE, AND THAT IS WHAT THE TOKEN TESTS PIN.
``AuthMiddleware`` is a ``BaseHTTPMiddleware``, which Starlette runs on ``http``
scope only, so NOTHING in the fleet auth stack stamps ``websocket.state.tenant_id``
on a handshake. A route carrying only ``LICENSE_GATE_WS`` therefore resolves no
tenant, derives ``none`` and closes 1008 ``license_none`` for **every caller,
entitled or not** — a total outage of the socket, on a service armed in
production. ``router_ws`` closes that by mounting ``stamp_ws_identity`` BEFORE the
gate: it introspects the handshake licence token and stamps the tenant the
licences service asserts. :func:`test_entitled_caller_with_a_handshake_token_is_accepted`
is the test that would have caught the outage — it drives a real handshake with a
real token through the real dependency chain, so it fails the moment the ordering
or the stamping regresses.

⛔ AND IDENTITY IS NEVER TAKEN FROM A RAW HEADER.
:func:`test_a_bare_x_tenant_id_header_is_not_an_identity` asserts the socket is
refused AND that introspection was never even attempted for a caller who merely
names a tenant in ``X-Tenant-Id`` without a ``Bearer`` credential. Trusting that
text would mint a header-spoofable identity lane the HTTP gate does not have.

The older tests below stamp ``scope["state"]["tenant_id"]`` from a pure-ASGI
middleware instead of presenting a token; they isolate the GATE's behaviour from
the identity step, which is why they are kept alongside the token tests rather
than replaced by them.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import APIRouter, FastAPI, WebSocket
from starlette.testclient import WebSocketDenialResponse
from w2v_common.auth import LICENSE_GATE, clear_license_status_cache
from w2v_common.auth.entitlement import (
    LICENSE_WS_CLOSE_POLICY,
    LICENSE_WS_CLOSE_UNAVAILABLE,
)
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.routes.compilation import router_ws
from src.api.ws import ConnectionManager, compilation_job_channel
from tests.conftest import TENANT_U7, TENANT_U8

pytestmark = pytest.mark.timeout(20)

JOB_ID = "job-under-test"
WS_PATH = f"/api/compile/jobs/{JOB_ID}/ws"
SUBJECT = "user-under-test"
#: A second, fully entitled tenant. The cross-tenant test impersonates OUR
#: tenant and points the job at this one — an attacker with a perfectly valid
#: licence, which is the only interesting case.
OTHER_TENANT = TENANT_U7


# ---------------------------------------------------------------------------
# Fakes — same shapes as tests/unit/test_license_gate.py, kept local so the two
# files cannot drift into disagreeing about what a licence record looks like.
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
        tenant_id=str(TENANT_U8),
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


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


class _FakeIntrospector:
    """Stand-in for ``_introspect_license``: a token -> introspection map.

    It RECORDS every token it is handed, which is the observable
    :func:`test_a_bare_x_tenant_id_header_is_not_an_identity` asserts on — "the
    socket was refused" alone would also pass if the header text had been
    submitted upstream as a credential and merely rejected there. The recording
    proves the header was never treated as one.
    """

    def __init__(self, by_token: dict[str, dict[str, Any]] | None = None) -> None:
        self.by_token = by_token or {}
        self.calls: list[str] = []

    async def __call__(self, token: str) -> dict[str, Any] | None:
        self.calls.append(token)
        return self.by_token.get(token)


#: An opaque licence token, deliberately NOT JWT-shaped: a three-segment bearer
#: on ``Authorization`` is an upstream caller's own identity token and must never
#: reach licence introspection.
LICENCE_TOKEN = "lic-opaque-handshake-token"


def _introspects_to_our_tenant() -> _FakeIntrospector:
    return _FakeIntrospector(
        {LICENCE_TOKEN: {"active": True, "tenant_id": str(TENANT_U8), "subject_id": SUBJECT}}
    )


def _introspection(monkeypatch, fake: _FakeIntrospector) -> _FakeIntrospector:
    """Patch the name ``src.api.ws_identity`` actually calls, not its origin."""
    monkeypatch.setattr("src.api.ws_identity._introspect_license", fake)
    return fake


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches 60s per (tenant, app, subject); without this the second
    test in the process would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


class _JobStore:
    """A ``CompilationJobManager.get_job`` that emulates the RLS predicate.

    The real ownership rule is postgres RLS: ``get_job`` carries NO tenant
    predicate, and under ``TENANT_RLS_MODE=enforce`` a foreign row is simply not
    visible, so the query returns ``None``. A unit test has no postgres, so this
    stands in for the DATABASE, not for the check — it returns the row only when
    the tenant bound into the identity ContextVar (which is what
    ``get_db_session`` binds the GUC from) matches the job's owner. Reading the
    ContextVar rather than an argument is deliberate: it is the same value
    postgres would be deciding on, so a regression that forgot to seed the
    context fails here exactly as it would fail in production.
    """

    def __init__(self, owner: str | None) -> None:
        self.owner = owner
        self.bound_tenants: list[str | None] = []

    async def get_job(self, _db: Any, job_id: str) -> Any:
        from w2v_common.auth.identity_context import get_tenant_id_from_context

        bound = get_tenant_id_from_context()
        self.bound_tenants.append(str(bound) if bound else None)
        if self.owner is None or bound is None or str(bound) != self.owner:
            return None
        return object()


@pytest.fixture(autouse=True)
def job_store(monkeypatch):
    """Every socket in this file asks "is this job mine?" before subscribing.

    Default: the job belongs to ``TENANT_U8``, i.e. to the caller the entitled
    fixtures impersonate, so the licence-gate tests keep measuring the licence
    gate. ``job_store.owner`` is reassigned by the ownership tests.
    """
    from contextlib import asynccontextmanager

    store = _JobStore(owner=str(TENANT_U8))

    @asynccontextmanager
    async def _fake_session():
        yield object()

    monkeypatch.setattr("src.api.routes.compilation.get_db_session", _fake_session)
    monkeypatch.setattr(
        "src.api.routes._compilation_jobs.CompilationJobManager.get_job",
        store.get_job,
    )
    return store


def _stamp_ws_identity(app: FastAPI) -> None:
    """Stamp tenant/subject on the WEBSOCKET scope, the way handshake auth would.

    Deliberately a pure-ASGI middleware, not ``@app.middleware("http")``: the
    whole point of this file is that a ``BaseHTTPMiddleware`` is skipped on a
    ``websocket`` scope, so a test that stamped identity through one would prove
    nothing. ``HTTPConnection.state`` is a view over ``scope["state"]``, which is
    what ``require_license_ws`` reads as ``websocket.state.tenant_id``.
    """

    class _Stamp:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] == "websocket":
                state = scope.setdefault("state", {})
                state["tenant_id"] = str(TENANT_U8)
                state["subject_id"] = SUBJECT
            await self._inner(scope, receive, send)

    app.add_middleware(_Stamp)


def _app(*, stamp_identity: bool, redis_client: Any | None = None) -> FastAPI:
    """The REAL ``router_ws``, with a real ConnectionManager behind it.

    ``redis_client`` selects the transport the handler takes: ``None`` is the
    in-memory ``ConnectionManager`` fallback, a fake exercises the Redis pub/sub
    lane — the one with no RLS backstop, and therefore the one the ownership
    tests must drive.
    """
    application = FastAPI()
    application.include_router(router_ws)
    application.state.redis_client = redis_client
    application.state.ws_manager = ConnectionManager()
    if stamp_identity:
        _stamp_ws_identity(application)
    return application


def _was_accepted(client: Any, headers: dict[str, str] | None = None) -> bool:
    """``True`` if the gate let the handshake through, ``False`` if it closed it.

    The observable the accept-path tests assert on. ``websocket_connect`` raising
    ``WebSocketDisconnect`` at ``__enter__`` IS the refusal — the dependency runs
    before ``websocket.accept()``, so a refused peer never reaches the handler —
    and reaching the body means it did. Returned as a bool rather than left as an
    implicit "did not raise" so the assertion is explicit and a regression names
    the property that broke.
    """
    from starlette.websockets import WebSocketDisconnect

    try:
        with client.websocket_connect(WS_PATH, headers=dict(headers or {})) as ws:
            ws.send_text("ping")
    except WebSocketDisconnect, WebSocketDenialResponse:
        return False
    return True


def _closure(client: Any, headers: dict[str, str] | None = None) -> tuple[int, str]:
    """Open the socket and return the ``(code, reason)`` the server closed with."""
    from starlette.websockets import WebSocketDisconnect

    try:
        with client.websocket_connect(WS_PATH, headers=dict(headers or {})):
            pass
    except WebSocketDisconnect as exc:
        return exc.code, exc.reason or ""
    except WebSocketDenialResponse as exc:  # pragma: no cover - shape guard
        pytest.fail(f"handshake denied with HTTP {exc.status_code}, expected a close frame")
    pytest.fail("socket was ACCEPTED; expected the gate to close it")


# ---------------------------------------------------------------------------
# The defect, reproduced
# ---------------------------------------------------------------------------


def test_the_http_gate_is_unmountable_on_a_websocket_route():
    """``LICENSE_GATE`` on a WS route is a TypeError on every handshake.

    Not a style point and not "fails closed for the right reason": with the flag
    OFF — where ``require_license`` is a documented no-op — the socket still
    500s, because the dependency cannot even be CALLED. This is what
    ``/api/compile/jobs/{job_id}/ws`` did on origin/main.
    """
    from starlette.testclient import TestClient

    async def _echo(websocket: WebSocket) -> None:  # pragma: no cover - never reached
        await websocket.accept()

    broken = APIRouter(dependencies=list(LICENSE_GATE))
    broken.add_api_websocket_route("/boom", _echo)
    application = FastAPI()
    application.include_router(broken)

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'request'"):
        with TestClient(application).websocket_connect("/boom"):
            pass


# ---------------------------------------------------------------------------
# Dormant: the flag off is a real no-op, even with no identity at all
# ---------------------------------------------------------------------------


def test_handshake_succeeds_while_the_gate_is_disarmed(monkeypatch):
    """Flag off -> the socket connects. This is the non-vacuity anchor: it is the
    case the old HTTP-typed dependency ALSO broke, so it can only pass once the
    mechanism is right."""
    from starlette.testclient import TestClient

    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    assert _was_accepted(TestClient(_app(stamp_identity=False))) is True


# ---------------------------------------------------------------------------
# Armed
# ---------------------------------------------------------------------------


def test_armed_gate_accepts_an_entitled_tenant(monkeypatch):
    """Armed + a stamped, entitled tenant -> the socket is ACCEPTED.

    The pair to every close below: without it a gate that closed unconditionally
    would pass this file.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    assert _was_accepted(TestClient(_app(stamp_identity=True))) is True


@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
def test_armed_gate_closes_1008_with_the_licence_reason(monkeypatch, inventory, code):
    """An unentitled tenant gets a CLOSE carrying the same ``license_<state>``
    vocabulary the HTTP path puts in its 402 body."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    close_code, reason = _closure(TestClient(_app(stamp_identity=True)))

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert reason.startswith(code), reason


def test_armed_gate_closes_1013_when_entitlement_cannot_be_established(monkeypatch):
    """An upstream failure is not an allow, and not a policy close either: 1013
    ('try again later'), mirroring the HTTP path's 503 + Retry-After."""
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    close_code, reason = _closure(TestClient(_app(stamp_identity=True)))

    assert close_code == LICENSE_WS_CLOSE_UNAVAILABLE == 1013
    assert reason.startswith("license_unavailable"), reason


def test_armed_gate_closes_an_unauthenticated_socket(monkeypatch):
    """No handshake auth exists, so production stamps nothing and the gate closes.

    This is the CURRENT production shape, asserted rather than left implicit: a
    licences backend that would answer ``active`` is still refused, because with
    no tenant the gate never asks it. Deriving a tenant from a raw header instead
    would make this test pass by inventing a spoofable identity — which is
    exactly why it is not done.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    close_code, reason = _closure(TestClient(_app(stamp_identity=False)))

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert reason.startswith("license_none"), reason


# ---------------------------------------------------------------------------
# Handshake identity: the half that makes the gate usable rather than fatal
# ---------------------------------------------------------------------------


def test_entitled_caller_with_a_handshake_token_is_accepted(monkeypatch):
    """THE OUTAGE TEST. Armed gate + a real licence token + an entitled tenant
    -> the socket is ACCEPTED.

    Nothing else in this file fails if ``stamp_ws_identity`` is dropped from
    ``router_ws`` or ordered after ``LICENSE_GATE_WS``: every close-path test
    below still closes, and the ASGI-stamped accept test still passes because it
    bypasses the handshake entirely. This one goes through the real dependency
    chain with no stamping middleware, so it is the only assertion that dies when
    the socket starts refusing the customers who pay for it.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    introspector = _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    accepted = _was_accepted(
        TestClient(_app(stamp_identity=False)),
        headers={"x-tenant-id": f"Bearer {LICENCE_TOKEN}"},
    )

    assert accepted is True
    assert introspector.calls == [LICENCE_TOKEN], introspector.calls


def test_unentitled_caller_with_a_handshake_token_is_closed_1008(monkeypatch):
    """Identity resolves, entitlement does not -> 1008 ``license_none``.

    The complement of the test above: proves the accept is the GATE's verdict on
    a real tenant, not the identity step waving everyone through once a token
    parses. Same token, same introspection, opposite licence state.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=[]))

    close_code, reason = _closure(
        TestClient(_app(stamp_identity=False)),
        headers={"x-tenant-id": f"Bearer {LICENCE_TOKEN}"},
    )

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert reason.startswith("license_none"), reason


def test_a_bare_x_tenant_id_header_is_not_an_identity(monkeypatch):
    """A caller who merely NAMES a tenant is refused, and never introspected.

    ``X-Tenant-Id: <uuid>`` with no ``Bearer`` is unauthenticated caller-supplied
    text. Reading it as identity would mint a header-spoofable lane the HTTP gate
    does not have — any client could name any tenant and be gated as that tenant.
    The licences backend here would answer ACTIVE for that tenant, so a socket
    that trusted the header would be accepted; it is closed instead.

    ``introspector.calls == []`` is the second, stronger assertion: the header
    text was not even offered upstream as a credential.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    introspector = _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    close_code, reason = _closure(
        TestClient(_app(stamp_identity=False)),
        headers={"x-tenant-id": str(TENANT_U8)},
    )

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert reason.startswith("license_none"), reason
    assert introspector.calls == [], introspector.calls


def test_a_jwt_on_authorization_is_not_submitted_to_licence_introspection(monkeypatch):
    """A three-segment bearer is the caller's own IdP token, not a licence.

    Submitting it upstream is the 2026-04-26 prod-401 class. It resolves no
    identity, so the gate closes — but the load-bearing assertion is again that
    introspection was never attempted.
    """
    from starlette.testclient import TestClient

    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    introspector = _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    close_code, _reason = _closure(
        TestClient(_app(stamp_identity=False)),
        headers={"authorization": "Bearer aaa.bbb.ccc"},
    )

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert introspector.calls == [], introspector.calls


@pytest.mark.parametrize(
    "headers",
    [
        None,
        {"x-tenant-id": str(TENANT_U8)},
        {"x-tenant-id": f"Bearer {LICENCE_TOKEN}"},
    ],
    ids=["no-headers", "bare-tenant-header", "licence-token"],
)
def test_disarmed_gate_accepts_regardless_of_the_handshake(monkeypatch, headers):
    """Flag off -> the socket connects, whatever the handshake carried.

    The identity step must not smuggle in enforcement of its own: dormancy is a
    property of the gate, and a ``stamp_ws_identity`` that closed on an
    unresolvable handshake would revoke it silently — turning a flag nobody has
    flipped into an outage. The licences backend is wired to REVOKED here, so
    this also re-proves the gate itself is a genuine no-op while disarmed.
    """
    from starlette.testclient import TestClient

    _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    assert _was_accepted(TestClient(_app(stamp_identity=False)), headers=headers) is True


# ---------------------------------------------------------------------------
# Job ownership: authentication is not authorization
# ---------------------------------------------------------------------------


class _FakePubSub:
    def __init__(self, parent: _FakeRedis) -> None:
        self._parent = parent
        self._pending = list(parent.messages)

    async def subscribe(self, channel: str) -> None:
        self._parent.subscribed.append(channel)

    async def get_message(
        self,
        ignore_subscribe_messages: bool = True,
        timeout: float = 30.0,  # noqa: ASYNC109 - mirrors redis-py's own signature
    ) -> dict[str, Any] | None:
        if self._pending:
            return {"type": "message", "data": self._pending.pop(0)}
        # Idle, not blocked: the handler sends a keepalive ping, which is what
        # surfaces the client's disconnect and ends the test deterministically.
        await asyncio.sleep(0.01)
        return None

    async def unsubscribe(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _FakeRedis:
    """Records whether the socket ever reached the pub/sub lane at all.

    ``subscribed`` is the observable the cross-tenant test asserts is EMPTY.
    "The socket was closed" alone would also pass if the handler had subscribed
    first and closed afterwards — by which point the Redis channel had already
    been joined and a racing publish already delivered.
    """

    def __init__(self, messages: list[str] | None = None) -> None:
        self.messages = messages or []
        self.subscribed: list[str] = []
        self.pubsub_calls = 0

    def pubsub(self) -> _FakePubSub:
        self.pubsub_calls += 1
        return _FakePubSub(self)


def _entitled_handshake(monkeypatch) -> dict[str, str]:
    """Arm the gate and hand back the headers of a caller who is beyond reproach.

    Identity introspects to ``TENANT_U8`` and the licence is ACTIVE, so anything
    that refuses this caller is refusing on job ownership and on nothing else.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _introspection(monkeypatch, _introspects_to_our_tenant())
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))
    return {"x-tenant-id": f"Bearer {LICENCE_TOKEN}"}


def test_owner_tenant_is_subscribed_and_receives_job_events(monkeypatch, job_store):
    """The job is ours -> the socket subscribes to its channel and relays events.

    The non-vacuity partner of the refusal tests: without it, an ownership check
    that refused unconditionally would leave this file green while taking the
    feature away from every legitimate customer.
    """
    from starlette.testclient import TestClient

    headers = _entitled_handshake(monkeypatch)
    job_store.owner = str(TENANT_U8)
    redis = _FakeRedis(messages=['{"event": "progress", "percent": 42}'])

    with TestClient(_app(stamp_identity=False, redis_client=redis)).websocket_connect(
        WS_PATH, headers=dict(headers)
    ) as ws:
        assert json.loads(ws.receive_text()) == {"event": "progress", "percent": 42}

    assert redis.subscribed == [compilation_job_channel(JOB_ID)]
    # The ownership question was asked AS our tenant — proving the refusals below
    # come from the RLS predicate and not from an empty identity context.
    assert job_store.bound_tenants == [str(TENANT_U8)], job_store.bound_tenants


def test_another_tenants_job_is_refused_before_any_subscribe(monkeypatch, job_store):
    """THE CROSS-TENANT TEST. A valid, entitled licence does not buy someone
    else's job.

    Redis pub/sub has no RLS backstop, so before this check any entitled tenant
    who learned (or guessed) a job id streamed that job's live compilation
    progress. The caller here passes identity AND the licence gate cleanly; the
    only thing wrong is that the job belongs to ``OTHER_TENANT``.

    ``redis.subscribed == []`` is the load-bearing half: refusal must happen
    BEFORE the channel is joined, not after.
    """
    from starlette.testclient import TestClient

    headers = _entitled_handshake(monkeypatch)
    job_store.owner = str(OTHER_TENANT)
    redis = _FakeRedis(messages=['{"event": "progress", "percent": 42}'])

    close_code, reason = _closure(
        TestClient(_app(stamp_identity=False, redis_client=redis)), headers=headers
    )

    assert close_code == LICENSE_WS_CLOSE_POLICY == 1008
    assert reason.startswith("job_not_found"), reason
    assert redis.subscribed == []
    assert redis.pubsub_calls == 0
    assert job_store.bound_tenants == [str(TENANT_U8)], job_store.bound_tenants


def test_a_job_that_does_not_exist_closes_instead_of_hanging(monkeypatch, job_store):
    """An unknown job id is closed, not left open waiting for events forever.

    And it is closed with a reason BYTE-IDENTICAL to the cross-tenant one: a
    caller who could tell "no such job" from "not your job" could walk the id
    space and enumerate every tenant's jobs from an unprivileged licence.
    """
    from starlette.testclient import TestClient

    headers = _entitled_handshake(monkeypatch)
    job_store.owner = None
    redis = _FakeRedis()

    absent_code, absent_reason = _closure(
        TestClient(_app(stamp_identity=False, redis_client=redis)), headers=headers
    )

    job_store.owner = str(OTHER_TENANT)
    foreign_code, foreign_reason = _closure(
        TestClient(_app(stamp_identity=False, redis_client=_FakeRedis())), headers=headers
    )

    assert absent_code == foreign_code == 1008
    assert absent_reason == foreign_reason, "refusals must be indistinguishable"
    assert redis.subscribed == []
