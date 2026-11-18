"""``POST /api/record/animation`` honours the orchestrator's ``Idempotency-Key``.

The orchestrator sends ``Idempotency-Key: {run_id}:ken_burns:{media_id}`` (and
``:title_card:``). Before vp composed ``IdempotencySettings`` the canonical
middleware was never registered, so a resumed run re-POSTed and paid for a second
recording. These tests drive the REAL w2v_common middleware over the REAL route
with a fakeredis-backed store and count recorder calls.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from w2v_common.cache.redis_client import RedisClient
from w2v_common.middleware import IdempotencyMiddleware
from w2v_common.testing import inject_fakeredis

from src.api.dependencies import get_recording_job_manager
from src.api.routes.recording import router
from tests.conftest import TENANT_U7, bind_test_tenant
from tests.unit.recording.test_routes import _fake_create_task, _make_mock_jm

_URL = "/api/record/animation"
_HTML = "<html><body><script>window.RENDER_DURATION_MS=1000</script></body></html>"
_BODY = {"html": _HTML, "part_id": "kb_1"}


def _app() -> tuple[FastAPI, AsyncMock]:
    service = AsyncMock()

    async def _slow_create(*_args, **_kwargs) -> str:
        # Hold the first request in flight so the duplicate genuinely overlaps it.
        await asyncio.sleep(0.3)
        return str(uuid.uuid4())

    service.create_animation_job = AsyncMock(side_effect=_slow_create)
    app = FastAPI()
    app.include_router(router)
    app.state.recorder_service = service
    app.state.storage_service = AsyncMock()
    store = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(store)
    app.state.redis = store
    jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: jm
    # Registration order = canonical stack: Idempotency INNER of the tenant binder
    # (create_app puts it inner of AuthMiddleware), so the tenant is on state.
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=86400)
    bind_test_tenant(app, TENANT_U7)
    return app, service


async def _post_pair(key_a: str, key_b: str) -> tuple[list[httpx.Response], AsyncMock]:
    app, service = _app()
    transport = httpx.ASGITransport(app=app)
    with patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task):
        async with httpx.AsyncClient(transport=transport, base_url="http://vp") as client:
            responses = await asyncio.gather(
                client.post(_URL, json=_BODY, headers={"Idempotency-Key": key_a}),
                client.post(_URL, json=_BODY, headers={"Idempotency-Key": key_b}),
            )
    return list(responses), service


@pytest.mark.asyncio
async def test_concurrent_same_key_records_once():
    key = f"{uuid.uuid4()}:ken_burns:media-1"
    responses, service = await _post_pair(key, key)
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert service.create_animation_job.await_count == 1
    # The duplicate attaches to the SAME job instead of a second recording.
    assert responses[0].json()["job_id"] == responses[1].json()["job_id"]
    assert sorted(r.headers.get("Idempotent-Replay", "") for r in responses) == ["", "true"]


@pytest.mark.asyncio
async def test_different_keys_record_twice():
    run = uuid.uuid4()
    responses, service = await _post_pair(f"{run}:ken_burns:media-1", f"{run}:title_card:media-1")
    assert [r.status_code for r in responses] == [200, 200]
    assert service.create_animation_job.await_count == 2
    assert responses[0].json()["job_id"] != responses[1].json()["job_id"]


def test_settings_force_idempotency_on(monkeypatch):
    monkeypatch.setenv("IDEMPOTENCY_ENABLED", "false")
    from src.config import Settings

    assert Settings().idempotency.enabled is True


def test_production_app_registers_idempotency_middleware():
    from src.asgi import app

    assert any(m.cls is IdempotencyMiddleware for m in app.user_middleware)


@pytest.mark.asyncio
async def test_init_redis_exposes_store_where_middleware_reads_it():
    # The middleware reads ``app.state.redis``; vp historically only set
    # ``redis_client``, which would leave the middleware falling open forever.
    from src import asgi

    store = AsyncMock()
    app = FastAPI()
    with (
        patch.object(asgi, "init_redis", AsyncMock(return_value=store)),
        patch.object(asgi, "get_redis_client", return_value=store),
    ):
        await asgi._init_redis(app)
    assert app.state.redis is store


# ---------------------------------------------------------------------------
# Orchestrator-shaped request through the REAL factory stack (Auth -> Idempotency)
# ---------------------------------------------------------------------------
# What w2v_orchestrator's VideoProductionClient (LicensedServiceClient ->
# w2v_common BaseServiceClient + identity_event_hook) puts on the wire for a
# ken_burns record_animation: the SA licence token as ``X-Tenant-Id: Bearer``,
# and — video_production is a DELEGATING destination — the run's real tenant as
# ``X-Effective-Tenant-Id``. vp's AuthMiddleware introspects the token, honours
# the delegation (SA subject is in AUTH_DELEGATING_SUBJECTS, as in gitops), and
# sets ``request.state.tenant_id`` BEFORE IdempotencyMiddleware runs.
_SA_TENANT = "019de900-0000-7000-8000-0000000005a0"
_SA_SUBJECT = "8382f0b3-c082-4f97-8d47-120ce5a18e05"
_RUN_TENANT = uuid.UUID("019de900-0000-7000-8000-0000000005a1")
_SA_TOKEN = "opaque-sa-licence-token"


def _orchestrator_headers(idempotency_key: str | None) -> dict[str, str]:
    headers = {
        "X-Tenant-Id": f"Bearer {_SA_TOKEN}",
        "X-Effective-Tenant-Id": str(_RUN_TENANT),
        "X-License-Id": "lic-sa",
        "X-Subject-Id": _SA_SUBJECT,
        "X-Correlation-Id": str(uuid.uuid4()),
        "X-Stage": "ken_burns",
        "X-Request-Id": str(uuid.uuid4()),
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _factory_app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, AsyncMock]:
    import w2v_common.auth.middleware as w2v_auth
    from w2v_common.asgi import create_app

    from src.config import Settings

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_DELEGATING_SUBJECTS", _SA_SUBJECT)
    w2v_auth._get_license_cache().clear()

    async def _introspect(token: str) -> dict[str, object]:
        assert token == _SA_TOKEN
        return {
            "active": True,
            "tenant_id": _SA_TENANT,
            "subject_id": _SA_SUBJECT,
            "license_id": "lic-sa",
            "roles": ["service"],
        }

    monkeypatch.setattr(w2v_auth, "_introspect_license", _introspect)
    settings = Settings()
    assert settings.idempotency.enabled is True
    app = create_app(settings, routers=[router], auth_enabled=True)
    template, service = _app()
    app.state.recorder_service = service
    app.state.storage_service = template.state.storage_service
    app.state.redis = template.state.redis
    jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: jm
    return app, service


async def _post_orchestrator_pair(app: FastAPI, key: str | None) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    with patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task):
        async with httpx.AsyncClient(transport=transport, base_url="http://vp") as client:
            return list(
                await asyncio.gather(
                    client.post(_URL, json=_BODY, headers=_orchestrator_headers(key)),
                    client.post(_URL, json=_BODY, headers=_orchestrator_headers(key)),
                )
            )


@pytest.mark.asyncio
async def test_orchestrator_shaped_request_is_deduped_not_refused(monkeypatch):
    app, service = _factory_app(monkeypatch)
    # Factory order: Auth must be OUTER of Idempotency (it supplies the tenant).
    from w2v_common.auth.middleware import AuthMiddleware

    order: list[object] = [m.cls for m in app.user_middleware]
    assert order.index(AuthMiddleware) < order.index(IdempotencyMiddleware), order

    responses = await _post_orchestrator_pair(app, f"{uuid.uuid4()}:ken_burns:media-1")
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert service.create_animation_job.await_count == 1
    assert responses[0].json()["job_id"] == responses[1].json()["job_id"]
    # The delegated run tenant (not the SA tenant) owns the job.
    assert service.create_animation_job.await_args.kwargs["tenant_id"] == _RUN_TENANT


@pytest.mark.asyncio
async def test_orchestrator_shaped_request_without_key_is_unaffected(monkeypatch):
    app, service = _factory_app(monkeypatch)
    responses = await _post_orchestrator_pair(app, None)
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert service.create_animation_job.await_count == 2
    assert all("Idempotent-Replay" not in r.headers for r in responses)


# ---------------------------------------------------------------------------
# A failed STARTUP connect must not disable dedupe for the life of the pod
# ---------------------------------------------------------------------------
# These run the REAL ``asgi._init_redis`` -> real ``init_redis`` -> real
# ``RedisClient.connect()``; only the socket is fake (a fakeredis server whose
# ``connected`` flag is the "is Redis reachable" switch).
_STARTUP_WARNING = (
    "idempotency dedupe unavailable until Redis is reachable; "
    "duplicate recordings may be paid twice"
)


@pytest.fixture
def unreachable_redis(monkeypatch: pytest.MonkeyPatch):
    import fakeredis
    import fakeredis.aioredis
    import redis.asyncio as aioredis
    import w2v_common.cache.redis_client as rc

    server = fakeredis.FakeServer()
    server.connected = False

    def _from_url(_url: str, **kwargs: object) -> aioredis.ConnectionPool:
        return aioredis.ConnectionPool(
            connection_class=fakeredis.aioredis.FakeAsyncRedisConnection,
            server=server,
            decode_responses=kwargs.get("decode_responses", True),
        )

    def _reset_singleton() -> None:
        rc._redis_client = None
        rc.get_redis_client.cache_clear()

    _reset_singleton()
    monkeypatch.setattr(rc.ConnectionPool, "from_url", staticmethod(_from_url))
    yield server
    _reset_singleton()


async def _boot_with_failed_redis_connect(
    caplog: pytest.LogCaptureFixture,
) -> tuple[FastAPI, AsyncMock]:
    from src import asgi

    service = AsyncMock()
    service.create_animation_job = AsyncMock(side_effect=lambda *_a, **_k: str(uuid.uuid4()))
    app = FastAPI()
    app.include_router(router)
    app.state.recorder_service = service
    app.state.storage_service = AsyncMock()
    with caplog.at_level("WARNING", logger="src.asgi"):
        await asgi._init_redis(app)  # the startup connect FAILS here
    jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: jm
    app.add_middleware(IdempotencyMiddleware, ttl_seconds=86400)
    bind_test_tenant(app, TENANT_U7)
    return app, service


async def _post_twice(app: FastAPI, key: str) -> list[httpx.Response]:
    transport = httpx.ASGITransport(app=app)
    with patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task):
        async with httpx.AsyncClient(transport=transport, base_url="http://vp") as client:
            headers = {"Idempotency-Key": key}
            return [
                await client.post(_URL, json=_BODY, headers=headers),
                await client.post(_URL, json=_BODY, headers=headers),
            ]


def _startup_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == "src.asgi" and r.levelname == "WARNING"
    ]


@pytest.mark.asyncio
async def test_dedupe_recovers_after_failed_startup_connect(unreachable_redis, caplog):
    app, service = await _boot_with_failed_redis_connect(caplog)
    # Per-pod fallback for the recorder/lease wiring is unchanged.
    assert app.state.redis_client is None

    unreachable_redis.connected = True  # the blip is over; NO pod restart

    responses = await _post_twice(app, f"{uuid.uuid4()}:ken_burns:media-1")
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert service.create_animation_job.await_count == 1
    assert responses[0].json()["job_id"] == responses[1].json()["job_id"]
    assert responses[1].headers.get("Idempotent-Replay") == "true"


@pytest.mark.asyncio
async def test_redis_never_reachable_falls_open_and_warns_once(unreachable_redis, caplog):
    app, service = await _boot_with_failed_redis_connect(caplog)

    responses = await _post_twice(app, f"{uuid.uuid4()}:ken_burns:media-1")
    # Fall open (the documented w2v_common trade-off): both succeed, both record.
    assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
    assert service.create_animation_job.await_count == 2
    assert all("Idempotent-Replay" not in r.headers for r in responses)

    warnings = _startup_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert _STARTUP_WARNING in warnings[0]
