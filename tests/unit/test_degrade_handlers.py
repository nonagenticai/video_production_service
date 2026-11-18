"""Regression tests for degrade handlers that could never fire.

WHY this module exists
----------------------
Six handlers in this service documented a graceful degrade -- "allowing
request", "falls back to in-memory tracking", "reports Redis unhealthy",
"returns ``None`` so one bad image cannot abort the compile", "emits a
``data: {"error": ...}`` SSE frame" -- and every one of them was **dead code**.

Each was written ``except (OSError, ConnectionError, TimeoutError, ...)``:

* redis-py declares its OWN ``ConnectionError`` and ``TimeoutError`` under
  ``redis.exceptions.RedisError(Exception)``. Same names as the builtins,
  entirely different classes, and no ``OSError`` ancestry.
* httpx declares ``HTTPStatusError``/``RequestError`` under
  ``httpx.HTTPError(Exception)``, which is likewise not an ``OSError``.

So the ``except`` clause named exceptions that the call inside the ``try`` can
never raise. The documented degrade never ran once in production; the real
behaviour was a hard failure. For ``src/middleware/rate_limiter.py`` -- ASGI
middleware on every ``/api/record`` and ``/api/compile`` request -- that meant a
Redis blip 500'd the entire recording and compilation surface, which is the
exact outage the fail-open branch was written to prevent.

These tests pin the *behaviour* (degrade actually happens) and, in
:func:`test_redis_and_httpx_errors_are_not_oserror`, the *fact* that made the
handlers dead -- so that "simplifying" the tuples back to builtins fails CI
instead of silently re-arming the outage.

Every case is hermetic: no real Redis, no network. Each fake is injected at the
seam the production code actually calls.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import redis.exceptions
from fastapi import FastAPI
from fastapi.testclient import TestClient
from redis.exceptions import RedisError
from starlette.requests import Request

from src.api import events as events_module
from src.api import health as health_module
from src.middleware import rate_limiter as rate_limiter_middleware
from src.services.compilation import _effects, _finalize
from src.services.compilation import _segment_builders as segment_builders
from src.services.compilation import rate_limiter as compilation_rate_limiter
from src.services.compilation._compiler_context import _JobContext
from src.services.recording.event_publisher import EventPublisher
from tests.conftest import TENANT_U4

# The three redis-py failures a caller can realistically see. Parametrising over
# all three proves the fix is the ``RedisError`` base, not a one-class patch.
_REDIS_FAULTS: tuple[type[RedisError], ...] = (
    redis.exceptions.ConnectionError,
    redis.exceptions.TimeoutError,
    redis.exceptions.RedisError,
)


# ---------------------------------------------------------------------------
# 1. The fact that made every handler dead code
# ---------------------------------------------------------------------------


def test_redis_and_httpx_errors_are_not_oserror() -> None:
    """Pin the class hierarchy that made the builtin-only tuples unreachable.

    Kept deliberately as a documentation assertion. If a future reader
    "simplifies" ``_REDIS_ERRORS`` back to ``(OSError, ConnectionError,
    TimeoutError)`` believing redis-py's exceptions are the builtins, this test
    is the one that explains why that is wrong.
    """
    assert not issubclass(redis.exceptions.RedisError, OSError)
    assert not issubclass(httpx.HTTPError, OSError)

    # redis-py's same-named exceptions are NOT the builtins.
    assert not issubclass(redis.exceptions.ConnectionError, ConnectionError)
    assert not issubclass(redis.exceptions.TimeoutError, TimeoutError)

    # ...and both really do descend from the bases the handlers now name.
    assert issubclass(redis.exceptions.ConnectionError, RedisError)
    assert issubclass(redis.exceptions.TimeoutError, RedisError)
    assert issubclass(httpx.HTTPStatusError, httpx.HTTPError)
    assert not issubclass(httpx.HTTPStatusError, OSError)


# ---------------------------------------------------------------------------
# Shared redis fakes (seam: ``client._ensure_connected().pipeline()``)
# ---------------------------------------------------------------------------


class _FakePipeline:
    """Async redis pipeline whose ``execute`` either returns or raises.

    Mirrors the real usage: the queueing calls are synchronous and chainable,
    only ``execute`` is awaited.

    Attributes:
        results: Values returned by ``execute`` when no error is configured.
        error: Exception raised by ``execute`` instead of returning results.
    """

    def __init__(
        self,
        results: list[object] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.results = results or []
        self.error = error

    def zremrangebyscore(self, *args: object, **kwargs: object) -> _FakePipeline:
        """Queue a ``ZREMRANGEBYSCORE`` (no-op)."""
        return self

    def zcard(self, *args: object, **kwargs: object) -> _FakePipeline:
        """Queue a ``ZCARD`` (no-op)."""
        return self

    def zadd(self, *args: object, **kwargs: object) -> _FakePipeline:
        """Queue a ``ZADD`` (no-op)."""
        return self

    def expire(self, *args: object, **kwargs: object) -> _FakePipeline:
        """Queue an ``EXPIRE`` (no-op)."""
        return self

    async def execute(self) -> list[object]:
        """Run the queued pipeline.

        Returns:
            The configured results.

        Raises:
            BaseException: The configured ``error``, when one is set.
        """
        if self.error is not None:
            raise self.error
        return self.results


class _FakeRedisClient:
    """Stand-in for ``w2v_common``'s ``RedisClient`` wrapper.

    Production code calls ``redis_client._ensure_connected()`` and then
    ``.pipeline()`` / ``.zrange()`` on the result, so this object serves as both
    the wrapper and the underlying client.

    Attributes:
        pipe: The pipeline handed back by :meth:`pipeline`.
    """

    def __init__(self, pipe: _FakePipeline) -> None:
        self.pipe = pipe

    def _ensure_connected(self) -> _FakeRedisClient:
        """Return the underlying client (itself)."""
        return self

    def pipeline(self) -> _FakePipeline:
        """Return the configured fake pipeline."""
        return self.pipe

    async def zrange(self, *args: object, **kwargs: object) -> list[tuple[str, float]]:
        """Return an empty sorted-set range (drives the 60 s retry-after default)."""
        return []


# ---------------------------------------------------------------------------
# 2. src/middleware/rate_limiter.py -- the ASGI middleware must FAIL OPEN
# ---------------------------------------------------------------------------


@pytest.fixture
def _middleware_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the middleware's window to 5 requests / 60 s.

    Patches ``get_settings`` in the middleware module -- the exact name the
    ``dispatch`` body resolves -- so the test never touches real config.
    """
    settings = MagicMock()
    settings.api.rate_limit_requests = 5
    settings.api.rate_limit_window_seconds = 60
    monkeypatch.setattr(rate_limiter_middleware, "get_settings", lambda: settings)


def _build_middleware_app(redis_client: object, hits: list[str]) -> FastAPI:
    """Build a minimal app behind ``RateLimitMiddleware``.

    Args:
        redis_client: Object placed on ``app.state.redis_client``.
        hits: List appended to whenever the downstream route actually executes.

    Returns:
        A FastAPI app exposing ``GET /api/record/ping`` (a rate-limited prefix).
    """
    app = FastAPI()

    @app.get("/api/record/ping")
    async def _ping() -> dict[str, str]:
        hits.append("downstream")
        return {"ok": "yes"}

    app.add_middleware(rate_limiter_middleware.RateLimitMiddleware)
    app.state.redis_client = redis_client
    return app


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
@pytest.mark.usefixtures("_middleware_limits")
def test_middleware_fails_open_on_redis_fault(fault: type[RedisError]) -> None:
    """A Redis fault must let the request through, not 500 the whole service.

    Before the fix the builtin-only tuple never matched ``redis.exceptions.*``,
    so ``pipe.execute()`` escaped ``dispatch`` and every request to
    ``/api/record/*`` and ``/api/compile/*`` returned 500 while Redis was down.

    Args:
        fault: The redis-py exception class raised by the pipeline.
    """
    hits: list[str] = []
    client = _FakeRedisClient(_FakePipeline(error=fault("redis is down")))
    app = _build_middleware_app(client, hits)

    response = TestClient(app).get("/api/record/ping")

    assert response.status_code == 200
    assert response.json() == {"ok": "yes"}
    # Fail-OPEN, not fail-quiet: the request must actually reach the app.
    assert hits == ["downstream"]


@pytest.mark.usefixtures("_middleware_limits")
def test_middleware_still_limits_when_redis_is_healthy() -> None:
    """Positive control: "fails open" must not have become "never limits".

    With a healthy pipeline reporting the window already full, the middleware
    still returns 429 and the downstream app is never reached.
    """
    hits: list[str] = []
    # Pipeline order is zremrangebyscore, zcard, zadd, expire -> results[1] is
    # the pre-insert count. 5 >= rate_limit_requests(5) => limited.
    client = _FakeRedisClient(_FakePipeline(results=[0, 5, 1, True]))
    app = _build_middleware_app(client, hits)

    response = TestClient(app).get("/api/record/ping")

    assert response.status_code == 429
    assert response.json()["detail"] == "Rate limit exceeded"
    assert "Retry-After" in response.headers
    assert hits == []


@pytest.mark.usefixtures("_middleware_limits")
def test_middleware_allows_when_under_limit() -> None:
    """Positive control: a healthy under-limit request passes through normally."""
    hits: list[str] = []
    client = _FakeRedisClient(_FakePipeline(results=[0, 1, 1, True]))
    app = _build_middleware_app(client, hits)

    response = TestClient(app).get("/api/record/ping")

    assert response.status_code == 200
    assert hits == ["downstream"]


# ---------------------------------------------------------------------------
# 3. src/api/health.py -- report Redis unhealthy instead of 500ing /health
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Async DB connection whose ``execute`` succeeds."""

    async def execute(self, statement: object) -> None:
        """Accept and ignore ``SELECT 1``."""
        return


class _FakeConnectCM:
    """Async context manager returned by ``engine.connect()``."""

    async def __aenter__(self) -> _FakeConnection:
        """Enter the connection scope."""
        return _FakeConnection()

    async def __aexit__(self, *exc_info: object) -> bool:
        """Leave the connection scope without suppressing anything."""
        return False


class _FakeEngine:
    """SQLAlchemy async engine stand-in that always connects successfully."""

    def connect(self) -> _FakeConnectCM:
        """Return a healthy connection scope."""
        return _FakeConnectCM()


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_check_redis_degrades_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
    fault: type[RedisError],
) -> None:
    """``_check_redis`` must return an "unhealthy" string, not raise.

    Seam: ``src.api.health.get_redis_client`` -- the name the function resolves.

    Args:
        monkeypatch: Pytest patcher.
        fault: The redis-py exception class raised by ``ping()``.
    """
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(side_effect=fault("redis is down"))
    monkeypatch.setattr(health_module, "get_redis_client", lambda: redis_client)

    status = await health_module._check_redis()

    assert status.startswith("unhealthy")


def test_health_endpoint_reports_redis_component_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GET /health`` must answer with a report, not 500, when Redis is down.

    Before the fix, ``ping()`` on a down Redis escaped ``_check_redis`` and blew
    through ``health_check``'s own builtin-only handler too, so the aggregate
    endpoint reported the whole service unreachable rather than reporting the
    one unhealthy component it exists to isolate.
    """
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(side_effect=redis.exceptions.ConnectionError("down"))
    monkeypatch.setattr(health_module, "get_redis_client", lambda: redis_client)

    async def _get_engine() -> _FakeEngine:
        return _FakeEngine()

    monkeypatch.setattr(health_module, "get_engine", _get_engine)

    app = FastAPI()
    app.include_router(health_module.router)

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["components"]["database"] == "healthy"
    assert body["components"]["redis"].startswith("unhealthy")
    # Redis down is a DEGRADED service, not an unreachable one.
    assert body["status"] == "degraded"


@pytest.mark.parametrize(
    ("ping_result", "redis_status", "overall"),
    [
        (True, "healthy", "healthy"),
        (False, "unhealthy: ping failed", "degraded"),
        (None, "unhealthy: ping failed", "degraded"),
    ],
)
def test_health_endpoint_treats_a_false_ping_as_unhealthy(
    monkeypatch: pytest.MonkeyPatch,
    ping_result: bool | None,
    redis_status: str,
    overall: str,
) -> None:
    """A ``ping()`` that RETURNS False is a down Redis, not a healthy one.

    ``w2v_common`` ``RedisClient.ping`` catches ``RedisError`` and returns
    False, so the raise-path tests above never see the real client's failure
    shape. The HTTP status stays 200 either way: Redis is a degraded
    dependency, never a reason to fail a probe.
    """
    redis_client = MagicMock()
    redis_client.ping = AsyncMock(return_value=ping_result)
    monkeypatch.setattr(health_module, "get_redis_client", lambda: redis_client)

    async def _get_engine() -> _FakeEngine:
        return _FakeEngine()

    monkeypatch.setattr(health_module, "get_engine", _get_engine)

    app = FastAPI()
    app.include_router(health_module.router)
    storage = MagicMock()
    storage.health_check = AsyncMock(return_value=True)
    app.state.storage_service = storage

    response = TestClient(app).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["components"]["redis"] == redis_status
    assert body["status"] == overall


# ---------------------------------------------------------------------------
# 4. src/services/compilation/_segment_builders.py -- one bad image ref
#    must not abort the compile
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    """Minimal ``httpx.Response`` stand-in for the image fetch.

    Attributes:
        status_code: HTTP status used by :meth:`raise_for_status`.
        content: Response body bytes.
        url: URL echoed into the raised ``HTTPStatusError``.
    """

    def __init__(self, status_code: int, content: bytes, url: str) -> None:
        self.status_code = status_code
        self.content = content
        self.url = url

    def raise_for_status(self) -> None:
        """Raise ``httpx.HTTPStatusError`` for 4xx/5xx, exactly as httpx does.

        Raises:
            httpx.HTTPStatusError: When ``status_code`` is >= 400.
        """
        if self.status_code >= 400:
            request = httpx.Request("GET", self.url)
            raise httpx.HTTPStatusError(
                f"{self.status_code} for {self.url}",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )


class _Builder(segment_builders._SegmentBuildersMixin):
    """Concrete host for the mixin under test.

    ``_storage`` is only *declared* on the mixin (no value), and neither the
    http(s) materialisation branch nor ``_process_visual_asset_part``'s
    skip-path touches it, so it is deliberately left unbound rather than
    stubbed -- an unfaithful storage mock would obscure which seam is under
    test.
    """


def _make_ctx(tmp_path: Path) -> _JobContext:
    """Build a job context whose asset directory lives in ``tmp_path``.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        A ``_JobContext`` sufficient for the image-materialisation path.
    """
    return _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        title_cards_dir=tmp_path,
    )


_IMAGE_REF = "https://images.example.com/shot.png"


def test_materialize_image_ref_returns_none_on_http_status_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A 404 on one image ref returns ``None`` instead of propagating.

    Seam: ``_segment_builders.safe_get_sync`` -- the function the http(s) branch
    calls before ``resp.raise_for_status()``. Before the fix the resulting
    ``httpx.HTTPStatusError`` was not an ``OSError``, so it escaped the
    fail-soft handler and hard-failed the entire compilation.
    """
    monkeypatch.setattr(
        segment_builders,
        "safe_get_sync",
        lambda client, url, **kwargs: _FakeHTTPResponse(404, b"", url),
    )
    ctx = _make_ctx(tmp_path)

    result = _Builder()._materialize_image_ref(ctx, "part-1", _IMAGE_REF)

    assert result is None
    assert ctx.asset_cache == {}


def test_materialize_image_ref_returns_none_on_request_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A connect/read failure (``httpx.RequestError``) also degrades to ``None``."""

    def _boom(client: object, url: str, **kwargs: object) -> _FakeHTTPResponse:
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", url))

    monkeypatch.setattr(segment_builders, "safe_get_sync", _boom)
    ctx = _make_ctx(tmp_path)

    assert _Builder()._materialize_image_ref(ctx, "part-1", _IMAGE_REF) is None


def test_materialize_image_ref_materializes_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Positive control: a 200 still writes the asset and caches it."""
    monkeypatch.setattr(
        segment_builders,
        "safe_get_sync",
        lambda client, url, **kwargs: _FakeHTTPResponse(200, b"PNGBYTES", url),
    )
    ctx = _make_ctx(tmp_path)

    result = _Builder()._materialize_image_ref(ctx, "part-1", _IMAGE_REF)

    assert result is not None
    assert result.read_bytes() == b"PNGBYTES"
    assert ctx.asset_cache[_IMAGE_REF] == result


def test_visual_asset_part_is_skipped_not_fatal_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The surrounding compile continues: the part is excluded, nothing raises.

    This is the behaviour the degrade exists for -- one unreachable image drops
    one segment rather than failing the whole job.
    """
    monkeypatch.setattr(
        segment_builders,
        "safe_get_sync",
        lambda client, url, **kwargs: _FakeHTTPResponse(500, b"", url),
    )
    ctx = _make_ctx(tmp_path)

    _Builder()._process_visual_asset_part(ctx, "part-1", {"image_url": _IMAGE_REF})

    assert ctx.segments_excluded == 1
    assert ctx.segments == []


# ---------------------------------------------------------------------------
# 5. src/services/recording/event_publisher.py -- a dropped event must not
#    fail the recording that emitted it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_publish_status_swallows_redis_error(
    fault: type[RedisError], caplog: pytest.LogCaptureFixture
) -> None:
    """A failing ``publish`` is logged and swallowed, never raised at the caller.

    Seam: the injected redis client's ``publish`` -- the only awaited call in
    the ``try`` body.

    Args:
        fault: The redis-py exception class raised by ``publish``.
    """
    redis_client = MagicMock()
    redis_client.publish = AsyncMock(side_effect=fault("redis is down"))

    with caplog.at_level(logging.WARNING, logger="src.services.recording.event_publisher"):
        result = await EventPublisher(redis_client).publish_status(TENANT_U4, "job-1", "completed")

    # The fault really reached the seam — otherwise this proves nothing.
    redis_client.publish.assert_awaited_once()
    # Degrading means returning normally…
    assert result is None
    # …but NOT silently: a dropped status event has to leave a trace, or the
    # next "the UI never updated" report has nothing to read.
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == [
        "Failed to publish event: redis is down"
    ]


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_publish_swallows_redis_error(
    fault: type[RedisError], caplog: pytest.LogCaptureFixture
) -> None:
    """The generic ``publish`` helper degrades identically.

    Args:
        fault: The redis-py exception class raised by ``publish``.
    """
    redis_client = MagicMock()
    redis_client.publish = AsyncMock(side_effect=fault("redis is down"))

    with caplog.at_level(logging.WARNING, logger="src.services.recording.event_publisher"):
        result = await EventPublisher(redis_client).publish(
            f"recording:jobs:{TENANT_U4}", {"event": "ping"}
        )

    redis_client.publish.assert_awaited_once()
    assert result is None
    # The generic helper names the channel it lost, the status helper does not.
    assert [r.message for r in caplog.records if r.levelno >= logging.WARNING] == [
        f"Failed to publish to recording:jobs:{TENANT_U4}: redis is down"
    ]


async def test_publish_status_happy_path_still_publishes() -> None:
    """Positive control: swallowing failures did not stop real publishes."""
    redis_client = MagicMock()
    redis_client.publish = AsyncMock()

    await EventPublisher(redis_client).publish_status(
        TENANT_U4, "job-1", "completed", {"artifact": "x.mp4"}
    )

    await_args = redis_client.publish.await_args
    assert await_args is not None, "publish was never awaited"
    channel, raw = await_args.args
    assert channel == f"recording:jobs:{TENANT_U4}"
    payload = json.loads(raw)
    assert payload["event"] == "status_changed"
    assert payload["job_id"] == "job-1"
    assert payload["status"] == "completed"
    assert payload["artifact"] == "x.mp4"


# ---------------------------------------------------------------------------
# 6. src/services/compilation/rate_limiter.py -- the documented in-memory
#    fallback must actually run
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_inmemory_rate_state() -> Iterator[None]:
    """Clear the module-global in-memory rate-limit state around every test."""
    compilation_rate_limiter._inmemory_timestamps.clear()
    yield
    compilation_rate_limiter._inmemory_timestamps.clear()


def _make_compilation_limiter(
    requests_per_minute: int,
) -> compilation_rate_limiter.RateLimitMiddleware:
    """Build the compilation rate-limit middleware with an explicit budget.

    Args:
        requests_per_minute: Per-tenant request budget.

    Returns:
        A configured ``RateLimitMiddleware``. ``dispatch`` is never invoked in
        these tests, so the wrapped app is a bare mock (the repo's existing
        ``test_rate_limiter_redis.py`` uses the same shape).
    """
    settings = compilation_rate_limiter.RateLimitSettings(
        enabled=True, requests_per_minute=requests_per_minute
    )
    return compilation_rate_limiter.RateLimitMiddleware(app=MagicMock(), settings=settings)


def _patch_broken_redis(monkeypatch: pytest.MonkeyPatch, fault: type[RedisError]) -> None:
    """Make ``w2v_common``'s ``get_redis_client`` yield a pipeline that always fails.

    ``_check_rate_redis`` imports ``get_redis_client`` from
    ``w2v_common.cache.redis_client`` *inside* the function body, so that module
    attribute is the real seam.

    Args:
        monkeypatch: Pytest patcher.
        fault: The redis-py exception class raised by ``pipe.execute()``.
    """
    import w2v_common.cache.redis_client as w2v_redis

    client = _FakeRedisClient(_FakePipeline(error=fault("redis is down")))
    monkeypatch.setattr(w2v_redis, "get_redis_client", lambda: client)


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_check_rate_falls_back_to_inmemory(
    monkeypatch: pytest.MonkeyPatch,
    fault: type[RedisError],
) -> None:
    """``_check_rate`` degrades to in-memory tracking rather than raising.

    Args:
        monkeypatch: Pytest patcher.
        fault: The redis-py exception class raised by the Redis path.
    """
    _patch_broken_redis(monkeypatch, fault)
    limiter = _make_compilation_limiter(requests_per_minute=5)

    exceeded, retry_after = await limiter._check_rate("tenant-1")

    assert exceeded is False
    assert retry_after == 0
    # Proof the FALLBACK ran (not merely that nothing raised): the in-memory
    # tracker recorded the request.
    assert len(compilation_rate_limiter._inmemory_timestamps["tenant-1"]) == 1


async def test_inmemory_fallback_still_enforces_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the fallback limits too -- it is not a blanket allow."""
    _patch_broken_redis(monkeypatch, redis.exceptions.ConnectionError)
    limiter = _make_compilation_limiter(requests_per_minute=2)

    assert (await limiter._check_rate("tenant-1"))[0] is False
    assert (await limiter._check_rate("tenant-1"))[0] is False

    exceeded, retry_after = await limiter._check_rate("tenant-1")

    assert exceeded is True
    assert retry_after >= 1


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_concurrent_and_quota_checks_allow_on_redis_error(
    monkeypatch: pytest.MonkeyPatch,
    fault: type[RedisError],
) -> None:
    """The quota helpers allow the request when Redis is unreachable.

    Seam: ``compilation.rate_limiter._redis_client`` -- the module-level helper
    every one of these functions awaits first.

    Args:
        monkeypatch: Pytest patcher.
        fault: The redis-py exception class raised by ``_redis_client``.
    """
    monkeypatch.setattr(
        compilation_rate_limiter,
        "_redis_client",
        AsyncMock(side_effect=fault("redis is down")),
    )

    assert await compilation_rate_limiter.check_concurrent_limit("tenant-1", 3) is True
    assert await compilation_rate_limiter.check_daily_quota("tenant-1", 100) is True


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_counter_mutations_swallow_redis_error(
    monkeypatch: pytest.MonkeyPatch,
    fault: type[RedisError],
) -> None:
    """The three counter mutators log and continue instead of raising.

    Args:
        monkeypatch: Pytest patcher.
        fault: The redis-py exception class raised by ``_redis_client``.
    """
    monkeypatch.setattr(
        compilation_rate_limiter,
        "_redis_client",
        AsyncMock(side_effect=fault("redis is down")),
    )

    await compilation_rate_limiter.increment_concurrent("tenant-1")
    await compilation_rate_limiter.decrement_concurrent("tenant-1")
    await compilation_rate_limiter.increment_daily_quota("tenant-1")


async def test_quota_check_uses_redis_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: with a healthy client the Redis path decides the answer.

    Guards against the fix degenerating into "always allow".

    Args:
        monkeypatch: Pytest patcher.
    """
    client = MagicMock()
    client.get = AsyncMock(return_value=b"7")
    monkeypatch.setattr(compilation_rate_limiter, "_redis_client", AsyncMock(return_value=client))

    assert await compilation_rate_limiter.check_concurrent_limit("tenant-1", 3) is False
    assert await compilation_rate_limiter.check_concurrent_limit("tenant-1", 10) is True


# ---------------------------------------------------------------------------
# 7. src/api/events.py -- the SSE generator's promised error frame
# ---------------------------------------------------------------------------


def _bare_request() -> Request:
    """Build a real Starlette ``Request`` with a minimal HTTP scope.

    The failure under test is raised before the generator ever consults the
    request, so a real (if inert) ``Request`` is both sufficient and more
    faithful than a duck-typed stand-in.

    Returns:
        A ``Request`` over a minimal ``http`` scope.
    """
    return Request({"type": "http", "method": "GET", "path": "/events/t", "headers": []})


@pytest.mark.parametrize("fault", _REDIS_FAULTS)
async def test_sse_generator_emits_error_frame_on_redis_error(
    fault: type[RedisError],
) -> None:
    """The SSE stream yields ``data: {"error": ...}`` instead of raising mid-stream.

    Seam: the injected redis client's ``_ensure_connected()``, which the
    generator calls before subscribing. Before the fix the redis-py exception
    escaped the builtin-only tuple, so the documented error frame was
    unreachable and the stream died on the client with no explanation.

    Args:
        fault: The redis-py exception class raised on connect.
    """
    redis_client = MagicMock()
    redis_client._ensure_connected = MagicMock(side_effect=fault("redis is down"))

    chunks = [
        chunk
        async for chunk in events_module._sse_event_generator(
            redis_client, TENANT_U4, _bare_request()
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].startswith("data: ")
    payload = json.loads(chunks[0].removeprefix("data: ").strip())
    assert "error" in payload


# ---------------------------------------------------------------------------
# 7. Degrade handlers that FIRE but were never reported
# ---------------------------------------------------------------------------
#
# The sibling defect to the rest of this module. The handlers above were dead
# code -- they never ran. These ones run correctly, drop a requested feature to
# keep an otherwise-deliverable video, log a warning... and then told the caller
# nothing. The job row is marked ``completed``, ``error`` is None, and before
# ``ctx.degradations`` existed the response body had no field that could carry
# it. A user paid for a watermark, got a video without one, and every signal
# the API offered said the compile had fully succeeded.


class _Effects(_effects._EffectsMixin):
    """Concrete host for the effects mixin under test."""


def _effects_ctx(tmp_path: Path) -> _JobContext:
    out = tmp_path / "demo.mp4"
    out.write_bytes(b"video")
    return _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        work_dir=tmp_path,
        output_path=out,
    )


def test_watermark_refused_by_the_lfi_guard_is_recorded_as_a_degradation(
    tmp_path: Path,
) -> None:
    """The LFI guard skips the overlay rather than failing the job.

    That is the right call -- but before this channel existed the skip left no
    trace anywhere the caller could reach, so a watermark-less video was
    indistinguishable from a clean compile.
    """
    ctx = _effects_ctx(tmp_path)

    _Effects()._apply_watermark(ctx, {"image_path": "/etc/passwd"})

    assert [d["feature"] for d in ctx.degradations] == ["watermark"]
    assert "local-path guard" in ctx.degradations[0]["reason"]


def test_color_correction_failure_is_recorded_as_a_degradation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed colour correction keeps the un-corrected video and says so."""
    ctx = _effects_ctx(tmp_path)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("ffmpeg eq filter unavailable")

    monkeypatch.setattr("src.services.compilation.video_filters.apply_color_correction", _boom)

    _Effects()._apply_color_correction(ctx, {})

    assert [d["feature"] for d in ctx.degradations] == ["color_correction"]
    # The ffmpeg message must NOT be echoed into the report.
    assert "eq filter unavailable" not in ctx.degradations[0]["reason"]
    assert ctx.degradations[0]["error_type"] == "RuntimeError"


def test_clean_run_records_no_degradations(tmp_path: Path) -> None:
    """The channel must stay EMPTY when nothing was dropped.

    Without this, a ``record_degradation`` accidentally called on the happy path
    would flip every compile to ``partial_success`` and the discriminator would
    be worthless in the other direction.
    """
    ctx = _effects_ctx(tmp_path)
    assert ctx.degradations == []


# --- the derivation itself -------------------------------------------------


class _FinalizeHost(_finalize._FinalizeMixin):
    """Host that stubs out everything around the report-building block.

    Only the status derivation is under test, so the encode probe, the upload
    and the DB write are replaced -- but ``_finalize`` itself is the real
    function, so deleting the ``partial_success`` write reds these tests.
    """

    def __init__(self, ctx_degradations: list[dict[str, str]]) -> None:
        self._ctx_degradations = ctx_degradations

    def _finalize_sync(self, ctx: _JobContext) -> tuple:
        return (12.0, 1024, [], None, None)

    async def _upload_final_artifacts(self, ctx: _JobContext, thumbnail_path: object) -> tuple:
        return ("s3://bucket/demo.mp4", None, None)


async def _report_for(degradations: list[dict[str, str]], subtitle_error: str | None) -> dict:
    """Run the REAL ``_finalize`` and return the report it persisted."""
    ctx = _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=MagicMock(),
        work_dir=Path("/nonexistent"),
    )
    ctx.degradations = list(degradations)
    ctx.subtitle_error = subtitle_error
    ctx.settings.compilation.cleanup_intermediates = False

    captured: dict = {}

    async def _capture(db: object, job_id: str, **kwargs: object) -> None:
        captured.update(kwargs)

    with (
        mock.patch(
            "src.services.compilation.job_manager.CompilationJobManager.mark_completed",
            new=_capture,
        ),
        mock.patch(
            "src.services.compilation.job_manager.CompilationJobManager.record_usage",
            new=AsyncMock(),
        ),
    ):
        await _FinalizeHost(degradations)._finalize(ctx, AsyncMock(), None)

    return captured["compilation_report"]


async def test_finalize_reports_success_only_when_nothing_was_dropped() -> None:
    report = await _report_for([], None)
    assert report["status"] == "success"
    assert "degradations" not in report
    assert "subtitle_error" not in report


async def test_finalize_reports_partial_success_for_a_dropped_watermark() -> None:
    """A NON-subtitle drop must flip the status.

    This is the case the original code could not express at all: it keyed the
    status off ``ctx.subtitle_error`` alone, so a dropped watermark reported a
    fully-successful compile.
    """
    report = await _report_for(
        [{"feature": "watermark", "reason": "refused by the local-path guard"}], None
    )
    assert report["status"] == "partial_success"
    assert report["degradations"] == [
        {"feature": "watermark", "reason": "refused by the local-path guard"}
    ]
    # No subtitle failure occurred, so that field must not be what carries it.
    assert "subtitle_error" not in report


async def test_finalize_folds_a_subtitle_error_into_degradations() -> None:
    report = await _report_for([], "burn-in failed: font not found")
    assert report["status"] == "partial_success"
    assert report["subtitle_error"] == "burn-in failed: font not found"
    # The degradations entry is SANITISED: subtitle_error can carry a resolved
    # server path (the "file not found" branch echoes it), so the fold uses a
    # fixed literal rather than copying the message through.
    assert report["degradations"] == [
        {"feature": "subtitles", "reason": "Subtitles could not be applied"}
    ]


# --- the sites wired in round 3 -------------------------------------------


async def test_narration_download_failure_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worst drop on the path: the whole video ships SILENT.

    ``_download_narration_audio``'s own docstring said this produced "a SILENT
    video with the job still reported successful". It set
    ``narration_audio_dir = None`` and recorded nothing, so ``_finalize``
    derived ``status: "success"`` for a video with no narration at all.
    """
    from src.services.compilation import compiler_service as cs

    async def _boom(*a: object, **k: object) -> object:
        raise httpx.ConnectError("connection refused", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(cs, "safe_get", _boom, raising=False)
    monkeypatch.setattr("w2v_common.net.safe_get", _boom)
    ctx = _JobContext(
        job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None, work_dir=tmp_path
    )

    await cs._download_narration_audio(ctx, "https://media.example.com/narration.mp3")

    assert ctx.narration_audio_dir is None
    assert [d["feature"] for d in ctx.degradations] == ["narration"]


def test_missing_background_music_file_is_recorded(tmp_path: Path) -> None:
    """Previously a bare ``return`` with NO log line at all."""
    ctx = _effects_ctx(tmp_path)
    _Effects()._mix_background_music(ctx, str(tmp_path / "does_not_exist.mp3"))
    assert [d["feature"] for d in ctx.degradations] == ["background_music"]


def test_crossfade_fallback_is_recorded_when_fewer_transitions_than_requested() -> None:
    """`transitions_used` alone cannot distinguish "8 wanted" from "9th died"."""
    from src.services.compilation import _assembly

    class _Host(_assembly._AssemblyMixin):
        def _assemble_with_xfade(self, *a: object, **k: object) -> int:
            return 1  # 3 segments => 2 requested, only 1 applied

    ctx = _JobContext(job_id="j", tenant_id=TENANT_U4, job=MagicMock(), settings=MagicMock())
    ctx.job.filter_config = {}
    ctx.transition_type = "fade"
    ctx.work_dir = Path("/tmp")
    ctx.segments = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    ctx.settings.compilation.output_filename = "demo.mp4"

    _Host()._assemble_video(ctx)

    assert [d["feature"] for d in ctx.degradations] == ["transitions"]


def test_full_crossfade_run_records_nothing() -> None:
    """The other direction: all transitions applied must NOT flag a degradation."""
    from src.services.compilation import _assembly

    class _Host(_assembly._AssemblyMixin):
        def _assemble_with_xfade(self, *a: object, **k: object) -> int:
            return 2  # all requested transitions applied

    ctx = _JobContext(job_id="j", tenant_id=TENANT_U4, job=MagicMock(), settings=MagicMock())
    ctx.job.filter_config = {}
    ctx.transition_type = "fade"
    ctx.work_dir = Path("/tmp")
    ctx.segments = [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")]
    ctx.settings.compilation.output_filename = "demo.mp4"

    _Host()._assemble_video(ctx)

    assert ctx.degradations == []


# --- the leak ---------------------------------------------------------------


class _FakeS3Error(Exception):
    """Shaped like a real minio S3Error, whose str() names bucket, key and host."""


# The request_id and job uuid below are deliberately NOT high-entropy hex.
# An earlier draft used realistic values and detect-secrets flagged them as a
# "Hex High Entropy String" -- a false positive, but the fix belongs here rather
# than in an allowlist: this fixture only needs tokens that are DISTINCTIVE
# enough to spot in the degradation reason, not tokens that look like keys.
_REALISTIC_S3_MESSAGE = (
    "S3Error code: NoSuchKey, message: The specified key does not exist., "
    "resource: /w2v-media/jobs/job-not-a-real-uuid-0001/demo.mp4, "
    "request_id: REQ-NOT-A-SECRET-0001, host_id: minio.test:9000, "
    "bucket_name: w2v-media, object_name: jobs/job-not-a-real/demo.mp4"
)


def test_degradation_reason_does_not_leak_storage_detail(tmp_path: Path) -> None:
    """A realistic S3Error must not reach the API-facing ``reason``.

    ``format_error`` returns ``f"{type}: {exc}"`` up to 4000 chars. Surfacing
    that verbatim would hand the caller the bucket name, the full object key,
    the internal MinIO host and the request id -- and ``_apply_subtitles``'
    docstring already refuses to echo a rejected path precisely to avoid that
    class of oracle. The reason is a fixed literal; only the exception CLASS
    name rides along.
    """
    ctx = _effects_ctx(tmp_path)
    exc = _FakeS3Error(_REALISTIC_S3_MESSAGE)

    ctx.record_degradation("thumbnail_upload", "Thumbnail upload to storage failed", exc)

    reason = ctx.degradations[0]["reason"]
    # Named `leaked_token`, not `secret`: detect-secrets' keyword plugin keys on
    # the VARIABLE NAME, so a loop variable called `secret` trips the scanner on
    # every literal beside it, whatever the literal is.
    for leaked_token in (
        "w2v-media",
        "minio.test",
        "REQ-NOT-A-SECRET-0001",
        "job-not-a-real",
        "NoSuchKey",
        "demo.mp4",
    ):
        assert leaked_token not in reason, f"{leaked_token!r} leaked into the API-facing reason"
    assert reason == "Thumbnail upload to storage failed"
    assert ctx.degradations[0]["error_type"] == "_FakeS3Error"


def test_no_wired_site_uses_format_error_for_a_reason() -> None:
    """Static guard: ``format_error`` must never feed a degradation reason.

    ``reason`` is returned verbatim to the caller, so a future site reaching for
    ``format_error(exc)`` -- as every site in the first draft of this fix did --
    would silently reopen the leak proven above.
    """
    import ast

    root = Path(__file__).resolve().parents[2] / "src" / "services" / "compilation"
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "record_degradation"
                and len(node.args) >= 2
                and not isinstance(node.args[1], ast.Constant)
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        "record_degradation reason must be a fixed literal, not a runtime value: "
        + ", ".join(offenders)
    )


@pytest.mark.parametrize(
    ("part", "feature"),
    [
        ({"background_image": "/etc/passwd"}, "background_image"),
        ({"logo_path": "/etc/shadow"}, "logo"),
    ],
)
def test_title_card_asset_refused_by_the_lfi_guard_is_recorded(
    tmp_path: Path, part: dict, feature: str
) -> None:
    """The background-image and logo twins of the watermark guard.

    All three end at the SAME ``resolve_local_asset_path`` boundary and all
    three composite into a downloadable video, so a refusal on any of them is
    the same class of silent drop -- the title card renders without the asset
    the tenant asked for.
    """
    from src.services.compilation import _segment_builders as sb

    class _Builder(sb._SegmentBuildersMixin):
        pass

    ctx = _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        work_dir=tmp_path,
        title_cards_dir=tmp_path,
        normalized_dir=tmp_path,
    )
    ctx.title_gen = MagicMock()

    # The guard fires early in the function; the mocked generator writes no file,
    # so the downstream probe raises. That is irrelevant to what is under test --
    # the degradation must already have been recorded by then.
    with (
        mock.patch.object(sb, "resolve_local_asset_path", return_value=None),
        contextlib.suppress(RuntimeError),
    ):
        _Builder()._process_title_card(ctx, "part-1", {**part, "title": "T"})

    assert [d["feature"] for d in ctx.degradations] == [feature]
    # The refused path must not be echoed back to the caller.
    assert "/etc/" not in ctx.degradations[0]["reason"]


# ---------------------------------------------------------------------------
# 8. The remaining wired sites, each exercised through its real code path
# ---------------------------------------------------------------------------
#
# Every test here drives the production function and asserts on
# ctx.degradations. None of them call record_degradation directly -- a test that
# does that passes whether or not the site is wired, which is how an earlier
# draft of this file recorded a "proven" site that was not exercised at all.

_FF = "src.services.compilation.compiler_service.FFmpegOps"


class _Fin(_finalize._FinalizeMixin):
    """Concrete host for the finalize mixin.

    ``storage`` is annotated ``Any`` rather than ``object`` because the mixin
    declares ``_storage: StorageService`` and these tests deliberately hand it
    a stub. ``object`` is not assignable to that attribute, so ty rejects it.
    """

    def __init__(self, storage: Any = None) -> None:
        self._storage = storage


def _fin_ctx(tmp_path: Path) -> _JobContext:
    out = tmp_path / "demo.mp4"
    out.write_bytes(b"video")
    settings = MagicMock()
    settings.compilation.output_filename = "demo.mp4"
    return _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=settings,
        work_dir=tmp_path,
        output_path=out,
    )


def test_chapter_marker_failure_is_recorded(tmp_path: Path) -> None:
    ctx = _fin_ctx(tmp_path)
    with mock.patch(f"{_FF}.embed_chapters", side_effect=RuntimeError("ffmpeg died")):
        _Fin()._embed_chapter_markers(ctx, [{"chapter_number": 1, "title": "a"}])
    assert [d["feature"] for d in ctx.degradations] == ["chapter_markers"]
    assert "ffmpeg died" not in ctx.degradations[0]["reason"]


def test_thumbnail_generation_failure_is_recorded(tmp_path: Path) -> None:
    ctx = _fin_ctx(tmp_path)
    with mock.patch(f"{_FF}.extract_thumbnail", side_effect=OSError("no such filter")):
        assert _Fin()._generate_thumbnail(ctx) is None
    assert [d["feature"] for d in ctx.degradations] == ["thumbnail"]


def test_quality_validation_failure_is_recorded(tmp_path: Path) -> None:
    ctx = _fin_ctx(tmp_path)
    ctx.target_w, ctx.target_h = 1920, 1080
    with mock.patch(
        "src.services.compilation.quality_validator.QualityValidator.validate",
        side_effect=RuntimeError("probe failed"),
    ):
        assert _Fin()._validate_quality(ctx) is None
    assert [d["feature"] for d in ctx.degradations] == ["quality_validation"]


async def test_presign_failure_is_recorded(tmp_path: Path) -> None:
    """The video IS stored; only the convenience URL is missing."""
    ctx = _fin_ctx(tmp_path)
    storage = MagicMock()
    storage.store_file = AsyncMock(return_value="s3://bucket/demo.mp4")
    storage.presign_url = AsyncMock(side_effect=RuntimeError(_REALISTIC_S3_MESSAGE))

    stored, _thumb, presigned = await _Fin(storage)._upload_final_artifacts(ctx, None)

    assert stored == "s3://bucket/demo.mp4"
    assert presigned is None
    assert [d["feature"] for d in ctx.degradations] == ["presigned_url"]
    assert "w2v-media" not in ctx.degradations[0]["reason"]


async def test_thumbnail_upload_failure_is_recorded(tmp_path: Path) -> None:
    ctx = _fin_ctx(tmp_path)
    thumb = tmp_path / "thumbnail.jpg"
    thumb.write_bytes(b"jpg")
    storage = MagicMock()
    storage.presign_url = AsyncMock(return_value=None)
    storage.store_file = AsyncMock(
        side_effect=[
            "s3://bucket/demo.mp4",
            RuntimeError(_REALISTIC_S3_MESSAGE),
        ]
    )

    await _Fin(storage)._upload_final_artifacts(ctx, str(thumb))

    assert [d["feature"] for d in ctx.degradations] == ["thumbnail_upload"]
    assert "minio.test" not in ctx.degradations[0]["reason"]


def test_watermark_overlay_exception_is_recorded(tmp_path: Path) -> None:
    """Distinct from the guard-refusal branch, which has its own test."""
    ctx = _effects_ctx(tmp_path)
    wm = tmp_path / "wm.png"
    wm.write_bytes(b"png")
    with (
        mock.patch("src.services.compilation._effects.resolve_local_asset_path", return_value=wm),
        mock.patch(
            "src.services.compilation.video_filters.overlay_image",
            side_effect=RuntimeError("overlay filter unavailable"),
        ),
    ):
        _Effects()._apply_watermark(ctx, {"image_path": str(wm)})
    assert [d["feature"] for d in ctx.degradations] == ["watermark"]
    assert ctx.degradations[0]["reason"] == "Watermark overlay failed"


def test_narration_loudnorm_failure_is_recorded(tmp_path: Path) -> None:
    ctx = _effects_ctx(tmp_path)
    narr_dir = tmp_path / "narration_audio"
    narr_dir.mkdir()
    (narr_dir / "narration_full.mp3").write_bytes(b"mp3")
    ctx.narration_audio_dir = narr_dir
    ctx.segments_with_narration = 0

    with mock.patch(f"{_FF}.loudnorm_audio", side_effect=RuntimeError("loudnorm failed")):
        result = _Effects()._resolve_narration_audio(ctx)

    # Falls back to the un-normalised track rather than failing the job.
    assert result == narr_dir / "narration_full.mp3"
    assert [d["feature"] for d in ctx.degradations] == ["narration_loudnorm"]


def test_ai_audio_strip_failure_is_recorded(tmp_path: Path) -> None:
    """The AI clip's own audio stays in the deliverable."""
    from src.services.compilation import _segment_builders as sb

    class _Builder(sb._SegmentBuildersMixin):
        pass

    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "part-1.mp4").write_bytes(b"clip")
    ctx = _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        work_dir=tmp_path,
        edited_dir=tmp_path,
        normalized_dir=tmp_path,
        clips_dir=clips,
    )

    with (
        mock.patch(f"{_FF}.strip_audio", side_effect=RuntimeError("no audio stream")),
        mock.patch(f"{_FF}.normalize_stream"),
    ):
        _Builder()._process_ai_clip(ctx, "part-1", {"strip_ai_audio": True})

    assert [d["feature"] for d in ctx.degradations] == ["ai_audio_strip"]


def test_per_segment_narration_mix_failure_is_recorded(tmp_path: Path) -> None:
    from src.services.compilation import _segment_builders as sb

    class _Builder(sb._SegmentBuildersMixin):
        pass

    ctx = _JobContext(job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None)
    video = tmp_path / "seg.mp4"
    video.write_bytes(b"v")

    with mock.patch(f"{_FF}.probe_duration", side_effect=OSError("probe failed")):
        result = _Builder()._mix_per_segment_narration(
            ctx, video, tmp_path / "a.mp3", tmp_path, "part-1"
        )

    # Returns the UN-narrated segment -- the drop this records.
    assert result == video
    assert [d["feature"] for d in ctx.degradations] == ["segment_narration"]


def test_background_image_download_failure_is_recorded(tmp_path: Path) -> None:
    from src.services.compilation import _segment_builders as sb

    class _Builder(sb._SegmentBuildersMixin):
        pass

    ctx = _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        work_dir=tmp_path,
        title_cards_dir=tmp_path,
        normalized_dir=tmp_path,
    )
    ctx.title_gen = MagicMock()

    def _boom(*a: object, **k: object) -> object:
        raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))

    with (
        mock.patch.object(sb, "safe_get_sync", _boom),
        contextlib.suppress(RuntimeError),
    ):
        _Builder()._process_title_card(
            ctx,
            "part-1",
            {"title": "T", "background_image": "https://images.example.com/bg.jpg"},
        )

    assert [d["feature"] for d in ctx.degradations] == ["background_image"]


async def test_motion_title_card_fallback_is_recorded(tmp_path: Path) -> None:
    """Animated card requested, static Pillow card delivered -- a real downgrade.

    Drives the real handler; an earlier draft asserted on a hand-made
    record_degradation call, which passed whether or not the site was wired.
    """
    from src.services.compilation import _segment_builders as sb

    class _Builder(sb._SegmentBuildersMixin):
        pass

    ctx = _JobContext(
        job_id="job-1",
        tenant_id=TENANT_U4,
        job=None,
        settings=None,
        work_dir=tmp_path,
        title_cards_dir=tmp_path,
        normalized_dir=tmp_path,
    )
    ctx.title_gen = MagicMock()

    from src.services.recording.animation_recorder import AnimationRecorder

    # Patch the recorder the handler actually calls. This used to patch
    # ``sb.render_animation`` with ``create=True`` -- a name the handler never
    # looks up -- so wherever Chromium was installed the test performed a REAL
    # ~70 s animation render and only reached the fallback when that render
    # failed, tripping the 60 s per-test timeout of the pre-push gate.
    with (
        mock.patch.object(
            AnimationRecorder,
            "record_animation",
            AsyncMock(side_effect=RuntimeError("playwright unavailable")),
        ),
        mock.patch.object(_Builder, "_process_title_card", MagicMock(return_value=None)),
    ):
        await _Builder()._process_title_card_motion(
            ctx, "part-1", {"title": "T", "animation": "fade-in"}
        )

    assert [d["feature"] for d in ctx.degradations] == ["motion_title_card"]


def test_progressive_xfade_fallback_is_recorded(tmp_path: Path) -> None:
    """The smart path records from inside the merge, not at the caller."""
    from src.services.compilation import _assembly

    class _Host(_assembly._AssemblyMixin):
        pass

    ctx = _JobContext(job_id="j", tenant_id=TENANT_U4, job=None, settings=None)
    vids = []
    for name in ("c0.mp4", "c1.mp4"):
        p = tmp_path / name
        p.write_bytes(b"v")
        vids.append(p)

    with (
        mock.patch(f"{_FF}.crossfade", side_effect=OSError("SIGKILL")),
        mock.patch(f"{_FF}.concat"),
    ):
        _Host()._progressive_xfade_merge(ctx, vids, tmp_path / "out.mp4", tmp_path, 0.5, "fade")

    assert [d["feature"] for d in ctx.degradations] == ["transitions"]
