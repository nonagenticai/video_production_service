"""Delivery-time webhook guard + Playwright BrowserContext interception.

Scope note: the *entry*-URL and compile-fetcher guards that used to live
alongside these are now owned by ``src/utils/url_guard.py`` (PR #109) and are
tested in ``tests/unit/test_tenant_url_ssrf_schemas.py``. What remains here is
only what #109 does not cover:

* the webhook POST at **delivery** time rather than at admission -- a queued job
  can be delivered long after it was accepted, and the two resolve
  independently;
* the browser itself -- an entry-URL check covers hop 0 of the top-level
  document only, not sub-resources, in-page ``fetch``, or 30x redirects.

The webhook tests assert that the **transport was never reached**
(``attempted == []``), not merely that the call returned ``False``. That
distinction matters: a refused-by-the-guard request and a dialled-then-failed
request produce the same return value, so an assertion on the return value
alone cannot tell them apart and would pass against the unguarded code.
"""

from __future__ import annotations

import ast
import asyncio
import ipaddress
import logging
import pathlib
import re
import socket
from collections.abc import Iterator
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from w2v_common.net import SSRFError

from src.shared.security.playwright_guard import install_ssrf_guard

# --- the estate's real internal targets -------------------------------------

REGISTRY_URL = "http://100.64.0.6:5000/v2/_catalog"  # RFC 6598 CGNAT tailnet
RFC1918_URL = "http://10.0.0.1/"
METADATA_URL = "http://169.254.169.254/latest/meta-data/"

INTERNAL_URLS = [REGISTRY_URL, RFC1918_URL, METADATA_URL]

PUBLIC_IP = "203.0.113.10"  # TEST-NET-3: not real, not in any block-list.
_DNS_MAP = {
    "site.example": PUBLIC_IP,
    "cdn.example": PUBLIC_IP,
    "login.example": PUBLIC_IP,
    "redirector.example": PUBLIC_IP,
    # A public-looking NAME that resolves straight into the tailnet.
    "rebind.example": "100.64.0.6",
}


def _addrinfo(ip: str) -> list[tuple[Any, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


@pytest.fixture
def stub_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve only the names in ``_DNS_MAP``; everything else fails closed.

    IP literals are echoed back because that is what the real
    :func:`socket.getaddrinfo` does -- ``validate_public_url`` has no literal
    fast-path. A stub that raised for literals would push those cases into the
    DNS-failure branch, so they would be refused for the wrong reason and the
    block-list assertions would be vacuous.
    """

    def _fake(host: str, *_a: Any, **_kw: Any) -> list[tuple[Any, ...]]:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            return _addrinfo(host)
        try:
            return _addrinfo(_DNS_MAP[host])
        except KeyError:
            raise socket.gaierror(f"stub resolver has no answer for {host}") from None

    monkeypatch.setattr("w2v_common.net.safe_fetch.socket.getaddrinfo", _fake)


# ---------------------------------------------------------------------------
# login_url -- the field that had NO validation whatsoever
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Webhook delivery
# ---------------------------------------------------------------------------


class TestWebhookDelivery:
    @pytest.mark.asyncio
    async def test_internal_webhook_is_refused_at_delivery_time(self, stub_dns: None) -> None:
        """Admission and delivery resolve independently; delivery is the one that counts."""
        from src.services.recording.webhook_service import WebhookService

        attempted: list[str] = []

        def _handler(request: httpx.Request) -> httpx.Response:
            attempted.append(str(request.url))
            return httpx.Response(200)

        real_client = httpx.AsyncClient
        with patch(
            "src.services.recording.webhook_service.httpx.AsyncClient",
            lambda **_kw: real_client(transport=httpx.MockTransport(_handler)),
        ):
            ok = await WebhookService().notify(REGISTRY_URL, "j1", "completed")

        assert ok is False
        assert attempted == [], "a job payload was POSTed at an internal address"

    @pytest.mark.asyncio
    async def test_refused_webhook_does_not_burn_retries(self, stub_dns: None) -> None:
        """A deterministic verdict must not be retried with backoff."""
        from src.services.recording.webhook_service import WebhookService

        sleeps = AsyncMock()
        real_client = httpx.AsyncClient
        with (
            patch("asyncio.sleep", sleeps),
            patch(
                "src.services.recording.webhook_service.httpx.AsyncClient",
                lambda **_kw: real_client(
                    transport=httpx.MockTransport(lambda r: httpx.Response(200))
                ),
            ),
        ):
            await WebhookService().notify(RFC1918_URL, "j1", "completed")

        sleeps.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_public_webhook_still_delivers(self, stub_dns: None) -> None:
        from src.services.recording.webhook_service import WebhookService

        real_client = httpx.AsyncClient
        with patch(
            "src.services.recording.webhook_service.httpx.AsyncClient",
            lambda **_kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        ):
            ok = await WebhookService().notify("https://cdn.example/hook", "j1", "completed")

        assert ok is True


# ---------------------------------------------------------------------------
# Playwright BrowserContext interception
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}


class _FakeRoute:
    def __init__(self, responses: list[_FakeResponse] | None = None) -> None:
        self.continued = False
        self.aborted_with: str | None = None
        self.fulfilled: _FakeResponse | None = None
        self.fetched: list[str] = []
        self.fetched_headers: list[dict[str, str]] = []
        self._responses = responses or [_FakeResponse(200)]

    async def continue_(self) -> None:
        self.continued = True

    async def abort(self, reason: str) -> None:
        self.aborted_with = reason

    async def fetch(
        self, *, url: str, max_redirects: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        assert max_redirects == 0, "the guard must never delegate redirect-following"
        self.fetched.append(url)
        self.fetched_headers.append(dict(headers or {}))
        return self._responses[min(len(self.fetched) - 1, len(self._responses) - 1)]

    async def fulfill(self, *, response: _FakeResponse) -> None:
        self.fulfilled = response


class _FakeRequest:
    def __init__(self, url: str, resource_type: str = "image") -> None:
        self.url = url
        self.resource_type = resource_type
        self.headers: dict[str, str] = {}


class _FakeWebSocketRoute:
    """Stands in for ``playwright.async_api.WebSocketRoute``.

    ``connect_to_server`` is deliberately synchronous, matching the real API:
    the guard's refusal is "never call it", so a double that made it awaitable
    would let a missing ``await`` read as a refusal.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed_with: tuple[int | None, str | None] | None = None
        self.connected = False

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed_with = (code, reason)

    def connect_to_server(self) -> object:
        self.connected = True
        return object()


class _UnclosableWebSocketRoute(_FakeWebSocketRoute):
    """...and the rescue ``close`` fails too, as it does on a torn-down socket."""

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        raise RuntimeError("socket already gone")


class _ConnectRaisingWebSocketRoute(_FakeWebSocketRoute):
    """The target is judged public, then opting in to the server blows up."""

    def connect_to_server(self) -> object:
        raise RuntimeError("playwright transport exploded")


class _FakeContext:
    def __init__(self) -> None:
        self.handler: Any = None
        self.pattern: Any = None
        self.ws_handler: Any = None
        self.ws_pattern: Any = None
        self.init_scripts: list[str] = []
        #: Registration order, so a test can assert the init script is installed
        #: before the route handlers without monkeypatching the double's methods.
        self.calls: list[str] = []
        #: ``(permissions, origin)`` per ``grant_permissions`` call.
        self.grants: list[tuple[tuple[str, ...], Any]] = []

    async def route(self, pattern: Any, handler: Any) -> None:
        self.calls.append("route")
        self.pattern = pattern
        self.handler = handler

    # Same reasoning as ``route_web_socket`` below: the real ``BrowserContext``
    # has this, so the double must too. The guard installs the SharedWorker
    # mitigation through it, and a double without it fails as "the guard is
    # broken" rather than "the double is narrower than production".
    async def add_init_script(self, script: Any) -> None:
        self.calls.append("add_init_script")
        self.init_scripts.append(str(script))

    # The real ``BrowserContext`` has this, so the double must too. A double
    # NARROWER than the object it stands in for cannot fail the way production
    # fails: without this method every test here would have died with
    # `AttributeError` the moment the guard started registering a WebSocket
    # route, which reports as "the guard is broken" rather than "the double is".
    async def route_web_socket(self, pattern: Any, handler: Any) -> None:
        self.calls.append("route_web_socket")
        self.ws_pattern = pattern
        self.ws_handler = handler

    # The real ``BrowserContext`` has this too. The guard grants Chromium's
    # ``local-network-access`` permission to allow-listed origins through it; the
    # double records every grant so a widened scope is visible without a browser.
    async def grant_permissions(self, permissions: Any, origin: Any = None) -> None:
        self.calls.append("grant_permissions")
        self.grants.append((tuple(permissions), origin))


async def _verdict_for(
    url: str,
    resource_type: str = "image",
    responses: list[_FakeResponse] | None = None,
) -> _FakeRoute:
    context = _FakeContext()
    await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
    route = _FakeRoute(responses)
    await context.handler(route, _FakeRequest(url, resource_type))
    return route


class TestPlaywrightRouteGuard:
    @pytest.mark.parametrize("url", INTERNAL_URLS)
    @pytest.mark.asyncio
    async def test_sub_resource_to_internal_host_is_aborted(self, url: str, stub_dns: None) -> None:
        """The case an entry-URL check cannot reach.

        The recorded page emits ``<img src="http://100.64.0.6:5000/...">`` and
        the browser fetches it independently of anything Python validated.
        """
        route = await _verdict_for(url)

        assert route.aborted_with == "blockedbyclient"
        assert route.fulfilled is None

    @pytest.mark.asyncio
    async def test_public_sub_resource_is_allowed(self, stub_dns: None) -> None:
        route = await _verdict_for("https://cdn.example/app.css", "stylesheet")

        assert route.aborted_with is None
        assert route.fetched == ["https://cdn.example/app.css"]
        assert route.fulfilled is not None and route.fulfilled.status == 200

    @pytest.mark.asyncio
    async def test_non_http_scheme_is_aborted(self, stub_dns: None) -> None:
        route = await _verdict_for("file:///etc/passwd", "document")

        assert route.aborted_with == "blockedbyclient"

    @pytest.mark.asyncio
    async def test_data_and_about_urls_pass_through(self, stub_dns: None) -> None:
        for url in ("data:image/png;base64,iVBORw0KGgo=", "about:blank"):
            route = await _verdict_for(url, "document")
            assert route.continued is True, url
            assert route.aborted_with is None, url

    @pytest.mark.asyncio
    async def test_redirect_to_internal_host_is_aborted(self, stub_dns: None) -> None:
        """Playwright does NOT re-invoke a route handler on redirect hops.

        A ``continue_()``-based handler therefore hands the whole redirect
        vector to the attacker; the guard walks the chain itself instead.
        """
        route = await _verdict_for(
            "https://redirector.example/go",
            "document",
            responses=[_FakeResponse(302, {"location": REGISTRY_URL})],
        )

        assert route.aborted_with == "blockedbyclient"
        assert route.fulfilled is None
        assert route.fetched == ["https://redirector.example/go"]

    @pytest.mark.asyncio
    async def test_redirect_to_public_host_is_followed(self, stub_dns: None) -> None:
        route = await _verdict_for(
            "https://redirector.example/go",
            "document",
            responses=[
                _FakeResponse(302, {"location": "https://cdn.example/final"}),
                _FakeResponse(200),
            ],
        )

        assert route.aborted_with is None
        assert route.fetched == ["https://redirector.example/go", "https://cdn.example/final"]
        assert route.fulfilled is not None and route.fulfilled.status == 200

    @pytest.mark.asyncio
    async def test_redirect_loop_is_capped(self, stub_dns: None) -> None:
        route = await _verdict_for(
            "https://cdn.example/loop",
            "document",
            responses=[_FakeResponse(302, {"location": "https://cdn.example/loop"})],
        )

        assert route.aborted_with == "blockedbyclient"
        assert len(route.fetched) == 6  # DEFAULT_MAX_REDIRECTS + 1

    @pytest.mark.asyncio
    async def test_host_resolving_into_the_tailnet_is_aborted(self, stub_dns: None) -> None:
        route = await _verdict_for("https://rebind.example/pixel.png")

        assert route.aborted_with == "blockedbyclient"

    @pytest.mark.asyncio
    async def test_verdict_is_cached_per_origin(
        self, stub_dns: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        context = _FakeContext()
        verdicts = await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        lookups: list[str] = []
        patched = socket.getaddrinfo

        def _counting(host: str, *a: Any, **kw: Any) -> Any:
            lookups.append(host)
            return patched(host, *a, **kw)

        monkeypatch.setattr("w2v_common.net.safe_fetch.socket.getaddrinfo", _counting)

        for path in ("/a.png", "/b.png", "/c.png"):
            route = _FakeRoute()
            await context.handler(route, _FakeRequest(f"https://cdn.example{path}"))
            assert route.aborted_with is None

        assert verdicts == {"https://cdn.example": True}
        assert lookups == ["cdn.example"]


# ---------------------------------------------------------------------------
# The route handler must be TOTAL
# ---------------------------------------------------------------------------


class _RaisingFetchRoute(_FakeRoute):
    """``route.fetch`` blows up with something the guard does not anticipate."""

    async def fetch(
        self, *, url: str, max_redirects: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        raise RuntimeError("playwright transport exploded")


class _TargetClosedFetchRoute(_FakeRoute):
    """``route.fetch`` fails because the page/context/browser is already gone.

    Playwright's real message, verbatim from a live pod on 2026-08-10:
    ``Route.fetch: Target page, context or browser has been closed``.
    """

    async def fetch(
        self, *, url: str, max_redirects: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        raise RuntimeError("Route.fetch: Target page, context or browser has been closed")


class _UnabortableRoute(_RaisingFetchRoute):
    """...and the rescue ``abort`` fails too, as it does on an already-resolved route."""

    async def abort(self, reason: str) -> None:
        raise RuntimeError("route is already handled")


class _CancellingFetchRoute(_FakeRoute):
    """``route.fetch`` is cancelled -- teardown, not a failure."""

    async def fetch(
        self, *, url: str, max_redirects: int, headers: dict[str, str] | None = None
    ) -> _FakeResponse:
        raise asyncio.CancelledError


class TestHandlerIsTotal:
    """An unanticipated failure must ABORT the route, never escape the handler.

    Playwright does not answer a route whose handler raised: it neither
    continues nor aborts, so Chromium holds the request until the navigation
    times out (~30 s, per affected request). That is fail-CLOSED, so it is not
    an SSRF bypass -- which is exactly why it survives a security review. It is
    still the wrong failure mode: it converts any bug on this path into a stall
    instead of an immediate, logged refusal.

    ``pytest`` fails a test on an escaping exception, so each case's real
    assertion is "the call returned at all"; the ``aborted_with`` assertion is
    what separates "returned because it was total" from a hypothetical handler
    that swallowed the error and left the route unanswered anyway.
    """

    @pytest.mark.asyncio
    async def test_raising_fetch_aborts_instead_of_stalling(self, stub_dns: None) -> None:
        """The origin is judged public, then the transport fails underneath us."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        route = _RaisingFetchRoute()

        await context.handler(route, _FakeRequest("https://cdn.example/app.css", "stylesheet"))

        assert route.aborted_with == "blockedbyclient"
        assert route.fulfilled is None
        assert route.continued is False

    @pytest.mark.asyncio
    async def test_unexpected_validator_error_aborts(
        self, stub_dns: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_origin_is_public`` handles SSRFError/ValueError/OSError -- and only those."""

        def _boom(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("resolver library regression")

        monkeypatch.setattr("src.shared.security.playwright_guard.validate_public_url", _boom)
        route = await _verdict_for("https://cdn.example/app.css", "stylesheet")

        assert route.aborted_with == "blockedbyclient"
        assert route.fulfilled is None

    @pytest.mark.asyncio
    async def test_target_closed_is_teardown_not_an_error(
        self, stub_dns: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A closed target is teardown noise, logged at DEBUG -- but still aborted.

        Measured live on analysis_service 2026-08-10: this was the ONLY error line
        in the whole w2v namespace across an hour of traffic, logged as
        ``SSRF guard: unexpected error handling <url> -- refusing``. It reads as
        the guard rejecting a legitimate asset; in fact the page that asked for it
        no longer existed. A false ERROR is worse than no ERROR, because it is the
        one signal an operator actually looks at.
        """
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        route = _TargetClosedFetchRoute()

        with caplog.at_level(logging.DEBUG, logger="src.shared.security.playwright_guard"):
            await context.handler(route, _FakeRequest("https://cdn.example/app.css", "stylesheet"))

        # Still TOTAL: the route is answered, so Chromium never stalls.
        assert route.aborted_with == "blockedbyclient"
        assert route.fulfilled is None
        levels = {r.levelno for r in caplog.records}
        assert logging.ERROR not in levels, "teardown must not be reported as an error"
        assert logging.DEBUG in levels

    @pytest.mark.asyncio
    async def test_a_real_fetch_failure_is_still_an_error(
        self, stub_dns: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The negative control for the demotion above."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        route = _RaisingFetchRoute()

        with caplog.at_level(logging.DEBUG, logger="src.shared.security.playwright_guard"):
            await context.handler(route, _FakeRequest("https://cdn.example/app.css", "stylesheet"))

        assert route.aborted_with == "blockedbyclient"
        assert logging.ERROR in {r.levelno for r in caplog.records}

    @pytest.mark.asyncio
    async def test_failing_rescue_abort_still_does_not_escape(self, stub_dns: None) -> None:
        """A route already resolved raises on a second ``abort``; that must not escape either."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        route = _UnabortableRoute()

        await context.handler(route, _FakeRequest("https://cdn.example/app.css", "stylesheet"))

        assert route.fulfilled is None

    @pytest.mark.asyncio
    async def test_cancellation_is_not_swallowed(self, stub_dns: None) -> None:
        """Cancellation is control flow. Absorbing it would break browser teardown."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        route = _CancellingFetchRoute()

        with pytest.raises(asyncio.CancelledError):
            await context.handler(route, _FakeRequest("https://cdn.example/app.css", "stylesheet"))

        assert route.aborted_with is None


# ---------------------------------------------------------------------------
# The recorder must actually INSTALL the guard on every context it creates
# ---------------------------------------------------------------------------


SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src"

# Names that mint a BrowserContext. ``launch_persistent_context`` returns one
# directly from the browser type, bypassing ``new_context`` entirely.
_CONTEXT_FACTORIES = frozenset({"new_context", "launch_persistent_context"})
_BROWSER_FACTORIES = frozenset({"launch", "connect", "connect_over_cdp"})


class _Ref(NamedTuple):
    """One Playwright sink, located precisely enough to be matched to a context.

    ``scope`` is the *innermost enclosing function*, not just the module. That
    is load-bearing: ``url_recorder`` creates three contexts in three different
    methods and names every one of them ``context``, so a module-scoped match
    lets a guard in method B satisfy an unguarded context in method A. Verified
    by mutation -- deleting the guard from the first of the three left a
    module-scoped check green.
    """

    module: str
    scope: str
    var: str
    line: int

    def same_binding(self, other: _Ref) -> bool:
        return (self.module, self.scope, self.var) == (other.module, other.scope, other.var)


class _Sinks(NamedTuple):
    contexts: list[_Ref]
    guards: list[_Ref]
    pages: list[_Ref]
    browsers: set[tuple[str, str, str]]  # (module, scope, browser var)


def _assigned_name(node: ast.stmt) -> str | None:
    """Return the single bare name *node* assigns to, if it assigns to one."""
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _called_attr(value: ast.expr | None) -> str | None:
    """Return ``foo`` for ``<expr>.foo(...)`` / ``await <expr>.foo(...)``."""
    if isinstance(value, ast.Await):
        value = value.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
        return value.func.attr
    return None


def _scoped_nodes(parent: ast.AST, scope: str) -> Iterator[tuple[str, ast.AST]]:
    """Yield every descendant of *parent* tagged with its innermost function scope."""
    for child in ast.iter_child_nodes(parent):
        inner = (
            f"{scope}.{child.name}"
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            else scope
        )
        yield inner, child
        yield from _scoped_nodes(child, inner)


def _classify(node: ast.AST) -> tuple[str, str, int] | None:
    """Return ``(kind, variable, lineno)`` for a Playwright sink node, else None.

    ``kind`` is one of ``context``, ``browser``, ``guard``, ``page``. The line
    number is read here rather than at the call site because only the branches
    below have narrowed the node to a type that carries one.
    """
    if isinstance(node, ast.Assign | ast.AnnAssign):
        name = _assigned_name(node)
        attr = _called_attr(node.value)
        if name is None:
            return None
        if attr in _CONTEXT_FACTORIES:
            return "context", name, node.lineno
        if attr in _BROWSER_FACTORIES:
            return "browser", name, node.lineno
        return None

    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "install_ssrf_guard":
        if node.args and isinstance(node.args[0], ast.Name):
            return "guard", node.args[0].id, node.lineno
        return None
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "new_page"
        and isinstance(func.value, ast.Name)
    ):
        return "page", func.value.id, node.lineno
    return None


def _collect_sinks() -> _Sinks:
    """Walk every module under ``src/`` and locate every Playwright sink.

    Nothing here is keyed to a filename or to a count: the modules come from a
    glob and the sites come from the AST, so a context created in a file that
    does not exist yet is still covered.
    """
    found: dict[str, list[_Ref]] = {"context": [], "browser": [], "guard": [], "page": []}

    for path in sorted(SRC_ROOT.rglob("*.py")):
        module = str(path.relative_to(SRC_ROOT.parent))
        tree = ast.parse(path.read_text(), filename=str(path))
        for scope, node in _scoped_nodes(tree, "<module>"):
            hit = _classify(node)
            if hit is not None:
                kind, var, lineno = hit
                found[kind].append(_Ref(module, scope, var, lineno))

    return _Sinks(
        contexts=found["context"],
        guards=found["guard"],
        pages=found["page"],
        browsers={(r.module, r.scope, r.var) for r in found["browser"]},
    )


class TestEveryBrowserContextIsGuarded:
    """Every Playwright ``BrowserContext`` created under ``src/`` is SSRF-guarded.

    Reading the source is the honest check here: the recorder entry points need
    a full Playwright drive to reach, and a test that renders happily while
    never calling ``context.route`` is exactly the regression that ships
    silently.

    What this replaced, and why: the previous cover hardcoded the single path
    ``src/services/recording/url_recorder.py`` and asserted a magic
    ``contexts == 3``. Both facts were true and the suite was green while
    ``content_recorder.py`` and ``animation_recorder.py`` each created a
    context with **no guard at all** -- two live, tenant-reachable in-pod SSRF
    sinks that the assertion could not see, because it never looked. Nothing
    below names a file or a count; both are derived from a glob over ``src/``,
    so a new unguarded context anywhere in the repo fails here.
    """

    def test_the_finder_sees_every_context_creation(self) -> None:
        """Non-vacuity: the AST walk must not silently miss a textual sink.

        Without this, an AST shape the collector does not understand (a context
        assigned via tuple-unpack, or created inside a ``with`` item) would
        make the guard assertions below pass by finding nothing at all.
        """
        sinks = _collect_sinks()
        textual = sum(
            len(re.findall(rf"\.{factory}\(", path.read_text()))
            for path in SRC_ROOT.rglob("*.py")
            for factory in _CONTEXT_FACTORIES
        )

        assert textual > 0, f"no BrowserContext creation found under {SRC_ROOT} at all"
        assert len(sinks.contexts) == textual, (
            f"{textual} context creations in the source text but the AST walk bound "
            f"only {len(sinks.contexts)} to a variable: "
            f"{[(s.module, s.line) for s in sinks.contexts]}. An unbound context "
            "cannot be guarded -- assign it to a name and call install_ssrf_guard on it."
        )

    def test_every_browser_context_gets_the_guard(self) -> None:
        sinks = _collect_sinks()
        unguarded = [
            site
            for site in sinks.contexts
            if not any(g.same_binding(site) and g.line > site.line for g in sinks.guards)
        ]
        assert not unguarded, "BrowserContext created without install_ssrf_guard: " + ", ".join(
            f"{s.module}:{s.line} ({s.scope}, {s.var})" for s in unguarded
        )

    def test_guard_is_installed_before_any_page_exists(self) -> None:
        """A handler registered after the first navigation misses it."""
        sinks = _collect_sinks()
        late: list[str] = []

        for site in sinks.contexts:
            first_guard = min(
                (g.line for g in sinks.guards if g.same_binding(site) and g.line > site.line),
                default=None,
            )
            first_page = min(
                (p.line for p in sinks.pages if p.same_binding(site) and p.line > site.line),
                default=None,
            )
            if first_page is None:
                continue
            if first_guard is None or first_guard > first_page:
                late.append(f"{site.module}:{site.line} ({site.var}) -> page at :{first_page}")

        assert not late, "context creates its page before the guard is installed: " + ", ".join(
            late
        )

    def test_no_page_is_minted_straight_off_a_browser(self) -> None:
        """``browser.new_page()`` mints an implicit context no guard can reach.

        It is the one-line bypass of everything above, so it is refused
        outright rather than left to be noticed in review.
        """
        sinks = _collect_sinks()
        bypasses = [
            f"{p.module}:{p.line} ({p.var}.new_page())"
            for p in sinks.pages
            if (p.module, p.scope, p.var) in sinks.browsers
        ]
        assert not bypasses, (
            "page created directly on a Browser, so its context is unguarded: "
            + ", ".join(bypasses)
        )


def test_ssrf_error_is_a_value_error() -> None:
    """Degrade clauses catching ``ValueError`` absorb guard refusals by design."""
    assert issubclass(SSRFError, ValueError)


class TestWebSocketHandlerIsTotal:
    """The same totality requirement, on the ``route_web_socket`` path.

    Sharper here than on the HTTP side. Playwright dials the real server only
    when the handler calls ``connect_to_server()``, so a raising WebSocket
    handler *is* fail-closed -- but silently: the page is left holding a socket
    that never opens and never closes, with nothing logged. Every case below
    therefore asserts the socket was explicitly closed, not merely that the
    handler returned.

    Which targets are judged public, and how ``ws``/``wss`` map onto the http
    verdict cache, is covered against a real Chromium in
    ``tests/integration/test_playwright_ssrf_guard.py``.
    """

    @pytest.mark.asyncio
    async def test_the_guard_registers_a_websocket_route_at_all(self, stub_dns: None) -> None:
        """Non-vacuity: without this registration the whole class is decorative.

        ``context.route`` is never consulted for a ``ws://`` handshake, so a
        guard that installs only the HTTP handler leaves a full-duplex channel
        to any internal address wide open.
        """
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.ws_handler is not None, (
            "install_ssrf_guard did not register a WebSocket route -- ws:// and "
            "wss:// targets are not judged at all"
        )

    @pytest.mark.asyncio
    async def test_internal_target_is_closed_not_connected(self, stub_dns: None) -> None:
        """The decision itself, at the unit layer: loopback must be refused."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _FakeWebSocketRoute("ws://127.0.0.1:9999/internal-endpoint")

        await context.ws_handler(ws)

        assert ws.connected is False, "the guard dialled an internal WebSocket target"
        assert ws.closed_with == (1008, "blocked by client")

    @pytest.mark.asyncio
    async def test_public_target_is_connected(self, stub_dns: None) -> None:
        """False-positive control: the handler must not simply close everything.

        Without this, every other case in this class would also pass against a
        handler that refused unconditionally -- which would break live-chat,
        analytics and HMR sockets on every page the service touches.
        """
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _FakeWebSocketRoute("wss://cdn.example/socket")

        await context.ws_handler(ws)

        assert ws.connected is True, "an allowed wss:// target was refused"
        assert ws.closed_with is None

    @pytest.mark.asyncio
    async def test_unexpected_validator_error_closes(
        self, stub_dns: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_origin_is_public`` handles SSRFError/ValueError/OSError -- and only those."""

        def _boom(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("resolver library regression")

        monkeypatch.setattr("src.shared.security.playwright_guard.validate_public_url", _boom)
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _FakeWebSocketRoute("wss://cdn.example/socket")

        await context.ws_handler(ws)

        assert ws.connected is False
        assert ws.closed_with == (1008, "blocked by client")

    @pytest.mark.asyncio
    async def test_raising_connect_closes_instead_of_stalling(self, stub_dns: None) -> None:
        """The target is judged public, then the transport fails underneath us."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _ConnectRaisingWebSocketRoute("wss://cdn.example/socket")

        await context.ws_handler(ws)

        assert ws.closed_with == (1008, "blocked by client")

    @pytest.mark.asyncio
    async def test_failing_rescue_close_still_does_not_escape(self, stub_dns: None) -> None:
        """A socket already torn down raises on ``close``; that must not escape either."""
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _UnclosableWebSocketRoute("ws://127.0.0.1:9999/internal-endpoint")

        await context.ws_handler(ws)

        assert ws.connected is False

    @pytest.mark.asyncio
    async def test_cancellation_is_not_swallowed(self, stub_dns: None) -> None:
        """Cancellation is control flow, exactly as on the HTTP path.

        False-positive control for the catch-all: it must fail if that clause is
        ever widened to ``BaseException``, not prove the fix.
        """

        def _cancel(*_a: Any, **_kw: Any) -> None:
            raise asyncio.CancelledError

        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext
        ws = _FakeWebSocketRoute("wss://cdn.example/socket")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("src.shared.security.playwright_guard.validate_public_url", _cancel)
            with pytest.raises(asyncio.CancelledError):
                await context.ws_handler(ws)

        assert ws.closed_with is None


class TestSharedWorkerMitigationIsInstalled:
    """The mitigation must be installed by ``install_ssrf_guard`` itself.

    The real proof that it *works* is the Chromium integration suite. This is
    the cheap structural half: a refactor that moves the route registrations
    around must not drop the init script, and the double records it so the
    omission is visible without a browser.

    Sibling of ``analysis_service``'s ``tests/unit/shared/
    test_playwright_guard_total.py::TestSharedWorkerMitigationIsInstalled``;
    keep the two in step.
    """

    @pytest.mark.asyncio
    async def test_install_ssrf_guard_adds_the_shared_worker_init_script(self) -> None:
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.init_scripts, (
            "install_ssrf_guard registered no init script, so the SharedWorker channel -- "
            "which cannot be routed and is NOT fail-closed (it completes a TCP handshake "
            "against the target) -- is open again"
        )
        assert any("SharedWorker" in script for script in context.init_scripts), (
            f"an init script was installed but it does not mention SharedWorker: "
            f"{context.init_scripts}"
        )

    @pytest.mark.asyncio
    async def test_the_init_script_is_installed_before_the_route_handlers(self) -> None:
        """Ordering is load-bearing, not cosmetic.

        If a route registration raised, an init script added afterwards would
        never run, leaving a context the caller believes is guarded but whose
        SharedWorker channel is open. Pinning the order keeps a later edit from
        reintroducing that window.
        """
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.calls, "install_ssrf_guard registered nothing at all"
        assert context.calls[0] == "add_init_script", (
            f"the SharedWorker init script must be installed before the route handlers, "
            f"or a raising route registration leaves the channel open: {context.calls}"
        )
        assert "route" in context.calls and "route_web_socket" in context.calls, context.calls


class _GrantRaisingContext(_FakeContext):
    async def grant_permissions(self, permissions: Any, origin: Any = None) -> None:
        self.calls.append("grant_permissions")
        raise RuntimeError("Unknown permission: local-network-access")


class TestLocalNetworkAccessGrantIsScoped:
    """The LNA grant restores the ``allow_hosts`` contract on Playwright >= 1.60.

    The behavioural proof is the Chromium integration test
    ``test_guarded_context_still_allows_an_allow_listed_websocket``. That test
    cannot tell a scoped grant from a context-wide one (the guard still refuses
    the internal socket either way), so the scope is pinned here: a grant with
    ``origin=None`` would strip Chromium's Local Network Access check from every
    public page in the context, removing the browser's own second layer.
    """

    @pytest.mark.asyncio
    async def test_no_allow_hosts_means_no_grant(self) -> None:
        context = _FakeContext()
        await install_ssrf_guard(context)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.grants == [], (
            f"a guard with no allow_hosts granted local-network-access: {context.grants}"
        )

    @pytest.mark.asyncio
    async def test_grant_names_exactly_the_allow_listed_origins(self) -> None:
        context = _FakeContext()
        hosts = ["minio.internal:9000", "Artifacts.Example"]
        await install_ssrf_guard(context, allow_hosts=hosts)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.grants == [
            (("local-network-access",), "http://minio.internal:9000"),
            (("local-network-access",), "https://minio.internal:9000"),
            (("local-network-access",), "http://artifacts.example"),
            (("local-network-access",), "https://artifacts.example"),
        ], context.grants
        assert all(origin for _, origin in context.grants), (
            f"a grant without an origin applies to EVERY page in the context: {context.grants}"
        )

    @pytest.mark.asyncio
    async def test_an_entry_that_is_not_host_port_is_never_granted(self) -> None:
        context = _FakeContext()
        hosts = ["http://evil.example/x", "user@host", "  "]
        await install_ssrf_guard(context, allow_hosts=hosts)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.grants == [], context.grants

    @pytest.mark.asyncio
    async def test_a_failed_grant_leaves_the_route_handlers_in_force(self) -> None:
        """Fail-closed: a failed grant only keeps Chromium refusing those sockets."""
        context = _GrantRaisingContext()
        hosts = ["minio.internal:9000"]
        await install_ssrf_guard(context, allow_hosts=hosts)  # ty: ignore[invalid-argument-type]  # why: _FakeContext duck-types BrowserContext

        assert context.handler is not None and context.ws_handler is not None, context.calls
        assert context.calls.index("route_web_socket") < context.calls.index("grant_permissions")
