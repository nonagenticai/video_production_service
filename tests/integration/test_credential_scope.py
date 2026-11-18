"""Real-Chromium proof that recording credentials reach ONLY the login's bound origin.

Two loopback servers stand in for two origins: ``A`` (``127.0.0.1``, the recorded
site the login is bound to) and ``B`` (``localhost``, any other origin — a third-
party sub-resource, a redirect target, a host that answers 401). Two different HOSTS
(not just ports) so host-scoped cookies are tested too. Each server records the
``Authorization`` / ``Cookie`` headers it received per path.

The recorder's own helpers build the context exactly as the recording paths do
(``_context_auth_kwargs`` + ``install_ssrf_guard(**_credential_scope)``,
and nothing else), so these tests exercise the shipped wiring, and a
NON-VACUITY case shows the harness does observe a leak when credentials are set
the old way (context ``extra_http_headers``).

Skipped when Chromium is not installed, like ``test_playwright_ssrf_guard.py``.
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from src.services.recording.url_recorder import UrlRecorder
from src.shared.security.playwright_guard import install_ssrf_guard
from tests.integration.test_playwright_ssrf_guard import requires_chromium

pytestmark = pytest.mark.integration

_TOKEN = "tok-bound-only"  # pragma: allowlist secret
_PASSWORD = "basic-bound-only"  # pragma: allowlist secret
_SID = "sid=SECRET-cookie"  # pragma: allowlist secret
_SITE_COOKIE = "site=SITE-SECRET"  # pragma: allowlist secret


class _Origin:
    """One loopback HTTP server that records what each path received."""

    def __init__(self, host: str) -> None:
        self.host = host
        self.seen: dict[str, dict[str, str | None]] = {}
        self.other = ""  # base URL of the other origin, set after both exist
        self.same_host = ""  # same HOST, different PORT (another origin)
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                auth = self.headers.get("Authorization")
                origin.seen[self.path] = {"auth": auth, "cookie": self.headers.get("Cookie")}
                self._send(*_respond(origin, self.path, auth))

            def _send(self, status: int, headers: dict[str, str], body: bytes = b"") -> None:
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"


def _respond(origin: _Origin, path: str, auth: str | None) -> tuple[int, dict[str, str], bytes]:
    """What each test route answers: (status, headers, body)."""
    if path == "/page":
        body = (
            f'<html><body><img src="/same.png"><img src="{origin.other}/third.png"></body></html>'
        ).encode()
        return (200, {"Content-Type": "text/html"}, body)
    if path == "/redirect":
        return (302, {"Location": f"{origin.other}/landing"}, b"")
    if path == "/fetchpage":
        # The PAGE's own fetch: same-origin, credentials included, with a
        # page-set Authorization — then the server 302s it off-origin.
        body = (
            b"<html><body><script>fetch('/redir-fetch', {credentials:"
            b"'include', headers: {Authorization: 'Bearer page-set'}})"
            b".finally(() => { window.done = true; });</script></body></html>"
        )
        # The SITE's own session cookie lands in the browser jar, so the page's
        # fetch below carries it (in request.headers) to the 302ing hop.
        return (200, {"Content-Type": "text/html", "Set-Cookie": f"{_SITE_COOKIE}; Path=/"}, body)
    if path == "/redir-fetch":
        return (
            302,
            {
                "Location": f"{origin.other}/fetch-landing",
                "Access-Control-Allow-Origin": "*",
            },
            b"",
        )
    if path == "/portpage":
        body = f'<html><body><img src="{origin.same_host}/port.png"></body></html>'.encode()
        return (200, {"Content-Type": "text/html"}, body)
    if path.startswith("/basic") and not auth:
        return (401, {"WWW-Authenticate": 'Basic realm="r"'}, b"")
    return (200, {"Content-Type": "text/plain"}, b"ok")


@pytest.fixture
def origins() -> Iterator[tuple[_Origin, _Origin]]:
    a, b = _Origin("127.0.0.1"), _Origin("localhost")
    a.other, b.other = b.base, a.base
    yield a, b
    a.server.shutdown()
    b.server.shutdown()


@pytest.fixture
def other_port() -> Iterator[_Origin]:
    c = _Origin("127.0.0.1")
    yield c
    c.server.shutdown()


async def _visit(
    a: _Origin,
    b: _Origin,
    auth: dict[str, Any],
    path: str,
    *,
    legacy: bool = False,
    extra_hosts: tuple[str, ...] = (),
) -> None:
    from playwright.async_api import async_playwright

    url = f"{a.base}{path}"
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            kwargs = UrlRecorder._context_auth_kwargs(auth, url)
            if legacy:  # the pre-fix wiring: a context-wide header
                kwargs["extra_http_headers"] = {"Authorization": f"Bearer {_TOKEN}"}
            context = await browser.new_context(**kwargs)
            scope = {} if legacy else UrlRecorder._credential_scope(auth, url)
            await install_ssrf_guard(
                context,
                allow_hosts=[f"127.0.0.1:{a.port}", f"localhost:{b.port}", *extra_hosts],
                **scope,
            )
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle")
            if path == "/fetchpage":
                await page.wait_for_function("window.done === true", timeout=10_000)
        finally:
            await browser.close()


_BEARER = {"type": "bearer", "token": _TOKEN}


@requires_chromium
class TestBearerScopedToTheBoundOrigin:
    @pytest.mark.asyncio
    async def test_same_origin_requests_get_the_token(self, origins) -> None:
        a, b = origins
        await _visit(a, b, _BEARER, "/page")
        assert a.seen["/page"]["auth"] == f"Bearer {_TOKEN}"
        assert a.seen["/same.png"]["auth"] == f"Bearer {_TOKEN}"

    @pytest.mark.asyncio
    async def test_a_cross_origin_sub_resource_gets_no_token(self, origins) -> None:
        a, b = origins
        await _visit(a, b, _BEARER, "/page")
        assert "/third.png" in b.seen  # the request was made...
        assert b.seen["/third.png"]["auth"] is None  # ...without the credential

    @pytest.mark.asyncio
    async def test_a_302_to_another_origin_gets_no_token(self, origins) -> None:
        a, b = origins
        await _visit(a, b, _BEARER, "/redirect")
        assert a.seen["/redirect"]["auth"] == f"Bearer {_TOKEN}"
        assert "/landing" in b.seen
        assert b.seen["/landing"]["auth"] is None

    @pytest.mark.asyncio
    async def test_non_vacuity_the_old_wiring_leaks(self, origins) -> None:
        """The harness can SEE a leak: a context-wide header reaches origin B."""
        a, b = origins
        await _visit(a, b, _BEARER, "/page", legacy=True)
        assert b.seen["/third.png"]["auth"] == f"Bearer {_TOKEN}"


@requires_chromium
class TestBasicAuthAnsweredOnlyForTheBoundOrigin:
    @pytest.mark.asyncio
    async def test_the_bound_origin_challenge_is_answered(self, origins) -> None:
        a, b = origins
        auth = {"type": "basic", "username": "u", "password": _PASSWORD}
        await _visit(a, b, auth, "/basic")
        assert (a.seen["/basic"]["auth"] or "").startswith("Basic ")

    @pytest.mark.asyncio
    async def test_a_401_from_another_origin_gets_no_basic(self, origins) -> None:
        a, b = origins
        a.other = f"{b.base}/basic?"  # the page's third-party image now answers 401
        auth = {"type": "basic", "username": "u", "password": _PASSWORD}
        await _visit(a, b, auth, "/page")
        challenged = [p for p in b.seen if p.startswith("/basic")]
        assert challenged, "the cross-origin 401 was never requested"
        assert all(b.seen[p]["auth"] is None for p in challenged)


@requires_chromium
class TestCookiesAreHostOnly:
    @pytest.mark.asyncio
    async def test_a_cookie_with_a_foreign_domain_still_only_reaches_the_bound_host(
        self, origins
    ) -> None:
        a, b = origins
        auth = {
            "type": "cookies",
            "cookies": [{"name": "sid", "value": "s3cr3t", "domain": "localhost"}],
        }
        await _visit(a, b, auth, "/page")
        assert "sid=s3cr3t" in (a.seen["/page"]["cookie"] or "")
        assert b.seen["/third.png"]["cookie"] is None


@requires_chromium
class TestPageFetchRedirectedOffOrigin:
    @pytest.mark.asyncio
    async def test_the_other_origin_gets_no_cookie_and_no_authorization(self, origins) -> None:
        """A same-origin fetch (cookie + page-set Authorization) that 302s to another
        origin: the guard performs the hop itself, so IT must drop both — the
        browser's own cross-origin-redirect stripping never runs."""
        a, b = origins
        auth = {"type": "cookies", "cookie_header": _SID}
        await _visit(a, b, auth, "/fetchpage")
        sent = a.seen["/redir-fetch"]
        # Non-vacuity: the entry hop carried the page's Authorization, the site's
        # jar cookie AND the login cookie — so the next assertion can fail.
        assert sent["auth"] == "Bearer page-set"
        assert _SITE_COOKIE in (sent["cookie"] or "") and _SID in (sent["cookie"] or "")
        assert "/fetch-landing" in b.seen
        assert b.seen["/fetch-landing"] == {"auth": None, "cookie": None}


@requires_chromium
class TestCookiesStayOnTheBoundPort:
    @pytest.mark.asyncio
    async def test_same_host_other_port_gets_no_cookie(self, origins, other_port) -> None:
        """RFC 6265 cookies ignore ports, so a login cookie in the browser jar would
        also reach 127.0.0.1:<other port>. The login cookie is therefore never put in
        the jar: the guard sends it as a header on hops to the bound ORIGIN only."""
        a, b = origins
        a.same_host = other_port.base
        auth = {"type": "cookies", "cookie_header": _SID}
        await _visit(a, b, auth, "/portpage", extra_hosts=(f"127.0.0.1:{other_port.port}",))
        assert _SID in (a.seen["/portpage"]["cookie"] or "")
        assert "/port.png" in other_port.seen
        assert other_port.seen["/port.png"]["cookie"] is None
