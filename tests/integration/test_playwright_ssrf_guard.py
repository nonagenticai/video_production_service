"""Real-Chromium proof that the BrowserContext SSRF guard actually intercepts.

Everything else about this guard can be asserted with doubles. The one claim
that cannot is the claim the guard exists for: *a real browser, rendering a real
page, issues its own requests for sub-resources and redirect targets, and those
requests are intercepted.* A fake ``Route``/``Request`` pair proves the handler
decides correctly; it cannot prove the handler is ever consulted.

So each case here is paired with a **non-vacuity** run: the same page is loaded
once WITHOUT the guard, where the internal request must be observed going out,
and once WITH it, where it must not be. If the guard were a no-op the guarded
half fails; if the fixture were not actually reaching the internal target the
unguarded half fails.

Why the test server is reached through ``allow_hosts``
------------------------------------------------------
The guard blocks ``127.0.0.1``, so the loopback test server cannot be addressed
by default. Rather than invent fake public hostnames -- which cannot work here,
because ``route.fetch`` resolves through the browser *server's* resolver and
ignores Chromium's ``--host-resolver-rules`` -- the server is named in the
guard's ``allow_hosts`` trust-list, exactly as a presigned object-store host
would be in production. That also gets the allow-list itself under test: the
allowed host is reachable, and a redirect *from* it to a non-allowed internal
address is still refused.

Skipped when the Chromium build is not present: the CI lint/unit job runs on a
bare runner via ``uv``, not inside the service image (which does bake in
Chromium, see the Dockerfile). The double-based tests in
``tests/unit/test_recording_and_compile_ssrf.py`` cover the same decision logic
unconditionally.

This file is the sibling of ``analysis_service``'s test of the same name; both
exercise the byte-identical ``src/shared/security/playwright_guard.py``. It is
duplicated rather than shared for the same reason the helper is -- see that
module's consolidation note.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import http.server
import socket
import threading
import urllib.parse
from collections.abc import Iterator
from typing import Any, cast

import pytest

# The Docker registry on worker3 -- RFC 6598 CGNAT, inside the tailnet, and the
# range `ipaddress.is_private` reports as False.
# The redirect/sub-resource target: a loopback URL served by the fixture server.
# In the private estate this was a registry on the tailnet, unreachable from a
# public runner. Loopback is inside the guard's blocked ranges, so the guarded
# arm must refuse it and the unguarded control must reach it.
REGISTRY_URL = ""
REGISTRY_ORIGIN = ""


# The reachable stand-ins for the internal target and the legitimate resource.
class _TargetHandler(http.server.BaseHTTPRequestHandler):
    """Serves the internal-target URL on its own loopback port.

    It must be a DIFFERENT origin from the page server: the page server's own
    origin is in the guard's allow-list, so pointing the "internal" target at it
    makes the guarded arm pass for the wrong reason (measured 2026-11-18).
    """

    def do_GET(self) -> None:
        body = b"{}"
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        return


_STATIC_RESPONSES = {
    "/ok.png": ("image/png", bytes.fromhex("89504e470d0a1a0a")),  # pragma: allowlist secret
    "/v2/_catalog": ("application/json", b"{}"),
}

pytestmark = pytest.mark.integration


def _chromium_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - playwright is a hard dependency
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            browser.close()
    except Exception:  # noqa: BLE001 - any launch failure means "not available"
        return False
    return True


requires_chromium = pytest.mark.skipif(
    not _chromium_available(),
    reason="Chromium build not installed (`playwright install chromium`)",
)


# Reports ``typeof SharedWorker`` in the top realm and in every fresh realm a
# hostile page can mint. Module-level so the handler stays simple and so the
# blob is a named constant rather than an inline literal.
_SHARED_WORKER_AVAILABILITY_PAGE = """<!doctype html><meta charset=utf-8><body><script>
window.__avail='pending';
function T(w,l){try{return l+'='+(typeof w.SharedWorker);}
  catch(e){return l+'=err:'+e.name;}}
window.addEventListener('DOMContentLoaded',()=>{(async()=>{try{
  const mk=(cfg)=>new Promise(res=>{const f=document.createElement('iframe');
    if(cfg.src)f.src=cfg.src; if(cfg.srcdoc)f.srcdoc=cfg.srcdoc;
    f.onload=()=>res(f); f.onerror=()=>res(f);
    document.body.appendChild(f);
    if(!cfg.src&&!cfg.srcdoc)res(f);});
  const o=[T(self,'top')];
  const a=await mk({src:'/blank'});     o.push(T(a.contentWindow,'iframe_src'));
  const b=await mk({srcdoc:'<p>x</p>'});o.push(T(b.contentWindow,'iframe_srcdoc'));
  const c=await mk({});                 o.push(T(c.contentWindow,'iframe_blank'));
  try{const u=URL.createObjectURL(new Blob(['<p>x</p>'],{type:'text/html'}));
      const d=await mk({src:u});        o.push(T(d.contentWindow,'iframe_blob'));}
  catch(e){o.push('iframe_blob=err:'+e.name);}
  try{const w=a.contentWindow;const g=w.document.createElement('iframe');
      w.document.body.appendChild(g);   o.push(T(g.contentWindow,'iframe_nested'));}
  catch(e){o.push('iframe_nested=err:'+e.name);}
  window.__avail=o.join('|');
}catch(e){window.__avail='page_threw:'+e;}})();});
</script></body>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serves the attacker-controlled page, plus one legitimate sub-resource."""

    # Signature must match BaseHTTPRequestHandler.log_message exactly, or ty
    # reports an invalid override.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(self, status: int, headers: dict[str, str], body: bytes = b"") -> None:
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _target_port(self) -> str:
        """The ``?target=`` port the WebSocket / Service Worker pages aim at.

        Passed in the URL rather than read from a module global so each case
        gets its own freshly-recording internal server and the pages stay
        stateless.
        """
        query = urllib.parse.urlparse(self.path).query
        return urllib.parse.parse_qs(query).get("target", [""])[0]

    def _serve_worker_route(self, route: str) -> bool:
        """Serve the SharedWorker / dedicated-Worker pages. True if handled.

        Split out of :meth:`do_GET` purely to keep that method under the
        complexity ceiling; these routes are otherwise ordinary.
        """
        if route == "/shw-page":
            # A SharedWorker whose fetch aims at the internal host. ``no-cors``
            # for the same reason as the Service Worker page: the response is
            # irrelevant, the SSRF is the request going out.
            body = (
                "<!doctype html><meta charset=utf-8><body><script>\n"
                "window.__shw='pending';\n"
                "try{\n"
                f"  const w=new SharedWorker('/shw.js?target={self._target_port()}');\n"
                "  w.port.onmessage=e=>{window.__shw=e.data;};\n"
                "  w.onerror=()=>{window.__shw='workererror';};\n"
                "  w.port.start();\n"
                "  w.port.postMessage('go');\n"
                "}catch(err){window.__shw='ctor_threw:'+err;}\n"
                "</script></body>"
            ).encode()
            self._send(200, {"content-type": "text/html", "cache-control": "no-store"}, body)
            return True
        if route == "/shw.js":
            body = (
                "self.onconnect=function(e){const p=e.ports[0];p.onmessage=function(){\n"
                "  (async()=>{try{\n"
                f"    await fetch('http://127.0.0.1:{self._target_port()}{_SHW_FETCH_PATH}',"
                "{mode:'no-cors'});\n"
                "    p.postMessage('shw_fetch_ok');\n"
                "  }catch(err){p.postMessage('threw:'+err);}})();\n"
                "};p.start();};\n"
            ).encode()
            self._send(
                200,
                {"content-type": "text/javascript", "cache-control": "no-store"},
                body,
            )
            return True
        if route == "/dedicated-worker-page":
            # Control for the mitigation: a DEDICATED Worker is routed normally
            # and must keep working after SharedWorker is removed.
            body = (
                "<!doctype html><meta charset=utf-8><body><script>\n"
                "window.__dw='pending';\n"
                f"const w=new Worker('/dw.js?target={self._target_port()}');\n"
                "w.onmessage=e=>{window.__dw=e.data;};\n"
                "w.postMessage('go');\n"
                "</script></body>"
            ).encode()
            self._send(200, {"content-type": "text/html", "cache-control": "no-store"}, body)
            return True
        if route == "/dw.js":
            body = (
                "self.onmessage=function(){(async()=>{try{\n"
                f"  await fetch('http://127.0.0.1:{self._target_port()}/FROM-DEDICATED-WORKER',"
                "{mode:'no-cors'});\n"
                "  postMessage('dw_fetch_ok');\n"
                "}catch(err){postMessage('threw:'+err);}})();};\n"
            ).encode()
            self._send(
                200,
                {"content-type": "text/javascript", "cache-control": "no-store"},
                body,
            )
            return True
        if route == "/shared-worker-availability":
            self._send(
                200,
                {"content-type": "text/html", "cache-control": "no-store"},
                _SHARED_WORKER_AVAILABILITY_PAGE.encode(),
            )
            return True
        return False

    def do_GET(self) -> None:
        # ``self.server`` is typed as BaseServer, which has no port; the
        # concrete server here is always an HTTPServer.
        port = cast("http.server.HTTPServer", self.server).server_port
        route = urllib.parse.urlparse(self.path).path
        if self._serve_worker_route(route):
            return
        if route == "/ws-page":
            # The attack: a page the guard cleared opens a full-duplex socket to
            # an internal address. It reads internal bytes back into ``__ws``
            # AND writes a frame the internal host receives.
            body = (
                "<!doctype html><meta charset=utf-8><body><script>\n"
                "window.__ws='pending';\n"
                f"const ws=new WebSocket('ws://127.0.0.1:{self._target_port()}"
                "/internal-endpoint');\n"
                "ws.onopen=()=>{ws.send('PAGE-WROTE-THIS');};\n"
                "ws.onmessage=e=>{window.__ws='read:'+e.data;};\n"
                "ws.onerror=()=>{if(window.__ws==='pending')window.__ws='error';};\n"
                "ws.onclose=e=>{if(window.__ws==='pending')"
                "window.__ws='closed:'+e.code;};\n"
                "</script></body>"
            ).encode()
            self._send(200, {"content-type": "text/html", "cache-control": "no-store"}, body)
            return
        if route == "/sw-page":
            body = (
                "<!doctype html><meta charset=utf-8><body><script>\n"
                "window.__sw='pending';\n"
                f"navigator.serviceWorker.register('/sw.js?target={self._target_port()}')\n"
                "  .then(async (reg)=>{\n"
                "    const r=await navigator.serviceWorker.ready;\n"
                "    const worker=r.active||reg.active;\n"
                "    const ch=new MessageChannel();\n"
                "    ch.port1.onmessage=e=>{window.__sw=e.data;};\n"
                "    window.__sw='sent';\n"
                "    worker.postMessage('go',[ch.port2]);\n"
                "  }).catch(e=>{window.__sw='register_failed:'+e;});\n"
                "</script></body>"
            ).encode()
            self._send(200, {"content-type": "text/html", "cache-control": "no-store"}, body)
            return
        if route == "/sw.js":
            # Fetches from INSIDE a Service Worker. ``no-cors`` because the
            # response is irrelevant -- the SSRF is the request going out.
            body = (
                "self.addEventListener('install',e=>self.skipWaiting());\n"
                "self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));\n"
                "self.addEventListener('message',e=>{const p=e.ports[0];(async()=>{\n"
                f"  try{{await fetch('http://127.0.0.1:{self._target_port()}"
                "/FROM-SERVICE-WORKER',{mode:'no-cors'});\n"
                "        p.postMessage('sw_fetch_ok');}\n"
                "  catch(err){p.postMessage('threw:'+err);}\n"
                "})();});\n"
            ).encode()
            self._send(
                200,
                {"content-type": "text/javascript", "cache-control": "no-store"},
                body,
            )
            return
        if route == "/blank":
            self._send(200, {"content-type": "text/html"}, b"<p>blank</p>")
            return
        if self.path == "/page":
            body = (
                "<html><body>"
                # The whole attack: the tenant's page names an internal target
                # and the BROWSER fetches it. No Python-side check of the
                # top-level URL can see this.
                f'<img src="{REGISTRY_URL}">'
                f'<img src="http://127.0.0.1:{port}/ok.png">'
                "</body></html>"
            ).encode()
            self._send(200, {"content-type": "text/html"}, body)
        elif self.path in ("/redirect", "/imgredirect"):
            self._send(302, {"location": REGISTRY_URL})
        elif self.path == "/page-with-redirecting-image":
            body = (
                f'<html><body><img src="http://127.0.0.1:{port}/imgredirect"></body></html>'
            ).encode()
            self._send(200, {"content-type": "text/html"}, body)
        elif self.path in _STATIC_RESPONSES:
            content_type, body = _STATIC_RESPONSES[self.path]
            self._send(200, {"content-type": content_type}, body)
        else:
            self._send(404, {})


@pytest.fixture(scope="module")
def server() -> Iterator[int]:
    global REGISTRY_URL, REGISTRY_ORIGIN
    target = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TargetHandler)
    threading.Thread(target=target.serve_forever, daemon=True).start()
    REGISTRY_URL = f"http://127.0.0.1:{target.server_port}/v2/_catalog"
    REGISTRY_ORIGIN = f"http://127.0.0.1:{target.server_port}"
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        target.shutdown()
        target.server_close()


async def _load(
    path: str, port: int, *, guarded: bool
) -> tuple[list[str], list[str], dict[str, bool]]:
    """Load ``http://127.0.0.1:<port><path>``.

    Returns ``(requested, succeeded, verdicts)``.

    ``requested`` is deliberately NOT the "did it reach the network" signal: the
    ``request`` event fires when Chromium *announces* a request, before route
    interception resolves, so a request the guard aborts still appears there.
    ``succeeded`` (from the ``response`` event) is the signal that bytes came
    back, and is what the guarded assertions read.
    """
    from playwright.async_api import async_playwright

    from src.shared.security.playwright_guard import install_ssrf_guard

    requested: list[str] = []
    succeeded: list[str] = []
    verdicts: dict[str, bool] = {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            if guarded:
                # The test server stands in for a legitimately-reachable
                # artifact host, the way a presigned MinIO endpoint would.
                verdicts = await install_ssrf_guard(context, allow_hosts=[f"127.0.0.1:{port}"])
            page = await context.new_page()
            page.on("request", lambda r: requested.append(r.url))
            page.on("response", lambda r: succeeded.append(r.url))
            # A blocked top-level navigation raises; the collected events are
            # what the assertions read, so the raise itself is not a failure.
            with contextlib.suppress(Exception):
                await page.goto(f"http://127.0.0.1:{port}{path}", wait_until="load", timeout=15000)
            await page.wait_for_timeout(700)
            await page.close()
            await context.close()
        finally:
            await browser.close()

    return requested, succeeded, verdicts


@requires_chromium
class TestSubResourceInterception:
    @pytest.mark.asyncio
    async def test_unguarded_context_does_request_the_internal_target(self, server: int) -> None:
        """Non-vacuity: without the guard the browser really does reach out.

        This is what unmodified ``origin/main`` does, and it is what makes the
        guarded assertion below meaningful rather than decorative.
        """
        _requested, succeeded, _ = await _load("/page", server, guarded=False)

        assert REGISTRY_URL in succeeded, (
            "the sub-resource request never reached the registry -- the fixture "
            "is not exercising the vulnerability, so the guarded test proves nothing"
        )

    @pytest.mark.asyncio
    async def test_guarded_context_blocks_the_internal_sub_resource(self, server: int) -> None:
        _requested, succeeded, verdicts = await _load("/page", server, guarded=True)

        assert verdicts.get(REGISTRY_ORIGIN) is False, (
            f"the registry origin was never judged: {verdicts}"
        )
        assert REGISTRY_URL not in succeeded, "the registry answered a blocked request"

    @pytest.mark.asyncio
    async def test_guarded_context_still_loads_the_legitimate_sub_resource(
        self, server: int
    ) -> None:
        """The guard must not simply block everything."""
        ok_url = f"http://127.0.0.1:{server}/ok.png"
        _requested, succeeded, verdicts = await _load("/page", server, guarded=True)

        # The allow-listed host is fetched and really answers.
        assert ok_url in succeeded
        assert verdicts.get(REGISTRY_ORIGIN) is False


@requires_chromium
class TestRedirectInterception:
    """``route.continue_()`` is NOT re-invoked for redirect hops.

    Verified against Chromium 147: with a ``continue_()``-based handler, a 302
    from the allowed host to the registry produced a real request *and a real
    400 response from worker3*, while the handler ran exactly once (for the
    entry URL). The guard therefore walks the chain itself via
    ``route.fetch(max_redirects=0)``, which is what these tests pin.
    """

    @pytest.mark.asyncio
    async def test_unguarded_navigation_follows_redirect_to_internal_host(
        self, server: int
    ) -> None:
        """Non-vacuity for the top-level redirect case."""
        _requested, succeeded, _ = await _load("/redirect", server, guarded=False)

        assert REGISTRY_URL in succeeded

    @pytest.mark.asyncio
    async def test_guarded_navigation_blocks_redirect_to_internal_host(self, server: int) -> None:
        """Hop 0 is an allowed host; hop 1 is the registry."""
        requested, succeeded, verdicts = await _load("/redirect", server, guarded=True)

        assert verdicts.get(REGISTRY_ORIGIN) is False, (
            f"the redirect target was never judged: {verdicts}"
        )
        # The guard walks the chain itself, so the browser is never even told
        # about the redirect target.
        assert REGISTRY_URL not in requested
        assert REGISTRY_URL not in succeeded

    @pytest.mark.asyncio
    async def test_unguarded_sub_resource_follows_redirect_to_internal_host(
        self, server: int
    ) -> None:
        """Non-vacuity for the sub-resource redirect case."""
        _requested, succeeded, _ = await _load(
            "/page-with-redirecting-image", server, guarded=False
        )

        assert REGISTRY_URL in succeeded

    @pytest.mark.asyncio
    async def test_guarded_sub_resource_blocks_redirect_to_internal_host(self, server: int) -> None:
        requested, succeeded, verdicts = await _load(
            "/page-with-redirecting-image", server, guarded=True
        )

        assert verdicts.get(REGISTRY_ORIGIN) is False
        assert REGISTRY_URL not in requested
        assert REGISTRY_URL not in succeeded


# --------------------------------------------------------------------------
# WebSocket + Service Worker coverage
#
# ``context.route`` is not the whole story, and both of the confident claims
# previously made about the rest of it were wrong. These cases measure it.
#
# The target below is a raw-socket server on a SECOND loopback port that is NOT
# in ``allow_hosts``, so the guard is required to refuse it. It records the
# literal request line it received, which is the only signal that cannot be
# faked by a broken fixture: if nothing arrives, either the guard worked or the
# harness is dead -- which is exactly why every case here is paired with an
# unguarded control arm that MUST see the request arrive.
# --------------------------------------------------------------------------

_WS_ACCEPT_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_WS_EXFIL_PAYLOAD = b"INTERNAL-DATA-EXFILTRATED"
_SW_FETCH_PATH = "/FROM-SERVICE-WORKER"
_SHW_FETCH_PATH = "/FROM-SHARED-WORKER"
_DW_FETCH_PATH = "/FROM-DEDICATED-WORKER"


class _InternalTarget:
    """A loopback stand-in for an internal host, speaking HTTP *and* WebSocket.

    Deliberately raw sockets rather than ``http.server``: the point of the
    WebSocket case is the 101 upgrade and the frames after it, which
    ``BaseHTTPRequestHandler`` cannot carry.
    """

    def __init__(self) -> None:
        self.request_lines: list[str] = []
        self.frames_from_page: list[bytes] = []
        #: Completed TCP handshakes, counted SEPARATELY from application bytes.
        #: ``accept()`` only returns once the 3-way handshake is done, so this is
        #: server-side proof a connection LANDED -- which is a different fact
        #: from "a request arrived", and the difference is the entire
        #: SharedWorker finding: connections landed while ``request_lines``
        #: stayed empty. Asserting only on ``request_lines`` would have called
        #: that fail-closed.
        self.accepts = 0
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self.port: int = self._sock.getsockname()[1]
        self._stop = False

    @property
    def origin(self) -> str:
        return f"127.0.0.1:{self.port}"

    def serve_forever(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self.accepts += 1
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def close(self) -> None:
        self._stop = True
        with contextlib.suppress(OSError):
            self._sock.close()

    @staticmethod
    def _unmask(frame: bytes) -> bytes:
        length = frame[1] & 0x7F
        offset = 2
        if length == 126:
            length = int.from_bytes(frame[2:4], "big")
            offset = 4
        mask = frame[offset : offset + 4]
        body = frame[offset + 4 : offset + 4 + length]
        return bytes(byte ^ mask[i % 4] for i, byte in enumerate(body))

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(5)
            raw = conn.recv(8192)
            if not raw:
                return
            text = raw.decode("latin1", "replace")
            self.request_lines.append(text.split("\r\n")[0])

            keys = [
                line.split(":", 1)[1].strip()
                for line in text.split("\r\n")
                if line.lower().startswith("sec-websocket-key:")
            ]
            if not keys:
                body = b"INTERNAL-HTTP-BODY"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/plain\r\n"
                    b"access-control-allow-origin: *\r\n"
                    b"content-length: " + str(len(body)).encode() + b"\r\n\r\n" + body
                )
                return

            accept = base64.b64encode(
                hashlib.sha1(keys[0].encode() + _WS_ACCEPT_MAGIC).digest()  # noqa: S324
            ).decode()
            conn.sendall(
                b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept.encode() + b"\r\n\r\n"
            )
            # Exfiltration half: internal bytes travel back to the page.
            conn.sendall(b"\x81" + bytes([len(_WS_EXFIL_PAYLOAD)]) + _WS_EXFIL_PAYLOAD)
            # Write half: whatever the page sends reaches the internal host.
            with contextlib.suppress(OSError, TimeoutError):
                frame = conn.recv(4096)
                if frame:
                    self.frames_from_page.append(self._unmask(frame))
        finally:
            with contextlib.suppress(OSError):
                conn.close()


@pytest.fixture
def internal_target() -> Iterator[_InternalTarget]:
    """Function-scoped: each case needs its own empty recording."""
    target = _InternalTarget()
    thread = threading.Thread(target=target.serve_forever, daemon=True)
    thread.start()
    try:
        yield target
    finally:
        target.close()


async def _drive(
    path: str, port: int, *, allow: list[str] | None, probe: str, settle_ms: int = 4000
) -> tuple[str, dict[str, bool]]:
    """Load ``path``, wait, and return ``(page's own verdict, guard verdict cache)``.

    ``allow`` is ``None`` for the unguarded control arm -- no guard installed at
    all, which is what ``origin/main`` does and what makes the guarded assertion
    mean something.
    """
    from playwright.async_api import async_playwright

    from src.shared.security.playwright_guard import install_ssrf_guard

    verdicts: dict[str, bool] = {}
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            if allow is not None:
                verdicts = await install_ssrf_guard(context, allow_hosts=allow)
            page = await context.new_page()
            with contextlib.suppress(Exception):
                await page.goto(f"http://127.0.0.1:{port}{path}", wait_until="load", timeout=15000)
            await page.wait_for_timeout(settle_ms)
            result = str(await page.evaluate(probe))
            await page.close()
            await context.close()
        finally:
            await browser.close()
    return result, verdicts


@requires_chromium
class TestWebSocketInterception:
    """``context.route`` is NEVER consulted for a ``ws://`` handshake.

    Measured live before the fix: a page on an allowed origin opened
    ``ws://127.0.0.1:<internal>/internal-endpoint``, the route handler was never
    called, the internal host logged the request, and the page read internal
    bytes back. That is a full-duplex channel -- strictly worse than the ``<img>``
    vector this module was built for, because it writes as well as reads. The
    guard therefore registers ``context.route_web_socket`` as well, and these
    cases pin both halves.
    """

    @pytest.mark.asyncio
    async def test_unguarded_context_opens_a_channel_to_the_internal_host(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """Non-vacuity: the vector is real and the fixture exercises it."""
        verdict, _ = await _drive(
            f"/ws-page?target={internal_target.port}", server, allow=None, probe="window.__ws"
        )

        assert internal_target.request_lines, (
            "the internal host was never contacted -- the fixture is not "
            "exercising the vulnerability, so the guarded case below proves nothing"
        )
        assert _WS_EXFIL_PAYLOAD.decode() in verdict, (
            f"the page did not read internal bytes back: {verdict!r}"
        )
        assert internal_target.frames_from_page == [b"PAGE-WROTE-THIS"], (
            f"the write half did not reach the internal host: {internal_target.frames_from_page!r}"
        )

    @pytest.mark.asyncio
    async def test_guarded_context_refuses_the_internal_websocket(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """The fix. Fails on the branch before ``route_web_socket`` was added."""
        verdict, verdicts = await _drive(
            f"/ws-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}"],
            probe="window.__ws",
        )

        # The ws:// target is judged under its http:// equivalent, so one origin
        # cannot be refused over http and quietly allowed over ws.
        assert verdicts.get(f"http://127.0.0.1:{internal_target.port}") is False, (
            f"the WebSocket target was never judged by the guard: {verdicts}"
        )
        assert internal_target.request_lines == [], (
            f"the internal host answered a WebSocket the guard had to refuse: "
            f"{internal_target.request_lines}"
        )
        assert internal_target.frames_from_page == []
        assert _WS_EXFIL_PAYLOAD.decode() not in verdict, (
            f"internal bytes still reached the page: {verdict!r}"
        )

    @pytest.mark.asyncio
    async def test_guarded_context_still_allows_an_allow_listed_websocket(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """The guard must judge WebSockets, not simply kill them.

        Without this, ``test_guarded_context_refuses_...`` would also pass if the
        handler closed every socket unconditionally -- which would silently break
        live-chat, analytics and HMR sockets on every recorded page.
        """
        verdict, verdicts = await _drive(
            f"/ws-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}", internal_target.origin],
            probe="window.__ws",
        )

        assert verdicts.get(f"http://127.0.0.1:{internal_target.port}") is True, verdicts
        assert internal_target.request_lines, "an allow-listed WebSocket was refused"
        assert _WS_EXFIL_PAYLOAD.decode() in verdict, (
            f"frames did not pass through to the page: {verdict!r}"
        )
        assert internal_target.frames_from_page == [b"PAGE-WROTE-THIS"], (
            f"frames did not pass through to the server: {internal_target.frames_from_page!r}"
        )


@requires_chromium
class TestServiceWorkerFetchesAreRouted:
    """Regression alarm for the claim that replaced ``service_workers="block"``.

    That option was added on the belief that context routing does not intercept
    Service Worker requests. It does -- measured on both pinned Playwright
    versions, with provenance read from ``Request.service_worker``. The option
    was therefore removed, because it closed nothing and it hangs
    ``navigator.serviceWorker.ready`` forever, which records a spinner instead of
    the page (``networkidle`` does not catch it: the network really is idle
    while the promise hangs).

    The residual risk is a future Playwright/Chromium version regressing that
    coverage. This pair is the alarm for it, and it is the reason dropping the
    option is safe rather than merely cheap.
    """

    @pytest.mark.asyncio
    async def test_unguarded_service_worker_reaches_the_internal_host(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """Non-vacuity: a Service Worker really can issue the request."""
        verdict, _ = await _drive(
            f"/sw-page?target={internal_target.port}",
            server,
            allow=None,
            probe="window.__sw",
            settle_ms=6000,
        )

        assert verdict == "sw_fetch_ok", f"the Service Worker never fetched: {verdict!r}"
        assert [line for line in internal_target.request_lines if _SW_FETCH_PATH in line], (
            f"the Service Worker's fetch never reached the internal host: "
            f"{internal_target.request_lines}"
        )

    @pytest.mark.asyncio
    async def test_guarded_context_intercepts_the_service_worker_fetch(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        verdict, verdicts = await _drive(
            f"/sw-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}"],
            probe="window.__sw",
            settle_ms=6000,
        )

        assert verdicts.get(f"http://127.0.0.1:{internal_target.port}") is False, (
            "the Service Worker's target was never judged -- context routing no "
            f"longer covers Service Workers, which is the regression this test exists for: {verdicts}"
        )
        assert internal_target.request_lines == [], (
            f"a Service Worker fetch reached the internal host past the guard: "
            f"{internal_target.request_lines}"
        )
        assert verdict.startswith("threw:"), (
            f"the Service Worker's fetch was expected to be refused: {verdict!r}"
        )


# --------------------------------------------------------------------------
# SharedWorker
#
# The third channel, and the one that could not be routed. Playwright's
# ``CRBrowser._onAttachedToTarget`` builds a session only for ``page`` and
# ``service_worker`` targets and detaches everything else, so a
# ``shared_worker`` target never gets ``Fetch`` enabled and its requests are
# never offered to ``context.route``. There is no ``route_shared_worker`` and no
# ``shared_workers`` context option on either pinned version.
#
# A previous revision recorded that as "fail-closed by accident". These cases
# exist because that was measurably false, and because the fix must be shown to
# cost nothing that still works.
# --------------------------------------------------------------------------


@requires_chromium
class TestSharedWorkerCannotReachInternalHosts:
    """The guard removes ``SharedWorker``; these pin why, and that it holds.

    The load-bearing assertion is ``internal_target.accepts == 0``, NOT
    ``request_lines == []``. Before the fix, ``request_lines`` was already empty
    while the TCP handshake completed -- so a suite that only asserted on
    request lines would have gone green over a live reachability oracle. That is
    the exact shape of "a green test kept the bug alive", and separating the two
    counters is what stops it recurring.
    """

    @pytest.mark.asyncio
    async def test_unguarded_shared_worker_reaches_the_internal_host(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """Non-vacuity: the vector is real and the fixture exercises it."""
        verdict, _ = await _drive(
            f"/shw-page?target={internal_target.port}",
            server,
            allow=None,
            probe="window.__shw",
            settle_ms=6000,
        )

        assert verdict == "shw_fetch_ok", f"the SharedWorker never fetched: {verdict!r}"
        assert [line for line in internal_target.request_lines if _SHW_FETCH_PATH in line], (
            f"the SharedWorker's fetch never reached the internal host -- the fixture is "
            f"not exercising the vector: {internal_target.request_lines}"
        )
        assert internal_target.accepts > 0

    @pytest.mark.asyncio
    async def test_guarded_context_opens_no_connection_to_the_internal_host(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """The fix.

        RED before ``_DISABLE_SHARED_WORKER_JS``: ``accepts`` was 1 -- the
        handshake completed against the internal host and then no bytes were
        sent, which is precisely the timing oracle. GREEN after: the constructor
        is gone, so no request is ever issued and no socket is ever opened.
        """
        verdict, _verdicts = await _drive(
            f"/shw-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}"],
            probe="window.__shw",
            settle_ms=6000,
        )

        assert internal_target.accepts == 0, (
            f"a SharedWorker still opened {internal_target.accepts} TCP connection(s) to the "
            f"internal host. Bytes are not required for this to matter: a completed handshake "
            f"is a reachability oracle, and it is what the pre-fix measurement found."
        )
        assert internal_target.request_lines == [], internal_target.request_lines
        # The page sees a plain ReferenceError from `new SharedWorker(...)`,
        # which is what a feature-detecting site branches on.
        assert verdict.startswith("ctor_threw:"), (
            f"expected the SharedWorker constructor to be unavailable: {verdict!r}"
        )

    @pytest.mark.asyncio
    async def test_allow_listing_the_target_does_not_change_the_outcome(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """Rules out "the guard judged it".

        ``allow_hosts`` is the guard's only widening input, and for a channel the
        guard actually decides about it is decisive -- see
        ``test_guarded_context_still_allows_an_allow_listed_websocket``. Here it
        changes nothing, before or after the fix, because this channel was never
        judged at all. That is what makes "not routed" a measurement.
        """
        _verdict, verdicts = await _drive(
            f"/shw-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}", internal_target.origin],
            probe="window.__shw",
            settle_ms=6000,
        )

        assert internal_target.accepts == 0
        assert f"http://127.0.0.1:{internal_target.port}" not in verdicts, (
            f"the SharedWorker's target was judged by the guard -- context routing now covers "
            f"SharedWorkers, so the module docstring's account is stale: {verdicts}"
        )


@requires_chromium
class TestSharedWorkerMitigationScopeAndCost:
    """False-positive controls: the mitigation must bite, and bite *only* here.

    Without the first case, every assertion above would pass just as readily in
    a world where the whole browser was broken. Without the second and third, it
    would pass in a world where the fix had also killed dedicated Workers or
    ordinary page fetches -- which is the ``service_workers="block"`` mistake
    this module already made once.
    """

    @pytest.mark.asyncio
    async def test_shared_worker_is_gone_from_every_realm_a_page_can_mint(
        self, server: int
    ) -> None:
        """The bypass measurement, as an assertion.

        An init script that only covered the top realm would be a fake fix: a
        hostile page mints a fresh realm and reads a pristine constructor out of
        it. Measured against six of them; all must report ``undefined``.
        """
        avail, _ = await _drive(
            "/shared-worker-availability",
            server,
            allow=[f"127.0.0.1:{server}"],
            probe="window.__avail",
            settle_ms=5000,
        )

        assert not avail.startswith("page_threw"), avail
        realms = dict(part.split("=", 1) for part in avail.split("|"))
        assert realms, avail
        # Every realm the page could actually reach into must be undefined. The
        # blob/nested arms can legitimately report an err:SecurityError on an
        # opaque realm -- what must never appear is a usable constructor.
        assert "function" not in realms.values(), (
            f"a pristine SharedWorker constructor survived in at least one realm, so the "
            f"mitigation is bypassable and must not be relied on: {realms}"
        )
        assert realms.get("top") == "undefined", realms

    @pytest.mark.asyncio
    async def test_unguarded_context_still_has_shared_worker(self, server: int) -> None:
        """Mutation control for the case above.

        If ``typeof SharedWorker`` were ``undefined`` for some unrelated reason
        -- a Chromium build without it, a broken probe page -- the assertion
        above would pass while measuring nothing. Without the guard installed it
        must be a ``function`` in every realm.
        """
        avail, _ = await _drive(
            "/shared-worker-availability",
            server,
            allow=None,
            probe="window.__avail",
            settle_ms=5000,
        )

        assert not avail.startswith("page_threw"), avail
        realms = dict(part.split("=", 1) for part in avail.split("|"))
        assert realms.get("top") == "function", (
            f"SharedWorker is absent even WITHOUT the guard, so the mitigation test above "
            f"proves nothing: {realms}"
        )

    @pytest.mark.asyncio
    async def test_dedicated_workers_still_work_and_are_still_judged(
        self, server: int, internal_target: _InternalTarget
    ) -> None:
        """The collateral check.

        A dedicated ``Worker`` is a different worker type, IS routed, and must
        keep working -- refused by the guard's decision (its target appears in
        the verdict cache as ``False``), not by having its constructor removed.
        """
        verdict, verdicts = await _drive(
            f"/dedicated-worker-page?target={internal_target.port}",
            server,
            allow=[f"127.0.0.1:{server}"],
            probe="window.__dw",
            settle_ms=6000,
        )

        # The worker ran (it reported back), and the guard judged its target.
        assert verdict.startswith("threw:"), (
            f"the dedicated Worker did not run or was not refused: {verdict!r}"
        )
        assert verdicts.get(f"http://127.0.0.1:{internal_target.port}") is False, (
            f"the dedicated Worker's target was never judged -- the mitigation has broken "
            f"routing for a worker type that WAS covered: {verdicts}"
        )
        assert internal_target.accepts == 0, internal_target.accepts

    @pytest.mark.asyncio
    async def test_allow_listed_sub_resource_still_loads(self, server: int) -> None:
        """Ordinary page traffic is untouched by the init script."""
        ok_url = f"http://127.0.0.1:{server}/ok.png"
        _requested, succeeded, verdicts = await _load("/page", server, guarded=True)

        assert ok_url in succeeded, "the mitigation broke ordinary sub-resource loading"
        assert verdicts.get(REGISTRY_ORIGIN) is False
