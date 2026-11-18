"""BrowserContext-level SSRF interception for Playwright.

Why this exists
---------------
``validate_public_url`` before ``page.goto`` covers exactly **hop 0 of the
top-level document** and nothing else. A headless browser is not an HTTP
client: after that first navigation it independently issues requests for
every ``<img>``, ``<script>``, ``<link>``, ``<iframe>``, stylesheet ``@import``,
web font, in-page ``fetch``/XHR, ``window.location`` assignment and 30x
redirect target -- resolving each hostname itself, through Chromium's own
resolver, which no Python-side guard can see or pin.

So a page the guard *did* clear can serve::

    <img src="http://100.64.0.6:5000/v2/_catalog">

and the browser will fetch it. Whether the bytes come back to the caller is
beside the point: the request is issued from inside the pod network, load-bearing
side effects happen, and response *timing* alone is a working port scanner.

The fix is to intercept at the layer where the browser announces every request
it is about to make. :func:`install_ssrf_guard` registers a route handler on the
**BrowserContext**, not the Page -- context level covers all current and future
pages, popups and frames created in that context, which ``page.route`` does not.

What is and is not routed (measured, not assumed)
-------------------------------------------------
Two successive versions of this docstring made confident claims about worker and
WebSocket coverage, in opposite directions, and both were wrong. The following
was measured against real headless Chromium on **both** pinned Playwright
versions (1.58.0 in ``analysis_service``, 1.59.0 in
``video_production_service``), with request provenance read from
``Request.service_worker`` rather than inferred from the URL, and each case
paired with an unguarded control arm:

* **Service Workers ARE routed.** A ``fetch`` issued from inside a registered
  Service Worker reaches this handler and is refused there; the instrumented
  internal host recorded no request at all, while the unguarded control arm
  recorded one. The claim that context routing "does not intercept requests
  issued from a Service Worker" is therefore false and is retracted. So is the
  ``service_workers="block"`` option that had been added at every call site to
  compensate for it: it closed nothing that was open, and it broke pages whose
  first paint gates on ``navigator.serviceWorker.ready`` -- that promise never
  settles under ``"block"``, and ``wait_until="networkidle"`` does not catch it
  because the network genuinely *is* idle while the promise hangs.
* **WebSockets were NOT routed, and now are.** ``context.route`` is never
  consulted for a ``ws://`` / ``wss://`` handshake. Until this module also
  registered ``context.route_web_socket`` (below), a page the guard had cleared
  could open a full-duplex channel to any internal address: measured live, the
  page read internal bytes back *and* the internal host received the page's
  frames. That is strictly worse than the ``<img>`` vector this module was
  built for, since it both exfiltrates and writes. Both schemes are now judged
  by the same validator, against the same per-origin verdict cache, and refused
  the same way.
* **An allowed WebSocket is opened by the PAGE, so Chromium's Local Network
  Access (LNA) check applies to it -- from the Chromium bundled with Playwright
  1.60.** ``WebSocketRoute.connect_to_server()`` does not dial from the driver:
  Playwright's injected WebSocket mock constructs the real socket inside the
  page (``_apiConnect``). Every document this guard lets through is delivered by
  ``route.fulfill`` (redirects are followed hop by hop via ``route.fetch``), and
  a fulfilled response carries no remote endpoint, so Chromium classes the
  document as **public** address space. From Playwright 1.60 the bundled
  Chromium enforces LNA on WebSockets, so a socket the guard had *approved* to an
  allow-listed internal host was refused by the browser with
  ``net::ERR_BLOCKED_BY_LOCAL_NETWORK_ACCESS_CHECKS`` (measured over the
  protocol on 1.64.1; 1.59.0 never enforced it). That is fail-closed, not a
  bypass -- but it silently broke the ``allow_hosts`` contract, and it is why
  ``playwright`` was briefly capped ``<1.60``, which froze Chromium security
  fixes. :func:`install_ssrf_guard` therefore grants the ``local-network-access``
  permission to the **allow-listed origins only** (see
  :func:`_grant_local_network_access`), never context-wide: documents from any
  other origin keep LNA as a second, independent layer under this guard.
* **SharedWorkers are NOT routed, they were NOT fail-closed, and they are now
  removed.** A ``fetch`` issued from a ``SharedWorker`` never reaches this
  handler at all -- confirmed with the ``_MATCH_ALL`` regex *and* with a
  ``"**/*"`` glob, so it is not an artifact of the pattern. The mechanism is
  Playwright's, not Chromium's: ``CRBrowser._onAttachedToTarget`` builds a
  session only for ``page`` and ``service_worker`` targets and calls
  ``session.detach()`` on every other target type, so a ``shared_worker`` target
  is released before any ``Fetch`` domain is enabled on it. There is no
  ``route_shared_worker`` and no ``shared_workers`` context option on either
  pinned version to opt back in with -- the routing surface is exactly
  ``route`` / ``route_from_har`` / ``route_web_socket``.

  An earlier revision of this docstring called that "fail-closed by an accident
  of interception". **That was wrong, and it is retracted.** Measured
  server-side on a recording listener -- ``accept()`` only returns for a
  COMPLETED TCP handshake, so this is not inferred from a client-side error
  string -- with an unguarded control arm on every case:

  - **The TCP connection lands.** A guarded cross-origin SharedWorker fetch
    produced ``accepts=1, request_lines=[], byteless_conns=1``: the handshake
    completed against the internal host and then **zero application bytes** were
    sent. Compare a properly routed source in the same harness -- a page
    ``fetch``, an ``<img>``, a *dedicated* ``Worker`` -- all of which produced
    ``accepts=0``, because the guard aborts the route before a socket is opened.
    So this was never the same fail-closed as everything else here.
  - **It is a working reachability oracle.** From inside the worker, timing a
    capped ``fetch`` at n=20 per class: an OPEN internal port answered in a
    median **4.75 ms**, a CLOSED one in **0.85 ms**, and a FILTERED/blackholed
    address ran to the **1200 ms** abort cap every single time. Filtered vs
    reachable is unambiguous on a single sample; open vs closed separates on the
    median (5.6x) but its per-sample ranges overlap, so that half is statistical
    rather than single-shot. The unguarded control arm shows the same shape
    (7.00 / 0.70 / 1208 ms), which is what proves the instrument is not blunt.
    This module's own opening paragraph names exactly this as the threat:
    "response *timing* alone is a working port scanner".
  - **No exfiltration.** No request bytes ever reached the internal host and no
    response bytes ever reached the page. The exposure is host/port *discovery*
    plus whatever a bare connect costs the target -- materially less than the
    ``<img>`` and WebSocket vectors above, and materially more than nothing.
  - Allow-listing the target changed none of this, which is the measurement that
    rules out "the guard judged it".

  Because it is a real primitive rather than a curiosity, it is closed, and
  closed at the layer this module actually owns: :func:`install_ssrf_guard`
  installs :data:`_DISABLE_SHARED_WORKER_JS` via ``context.add_init_script``.
  See that constant for why an init script -- rather than a launch flag -- is
  the right instrument, and for the realm-by-realm bypass measurements behind
  the claim that it holds.

What this does NOT close
------------------------
* **DNS rebinding inside Chromium.** The handler resolves the URL to decide,
  then hands the request back to Chromium, which resolves it *again* on its own.
  A hostile authoritative server can answer differently the second time. Unlike
  httpx (where :func:`w2v_common.net.safe_get` pins the validated IP into the
  connection) there is no supported way to pin a Chromium connection, so this
  window is open by construction. It is narrowed by the per-origin verdict cache
  below -- one verdict per origin per context, so a name cannot be re-judged
  favourably later within the same page load.
* **Non-network schemes.** ``file://``, ``chrome://`` and friends are loaded by
  Chromium without going through the network stack, so a route handler is never
  consulted for them. Those must be rejected on the **entry URL** before
  ``goto`` -- which is why callers still run ``validate_public_url`` first
  (it raises ``ValueError`` on any non-http(s) scheme). The two checks are
  complementary, not redundant; neither one alone is sufficient.

Consolidation note
------------------
This module is intentionally byte-identical to
``analysis_service``'s ``src/shared/security/playwright_guard.py``.
Its correct long-term home is ``w2v_common.net`` (alongside ``safe_get``) so
Playwright sinks get shared fixes the way httpx sinks already do; publishing a
new ``w2v_common`` was out of scope for the change that introduced it. Until
then, a fix here must be mirrored in the sibling repo.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING
from urllib.parse import urljoin, urlsplit, urlunsplit

from w2v_common.net import SSRFError, validate_public_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from playwright.async_api import BrowserContext, Request, Route, WebSocketRoute

logger = logging.getLogger(__name__)


def _is_target_closed(exc: BaseException) -> bool:
    """True when ``exc`` is Playwright's "target has been closed" teardown error.

    Matched on the message rather than the class because ``TargetClosedError``
    lives in ``playwright._impl._errors`` — a private module whose path has moved
    between releases, so importing it would couple this security-critical handler
    to Playwright internals. The public ``playwright.async_api.Error`` is the base
    of every Playwright error, so it cannot distinguish this case on its own.
    """
    return "has been closed" in str(exc)


# Matches every request URL. A glob such as ``"**/*"`` is resolved relative to
# the context's ``base_url`` and does not reliably match non-http schemes, so
# the verdict would silently depend on how the context was constructed. An
# explicit "match everything" regex cannot drift that way.
_MATCH_ALL = re.compile(r".*")

# Schemes a page may legitimately use that never leave the browser. ``data:``
# and ``blob:`` carry their payload inline; ``about:blank`` is the empty page
# Playwright itself navigates to when opening a tab. None of them can name a
# network target, so there is nothing to resolve.
_NON_NETWORK_SCHEMES = frozenset({"data", "blob", "about"})

# Chromium's error code for "the client refused this request". Surfaces to the
# page as a normal network failure (net::ERR_BLOCKED_BY_CLIENT) and to a blocked
# top-level ``goto`` as a Playwright error, so a blocked navigation is loud.
_ABORT_REASON = "blockedbyclient"

# WebSocket schemes, mapped to the http(s) scheme that names the SAME connection
# target. ``validate_public_url`` raises ``ValueError`` on anything that is not
# http(s), so a ``ws://`` URL handed to it straight would be refused for the
# wrong reason -- and would refuse every *public* ``wss://`` endpoint too, which
# is a behavioural break (live chat, analytics, HMR) rather than a security win.
# The mapping is exact: scheme is not an input to the host/port resolution the
# validator performs, so ``ws://h:p/x`` and ``http://h:p/x`` are the same
# question. It also means a host judged once is judged once for both -- the
# per-origin verdict cache is keyed on the mapped origin, so an origin already
# refused over http cannot be re-judged favourably over ws.
_WS_TO_HTTP_SCHEME = {"ws": "http", "wss": "https"}

# RFC 6455 "Policy Violation". Delivered to the page as the ``code`` on the
# ``close`` event, so a refused socket looks like a deliberate rejection rather
# than a network flake, and never silently hangs half-open.
_WS_CLOSE_CODE = 1008
_WS_CLOSE_REASON = "blocked by client"

# Removes the ``SharedWorker`` constructor from every realm the page can reach.
#
# Why this exists: SharedWorker network traffic is invisible to ``context.route``
# (Playwright detaches ``shared_worker`` targets before enabling ``Fetch`` on
# them -- see the module docstring), and measurement showed that is NOT
# fail-closed: the TCP handshake completes against the target while no
# application bytes are sent, which hands the page a reachability/port oracle
# that every routed request source in the same harness does not have.
#
# Why an init script rather than ``--disable-blink-features=SharedWorker``: that
# flag does work (measured -- and a bogus feature name leaves the constructor
# intact, so the flag is really doing the work), but it is a **launch**
# argument. This function is handed a context that is already open, so adopting
# the flag would mean editing every ``chromium.launch`` call site in both repos
# and would leave any future call site silently unprotected. The guard must be
# able to close its own gap at the layer it is given.
#
# Why it is not a fake fix: an init script runs before page script in EVERY
# realm, so the usual "grab a pristine constructor from a fresh realm" bypass
# does not apply. Measured against a page that mints six of them -- a
# same-origin ``src`` iframe, a ``srcdoc`` iframe, a src-less ``about:blank``
# iframe, a ``data:`` iframe, a ``blob:`` iframe, and a grandchild iframe
# created from inside a child realm. Without the script all six report
# ``typeof SharedWorker === 'function'`` and construction succeeds; with it all
# six report ``undefined`` and construction fails. (The ``data:`` iframe is a
# cross-origin opaque realm the page cannot reach into at all.)
#
# ``delete`` rather than a throwing stub is deliberate: it leaves ``typeof
# SharedWorker === 'undefined'``, which is the check every real feature-detecting
# site already performs, so such a page takes its own documented fallback path.
# A throwing getter would instead break feature detection itself. This is the
# opposite of the ``service_workers="block"`` mistake, which hung
# ``navigator.serviceWorker.ready`` forever and recorded a spinner: nothing here
# gates first paint, and pages depending on cross-origin SharedWorker fetches
# were already failing -- silently, at fetch time -- before this change.
_DISABLE_SHARED_WORKER_JS = "try { delete self.SharedWorker; } catch (e) {}"

# Redirect statuses whose Location must be re-judged before the next hop.
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})

# Chain length cap, matching ``w2v_common.net.DEFAULT_MAX_REDIRECTS`` so the
# browser sinks and the httpx sinks agree on what "too many" means.
DEFAULT_MAX_REDIRECTS = 5


def _origin_key(url: str) -> str:
    """Return the ``scheme://host:port`` key a verdict is cached under.

    Path and query are deliberately excluded: SSRF is decided entirely by where
    the connection goes, so every URL sharing an origin shares a verdict. This
    keeps one page load from issuing hundreds of duplicate DNS lookups, and
    doubles as a rebinding narrowing -- an origin judged unsafe stays unsafe for
    the life of the context.
    """
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


async def _origin_is_public(
    url: str,
    allow_hosts: tuple[str, ...],
    verdicts: dict[str, bool],
) -> bool:
    """Return True if *url*'s origin is a safe public target, memoised per origin.

    ``validate_public_url`` calls :func:`socket.getaddrinfo`, which is blocking;
    running it inline would stall the event loop driving the browser for the
    duration of every uncached DNS lookup, on a code path that fires once per
    sub-resource. It is therefore dispatched to a worker thread.
    """
    key = _origin_key(url)
    cached = verdicts.get(key)
    if cached is not None:
        return cached

    try:
        await asyncio.to_thread(validate_public_url, url, allow_hosts=allow_hosts)
    except (SSRFError, ValueError) as exc:
        logger.warning("SSRF guard: refusing browser request to %s (%s)", key, exc)
        verdicts[key] = False
        return False
    except OSError as exc:
        # Resolver failure that is not gaierror (e.g. EAI_AGAIN surfacing as a
        # bare OSError). Fail CLOSED, matching validate_public_url's own stance.
        logger.warning("SSRF guard: resolver error for %s, refusing (%s)", key, exc)
        verdicts[key] = False
        return False

    verdicts[key] = True
    return True


async def install_ssrf_guard(
    context: BrowserContext,
    *,
    allow_hosts: Iterable[str] | None = None,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    credential_origin: str | None = None,
    credential_headers: Mapping[str, str] | None = None,
) -> dict[str, bool]:
    """Route every request the context makes through the fleet SSRF guard.

    Registers **two** handlers, because one does not cover the other:
    ``context.route`` for HTTP(S) traffic, and ``context.route_web_socket`` for
    ``ws``/``wss`` handshakes, which ``context.route`` is never consulted for.
    Both share the single per-origin verdict cache returned below, so a target
    is judged once regardless of which channel asks for it.

    Also installs :data:`_DISABLE_SHARED_WORKER_JS`, because there is a third
    channel that **cannot** be routed: SharedWorker traffic is never offered to
    a route handler on either pinned Playwright version, and measurement showed
    the resulting failure is not fail-closed -- the TCP handshake completes
    against the target while no bytes are sent, which is a reachability oracle.
    Removing the constructor is the only instrument available at context level.

    Args:
        context: The :class:`~playwright.async_api.BrowserContext` to guard.
            Install this **before** the first navigation -- a handler registered
            afterwards does not retroactively cover requests already in flight.
        allow_hosts: Narrow, config-driven list of hosts (``host`` or
            ``host:port``) that legitimately resolve into an internal range --
            e.g. a presigned object-store endpoint. Anything not named here is
            judged on its resolved IP. Blocking by *name* is never the mechanism;
            this list is the only name-based input and it only ever *widens*.
        max_redirects: Maximum hops in a redirect chain before it is refused.
        credential_origin: The ONE origin (``scheme://host[:port]``) a login is
            bound to. ``credential_headers`` ride only on hops to exactly this
            origin — never on a third-party sub-resource, never on a hop a
            cross-origin redirect leads to. Because this guard performs every hop
            itself (``route.fetch``, ``max_redirects=0``), it is the one place a
            per-hop decision is possible: Playwright applies
            ``route.continue_(headers=...)`` and context ``extra_http_headers`` to
            EVERY request and every redirect they initiate.
        credential_headers: Auth headers (e.g. ``Authorization: Bearer``) to add
            on hops to ``credential_origin`` only.

    Returns:
        The live per-origin verdict cache (``origin -> allowed``). Returned so
        tests can assert which origins were judged, and what the verdict was,
        without reaching into the closure.
    """
    verdicts: dict[str, bool] = {}
    allowed = tuple(allow_hosts or ())
    scope = _CredentialScope(credential_origin, credential_headers)

    async def _handler(route: Route, request: Request) -> None:
        url = request.url
        try:
            scheme = urlsplit(url).scheme.lower()

            if scheme in _NON_NETWORK_SCHEMES:
                await route.continue_()
                return

            if scheme not in ("http", "https"):
                logger.warning(
                    "SSRF guard: refusing %s request with scheme %r: %s",
                    request.resource_type,
                    scheme,
                    url,
                )
                await route.abort(_ABORT_REASON)
                return

            if not await _origin_is_public(url, allowed, verdicts):
                logger.warning(
                    "SSRF guard: blocked %s request to non-public target %s",
                    request.resource_type,
                    url,
                )
                await route.abort(_ABORT_REASON)
                return

            await _follow_redirects_guarded(route, request, allowed, verdicts, max_redirects, scope)
        except asyncio.CancelledError:
            # Control flow, not a failure: the task driving this request is
            # being torn down (context/browser closing, caller timing out).
            # Answering the route here would race the teardown, and swallowing
            # it would break cancellation. Re-raise untouched. Redundant under
            # CPython -- CancelledError derives from BaseException, so the
            # clause below cannot see it -- but it pins that fact so widening
            # the catch-all later cannot silently start eating cancellations.
            raise
        except Exception as exc:
            # Teardown is not a fault — the sibling of the CancelledError above.
            # When the context or browser closes with sub-resource requests still
            # in flight, `route.fetch` raises TargetClosedError ("Target page,
            # context or browser has been closed"). It is not a CancelledError,
            # so it landed in the catch-all below and was logged at ERROR with
            # "refusing" — implying the guard had rejected a legitimate asset,
            # when in fact the page it belonged to no longer existed.
            #
            # This matters beyond tidiness: it was the ONLY error line in the
            # entire w2v namespace across an hour of live traffic (measured
            # 2026-08-10), so the single signal an operator has was a false one.
            #
            # Kept inside the same clause, not a sibling `except`, because a
            # sibling would have to re-`raise` to reach the catch-all — and a
            # raising route handler is exactly the ~30s Chromium stall the
            # totality comment below exists to prevent. Only a CLOSED target is
            # demoted; every other error still logs at ERROR.
            if _is_target_closed(exc):
                logger.debug(
                    "SSRF guard: %s abandoned, target already closed -- refusing",
                    url,
                    exc_info=True,
                )
                await _abort_quietly(route, url)
                return
            # The handler must be TOTAL. A Playwright route handler that raises
            # neither continues nor aborts the request: Chromium is left holding
            # an unanswered route until the navigation times out (~30 s per
            # affected request). That is fail-closed, not fail-open, so it is
            # not a bypass -- but it turns any unanticipated bug (a raising
            # `route.fetch`, a validator raising something outside the
            # SSRFError/ValueError/OSError set `_origin_is_public` handles) into
            # a stall instead of an immediate, logged refusal.
            logger.exception("SSRF guard: unexpected error handling %s -- refusing", url)
            await _abort_quietly(route, url)

    async def _ws_handler(ws: WebSocketRoute) -> None:
        # A thin closure over the two pieces of per-context state; the decision
        # itself lives at module scope so it can be read (and tested) without
        # reaching through ``install_ssrf_guard``.
        await _guard_websocket(ws, allowed, verdicts)

    # Installed BEFORE the route handlers so that an exception from either
    # registration cannot leave a context whose SharedWorker channel is open
    # while the caller believes the guard is installed. Init scripts apply to
    # every document created after this point, which is the same "install before
    # the first navigation" contract the route handlers already carry.
    await context.add_init_script(_DISABLE_SHARED_WORKER_JS)

    await context.route(_MATCH_ALL, _handler)
    # Registered separately because ``context.route`` is NEVER consulted for a
    # WebSocket handshake -- measured, see the module docstring. Without this a
    # page cleared by the guard opens a full-duplex read/write channel to any
    # internal address, which is a strictly worse SSRF primitive than the
    # sub-resource vector above.
    await context.route_web_socket(_MATCH_ALL, _ws_handler)
    await _grant_local_network_access(context, allowed)
    return verdicts


def _allow_listed_origins(allowed: tuple[str, ...]) -> list[str]:
    """Return the ``http``/``https`` origins named by the ``allow_hosts`` entries.

    An entry is ``host`` or ``host:port`` (no scheme), so each yields both
    schemes. A bare ``host`` names the scheme-default port, which is exactly the
    browser-origin reading of ``http://host`` / ``https://host``.
    """
    origins: list[str] = []
    for entry in allowed:
        netloc = entry.strip().lower()
        if not netloc or "/" in netloc or "@" in netloc:
            # Not a host[:port] entry; granting a permission on a guessed origin
            # would be widening on a misread, so skip it (the guard still judges
            # every request from such a page; Chromium's LNA stays in force).
            logger.warning("SSRF guard: not granting local-network-access for entry %r", entry)
            continue
        origins.extend(f"{scheme}://{netloc}" for scheme in ("http", "https"))
    return origins


async def _grant_local_network_access(context: BrowserContext, allowed: tuple[str, ...]) -> None:
    """Let documents from allow-listed origins open the sockets the guard approved.

    See the module docstring: an approved WebSocket is dialled by the page, and
    from Playwright 1.60 the bundled Chromium refuses a page-initiated socket to
    a private/loopback address unless the *requesting origin* holds the
    ``local-network-access`` permission. A guard-fulfilled document is always
    classed public, so without this an allow-listed internal WebSocket can never
    open, whatever the guard decided.

    Scope is deliberately the allow-listed origins, never ``"*"``:

    * ``allow_hosts`` is already the operator's declaration that those origins
      are legitimately internal, so their documents reaching the local network
      is the trust the list expresses -- and everything they request is still
      judged here (HTTP via ``context.route``, WebSockets via
      ``route_web_socket``).
    * Documents from every other origin keep Chromium's LNA check as a second,
      independent layer, which also covers channels Playwright cannot route.
    * No ``allow_hosts`` => no grant => no behaviour change at all.

    Known, fail-closed limit: a document from a NON-allow-listed (public) origin
    opening a WebSocket to an allow-listed internal host is still refused by
    Chromium's LNA even though this guard approves it. Widening the grant to
    cover that would remove LNA from every public page in the context.

    A failed grant is logged and swallowed: the only consequence is that
    Chromium keeps refusing those sockets, which is the fail-closed direction,
    and it must not prevent the route handlers above from being in force.
    """
    origins = _allow_listed_origins(allowed)
    for origin in origins:
        try:
            await context.grant_permissions(["local-network-access"], origin=origin)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "SSRF guard: could not grant local-network-access to %s; allow-listed "
                "internal WebSockets from it stay blocked by the browser",
                origin,
                exc_info=True,
            )


async def _guard_websocket(
    ws: WebSocketRoute,
    allowed: tuple[str, ...],
    verdicts: dict[str, bool],
) -> None:
    """Judge one WebSocket handshake and either connect it or close it.

    Fail-closed by construction: Playwright dials the real server only when a
    handler calls ``connect_to_server()``. Every path that is not an explicit
    allow -- an exception, an early return, a future edit that forgets a branch
    -- therefore refuses, because it simply never opts in. With no message
    handlers registered on the returned server route, frames pass through
    untouched in both directions, so an allowed socket behaves exactly as it
    would with no guard installed.
    """
    url = ws.url
    try:
        if not await _ws_target_is_public(url, allowed, verdicts):
            await ws.close(code=_WS_CLOSE_CODE, reason=_WS_CLOSE_REASON)
            return
        ws.connect_to_server()
    except asyncio.CancelledError:
        raise
    except Exception:
        # Same totality requirement as the HTTP handler: a raising WebSocket
        # handler leaves ``connect_to_server`` uncalled, which is fail-closed
        # but silent. Log it and close the page side explicitly so the page
        # gets a definite answer instead of a socket that never opens.
        logger.exception("SSRF guard: unexpected error handling WebSocket %s -- refusing", url)
        await _close_quietly(ws, url)


async def _ws_target_is_public(
    url: str,
    allowed: tuple[str, ...],
    verdicts: dict[str, bool],
) -> bool:
    """Judge a ``ws://``/``wss://`` target with the same validator as everything else.

    Any other scheme is refused outright rather than mapped: a WebSocket route is
    only ever invoked for a handshake, so an unrecognised scheme here means an
    assumption has broken, and the safe reading of a broken assumption is "no".
    """
    parts = urlsplit(url)
    http_scheme = _WS_TO_HTTP_SCHEME.get(parts.scheme.lower())
    if http_scheme is None:
        logger.warning("SSRF guard: refusing WebSocket with unexpected scheme: %s", url)
        return False

    probe = urlunsplit((http_scheme, parts.netloc, parts.path, parts.query, ""))
    if not await _origin_is_public(probe, allowed, verdicts):
        logger.warning("SSRF guard: blocked WebSocket to non-public target %s", url)
        return False
    return True


async def _close_quietly(ws: WebSocketRoute, url: str) -> None:
    """Close *ws* from the page side, tolerating a socket that is already gone.

    Only reached from the WebSocket handler's catch-all, where the socket may
    already have been closed or torn down. ``WebSocketRoute.close`` swallows its
    own transport errors, but the belt-and-braces wrapper keeps a future change
    to that behaviour from re-raising out of a handler -- which would strand the
    page on an unanswered route, exactly the stall ``_abort_quietly`` exists to
    prevent on the HTTP side.
    """
    try:
        await ws.close(code=_WS_CLOSE_CODE, reason=_WS_CLOSE_REASON)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("SSRF guard: could not close WebSocket %s after an unexpected error", url)


async def _abort_quietly(route: Route, url: str) -> None:
    """Abort *route*, tolerating a route that is already resolved.

    Only reached from the handler's catch-all, where the request may already
    have been fulfilled or aborted before the failure. Playwright raises on a
    second attempt to resolve the same route, and letting that escape would
    re-open the very stall this exists to close.
    """
    try:
        await route.abort(_ABORT_REASON)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("SSRF guard: could not abort %s after an unexpected error", url)


#: Request headers that carry a credential; dropped on any hop that leaves the
#: request's own origin (the Fetch spec drops Authorization on a cross-origin
#: redirect; a hand-rolled redirect loop has to do it itself).
_CREDENTIAL_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie"})


def _origin(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    port = parts.port
    default = {"http": 80, "https": 443}.get(parts.scheme.lower())
    netloc = host if port in (None, default) else f"{host}:{port}"
    return f"{parts.scheme.lower()}://{netloc}"


class _CredentialScope:
    """Which headers one hop may carry: credentials only on the bound origin."""

    def __init__(self, origin: str | None, headers: Mapping[str, str] | None) -> None:
        self._origin = _origin(origin) if origin else None
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}

    def hop_headers(
        self, request_headers: Mapping[str, str], entry: str, hop: str
    ) -> dict[str, str]:
        hop_origin = _origin(hop)
        out = {k: v for k, v in request_headers.items()}
        # The Fetch spec drops credentials on a redirect that leaves the request's
        # origin; this loop performs the hops itself, so it must do the same — else
        # the page's own Cookie / Authorization ride to whatever host a 302 names.
        # (Not stripped on a page's own cross-origin request: those are the app's
        # deliberate calls, and the browser jar already scopes cookies by host.)
        if hop_origin != _origin(entry):
            out = {k: v for k, v in out.items() if k.lower() not in _CREDENTIAL_HEADERS}
        if self._headers and hop_origin == self._origin:
            for name, value in self._headers.items():
                current = next((k for k in out if k.lower() == name), None)
                if name == "cookie" and current is not None:
                    out[current] = f"{out[current]}; {value}"
                else:
                    if current is not None:
                        del out[current]
                    out[name] = value
        return out


async def _follow_redirects_guarded(
    route: Route,
    request: Request,
    allowed: tuple[str, ...],
    verdicts: dict[str, bool],
    max_redirects: int,
    scope: _CredentialScope | None = None,
) -> None:
    """Walk the redirect chain, judging every hop, then fulfil the final response.

    ``route.continue_()`` cannot be used once the entry URL is cleared, because
    **Playwright does not re-invoke a route handler for redirect hops**. Verified
    against Chromium 147 with a 302 from an allowed host to the estate's Docker
    registry: with a ``continue_()``-based handler the registry request was
    issued and answered (400) while the handler was called exactly once, for the
    entry URL. That holds for top-level navigations *and* sub-resources, so
    "validate on the way in and let the browser continue" leaves the entire
    redirect vector open -- which is the single easiest SSRF bypass to write.

    Instead the request is performed with ``max_redirects=0`` and each
    ``Location`` is resolved and re-judged here before the next hop is made. The
    final response is handed back to the page via ``route.fulfill``, so the page
    still sees exactly what it would have seen.
    """
    current = request.url
    scope = scope or _CredentialScope(None, None)
    for _hop in range(max_redirects + 1):
        headers = scope.hop_headers(request.headers, request.url, current)
        response = await route.fetch(url=current, headers=headers, max_redirects=0)
        if response.status not in _REDIRECT_STATUS:
            await route.fulfill(response=response)
            return

        location = response.headers.get("location")
        if not location:
            logger.warning("SSRF guard: redirect from %s carried no Location", current)
            await route.abort(_ABORT_REASON)
            return

        # Relative Locations are legal and common; resolve before judging.
        nxt = urljoin(current, location)
        scheme = urlsplit(nxt).scheme.lower()
        if scheme not in ("http", "https") or not await _origin_is_public(nxt, allowed, verdicts):
            logger.warning(
                "SSRF guard: blocked redirect %s -> %s (non-public target)", current, nxt
            )
            await route.abort(_ABORT_REASON)
            return
        current = nxt

    logger.warning("SSRF guard: too many redirects (> %d) from %s", max_redirects, request.url)
    await route.abort(_ABORT_REASON)
