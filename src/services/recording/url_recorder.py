"""Record a live URL (e.g. an authenticated SPA) via Playwright `page.goto`.

This is the third recorder in the service, alongside :class:`ContentRecorder`
(raw HTML via ``set_content``) and :class:`PlaywrightRecorder` (pre-authored
``npx playwright test`` scripts). Unlike either, it drives a headless Chromium
against a *live* URL and can authenticate first — HTTP basic, bearer header,
a pre-collected cookie jar, or an interactive form login — which is what
capturing the real, logged-in product UI requires.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from typing import Any, TypedDict, cast
from urllib.parse import urlparse

from PIL import Image
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

from src.config import get_settings
from src.exceptions import SiteLoginFailedError
from src.services.ffmpeg import FFmpegOps
from src.services.recording.artifacts import ARTIFACT_KIND_SCREENS_ZIP
from src.services.recording.quality_presets import get_quality_preset
from src.shared.security.playwright_guard import install_ssrf_guard

logger = logging.getLogger(__name__)


class CapturedScreens(list[tuple[str, bytes]]):
    """Screens and bounded page content captured from the same authenticated DOM."""

    def __init__(self) -> None:
        super().__init__()
        self.page_contents: dict[str, dict[str, str]] = {}


ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

# Default scroll configuration (shared semantics with ContentRecorder).
_DEFAULT_SCROLL_SPEED = 300  # px per second
_DEFAULT_START_DELAY = 1.0  # seconds to pause before scrolling
_DEFAULT_END_DELAY = 2.0  # seconds to pause after scrolling
# Floor on the recorded scroll phase so single-viewport pages (which scroll
# little or not at all) still yield usable footage instead of a ~3s clip.
_DEFAULT_MIN_CLIP_SECONDS = 8.0

# Settle robustness: cap the caller-supplied post-load delay so a runaway value
# can't hang the recorder, and treat a near-uniform captured frame as blank.
_MAX_POST_LOAD_DELAY_MS = 30000
_BLANK_FRAME_LUMA_RANGE = 8  # max-min luma below this ⇒ effectively blank
_BLANK_RECHECK_MAX_MS = 5000

# Best-effort form-login selectors when auth_info doesn't override them. Broad
# enough to match Keycloak's login form (#username / #password / #kc-login) and
# most conventional forms.
_DEFAULT_USERNAME_SELECTOR = "#username, input[name='username'], input[type='email']"
_DEFAULT_PASSWORD_SELECTOR = "#password, input[name='password'], input[type='password']"  # nosec B105  # noqa: S105 -- CSS selector for the password field, not a secret
_DEFAULT_SUBMIT_SELECTOR = "#kc-login, button[type='submit'], input[type='submit']"

_GOTO_TIMEOUT_MS = 60000
_LOGIN_TIMEOUT_MS = 30000
#: Prefix of every sign-in failure message (see ``SiteLoginFailedError``).
SITE_LOGIN_FAILED = "site_login_failed"
#: A visible password input after signing in means the sign-in page is still showing.
_SIGNED_OUT_PASSWORD_SELECTOR = "input[type='password']"  # nosec B105  # noqa: S105 -- CSS selector, not a secret


class _SignInPageUnreachableError(PlaywrightError):
    """The sign-in PAGE itself could not be opened (DNS / TLS / timeout).

    Distinct from a form that could not be filled in: the page never rendered, so
    "no username / password field found" would be a false diagnosis.
    """


#: Wording that makes a page a SIGN-IN screen. Matched against the visible
#: heading and the document title only — never against a nav link, which is how
#: an ordinary content page says "Sign in" without being a sign-in screen.
#: CONTAINS, any separator ("Acme—Sign in", "Login/Register", "Sign into Acme"),
#: except when the phrase names an account-page section: a signed-in page headed
#: "Sign-in activity" or "Login history" is not a sign-in screen.
_SIGN_IN_HEADING_RE = re.compile(
    r"\b(?:sign|log)[\s\-\u2010-\u2015]*(?:into|in|on)\b"  # sign in / sign-in / login / logon
    r"(?![\s\-\u2010-\u2015]*(?:activity|history|sessions|settings|security|methods"
    r"|options|attempts|records|alerts|notifications|devices)\b)"
    r"|\bauthenticate\b",
    re.IGNORECASE,
)
#: An authentication ACTION ("Sign in with SSO"), not merely an authentication
#: LINK: the provider/SSO wording a sign-in wall carries and a header does not.
_SSO_CONTROL_RE = re.compile(
    r"\b(?:sign|log)\s*-?\s*(?:in|on)\s+with\s+\w"
    r"|\bcontinue\s+with\s+(?:google|microsoft|apple|okta|azure|saml|sso|your)\b"
    r"|\bsingle\s*-?\s*sign\s*-?\s*on\b"
    r"|\bsso\b",
    re.IGNORECASE,
)

#: One read of the page per recording, answering "what can this page see?".
#: A SNAPSHOT, not a verdict: the decision lives in ``_sign_in_evidence`` where it
#: is unit-tested. Bounded (``slice``) so a huge nav never bloats the payload.
_SIGN_IN_SNAPSHOT_JS = r"""
(passwordSelector) => {
  const isVisible = (el) => {
    if (!el || typeof el.getBoundingClientRect !== 'function') return false;
    const style = window.getComputedStyle(el);
    if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
    if (style.opacity !== '' && Number.parseFloat(style.opacity) === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect.width > 1 && rect.height > 1;
  };
  const textOf = (el) =>
    ((el.innerText || el.textContent || '') + '').replace(/\s+/g, ' ').trim();
  const firstVisible = (selector) => {
    for (const el of document.querySelectorAll(selector)) if (isVisible(el)) return el;
    return null;
  };
  const password = firstVisible(passwordSelector);
  const heading = firstVisible('h1, h2, [role="heading"]');
  const controls = Array.from(
    document.querySelectorAll(
      'a, button, [role="button"], input[type="submit"], input[type="button"]'
    )
  )
    .filter(isVisible)
    .slice(0, 60)
    .map(textOf)
    .filter((text) => text.length > 0);
  const authFields = Array.from(
    document.querySelectorAll(
      'form input[type="password"], form input[type="email"],' +
        ' input[autocomplete="username"], form input[name="username" i],' +
        ' form input[name="email" i]'
    )
  ).filter(isVisible).length;
  return {
    password: Boolean(password),
    heading: heading ? textOf(heading) : '',
    title: document.title || '',
    controls: controls,
    auth_fields: authFields,
  };
}
"""

# Navigation retry. ``networkidle`` is the strictest readiness and best for
# static pages, but it reliably TIMES OUT on SPAs that hold connections open
# (SSE / websockets / polling never go idle). Without a retry, a failed goto was
# soft-swallowed and the recorder stayed on the PREVIOUS page — so the next
# screenshot was a stale duplicate of the prior screen (a screen the narration
# never matches). Retry with progressively more lenient readiness so a transient
# or never-idle navigation still lands on the target page.
_GOTO_WAIT_STRATEGIES = ("networkidle", "load", "domcontentloaded")
_GOTO_RETRY_BACKOFF_S = 0.5

# Named per-step viewports (authored-shotlist contract values) → pixel dims.
# The recorder owns this table so the contract value ("desktop"|"tablet"|
# "mobile") flows straight through from the orchestrator; mobile is iPhone-class.
_STEP_VIEWPORTS: dict[str, tuple[int, int]] = {
    "desktop": (1920, 1080),
    "tablet": (1024, 768),
    "mobile": (390, 844),
}
# Cap on waiting for a per-step ``wait_for`` content selector before the shot.
_STEP_WAIT_FOR_TIMEOUT_MS = 10000

# Safety margin (s) kept when trimming the blank page-load lead-in, so a slightly
# imprecise wall-clock estimate never cuts into the first content frames. Only
# trim when the resulting offset is at least _MIN_TRIM_S (not worth it otherwise).
_LEAD_TRIM_MARGIN_S = 0.5
_MIN_TRIM_S = 0.5

# Per-screen interaction clips: a small pre-roll so the click that reveals a
# screen is in-frame, and a default dwell after the click so each clip is long
# enough to cover the answer line that plays over it on render.
_CLIP_PREROLL_S = 0.3
_DEFAULT_CLIP_DWELL_MS = 6500.0


class CredentialScope(TypedDict, total=False):
    """``install_ssrf_guard`` keyword arguments that bind a login to one origin."""

    credential_origin: str
    credential_headers: dict[str, str]


class UrlRecorder:
    """Navigate a live URL in headless Chromium, scroll, and capture video."""

    def __init__(self) -> None:
        settings = get_settings()
        self._viewport_width = settings.recording.viewport_width
        self._viewport_height = settings.recording.viewport_height
        self._headless = settings.recording.headless
        self._output_format = settings.recording.output_format
        self._default_quality_preset = settings.recording.default_quality_preset
        self._ffmpeg_preset = settings.ffmpeg.preset
        self._ffmpeg_timeout = settings.ffmpeg.timeout
        self._ffmpeg_default_crf = settings.ffmpeg.default_crf

    async def record_url(
        self,
        job_id: str,
        url: str,
        auth_info: dict[str, Any] | None = None,
        recording_config: dict[str, Any] | None = None,
        scroll_config: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Record a live URL being scrolled through.

        Args:
            job_id: Unique job identifier.
            url: Live URL to navigate and record.
            auth_info: Optional authentication dict (see :class:`UrlAuthInfo`).
            recording_config: Optional viewport/quality overrides.
            scroll_config: Optional scroll speed/delay overrides.
            progress_callback: Async callback for progress updates.

        Returns:
            Result dict with video_path, duration_seconds, etc.
        """
        resolved = self._resolve_config(recording_config)
        scroll = self._resolve_scroll_config(scroll_config)
        start = time.time()

        # One per-job staging root holds every temp artifact (webm + trimmed mp4);
        # recorder_service rmtrees ``staging_dir`` once the output is stored.
        staging = Path(tempfile.mkdtemp(prefix=f"url_rec_{job_id}_"))

        try:
            if progress_callback:
                await progress_callback(0.3, "launching_browser")

            video_path = await self._record_with_playwright(
                job_id, url, auth_info, resolved, scroll, progress_callback, staging
            )

            if progress_callback:
                await progress_callback(0.8, "post_processing")

            final_path = await self._post_process(video_path, resolved)

            probe = await asyncio.to_thread(FFmpegOps.probe_all, final_path)
            elapsed = time.time() - start

            if progress_callback:
                await progress_callback(1.0, "completed")

            return {
                "success": True,
                "video_path": str(final_path),
                "duration_seconds": probe.duration,
                "file_size_bytes": final_path.stat().st_size,
                "width": probe.width or resolved["viewport_width"],
                "height": probe.height or resolved["viewport_height"],
                "has_audio": probe.has_audio,
                "recording_time_seconds": elapsed,
                "staging_dir": str(staging),
            }

        except (OSError, RuntimeError, ValueError, TimeoutError, PlaywrightError) as exc:
            logger.error("URL recording failed for job %s: %s", job_id, exc)
            return {
                "success": False,
                "error": str(exc),
                "recording_time_seconds": time.time() - start,
                "staging_dir": str(staging),
            }
        except BaseException:
            # A failed sign-in (SiteLoginFailedError) propagates to
            # RecorderService instead of returning a result dict, so this is the
            # only place its staging dir can be freed.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    async def capture_screens(
        self,
        job_id: str,
        url: str,
        steps: list[dict[str, Any]],
        auth_info: dict[str, Any] | None = None,
        recording_config: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Capture one screenshot per screen in a SINGLE navigation pass.

        Navigates ``url`` once (authenticating first when ``auth_info`` is set),
        settles, then walks ``steps``: each ``click`` advances the SPA
        (soft-fail per step, like :meth:`_run_interactions`) and after the
        step's ``settle_ms`` a screenshot is taken iff the step carries a
        ``capture_id``. All PNGs are packed into ONE zip whose entry names are
        the ``capture_id``\\s, so the caller unzips to recover
        ``{screen_id: png}`` — far lighter than a 75s clip and perfectly
        static, so promo render can key each screen to its narration line.

        Returns a result dict shaped like :meth:`record_url` (``video_path`` is
        the zip path), so the recorder-service success/storage path is reused
        unchanged.
        """
        resolved = self._resolve_config(recording_config)
        start = time.time()
        # One per-job staging root holds the zip; recorder_service rmtrees
        # ``staging_dir`` once the output is stored.
        staging = Path(tempfile.mkdtemp(prefix=f"screens_{job_id}_"))
        try:
            if progress_callback:
                await progress_callback(0.3, "launching_browser")

            shots = await self._capture_with_playwright(
                url, auth_info, resolved, steps, progress_callback
            )
            if not shots:
                return {
                    "success": False,
                    "error": "no screens captured (all selectors missed or screenshots failed)",
                    "recording_time_seconds": time.time() - start,
                    "staging_dir": str(staging),
                }

            if progress_callback:
                await progress_callback(0.8, "packaging")
            zip_path = await asyncio.to_thread(self._zip_screens, shots, staging)
            elapsed = time.time() - start

            if progress_callback:
                await progress_callback(1.0, "completed")

            return {
                "success": True,
                "video_path": str(zip_path),
                # ``video_path`` is a ZIP here, not a video (the key name is
                # shared so the recorder-service success path is reused). Say so
                # explicitly: the success path verifies each kind differently and
                # ffprobe cannot read an archive.
                "artifact_kind": ARTIFACT_KIND_SCREENS_ZIP,
                "duration_seconds": 0.0,
                "file_size_bytes": zip_path.stat().st_size,
                "width": resolved["viewport_width"],
                "height": resolved["viewport_height"],
                "has_audio": False,
                "screen_ids": [cid for cid, _ in shots],
                "recording_time_seconds": elapsed,
                "staging_dir": str(staging),
            }
        except (OSError, RuntimeError, ValueError, TimeoutError, PlaywrightError) as exc:
            logger.error("Screen capture failed for job %s: %s", job_id, exc)
            return {
                "success": False,
                "error": str(exc),
                "recording_time_seconds": time.time() - start,
                "staging_dir": str(staging),
            }
        except BaseException:
            # Deliberately NOT one of the swallowed errors: a failed sign-in
            # (SiteLoginFailedError) must reach RecorderService so the job takes
            # the normal failure path. No result dict is ever returned on that
            # path, so the staging dir is removed here or it leaks.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    async def _capture_with_playwright(
        self,
        url: str,
        auth_info: dict[str, Any] | None,
        config: dict[str, Any],
        steps: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
    ) -> list[tuple[str, bytes]]:
        """Launch Playwright, authenticate, navigate, settle, walk+screenshot.

        No ``record_video_dir`` — discrete screenshots only, so this is much
        lighter than the clip recorder (no webm encode, no ffmpeg pass).
        """
        from playwright.async_api import async_playwright

        auth = auth_info or {}
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.get("headless", True))
            context = await browser.new_context(
                viewport={
                    "width": config["viewport_width"],
                    "height": config["viewport_height"],
                },
                color_scheme=config.get("color_scheme", "light"),
                **self._context_auth_kwargs(auth, url),
            )
            await install_ssrf_guard(context, **self._credential_scope(auth, url))
            page = await context.new_page()

            if self._signs_in(auth):
                if progress_callback:
                    await progress_callback(0.35, "authenticating")
                await self._sign_in(page, auth)

            if progress_callback:
                await progress_callback(0.4, "loading_url")
            await page.goto(url, wait_until="networkidle", timeout=_GOTO_TIMEOUT_MS)
            await self._settle_after_load(page, config)
            await self._assert_signed_in(page, auth, url)

            if progress_callback:
                await progress_callback(0.5, "capturing")
            shots = await self._walk_and_capture(page, steps, base_url=url, base_viewport="desktop")

            await context.close()
            await browser.close()
        return shots

    @staticmethod
    async def _goto_with_retry(page: Any, url: str) -> bool:
        """Navigate to ``url``, retrying with progressively more lenient readiness.

        Returns ``True`` iff a navigation completed. Tries ``networkidle`` first
        (best for static pages), then ``load``, then ``domcontentloaded`` — so an
        SPA whose connections never go idle still lands instead of soft-failing and
        leaving the page on the previous URL. A short backoff absorbs transient
        errors. Callers treat ``False`` as "do not screenshot" so a failed
        navigation never yields a stale duplicate of the prior screen.
        """
        last: Exception | None = None
        for attempt, wait_until in enumerate(_GOTO_WAIT_STRATEGIES):
            try:
                await page.goto(url, wait_until=wait_until, timeout=_GOTO_TIMEOUT_MS)
                if attempt:
                    logger.info("goto %r recovered on retry (%s)", url, wait_until)
                return True
            except (PlaywrightError, OSError, ValueError) as exc:
                last = exc
                logger.warning(
                    "goto %r attempt %d/%d (%s) failed: %s",
                    url,
                    attempt + 1,
                    len(_GOTO_WAIT_STRATEGIES),
                    wait_until,
                    exc,
                )
                await asyncio.sleep(_GOTO_RETRY_BACKOFF_S * (attempt + 1))
        logger.error("goto %r failed after %d attempts: %s", url, len(_GOTO_WAIT_STRATEGIES), last)
        return False

    @classmethod
    async def _apply_step_setup(
        cls, page: Any, step: dict[str, Any], state: dict[str, Any]
    ) -> bool:
        """Page setup that must precede the reveal: viewport resize, then goto.

        Both are idempotent against ``state`` (the running ``{url, viewport}``):
        a viewport equal to the last-applied one or a goto equal to the current
        URL is a no-op, so the first shot whose URL is already loaded does not
        re-navigate. Kept separate from :meth:`_apply_step_interaction` so the
        clip recorder can run it BEFORE the clip's start mark (the navigation
        must not land in-frame).

        Returns ``False`` iff a goto was requested but failed every retry — the
        caller then SKIPS this step's screenshot rather than capture a stale
        duplicate of the page still on screen. Viewport failures are non-fatal.
        """
        viewport = step.get("viewport")
        if viewport and viewport != state.get("viewport"):
            dims = _STEP_VIEWPORTS.get(str(viewport))
            if dims:
                try:
                    await page.set_viewport_size({"width": dims[0], "height": dims[1]})
                    state["viewport"] = viewport
                except (PlaywrightError, OSError, ValueError) as exc:
                    logger.warning("viewport %r resize failed: %s", viewport, exc)
        goto = step.get("goto")
        if goto and str(goto) != state.get("url"):
            if await cls._goto_with_retry(page, str(goto)):
                state["url"] = str(goto)
            else:
                return False
        return True

    @staticmethod
    async def _apply_step_interaction(
        page: Any, step: dict[str, Any], idx: int, capture_scale: float | None = None
    ) -> dict[str, Any] | None:
        """The reveal: click (advance the SPA), then scroll_to, then wait_for.

        Mirrors the legacy soft-fail click and adds the new modifiers in a fixed
        order. ``scroll_to`` accepts a selector (scrolled into view) or the
        literal ``"bottom"``; ``wait_for`` awaits a content selector before the
        shot. Each modifier soft-fails independently.

        When ``capture_scale`` is not ``None`` (clip mode), the clicked element's
        geometry is measured AFTER the reveal (click → scroll_to → wait_for) and
        returned as a focus/cursor record — the C15 track sidecar's per-clip
        keyframe. Screenshot mode passes ``None`` and gets ``None`` back, so its
        behaviour (and byte output) is unchanged.
        """
        action = str(step.get("action") or "click")
        locator: Any = None
        try:
            if action == "click":
                selector = step.get("selector")
                if selector:
                    locator = page.locator(str(selector)).first
                    await locator.click(timeout=8000)
            # action == "wait": just the settle handled by the caller
        except (PlaywrightError, OSError, ValueError) as exc:
            logger.warning("capture step %d (%s) failed: %s", idx, action, exc)
        scroll_to = step.get("scroll_to")
        if scroll_to:
            try:
                if str(scroll_to).lower() == "bottom":
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                else:
                    await page.locator(str(scroll_to)).first.scroll_into_view_if_needed(
                        timeout=8000
                    )
            except (PlaywrightError, OSError, ValueError) as exc:
                logger.warning("scroll_to %r failed: %s", scroll_to, exc)
        wait_for = step.get("wait_for")
        if wait_for:
            try:
                await page.wait_for_selector(str(wait_for), timeout=_STEP_WAIT_FOR_TIMEOUT_MS)
            except (PlaywrightError, OSError, ValueError) as exc:
                logger.warning("wait_for %r timed out; continuing: %s", wait_for, exc)
        if capture_scale is not None and locator is not None:
            return await UrlRecorder._focus_record(locator, action, capture_scale)
        return None

    @staticmethod
    async def _focus_record(
        locator: Any, action: str, capture_scale: float
    ) -> dict[str, Any] | None:
        """Measure the target element's post-reveal geometry for the C15 sidecar.

        Playwright's ``bounding_box`` returns ``{x, y, width, height}`` in CSS px
        (main-frame viewport space) or ``None`` when the element is hidden. We
        scale every coordinate by ``capture_scale`` so the bbox, the cursor
        keyframe (its centre) and the sidecar viewport all live in ONE
        frame-pixel space (the recorded video is ``dims x capture_scale``). A
        hidden element (``None``) yields no focus entry, so the clip gets no
        sidecar.
        """
        try:
            box = await locator.bounding_box()
        except (PlaywrightError, OSError, ValueError) as exc:
            logger.warning("bounding_box failed; no focus entry: %s", exc)
            return None
        if not box:
            return None
        s = capture_scale
        x = float(box["x"]) * s
        y = float(box["y"]) * s
        w = float(box["width"]) * s
        h = float(box["height"]) * s
        act = action if action in ("click", "type", "reveal") else "click"
        return {
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "action": act,
            "cx": x + w / 2.0,
            "cy": y + h / 2.0,
        }

    @classmethod
    async def _walk_and_capture(
        cls,
        page: Any,
        steps: list[dict[str, Any]],
        base_url: str = "",
        base_viewport: str | None = None,
    ) -> list[tuple[str, bytes]]:
        """Walk the step list; screenshot after each step that has a capture_id.

        Per step: setup (viewport, goto) → interaction (click, scroll_to,
        wait_for) → settle → screenshot. ``base_url``/``base_viewport`` seed the
        idempotency state so a goto/viewport that matches what
        :meth:`_capture_with_playwright` already established is skipped. A step
        without a ``capture_id`` only advances the SPA (e.g. a persona switch
        that reveals nav that is ``display:none`` until clicked). Duplicate ids
        are skipped.
        """
        shots = CapturedScreens()
        seen: set[str] = set()
        state: dict[str, Any] = {"url": base_url, "viewport": base_viewport}
        for i, step in enumerate(steps):
            setup_ok = await cls._apply_step_setup(page, step, state)
            await cls._apply_step_interaction(page, step, i)
            settle_ms = step.get("settle_ms")
            settle = (float(settle_ms) if settle_ms is not None else 800.0) / 1000.0
            if settle > 0:
                await asyncio.sleep(settle)
            capture_id = step.get("capture_id")
            if not capture_id:
                continue
            cid = str(capture_id)
            if cid in seen:
                logger.warning("duplicate capture_id %r; skipping screenshot", cid)
                continue
            if not setup_ok:
                # Navigation failed every retry — skip rather than capture a stale
                # duplicate of the previous screen (the render falls back to the
                # line's slide, which beats showing the wrong page).
                logger.warning("nav failed for capture_id %r; skipping screenshot", cid)
                continue
            try:
                png = await page.screenshot()
                shots.append((cid, png))
                content = await cls._page_content(page)
                if content:
                    shots.page_contents[cid] = content
                seen.add(cid)
            except (PlaywrightError, OSError, ValueError) as exc:
                logger.warning("screenshot for capture_id %r failed: %s", cid, exc)
        return shots

    @staticmethod
    async def _page_content(page: Page) -> dict[str, str]:
        """Capture visible text and inert markup; never serialize form credentials."""
        try:
            content = await page.evaluate("""() => {
                const root = document.querySelector('main') || document.body;
                const copy = root.cloneNode(true);
                copy.querySelectorAll('script,style,link,meta,iframe,object,embed,input,textarea,select,[hidden],[aria-hidden="true"]').forEach(el => el.remove());
                for (const el of [copy, ...copy.querySelectorAll('*')]) {
                    for (const attr of [...el.attributes]) {
                        if (!['id','class','role','aria-label'].includes(attr.name)) el.removeAttribute(attr.name);
                    }
                }
                return {text: root.innerText.slice(0, 50000), html: copy.outerHTML.slice(0, 150000)};
            }""")
            if isinstance(content, dict) and all(
                isinstance(content.get(k), str) for k in ("text", "html")
            ):
                return {"text": content["text"][:50000], "html": content["html"][:150000]}
        except PlaywrightError, OSError, ValueError:
            logger.warning("Could not read captured page content")
        return {}

    @staticmethod
    def _zip_screens(shots: list[tuple[str, bytes]], staging: Path) -> Path:
        """Pack captured PNGs into one self-describing zip (entry = ``<id>.png``)."""
        out_dir = Path(tempfile.mkdtemp(prefix="zip_", dir=staging))
        zip_path = out_dir / "screens.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for cid, png in shots:
                zf.writestr(f"{cid}.png", png)
                content = getattr(shots, "page_contents", {}).get(cid)
                if content:
                    zf.writestr(f"{cid}.page.json", json.dumps(content))
        return zip_path

    async def capture_screen_clips(
        self,
        job_id: str,
        url: str,
        steps: list[dict[str, Any]],
        auth_info: dict[str, Any] | None = None,
        recording_config: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Capture, in ONE navigation pass, a short interaction *clip* per screen
        plus a still — recording the click that reveals each screen.

        Mirrors :meth:`capture_screens` but records one continuous WebM while
        walking ``steps``; for every capture step it remembers ``(capture_id,
        t_start, t_end)`` (``t_start`` just before the click so the click lands
        in-frame, ``t_end`` after ``clip_dwell_ms``) AND takes a screenshot.
        After the context closes, one short MP4 per screen is trimmed from the
        WebM. The zip holds BOTH ``<id>.mp4`` (the reveal, with motion) and
        ``<id>.png`` (the steady still) — self-describing, like
        :meth:`capture_screens`, so promo render keys the reveal line to the
        clip and held lines to the still. Result shape matches
        :meth:`capture_screens` (``video_path`` is the zip), so the
        recorder-service success/storage path is reused unchanged.
        """
        resolved = self._resolve_config(recording_config)
        dwell_ms = float((recording_config or {}).get("clip_dwell_ms") or _DEFAULT_CLIP_DWELL_MS)
        start = time.time()
        # One per-job staging root holds the continuous webm, the trimmed per-screen
        # clips and the final zip; recorder_service rmtrees ``staging_dir`` once stored.
        staging = Path(tempfile.mkdtemp(prefix=f"screen_clips_{job_id}_"))
        try:
            if progress_callback:
                await progress_callback(0.3, "launching_browser")

            webm, boundaries, stills, tracks = await self._capture_clips_with_playwright(
                job_id, url, auth_info, resolved, steps, dwell_ms, progress_callback, staging
            )
            if not boundaries and not stills:
                return {
                    "success": False,
                    "error": "no screens captured (all selectors missed or screenshots failed)",
                    "recording_time_seconds": time.time() - start,
                    "staging_dir": str(staging),
                }

            if progress_callback:
                await progress_callback(0.8, "packaging")
            crf = get_quality_preset(resolved["quality_preset"]).crf
            clips = await asyncio.to_thread(self._trim_clips, webm, boundaries, crf, staging)
            # ``_trim_clips`` skips non-positive windows and swallows per-clip
            # trim errors, so EVERY trim can fail and it still returns []. The
            # guard above only covers "nothing captured at all", so a stills-only
            # zip used to be reported as success with a full ``screen_ids`` list
            # and zero usable clips — the caller has no way to notice.
            clip_failure = self._clip_capture_failure(boundaries, clips, stills)
            if clip_failure is not None:
                return {
                    "success": False,
                    "error": clip_failure,
                    "recording_time_seconds": time.time() - start,
                    "staging_dir": str(staging),
                }
            zip_path = await asyncio.to_thread(self._zip_clips, clips, stills, tracks, staging)
            elapsed = time.time() - start

            if progress_callback:
                await progress_callback(1.0, "completed")

            screen_ids = [cid for cid, _ in stills] or [cid for cid, _, _ in boundaries]
            return {
                "success": True,
                "video_path": str(zip_path),
                # ``video_path`` is a ZIP here, not a video (the key name is
                # shared so the recorder-service success path is reused). Say so
                # explicitly: the success path verifies each kind differently and
                # ffprobe cannot read an archive.
                "artifact_kind": ARTIFACT_KIND_SCREENS_ZIP,
                "duration_seconds": 0.0,
                "file_size_bytes": zip_path.stat().st_size,
                "width": resolved["viewport_width"],
                "height": resolved["viewport_height"],
                "has_audio": False,
                "screen_ids": screen_ids,
                "recording_time_seconds": elapsed,
                "staging_dir": str(staging),
            }
        except (OSError, RuntimeError, ValueError, TimeoutError, PlaywrightError) as exc:
            logger.error("Screen clip capture failed for job %s: %s", job_id, exc)
            return {
                "success": False,
                "error": str(exc),
                "recording_time_seconds": time.time() - start,
                "staging_dir": str(staging),
            }
        except BaseException:
            # See capture_screens: a sign-in failure must propagate to
            # RecorderService, and nobody else can free this staging dir.
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _clip_record_dims(
        steps: list[dict[str, Any]], default_w: int, default_h: int
    ) -> tuple[str | None, int, int]:
        """Resolve the clip recording canvas (uniform_viewport, width, height).

        Playwright fixes ``record_video_size`` at context creation while a per-step
        ``viewport`` only resizes the page, so a clip recorded at a different size
        than the page renders the page into a corner with a gray surround. When
        every capturing step shares ONE named viewport, record at that viewport's
        dims; otherwise (mixed or absent) keep the config default.
        """
        cap_viewports = {
            str(s["viewport"]) for s in steps if s.get("capture_id") and s.get("viewport")
        }
        uniform = next(iter(cap_viewports)) if len(cap_viewports) == 1 else None
        dims = _STEP_VIEWPORTS.get(uniform) if uniform else None
        w, h = dims or (default_w, default_h)
        return uniform, w, h

    async def _capture_clips_with_playwright(
        self,
        job_id: str,
        url: str,
        auth_info: dict[str, Any] | None,
        config: dict[str, Any],
        steps: list[dict[str, Any]],
        dwell_ms: float,
        progress_callback: ProgressCallback | None,
        staging: Path,
    ) -> tuple[
        Path,
        list[tuple[str, float, float]],
        list[tuple[str, bytes]],
        dict[str, dict[str, Any]],
    ]:
        """Record one continuous WebM while walking the steps; return the WebM
        path plus per-screen ``(id, t_start, t_end)`` clip boundaries, stills and
        the C15 focus/cursor tracks keyed by ``capture_id``.

        Mirrors :meth:`_record_with_playwright` (continuous ``record_video_dir``)
        but drives :meth:`_walk_and_record_clips` instead of the auto-scroll.

        ``capture_scale`` (default 1.0) records at a higher device-scale-factor
        for zoom headroom: the video pixel dims become ``dims x capture_scale``
        and the track sidecar's bbox/cursor coords + viewport are scaled to the
        same frame-pixel space. At 1.0 the context args are byte-identical to the
        un-scaled recording (``device_scale_factor`` is only passed when > 1.0).
        """
        from playwright.async_api import async_playwright

        video_dir = Path(tempfile.mkdtemp(prefix="webm_", dir=staging))
        auth = auth_info or {}
        # ``record_video_size`` is FIXED at context creation, but a per-step
        # ``viewport`` only resizes the page — so a uniform non-default viewport
        # (e.g. an all-mobile portrait shotlist) would render the page into a
        # corner of a differently-sized clip canvas, leaving a gray surround. Size
        # the clip canvas to a uniform per-step viewport when there is one.
        uniform_viewport, rec_w, rec_h = self._clip_record_dims(
            steps, config["viewport_width"], config["viewport_height"]
        )
        scale = float(config.get("capture_scale") or 1.0)
        context_kwargs: dict[str, Any] = {
            "viewport": {"width": rec_w, "height": rec_h},
            "record_video_dir": str(video_dir),
            "record_video_size": {"width": int(rec_w * scale), "height": int(rec_h * scale)},
            "color_scheme": config.get("color_scheme", "light"),
            **self._context_auth_kwargs(auth, url),
        }
        # Only pass device_scale_factor when scaling up — omitting it at 1.0 keeps
        # the recorded bytes identical to the pre-feature capture.
        if scale != 1.0:
            context_kwargs["device_scale_factor"] = scale
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.get("headless", True))
            context = await browser.new_context(**context_kwargs)
            await install_ssrf_guard(context, **self._credential_scope(auth, url))
            # Playwright's video clock starts at context creation; remember the
            # instant so clip boundaries can be expressed on the WebM timeline.
            rec_start = time.monotonic()
            page = await context.new_page()

            if self._signs_in(auth):
                if progress_callback:
                    await progress_callback(0.35, "authenticating")
                await self._sign_in(page, auth)

            if progress_callback:
                await progress_callback(0.4, "loading_url")
            await page.goto(url, wait_until="networkidle", timeout=_GOTO_TIMEOUT_MS)
            await self._settle_after_load(page, config)
            await self._assert_signed_in(page, auth, url)

            if progress_callback:
                await progress_callback(0.5, "capturing")
            boundaries, stills, tracks = await self._walk_and_record_clips(
                page,
                steps,
                rec_start,
                dwell_ms,
                base_url=url,
                base_viewport=uniform_viewport or "desktop",
                capture_scale=scale,
                viewport_w=rec_w,
                viewport_h=rec_h,
            )

            await context.close()
            await browser.close()

        video_files = list(video_dir.glob("*.webm"))
        if not video_files:
            msg = f"No video file produced in {video_dir}"
            raise RuntimeError(msg)
        webm = max(video_files, key=lambda f: f.stat().st_mtime)
        return webm, boundaries, stills, tracks

    @classmethod
    async def _walk_and_record_clips(
        cls,
        page: Any,
        steps: list[dict[str, Any]],
        rec_start: float,
        dwell_ms: float,
        base_url: str = "",
        base_viewport: str | None = None,
        capture_scale: float = 1.0,
        viewport_w: int = 0,
        viewport_h: int = 0,
    ) -> tuple[
        list[tuple[str, float, float]],
        list[tuple[str, bytes]],
        dict[str, dict[str, Any]],
    ]:
        """Walk the step list on the continuous-recording timeline.

        Mirrors :meth:`_walk_and_capture` (same step interpreter) but, for each
        step that carries a ``capture_id``, records the clip window — ``t_start``
        a small pre-roll BEFORE the interaction (so the click that reveals the
        screen is in-frame), ``t_end`` after ``dwell_ms`` — and takes a still.
        Setup (viewport/goto) runs BEFORE ``t_start`` so the navigation never
        lands in the clip. Offsets are relative to ``rec_start`` (the WebM
        clock). Steps without a ``capture_id`` only advance the SPA (e.g. a
        persona switch). Duplicate ids are skipped.

        Also returns the C15 focus/cursor track keyed by ``capture_id``. The
        clicked element's scaled geometry becomes a single clip-local keyframe
        (``t = _CLIP_PREROLL_S``, the moment the reveal lands in-frame). Clips
        whose step has no clickable target (``wait`` steps, hidden elements)
        get an empty ``focus`` list, so :meth:`_zip_clips` writes no sidecar.
        """
        boundaries: list[tuple[str, float, float]] = []
        stills = CapturedScreens()
        tracks: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        state: dict[str, Any] = {"url": base_url, "viewport": base_viewport}
        dwell = max(0.0, dwell_ms / 1000.0)
        viewport = {
            "w": int(viewport_w * capture_scale),
            "h": int(viewport_h * capture_scale),
        }
        for i, step in enumerate(steps):
            capture_id = step.get("capture_id")
            cid = str(capture_id) if capture_id else None
            # Setup (resize/navigate) happens before the clip start so it is not
            # recorded in-frame.
            setup_ok = await cls._apply_step_setup(page, step, state)
            # Mark the clip start before the interaction so the reveal is in-frame.
            t_start = max(0.0, time.monotonic() - rec_start - _CLIP_PREROLL_S) if cid else 0.0
            focus = await cls._apply_step_interaction(page, step, i, capture_scale)

            if not cid:
                settle_ms = step.get("settle_ms")
                settle = (float(settle_ms) if settle_ms is not None else 800.0) / 1000.0
                if settle > 0:
                    await asyncio.sleep(settle)
                continue
            if cid in seen:
                logger.warning("duplicate capture_id %r; skipping clip", cid)
                continue
            if not setup_ok:
                # Nav failed every retry — skip the clip+still so the render falls
                # back to the line's slide instead of a stale duplicate screen.
                logger.warning("nav failed for capture_id %r; skipping clip", cid)
                continue
            if dwell > 0:
                await asyncio.sleep(dwell)
            t_end = time.monotonic() - rec_start
            try:
                png = await page.screenshot()
            except (PlaywrightError, OSError, ValueError) as exc:
                logger.warning("screenshot for capture_id %r failed: %s", cid, exc)
                continue
            boundaries.append((cid, t_start, t_end))
            stills.append((cid, png))
            content = await cls._page_content(page)
            if content:
                stills.page_contents[cid] = content
            # The reveal lands one pre-roll into the clip (t_start marks a
            # pre-roll BEFORE the click), so the keyframe's clip-local t is the
            # pre-roll. Empty focus ⇒ no sidecar for this clip.
            track: dict[str, Any] = {"focus": [], "cursor": [], "viewport": viewport}
            if focus is not None:
                track["focus"].append(
                    {"t": _CLIP_PREROLL_S, "bbox": focus["bbox"], "action": focus["action"]}
                )
                track["cursor"].append(
                    {"t": _CLIP_PREROLL_S, "x": focus["cx"], "y": focus["cy"], "click": True}
                )
            tracks[cid] = track
            seen.add(cid)
        return boundaries, stills, tracks

    @staticmethod
    def _clip_capture_failure(
        boundaries: list[tuple[str, float, float]],
        clips: list[tuple[str, Path]],
        stills: list[tuple[str, bytes]],
    ) -> str | None:
        """Return an error string when the trim phase produced nothing usable.

        ``None`` means the result is shippable. A capture that recorded no
        boundaries at all is *not* a clip failure (the ``not boundaries and not
        stills`` guard upstream already covers the empty case), but a capture
        that recorded boundaries and trimmed zero clips is.
        """
        if not boundaries:
            return None
        if clips:
            return None
        return (
            f"clip trimming produced 0 usable clips from {len(boundaries)} recorded "
            f"window(s) ({len(stills)} still(s) captured); the recording holds no motion"
        )

    @staticmethod
    def _trim_clips(
        webm: Path,
        boundaries: list[tuple[str, float, float]],
        crf: int,
        staging: Path,
    ) -> list[tuple[str, Path]]:
        """Trim one short H.264 MP4 per screen from the continuous WebM, using
        the recorded ``(id, t_start, t_end)`` windows (frame-accurate re-encode)."""
        out_dir = Path(tempfile.mkdtemp(prefix="clips_", dir=staging))
        clips: list[tuple[str, Path]] = []
        for cid, t_start, t_end in boundaries:
            duration = t_end - t_start
            if duration <= 0:
                logger.warning("clip %r has non-positive duration %.3f; skipping", cid, duration)
                continue
            out = out_dir / f"{cid}.mp4"
            try:
                FFmpegOps.trim_reencode(webm, out, start_s=t_start, duration_s=duration, crf=crf)
                clips.append((cid, out))
            except (OSError, ValueError, RuntimeError) as exc:
                logger.warning("trim for clip %r failed: %s", cid, exc)
        return clips

    @staticmethod
    def _zip_clips(
        clips: list[tuple[str, Path]],
        stills: list[tuple[str, bytes]],
        tracks: dict[str, dict[str, Any]] | None,
        staging: Path,
    ) -> Path:
        """Pack per-screen clips (``<id>.mp4``), stills (``<id>.png``) and the C15
        focus/cursor track sidecars (``<id>.track.json``) into one self-describing
        zip — same convention as :meth:`_zip_screens`.

        A sidecar is written ONLY for a clip whose track has a non-empty
        ``focus`` list, so clips with no clickable target (no reveal to zoom to)
        add nothing and the zip stays byte-identical to the pre-feature output.
        """
        out_dir = Path(tempfile.mkdtemp(prefix="zip_", dir=staging))
        zip_path = out_dir / "screens.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for cid, png in stills:
                zf.writestr(f"{cid}.png", png)
                content = getattr(stills, "page_contents", {}).get(cid)
                if content:
                    zf.writestr(f"{cid}.page.json", json.dumps(content))
            for cid, clip_path in clips:
                zf.write(clip_path, f"{cid}.mp4")
            for cid, track in (tracks or {}).items():
                if track.get("focus"):
                    zf.writestr(f"{cid}.track.json", json.dumps(track))
        return zip_path

    async def _record_with_playwright(
        self,
        job_id: str,
        url: str,
        auth_info: dict[str, Any] | None,
        config: dict[str, Any],
        scroll: dict[str, Any],
        progress_callback: ProgressCallback | None,
        staging: Path,
    ) -> Path:
        """Launch Playwright, optionally authenticate, navigate, scroll, capture."""
        from playwright.async_api import async_playwright

        video_dir = Path(tempfile.mkdtemp(prefix="webm_", dir=staging))
        auth = auth_info or {}

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=config.get("headless", True))
            context = await browser.new_context(
                viewport={
                    "width": config["viewport_width"],
                    "height": config["viewport_height"],
                },
                record_video_dir=str(video_dir),
                record_video_size={
                    "width": config["viewport_width"],
                    "height": config["viewport_height"],
                },
                color_scheme=config.get("color_scheme", "light"),
                **self._context_auth_kwargs(auth, url),
            )
            await install_ssrf_guard(context, **self._credential_scope(auth, url))

            # Playwright records video for the whole context lifetime, so the
            # clock starts here. Remember this instant to trim the blank
            # page-load lead-in (login + navigation + settle) later.
            rec_start = time.monotonic()

            page = await context.new_page()

            if self._signs_in(auth):
                if progress_callback:
                    await progress_callback(0.35, "authenticating")
                await self._sign_in(page, auth)

            if progress_callback:
                await progress_callback(0.4, "loading_url")

            await page.goto(url, wait_until="networkidle", timeout=_GOTO_TIMEOUT_MS)

            # Data-driven SPAs render their main content *after* networkidle, so
            # settle before recording the scroll — otherwise the capture is a
            # blank shell (the white-screen bug).
            await self._settle_after_load(page, config)
            await self._assert_signed_in(page, auth, url)

            # Everything recorded up to now (login + navigation + settle) is
            # blank/loading; stash the lead so _post_process can trim it off the
            # front of the clip. Shares the per-call `config` dict with
            # _post_process (no signature change to record_url).
            config["_lead_seconds"] = max(0.0, time.monotonic() - rec_start)

            # A scripted demo walkthrough (clicks through the SPA) REPLACES the
            # auto-scroll when present: the steps define the recorded clip.
            steps = config.get("interaction_steps")
            if steps:
                if progress_callback:
                    await progress_callback(0.5, "interacting")
                await self._run_interactions(page, steps)
            else:
                if progress_callback:
                    await progress_callback(0.5, "scrolling")
                await self._auto_scroll(page, scroll)

            if progress_callback:
                await progress_callback(0.7, "finalizing_video")

            # Close context to finalize the video
            await context.close()
            await browser.close()

        video_files = list(video_dir.glob("*.webm"))
        if not video_files:
            msg = f"No video file produced in {video_dir}"
            raise RuntimeError(msg)

        return max(video_files, key=lambda f: f.stat().st_mtime)

    @classmethod
    async def _settle_after_load(cls, page: Any, config: dict[str, Any]) -> None:
        """Wait for real content after navigation, before recording the scroll.

        Two best-effort knobs (both optional, no-op when unset):
        - ``wait_for_selector``: wait until a content element appears (10s cap).
        - ``post_load_delay_ms``: a fixed settle delay for SPAs that hydrate
          their main pane via XHR *after* the network goes idle (clamped to
          ``_MAX_POST_LOAD_DELAY_MS`` so a runaway value can't hang the job).

        After settling, verify the frame isn't blank: a slow-hydrating SPA can
        leave a white/empty shell that would otherwise be recorded silently
        (the white-screen bug). On a blank frame we wait one bounded extra delay
        and re-check, logging loudly either way — never raising, so a genuinely
        blank page still records but is surfaced.
        """
        selector = config.get("wait_for_selector")
        if selector:
            try:
                await page.wait_for_selector(selector, timeout=10000)
            except PlaywrightError:
                logger.warning("wait_for_selector %r timed out; continuing with delay", selector)

        delay_ms = config.get("post_load_delay_ms")
        if delay_ms:
            delay_ms = float(delay_ms)
            if delay_ms > _MAX_POST_LOAD_DELAY_MS:
                logger.warning(
                    "post_load_delay_ms %.0f exceeds cap %d; clamping",
                    delay_ms,
                    _MAX_POST_LOAD_DELAY_MS,
                )
                delay_ms = float(_MAX_POST_LOAD_DELAY_MS)
            await asyncio.sleep(delay_ms / 1000.0)

        if await cls._frame_is_blank(page):
            recheck_ms = min(float(delay_ms or 2000), float(_BLANK_RECHECK_MAX_MS))
            logger.warning(
                "settled frame looks blank; waiting %.0fms and re-checking before capture",
                recheck_ms,
            )
            await asyncio.sleep(recheck_ms / 1000.0)
            if await cls._frame_is_blank(page):
                logger.warning("frame still blank after re-check; recording may be empty")

    @staticmethod
    async def _frame_is_blank(page: Any) -> bool:
        """Best-effort: True if the current viewport is near-uniform (blank).

        Screenshot failures degrade to "not blank" so a flaky screenshot never
        blocks a recording.
        """
        try:
            shot = await page.screenshot()
            with Image.open(io.BytesIO(shot)) as raw:
                luma = raw.convert("L")
            # Single-band "L" image ⇒ (min, max); guard the multiband typing.
            extrema = luma.getextrema()
            if not (isinstance(extrema, tuple) and len(extrema) == 2):
                return False
            lo, hi = extrema
            if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float))):
                return False
            return (hi - lo) < _BLANK_FRAME_LUMA_RANGE
        except (PlaywrightError, OSError, ValueError) as exc:
            logger.debug("blank-frame check skipped: %s", exc)
            return False

    @staticmethod
    def _bound_origin(url: str) -> str:
        """The ONE origin a recording's login is bound to: the recorded URL's."""
        parts = urlparse(url)
        return f"{parts.scheme}://{parts.netloc}"

    @staticmethod
    def _context_auth_kwargs(auth: dict[str, Any], url: str) -> dict[str, Any]:
        """Basic auth as ``http_credentials`` restricted to the bound origin.

        Without ``origin`` Chromium answers a Basic challenge from ANY origin that
        returns 401 — a third-party script or a redirect target harvests the
        password. A bearer token is deliberately NOT set here: context
        ``extra_http_headers`` ride on every request the page makes, every
        origin; it is attached per hop by the SSRF guard instead
        (:meth:`_credential_scope`).

        A login with a ``login_url`` is a FORM sign-in (often single sign-on: the
        form is on an identity provider, e.g. Keycloak) and NEVER becomes
        ``http_credentials``, whatever its ``type`` says. The Brief stores a
        username + password sign-in as ``type: "basic"``, so without this a 401
        Basic challenge from the RECORDED site would have been answered with the
        password the tenant entered for the identity provider. Genuine HTTP Basic
        (no sign-in page) is answered only for the recorded site's origin."""
        kwargs: dict[str, Any] = {}
        if auth.get("login_url"):
            return kwargs
        if auth.get("type") == "basic" and auth.get("username"):
            kwargs["http_credentials"] = {
                "username": auth["username"],
                "password": auth.get("password") or "",
                "origin": UrlRecorder._bound_origin(url),
            }
        return kwargs

    @staticmethod
    def _cookie_header(auth: Mapping[str, object]) -> str:
        """The login's cookies as ONE ``Cookie`` header value ("" when none)."""
        jar = auth.get("cookies")
        entries = [
            cast("Mapping[str, object]", c)
            for c in (jar if isinstance(jar, list) else [])
            if isinstance(c, Mapping)
        ]
        pairs = [f"{c['name']}={c.get('value') or ''}" for c in entries if c.get("name")]
        if not pairs and auth.get("cookie_header"):
            pairs = [p.strip() for p in str(auth["cookie_header"]).split(";") if "=" in p]
        return "; ".join(pairs)

    @staticmethod
    def _credential_scope(auth: Mapping[str, object], url: str) -> CredentialScope:
        """``install_ssrf_guard`` kwargs: the login, attached only on the bound origin.

        Bearer AND cookies are sent as headers by the guard on hops to the bound
        ORIGIN only. Cookies are deliberately NOT put in the browser jar: RFC 6265
        jars ignore ports (and ``route.fetch`` re-adds jar cookies to every hop), so
        a jar cookie for 127.0.0.1:443 also reaches 127.0.0.1:<any port>. A caller's
        ``domain`` is ignored for the same reason. Basic carries no header here — it
        is answered by ``http_credentials`` restricted to the same origin."""
        origin = UrlRecorder._bound_origin(url)
        kind = auth.get("type")
        if kind == "bearer" and auth.get("token"):
            return {
                "credential_origin": origin,
                "credential_headers": {"Authorization": f"Bearer {auth['token']}"},
            }
        cookie = UrlRecorder._cookie_header(auth) if kind == "cookies" else ""
        if cookie:
            return {"credential_origin": origin, "credential_headers": {"Cookie": cookie}}
        if kind in ("basic", "cookies"):
            return {"credential_origin": origin}
        return {}

    @staticmethod
    def _signs_in(auth: Mapping[str, object]) -> bool:
        """True when this recording performs an interactive form sign-in."""
        return bool(auth.get("login_url") and auth.get("username"))

    @staticmethod
    def _verifies_sign_in(auth: Mapping[str, object]) -> bool:
        """True when this recording replays a credential a session must come from.

        EVERY auth method is verified: cookies and bearer tokens are replayed by
        the SSRF guard, ``basic`` answers an HTTP challenge, and ``login_url`` +
        username submits a form. A recording with NO credential at all
        (``type: "none"``, or a cookie login whose jar is empty) is deliberately
        not checked — it may be recording a public sign-in page on purpose.
        """
        if auth.get("login_url") and auth.get("username"):
            return True
        kind = auth.get("type")
        if kind == "cookies":
            return bool(auth.get("cookies") or auth.get("cookie_header"))
        if kind == "bearer":
            return bool(auth.get("token"))
        if kind == "basic":
            return bool(auth.get("username"))
        return False

    @staticmethod
    def _credential_noun(auth: Mapping[str, object]) -> str:
        """What the user supplied, phrased for the message they read."""
        kind = str(auth.get("type") or "")
        if kind == "cookies":
            return "saved cookie session"
        if kind == "bearer":
            return "saved access token"
        return "username or password"

    @staticmethod
    def _landed_on_sign_in_url(page_url: str, auth: Mapping[str, object], url: str) -> bool:
        """True when the browser is on the named sign-in page itself.

        Two shapes: the recorded site bounced us to a DIFFERENT origin (single
        sign-on — any page back on the identity provider means no session was
        established), or the app's own sign-in page (same origin as the recorded
        site) is showing, which is exactly ``login_url``'s path.
        """
        login_url = auth.get("login_url")
        if not login_url:
            return False
        login = urlparse(str(login_url))
        here = urlparse(page_url)
        login_host = login.netloc.lower()
        if login_host != here.netloc.lower():
            return False
        if login_host != urlparse(url).netloc.lower():
            return True
        login_path = (login.path or "").rstrip("/").lower()
        return bool(login_path) and login_path == (here.path or "").rstrip("/").lower()

    @staticmethod
    async def _sign_in_snapshot(page: Page, auth: Mapping[str, object]) -> Mapping[str, Any]:
        """Read what the page can see. A probe that cannot answer FAILS the sign-in.

        ``page.evaluate`` raises when the page navigated out from under the probe
        ("execution context destroyed") — exactly what an app bouncing an
        unauthenticated session to its sign-in page looks like. Reading that as
        signed-in reproduces the defect this check exists to catch, so an
        uninspectable page is a failed sign-in. A probe that returns anything
        other than a snapshot cannot prove the sign-in took effect either.
        """
        selector = str(auth.get("password_selector") or _SIGNED_OUT_PASSWORD_SELECTOR)
        try:
            seen = await page.evaluate(_SIGN_IN_SNAPSHOT_JS, selector)
        except PlaywrightError as exc:
            raise SiteLoginFailedError(
                f"{SITE_LOGIN_FAILED}: the page could not be checked for a sign-in page "
                "— the session was not accepted"
            ) from exc
        if not isinstance(seen, Mapping):
            raise SiteLoginFailedError(
                f"{SITE_LOGIN_FAILED}: the page could not be checked for a sign-in page "
                "— the session was not accepted"
            )
        return seen

    @staticmethod
    def _sign_in_evidence(snapshot: Mapping[str, Any]) -> str | None:
        """Does this page look like a sign-in screen? ONE verdict, any auth method.

        ``"password"`` — a visible password field is showing;
        ``"wall"``     — a sign-in screen without one: an authentication action
                         plus sign-in wording in the heading/title, or a bare SSO
                         wall with nothing else on it.

        Deliberately conservative: a plain "Sign in" link in a content page's
        header has neither the SSO/provider wording nor a sign-in heading, so an
        ordinary page never trips it.
        """
        if snapshot.get("password"):
            return "password"
        controls = [str(control) for control in (snapshot.get("controls") or ())]
        sso_actions = [control for control in controls if _SSO_CONTROL_RE.search(control)]
        if not snapshot.get("auth_fields") and not sso_actions:
            return None
        # Each field on its own: joined, a heading ending "Sign in" plus a title
        # starting "Settings" would read as "Sign in Settings" and be rejected.
        if any(
            _SIGN_IN_HEADING_RE.search(str(text or ""))
            for text in (snapshot.get("heading"), snapshot.get("title"))
        ):
            return "wall"
        # A wall whose heading is just the product name still has nothing else on
        # it: the only actionable things are signing in and getting help.
        if sso_actions and len(controls) <= 2:
            return "wall"
        return None

    @classmethod
    def _sign_in_failure(cls, evidence: str, auth: Mapping[str, object]) -> str:
        """The user-facing reason; the ``site_login_failed`` code stays the prefix."""
        noun = cls._credential_noun(auth)
        if evidence == "password":
            return (
                f"{SITE_LOGIN_FAILED}: the page still asks for a password after signing in "
                f"— the {noun} was not accepted"
            )
        return (
            f"{SITE_LOGIN_FAILED}: the page is a sign-in screen, not the recorded site "
            f"— the {noun} was not accepted"
        )

    @classmethod
    async def _sign_in(cls, page: Page, auth: dict[str, Any]) -> None:
        """Submit the sign-in form; a form that cannot be filled is a FAILED sign-in.

        Recording on after a failed sign-in records the sign-in page and passes it
        off as footage, so this raises :class:`SiteLoginFailedError` instead. The
        message names what actually went wrong: a sign-in page that could not be
        OPENED is a reachability problem, not a missing form field."""
        host = urlparse(str(auth["login_url"])).hostname or "the sign-in page"
        try:
            await cls._form_login(page, auth)
        except _SignInPageUnreachableError as exc:
            raise SiteLoginFailedError(
                f"{SITE_LOGIN_FAILED}: the sign-in page on {host} could not be opened — "
                "the site did not respond, so the sign-in never happened"
            ) from exc
        except PlaywrightError as exc:
            raise SiteLoginFailedError(
                f"{SITE_LOGIN_FAILED}: the sign-in form on {host} could not be filled in "
                "(no username / password field or submit button found)"
            ) from exc

    @classmethod
    async def _assert_signed_in(cls, page: Page, auth: Mapping[str, object], url: str) -> None:
        """After reaching ``url``, the PAGE must not be a sign-in screen.

        ONE predicate for every auth method — form/SSO, cookies, bearer and
        ``basic`` alike. The verdict is read from the page that is about to be
        recorded, never from the auth METHOD: an expired cookie and a rejected
        password produce the same page, and recording that page is the defect
        this stops.
        """
        if not cls._verifies_sign_in(auth):
            return
        if cls._landed_on_sign_in_url(page.url, auth, url):
            raise SiteLoginFailedError(
                f"{SITE_LOGIN_FAILED}: still on the sign-in page after signing in — the "
                f"{cls._credential_noun(auth)} was not accepted"
            )
        snapshot = await cls._sign_in_snapshot(page, auth)
        evidence = cls._sign_in_evidence(snapshot)
        if evidence is not None:
            raise SiteLoginFailedError(cls._sign_in_failure(evidence, auth))

    @staticmethod
    async def _form_login(page: Any, auth: dict[str, Any]) -> None:
        """Submit an interactive login form before recording the target URL."""
        username_sel = auth.get("username_selector") or _DEFAULT_USERNAME_SELECTOR
        password_sel = auth.get("password_selector") or _DEFAULT_PASSWORD_SELECTOR
        submit_sel = auth.get("submit_selector") or _DEFAULT_SUBMIT_SELECTOR

        try:
            await page.goto(auth["login_url"], wait_until="networkidle", timeout=_GOTO_TIMEOUT_MS)
        except PlaywrightError as exc:
            # Mark the phase: the caller must not report this as a missing field.
            raise _SignInPageUnreachableError(str(exc)) from exc
        await page.fill(username_sel, auth["username"], timeout=_LOGIN_TIMEOUT_MS)
        if auth.get("password"):
            await page.fill(password_sel, auth["password"], timeout=_LOGIN_TIMEOUT_MS)
        await page.click(submit_sel, timeout=_LOGIN_TIMEOUT_MS)
        # Give the post-login redirect / SPA bootstrap time to settle.
        try:
            await page.wait_for_load_state("networkidle", timeout=_LOGIN_TIMEOUT_MS)
        except PlaywrightError:
            logger.warning("Post-login networkidle wait timed out; continuing")

    async def _auto_scroll(self, page: Any, scroll_config: dict[str, Any]) -> None:
        """Scroll the page from top to bottom at a configurable speed.

        Holds the final frame so the total scroll phase is at least
        ``min_clip_seconds`` — single-viewport pages scroll little or not at all
        and would otherwise produce a ~3s clip (start + end delay only).
        """
        start_delay = scroll_config.get("start_delay_seconds", _DEFAULT_START_DELAY)
        end_delay = scroll_config.get("end_delay_seconds", _DEFAULT_END_DELAY)
        speed = scroll_config.get("speed_px_per_second", _DEFAULT_SCROLL_SPEED)
        enabled = scroll_config.get("enabled", True)
        min_clip = scroll_config.get("min_clip_seconds", _DEFAULT_MIN_CLIP_SECONDS)

        phase_start = time.monotonic()

        if start_delay > 0:
            await asyncio.sleep(start_delay)

        if not enabled:
            await self._hold_until_min(phase_start, end_delay, min_clip)
            return

        step_px = 100
        interval = step_px / speed

        scroll_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = await page.evaluate("window.innerHeight")
        total_scroll = max(0, scroll_height - viewport_height)

        scrolled = 0
        while scrolled < total_scroll:
            step = min(step_px, total_scroll - scrolled)
            await page.evaluate(f"window.scrollBy(0, {step})")
            scrolled += step
            await asyncio.sleep(interval)

        await self._hold_until_min(phase_start, end_delay, min_clip)

    @staticmethod
    async def _hold_until_min(phase_start: float, end_delay: float, min_clip: float) -> None:
        """Sleep the end-hold, padded so the scroll phase reaches ``min_clip``."""
        elapsed = time.monotonic() - phase_start
        hold = max(end_delay, min_clip - elapsed)
        if hold > 0:
            await asyncio.sleep(hold)

    @staticmethod
    async def _run_interactions(page: Any, steps: list[dict[str, Any]]) -> None:
        """Click/scroll/wait through a scripted demo while recording.

        Each step soft-fails: a missing click target (or any Playwright error)
        is logged and skipped so one stale selector never empties the clip.
        ``hold_ms`` after each step keeps the resulting screen on-camera.
        """
        for i, step in enumerate(steps):
            action = str(step.get("action") or "click")
            try:
                if action == "click":
                    selector = step.get("selector")
                    if selector:
                        timeout = float(step.get("timeout_ms") or 8000)
                        await page.locator(str(selector)).first.click(timeout=timeout)
                elif action == "scroll":
                    px = int(step.get("pixels") or 600)
                    await page.mouse.wheel(0, px)
                # action == "wait": just the hold below
            except (PlaywrightError, OSError, ValueError) as exc:
                logger.warning("interaction step %d (%s) failed: %s", i, action, exc)
            hold_ms = step.get("hold_ms")
            hold = (float(hold_ms) if hold_ms is not None else 2500.0) / 1000.0
            if hold > 0:
                await asyncio.sleep(hold)

    async def _post_process(self, video_path: Path, config: dict[str, Any]) -> Path:
        """Convert WebM to MP4 if needed, trimming the blank page-load lead-in.

        The lead-in trim is opt-in: it only fires when ``post_load_delay_ms`` was
        supplied (the settle knob). ``_lead_seconds`` is the wall-clock from
        context creation to content-ready, stashed by _record_with_playwright; we
        trim slightly less than that (``_LEAD_TRIM_MARGIN_S``) so we never cut
        into the first rendered content frames.
        """
        trim_s = 0.0
        if config.get("post_load_delay_ms") is not None:
            lead = float(config.get("_lead_seconds") or 0.0)
            trim_s = max(0.0, lead - _LEAD_TRIM_MARGIN_S)
            if trim_s < _MIN_TRIM_S:
                trim_s = 0.0

        output_format = config.get("output_format", self._output_format)
        if output_format == "mp4" and video_path.suffix.lower() == ".webm":
            preset_name = config.get("quality_preset", self._default_quality_preset)
            preset = get_quality_preset(preset_name)
            mp4_output = video_path.with_suffix(".mp4")
            return await asyncio.to_thread(
                FFmpegOps.convert_to_mp4,
                video_path,
                mp4_output,
                crf=preset.crf,
                start_offset_s=trim_s,
            )

        # No format conversion: trim in place when there is a lead to drop.
        if trim_s > 0:
            probe = await asyncio.to_thread(FFmpegOps.probe_all, video_path)
            if probe.duration and probe.duration > trim_s + _MIN_TRIM_S:
                trimmed = video_path.with_name(f"{video_path.stem}-trim{video_path.suffix}")
                return await asyncio.to_thread(
                    FFmpegOps.trim, video_path, trimmed, trim_s, probe.duration
                )
        return video_path

    def _resolve_config(self, recording_config: dict[str, Any] | None) -> dict[str, Any]:
        """Merge defaults with optional overrides."""
        preset_name = self._default_quality_preset
        if recording_config and recording_config.get("quality_preset"):
            preset_name = recording_config["quality_preset"]

        preset = get_quality_preset(preset_name)
        config: dict[str, Any] = {
            "viewport_width": preset.viewport_width,
            "viewport_height": preset.viewport_height,
            "headless": self._headless,
            "output_format": self._output_format,
            "quality_preset": preset_name,
            "color_scheme": "light",
            "capture_scale": 1.0,
        }

        if recording_config:
            for key in (
                "viewport_width",
                "viewport_height",
                "headless",
                "output_format",
                "color_scheme",
                "post_load_delay_ms",
                "wait_for_selector",
                "interaction_steps",
                "capture_scale",
            ):
                if recording_config.get(key) is not None:
                    config[key] = recording_config[key]

        return config

    def _resolve_scroll_config(self, scroll_config: dict[str, Any] | None) -> dict[str, Any]:
        """Merge default scroll config with overrides."""
        defaults: dict[str, Any] = {
            "enabled": True,
            "speed_px_per_second": _DEFAULT_SCROLL_SPEED,
            "start_delay_seconds": _DEFAULT_START_DELAY,
            "end_delay_seconds": _DEFAULT_END_DELAY,
            "min_clip_seconds": _DEFAULT_MIN_CLIP_SECONDS,
        }
        if scroll_config:
            defaults.update({k: v for k, v in scroll_config.items() if v is not None})
        return defaults
