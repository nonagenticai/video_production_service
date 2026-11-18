"""Deterministic frame-step capture of branded motion-graphics templates.

Unlike :class:`ContentRecorder` (Playwright ``record_video_dir`` + auto-scroll,
which captures wall-clock real time and drops/jitters frames under load), this
recorder steps a timeline frame-by-frame:

1. load the template HTML (``window.PARAMS`` injected for brand/content),
2. for ``frame = 0 .. N`` at the requested fps, seek the page to
   ``t = frame * 1000/fps`` ms — pausing every ``Animation.currentTime`` (so CSS
   ``@keyframes`` advance deterministically) and calling ``window.render(t)`` when
   the template exposes it,
3. screenshot each frame, then assemble the PNG sequence with ffmpeg at the same
   fps.

The result is pixel-identical regardless of machine load — the property that
makes this a credible, on-brand alternative to generative (AI) video. The result
dict shape mirrors :class:`ContentRecorder` so ``RecorderService._handle_success``
consumes it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

from playwright.async_api import Error as PlaywrightError

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page

from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.services.ffmpeg import FFmpegOps
from src.services.ffmpeg.alpha_probe import require_vp9_alpha_support
from src.services.recording.quality_presets import get_quality_preset
from src.shared.security.playwright_guard import install_ssrf_guard

logger = logging.getLogger(__name__)


class FitOverflow(BaseModel):
    """One element whose text the template drew past its line budget."""

    lines: int
    max_lines: int


class RenderFlags(BaseModel):
    """What a recording noticed about its own output (render 01a1179d)."""

    fit_overflow: dict[str, FitOverflow] = {}


ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

_DEFAULT_FPS = 30
_MIN_FPS = 1
_MAX_FPS = 60
_DEFAULT_DURATION_MS = 5000.0
_MIN_DURATION_MS = 100.0
# Hard cap on frames per clip — a runaway duration/fps would otherwise fill disk
# and wedge the worker. 3600 frames = 2 min @ 30fps / 1 min @ 60fps.
_MAX_FRAMES = 3600
# Default output dimensions — the seed templates (social_card, pipeline_story)
# are 1080x1080; callers override for 16:9 in-video clips.
_DEFAULT_WIDTH = 1080
_DEFAULT_HEIGHT = 1080

# Supersampling factor for the capture. Chromium rasterises glyph edges and
# stroke antialiasing at the context's device_scale_factor, so a 1:1 capture of
# a type-led shot (kinetic_typography, lower_third, quote, data_viz, every
# animated_slide) bakes in aliased edges that no downstream filter can recover.
# Capturing at 2x and downscaling the LOSSLESS PNG sequence with lanczos before
# the encode resolves each output pixel from 4 rasterised samples — real detail,
# not an upscale. Cost is 4x the per-frame bitmap area: negligible in RAM (one
# RGBA surface, ~18MB at 2160x2160 vs the ~2.7Gi per-recording envelope the pod
# is sized for) but 4x on the temp PNG bytes, which is why the scaled sequence
# replaces the originals on disk rather than sitting beside them, and why
# _MAX_FRAMES stays where it is.
_CAPTURE_SCALE = 2
# Raster-area ceiling for the supersampled capture, in pixels. 2560x1440 is the
# largest *ordered* tier that still fits 2x supersampling inside a sane Chromium
# surface, so the budget is exactly its 2x area. A 4k order (3840x2160 -> 2x is
# 7680x4320, 33.2M px, ~133MB of RGBA per frame) therefore captures at 1x: at
# that output resolution a glyph already has the samples it needs, and pushing a
# 4x surface through a pod sized for three concurrent recordings is how the
# recorder gets OOM-killed instead of sharper.
_MAX_CAPTURE_RASTER_PIXELS = 2560 * 1440 * _CAPTURE_SCALE * _CAPTURE_SCALE
# Subdirectory the downscaled sequence is written into (a sibling of the raw
# frames inside the per-job staging root, so recorder_service's rmtree of
# ``staging_dir`` still reclaims everything).
_SCALED_DIRNAME = "scaled"

# Bound on how long a single teardown call (context.close() / browser.close())
# may take during _capture_frames's finally. Called on EVERY exit path,
# including a timeout-driven cancellation -- if Playwright's own close() hangs
# (a wedged Chromium process), an unbounded await here would hang the
# cancellation itself and defeat record_animation's whole execution timeout.
# Best-effort: a failure to close within this window is logged, never raised.
_TEARDOWN_TIMEOUT_S = 10.0

# Cadence of the optional DEBUG progress line inside the frame loop. The
# per-recording INFO summary (``animation_render_timing``) is the one meant for
# production logs; this exists only to watch a long capture unfold.
_TIMING_DEBUG_EVERY_FRAMES = 30

# Wall-time phases accumulated per recording (time.perf_counter), reported in
# ONE ``animation_render_timing`` INFO line. ``setup`` = Chromium launch +
# context/page + set_content + fonts; ``seek`` = the per-frame _STEP_JS
# evaluate; ``screenshot`` = the per-frame page.screenshot (PNG encode + write
# at the supersampled raster); ``downscale`` = the whole-sequence lanczos
# ffmpeg pass; ``encode`` = the final container encode; ``probe`` = ffprobe.
_TIMING_PHASES = ("setup", "seek", "screenshot", "downscale", "encode", "probe")


def _new_timings() -> dict[str, float]:
    timings: dict[str, float] = {f"{phase}_s": 0.0 for phase in _TIMING_PHASES}
    timings["frames"] = 0.0
    return timings


def _log_render_timing(
    job_id: str,
    timings: dict[str, float],
    *,
    capture_scale: float,
    fps: int,
    width: int,
    height: int,
    total_s: float,
    success: bool,
) -> None:
    """Emit the single per-recording timing summary (INFO)."""
    frames = int(timings.get("frames", 0))
    per_frame_ms = (
        (timings["seek_s"] + timings["screenshot_s"]) / frames * 1000.0 if frames else 0.0
    )
    logger.info(
        "animation_render_timing job_id=%s success=%s frames=%d fps=%d size=%sx%s "
        "capture_scale=%s total_s=%.2f setup_s=%.2f seek_s=%.2f screenshot_s=%.2f "
        "downscale_s=%.2f encode_s=%.2f probe_s=%.2f per_frame_ms=%.1f",
        job_id,
        success,
        frames,
        fps,
        width,
        height,
        capture_scale,
        total_s,
        timings["setup_s"],
        timings["seek_s"],
        timings["screenshot_s"],
        timings["downscale_s"],
        timings["encode_s"],
        timings["probe_s"],
        per_frame_ms,
    )


# Per-frame seek: pause + seek every running animation (deterministic CSS
# @keyframes), then drive the JS timeline if the template exposes render(t).
_STEP_JS = """(t) => {
  for (const a of document.getAnimations()) { try { a.pause(); a.currentTime = t; } catch (e) {} }
  if (typeof window.render === 'function') { window.render(t); }
}"""


def _inject_params(html: str, params: Any) -> str:
    """Inline ``window.PARAMS`` into the document so the template's inline script
    sees it as it executes.

    Playwright's ``add_init_script`` only fires on a real navigation; ``set_content``
    rewrites the document in place (no navigation), so an init script never runs
    and ``window.PARAMS`` stays undefined — every template then silently renders
    its built-in defaults. Inlining the params as the first ``<head>`` script
    (before the template's own ``<script>``) is the reliable path. ``</`` is
    escaped so a string param can't close the tag early.
    """
    if params is None:
        return html
    blob = json.dumps(params).replace("</", "<\\/")
    tag = f"<script>window.PARAMS={blob};</script>"
    if "<head>" in html:
        return html.replace("<head>", "<head>" + tag, 1)
    return tag + html


# Neutralises whatever backdrop the document paints itself. ``!important`` beats
# the templates' own rules regardless of specificity, ``background-image`` is
# cleared separately because clearing only the colour would leave a gradient
# painted over the now-transparent base, and the block is appended LAST in
# ``<head>`` so it also wins on source order.
_TRANSPARENT_BG_STYLE = (
    "<style>html,body{background:transparent !important;"
    "background-color:transparent !important;"
    "background-image:none !important}</style>"
)


def _inject_transparent_style(html: str) -> str:
    """Strip the page's own opaque background for a transparent capture.

    ``omit_background=True`` only makes Chromium's *default* white base
    transparent — every pixel the document itself paints still lands in the
    PNG. All bundled templates set ``body{background:...}``, so without this
    override the frames come back fully opaque and the ``transparent`` flag
    looks broken while reporting success. Note this handles ``html``/``body``
    only: templates that paint an opaque full-bleed stage element are NOT
    overlay-safe and are rejected at the API boundary instead (see
    ``templates.OVERLAY_SAFE_TEMPLATES``).
    """
    if "</head>" in html:
        return html.replace("</head>", _TRANSPARENT_BG_STYLE + "</head>", 1)
    return _TRANSPARENT_BG_STYLE + html


def _reject_opaque_container(output_format: str) -> None:
    """Refuse a transparent render that would be encoded without an alpha plane.

    H.264 has no alpha plane at all and GIF's is one bit (hard-keyed, jagged
    edges — not what a lower-third needs), so neither container can carry what
    was asked for. Handing back an opaque clip anyway is the silent degradation
    this service's error taxonomy exists to prevent: the caller would only find
    out when the "overlay" turned into a solid box over their footage, long
    after the job reported success.
    """
    if output_format != "webm":
        msg = (
            f"transparent rendering requires output_format='webm' (h264 has no "
            f"alpha plane, gif alpha is 1-bit); got '{output_format}'"
        )
        raise ValueError(msg)


class AnimationRecorder:
    """Capture a motion-graphics template to MP4 / WebM / GIF via frame stepping."""

    def __init__(self) -> None:
        settings = get_settings()
        self._headless = settings.recording.headless
        self._default_quality_preset = settings.recording.default_quality_preset
        # Execution-timeout sizing (Defect: an unbounded render could hold its
        # fleet slot forever). See the settings' own docstrings for why each
        # exists and how they compose.
        self._timeout_floor_s = settings.recording.animation_render_timeout_floor_seconds
        self._timeout_ceiling_s = settings.recording.animation_render_timeout_ceiling_seconds
        self._seconds_per_frame = settings.recording.animation_render_seconds_per_frame
        self._ffmpeg_seconds_per_frame = (
            settings.recording.animation_render_ffmpeg_seconds_per_frame
        )
        self._encode_reserve_s = settings.recording.animation_render_encode_reserve_seconds
        # Primary hang detector for the frame-step loop (see the setting's own
        # docstring for the per-frame capture time distribution this is sized
        # from) and the bound on Chromium launch/new_context/new_page.
        self._stall_window_s = settings.recording.animation_render_stall_window_seconds
        self._launch_timeout_s = settings.recording.animation_render_launch_timeout_seconds

    async def record_animation(
        self,
        job_id: str,
        html_content: str,
        animation_config: dict[str, Any] | None = None,
        recording_config: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Render and capture a motion template.

        Args:
            job_id: Unique job identifier (used to name the temp frame dir).
            html_content: Template HTML (already font-embedded; params injected
                separately via ``animation_config['params']``).
            animation_config: ``{params, fps, duration_ms, output_format,
                transparent}``. ``transparent`` (default False) captures with
                an alpha channel for compositing over footage; it forces the
                webm/VP9 path and is refused outright on a build whose VP9
                encoder drops alpha.
            recording_config: Optional viewport/quality overrides (same shape as
                the content recorder).
            progress_callback: Async progress reporter.

        Returns:
            Result dict mirroring :meth:`ContentRecorder.record_content`.
        """
        anim = animation_config or {}
        resolved = self._resolve_config(recording_config)
        fps = _clamp_int(anim.get("fps"), _DEFAULT_FPS, _MIN_FPS, _MAX_FPS)
        output_format = str(anim.get("output_format") or "mp4").lower()
        transparent = bool(anim.get("transparent"))
        params = anim.get("params")
        start = time.time()
        perf_start = time.perf_counter()
        timings = _new_timings()

        # Execution timeout, sized from the job's OWN work rather than a flat
        # number: an estimated frame count (the page-reported duration isn't
        # known until the page loads, so this uses the same request>default
        # precedence _resolve_duration_ms applies, minus the page candidate)
        # times a generous per-frame budget, plus a frame-scaled ffmpeg budget
        # and a fixed launch/overhead reserve, clamped to [floor, ceiling].
        # This is a BACKSTOP behind the per-frame stall watchdog in
        # _capture_frames (animation_render_stall_window_seconds) -- it only
        # has to bound the sum of a job's own frame times, not detect a hang
        # (a hang is caught per-frame, far sooner). The capture phase gets the
        # remainder after carving out the ffmpeg budget, and the encode
        # phase's ffmpeg subprocess(es) get that budget itself as their OWN
        # timeout (see _assemble) -- that is what actually guarantees a stuck
        # ffmpeg is killed at the OS level rather than merely orphaned by
        # cancelling the asyncio task awaiting it.
        duration_estimate_ms = _resolve_duration_ms(anim.get("duration_ms"), None)
        frame_estimate = min(_MAX_FRAMES, max(1, round(duration_estimate_ms / 1000.0 * fps)))
        encode_budget_s = self._encode_reserve_s + frame_estimate * self._ffmpeg_seconds_per_frame
        execution_timeout_s = max(
            self._timeout_floor_s,
            min(
                frame_estimate * self._seconds_per_frame + encode_budget_s, self._timeout_ceiling_s
            ),
        )
        capture_timeout_s = max(1.0, execution_timeout_s - encode_budget_s)

        # One per-job staging root holds the captured PNG frames and the encoded
        # output; recorder_service rmtrees ``staging_dir`` once the clip is stored.
        staging = Path(tempfile.mkdtemp(prefix=f"anim_{job_id}_"))

        try:
            if transparent:
                # Both gates run BEFORE the browser launches, so a request that
                # cannot possibly produce a usable overlay costs no capture.
                # They raise into the handler below, which turns them into an
                # explicit job failure (never a silently-opaque success).
                _reject_opaque_container(output_format)
                require_vp9_alpha_support()

            if progress_callback:
                await progress_callback(0.2, "launching_browser")

            flags = RenderFlags()

            try:
                frames_dir, frame_count, _duration_ms, max_frame_seconds = await asyncio.wait_for(
                    self._capture_frames(
                        job_id,
                        staging,
                        html_content,
                        resolved,
                        fps,
                        anim.get("duration_ms"),
                        params,
                        progress_callback,
                        transparent=transparent,
                        timings=timings,
                        flags=flags,
                    ),
                    timeout=capture_timeout_s,
                )
            except TimeoutError as exc:
                # _capture_frames's own finally (context.close()/browser.close(),
                # bounded by _TEARDOWN_TIMEOUT_S) still runs during this
                # cancellation -- the slot release happens in the caller
                # (RecorderService._attempt_recording's finally), which is why
                # failing loudly here (rather than hanging) is what lets it run.
                if str(exc):
                    # A per-frame stall or a bounded-launch timeout inside
                    # _capture_frames already raised its own descriptive
                    # TimeoutError (message present) -- preserve it verbatim
                    # so a caller/log can tell "no progress on frame N" or "
                    # launch hung" apart from the generic backstop below,
                    # rather than every timeout collapsing into one message.
                    raise
                # A bare asyncio.wait_for cancellation (str(exc) == "") means
                # the OUTER capture-phase backstop fired instead -- the whole
                # phase ran long without any single frame stalling.
                raise TimeoutError(
                    f"animation render exceeded its {capture_timeout_s:.0f}s capture "
                    f"budget ({frame_estimate} frames est. @ {fps}fps, "
                    f"execution ceiling {execution_timeout_s:.0f}s); browser "
                    "capture terminated"
                ) from exc

            if progress_callback:
                await progress_callback(0.8, "encoding")

            final_path = await asyncio.to_thread(
                self._assemble,
                frames_dir,
                fps,
                output_format,
                resolved["crf"],
                transparent,
                # Keyword-only: ``transparent`` stays the last POSITIONAL arg.
                target_width=resolved["width"],
                target_height=resolved["height"],
                capture_scale=resolved["capture_scale"],
                # The ffmpeg subprocess's OWN timeout (see FFmpegOps.run): a
                # stuck encode is killed at the OS level within this window,
                # not merely orphaned by asyncio cancelling the to_thread call
                # (cancellation cannot stop a blocking subprocess.run() call
                # already running on a worker thread).
                timeout=int(encode_budget_s),
                timings=timings,
            )

            probe_start = time.perf_counter()
            probe = await asyncio.to_thread(FFmpegOps.probe_all, final_path)
            timings["probe_s"] += time.perf_counter() - probe_start
            elapsed = time.time() - start
            _log_render_timing(
                job_id,
                timings,
                capture_scale=resolved["capture_scale"],
                fps=fps,
                width=resolved["width"],
                height=resolved["height"],
                total_s=time.perf_counter() - perf_start,
                success=True,
            )

            if progress_callback:
                await progress_callback(1.0, "completed")

            # The raster Chromium ACTUALLY rasterised: the viewport times the
            # EFFECTIVE supersampling factor _resolve_capture_scale returned --
            # not the request, and not _CAPTURE_SCALE. The budget check steps
            # the factor down silently (a 4k order captures at 1x), and until
            # this key existed no consumer could tell. Reported as absolute
            # pixel dimensions rather than the factor so a caller sizing source
            # assets to the capture needs no arithmetic and no knowledge of this
            # module's private constant; a clamp-down is then visible as
            # capture_width == width.
            capture_scale = int(resolved["capture_scale"])
            return {
                "success": True,
                "video_path": str(final_path),
                "duration_seconds": probe.duration or (frame_count / fps),
                "file_size_bytes": final_path.stat().st_size,
                "width": probe.width or resolved["width"],
                "height": probe.height or resolved["height"],
                "capture_width": resolved["width"] * capture_scale,
                "capture_height": resolved["height"] * capture_scale,
                "has_audio": probe.has_audio,
                "recording_time_seconds": elapsed,
                # Slowest single frame's own capture time (see the stall-window
                # setting's docstring) -- the per-frame telemetry that did not
                # exist before this field, so the stall window can eventually be
                # tuned from real data instead of job-level averages.
                "max_frame_seconds": max_frame_seconds,
                "staging_dir": str(staging),
                # text the template drew past its line budget (see _flag_fit_overflow)
                **(
                    {"fit_overflow": {k: v.model_dump() for k, v in flags.fit_overflow.items()}}
                    if flags.fit_overflow
                    else {}
                ),
            }
        except (OSError, RuntimeError, ValueError, TimeoutError, PlaywrightError) as exc:
            logger.error("Animation recording failed for job %s: %s", job_id, exc)
            _log_render_timing(
                job_id,
                timings,
                capture_scale=resolved["capture_scale"],
                fps=fps,
                width=resolved["width"],
                height=resolved["height"],
                total_s=time.perf_counter() - perf_start,
                success=False,
            )
            return {
                "success": False,
                "error": str(exc),
                "recording_time_seconds": time.time() - start,
                "staging_dir": str(staging),
            }

    @staticmethod
    async def _flag_fit_overflow(page: Any, job_id: str, flags: RenderFlags | None) -> None:
        """Text the template could not fit at its smallest type is drawn, never
        clamped — and flagged here, loudly (render 01a1179d: a caller had cut its
        own text to fit and nothing knew). ``flags.fit_overflow`` carries
        ``{element: FitOverflow}`` into the recording result."""
        try:
            overflow = await page.evaluate("window.FIT_OVERFLOW || null")
        except PlaywrightError, RuntimeError, TypeError:
            return
        if not isinstance(overflow, dict) or not overflow:
            return
        try:
            report = RenderFlags.model_validate({"fit_overflow": overflow}).fit_overflow
        except ValidationError:
            report = {}
        logger.warning(
            "Animation %s: text overflows its template at the smallest type (drawn, not "
            "clamped): %s",
            job_id,
            overflow,
        )
        if flags is not None and report:
            flags.fit_overflow = report

    async def _capture_frames(
        self,
        job_id: str,
        frames_dir: Path,
        html_content: str,
        config: dict[str, Any],
        fps: int,
        requested_duration_ms: Any,
        params: Any,
        progress_callback: ProgressCallback | None,
        *,
        transparent: bool = False,
        timings: dict[str, float] | None = None,
        flags: RenderFlags | None = None,
    ) -> tuple[Path, int, float, float]:
        """Launch Chromium, seek the timeline per frame, and screenshot each.

        Frames are written into ``frames_dir`` (the per-job staging root); the
        caller owns its lifecycle and cleanup.

        ``transparent`` captures RGBA frames: the template's own backdrop is
        overridden away and Chromium's default white base is omitted. Both
        halves are required — either one alone still yields opaque PNGs.

        Returns ``(frames_dir, frame_count, duration_ms, max_frame_seconds)`` --
        the last element is the slowest single frame's own capture time (see
        ``Settings.animation_render_stall_window_seconds`` for why it exists),
        which ``record_animation`` persists into its result dict.

        ``timings`` (optional) accumulates per-phase wall time in place -- see
        ``_TIMING_PHASES``.
        """
        from playwright.async_api import async_playwright

        if timings is None:
            timings = _new_timings()

        width, height = config["width"], config["height"]
        # The viewport is always the TARGET size (layout must not reflow), while
        # device_scale_factor multiplies the rasterisation — so each PNG comes
        # back at width*scale x height*scale and _assemble scales it back down.
        # Mirrors url_recorder's capture_scale handling; unlike that path the
        # factor is always passed, because this context already pinned an
        # explicit device_scale_factor of 1 (so 1 is not an omission here).
        scale = int(config.get("capture_scale") or 1)

        async with async_playwright() as p:
            # Both handles start unset and _safe_close tolerates either being
            # None -- a hang or error partway through launch (Chromium starts
            # but new_context/new_page fails) must still close whatever DID
            # get created. Before this fix, launch/new_context/install_ssrf_
            # guard/new_page ran BEFORE this try/finally existed at all, so a
            # hang during startup leaked the browser/context with no teardown.
            # Both handles start unset and _safe_close tolerates either being
            # None -- a hang or error partway through launch (Chromium starts
            # but new_context/new_page fails) must still close whatever DID
            # get created. Before this fix, launch/new_context/install_ssrf_
            # guard/new_page ran BEFORE this try/finally existed at all, so a
            # hang during startup leaked the browser/context with no teardown.
            browser = None
            context = None
            setup_start = time.perf_counter()
            try:
                browser = await asyncio.wait_for(
                    p.chromium.launch(headless=config.get("headless", True)),
                    timeout=self._launch_timeout_s,
                )
                # A direct `context = await browser DOT new_context(...)` shape
                # (rather than wrapping the call as an argument to
                # asyncio.wait_for, as launch/new_page below still do) is
                # load-bearing: the repo-wide SSRF-guard walker (tests/unit/
                # test_playwright_guard_and_webhook_ssrf.py) finds every
                # context-factory call site textually and separately asserts
                # each one is bound to a variable by the AST -- a context
                # created as a nested call argument matches textually but is
                # invisible to that binding check, so it fails closed instead
                # of being verified guarded. asyncio's `timeout` context
                # manager gives the same bounded-launch behaviour (raises
                # TimeoutError on expiry) while keeping the assignment
                # directly recognisable to the walker.
                async with asyncio.timeout(self._launch_timeout_s):
                    context = await browser.new_context(
                        viewport={"width": width, "height": height},
                        device_scale_factor=scale,
                        color_scheme=config.get("color_scheme", "light"),
                    )
                # ``html_content`` is a whole document that reaches here
                # straight from ``POST /api/record/animation`` when the
                # caller supplies ``body.html`` inline, so every sub-resource
                # it names is a request Chromium issues from inside the pod
                # network. The bundled templates are trusted; the inline path
                # is not, and both share this context.
                await asyncio.wait_for(install_ssrf_guard(context), timeout=self._launch_timeout_s)
                page = await asyncio.wait_for(context.new_page(), timeout=self._launch_timeout_s)

                # Inline window.PARAMS into the markup — set_content does not fire
                # add_init_script, so this is the only way the template's inline
                # script sees the params (see _inject_params).
                document = _inject_params(html_content, params)
                if transparent:
                    document = _inject_transparent_style(document)
                await page.set_content(
                    document,
                    wait_until="networkidle",
                    timeout=30000,
                )
                # Embedded fonts are data-URIs but still need a layout pass.
                await page.evaluate("document.fonts && document.fonts.ready")
                await self._flag_fit_overflow(page, job_id, flags)

                duration_ms = _resolve_duration_ms(
                    requested_duration_ms,
                    await page.evaluate("window.RENDER_DURATION_MS || null"),
                )
                frame_count = max(1, round(duration_ms / 1000.0 * fps))
                if frame_count > _MAX_FRAMES:
                    logger.warning(
                        "Animation %s clamped from %d to %d frames",
                        job_id,
                        frame_count,
                        _MAX_FRAMES,
                    )
                    frame_count = _MAX_FRAMES
                timings["setup_s"] += time.perf_counter() - setup_start

                # Slowest single frame's own capture time (seek + screenshot),
                # tracked so the stall window can eventually be tuned from real
                # per-frame telemetry instead of the job-level AVERAGES it is
                # justified from today (see the setting's docstring). Persisted
                # as ``max_frame_seconds`` on the job row.
                max_frame_seconds = 0.0
                for i in range(frame_count):
                    t = i * 1000.0 / fps
                    frame_path = frames_dir / f"frame_{i:06d}.png"
                    frame_start = time.monotonic()
                    try:
                        # PRIMARY hang detector: a stall is lack of PROGRESS,
                        # not total duration, so each frame's own work (seek +
                        # screenshot) gets its own bounded window rather than
                        # the whole loop sharing one deadline. A render with
                        # many frames that each complete on time is never
                        # killed here regardless of total elapsed time --
                        # that is the backstop's job (capture_timeout_s in
                        # record_animation).
                        await asyncio.wait_for(
                            self._capture_one_frame(page, t, frame_path, transparent, timings),
                            timeout=self._stall_window_s,
                        )
                    except TimeoutError as exc:
                        raise TimeoutError(
                            f"animation capture stalled: no progress on frame "
                            f"{i}/{frame_count} within {self._stall_window_s:.0f}s "
                            f"(job {job_id})"
                        ) from exc
                    max_frame_seconds = max(max_frame_seconds, time.monotonic() - frame_start)
                    timings["frames"] += 1
                    if (i + 1) % _TIMING_DEBUG_EVERY_FRAMES == 0:
                        logger.debug(
                            "animation_render_progress job_id=%s frame=%d/%d seek_s=%.2f "
                            "screenshot_s=%.2f",
                            job_id,
                            i + 1,
                            frame_count,
                            timings["seek_s"],
                            timings["screenshot_s"],
                        )
                    if progress_callback and frame_count > 1 and i % 15 == 0:
                        await progress_callback(0.2 + 0.6 * (i / frame_count), "capturing_frames")
            finally:
                # Runs on EVERY exit, including a timeout-driven cancellation
                # from record_animation's asyncio.wait_for, or a stall/launch
                # timeout raised above -- see _safe_close.
                await self._safe_close(context, browser)

        return frames_dir, frame_count, duration_ms, max_frame_seconds

    @staticmethod
    async def _capture_one_frame(
        page: Page,
        t: float,
        frame_path: Path,
        transparent: bool,
        timings: dict[str, float] | None = None,
    ) -> None:
        """Seek the timeline to ``t`` and screenshot it -- the unit of PROGRESS
        the stall watchdog in ``_capture_frames`` measures against."""
        seek_start = time.perf_counter()
        await page.evaluate(_STEP_JS, t)
        shot_start = time.perf_counter()
        # omit_background drops Chromium's default white base so the PNG
        # carries real alpha. False is Playwright's own default, so the
        # opaque path is unchanged.
        await page.screenshot(path=str(frame_path), omit_background=transparent)
        if timings is not None:
            shot_end = time.perf_counter()
            timings["seek_s"] += shot_start - seek_start
            timings["screenshot_s"] += shot_end - shot_start

    @staticmethod
    async def _safe_close(context: BrowserContext | None, browser: Browser | None) -> None:
        """Best-effort, BOUNDED teardown of the capture context/browser.

        Called from ``_capture_frames``'s ``finally`` on every exit path,
        including cancellation from an outer ``asyncio.wait_for`` timeout, AND
        a hang/error during launch itself (``context`` and/or ``browser`` may
        be ``None`` when launch never completed -- only the handles that were
        actually acquired get closed). A hung Playwright ``close()`` here
        would otherwise hang the CANCELLATION itself -- defeating the whole
        point of the execution timeout, since the caller would keep waiting
        past its own deadline for a teardown that never finishes. Each half
        gets its own bounded window; a failure to close is logged, never
        re-raised -- the render has already failed for its own reason (or
        succeeded) and a teardown error must not replace that outcome.
        """
        for label, handle in (("context", context), ("browser", browser)):
            if handle is None:
                continue
            try:
                await asyncio.wait_for(handle.close(), timeout=_TEARDOWN_TIMEOUT_S)
            except Exception:
                logger.warning(
                    "Animation recorder: failed to close %s cleanly within %.0fs",
                    label,
                    _TEARDOWN_TIMEOUT_S,
                    exc_info=True,
                )

    @classmethod
    def _assemble(
        cls,
        frames_dir: Path,
        fps: int,
        output_format: str,
        crf: int,
        transparent: bool = False,
        *,
        target_width: int | None = None,
        target_height: int | None = None,
        capture_scale: int = 1,
        timeout: int | None = None,
        timings: dict[str, float] | None = None,
    ) -> Path:
        """Encode the captured frame sequence into the requested container.

        When the frames were supersampled (``capture_scale > 1``) they are first
        downscaled to ``target_width`` x ``target_height`` with lanczos — the
        second half of the supersampling win, and the reason the encoders below
        see the ordered output size rather than a 2x one.

        ``capture_scale`` defaults to 1, so a direct caller (the compiler's
        segment builders build animation configs by hand) gets the pre-feature
        argv verbatim.

        ``timeout`` overrides each ffmpeg subprocess's own timeout (the
        downscale pass and the final encode each get this budget, so worst
        case is up to 2x it, not shared between them). ``None`` keeps
        ``FFmpegOps``'s class-wide default, which is what every caller other
        than ``record_animation`` gets.

        ``timings`` (optional) accumulates ``downscale_s`` / ``encode_s`` in place.
        """
        # Defence in depth. record_animation already refused this combination,
        # but _assemble is also reachable directly (the compiler's segment
        # builders construct animation configs by hand), and an alpha-carrying
        # frame sequence flattened into h264/gif here would fail exactly the
        # same silent way.
        if transparent:
            _reject_opaque_container(output_format)
        if capture_scale > 1 and target_width and target_height:
            downscale_start = time.perf_counter()
            frames_dir = cls._downscale_frames(
                frames_dir, target_width, target_height, transparent=transparent, timeout=timeout
            )
            if timings is not None:
                timings["downscale_s"] += time.perf_counter() - downscale_start
        encode_start = time.perf_counter()
        try:
            return cls._encode(frames_dir, fps, output_format, crf, transparent, timeout)
        finally:
            if timings is not None:
                timings["encode_s"] += time.perf_counter() - encode_start

    @staticmethod
    def _encode(
        frames_dir: Path,
        fps: int,
        output_format: str,
        crf: int,
        transparent: bool,
        timeout: int | None,
    ) -> Path:
        """Encode the (already output-sized) frame sequence into the container."""
        if output_format == "gif":
            return FFmpegOps.frames_to_gif(
                frames_dir, frames_dir / "out.gif", fps=fps, timeout=timeout
            )
        if output_format == "webm":
            return FFmpegOps.frames_to_webm(
                frames_dir,
                frames_dir / "out.webm",
                fps=fps,
                transparent=transparent,
                timeout=timeout,
            )
        if output_format == "mp4":
            return FFmpegOps.frames_to_mp4(
                frames_dir, frames_dir / "out.mp4", fps=fps, crf=crf, timeout=timeout
            )
        raise ValueError(f"output_format must be one of mp4|webm|gif, got '{output_format}'")

    @staticmethod
    def _downscale_frames(
        frames_dir: Path,
        width: int,
        height: int,
        *,
        transparent: bool = False,
        timeout: int | None = None,
    ) -> Path:
        """Lanczos-downscale a supersampled PNG sequence to the output size.

        Returns the directory holding the scaled sequence (same
        ``frame_%06d.png`` naming, so every encoder below is unchanged). The
        originals are unlinked once the pass succeeds: at 2x they are 4x the
        bytes, and keeping both sequences for the length of the encode is how a
        long clip fills the pod's ephemeral storage.
        """
        scaled_dir = frames_dir / _SCALED_DIRNAME
        scaled_dir.mkdir(parents=True, exist_ok=True)
        vf = f"scale={width}:{height}:flags=lanczos"
        args = [
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-vf",
            vf,
        ]
        if transparent:
            # scale() on an RGBA input keeps alpha, but the PNG encoder picks
            # its pix_fmt from the negotiated format — pin rgba so an overlay
            # clip cannot lose its alpha plane here instead of at the encode.
            args += ["-pix_fmt", "rgba"]
        args.append(str(scaled_dir / "frame_%06d.png"))
        FFmpegOps.run(args, timeout=timeout)
        for stale in frames_dir.glob("frame_*.png"):
            stale.unlink(missing_ok=True)
        return scaled_dir

    def _resolve_config(self, recording_config: dict[str, Any] | None) -> dict[str, Any]:
        """Merge defaults with optional viewport/quality overrides.

        Resolution precedence is explicit > preset > default. An explicitly
        requested tier carries its RESOLUTION as well as its CRF: before this,
        a ``2k``/``4k`` order reaching ``POST /api/recording`` without an
        explicit ``body.width`` got a square 1080x1080 clip at a 2k CRF, then
        was upscaled and mis-framed downstream. The default tier (no
        ``quality_preset`` in the request at all) keeps the 1080x1080 square the
        seed templates are authored at.
        """
        overrides = recording_config or {}
        requested_preset = overrides.get("quality_preset")
        explicit_preset = bool(requested_preset)
        preset_name = str(requested_preset) if explicit_preset else self._default_quality_preset
        preset = get_quality_preset(preset_name)

        config: dict[str, Any] = {
            "width": preset.viewport_width if explicit_preset else _DEFAULT_WIDTH,
            "height": preset.viewport_height if explicit_preset else _DEFAULT_HEIGHT,
            "crf": preset.crf,
            "headless": self._headless,
            "color_scheme": "light",
        }
        if overrides.get("viewport_width"):
            config["width"] = int(overrides["viewport_width"])
        if overrides.get("viewport_height"):
            config["height"] = int(overrides["viewport_height"])
        for key in ("headless", "color_scheme"):
            if overrides.get(key) is not None:
                config[key] = overrides[key]
        # Resolved LAST: the budget below has to be measured against the final
        # output size, not the preset's.
        config["capture_scale"] = _resolve_capture_scale(
            config["width"], config["height"], overrides.get("capture_scale")
        )
        return config


def _resolve_capture_scale(
    width: int, height: int, requested: int | str | float | None = None
) -> int:
    """Pick the supersampling factor that fits ``_MAX_CAPTURE_RASTER_PIXELS``.

    Returns 1 when even 2x would exceed the raster budget (a 4k order), so the
    quality win is taken wherever it is affordable and silently skipped where it
    would trade sharpness for an OOM kill. An explicit ``capture_scale`` is
    honoured but clamped by the same budget — the ceiling protects the pod, not
    the default.
    """
    scale = _clamp_int(requested, _CAPTURE_SCALE, 1, _CAPTURE_SCALE)
    area = max(1, width) * max(1, height)
    while scale > 1 and area * scale * scale > _MAX_CAPTURE_RASTER_PIXELS:
        scale -= 1
    return scale


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        n = int(value) if value is not None else default
    except TypeError, ValueError:
        n = default
    return max(low, min(n, high))


def _resolve_duration_ms(requested: Any, from_page: Any) -> float:
    """Pick the timeline length: explicit request > template default > fallback."""
    for candidate in (requested, from_page):
        try:
            if candidate is not None:
                ms = float(candidate)
                if ms >= _MIN_DURATION_MS:
                    return ms
        except TypeError, ValueError:
            continue
    return _DEFAULT_DURATION_MS
