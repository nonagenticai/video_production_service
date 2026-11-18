"""Record scrolling HTML content via Playwright using `page.set_content`."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from playwright.async_api import Error as PlaywrightError

from src.config import get_settings
from src.services.ffmpeg import FFmpegOps
from src.services.recording.quality_presets import get_quality_preset
from src.shared.security.playwright_guard import install_ssrf_guard

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]

# Default scroll configuration
_DEFAULT_SCROLL_SPEED = 300  # px per second
_DEFAULT_START_DELAY = 1.0  # seconds to pause before scrolling
_DEFAULT_END_DELAY = 2.0  # seconds to pause after scrolling


class ContentRecorder:
    """Record scrolling HTML content as a video via Playwright."""

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

    async def record_content(
        self,
        job_id: str,
        html_content: str,
        recording_config: dict[str, Any] | None = None,
        scroll_config: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Record HTML content being scrolled through.

        Args:
            job_id: Unique job identifier.
            html_content: Raw HTML string to render and record.
            recording_config: Optional viewport/quality overrides.
            scroll_config: Optional scroll speed/delay overrides.
            progress_callback: Async callback for progress updates.

        Returns:
            Result dict with video_path, duration_seconds, etc.
        """
        resolved = self._resolve_config(recording_config)
        scroll = self._resolve_scroll_config(scroll_config)
        start = time.time()

        # One per-job staging root holds every temp artifact (webm + post-processed
        # mp4); recorder_service rmtrees ``staging_dir`` once the output is stored.
        staging = Path(tempfile.mkdtemp(prefix=f"content_rec_{job_id}_"))

        try:
            if progress_callback:
                await progress_callback(0.3, "launching_browser")

            video_path = await self._record_with_playwright(
                job_id, html_content, resolved, scroll, progress_callback, staging
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
            logger.error("Content recording failed for job %s: %s", job_id, exc)
            return {
                "success": False,
                "error": str(exc),
                "recording_time_seconds": time.time() - start,
                "staging_dir": str(staging),
            }

    async def _record_with_playwright(
        self,
        job_id: str,
        html_content: str,
        config: dict[str, Any],
        scroll: dict[str, Any],
        progress_callback: ProgressCallback | None,
        staging: Path,
    ) -> Path:
        """Launch Playwright, load HTML via set_content, scroll, and capture video."""
        from playwright.async_api import async_playwright

        video_dir = Path(tempfile.mkdtemp(prefix="webm_", dir=staging))

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
            )
            # Before any page exists. ``html_content`` is tenant-supplied
            # (``POST /api/record/content``) and unvalidated by construction --
            # it is a whole document. Every <img>/<script>/<link>/<iframe>/fetch
            # in it is a request Chromium issues from inside the pod network,
            # and ``wait_until="networkidle"`` below guarantees the recorder
            # *waits* for them. Without this the endpoint is a working in-pod
            # SSRF and port scanner; a handler installed later misses the load.
            await install_ssrf_guard(context)

            page = await context.new_page()

            if progress_callback:
                await progress_callback(0.4, "loading_content")

            await page.set_content(html_content, wait_until="networkidle", timeout=30000)

            if progress_callback:
                await progress_callback(0.5, "scrolling")

            await self._auto_scroll(page, scroll)

            if progress_callback:
                await progress_callback(0.7, "finalizing_video")

            # Close context to finalize the video
            await context.close()
            await browser.close()

        # Find the recorded video file
        video_files = list(video_dir.glob("*.webm"))
        if not video_files:
            msg = f"No video file produced in {video_dir}"
            raise RuntimeError(msg)

        return max(video_files, key=lambda f: f.stat().st_mtime)

    async def _auto_scroll(self, page: Any, scroll_config: dict[str, Any]) -> None:
        """Scroll the page from top to bottom at a configurable speed."""
        start_delay = scroll_config.get("start_delay_seconds", _DEFAULT_START_DELAY)
        end_delay = scroll_config.get("end_delay_seconds", _DEFAULT_END_DELAY)
        speed = scroll_config.get("speed_px_per_second", _DEFAULT_SCROLL_SPEED)
        enabled = scroll_config.get("enabled", True)

        if start_delay > 0:
            await asyncio.sleep(start_delay)

        if not enabled:
            # Just wait at top for a fixed duration
            await asyncio.sleep(end_delay)
            return

        # Scroll in steps: 100px per tick, calculate interval from speed
        step_px = 100
        interval = step_px / speed  # seconds between steps

        scroll_height = await page.evaluate("document.body.scrollHeight")
        viewport_height = await page.evaluate("window.innerHeight")
        total_scroll = max(0, scroll_height - viewport_height)

        scrolled = 0
        while scrolled < total_scroll:
            step = min(step_px, total_scroll - scrolled)
            await page.evaluate(f"window.scrollBy(0, {step})")
            scrolled += step
            await asyncio.sleep(interval)

        if end_delay > 0:
            await asyncio.sleep(end_delay)

    async def _post_process(self, video_path: Path, config: dict[str, Any]) -> Path:
        """Convert WebM to MP4 if needed."""
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
        }

        if recording_config:
            for key in (
                "viewport_width",
                "viewport_height",
                "headless",
                "output_format",
                "color_scheme",
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
        }
        if scroll_config:
            defaults.update({k: v for k, v in scroll_config.items() if v is not None})
        return defaults
