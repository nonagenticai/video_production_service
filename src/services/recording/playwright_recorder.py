"""Execute Playwright scripts with video capture.

Manages subprocess execution of Playwright test scripts, video discovery,
optional MP4 post-processing, and ffprobe metadata extraction.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from src.api.schemas.recording import sanitize_auth_env_vars
from src.config import get_settings
from src.exceptions import FFmpegConversionError, FFmpegError, FFmpegProbeError
from src.services.recording.quality_presets import get_quality_preset

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], Coroutine[Any, Any, None]]


class PlaywrightRecorder:
    """Execute each accepted script with video capture via Playwright."""

    def __init__(self) -> None:
        settings = get_settings()
        self._viewport_width = settings.recording.viewport_width
        self._viewport_height = settings.recording.viewport_height
        self._timeout_ms = settings.recording.timeout_ms
        self._timeout_buffer_s = settings.recording.timeout_buffer_s
        self._min_video_size = settings.recording.min_video_size_bytes
        self._slow_mo_ms = settings.recording.slow_mo_ms
        self._color_scheme = settings.recording.color_scheme
        self._headless = settings.recording.headless
        self._output_format = settings.recording.output_format
        self._default_quality_preset = settings.recording.default_quality_preset
        self._ffmpeg_preset = settings.ffmpeg.preset
        self._ffmpeg_timeout = settings.ffmpeg.timeout
        self._ffmpeg_default_crf = settings.ffmpeg.default_crf
        self._error_output_max_length = settings.recording.error_output_max_length
        self._cancel_wait_timeout_s = settings.recording.cancel_wait_timeout_s
        # Track running subprocesses for cancellation
        self._running_processes: dict[str, asyncio.subprocess.Process] = {}

    def _resolve_recording_config(self, recording_config: dict | None) -> dict[str, Any]:
        """Merge per-request config overrides with instance defaults.

        When a quality_preset is specified (in recording_config or as the default),
        the preset's viewport dimensions are applied first, then explicit overrides
        take precedence.
        """
        config: dict[str, Any] = {
            "viewport_width": self._viewport_width,
            "viewport_height": self._viewport_height,
            "timeout_ms": self._timeout_ms,
            "slow_mo_ms": self._slow_mo_ms,
            "color_scheme": self._color_scheme,
            "headless": self._headless,
            "output_format": self._output_format,
            "quality_preset": self._default_quality_preset,
        }

        # Determine the quality preset name (request override > default)
        raw_preset = (
            recording_config.get("quality_preset")
            if recording_config and recording_config.get("quality_preset") is not None
            else None
        )
        preset_name: str = (
            str(raw_preset) if raw_preset is not None else self._default_quality_preset
        )

        # Apply quality preset viewport dimensions
        try:
            preset = get_quality_preset(preset_name)
            config["viewport_width"] = preset.viewport_width
            config["viewport_height"] = preset.viewport_height
            config["quality_preset"] = preset_name
        except ValueError:
            logger.warning("Unknown quality preset '%s', using defaults", preset_name)

        # Apply explicit per-request overrides (these take precedence over preset)
        if recording_config:
            override_keys = {
                "viewport_width",
                "viewport_height",
                "timeout_ms",
                "slow_mo_ms",
                "color_scheme",
                "headless",
                "output_format",
            }
            for key in override_keys:
                if recording_config.get(key) is not None:
                    config[key] = recording_config[key]

        return config

    async def record_part(
        self,
        part_id: str,
        scripts_dir: str,
        recording_config: dict | None = None,
        auth_config: dict | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict:
        """Record a single part by running its Playwright script.

        Returns dict with: part_id, success, video_path, duration_seconds,
        file_size_bytes, width, height, has_audio, error
        """
        logger.info("Recording part %s from %s", part_id, scripts_dir)
        t0 = time.monotonic()

        result = self._validate_and_resolve_script(part_id, scripts_dir)
        if isinstance(result, dict):
            return result
        scripts_path, script_file = result

        if progress_callback:
            await progress_callback(0.1, "initializing")

        resolved_config = self._resolve_recording_config(recording_config)

        sub_result = await self._run_recording_subprocess(
            part_id, script_file, scripts_path, resolved_config, auth_config, progress_callback
        )
        if isinstance(sub_result, dict):
            return sub_result
        returncode, stdout_str, stderr_str = sub_result

        if progress_callback:
            await progress_callback(0.7, "processing_video")

        video_path = self._find_video(scripts_path / "test-results", part_id)
        output_combined = (stderr_str + "\n" + stdout_str).strip()

        if returncode != 0 and not video_path:
            return {
                "part_id": part_id,
                "success": False,
                "error": f"Recording failed: {output_combined[: self._error_output_max_length]}",
            }

        accepted_with_warning = False
        if returncode != 0 and video_path:
            video_size = video_path.stat().st_size if video_path.exists() else 0
            if video_size < self._min_video_size:
                return {
                    "part_id": part_id,
                    "success": False,
                    "video_path": str(video_path),
                    "error": f"Playwright exited {returncode}, video too small ({video_size}B)",
                }
            logger.warning(
                "Playwright exited with code %d for %s but video looks usable (%d bytes)",
                returncode,
                part_id,
                video_size,
            )
            accepted_with_warning = True

        if video_path:
            video_path = await self._post_process_video(
                part_id, video_path, resolved_config, progress_callback
            )

        elapsed = time.monotonic() - t0
        return await self._probe_and_return(
            part_id, video_path, resolved_config, returncode, accepted_with_warning, elapsed
        )

    # ------------------------------------------------------------------
    # Private helpers extracted from record_part
    # ------------------------------------------------------------------

    def _validate_and_resolve_script(
        self, part_id: str, scripts_dir: str
    ) -> tuple[Path, Path] | dict:
        """Validate *scripts_dir* and locate the Playwright script file.

        Returns ``(scripts_path, script_file)`` on success, or an error dict
        that can be returned directly from :meth:`record_part`.
        """
        scripts_path = Path(scripts_dir)

        try:
            scripts_path = scripts_path.resolve()
        except OSError, ValueError:
            return {
                "part_id": part_id,
                "success": False,
                "error": f"Invalid scripts directory: {scripts_dir}",
            }

        allowed_base = Path(get_settings().recording.scripts_base_dir).resolve()
        if not scripts_path.is_relative_to(allowed_base):
            return {
                "part_id": part_id,
                "success": False,
                "error": f"scripts_dir must be within {allowed_base}",
            }

        script_file = None
        for ext in (".demo.spec.ts", ".test.ts", ".ts"):
            candidate = scripts_path / "parts" / f"{part_id}{ext}"
            if candidate.exists():
                script_file = candidate
                break

        if not script_file:
            return {
                "part_id": part_id,
                "success": False,
                "error": f"Script file not found for {part_id}",
            }

        # Playwright's default test matcher excludes plain "*.ts" files.
        # Create a discoverable alias for legacy title-card scripts.
        if script_file.name.endswith(".ts") and not (
            script_file.name.endswith(".demo.spec.ts") or script_file.name.endswith(".test.ts")
        ):
            alias_file = script_file.with_name(f"{script_file.stem}.demo.spec.ts")
            if not alias_file.exists():
                try:
                    alias_file.write_text(script_file.read_text(encoding="utf-8"), encoding="utf-8")
                except OSError as exc:
                    return {
                        "part_id": part_id,
                        "success": False,
                        "error": f"Failed to prepare script for Playwright: {exc}",
                    }
            script_file = alias_file

        return scripts_path, script_file

    @staticmethod
    def _build_recording_env(auth_config: dict | None, resolved_config: dict) -> dict[str, str]:
        env = os.environ.copy()
        if auth_config and auth_config.get("env_vars"):
            safe_env_vars = sanitize_auth_env_vars(auth_config["env_vars"])
            if len(safe_env_vars) != len(auth_config["env_vars"]):
                dropped = sorted(set(auth_config["env_vars"].keys()) - set(safe_env_vars.keys()))
                logger.warning(
                    "Dropped %d unsafe auth_config env_vars before subprocess launch: %s",
                    len(dropped),
                    dropped,
                )
            env.update(safe_env_vars)
        env["RECORDING_VIEWPORT_WIDTH"] = str(resolved_config["viewport_width"])
        env["RECORDING_VIEWPORT_HEIGHT"] = str(resolved_config["viewport_height"])
        env["RECORDING_TIMEOUT_MS"] = str(resolved_config["timeout_ms"])
        env["RECORDING_SLOW_MO_MS"] = str(resolved_config["slow_mo_ms"])
        env["RECORDING_COLOR_SCHEME"] = str(resolved_config["color_scheme"])
        env["RECORDING_HEADLESS"] = str(resolved_config["headless"]).lower()
        return env

    async def _run_recording_subprocess(
        self,
        part_id: str,
        script_file: Path,
        scripts_path: Path,
        resolved_config: dict,
        auth_config: dict | None,
        progress_callback: ProgressCallback | None,
    ) -> tuple[int, str, str] | dict:
        """Build the environment, launch Playwright, and wait for completion.

        Returns ``(returncode, stdout_str, stderr_str)`` on success, or an
        error dict on timeout / cancellation.
        """
        env = self._build_recording_env(auth_config, resolved_config)

        effective_timeout_ms = resolved_config["timeout_ms"]
        total_timeout_s = effective_timeout_ms // 1000 + self._timeout_buffer_s

        try:
            relative_script = script_file.relative_to(scripts_path)
        except ValueError:
            relative_script = script_file

        if progress_callback:
            await progress_callback(0.2, "starting_subprocess")

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "npx",
                "playwright",
                "test",
                str(relative_script),
                "--config=playwright.recording.config.ts",
                "--reporter=line",
                "--output",
                str(Path("test-results") / part_id),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=scripts_path,
                env=env,
                # Own session/process group so the whole tree (npx → node →
                # Chromium) can be signalled at once on timeout/cancel. Without
                # this, terminating `proc` only reaches npx and orphans the
                # browser, which keeps writing video and consuming memory.
                start_new_session=True,
            )
            self._running_processes[part_id] = proc

            if progress_callback:
                await progress_callback(0.3, "recording")

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=total_timeout_s)
        except TimeoutError:
            logger.error("Recording timed out for %s", part_id)
            await self._handle_subprocess_timeout(part_id, proc, scripts_path)
            return {
                "part_id": part_id,
                "success": False,
                "error": f"Recording timed out after {total_timeout_s}s",
            }
        except asyncio.CancelledError:
            # Best-effort synchronous tree kill — awaiting here could re-raise the
            # cancellation before the browser is reaped, leaving it orphaned.
            if proc is not None:
                self._signal_process_group(proc, signal.SIGKILL)
            self._running_processes.pop(part_id, None)
            return {
                "part_id": part_id,
                "success": False,
                "error": "Recording was cancelled",
            }
        finally:
            self._running_processes.pop(part_id, None)

        stdout_str = stdout.decode("utf-8", errors="replace") or ""
        stderr_str = stderr.decode("utf-8", errors="replace") or ""
        # After communicate() the process has exited, so returncode is always set.
        if proc is None or proc.returncode is None:
            raise RuntimeError("returncode must be set after communicate()")
        return proc.returncode, stdout_str, stderr_str

    async def _post_process_video(
        self,
        part_id: str,
        video_path: Path,
        resolved_config: dict,
        progress_callback: ProgressCallback | None = None,
    ) -> Path:
        """Convert a WebM recording to MP4 via FFmpeg when requested.

        Returns the final video path (MP4 on success, original otherwise).
        """
        output_format = resolved_config.get("output_format", self._output_format)
        if not (output_format == "mp4" and video_path.exists() and video_path.suffix == ".webm"):
            return video_path

        if progress_callback:
            await progress_callback(0.75, "converting_to_mp4")
        try:
            from src.services.ffmpeg import FFmpegOps

            preset_name = resolved_config.get("quality_preset", self._default_quality_preset)
            try:
                qp = get_quality_preset(preset_name)
                crf = qp.crf
            except ValueError:
                crf = self._ffmpeg_default_crf

            mp4_output = video_path.with_suffix(".mp4")
            mp4_path = await asyncio.to_thread(
                FFmpegOps.convert_to_mp4,
                video_path,
                mp4_output,
                crf,
            )
            video_path.unlink(missing_ok=True)
            logger.info("Converted to MP4: %s", mp4_path)
            return mp4_path
        except (FFmpegConversionError, FFmpegError, RuntimeError, OSError) as exc:
            logger.warning("MP4 conversion failed for %s, keeping WebM: %s", part_id, exc)
            return video_path

    async def _probe_and_return(
        self,
        part_id: str,
        video_path: Path | None,
        resolved_config: dict,
        returncode: int,
        accepted_with_warning: bool,
        elapsed: float,
    ) -> dict:
        """Extract ffprobe metadata and build the final result dict."""
        duration = 0.0
        file_size = 0
        width = resolved_config["viewport_width"]
        height = resolved_config["viewport_height"]
        has_audio = False

        if video_path and video_path.exists():
            file_size = video_path.stat().st_size
            try:
                from src.services.ffmpeg import FFmpegOps

                probe = await asyncio.to_thread(FFmpegOps.probe_all, video_path)
                duration = probe.duration
                if probe.width > 0 and probe.height > 0:
                    width = probe.width
                    height = probe.height
                has_audio = probe.has_audio
            except (FFmpegProbeError, FFmpegError, RuntimeError, OSError) as exc:
                logger.debug("ffprobe metadata fallback for %s: %s", part_id, exc)

        success = video_path is not None and (returncode == 0 or accepted_with_warning)
        logger.info(
            "Recording part %s %s in %.1fs (video=%.1fs, %d bytes)",
            part_id,
            "succeeded" if success else "failed",
            elapsed,
            duration,
            file_size,
        )
        return {
            "part_id": part_id,
            "success": success,
            "video_path": str(video_path) if video_path else None,
            "duration_seconds": duration,
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "has_audio": has_audio,
        }

    async def _handle_subprocess_timeout(
        self,
        part_id: str,
        proc: asyncio.subprocess.Process | None,
        scripts_path: Path,
    ) -> None:
        """Kill the timed-out process tree, then reap its partial webm output.

        The tree is killed *before* deleting the webm so the orphaned browser
        isn't still writing to a file we're removing.
        """
        if proc is not None:
            await self._terminate_process_tree(part_id, proc)
        self._running_processes.pop(part_id, None)
        partial_dir = scripts_path / "test-results"
        if partial_dir.exists():
            for partial in partial_dir.rglob("*.webm"):
                partial.unlink(missing_ok=True)

    @staticmethod
    def _signal_process_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        """Send *sig* to the subprocess's whole process group, best-effort.

        The subprocess is launched with ``start_new_session=True`` so ``proc``
        leads a process group containing the spawned node + Chromium children.
        Signalling the group reaches them all; signalling only ``proc`` would
        orphan the browser. Falls back to the single process if the group can't
        be resolved (e.g. it already exited).
        """
        if proc.returncode is not None:
            return
        pid = proc.pid
        # Mock-pid broadcast guard: under unit tests ``proc`` may be a mock whose
        # ``.pid`` is not a real int (a ``MagicMock`` coerces via ``__index__`` to
        # 1). ``os.getpgid(1)`` resolves init's group, so ``os.killpg(...)`` would
        # become ``kill(-1, sig)`` — a broadcast that signals the whole process
        # tree (e.g. the CI runner agent). Only signal a real child pid.
        if not isinstance(pid, int) or pid <= 1:
            return
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError, OSError:
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError, OSError:
            pass

    async def _terminate_process_tree(self, part_id: str, proc: asyncio.subprocess.Process) -> None:
        """Terminate the recorder process tree, escalating SIGTERM → SIGKILL.

        Reaps the leader after killing so it can't linger as a zombie.
        """
        if proc.returncode is not None:
            return
        self._signal_process_group(proc, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._cancel_wait_timeout_s)
            return
        except TimeoutError:
            logger.warning(
                "Recorder process group for %s ignored SIGTERM; sending SIGKILL", part_id
            )
        self._signal_process_group(proc, signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), timeout=self._cancel_wait_timeout_s)
        except TimeoutError:
            logger.error("Recorder process for %s did not exit after SIGKILL", part_id)

    async def cancel_recording(self, part_id: str) -> bool:
        """Cancel a running recording subprocess (and its browser child tree).

        Args:
            part_id: Part identifier whose subprocess to terminate.

        Returns:
            True if the process was found and terminated, False otherwise.
        """
        logger.info("Cancelling recording subprocess for part %s", part_id)
        proc = self._running_processes.pop(part_id, None)
        if proc and proc.returncode is None:
            await self._terminate_process_tree(part_id, proc)
            logger.info("Recording subprocess for part %s terminated", part_id)
            return True
        return False

    def _find_video(self, video_dir: Path, part_id: str) -> Path | None:
        """Find recorded video file in Playwright's test-results."""
        if not video_dir.exists():
            return None
        extensions = ("*.webm", "*.mp4")

        # Primary lookup: recorder writes to --output test-results/{part_id},
        # so prefer files under that exact directory regardless of nested
        # Playwright title-folder truncation behavior.
        part_output_dir = video_dir / part_id
        if part_output_dir.exists():
            candidates: list[Path] = []
            for ext in extensions:
                candidates.extend(part_output_dir.rglob(ext))
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)

        boundary_pattern = re.compile(
            r"(?:^|[-_/\\])" + re.escape(part_id) + r"(?:$|[-_/\\.])",
            re.IGNORECASE,
        )
        for ext in extensions:
            for video_file in video_dir.rglob(ext):
                if boundary_pattern.search(video_file.stem) or boundary_pattern.search(
                    video_file.parent.name
                ):
                    return video_file
        logger.warning("No video found for part %s in %s", part_id, video_dir)
        return None
