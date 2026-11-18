"""Tests for src/services/playwright_recorder.py -- PlaywrightRecorder."""

from __future__ import annotations

import asyncio
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.recording.playwright_recorder import PlaywrightRecorder

# ── Helpers ────────────────────────────────────────────────────────


def _mock_settings():
    """Return a MagicMock settings object for PlaywrightRecorder init."""
    return MagicMock(
        recording=MagicMock(
            viewport_width=1920,
            viewport_height=1080,
            timeout_ms=60000,
            timeout_buffer_s=30,
            min_video_size_bytes=10000,
            slow_mo_ms=0,
            color_scheme="light",
            headless=True,
            output_format="mp4",
            default_quality_preset="standard",
            error_output_max_length=500,
            cancel_wait_timeout_s=5.0,
        ),
        ffmpeg=MagicMock(
            preset="medium",
            timeout=300,
            default_crf=23,
        ),
    )


# ── _find_video ──────────────────────────────────────────────────


class TestFindVideo:
    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_no_dir(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        result = recorder._find_video(Path("/nonexistent/dir"), "part1")
        assert result is None

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_matching_video(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = Path(tmpdir) / "test-results" / "part1-test"
            video_dir.mkdir(parents=True)
            video_file = video_dir / "part1-recording.webm"
            video_file.write_bytes(b"fake-video-data")
            result = recorder._find_video(Path(tmpdir) / "test-results", "part1")
            assert result is not None
            assert result.suffix == ".webm"

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_no_match(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            video_dir = Path(tmpdir) / "test-results"
            video_dir.mkdir(parents=True)
            # Create a video that does NOT match part_id "partX"
            other_dir = video_dir / "other-test"
            other_dir.mkdir()
            (other_dir / "other-recording.webm").write_bytes(b"data")
            result = recorder._find_video(video_dir, "partX")
            assert result is None

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_finds_video_under_exact_part_output_dir(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            part_id = "ch01-part1-webapp-commercial-v1"
            video_dir = Path(tmpdir) / "test-results" / part_id
            nested = video_dir / "truncated-playwright-folder"
            nested.mkdir(parents=True)
            video_file = nested / "video.webm"
            video_file.write_bytes(b"fake-video-data")

            result = recorder._find_video(Path(tmpdir) / "test-results", part_id)
            assert result == video_file


# ── record_part ──────────────────────────────────────────────────


class TestRecordPart:
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_missing_script(self, mock_gs):
        settings = _mock_settings()
        mock_gs.return_value = settings
        recorder = PlaywrightRecorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            result = await recorder.record_part("nonexistent_part", tmpdir)
            assert result["success"] is False
            assert "Script file not found" in result["error"]

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_path_traversal_rejected(self, mock_gs):
        settings = _mock_settings()
        settings.recording.scripts_base_dir = "/app/scripts"
        mock_gs.return_value = settings
        recorder = PlaywrightRecorder()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await recorder.record_part("part1", tmpdir)
            assert result["success"] is False
            assert "scripts_dir must be within" in result["error"]


# ── _resolve_recording_config ────────────────────────────────────


class TestResolveRecordingConfig:
    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_defaults_without_overrides(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config(None)
        # Default quality_preset is "standard" -> 1920x1080
        assert config["viewport_width"] == 1920
        assert config["viewport_height"] == 1080
        assert config["headless"] is True
        assert config["output_format"] == "mp4"
        assert config["quality_preset"] == "standard"

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_merges_overrides(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config({"viewport_width": 800, "headless": False})
        assert config["viewport_width"] == 800
        assert config["headless"] is False
        # Non-overridden values stay at preset-derived defaults
        assert config["viewport_height"] == 1080

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_quality_preset_preview(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config({"quality_preset": "preview"})
        assert config["viewport_width"] == 1280
        assert config["viewport_height"] == 720
        assert config["quality_preset"] == "preview"

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_quality_preset_4k(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config({"quality_preset": "4k"})
        assert config["viewport_width"] == 3840
        assert config["viewport_height"] == 2160

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_explicit_viewport_overrides_preset(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config(
            {"quality_preset": "4k", "viewport_width": 2560}
        )
        # Explicit viewport_width takes precedence over preset
        assert config["viewport_width"] == 2560
        # viewport_height still comes from 4k preset
        assert config["viewport_height"] == 2160

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_output_format_override(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        config = recorder._resolve_recording_config({"output_format": "webm"})
        assert config["output_format"] == "webm"

    @patch("src.services.recording.playwright_recorder.get_settings")
    def test_unknown_preset_falls_back(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()
        # Unknown preset should log a warning and use defaults
        config = recorder._resolve_recording_config({"quality_preset": "nonexistent"})
        # Falls back to instance defaults (1920x1080 from _mock_settings)
        assert config["viewport_width"] == 1920
        assert config["viewport_height"] == 1080


# ── cancel_recording ─────────────────────────────────────────────


class TestCancelRecording:
    @patch("src.services.recording.playwright_recorder.os.killpg")
    @patch("src.services.recording.playwright_recorder.os.getpgid", return_value=4242)
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_terminates_running_process(self, mock_gs, _mock_getpgid, mock_killpg):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()

        mock_proc = AsyncMock()
        mock_proc.returncode = None  # still running
        mock_proc.pid = 1234
        mock_proc.wait = AsyncMock()
        recorder._running_processes["part1"] = mock_proc

        result = await recorder.cancel_recording("part1")
        assert result is True
        # SIGTERM sent to the whole process group, not just the npx parent.
        mock_killpg.assert_called_once_with(4242, signal.SIGTERM)

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_no_process_returns_false(self, mock_gs):
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()

        result = await recorder.cancel_recording("part-not-running")
        assert result is False

    @patch("src.services.recording.playwright_recorder.os.killpg")
    @patch("src.services.recording.playwright_recorder.os.getpgid", return_value=4242)
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_cancel_kills_after_wait_timeout(self, mock_gs, _mock_getpgid, mock_killpg):
        """SIGTERM wait times out -> escalate to SIGKILL on the group."""
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.pid = 1234
        # Make wait raise TimeoutError so we hit the SIGKILL escalation branch.
        mock_proc.wait = AsyncMock(side_effect=TimeoutError)
        recorder._running_processes["part1"] = mock_proc

        result = await recorder.cancel_recording("part1")
        assert result is True
        assert mock_killpg.call_args_list == [
            ((4242, signal.SIGTERM),),
            ((4242, signal.SIGKILL),),
        ]

    @patch("src.services.recording.playwright_recorder.os.killpg", side_effect=ProcessLookupError)
    @patch("src.services.recording.playwright_recorder.os.getpgid", return_value=4242)
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_cancel_process_lookup_error(self, mock_gs, _mock_getpgid, _mock_killpg):
        """A process that vanished mid-kill is swallowed, not raised."""
        mock_gs.return_value = _mock_settings()
        recorder = PlaywrightRecorder()

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.pid = 1234
        mock_proc.wait = AsyncMock()
        recorder._running_processes["part1"] = mock_proc

        result = await recorder.cancel_recording("part1")
        assert result is True

    @patch("src.services.recording.playwright_recorder.os.killpg")
    @patch("src.services.recording.playwright_recorder.os.getpgid")
    async def test_signal_process_group_guards_mock_pid(self, mock_getpgid, mock_killpg):
        """Regression: a mocked ``proc.pid`` (not a real int) must NOT reach
        ``os.getpgid``/``os.killpg``.

        Without the guard, a ``MagicMock`` ``proc.pid`` coerces via ``__index__``
        to 1, so ``os.killpg(os.getpgid(1), sig)`` becomes ``kill(-1, sig)`` — a
        broadcast that SIGKILLs the whole process tree (the CI-runner self-kill
        class fixed in coding-security #80). ``returncode`` is None so we get
        past the early-out and actually exercise the pid guard.
        """
        proc = AsyncMock()
        proc.returncode = None  # still running: passes the returncode early-out
        # proc.pid is left as the default mock attribute (would coerce to 1).
        PlaywrightRecorder._signal_process_group(proc, signal.SIGTERM)
        mock_getpgid.assert_not_called()
        mock_killpg.assert_not_called()


# ── record_part: invalid scripts_dir ────────────────────────────


class TestRecordPartInvalidScriptsDir:
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_invalid_scripts_dir_oserror(self, mock_gs):
        """Cover lines 108-109: OSError when resolving scripts_dir."""
        settings = _mock_settings()
        settings.recording.scripts_base_dir = "/app/scripts"
        mock_gs.return_value = settings
        recorder = PlaywrightRecorder()

        # Patch Path.resolve to raise OSError
        with patch.object(Path, "resolve", side_effect=OSError("bad path")):
            result = await recorder.record_part("part1", "/some/path")
        assert result["success"] is False
        assert "Invalid scripts directory" in result["error"]


# ── record_part: script file search ─────────────────────────────


class TestRecordPartScriptSearch:
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_finds_demo_spec_ts(self, mock_gs):
        """Cover lines 128-129: finds .demo.spec.ts script."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            script = parts_dir / "p1.demo.spec.ts"
            script.write_text("test")

            recorder = PlaywrightRecorder()

            # Mock subprocess to avoid actually running npx
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part("p1", tmpdir)
            # The test should have found the script (no "Script file not found")
            assert "Script file not found" not in result.get("error", "")


# ── record_part: subprocess execution ────────────────────────────


class TestRecordPartSubprocess:
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_successful_recording_with_webm_output(self, mock_gs):
        """Cover lines 138-306 happy path: subprocess success, video found, probe."""
        settings = _mock_settings()
        settings.recording.output_format = "webm"
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            # Create script file
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            # Create test-results video directory so _find_video works
            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"passed", b""))

            callback = AsyncMock()

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part(
                    "part1",
                    tmpdir,
                    recording_config={"output_format": "webm"},
                    auth_config={"env_vars": {"TOKEN": "abc"}},
                    progress_callback=callback,
                )

            assert result["success"] is True
            assert result["part_id"] == "part1"
            assert result["file_size_bytes"] > 0
            # Progress callback should have been called multiple times
            assert callback.await_count >= 3

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_successful_recording_with_mp4_conversion(self, mock_gs):
        """Cover MP4 conversion path (lines 252-284)."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            # Create mp4 file that ffmpeg "produces"
            mp4_path = video_file.with_suffix(".mp4")
            mp4_path.write_bytes(b"mp4data" * 100)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"passed", b""))

            mock_convert = MagicMock(return_value=mp4_path)
            mock_probe_result = MagicMock()
            mock_probe_result.duration = 15.0
            mock_probe_result.width = 1920
            mock_probe_result.height = 1080
            mock_probe_result.has_audio = True

            with (
                patch("asyncio.create_subprocess_exec", return_value=mock_proc),
                patch("src.services.ffmpeg.ops.FFmpegOps.convert_to_mp4", mock_convert),
                patch(
                    "src.services.ffmpeg.ops.FFmpegOps.probe_all",
                    return_value=mock_probe_result,
                ),
            ):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is True
            assert result["duration_seconds"] == 15.0
            assert result["has_audio"] is True
            mock_convert.assert_called_once()

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_mp4_conversion_failure_keeps_webm(self, mock_gs):
        """Cover line 284: MP4 conversion fails, keep original WebM."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"passed", b""))

            with (
                patch("asyncio.create_subprocess_exec", return_value=mock_proc),
                patch(
                    "src.services.ffmpeg.ops.FFmpegOps.convert_to_mp4",
                    side_effect=RuntimeError("ffmpeg crashed"),
                ),
            ):
                result = await recorder.record_part("part1", tmpdir)

            # Should still succeed with the WebM
            assert result["success"] is True
            assert result["video_path"] is not None

    @patch("src.services.recording.playwright_recorder.os.killpg")
    @patch("src.services.recording.playwright_recorder.os.getpgid", return_value=4242)
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_recording_timeout(self, mock_gs, _mock_getpgid, mock_killpg):
        """Cover lines 188-200: subprocess times out (and its tree is killed)."""
        settings = _mock_settings()
        settings.recording.timeout_ms = 1000
        settings.recording.timeout_buffer_s = 0
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 1234
            mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is False
            assert "timed out" in result["error"]
            # The orphan-prone browser tree is signalled on timeout.
            mock_killpg.assert_any_call(4242, signal.SIGTERM)

    @patch("src.services.recording.playwright_recorder.os.killpg")
    @patch("src.services.recording.playwright_recorder.os.getpgid", return_value=4242)
    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_recording_timeout_cleans_partial_videos(self, mock_gs, _mock_getpgid, _mock_kp):
        """Cover lines 192-195: partial video cleanup on timeout."""
        settings = _mock_settings()
        settings.recording.timeout_ms = 1000
        settings.recording.timeout_buffer_s = 0
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            # Create partial video that should be cleaned up
            results_dir = Path(tmpdir) / "test-results" / "partial"
            results_dir.mkdir(parents=True)
            partial_video = results_dir / "partial-video.webm"
            partial_video.write_bytes(b"partial")

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = None
            mock_proc.pid = 1234
            mock_proc.communicate = AsyncMock(side_effect=TimeoutError)

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is False
            # Partial video should have been cleaned up
            assert not partial_video.exists()

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_recording_cancelled(self, mock_gs):
        """Cover lines 201-207: asyncio.CancelledError."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is False
            assert "cancelled" in result["error"]

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_subprocess_failed_no_video(self, mock_gs):
        """Cover lines 224-229: non-zero exit, no video found."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: browser crashed"))

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is False
            assert "Recording failed" in result["error"]

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_subprocess_failed_video_too_small(self, mock_gs):
        """Cover lines 233-241: non-zero exit, video too small."""
        settings = _mock_settings()
        settings.recording.min_video_size_bytes = 10000
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            # Create a small video
            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"tiny")  # < 10000 bytes

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"warning"))

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part(
                    "part1", tmpdir, recording_config={"output_format": "webm"}
                )

            assert result["success"] is False
            assert "video too small" in result["error"]

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_subprocess_failed_video_large_enough(self, mock_gs):
        """Cover lines 233-248: non-zero exit but video is usable."""
        settings = _mock_settings()
        settings.recording.min_video_size_bytes = 100
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"warning"))

            with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
                result = await recorder.record_part(
                    "part1", tmpdir, recording_config={"output_format": "webm"}
                )

            # Accepted with warning since video is large enough
            assert result["success"] is True
            assert result["video_path"] is not None

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_probe_failure_uses_defaults(self, mock_gs):
        """Cover lines 303-304: ffprobe fails, fallback to config dimensions."""
        settings = _mock_settings()
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

            with (
                patch("asyncio.create_subprocess_exec", return_value=mock_proc),
                patch(
                    "src.services.ffmpeg.ops.FFmpegOps.convert_to_mp4",
                    side_effect=RuntimeError("no ffmpeg"),
                ),
            ):
                result = await recorder.record_part(
                    "part1", tmpdir, recording_config={"output_format": "webm"}
                )

            assert result["success"] is True
            # Width/height should come from resolved config defaults
            assert result["width"] == 1920
            assert result["height"] == 1080

    @patch("src.services.recording.playwright_recorder.get_settings")
    async def test_mp4_conversion_with_unknown_preset_fallback(self, mock_gs):
        """Cover lines 268-269: unknown preset during conversion, fallback crf=23."""
        settings = _mock_settings()
        settings.recording.default_quality_preset = "standard"
        mock_gs.return_value = settings

        with tempfile.TemporaryDirectory() as tmpdir:
            settings.recording.scripts_base_dir = tmpdir
            parts_dir = Path(tmpdir) / "parts"
            parts_dir.mkdir()
            (parts_dir / "part1.ts").write_text("test")

            results_dir = Path(tmpdir) / "test-results" / "part1-test"
            results_dir.mkdir(parents=True)
            video_file = results_dir / "part1-video.webm"
            video_file.write_bytes(b"x" * 20000)

            mp4_path = video_file.with_suffix(".mp4")
            mp4_path.write_bytes(b"mp4" * 100)

            recorder = PlaywrightRecorder()

            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

            mock_convert = MagicMock(return_value=mp4_path)

            # Make quality_preset unknown in the resolved config
            with (
                patch("asyncio.create_subprocess_exec", return_value=mock_proc),
                patch("src.services.ffmpeg.ops.FFmpegOps.convert_to_mp4", mock_convert),
                patch(
                    "src.services.ffmpeg.ops.FFmpegOps.probe_all",
                    return_value=MagicMock(duration=5.0, width=1920, height=1080, has_audio=False),
                ),
                patch.object(
                    recorder,
                    "_resolve_recording_config",
                    return_value={
                        "viewport_width": 1920,
                        "viewport_height": 1080,
                        "timeout_ms": 60000,
                        "slow_mo_ms": 0,
                        "color_scheme": "light",
                        "headless": True,
                        "output_format": "mp4",
                        "quality_preset": "nonexistent_preset",
                    },
                ),
            ):
                result = await recorder.record_part("part1", tmpdir)

            assert result["success"] is True
            # Verify convert_to_mp4 was called with fallback crf=23
            # Args are positional: (input_path, output_path, crf)
            call_args = mock_convert.call_args[0]
            assert call_args[2] == 23
