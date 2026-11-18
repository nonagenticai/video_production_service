"""Tests for src/services/ffmpeg_ops.py -- FFmpegOps probe methods and conversion."""

from __future__ import annotations

import subprocess
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.ffmpeg import FFmpegError, FFmpegOps, ProbeResult

# ── Constants ────────────────────────────────────────────────────


class TestFFmpegOpsConstants:
    def test_ffprobe_timeout_from_settings(self):
        from src.config import get_settings

        assert get_settings().ffmpeg.ffprobe_timeout == 10

    def test_probe_cache_is_dict(self):
        assert isinstance(FFmpegOps._probe_cache, dict)


# ── Delegation ───────────────────────────────────────────────────


class TestProbeDelegation:
    def setup_method(self):
        FFmpegOps._probe_cache.clear()

    def test_probe_duration_delegates(self):
        mock = MagicMock(
            return_value=ProbeResult(duration=10.5, width=1920, height=1080, has_audio=True)
        )
        with patch.object(FFmpegOps, "probe_all", mock):
            result = FFmpegOps.probe_duration(Path("/fake/video.mp4"))
            assert result == 10.5
            mock.assert_called_once()

    def test_probe_dimensions_delegates(self):
        mock = MagicMock(
            return_value=ProbeResult(duration=5.0, width=3840, height=2160, has_audio=False)
        )
        with patch.object(FFmpegOps, "probe_all", mock):
            w, h = FFmpegOps.probe_dimensions(Path("/fake/video.mp4"))
            assert w == 3840
            assert h == 2160

    def test_probe_has_audio_delegates(self):
        mock = MagicMock(
            return_value=ProbeResult(duration=5.0, width=1920, height=1080, has_audio=False)
        )
        with patch.object(FFmpegOps, "probe_all", mock):
            assert FFmpegOps.probe_has_audio(Path("/fake/video.mp4")) is False


# ── ProbeResult dataclass ────────────────────────────────────────


class TestProbeResult:
    def test_basic(self):
        result = ProbeResult(duration=5.0, width=1920, height=1080, has_audio=True)
        assert result.duration == 5.0
        assert result.width == 1920
        assert result.height == 1080
        assert result.has_audio is True

    def test_empty(self):
        result = ProbeResult(duration=0.0, width=0, height=0, has_audio=False)
        assert result.duration == 0.0
        assert result.has_audio is False


# ── convert_to_mp4 ──────────────────────────────────────────────


class TestConvertToMp4:
    def setup_method(self):
        FFmpegOps._probe_cache.clear()

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_successful_conversion_with_audio(self, mock_has_audio, mock_run):
        """Successful ffmpeg conversion returns the output path (input has audio)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm-data")
            dest = Path(tmpdir) / "video.mp4"

            result = FFmpegOps.convert_to_mp4(source, dest, crf=23)
            assert result == dest
            mock_run.assert_called_once()
            # Should NOT include anullsrc when audio is present
            call_args = mock_run.call_args[0][0]
            assert "anullsrc" not in " ".join(call_args)

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_successful_conversion_without_audio(self, mock_has_audio, mock_run):
        """Conversion adds silent audio track when input has no audio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm-data")
            dest = Path(tmpdir) / "video.mp4"

            result = FFmpegOps.convert_to_mp4(source, dest)
            assert result == dest
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "anullsrc" in " ".join(call_args)

    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_nonzero_exit_raises(self, mock_has_audio):
        """Non-zero ffmpeg exit code raises FFmpegError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm")
            dest = Path(tmpdir) / "video.mp4"

            mock_result = MagicMock(spec=subprocess.CompletedProcess)
            mock_result.returncode = 1
            mock_result.stderr = "encoding failed"

            with patch("subprocess.run", return_value=mock_result):
                with pytest.raises(FFmpegError):
                    FFmpegOps.convert_to_mp4(source, dest)

    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_ffmpeg_not_found_raises(self, mock_has_audio):
        """FileNotFoundError from subprocess raises FFmpegError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm")
            dest = Path(tmpdir) / "video.mp4"

            with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg")):
                with pytest.raises(FFmpegError, match="ffmpeg not found"):
                    FFmpegOps.convert_to_mp4(source, dest)

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_crf_parameter_passed(self, mock_has_audio, mock_run):
        """CRF parameter is included in the ffmpeg command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm")
            dest = Path(tmpdir) / "video.mp4"

            FFmpegOps.convert_to_mp4(source, dest, crf=18)

            call_args = mock_run.call_args[0][0]
            assert "-crf" in call_args
            crf_idx = call_args.index("-crf")
            assert call_args[crf_idx + 1] == "18"

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_output_dir_created(self, mock_has_audio, mock_run):
        """Output directory is created if it does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "video.webm"
            source.write_bytes(b"fake-webm")
            dest = Path(tmpdir) / "subdir" / "video.mp4"

            FFmpegOps.convert_to_mp4(source, dest)
            assert dest.parent.exists()


class TestRunConcurrencyGate:
    """The process-wide ffmpeg subprocess semaphore (OOM-under-load fix)."""

    def setup_method(self):
        # Save shared ClassVar gate state so a forced small cap here can't leak
        # into other tests.
        self._saved_max = FFmpegOps._max_concurrent_subprocesses
        self._saved_sem = FFmpegOps._run_semaphore

    def teardown_method(self):
        FFmpegOps._max_concurrent_subprocesses = self._saved_max
        FFmpegOps._run_semaphore = self._saved_sem

    def test_run_caps_concurrent_ffmpeg_subprocesses(self):
        """No more than _max_concurrent_subprocesses ffmpeg processes run at
        once, even when many compile-step threads call run() simultaneously."""
        FFmpegOps._max_concurrent_subprocesses = 2
        FFmpegOps._run_semaphore = None  # force a lazy rebuild at the new size

        active = 0
        peak = 0
        lock = threading.Lock()

        def fake_run(*_args, **_kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)  # hold the slot long enough to overlap
            with lock:
                active -= 1
            result = MagicMock(spec=subprocess.CompletedProcess)
            result.returncode = 0
            result.stderr = ""
            return result

        with patch("subprocess.run", side_effect=fake_run):
            threads = [
                threading.Thread(target=lambda: FFmpegOps.run(["-i", "in.mp4", "out.mp4"]))
                for _ in range(8)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert peak >= 1  # the work actually ran
        assert peak <= 2  # but never exceeded the configured ceiling

    def test_configure_from_settings_sizes_the_gate(self):
        from src.config import get_settings

        FFmpegOps.configure_from_settings()
        expected = get_settings().compilation.max_concurrent_ffmpeg_subprocesses
        assert FFmpegOps._max_concurrent_subprocesses == expected
        assert FFmpegOps._run_semaphore is not None
