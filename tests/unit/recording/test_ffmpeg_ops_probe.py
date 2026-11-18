"""Tests for src/services/ffmpeg_ops.py -- _probe and probe_all methods."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.ffmpeg import FFmpegOps

# ── _probe ───────────────────────────────────────────────────────


class TestProbeRaw:
    def test_success(self):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 0
        mock_result.stdout = "  output data  "
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = FFmpegOps._probe(["-show_format", "test.mp4"])
            assert result == "output data"
            mock_run.assert_called_once()

    def test_timeout_raises(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffprobe", 10)):
            with pytest.raises(RuntimeError, match="timed out"):
                FFmpegOps._probe(["-show_format", "test.mp4"])

    def test_nonzero_returncode_raises(self):
        mock_result = MagicMock(spec=subprocess.CompletedProcess)
        mock_result.returncode = 1
        mock_result.stderr = "No such file"
        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                FFmpegOps._probe(["-show_format", "test.mp4"])

    def test_ffprobe_not_found_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe")):
            with pytest.raises(RuntimeError, match="ffprobe not found"):
                FFmpegOps._probe(["-show_format", "test.mp4"])


# ── probe_all ────────────────────────────────────────────────────


class TestProbeAll:
    def setup_method(self):
        """Clear the probe cache before each test."""
        FFmpegOps._probe_cache.clear()

    def test_parses_json_correctly(self):
        data = {
            "format": {"duration": "12.5"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio"},
            ],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 12.5
            assert result.width == 1920
            assert result.height == 1080
            assert result.has_audio is True
            assert result.probed is True

    def test_no_audio_stream(self):
        data = {
            "format": {"duration": "5.0"},
            "streams": [
                {"codec_type": "video", "width": 640, "height": 480},
            ],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.has_audio is False
            assert result.width == 640
            assert result.height == 480

    def test_no_duration(self):
        data = {
            "format": {},
            "streams": [{"codec_type": "video", "width": 320, "height": 240}],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 0.0

    def test_invalid_duration_string(self):
        data = {
            "format": {"duration": "N/A"},
            "streams": [],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 0.0

    def test_probe_failure_returns_defaults(self):
        mock_probe = MagicMock(side_effect=RuntimeError("ffprobe not found"))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 0.0
            assert result.width == 0
            assert result.height == 0
            assert result.has_audio is False
            # The defect this guards: a probe FAILURE (ffprobe crashed/timed
            # out/unparseable) must be distinguishable from a genuinely
            # 0-second file that probed successfully -- both currently land
            # on duration=0.0, so ``probed`` is the only signal a caller can
            # check before billing off that 0.0.
            assert result.probed is False

    def test_invalid_json_returns_defaults(self):
        mock_probe = MagicMock(return_value="not json at all")

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 0.0
            assert result.probed is False

    def test_genuinely_zero_second_file_is_probed_true(self):
        """A real file that ffprobe reads successfully but reports 0 duration
        (e.g. an empty/near-empty container) must NOT be confused with a probe
        failure: ``probed`` stays True, only ``duration`` is 0.0."""
        data = {
            "format": {"duration": "0.0"},
            "streams": [{"codec_type": "video", "width": 320, "height": 240}],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.duration == 0.0
            assert result.probed is True

    def test_probe_failure_is_not_cached(self):
        """A transient probe failure must not poison the cache: a later probe
        of the SAME (unchanged) path should re-run ffprobe, not replay the
        stale failure forever."""
        mock_probe = MagicMock(side_effect=RuntimeError("ffprobe timed out"))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            path = Path(f.name)
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result1 = FFmpegOps.probe_all(path)
                result2 = FFmpegOps.probe_all(path)
            assert result1.probed is False
            assert result2.probed is False
            # Not served from cache -- _probe was invoked on every call.
            assert mock_probe.call_count == 2

    def test_caches_result(self):
        data = {
            "format": {"duration": "3.0"},
            "streams": [{"codec_type": "video", "width": 800, "height": 600}],
        }
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            path = Path(f.name)
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result1 = FFmpegOps.probe_all(path)
                result2 = FFmpegOps.probe_all(path)
            assert result1 == result2
            # _probe should be called only once due to caching
            mock_probe.assert_called_once()

    def test_nonexistent_file_stat_fallback(self):
        """probe_all on a nonexistent file should still work (mtime defaults to 0.0)."""
        FFmpegOps._probe_cache.clear()
        mock_probe = MagicMock(side_effect=RuntimeError("no file"))
        with patch.object(FFmpegOps, "_probe", mock_probe):
            result = FFmpegOps.probe_all(Path("/nonexistent/file.mp4"))
        assert result.duration == 0.0

    def test_no_streams(self):
        data = {"format": {"duration": "2.0"}}
        mock_probe = MagicMock(return_value=json.dumps(data))

        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            with patch.object(FFmpegOps, "_probe", mock_probe):
                result = FFmpegOps.probe_all(Path(f.name))
            assert result.width == 0
            assert result.height == 0
            assert result.has_audio is False
