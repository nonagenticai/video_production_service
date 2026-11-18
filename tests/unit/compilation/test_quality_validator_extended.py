"""Extended tests for quality_validator covering uncovered lines.

Covers:
  - Error-severity issues mark report invalid (line 89)
  - Resolution 0x0 warning (lines 182-189)
  - Audio level detection general exception (lines 250-259)
  - No max_volume parsed returns warning (lines 265-272)
  - Encoding integrity timeout (lines 331-343)
  - Encoding integrity general exception (lines 344-356)
  - _parse_volume ValueError branch (lines 398-399)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

_PATCH_PROBE_ALL = "src.services.compilation.quality_validator.FFmpegOps.probe_all"
_PATCH_SUBPROCESS_RUN = "src.services.compilation.quality_validator.subprocess.run"


def _make_probe_result(*, duration=60.0, width=1920, height=1080, has_audio=True) -> MagicMock:
    result = MagicMock()
    result.duration = duration
    result.width = width
    result.height = height
    result.has_audio = has_audio
    # Mirror the real ProbeResult: a bare MagicMock returns 1 from
    # __int__, which the video_has_frames check reads as a 1-frame file.
    result.has_video = True
    result.video_frames = 1800
    return result


class TestValidateErrorSeverity:
    """Test that error-severity issues in any check mark the report invalid (line 89)."""

    @patch(_PATCH_SUBPROCESS_RUN)
    @patch(_PATCH_PROBE_ALL)
    def test_error_severity_makes_report_invalid(self, mock_probe_all, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        # Duration 0 creates an error-severity issue
        mock_probe_all.return_value = _make_probe_result(duration=0.0, has_audio=False)

        # Encoding check succeeds
        enc_result = MagicMock()
        enc_result.returncode = 0
        enc_result.stderr = ""
        mock_subprocess_run.return_value = enc_result

        validator = QualityValidator()
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.stat.return_value = MagicMock(st_size=1000)

        report = validator.validate(fake_path)
        assert report.valid is False
        assert any(i["severity"] == "error" for i in report.issues)


class TestCheckResolutionZero:
    """Test resolution 0x0 warning (lines 182-189)."""

    def test_zero_resolution_returns_warning(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator(target_width=1920, target_height=1080)
        issues: list[dict] = []
        validator._check_resolution_matches(0, 0, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert issues[0]["check"] == "resolution_matches"
        assert "Could not determine" in issues[0]["message"]

    def test_zero_width_returns_warning(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator(target_width=1920, target_height=1080)
        issues: list[dict] = []
        validator._check_resolution_matches(0, 1080, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"


class TestCheckAudioLevelsExceptions:
    """Test audio level detection edge cases (lines 250-259, 265-272)."""

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_general_exception_returns_warning(self, mock_subprocess_run):
        """A non-timeout exception returns a warning and None."""
        from src.services.compilation.quality_validator import QualityValidator

        mock_subprocess_run.side_effect = OSError("permission denied")

        issues: list[dict] = []
        peak = QualityValidator._check_audio_levels(Path("/fake/video.mp4"), issues)
        assert peak is None
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "failed unexpectedly" in issues[0]["message"]

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_no_max_volume_parsed_returns_warning(self, mock_subprocess_run):
        """When max_volume can't be parsed, returns warning and None."""
        from src.services.compilation.quality_validator import QualityValidator

        result = MagicMock()
        result.stderr = "some output without volume info"
        mock_subprocess_run.return_value = result

        issues: list[dict] = []
        peak = QualityValidator._check_audio_levels(Path("/fake/video.mp4"), issues)
        assert peak is None
        assert any(
            i["check"] == "audio_levels" and "Could not parse" in i["message"] for i in issues
        )


class TestCheckEncodingIntegrityEdgeCases:
    """Test encoding integrity edge cases (lines 331-356)."""

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_timeout_returns_warning_and_false(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        issues: list[dict] = []
        ok = QualityValidator._check_encoding_integrity(Path("/fake/video.mp4"), issues)
        assert ok is False
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "timed out" in issues[0]["message"]

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_general_exception_returns_error_and_false(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        mock_subprocess_run.side_effect = OSError("disk failure")

        issues: list[dict] = []
        ok = QualityValidator._check_encoding_integrity(Path("/fake/video.mp4"), issues)
        assert ok is False
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert "exception" in issues[0]["message"].lower()


class TestParseVolumeEdgeCases:
    """Test _parse_volume ValueError branch (lines 398-399)."""

    def test_invalid_float_from_regex_match(self):
        """When the regex matches but float conversion fails (e.g. '1.2.3')."""
        from src.services.compilation.quality_validator import _parse_volume

        # The regex (-?[\d.]+) matches "1.2.3" which fails float()
        stderr = "[Parsed_volumedetect_0] max_volume: 1.2.3 dB\n"
        result = _parse_volume(stderr, "max_volume")
        # The regex actually matches "1.2.3" as the group, float("1.2.3") raises ValueError
        assert result is None

    def test_dots_only_fails_float(self):
        r"""Regex matches '...' as [\d.]+ but float('...') raises ValueError."""
        from src.services.compilation.quality_validator import _parse_volume

        stderr = "[Parsed_volumedetect_0] max_volume: ... dB\n"
        result = _parse_volume(stderr, "max_volume")
        assert result is None

    def test_parse_positive_volume(self):
        from src.services.compilation.quality_validator import _parse_volume

        stderr = "[Parsed_volumedetect_0] max_volume: 0.0 dB\n"
        result = _parse_volume(stderr, "max_volume")
        assert result == 0.0
