"""Unit tests for video quality validation service."""

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


class TestQualityReportDefaults:
    def test_dataclass_defaults(self):
        from src.services.compilation.quality_validator import QualityReport

        report = QualityReport(valid=True)
        assert report.valid is True
        assert report.issues == []
        assert report.file_size_bytes == 0
        assert report.duration_seconds == 0.0
        assert report.resolution == ""
        assert report.audio_peak_db is None
        assert report.encoding_valid is True

    def test_dataclass_with_values(self):
        from src.services.compilation.quality_validator import QualityReport

        report = QualityReport(
            valid=False,
            issues=[{"check": "test", "severity": "error", "message": "fail"}],
            file_size_bytes=1024,
            duration_seconds=30.0,
            resolution="1920x1080",
            audio_peak_db=-3.0,
            encoding_valid=False,
        )
        assert report.valid is False
        assert len(report.issues) == 1
        assert report.file_size_bytes == 1024
        assert report.audio_peak_db == -3.0


class TestQualityValidate:
    def test_file_not_found_returns_invalid_report(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator()
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = False

        report = validator.validate(fake_path)
        assert report.valid is False
        assert any(i["check"] == "file_exists" and i["severity"] == "error" for i in report.issues)

    def test_zero_byte_file_returns_invalid(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator()
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.stat.return_value = MagicMock(st_size=0)

        report = validator.validate(fake_path)
        assert report.valid is False
        assert any(
            i["check"] == "non_zero_size" and i["severity"] == "error" for i in report.issues
        )

    @patch(_PATCH_SUBPROCESS_RUN)
    @patch(_PATCH_PROBE_ALL)
    def test_valid_file_happy_path(self, mock_probe_all, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        mock_probe_all.return_value = _make_probe_result()

        # volumedetect subprocess call
        vol_result = MagicMock()
        vol_result.stderr = (
            "[Parsed_volumedetect_0] max_volume: -5.0 dB\n"
            "[Parsed_volumedetect_0] mean_volume: -20.0 dB\n"
        )

        # encoding integrity subprocess call
        enc_result = MagicMock()
        enc_result.returncode = 0
        enc_result.stderr = ""

        mock_subprocess_run.side_effect = [vol_result, enc_result]

        validator = QualityValidator()
        fake_path = MagicMock(spec=Path)
        fake_path.exists.return_value = True
        fake_path.stat.return_value = MagicMock(st_size=5000000)

        report = validator.validate(fake_path)
        assert report.valid is True
        assert report.file_size_bytes == 5000000
        assert report.duration_seconds == 60.0
        assert report.resolution == "1920x1080"
        assert report.encoding_valid is True


class TestCheckDurationValid:
    def test_zero_duration_returns_error(self):
        from src.services.compilation.quality_validator import QualityValidator

        issues: list[dict] = []
        QualityValidator._check_duration_valid(0.0, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert issues[0]["check"] == "duration_valid"

    def test_exceeds_max_duration_returns_error(self):
        from src.services.compilation.quality_validator import QualityValidator

        issues: list[dict] = []
        QualityValidator._check_duration_valid(7201.0, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "error"
        assert "exceeds" in issues[0]["message"]

    def test_valid_duration_returns_info(self):
        from src.services.compilation.quality_validator import QualityValidator

        issues: list[dict] = []
        QualityValidator._check_duration_valid(120.0, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "info"
        assert issues[0]["check"] == "duration_valid"


class TestCheckResolutionMatches:
    def test_matching_resolution_returns_info(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator(target_width=1920, target_height=1080)
        issues: list[dict] = []
        validator._check_resolution_matches(1920, 1080, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "info"
        assert "matches" in issues[0]["message"]

    def test_mismatched_resolution_returns_warning(self):
        from src.services.compilation.quality_validator import QualityValidator

        validator = QualityValidator(target_width=1920, target_height=1080)
        issues: list[dict] = []
        validator._check_resolution_matches(1280, 720, issues)
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "does not match" in issues[0]["message"]


class TestCheckAudioLevels:
    @patch(_PATCH_SUBPROCESS_RUN)
    def test_clipping_detected(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        result = MagicMock()
        result.stderr = (
            "[Parsed_volumedetect_0] max_volume: 0.0 dB\n"
            "[Parsed_volumedetect_0] mean_volume: -10.0 dB\n"
        )
        mock_subprocess_run.return_value = result

        issues: list[dict] = []
        peak = QualityValidator._check_audio_levels(Path("/fake/video.mp4"), issues)
        assert peak == 0.0
        assert any(
            i["check"] == "audio_levels"
            and i["severity"] == "warning"
            and "clipping" in i["message"]
            for i in issues
        )

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_silence_detected(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        result = MagicMock()
        result.stderr = (
            "[Parsed_volumedetect_0] max_volume: -10.0 dB\n"
            "[Parsed_volumedetect_0] mean_volume: -55.0 dB\n"
        )
        mock_subprocess_run.return_value = result

        issues: list[dict] = []
        peak = QualityValidator._check_audio_levels(Path("/fake/video.mp4"), issues)
        assert peak == -10.0
        assert any(i["check"] == "audio_levels" and "silent" in i["message"] for i in issues)

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_timeout_returns_warning(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

        issues: list[dict] = []
        peak = QualityValidator._check_audio_levels(Path("/fake/video.mp4"), issues)
        assert peak is None
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"
        assert "timed out" in issues[0]["message"]


class TestCheckEncodingIntegrity:
    @patch(_PATCH_SUBPROCESS_RUN)
    def test_valid_encoding_returns_true(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        mock_subprocess_run.return_value = result

        issues: list[dict] = []
        ok = QualityValidator._check_encoding_integrity(Path("/fake/video.mp4"), issues)
        assert ok is True
        assert any(i["check"] == "encoding_integrity" and i["severity"] == "info" for i in issues)

    @patch(_PATCH_SUBPROCESS_RUN)
    def test_corrupt_encoding_returns_false(self, mock_subprocess_run):
        from src.services.compilation.quality_validator import QualityValidator

        result = MagicMock()
        result.returncode = 1
        result.stderr = "decode error at frame 100"
        mock_subprocess_run.return_value = result

        issues: list[dict] = []
        ok = QualityValidator._check_encoding_integrity(Path("/fake/video.mp4"), issues)
        assert ok is False
        assert any(i["check"] == "encoding_integrity" and i["severity"] == "error" for i in issues)


class TestParseVolume:
    def test_parse_max_volume(self):
        from src.services.compilation.quality_validator import _parse_volume

        stderr = "[Parsed_volumedetect_0] max_volume: -3.2 dB\n"
        result = _parse_volume(stderr, "max_volume")
        assert result == -3.2

    def test_parse_mean_volume(self):
        from src.services.compilation.quality_validator import _parse_volume

        stderr = "[Parsed_volumedetect_0] mean_volume: -25.7 dB\n"
        result = _parse_volume(stderr, "mean_volume")
        assert result == -25.7

    def test_missing_key_returns_none(self):
        from src.services.compilation.quality_validator import _parse_volume

        stderr = "[Parsed_volumedetect_0] max_volume: -3.2 dB\n"
        result = _parse_volume(stderr, "nonexistent_key")
        assert result is None
