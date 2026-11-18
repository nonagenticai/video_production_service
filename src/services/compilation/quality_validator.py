"""Video quality validation service."""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404
from dataclasses import dataclass, field
from pathlib import Path

from src.services.ffmpeg import FFmpegOps

logger = logging.getLogger(__name__)

_MAX_DURATION_SECONDS: float = 7200.0
_PEAK_CLIPPING_THRESHOLD_DB: float = -1.0
_SILENCE_THRESHOLD_DB: float = -50.0


@dataclass
class QualityReport:
    """Aggregated result of all quality checks on a video file."""

    valid: bool
    issues: list[dict] = field(default_factory=list)
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    resolution: str = ""
    audio_peak_db: float | None = None
    encoding_valid: bool = True


class QualityValidator:
    """Runs a battery of checks against a compiled video file and returns
    a structured :class:`QualityReport`.
    """

    def __init__(
        self,
        target_width: int = 1920,
        target_height: int = 1080,
    ) -> None:
        self._target_width = target_width
        self._target_height = target_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, video_path: Path) -> QualityReport:
        """Run every check and return a consolidated report."""
        issues: list[dict] = []
        report = QualityReport(valid=True, issues=issues)

        # --- file_exists ---
        if not self._check_file_exists(video_path, issues):
            report.valid = False
            return report

        # --- non_zero_size ---
        file_size = video_path.stat().st_size
        report.file_size_bytes = file_size
        if not self._check_non_zero_size(file_size, issues):
            report.valid = False
            return report

        # --- probe metadata (duration + resolution) ---
        probe = FFmpegOps.probe_all(video_path)
        # `probe_all` folds an ffprobe FAILURE (crash / non-zero exit / the 10s
        # FFPROBE_TIMEOUT elapsing under load) into a zero-valued result whose
        # only distinguishing mark is `probed=False`. Without this branch the
        # report still failed -- but as "video_stream_present", because width,
        # height and has_video are all zero. That is a measurement GAP wearing
        # an encoder bug's name, and it sends the operator after the wrong
        # component. Report the gap as itself.
        if not probe.probed:
            issues.append(
                {
                    "check": "probe_readable",
                    "severity": "error",
                    "message": (
                        "ffprobe could not read the output file; its duration, "
                        "resolution and stream layout are UNMEASURED (not zero)."
                    ),
                }
            )
            report.valid = False
            return report
        report.duration_seconds = probe.duration
        report.resolution = f"{probe.width}x{probe.height}" if probe.width and probe.height else ""

        # --- video_stream_present / video_has_frames ---
        # Without these two, an audio-only or single-frame mp4 scored
        # ``valid=True``: resolution + integrity findings are only
        # ``severity="warning"``, so nothing could ever fail the report.
        self._check_video_stream(probe, issues)

        # --- duration_valid ---
        self._check_duration_valid(probe.duration, issues)

        # --- resolution_matches ---
        self._check_resolution_matches(probe.width, probe.height, issues)

        # --- audio_levels ---
        if probe.has_audio:
            peak_db = self._check_audio_levels(video_path, issues)
            report.audio_peak_db = peak_db

        # --- encoding_integrity ---
        encoding_ok = self._check_encoding_integrity(video_path, issues)
        report.encoding_valid = encoding_ok

        # Mark the whole report invalid when any error-severity issue exists.
        if any(i["severity"] == "error" for i in issues):
            report.valid = False

        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_video_stream(probe: object, issues: list[dict]) -> bool:
        """Error when the file carries no video stream, or only a single frame.

        ``has_video`` is defaulted to ``False`` on :class:`ProbeResult` for
        backwards compatibility, so a non-zero frame geometry is accepted as
        proof of a video stream too. ``video_frames == 0`` means "ffprobe did
        not report ``nb_frames``", not "no frames" — only an explicit 1 is a
        failure.
        """
        has_video = bool(getattr(probe, "has_video", False)) or (
            int(getattr(probe, "width", 0) or 0) > 0 and int(getattr(probe, "height", 0) or 0) > 0
        )
        if not has_video:
            issues.append(
                {
                    "check": "video_stream_present",
                    "severity": "error",
                    "message": "File contains no video stream",
                }
            )
            return False

        frames = int(getattr(probe, "video_frames", 0) or 0)
        if 0 < frames <= 1:
            issues.append(
                {
                    "check": "video_has_frames",
                    "severity": "error",
                    "message": f"Video stream holds only {frames} frame(s)",
                }
            )
            return False
        return True

    @staticmethod
    def _check_file_exists(
        video_path: Path,
        issues: list[dict],
    ) -> bool:
        if not video_path.exists():
            issues.append(
                {
                    "check": "file_exists",
                    "severity": "error",
                    "message": f"File not found: {video_path}",
                }
            )
            return False
        issues.append(
            {
                "check": "file_exists",
                "severity": "info",
                "message": "File exists.",
            }
        )
        return True

    @staticmethod
    def _check_non_zero_size(
        file_size: int,
        issues: list[dict],
    ) -> bool:
        if file_size <= 0:
            issues.append(
                {
                    "check": "non_zero_size",
                    "severity": "error",
                    "message": "File size is 0 bytes.",
                }
            )
            return False
        issues.append(
            {
                "check": "non_zero_size",
                "severity": "info",
                "message": f"File size: {file_size} bytes.",
            }
        )
        return True

    @staticmethod
    def _check_duration_valid(
        duration: float,
        issues: list[dict],
    ) -> None:
        if duration <= 0:
            issues.append(
                {
                    "check": "duration_valid",
                    "severity": "error",
                    "message": "Duration is 0 or negative.",
                }
            )
        elif duration > _MAX_DURATION_SECONDS:
            issues.append(
                {
                    "check": "duration_valid",
                    "severity": "error",
                    "message": (
                        f"Duration {duration:.1f}s exceeds maximum {_MAX_DURATION_SECONDS:.0f}s."
                    ),
                }
            )
        else:
            issues.append(
                {
                    "check": "duration_valid",
                    "severity": "info",
                    "message": f"Duration: {duration:.2f}s.",
                }
            )

    def _check_resolution_matches(
        self,
        width: int,
        height: int,
        issues: list[dict],
    ) -> None:
        if width == 0 or height == 0:
            issues.append(
                {
                    "check": "resolution_matches",
                    "severity": "warning",
                    "message": "Could not determine video resolution.",
                }
            )
            return

        if width != self._target_width or height != self._target_height:
            issues.append(
                {
                    "check": "resolution_matches",
                    "severity": "warning",
                    "message": (
                        f"Resolution {width}x{height} does not match "
                        f"target {self._target_width}x{self._target_height}."
                    ),
                }
            )
        else:
            issues.append(
                {
                    "check": "resolution_matches",
                    "severity": "info",
                    "message": (
                        f"Resolution matches target {self._target_width}x{self._target_height}."
                    ),
                }
            )

    @staticmethod
    def _check_audio_levels(
        video_path: Path,
        issues: list[dict],
    ) -> float | None:
        """Use FFmpeg volumedetect to measure peak and mean audio levels.

        Returns the parsed max_volume in dB, or ``None`` on failure.
        """
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(  # nosec B603  # noqa: S603 -- ffmpeg volumedetect with controlled args
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            logger.warning("volumedetect timed out for %s", video_path)
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "warning",
                    "message": "Audio level detection timed out.",
                }
            )
            return None
        except Exception:
            logger.exception("volumedetect failed for %s", video_path)
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "warning",
                    "message": "Audio level detection failed unexpectedly.",
                }
            )
            return None

        max_vol = _parse_volume(stderr, "max_volume")
        mean_vol = _parse_volume(stderr, "mean_volume")

        if max_vol is None:
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "warning",
                    "message": "Could not parse max_volume from volumedetect.",
                }
            )
            return None

        if max_vol > _PEAK_CLIPPING_THRESHOLD_DB:
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "warning",
                    "message": (
                        f"Peak audio {max_vol:.1f} dB exceeds "
                        f"{_PEAK_CLIPPING_THRESHOLD_DB} dB -- possible clipping."
                    ),
                }
            )
        else:
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "info",
                    "message": f"Peak audio: {max_vol:.1f} dB.",
                }
            )

        if mean_vol is not None and mean_vol < _SILENCE_THRESHOLD_DB:
            issues.append(
                {
                    "check": "audio_levels",
                    "severity": "warning",
                    "message": (
                        f"Mean volume {mean_vol:.1f} dB is below "
                        f"{_SILENCE_THRESHOLD_DB} dB -- audio may be silent."
                    ),
                }
            )

        return max_vol

    @staticmethod
    def _check_encoding_integrity(
        video_path: Path,
        issues: list[dict],
    ) -> bool:
        """Decode the entire file to /dev/null and look for errors."""
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(  # nosec B603  # noqa: S603 -- ffmpeg encoding integrity check
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "Encoding integrity check timed out for %s",
                video_path,
            )
            issues.append(
                {
                    "check": "encoding_integrity",
                    "severity": "warning",
                    "message": "Encoding integrity check timed out.",
                }
            )
            return False
        except Exception:
            logger.exception(
                "Encoding integrity check failed for %s",
                video_path,
            )
            issues.append(
                {
                    "check": "encoding_integrity",
                    "severity": "error",
                    "message": "Encoding integrity check raised an exception.",
                }
            )
            return False

        if result.returncode != 0:
            stderr_snippet = result.stderr[:500] if result.stderr else ""
            issues.append(
                {
                    "check": "encoding_integrity",
                    "severity": "error",
                    "message": (
                        f"Decode errors detected (rc={result.returncode}): {stderr_snippet}"
                    ),
                }
            )
            return False

        issues.append(
            {
                "check": "encoding_integrity",
                "severity": "info",
                "message": "No decode errors found.",
            }
        )
        return True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_volume(stderr: str, key: str) -> float | None:
    """Extract a dB value from FFmpeg volumedetect output.

    Example line::

        [Parsed_volumedetect_0 ...] max_volume: -3.2 dB
    """
    pattern = rf"{re.escape(key)}\s*:\s*(-?[\d.]+)\s*dB"
    match = re.search(pattern, stderr)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
