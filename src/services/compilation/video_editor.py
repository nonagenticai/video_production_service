"""Per-part video editing (speed adjust, trimming, dead time removal)."""

from __future__ import annotations

import contextlib
import logging
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any

from src.services.ffmpeg import FFmpegOps

logger = logging.getLogger(__name__)


class VideoEditor:
    """Edit individual recording segments before compilation."""

    def __init__(self, output_dir: Path | None = None, crf: int = 20) -> None:
        # Default path uses system temp dir; caller can override via output_dir
        self.output_dir = (output_dir or Path(tempfile.gettempdir()) / "edited").resolve()
        self.crf = crf

    def speed_adjust(self, video_path: Path, factor: float) -> Path:
        """Adjust playback speed of a video segment by the given factor."""
        output = self._edited_path(video_path, f"speed{factor}")
        return FFmpegOps.speed_adjust(video_path, output, factor)

    def trim(self, video_path: Path, start_s: float, end_s: float) -> Path:
        """Trim a video segment to the given start/end timestamps."""
        output = self._edited_path(video_path, "trimmed")
        return FFmpegOps.trim(video_path, output, start_s, end_s)

    def auto_trim(self, video_path: Path, trim_start: float = 0.5, trim_end: float = 0.5) -> Path:
        """Auto-trim a small amount from the start and end of a video."""
        duration = FFmpegOps.probe_duration(video_path)
        if duration <= trim_start + trim_end + 1.0:
            return video_path
        return self.trim(video_path, trim_start, duration - trim_end)

    @staticmethod
    def _run_freezedetect(video_path: Path, freeze_threshold_s: float) -> Any:
        try:
            return subprocess.run(  # nosec B603 B607  # noqa: S603 -- ffmpeg with controlled args
                [  # noqa: S607
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-vf",
                    f"freezedetect=n=0.003:d={freeze_threshold_s}",
                    "-an",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=FFmpegOps._default_timeout,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("freezedetect failed for %s: %s", video_path, e)
            return None

    @staticmethod
    def _parse_freeze_ranges(stderr: str) -> list[tuple[float, float]]:
        freeze_ranges: list[tuple[float, float]] = []
        freeze_start: float | None = None
        for line in stderr.splitlines():
            if "freeze_start" in line:
                parts = line.split("freeze_start:")
                if len(parts) >= 2:
                    try:
                        freeze_start = float(parts[1].strip().split()[0])
                    except ValueError, IndexError:
                        freeze_start = None
            elif "freeze_end" in line and freeze_start is not None:
                parts = line.split("freeze_end:")
                if len(parts) >= 2:
                    try:
                        freeze_end = float(parts[1].strip().split()[0])
                        freeze_ranges.append((freeze_start, freeze_end))
                    except ValueError, IndexError:
                        pass
                freeze_start = None
        return freeze_ranges

    def _build_dead_time_segments(
        self,
        video_path: Path,
        freeze_ranges: list[tuple[float, float]],
        duration: float,
        speed_factor: float,
    ) -> list[Path]:
        segment_paths: list[Path] = []
        current_pos = 0.0
        seg_idx = 0
        for fstart, fend in freeze_ranges:
            if fstart > current_pos + 0.1:
                seg_path = self._edited_path(video_path, f"live{seg_idx}")
                FFmpegOps.trim(video_path, seg_path, current_pos, fstart)
                segment_paths.append(seg_path)
                seg_idx += 1

            frozen_seg = self._edited_path(video_path, f"frozen{seg_idx}")
            FFmpegOps.trim(video_path, frozen_seg, fstart, fend)
            sped_seg = self._edited_path(video_path, f"sped{seg_idx}")
            FFmpegOps.speed_adjust(frozen_seg, sped_seg, speed_factor)
            segment_paths.append(sped_seg)
            with contextlib.suppress(OSError):
                frozen_seg.unlink()
            seg_idx += 1
            current_pos = fend

        if current_pos < duration - 0.1:
            seg_path = self._edited_path(video_path, f"live{seg_idx}")
            FFmpegOps.trim(video_path, seg_path, current_pos, duration)
            segment_paths.append(seg_path)
        return segment_paths

    def remove_dead_time(
        self,
        video_path: Path,
        speed_factor: float = 8.0,
        freeze_threshold_s: float = 2.0,
        fps: int = 30,
        cfr: bool = False,
    ) -> Path:
        """Detect frozen frames and speed them up to remove dead time.

        Threads ``fps``/``cfr`` into the final concat so dead-time-trimmed
        segments do not re-introduce source VFR ahead of the assembly concat.
        Defaults keep the emitted concat byte-identical.
        """
        duration = FFmpegOps.probe_duration(video_path)
        if duration < freeze_threshold_s * 2:
            return video_path

        result = self._run_freezedetect(video_path, freeze_threshold_s)
        if result is None:
            return video_path

        freeze_ranges = self._parse_freeze_ranges(result.stderr)
        if not freeze_ranges:
            return video_path

        logger.info("Detected %d frozen segment(s) in %s", len(freeze_ranges), video_path)

        segment_paths = self._build_dead_time_segments(
            video_path, freeze_ranges, duration, speed_factor
        )
        if not segment_paths:
            return video_path

        concat_file: Path | None = None
        try:
            output = self._edited_path(video_path, "nodeadtime")
            concat_file = self.output_dir / f"{video_path.stem}_deadtime_concat.txt"
            with concat_file.open("w") as f:
                for seg in segment_paths:
                    f.write(f"file '{seg.resolve()}'\n")
            FFmpegOps.concat(concat_file, output, fps=fps, cfr=cfr)
        finally:
            for seg in segment_paths:
                with contextlib.suppress(OSError):
                    seg.unlink()
            if concat_file is not None:
                with contextlib.suppress(OSError):
                    concat_file.unlink()

        return output

    def _edited_path(self, original: Path, suffix: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / f"{original.stem}-{suffix}{original.suffix}"
