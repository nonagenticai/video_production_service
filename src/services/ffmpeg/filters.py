"""Per-clip filter ops: normalize, thumbnail, loudnorm, preview."""

from __future__ import annotations

from pathlib import Path

from src.services.ffmpeg.base import COLOR_TAG_ARGS
from src.services.ffmpeg.probe import ProbeMixin


class FiltersMixin(ProbeMixin):
    """Filter-style single-input ops (scaling, padding, audio normalisation)."""

    @classmethod
    def normalize_stream(
        cls,
        input_path: Path,
        output_path: Path,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        crf: int = 20,
        cfr: bool = False,
    ) -> Path:
        """Scale, pad, and re-encode a video to uniform resolution, FPS, and audio format.

        When ``cfr`` is True, emit ``-fps_mode cfr -r {fps}`` so browser-captured
        VFR sources become true constant-frame-rate. Defaults to False → the emitted
        command is byte-identical to the pre-CFR behaviour.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_audio = cls.probe_has_audio(input_path)
        cfr_args = ["-fps_mode", "cfr", "-r", str(fps)] if cfr else []
        vfilter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}"
        )
        if has_audio:
            cls.run(
                [
                    "-i",
                    str(input_path),
                    "-vf",
                    vfilter,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-crf",
                    str(crf),
                    "-preset",
                    cls._default_preset,
                    "-pix_fmt",
                    "yuv420p",
                    *COLOR_TAG_ARGS,
                    "-movflags",
                    "+faststart",
                    *cfr_args,
                    str(output_path),
                ]
            )
        else:
            cls.run(
                [
                    "-i",
                    str(input_path),
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-vf",
                    vfilter,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-crf",
                    str(crf),
                    "-preset",
                    cls._default_preset,
                    "-pix_fmt",
                    "yuv420p",
                    *COLOR_TAG_ARGS,
                    "-movflags",
                    "+faststart",
                    *cfr_args,
                    "-shortest",
                    str(output_path),
                ]
            )
        return output_path

    @classmethod
    def extract_thumbnail(
        cls,
        video_path: Path,
        output_path: Path,
        timestamp: float = 1.0,
        width: int = 640,
        height: int = 360,
    ) -> Path:
        """Extract a single frame as a thumbnail image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cls.run(
            [
                "-ss",
                str(timestamp),
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-vf",
                (
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
                    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
                ),
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def loudnorm_audio(
        cls,
        input_path: Path,
        output_path: Path,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
    ) -> Path:
        """Apply EBU R128 loudness normalization to an audio file.

        Ensures consistent loudness across concatenated narration parts.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cls.run(
            [
                "-i",
                str(input_path),
                "-af",
                f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11",
                "-ar",
                "48000",
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def generate_preview(
        cls,
        input_path: Path,
        output_path: Path,
        duration: float = 15.0,
        width: int = 640,
        height: int = 360,
    ) -> Path:
        """Generate a low-quality preview of the first N seconds (WP-10)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vfilter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
        cls.run(
            [
                "-i",
                str(input_path),
                "-t",
                str(duration),
                "-vf",
                vfilter,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "28",
                *COLOR_TAG_ARGS,
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                str(output_path),
            ]
        )
        return output_path
