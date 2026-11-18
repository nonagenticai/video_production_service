"""Subtitle ops: hardcode/burn-in and soft-track embedding."""

from __future__ import annotations

from pathlib import Path

from src.services.ffmpeg.base import COLOR_TAG_ARGS
from src.services.ffmpeg.probe import ProbeMixin

# MP4/MOV carry exactly one text subtitle codec: ``mov_text`` (tx3g). ffmpeg
# refuses ``srt``/``webvtt`` there ("Could not find tag for codec subrip ...
# codec not currently supported in container"), so the codec is chosen by the
# OUTPUT container, never by the input subtitle file.
_MOV_TEXT_CONTAINERS = frozenset({".mp4", ".mov", ".m4v"})


def subtitle_codec_for(subtitle_path: Path, output_path: Path) -> str:
    """Soft-track subtitle codec that ``output_path``'s container accepts."""
    out_ext = output_path.suffix.lower()
    if out_ext in _MOV_TEXT_CONTAINERS:
        return "mov_text"
    if out_ext == ".webm":
        return "webvtt"  # WebM accepts only WebVTT text tracks
    # Matroska and anything else: keep the source's own text codec.
    return "webvtt" if subtitle_path.suffix.lower() == ".vtt" else "srt"


class SubtitlesMixin(ProbeMixin):
    """Subtitle burning + soft-track embedding."""

    @classmethod
    def burn_subtitles(
        cls,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
        font_size: int = 24,
        font_color: str = "white",
        position: str = "bottom",
    ) -> Path:
        """Burn subtitles into video (hardcode) (WP-10)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        margin_v = 20 if position == "bottom" else 10
        alignment = 2 if position == "bottom" else 6  # SSA alignment
        style = f"FontSize={font_size},PrimaryColour=&H00FFFFFF,MarginV={margin_v},Alignment={alignment}"
        sub_path_escaped = str(subtitle_path).replace("\\", "/").replace(":", "\\:")
        cls.run(
            [
                "-i",
                str(video_path),
                "-vf",
                f"subtitles={sub_path_escaped}:force_style='{style}'",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                *COLOR_TAG_ARGS,
                "-c:a",
                "copy",
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def embed_subtitles(cls, video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
        """Embed subtitles as a soft track (WP-10)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sub_codec = subtitle_codec_for(subtitle_path, output_path)
        cls.run(
            [
                "-i",
                str(video_path),
                "-i",
                str(subtitle_path),
                "-c:v",
                "copy",
                *cls.copy_color_tag_args(video_path),
                "-c:a",
                "copy",
                "-c:s",
                sub_codec,
                "-map",
                "0:v",
                "-map",
                "0:a?",
                "-map",
                "1:0",
                str(output_path),
            ]
        )
        return output_path
