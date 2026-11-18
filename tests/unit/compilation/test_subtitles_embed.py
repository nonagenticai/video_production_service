"""Subtitles are a soft track by default, and the soft track must fit the container.

Owner rule: subtitles are never burned into the picture by default. A burn is
irreversible -- the text becomes pixels and no player can turn it off -- so the
schema default is ``embed`` and only an explicit ``mode="burn"`` hardcodes.

The embed path used to pick the subtitle codec from the *input* file
(``.srt`` -> ``srt``, ``.vtt`` -> ``webvtt``) and write it into the compile's
``.mp4``. MP4 accepts neither: ffmpeg refuses with ``Could not find tag for
codec subrip ... codec not currently supported in container``. The only text
subtitle codec MP4/MOV carry is ``mov_text``, so the codec is chosen from the
*output* container.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.schemas import SubtitleInput
from src.services.ffmpeg import FFmpegOps


@pytest.fixture
def run_argv():
    with (
        patch("src.services.ffmpeg.base.subprocess.run") as mock_run,
        patch.object(FFmpegOps, "probe_video_codec", return_value="h264"),
    ):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        yield mock_run


def _codec_arg(argv: list[str]) -> str:
    return argv[argv.index("-c:s") + 1]


def test_schema_default_mode_is_embed_not_burn():
    assert SubtitleInput(file_path="/app/data/j/c.srt").mode == "embed"


def test_explicit_burn_is_still_honoured():
    assert SubtitleInput(file_path="/app/data/j/c.srt", mode="burn").mode == "burn"


@pytest.mark.parametrize("sub", ["s.srt", "s.vtt"])
@pytest.mark.parametrize("out", ["o.mp4", "o.mov", "o.m4v", "O.MP4"])
def test_embed_into_mp4_family_uses_mov_text(run_argv, tmp_path, sub, out):
    FFmpegOps.embed_subtitles(tmp_path / "v.mp4", tmp_path / sub, tmp_path / out)
    assert _codec_arg(run_argv.call_args[0][0]) == "mov_text"


@pytest.mark.parametrize(
    ("sub", "out", "codec"),
    [
        ("s.srt", "o.mkv", "srt"),
        ("s.vtt", "o.mkv", "webvtt"),
        ("s.srt", "o.webm", "webvtt"),
        ("s.vtt", "o.webm", "webvtt"),
    ],
)
def test_embed_into_matroska_family_keeps_text_codecs(run_argv, tmp_path, sub, out, codec):
    FFmpegOps.embed_subtitles(tmp_path / "v.mkv", tmp_path / sub, tmp_path / out)
    assert _codec_arg(run_argv.call_args[0][0]) == codec


def test_codec_mapping_is_by_output_container():
    from src.services.ffmpeg.subtitles import subtitle_codec_for

    assert subtitle_codec_for(Path("x.srt"), Path("o.mp4")) == "mov_text"
    assert subtitle_codec_for(Path("x.vtt"), Path("o.mkv")) == "webvtt"


@pytest.fixture
def require_ffmpeg() -> None:
    """Skip locally without ffmpeg/ffprobe, FAIL under CI (see recording/conftest.py)."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    if os.environ.get("CI"):
        pytest.fail("ffmpeg/ffprobe not on PATH but CI is set", pytrace=False)
    pytest.skip("ffmpeg not on PATH")


@pytest.mark.usefixtures("require_ffmpeg")
@pytest.mark.parametrize("sub_name", ["s.srt", "s.vtt"])
def test_embed_really_produces_a_subtitle_stream_in_mp4(tmp_path, sub_name):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    video = tmp_path / "v.mp4"
    subprocess.run(  # noqa: S603 -- fixed argv, absolute binary
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
            "-f", "lavfi", "-i", "anullsrc", "-t", "1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    sub = tmp_path / sub_name
    if sub_name.endswith(".srt"):
        sub.write_text("1\n00:00:00,000 --> 00:00:00,900\nHello\n")
    else:
        sub.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:00.900\nHello\n")
    out = tmp_path / "o.mp4"

    FFmpegOps.embed_subtitles(video, sub, out)

    probe = subprocess.run(  # noqa: S603 -- fixed argv, absolute binary
        [
            ffprobe, "-v", "error", "-select_streams", "s",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip
    assert probe.stdout.split() == ["mov_text"]
