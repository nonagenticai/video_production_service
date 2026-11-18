"""Every ffmpeg output this service emits must be tagged BT.709.

Untagged H.264 leaves the colour description to the decoder's guess, which is
what made ``colorspace_ok`` false on the delivered masters. Two mechanisms are
needed and they are NOT interchangeable:

* a re-encode takes the encoder flags (``COLOR_TAG_ARGS``);
* a ``-c:v copy`` hands the bitstream through untouched, so no encoder flag can
  reach it — only the ``h264_metadata`` bitstream filter can rewrite the SPS
  VUI (``COPY_COLOR_TAG_ARGS``).

These tests assert the ORDER-SENSITIVE, contiguous presence of each token group
in the real argv, not merely that the strings appear somewhere, so a splice into
an INPUT position (before ``-i``) would still fail.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.ffmpeg import COLOR_TAG_ARGS, COPY_COLOR_TAG_ARGS, FFmpegOps


def _contains_subseq(argv: list[str], sub: list[str]) -> bool:
    """True iff ``sub`` appears as a contiguous run inside ``argv``."""
    n = len(sub)
    return any(argv[i : i + n] == sub for i in range(len(argv) - n + 1))


def _output_side(argv: list[str]) -> list[str]:
    """Args after the LAST ``-i`` — i.e. the output-option side of the argv."""
    last_i = max(i for i, a in enumerate(argv) if a == "-i")
    return argv[last_i + 2 :]


@pytest.fixture
def run_argv():
    """Patch subprocess.run and yield a callable returning the argv of call N."""
    with patch("src.services.ffmpeg.base.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        yield mock_run


_CHAPTERS = [{"chapter_number": 1, "title": "One", "start_seconds": 0.0, "duration_seconds": 5.0}]


@pytest.fixture
def h264_source():
    """The stream-copy source probes as H.264 -- the bsf is legal."""
    with patch.object(FFmpegOps, "probe_video_codec", return_value="h264"):
        yield


@pytest.fixture
def vp9_source():
    """The stream-copy source probes as VP9 -- the H.264 bsf would hard-error."""
    with patch.object(FFmpegOps, "probe_video_codec", return_value="vp9"):
        yield


def test_reencode_argv_carries_the_literal_bt709_triplet(run_argv):
    """The VALUES, asserted against the real argv -- not against the constant.

    Comparing ``COLOR_TAG_ARGS`` to a copy of itself restates the definition
    and cannot fail on a behaviour regression. This spells the six tokens out
    as literals and demands them on the output side of an argv that ffmpeg
    actually received, so a typo'd primary OR a splice that never reaches the
    command line both fail here.
    """
    FFmpegOps.concat(Path("list.txt"), Path("out.mp4"))
    argv = run_argv.call_args[0][0]
    assert _contains_subseq(
        _output_side(argv),
        ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"],
    ), argv


def test_copy_argv_carries_the_literal_h264_metadata_bsf(run_argv, h264_source, tmp_path):
    """The bsf spec, asserted against the real argv of a stream-copy call."""
    FFmpegOps.embed_chapters(tmp_path / "v.mp4", tmp_path / "o.mp4", _CHAPTERS)
    argv = _output_side(run_argv.call_args[0][0])
    assert "-bsf:v" in argv, argv
    spec = argv[argv.index("-bsf:v") + 1]
    assert spec.startswith("h264_metadata="), spec
    # British spelling is what the bsf actually names the option; the American
    # one is silently ignored by ffmpeg's option parser.
    assert "colour_primaries=1" in spec
    assert "transfer_characteristics=1" in spec
    assert "matrix_coefficients=1" in spec
    assert "video_full_range_flag=0" in spec


# ---------------------------------------------------------------------------
# Re-encode branches -> encoder colour flags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "expect_encode"),
    [
        (lambda: FFmpegOps.concat(Path("list.txt"), Path("out.mp4")), True),
        (lambda: FFmpegOps.image_to_video(Path("i.png"), Path("out.mp4"), 3.0), True),
        (lambda: FFmpegOps.frames_to_mp4(Path("/frames"), Path("out.mp4"), fps=30), True),
        (lambda: FFmpegOps.trim_reencode(Path("i.mp4"), Path("o.mp4"), 0.0, 2.0), True),
        (
            lambda: FFmpegOps.burn_subtitles(Path("i.mp4"), Path("s.srt"), Path("o.mp4")),
            True,
        ),
        (lambda: FFmpegOps.generate_preview(Path("i.mp4"), Path("o.mp4")), True),
    ],
)
def test_reencode_paths_tag_bt709(run_argv, call, expect_encode):
    call()
    assert run_argv.call_count >= 1
    argv = run_argv.call_args[0][0]
    assert _contains_subseq(_output_side(argv), COLOR_TAG_ARGS), argv


def test_normalize_stream_tags_both_audio_branches(run_argv):
    with patch.object(FFmpegOps, "probe_has_audio", return_value=True):
        FFmpegOps.normalize_stream(Path("i.mp4"), Path("o.mp4"))
    assert _contains_subseq(_output_side(run_argv.call_args[0][0]), COLOR_TAG_ARGS)
    with patch.object(FFmpegOps, "probe_has_audio", return_value=False):
        FFmpegOps.normalize_stream(Path("i.mp4"), Path("o2.mp4"))
    assert _contains_subseq(_output_side(run_argv.call_args[0][0]), COLOR_TAG_ARGS)


# ---------------------------------------------------------------------------
# mix_voiceover: the branch that made colorspace_ok INTERMITTENT.
# freeze_video=True re-encodes (tpad); False stream-copies.
# ---------------------------------------------------------------------------


def _mix_voiceover_argv(run_argv, *, freeze: bool, video_has_audio: bool = True):
    # video 10s; voiceover 20s (longer -> freeze) or 5s (shorter -> copy).
    durations = {"video.mp4": 10.0, "vo.wav": 20.0 if freeze else 5.0}
    with (
        patch.object(FFmpegOps, "probe_duration", side_effect=lambda p: durations[Path(p).name]),
        patch.object(FFmpegOps, "probe_has_audio", return_value=video_has_audio),
    ):
        FFmpegOps.mix_voiceover(Path("video.mp4"), Path("vo.wav"), Path("out.mp4"))
    return run_argv.call_args[0][0]


def test_mix_voiceover_copy_branch_carries_the_h264_metadata_bsf(run_argv, h264_source):
    argv = _mix_voiceover_argv(run_argv, freeze=False)
    # Precondition: this really is the stream-copy branch.
    assert _contains_subseq(argv, ["-c:v", "copy"]), argv
    assert _contains_subseq(_output_side(argv), COPY_COLOR_TAG_ARGS), argv


def test_mix_voiceover_freeze_branch_carries_encoder_color_flags(run_argv):
    argv = _mix_voiceover_argv(run_argv, freeze=True)
    # Precondition: this really is the re-encode branch (tpad clone, no -c:v copy).
    assert not _contains_subseq(argv, ["-c:v", "copy"]), argv
    assert any("tpad=stop_mode=clone" in a for a in argv), argv
    assert _contains_subseq(_output_side(argv), COLOR_TAG_ARGS), argv


def test_mix_voiceover_silence_pad_copy_branch_carries_the_bsf(run_argv, h264_source):
    # Source video has no audio and the VO is shorter -> the apad + `-c:v copy`
    # path, a second stream-copy branch in the same function.
    argv = _mix_voiceover_argv(run_argv, freeze=False, video_has_audio=False)
    assert _contains_subseq(argv, ["-c:v", "copy"]), argv
    assert _contains_subseq(_output_side(argv), COPY_COLOR_TAG_ARGS), argv


def test_mix_background_music_copy_branches_carry_the_bsf(run_argv, h264_source):
    for has_audio in (True, False):
        with (
            patch.object(FFmpegOps, "probe_duration", return_value=30.0),
            patch.object(FFmpegOps, "probe_has_audio", return_value=has_audio),
        ):
            FFmpegOps.mix_background_music(Path("v.mp4"), Path("m.mp3"), Path("o.mp4"))
        argv = run_argv.call_args[0][0]
        assert _contains_subseq(argv, ["-c:v", "copy"]), argv
        assert _contains_subseq(_output_side(argv), COPY_COLOR_TAG_ARGS), (has_audio, argv)


# ---------------------------------------------------------------------------
# Multi-input compose ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["crossfade", "hstack", "fade_to_black"])
def test_compose_transition_ops_tag_bt709(run_argv, op):
    with (
        patch.object(FFmpegOps, "probe_duration", return_value=10.0),
        patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080)),
        patch.object(FFmpegOps, "probe_has_audio", return_value=True),
    ):
        getattr(FFmpegOps, op)(Path("a.mp4"), Path("b.mp4"), Path("o.mp4"))
    argv = run_argv.call_args[0][0]
    assert _contains_subseq(_output_side(argv), COLOR_TAG_ARGS), argv


# ---------------------------------------------------------------------------
# The codec gate: the h264_metadata bsf is H.264-ONLY.
#
# ffmpeg hard-errors when a bitstream filter meets a stream of the wrong
# codec, and this service really does emit VP9/WebM
# (``transcode.frames_to_webm``). An ungated splice would therefore turn a
# working webm stream-copy into a stage failure, so every ``-c(:v) copy``
# tags only what it has PROBED as H.264.
# ---------------------------------------------------------------------------

# Each entry takes a tmp dir: ``embed_chapters`` really writes an FFMETADATA1
# sidecar next to its input, so it must not run against the repo cwd.
_COPY_OPS = [
    (
        "embed_chapters",
        lambda d: FFmpegOps.embed_chapters(d / "v.mp4", d / "o.mp4", _CHAPTERS),
    ),
    (
        "embed_subtitles",
        lambda d: FFmpegOps.embed_subtitles(d / "v.webm", d / "s.vtt", d / "o.mkv"),
    ),
    ("trim", lambda d: FFmpegOps.trim(d / "v.mp4", d / "o.mp4", 0.0, 2.0)),
    ("strip_audio", lambda d: FFmpegOps.strip_audio(d / "v.mp4", d / "o.mp4")),
]


@pytest.mark.parametrize(("name", "call"), _COPY_OPS, ids=[n for n, _ in _COPY_OPS])
def test_stream_copy_ops_tag_bt709_when_source_is_h264(run_argv, h264_source, tmp_path, name, call):
    call(tmp_path)
    argv = run_argv.call_args[0][0]
    assert _contains_subseq(argv, ["-c", "copy"]) or _contains_subseq(argv, ["-c:v", "copy"]), argv
    assert _contains_subseq(_output_side(argv), COPY_COLOR_TAG_ARGS), (name, argv)


@pytest.mark.parametrize(("name", "call"), _COPY_OPS, ids=[n for n, _ in _COPY_OPS])
def test_stream_copy_ops_omit_the_bsf_on_a_non_h264_source(
    run_argv, vp9_source, tmp_path, name, call
):
    call(tmp_path)
    argv = run_argv.call_args[0][0]
    assert "-bsf:v" not in argv, (name, argv)


@pytest.mark.parametrize(("name", "call"), _COPY_OPS, ids=[n for n, _ in _COPY_OPS])
def test_stream_copy_ops_omit_the_bsf_when_the_codec_probe_fails(run_argv, tmp_path, name, call):
    # An unreadable probe is "I could not tell", and the safe answer to that is
    # "do not splice an H.264-only filter", never "probably fine".
    with patch.object(FFmpegOps, "probe_video_codec", return_value=""):
        call(tmp_path)
    argv = run_argv.call_args[0][0]
    assert "-bsf:v" not in argv, (name, argv)


def test_mix_voiceover_copy_branch_omits_the_bsf_on_a_non_h264_source(run_argv, vp9_source):
    argv = _mix_voiceover_argv(run_argv, freeze=False)
    assert _contains_subseq(argv, ["-c:v", "copy"]), argv
    assert "-bsf:v" not in argv, argv


def test_mix_background_music_omits_the_bsf_on_a_non_h264_source(run_argv, vp9_source):
    with (
        patch.object(FFmpegOps, "probe_duration", return_value=30.0),
        patch.object(FFmpegOps, "probe_has_audio", return_value=True),
    ):
        FFmpegOps.mix_background_music(Path("v.mp4"), Path("m.mp3"), Path("o.mp4"))
    assert "-bsf:v" not in run_argv.call_args[0][0]


def test_copy_color_tag_args_probes_the_source_stream_codec():
    """The gate reads codec_name off the first VIDEO stream of the SOURCE."""
    with patch.object(FFmpegOps, "_probe", return_value="h264") as probe:
        assert FFmpegOps.copy_color_tag_args(Path("v.mp4")) == COPY_COLOR_TAG_ARGS
    args = probe.call_args[0][0]
    assert _contains_subseq(args, ["-select_streams", "v:0"]), args
    assert _contains_subseq(args, ["-show_entries", "stream=codec_name"]), args
    assert args[-1] == "v.mp4", args
