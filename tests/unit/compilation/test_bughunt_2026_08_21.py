"""Regression tests for the 2026-08-21 bug hunt.

Each test here failed on ``origin/main`` (f2c5401) before its fix landed:

1. ``zoom_out`` (and any unknown preset) silently degraded to a static clip.
2. A stills-only screen-clip recording reported ``success: True`` with zero
   usable clips.
3. ``transitions_used`` in the compilation report was ``len(segments) - 1``
   even when every xfade failed and the assembler fell back to plain concat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.schemas.compilation import FindingInput, ShotListPartInput
from src.services.ffmpeg import FFmpegOps
from src.utils.motion_presets import MOTION_PRESETS

# ---------------------------------------------------------------------------
# 1. motion presets
# ---------------------------------------------------------------------------

# The documented vocabulary, mirrored from the recording template registry
# (``src/services/recording/templates/__init__.py``) and ``ken_burns.html``.
_DOCUMENTED_VOCABULARY = (
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
)


def test_motion_preset_constant_matches_documented_vocabulary():
    assert set(MOTION_PRESETS) == set(_DOCUMENTED_VOCABULARY)


@pytest.mark.parametrize("preset", _DOCUMENTED_VOCABULARY)
@patch.object(FFmpegOps, "run")
def test_every_documented_preset_renders_real_motion(mock_run, preset, tmp_path):
    """Each documented preset must reach a zoompan/crop branch.

    Before the fix ``zoom_out`` fell through to ``image_to_video`` (a bare
    ``scale=`` filter), producing a 100%-static clip while the job went green.
    """
    mock_run.return_value = MagicMock(returncode=0)
    FFmpegOps.image_to_motion_video(
        Path("/img.png"), tmp_path / "out.mp4", 5.0, motion_preset=preset
    )
    args = mock_run.call_args[0][0]
    vf = args[args.index("-vf") + 1]
    assert "zoompan=" in vf or "crop=" in vf, f"{preset!r} produced a static filter: {vf}"


@patch.object(FFmpegOps, "run")
def test_zoom_out_zooms_out_not_in(mock_run, tmp_path):
    """zoom_out must start wide and settle — i.e. a decreasing zoom expression."""
    mock_run.return_value = MagicMock(returncode=0)
    FFmpegOps.image_to_motion_video(
        Path("/img.png"), tmp_path / "out.mp4", 5.0, motion_preset="zoom_out"
    )
    args = mock_run.call_args[0][0]
    vf = args[args.index("-vf") + 1]
    assert "zoompan=" in vf
    assert "max(" in vf, f"zoom_out should clamp with max(), got {vf}"


@patch.object(FFmpegOps, "run")
def test_unknown_motion_preset_fails_loud(mock_run, tmp_path):
    """An unknown preset must raise, never degrade silently to a static clip."""
    mock_run.return_value = MagicMock(returncode=0)
    with pytest.raises(ValueError, match="motion_preset"):
        FFmpegOps.image_to_motion_video(
            Path("/img.png"), tmp_path / "out.mp4", 5.0, motion_preset="ken_burns_deluxe"
        )
    mock_run.assert_not_called()


# --- production call sites: the schema must not forward an unknown preset ---


_REQUIRED = {
    ShotListPartInput: {"id": "p1"},
    FindingInput: {"title": "f1"},
}


@pytest.mark.parametrize("model", [ShotListPartInput, FindingInput])
@pytest.mark.parametrize("preset", _DOCUMENTED_VOCABULARY)
def test_schema_accepts_documented_presets(model, preset):
    assert model(**_REQUIRED[model], motion_preset=preset).motion_preset == preset


@pytest.mark.parametrize("model", [ShotListPartInput, FindingInput])
def test_schema_rejects_unknown_preset(model):
    """Vacuity guard: the SAME payload without motion_preset must validate."""
    assert model(**_REQUIRED[model]).motion_preset is None
    with pytest.raises(ValidationError, match="motion_preset"):
        model(**_REQUIRED[model], motion_preset="ken_burns_deluxe")


@pytest.mark.parametrize("model", [ShotListPartInput, FindingInput])
def test_schema_still_allows_none_and_normalises_case(model):
    assert model(**_REQUIRED[model]).motion_preset is None
    assert model(**_REQUIRED[model], motion_preset="  Zoom_Out ").motion_preset == "zoom_out"


@patch.object(FFmpegOps, "run")
def test_segment_builder_zoom_out_reaches_ffmpeg_with_motion(mock_run, tmp_path):
    """Pin the PRODUCTION call site with the real FFmpegOps.

    Before the fix this reached ffmpeg with a bare ``scale=`` filter (static).
    """
    from src.services.compilation._segment_builders import _SegmentBuildersMixin

    mock_run.return_value = MagicMock(returncode=0)
    ctx = MagicMock()
    ctx.title_cards_dir = tmp_path
    ctx.target_w, ctx.target_h = 1920, 1080
    builder = _SegmentBuildersMixin()
    with (
        patch.object(
            _SegmentBuildersMixin, "_materialize_image_ref", return_value=tmp_path / "i.png"
        ),
        patch.object(
            _SegmentBuildersMixin,
            "_segment_with_optional_narration",
            return_value=tmp_path / "s.mp4",
        ),
    ):
        builder._process_visual_asset_part(  # type: ignore[attr-defined]
            ctx,
            "part1",
            {"image_url": "http://x/i.png", "motion_preset": "zoom_out"},
        )
    args = mock_run.call_args[0][0]
    vf = args[args.index("-vf") + 1]
    assert "zoompan=" in vf, f"production call site rendered a static clip: {vf}"


# ---------------------------------------------------------------------------
# 2. stills-only recording must not report success with zero clips
# ---------------------------------------------------------------------------


def test_trim_clips_all_failed_is_reported_as_failure():
    """``_trim_clips`` returning [] while boundaries existed is a hard failure."""
    from src.services.recording.url_recorder import UrlRecorder

    recorder = UrlRecorder.__new__(UrlRecorder)
    boundaries = [("s1", 0.0, 2.0), ("s2", 2.0, 4.0)]
    assert recorder._clip_capture_failure(boundaries, clips=[], stills=[("s1", b"x")]) is not None
    # every boundary trimmed -> no failure
    assert (
        recorder._clip_capture_failure(
            boundaries, clips=[("s1", Path("a")), ("s2", Path("b"))], stills=[]
        )
        is None
    )
    # no boundaries at all (stills-only capture by design) -> no clip failure
    assert recorder._clip_capture_failure([], clips=[], stills=[("s1", b"x")]) is None


@pytest.mark.asyncio
async def test_stills_only_capture_with_failed_trims_reports_failure(tmp_path):
    """End-to-end on ``capture_screen_clips``: every trim fails -> success False."""
    from src.services.recording.url_recorder import UrlRecorder

    recorder = UrlRecorder.__new__(UrlRecorder)
    boundaries = [("s1", 0.0, 2.0)]
    stills = [("s1", b"png")]
    webm = tmp_path / "rec.webm"
    webm.write_bytes(b"x")

    async def _capture(*_a, **_kw):
        return webm, boundaries, stills, {}

    with (
        patch.object(
            UrlRecorder,
            "_resolve_config",
            return_value={
                "quality_preset": "standard",
                "viewport_width": 1280,
                "viewport_height": 720,
            },
        ),
        patch.object(UrlRecorder, "_capture_clips_with_playwright", side_effect=_capture),
        patch.object(UrlRecorder, "_trim_clips", return_value=[]),
    ):
        result = await recorder.capture_screen_clips("job1", "http://x", None, {}, [])

    assert result["success"] is False
    assert "clip" in result["error"].lower()


# ---------------------------------------------------------------------------
# 3. transitions_used must count transitions actually applied
# ---------------------------------------------------------------------------


def _assembler() -> Any:
    """Bare instance of the assembly mixin."""
    from src.services.compilation._assembly import _AssemblyMixin

    return _AssemblyMixin.__new__(_AssemblyMixin)


def _segs(tmp_path, n):
    out = []
    for i in range(n):
        p = tmp_path / f"seg{i}.mp4"
        p.write_bytes(b"x")
        out.append(p)
    return out


def test_xfade_fallback_to_concat_reports_zero_transitions(tmp_path):
    segments = _segs(tmp_path, 3)
    out = tmp_path / "out.mp4"
    with patch("src.services.compilation._assembly._ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.return_value.crossfade.side_effect = RuntimeError("xfade died")
        used = _assembler()._assemble_with_xfade(segments, out, tmp_path, 0.5, "fade")
    assert used == 0, "concat fallback contains no transitions"


def test_xfade_partial_failure_reports_only_applied_transitions(tmp_path):
    segments = _segs(tmp_path, 4)
    out = tmp_path / "out.mp4"
    calls = {"n": 0}

    def _crossfade(_a, _b, temp_out, **_kw):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("xfade died")
        Path(temp_out).write_bytes(b"x")
        return temp_out

    with patch("src.services.compilation._assembly._ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.return_value.crossfade.side_effect = _crossfade
        used = _assembler()._assemble_with_xfade(segments, out, tmp_path, 0.5, "fade")
    assert used == 1


def test_assemble_video_records_actual_transitions_on_fallback(tmp_path):
    """Pin the production call site: ctx.transitions_used feeds the report."""
    from src.services.compilation._assembly import _AssemblyMixin

    ctx = MagicMock()
    ctx.segments = _segs(tmp_path, 3)
    ctx.work_dir = tmp_path
    ctx.edited_dir = tmp_path
    ctx.transition_type = "fade"
    ctx.crossfade_dur = 0.5
    ctx.target_fps = 30
    ctx.cfr_mode = False
    ctx.job.filter_config = {}
    ctx.settings.compilation.output_filename = "final.mp4"
    ctx.transitions_used = 0

    with patch("src.services.compilation._assembly._ffmpeg") as mock_ffmpeg:
        mock_ffmpeg.return_value.crossfade.side_effect = RuntimeError("xfade died")
        _AssemblyMixin._assemble_video(_AssemblyMixin.__new__(_AssemblyMixin), ctx)

    assert ctx.transitions_used == 0


# ---------------------------------------------------------------------------
# 4a. quality_validator must fail an audio-only / single-frame file
# ---------------------------------------------------------------------------


def _probe(**kw):
    from src.services.ffmpeg.base import ProbeResult

    defaults: dict[str, Any] = {
        "duration": 10.0,
        "width": 1920,
        "height": 1080,
        "has_audio": True,
        "has_video": True,
        "video_frames": 300,
    }
    defaults.update(kw)
    return ProbeResult(**defaults)


def _validate(tmp_path, probe):
    from src.services.compilation.quality_validator import QualityValidator

    f = tmp_path / "out.mp4"
    f.write_bytes(b"x" * 1024)
    with (
        patch("src.services.compilation.quality_validator.FFmpegOps.probe_all", return_value=probe),
        patch.object(QualityValidator, "_check_audio_levels", return_value=-6.0),
        patch.object(QualityValidator, "_check_encoding_integrity", return_value=True),
    ):
        return QualityValidator().validate(f)


def test_quality_validator_baseline_is_valid(tmp_path):
    """Vacuity guard for the two tests below."""
    assert _validate(tmp_path, _probe()).valid is True


def test_audio_only_file_is_invalid(tmp_path):
    report = _validate(tmp_path, _probe(has_video=False, width=0, height=0, video_frames=0))
    assert report.valid is False
    assert any(i["check"] == "video_stream_present" for i in report.issues)


def test_single_frame_video_is_invalid(tmp_path):
    report = _validate(tmp_path, _probe(video_frames=1))
    assert report.valid is False
    assert any(i["check"] == "video_has_frames" for i in report.issues)


# ---------------------------------------------------------------------------
# 4b. an over-long crossfade must not erase clip 1
# ---------------------------------------------------------------------------


@patch.object(FFmpegOps, "run")
@patch.object(FFmpegOps, "probe_has_audio", return_value=False)
@patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
@patch.object(FFmpegOps, "probe_duration", return_value=2.0)
def test_crossfade_longer_than_clip_is_clamped(_dur, _dims, _audio, mock_run, tmp_path):
    """``offset = max(0, d1 - fade_duration)`` used to hit 0, erasing clip 1."""
    mock_run.return_value = MagicMock(returncode=0)
    FFmpegOps.crossfade(
        tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "o.mp4", fade_duration=5.0
    )
    args = mock_run.call_args[0][0]
    vfilter = next(a for a in args if "xfade=" in a)
    offset = float(vfilter.split("offset=")[1].split("[")[0])
    assert offset > 0.0, f"clip 1 vanished entirely: {vfilter}"


@pytest.mark.parametrize(
    "model_path",
    ["CompileRequest", "CustomPresetCreateRequest", "TemplateCreateRequest"],
)
def test_crossfade_duration_is_bounded_in_request_schemas(model_path):
    import src.api.schemas.compilation as sch

    model = getattr(sch, model_path)
    field = model.model_fields["crossfade_duration"]
    bounds = {type(m).__name__ for m in field.metadata}
    assert {"Ge", "Le"} <= bounds, f"{model_path}.crossfade_duration is unbounded"
