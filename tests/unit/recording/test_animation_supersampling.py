"""Animation capture quality: 2x supersampling + preset-driven resolution.

Two defects that made every animated clip soft:

1. the single Chromium context that serves *every* motion template captured at
   ``device_scale_factor=1``, so glyph edges and strokes were rasterised 1:1 and
   aliased — worst on the shots where fine type IS the shot;
2. ``quality_preset`` changed only the CRF, never the size, so a ``2k``/``4k``
   order that did not hand-pass a viewport got a square 1080x1080 clip.

The assertions here are deliberately on the ffmpeg argv, the resolved config and
(where ffmpeg is present) the real scaled PNG's pixel dimensions — a
``deviceScaleFactor`` readback is NOT proof that a larger picture was drawn.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from src.services.ffmpeg import FFmpegOps
from src.services.recording import animation_recorder as ar_mod
from src.services.recording.animation_recorder import (
    _CAPTURE_SCALE,
    AnimationRecorder,
    _resolve_capture_scale,
)
from src.services.recording.quality_presets import QUALITY_PRESETS
from tests.unit.recording.test_animation_recorder import _make_recorder

# ---------------------------------------------------------------------------
# Defect 1a — the capture context asks Chromium for a 2x raster
# ---------------------------------------------------------------------------


class _RecordingContext:
    """Captures the kwargs ``new_context`` was called with."""

    def __init__(self) -> None:
        self.kwargs: dict = {}

    async def new_page(self):
        page = MagicMock()
        page.set_content = AsyncMock()
        page.evaluate = AsyncMock(return_value=1000)
        page.screenshot = AsyncMock()
        return page

    async def close(self) -> None:
        return None


async def _capture_context_kwargs(config: dict, tmp_path: Path) -> dict:
    ctx = _RecordingContext()

    class _Browser:
        async def new_context(self, **kwargs):
            ctx.kwargs = kwargs
            return ctx

        async def close(self) -> None:
            return None

    class _PW:
        def __init__(self) -> None:
            self.chromium = MagicMock()
            self.chromium.launch = AsyncMock(return_value=_Browser())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return None

    recorder = _make_recorder()
    with (
        patch("playwright.async_api.async_playwright", return_value=_PW()),
        patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
    ):
        await recorder._capture_frames(
            "job-ss", tmp_path, "<html><body></body></html>", config, 1, 1000, None, None
        )
    return ctx.kwargs


@pytest.mark.asyncio
async def test_capture_context_requests_2x_device_scale_factor(tmp_path):
    """Every motion template shares this context; 1x here aliases all of them."""
    kwargs = await _capture_context_kwargs(
        {"width": 1920, "height": 1080, "capture_scale": 2, "headless": True}, tmp_path
    )
    assert kwargs["device_scale_factor"] == 2


@pytest.mark.asyncio
async def test_viewport_stays_the_target_size_so_layout_does_not_reflow(tmp_path):
    """Supersampling must multiply the RASTER, not the CSS viewport.

    Widening the viewport instead would re-run the template's layout at a
    different size — a different picture, not a sharper one.
    """
    kwargs = await _capture_context_kwargs(
        {"width": 1920, "height": 1080, "capture_scale": 2, "headless": True}, tmp_path
    )
    assert kwargs["viewport"] == {"width": 1920, "height": 1080}


@pytest.mark.asyncio
async def test_capture_scale_1_still_pins_device_scale_factor_1(tmp_path):
    kwargs = await _capture_context_kwargs(
        {"width": 1080, "height": 1080, "capture_scale": 1, "headless": True}, tmp_path
    )
    assert kwargs["device_scale_factor"] == 1


# ---------------------------------------------------------------------------
# Defect 1b — the supersampled sequence is downscaled with lanczos
# ---------------------------------------------------------------------------


class TestDownscalePass:
    def test_assemble_inserts_a_lanczos_downscale_to_the_target_size(self, tmp_path):
        with (
            patch.object(FFmpegOps, "run") as run,
            patch.object(FFmpegOps, "frames_to_mp4", return_value=tmp_path / "out.mp4") as enc,
        ):
            AnimationRecorder._assemble(
                tmp_path,
                30,
                "mp4",
                18,
                target_width=1920,
                target_height=1080,
                capture_scale=2,
            )
        argv = run.call_args.args[0]
        assert "scale=1920:1080:flags=lanczos" in argv
        # ...and the encoder must read the SCALED sequence, not the 2x one.
        assert enc.call_args.args[0] == tmp_path / "scaled"

    def test_no_downscale_pass_when_capture_scale_is_1(self, tmp_path):
        """The pre-feature argv, verbatim — a direct caller pays nothing."""
        with (
            patch.object(FFmpegOps, "run") as run,
            patch.object(FFmpegOps, "frames_to_mp4", return_value=tmp_path / "out.mp4") as enc,
        ):
            AnimationRecorder._assemble(
                tmp_path, 30, "mp4", 18, target_width=1920, target_height=1080, capture_scale=1
            )
        run.assert_not_called()
        assert enc.call_args.args[0] == tmp_path

    def test_transparent_downscale_pins_rgba_so_alpha_survives_the_pass(self, tmp_path):
        with (
            patch.object(FFmpegOps, "run") as run,
            patch.object(FFmpegOps, "frames_to_webm", return_value=tmp_path / "out.webm"),
        ):
            AnimationRecorder._assemble(
                tmp_path,
                30,
                "webm",
                18,
                True,
                target_width=1080,
                target_height=1080,
                capture_scale=2,
            )
        argv = run.call_args.args[0]
        assert argv[argv.index("-pix_fmt") + 1] == "rgba"

    @pytest.mark.usefixtures("require_ffmpeg")
    def test_real_pass_writes_target_sized_pngs_and_frees_the_2x_originals(self, tmp_path):
        """The unambiguous proof: real ffmpeg, real PNG pixel dimensions."""
        for i in range(3):
            Image.new("RGB", (2160, 2160), (40, 90, 200)).save(tmp_path / f"frame_{i:06d}.png")

        scaled = AnimationRecorder._downscale_frames(tmp_path, 1080, 1080)

        produced = sorted(scaled.glob("frame_*.png"))
        assert len(produced) == 3
        for f in produced:
            with Image.open(f) as im:
                assert im.size == (1080, 1080)
        # 2x frames are 4x the bytes; both sequences co-resident is how a long
        # clip fills the pod's ephemeral storage.
        assert sorted(tmp_path.glob("frame_*.png")) == []


# ---------------------------------------------------------------------------
# Defect 1c — the raster budget keeps a 4k order from OOM-killing the pod
# ---------------------------------------------------------------------------


class TestRasterBudget:
    @pytest.mark.parametrize(("w", "h"), [(1080, 1080), (1920, 1080), (1280, 720), (2560, 1440)])
    def test_supersamples_every_tier_that_fits(self, w, h):
        assert _resolve_capture_scale(w, h) == _CAPTURE_SCALE

    def test_4k_falls_back_to_1x(self):
        """7680x4320 is ~133MB of RGBA per frame — sharper is not worth an OOM."""
        assert _resolve_capture_scale(3840, 2160) == 1

    def test_an_explicit_request_is_clamped_by_the_same_budget(self):
        assert _resolve_capture_scale(3840, 2160, 2) == 1
        assert _resolve_capture_scale(1080, 1080, 1) == 1


# ---------------------------------------------------------------------------
# Defect 2 — quality_preset now carries its RESOLUTION, not just its CRF
# ---------------------------------------------------------------------------


class TestPresetResolution:
    @pytest.mark.parametrize("name", sorted(QUALITY_PRESETS))
    def test_explicit_preset_seeds_its_own_viewport(self, name):
        preset = QUALITY_PRESETS[name]
        cfg = _make_recorder()._resolve_config({"quality_preset": name})
        assert (cfg["width"], cfg["height"]) == (preset.viewport_width, preset.viewport_height)

    def test_2k_is_1440p_not_a_square_1080(self):
        cfg = _make_recorder()._resolve_config({"quality_preset": "2k"})
        assert (cfg["width"], cfg["height"]) == (2560, 1440)
        assert cfg["crf"] == 18

    def test_4k_is_2160p_not_a_square_1080(self):
        cfg = _make_recorder()._resolve_config({"quality_preset": "4k"})
        assert (cfg["width"], cfg["height"]) == (3840, 2160)

    def test_explicit_viewport_outranks_the_preset(self):
        """Precedence is explicit > preset: the compiler hand-passes ctx.target_w/h."""
        cfg = _make_recorder()._resolve_config(
            {"quality_preset": "4k", "viewport_width": 1920, "viewport_height": 1080}
        )
        assert (cfg["width"], cfg["height"]) == (1920, 1080)
        # ...and the preset's CRF still applies.
        assert cfg["crf"] == QUALITY_PRESETS["4k"].crf

    def test_one_sided_override_leaves_the_other_axis_on_the_preset(self):
        cfg = _make_recorder()._resolve_config({"quality_preset": "2k", "viewport_width": 1920})
        assert (cfg["width"], cfg["height"]) == (1920, 1440)

    def test_default_tier_output_is_unchanged_1080_square(self):
        """No quality_preset in the request at all: the seed templates' size."""
        cfg = _make_recorder()._resolve_config(None)
        assert (cfg["width"], cfg["height"]) == (ar_mod._DEFAULT_WIDTH, ar_mod._DEFAULT_HEIGHT)
        assert (cfg["width"], cfg["height"]) == (1080, 1080)
        assert _make_recorder()._resolve_config({})["width"] == 1080
