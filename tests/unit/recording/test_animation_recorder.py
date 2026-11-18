"""Tests for src/services/recording/animation_recorder.py -- AnimationRecorder.

The browser capture is mocked (mirroring test_content_recorder.py); the ffmpeg
frame-assembly is exercised for real (ffmpeg is a service dependency) on a tiny
PNG sequence so the encode path is verified end-to-end without a browser.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from playwright.async_api import Error as PlaywrightError

from src.services.ffmpeg import FFmpegOps
from src.services.recording.animation_recorder import (
    AnimationRecorder,
    _clamp_int,
    _inject_params,
    _resolve_duration_ms,
)


def _mock_settings():
    s = MagicMock()
    s.recording.headless = True
    s.recording.default_quality_preset = "standard"
    # Execution-timeout sizing knobs -- MUST be real numbers, not MagicMocks:
    # record_animation does arithmetic on them (frame_estimate *
    # seconds_per_frame, min/max against the floor/ceiling) before ever
    # touching a browser, so any test that calls record_animation exercises
    # this even when _capture_frames itself is mocked away.
    s.recording.animation_render_timeout_floor_seconds = 30.0
    s.recording.animation_render_timeout_ceiling_seconds = 18500.0
    s.recording.animation_render_seconds_per_frame = 4.5
    s.recording.animation_render_ffmpeg_seconds_per_frame = 0.5
    s.recording.animation_render_encode_reserve_seconds = 120.0
    s.recording.animation_render_stall_window_seconds = 60.0
    s.recording.animation_render_launch_timeout_seconds = 45.0
    return s


def _make_recorder():
    with patch(
        "src.services.recording.animation_recorder.get_settings",
        return_value=_mock_settings(),
    ):
        return AnimationRecorder()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestClampInt:
    def test_default_on_none(self):
        assert _clamp_int(None, 30, 1, 60) == 30

    def test_default_on_garbage(self):
        assert _clamp_int("abc", 30, 1, 60) == 30

    def test_clamps_high_and_low(self):
        assert _clamp_int(999, 30, 1, 60) == 60
        assert _clamp_int(0, 30, 1, 60) == 1


class TestResolveDuration:
    def test_request_wins(self):
        assert _resolve_duration_ms(3000, 16800) == 3000.0

    def test_falls_back_to_page(self):
        assert _resolve_duration_ms(None, 16800) == 16800.0

    def test_falls_back_to_default_when_both_absent(self):
        assert _resolve_duration_ms(None, None) == 5000.0

    def test_rejects_too_small(self):
        # Below the floor → skip to the next candidate (page), then default.
        assert _resolve_duration_ms(10, None) == 5000.0


# ---------------------------------------------------------------------------
# _inject_params — the params must reach window.PARAMS via the markup, because
# set_content does NOT fire add_init_script (the bug this guards against: every
# template silently renders defaults when params are ignored).
# ---------------------------------------------------------------------------


class TestInjectParams:
    def test_none_returns_unchanged(self):
        html = "<html><head></head><body></body></html>"
        assert _inject_params(html, None) == html

    def test_injects_into_head_before_template_script(self):
        html = "<html><head><style>x{}</style></head><body><script>1</script></body></html>"
        out = _inject_params(html, {"title": "Hi"})
        # The params script is the FIRST thing in <head>, before the template's
        # own <style>/<script>, so window.PARAMS exists when the template runs.
        head_idx = out.index("<head>")
        params_idx = out.index("window.PARAMS=")
        style_idx = out.index("<style>")
        assert head_idx < params_idx < style_idx
        # Round-trips as the exact object.
        blob = out[out.index("window.PARAMS=") + len("window.PARAMS=") : out.index(";</script>")]
        assert json.loads(blob) == {"title": "Hi"}

    def test_escapes_closing_tag_in_string_param(self):
        out = _inject_params("<html><head></head></html>", {"t": "</script><b>x"})
        # The literal </script> must be escaped so it can't close the tag early.
        assert "</script><b>" not in out.split(";</script>")[0]
        assert "<\\/script>" in out

    def test_prepends_when_no_head(self):
        out = _inject_params("<div>no head</div>", {"a": 1})
        assert out.startswith("<script>window.PARAMS=")


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_defaults_square_1080(self):
        cfg = _make_recorder()._resolve_config(None)
        assert cfg["width"] == 1080
        assert cfg["height"] == 1080
        assert cfg["color_scheme"] == "light"
        assert "crf" in cfg

    def test_viewport_overrides(self):
        cfg = _make_recorder()._resolve_config({"viewport_width": 1920, "viewport_height": 1080})
        assert cfg["width"] == 1920
        assert cfg["height"] == 1080


# ---------------------------------------------------------------------------
# _assemble dispatch
# ---------------------------------------------------------------------------


class TestAssembleDispatch:
    def test_mp4(self):
        with patch.object(FFmpegOps, "frames_to_mp4", return_value=Path("/x/out.mp4")) as m:
            out = AnimationRecorder._assemble(Path("/frames"), 30, "mp4", 23)
        m.assert_called_once()
        assert out == Path("/x/out.mp4")

    def test_gif(self):
        with patch.object(FFmpegOps, "frames_to_gif", return_value=Path("/x/out.gif")) as m:
            out = AnimationRecorder._assemble(Path("/frames"), 30, "gif", 23)
        # The encoder's path is what the caller downloads, so it has to come back
        # out of the dispatcher — and the container extension has to match the
        # requested format, not whatever the previous branch produced.
        assert out == Path("/x/out.gif")
        assert m.call_args.args == (Path("/frames"), Path("/frames/out.gif"))
        assert m.call_args.kwargs == {"fps": 30, "timeout": None}

    def test_webm(self):
        with patch.object(FFmpegOps, "frames_to_webm", return_value=Path("/x/out.webm")) as m:
            out = AnimationRecorder._assemble(Path("/frames"), 30, "webm", 23)
        assert out == Path("/x/out.webm")
        assert m.call_args.args == (Path("/frames"), Path("/frames/out.webm"))
        # transparent defaults to False here: an alpha encode of opaque frames
        # is a different (and much slower) codepath.
        assert m.call_args.kwargs == {"fps": 30, "transparent": False, "timeout": None}

    def test_webm_forwards_the_alpha_request(self):
        """transparent=True must reach the encoder — dropping it ships an opaque clip."""
        with patch.object(FFmpegOps, "frames_to_webm", return_value=Path("/x/out.webm")) as m:
            out = AnimationRecorder._assemble(Path("/frames"), 30, "webm", 23, transparent=True)
        assert out == Path("/x/out.webm")
        assert m.call_args.kwargs["transparent"] is True

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="output_format"):
            AnimationRecorder._assemble(Path("/frames"), 30, "avi", 23)


# ---------------------------------------------------------------------------
# record_animation orchestration (browser mocked)
# ---------------------------------------------------------------------------


class TestRecordAnimation:
    @pytest.mark.asyncio
    async def test_success_shape(self, tmp_path):
        recorder = _make_recorder()
        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)

        probe = MagicMock(duration=3.0, width=1080, height=1080, has_audio=True)
        with (
            patch.object(
                recorder,
                "_capture_frames",
                new=AsyncMock(return_value=(tmp_path, 90, 3000.0, 0.8)),
            ),
            patch.object(recorder, "_assemble", return_value=out),
            patch.object(FFmpegOps, "probe_all", return_value=probe),
        ):
            result = await recorder.record_animation(
                "job1", "<html></html>", {"fps": 30, "output_format": "mp4"}
            )

        assert result["success"] is True
        assert result["video_path"] == str(out)
        assert result["duration_seconds"] == 3.0
        assert result["width"] == 1080
        assert result["file_size_bytes"] == 64

    @pytest.mark.asyncio
    async def test_playwright_error_caught(self):
        recorder = _make_recorder()
        with patch.object(
            recorder,
            "_capture_frames",
            new=AsyncMock(side_effect=PlaywrightError("boom")),
        ):
            result = await recorder.record_animation("job2", "<html></html>", {})
        assert result["success"] is False
        assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# Frame-assembly ffmpeg arg shape (mocked run — always-on, matches the repo's
# test_ffmpeg_ops delegation-test convention; the lint CI runner has no ffmpeg).
# ---------------------------------------------------------------------------


class TestFrameAssemblyArgs:
    def test_frames_to_mp4_args(self, tmp_path):
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_mp4(tmp_path, tmp_path / "out.mp4", fps=24, crf=20)
        args = run.call_args.args[0]
        assert "-framerate" in args and "24" in args
        assert "libx264" in args and "yuv420p" in args
        assert any("frame_%06d.png" in a for a in args)

    def test_frames_to_gif_two_pass(self, tmp_path):
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_gif(tmp_path, tmp_path / "out.gif", fps=12)
        # palettegen pass then paletteuse pass.
        first, second = run.call_args_list[0].args[0], run.call_args_list[1].args[0]
        assert any("palettegen" in a for a in first)
        assert any("paletteuse" in a for a in second)

    def test_frames_to_webm_vp9(self, tmp_path):
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_webm(tmp_path, tmp_path / "out.webm", fps=30)
        args = run.call_args.args[0]
        assert "libvpx-vp9" in args


# ---------------------------------------------------------------------------
# Real ffmpeg frame-assembly (no browser): a 2-frame sequence → MP4 / GIF.
# Skipped where ffmpeg is absent locally; FAILS in CI (see `require_ffmpeg` in
# conftest.py) so the encode is always proven end-to-end there.
# ---------------------------------------------------------------------------


def _write_frames(frames_dir: Path, n: int, size: tuple[int, int], color: str) -> None:
    for i in range(n):
        Image.new("RGB", size, color).save(frames_dir / f"frame_{i:06d}.png")


@pytest.mark.usefixtures("require_ffmpeg")
class TestRealFrameAssembly:
    def test_frames_to_mp4_produces_valid_video(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d)
            _write_frames(frames_dir, 2, (320, 240), "navy")
            out = FFmpegOps.frames_to_mp4(frames_dir, frames_dir / "out.mp4", fps=2, crf=23)
            assert out.exists()
            assert out.stat().st_size > 0
            probe = FFmpegOps.probe_all(out)
            assert probe.width == 320
            assert probe.height == 240

    def test_frames_to_mp4_forces_even_dimensions(self):
        # Odd dims (321x241) would fail yuv420p without the even-scale filter.
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d)
            _write_frames(frames_dir, 2, (321, 241), "teal")
            out = FFmpegOps.frames_to_mp4(frames_dir, frames_dir / "out.mp4", fps=2)
            probe = FFmpegOps.probe_all(out)
            assert probe.width % 2 == 0
            assert probe.height % 2 == 0

    def test_frames_to_gif_produces_valid_gif(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d)
            _write_frames(frames_dir, 2, (160, 120), "coral")
            out = FFmpegOps.frames_to_gif(frames_dir, frames_dir / "out.gif", fps=2)
            assert out.exists()
            assert out.stat().st_size > 0
            with Image.open(out) as img:
                assert img.format == "GIF"
            # The two-pass palette temp file is cleaned up.
            assert not (frames_dir / "out.palette.png").exists()


# ---------------------------------------------------------------------------
# max_frame_seconds telemetry -- the per-frame data the stall window was NOT
# actually sized from (see Settings.animation_render_stall_window_seconds).
# ---------------------------------------------------------------------------


class _TimedFakePage:
    """Page double whose screenshot call can be made to take real wall time,
    per-frame, via ``delays`` (seconds, indexed by frame number)."""

    def __init__(self, delays: dict[int, float]) -> None:
        self._delays = delays
        self._frame = 0

    async def set_content(self, *_args, **_kwargs) -> None:
        return None

    async def evaluate(self, *_args) -> object:
        # `window.RENDER_DURATION_MS || null` -> None so the caller's
        # requested_duration_ms wins; `document.fonts && document.fonts.ready`
        # also lands here and its return value is discarded either way.
        return None

    async def screenshot(self, *, path: str, **_kwargs) -> None:
        import asyncio as _asyncio

        delay = self._delays.get(self._frame, 0.0)
        if delay:
            await _asyncio.sleep(delay)
        Path(path).write_bytes(b"png")
        self._frame += 1


class _TimedFakeContext:
    def __init__(self, page: _TimedFakePage) -> None:
        self._page = page

    async def new_page(self) -> _TimedFakePage:
        return self._page

    async def close(self) -> None:
        return None


class _TimedFakeBrowser:
    def __init__(self, context: _TimedFakeContext) -> None:
        self._context = context

    async def new_context(self, **_kwargs) -> _TimedFakeContext:
        return self._context

    async def close(self) -> None:
        return None


class _TimedFakePlaywright:
    def __init__(self, browser: _TimedFakeBrowser) -> None:
        self.chromium = MagicMock()
        self.chromium.launch = AsyncMock(return_value=browser)

    async def __aenter__(self) -> _TimedFakePlaywright:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class TestMaxFrameSecondsTelemetry:
    """``_capture_frames`` must record the slowest single frame's own capture
    time (JS seek + screenshot), not just complete a render successfully --
    this is the per-frame data the stall window's 90s default is explicitly
    NOT derived from (there was none before this), so it has to actually be
    measured and returned for a future retune to use real numbers.
    """

    @pytest.mark.asyncio
    async def test_records_the_slowest_frame_not_the_first_or_last(self, tmp_path):
        # Frame 0 fast, frame 1 the slow outlier, frame 2 fast again -- proves
        # the max is a genuine max over the whole loop, not "whatever the last
        # frame happened to take" or "the first frame's cold-start cost".
        page = _TimedFakePage({1: 0.05})
        browser = _TimedFakeBrowser(_TimedFakeContext(page))
        recorder = _make_recorder()
        config = {"width": 64, "height": 64, "capture_scale": 1, "headless": True}

        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_TimedFakePlaywright(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
        ):
            (
                _frames_dir,
                frame_count,
                _duration_ms,
                max_frame_seconds,
            ) = await recorder._capture_frames(
                "job-timed", tmp_path, "<html></html>", config, 30, 100, None, None
            )

        assert frame_count == 3
        # Comfortably above the 0.05s injected delay's floor, comfortably
        # below the untimed frames' cost, so this cannot pass by accident.
        assert max_frame_seconds >= 0.04
        assert max_frame_seconds < 0.5

    @pytest.mark.asyncio
    async def test_zero_when_every_frame_is_effectively_instant(self, tmp_path):
        page = _TimedFakePage({})
        browser = _TimedFakeBrowser(_TimedFakeContext(page))
        recorder = _make_recorder()
        config = {"width": 64, "height": 64, "capture_scale": 1, "headless": True}

        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_TimedFakePlaywright(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
        ):
            (
                _frames_dir,
                _frame_count,
                _duration_ms,
                max_frame_seconds,
            ) = await recorder._capture_frames(
                "job-instant", tmp_path, "<html></html>", config, 30, 100, None, None
            )

        assert max_frame_seconds < 0.05

    @pytest.mark.asyncio
    async def test_reaches_the_result_dict_on_a_successful_render(self, tmp_path):
        """End-to-end through record_animation: the value _capture_frames
        measured must survive into the returned result, since that is what
        recorder_service persists onto the job row."""
        page = _TimedFakePage({0: 0.05})
        browser = _TimedFakeBrowser(_TimedFakeContext(page))
        recorder = _make_recorder()

        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)
        probe = MagicMock(duration=1.0, width=64, height=64, has_audio=False)

        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_TimedFakePlaywright(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
            patch.object(recorder, "_assemble", return_value=out),
            patch.object(FFmpegOps, "probe_all", return_value=probe),
        ):
            result = await recorder.record_animation(
                "job-e2e-timed",
                "<html></html>",
                {"fps": 30, "duration_ms": 66, "output_format": "mp4"},
            )

        assert result["success"] is True
        assert result["max_frame_seconds"] >= 0.04

    @pytest.mark.asyncio
    async def test_mutation_removing_the_measurement_makes_it_always_zero(self, tmp_path):
        """Mutation proof: with the ``frame_start``/``max()`` tracking
        deleted, the loop would still complete and return SOME 4th tuple
        element (or fail to unpack) -- prove the specific claim ("the slowest
        frame is measured") by checking against a hand-rolled version of the
        loop with the measurement stripped, mirroring the real
        ``_capture_frames`` body exactly except for that one omission.
        """
        page = _TimedFakePage({0: 0.05})

        # The exact per-frame body from AnimationRecorder._capture_frames,
        # minus max_frame_seconds tracking -- the mutation under test.
        async def _unmeasured_loop(page, frame_count, stall_window_s):
            for i in range(frame_count):
                await asyncio.wait_for(
                    AnimationRecorder._capture_one_frame(
                        page, i * 1000.0 / 30, tmp_path / f"frame_{i:06d}.png", False
                    ),
                    timeout=stall_window_s,
                )
            return 0.0  # no telemetry at all -- the pre-fix behaviour

        mutated = await _unmeasured_loop(page, 2, 5.0)
        assert mutated == 0.0  # the mutation always reports zero, hiding the 0.05s frame

        recorder = _make_recorder()
        browser = _TimedFakeBrowser(_TimedFakeContext(_TimedFakePage({0: 0.05})))
        config = {"width": 64, "height": 64, "capture_scale": 1, "headless": True}
        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_TimedFakePlaywright(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
        ):
            _fd, _fc, _dm, real_max = await recorder._capture_frames(
                "job-real", tmp_path, "<html></html>", config, 30, 66, None, None
            )
        # The real implementation, unlike the mutation, actually reports it.
        assert real_max >= 0.04


class TestRenderTimingSummary:
    """One ``animation_render_timing`` INFO line per recording, carrying the
    frame count and per-phase wall time -- the evidence needed to say where a
    slow render's time goes before touching any quality setting."""

    @pytest.mark.asyncio
    async def test_success_emits_one_summary_line_with_frame_count(self, tmp_path, caplog):
        # Frame 0 screenshot sleeps 50ms, so screenshot_s must be non-zero.
        page = _TimedFakePage({0: 0.05})
        browser = _TimedFakeBrowser(_TimedFakeContext(page))
        recorder = _make_recorder()
        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)
        probe = MagicMock(duration=1.0, width=64, height=64, has_audio=False)

        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_TimedFakePlaywright(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
            patch.object(recorder, "_assemble", return_value=out),
            patch.object(FFmpegOps, "probe_all", return_value=probe),
            caplog.at_level("INFO", logger="src.services.recording.animation_recorder"),
        ):
            result = await recorder.record_animation(
                "job-timing",
                "<html></html>",
                {"fps": 30, "duration_ms": 100, "output_format": "mp4"},
            )

        assert result["success"] is True
        lines = [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith("animation_render_timing ") and r.levelname == "INFO"
        ]
        assert len(lines) == 1, lines
        line = lines[0]
        assert "job_id=job-timing" in line
        assert "success=True" in line
        assert " frames=3 " in line  # 100ms @ 30fps
        assert "capture_scale=2" in line
        fields = dict(tok.split("=", 1) for tok in line.split()[1:])
        assert float(fields["screenshot_s"]) >= 0.04
        assert float(fields["per_frame_ms"]) > 0.0
        for key in ("total_s", "setup_s", "seek_s", "downscale_s", "encode_s", "probe_s"):
            assert float(fields[key]) >= 0.0

    def test_assemble_accumulates_downscale_and_encode(self, tmp_path):
        timings = {"downscale_s": 0.0, "encode_s": 0.0}
        scaled = tmp_path / "scaled"
        with (
            patch.object(AnimationRecorder, "_downscale_frames", return_value=scaled) as down,
            patch.object(FFmpegOps, "frames_to_mp4", return_value=scaled / "out.mp4") as enc,
            patch("src.services.recording.animation_recorder.time.perf_counter") as clock,
        ):
            clock.side_effect = [0.0, 2.0, 10.0, 15.0]
            AnimationRecorder._assemble(
                tmp_path,
                30,
                "mp4",
                23,
                target_width=64,
                target_height=64,
                capture_scale=2,
                timings=timings,
            )
        assert down.called and enc.called
        assert timings == {"downscale_s": 2.0, "encode_s": 5.0}
