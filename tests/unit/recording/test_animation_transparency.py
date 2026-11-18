"""Transparent (alpha) motion-graphics capture: schema, argv, and capability probe.

Three distinct failure modes are pinned here, because all three are silent:

1. **Regression on the opaque path.** ``transparent=False`` must produce the
   pre-alpha argv, plus the BT.709 tag triple that every output in this service
   now carries — so the webm case asserts the full list, not a membership
   check.
2. **Silent flattening.** A transparent request that lands in h264/gif comes
   back looking like a successful job and only reveals itself as an opaque box
   once composited. Both the API boundary and the encoder dispatch must refuse.
3. **Silent alpha drop.** An ffmpeg build can accept ``-pix_fmt yuva420p``,
   exit 0, and write a file with no alpha in it. The capability probe's verdict
   is exercised here in both directions on canned readings, and the readings
   themselves are taken against the real binary in
   :class:`TestRealVp9AlphaRoundTrip`.

Note what is deliberately NOT asserted anywhere: that the encoded stream's
``pix_fmt`` is ``yuva420p``. VP9 keeps alpha as WebM ``BlockAdditional`` side
data rather than a fourth plane, so a correct alpha file reports ``yuv420p``
and that assertion would fail on every build in existence.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.schemas.recording import RecordAnimationRequest
from src.services.ffmpeg import COLOR_TAG_ARGS, FFmpegOps
from src.services.ffmpeg import alpha_probe as alpha_probe_mod
from src.services.recording.animation_recorder import (
    AnimationRecorder,
    _inject_transparent_style,
)
from src.services.recording.templates import OVERLAY_SAFE_TEMPLATES, TEMPLATES


@pytest.fixture(autouse=True)
def _clear_alpha_cache():
    """The capability result is memoised process-wide; isolate every test."""
    alpha_probe_mod.reset_cache_for_tests()
    yield
    alpha_probe_mod.reset_cache_for_tests()


@pytest.fixture
def _alpha_supported():
    """Pretend this build handles VP9 alpha (the recorder gate is not under test)."""
    with patch.object(alpha_probe_mod, "probe_vp9_alpha_support", return_value=True):
        yield


def _mock_settings():
    s = MagicMock()
    s.recording.headless = True
    s.recording.default_quality_preset = "standard"
    # Real numbers required: record_animation does arithmetic on these before
    # touching a browser (see test_animation_recorder.py's identical comment).
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
# frames_to_webm argv — the opaque default is pinned byte-for-byte.
# ---------------------------------------------------------------------------


class TestFramesToWebmArgs:
    def test_opaque_argv_is_unchanged(self, tmp_path):
        """transparent=False must carry none of the alpha-path argv, byte-for-byte.

        The BT.709 tag triple is spliced in from ``COLOR_TAG_ARGS`` and is
        expected on BOTH branches: ``-color_primaries``/``-color_trc``/
        ``-colorspace`` are generic AVCodecContext options, so libvpx-vp9
        honours them and the WebM muxer writes them into the Colour element
        (verified against the real binary on both ``yuv420p`` and
        ``yuva420p``). This is NOT the stream-copy ``h264_metadata`` bsf, which
        would hard-error on a VP9 stream and is gated on a probed codec
        elsewhere. Splicing it from the constant rather than literal strings
        means dropping the tags from the builder fails this test.
        """
        out = tmp_path / "out.webm"
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_webm(tmp_path, out, fps=30, crf=32)
        assert run.call_args.args[0] == [
            "-framerate",
            "30",
            "-i",
            str(tmp_path / "frame_%06d.png"),
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "0",
            "-crf",
            "32",
            *COLOR_TAG_ARGS,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            "30",
            str(out),
        ]

    def test_default_matches_explicit_false(self, tmp_path):
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_webm(tmp_path, tmp_path / "a.webm", fps=24)
            FFmpegOps.frames_to_webm(tmp_path, tmp_path / "a.webm", fps=24, transparent=False)
        default_argv, explicit_argv = (c.args[0] for c in run.call_args_list)
        assert default_argv == explicit_argv
        # Equality alone would also hold if BOTH had silently gone alpha, which
        # is the failure this guards: the default must be the opaque encode.
        assert default_argv[default_argv.index("-pix_fmt") + 1] == "yuv420p"
        assert "yuva420p" not in default_argv
        assert "-auto-alt-ref" not in default_argv

    def test_transparent_argv_carries_alpha(self, tmp_path):
        out = tmp_path / "out.webm"
        with patch.object(FFmpegOps, "run") as run:
            FFmpegOps.frames_to_webm(tmp_path, out, fps=30, crf=32, transparent=True)
        args = run.call_args.args[0]
        # The encoder is asked for an alpha format...
        assert args[args.index("-pix_fmt") + 1] == "yuva420p"
        # ...AND the filtergraph is pinned to one, so format negotiation cannot
        # quietly resolve the scale filter's output to plain yuv420p.
        assert args[args.index("-vf") + 1].endswith(",format=yuva420p")
        assert args[args.index("-vf") + 1].startswith("scale=trunc(iw/2)*2:trunc(ih/2)*2")
        # Alt-ref frames have no alpha counterpart in libvpx.
        assert args[args.index("-auto-alt-ref") + 1] == "0"
        assert args[-1] == str(out)


# ---------------------------------------------------------------------------
# _assemble — transparent must never reach an alpha-less container.
# ---------------------------------------------------------------------------


class TestAssembleRejectsOpaqueContainers:
    @pytest.mark.parametrize("fmt", ["mp4", "gif"])
    def test_rejects(self, fmt):
        with pytest.raises(ValueError, match="requires output_format='webm'"):
            AnimationRecorder._assemble(Path("/frames"), 30, fmt, 23, True)

    def test_webm_passes_flag_through(self):
        with patch.object(FFmpegOps, "frames_to_webm", return_value=Path("/x/out.webm")) as m:
            AnimationRecorder._assemble(Path("/frames"), 30, "webm", 23, True)
        assert m.call_args.kwargs["transparent"] is True

    def test_opaque_webm_still_defaults_to_false(self):
        with patch.object(FFmpegOps, "frames_to_webm", return_value=Path("/x/out.webm")) as m:
            AnimationRecorder._assemble(Path("/frames"), 30, "webm", 23)
        assert m.call_args.kwargs["transparent"] is False


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class TestRecordAnimationRequestTransparent:
    def test_defaults_to_false_and_leaves_format_alone(self):
        req = RecordAnimationRequest(template_id="lower_third", part_id="p1")
        assert req.transparent is False
        assert req.output_format == "mp4"

    def test_unset_format_is_upgraded_to_webm(self):
        req = RecordAnimationRequest(template_id="lower_third", part_id="p1", transparent=True)
        assert req.output_format == "webm"

    def test_explicit_webm_accepted(self):
        req = RecordAnimationRequest(
            template_id="lower_third", part_id="p1", transparent=True, output_format="webm"
        )
        assert req.output_format == "webm"

    @pytest.mark.parametrize("fmt", ["mp4", "gif"])
    def test_explicit_alpha_less_format_rejected(self, fmt):
        """Contradictory instructions get a 422 — never a quietly opaque clip."""
        with pytest.raises(ValidationError, match="requires output_format='webm'"):
            RecordAnimationRequest(
                template_id="lower_third", part_id="p1", transparent=True, output_format=fmt
            )

    def test_opaque_mp4_request_unaffected(self):
        req = RecordAnimationRequest(
            template_id="animated_title_card", part_id="p1", output_format="mp4"
        )
        assert req.output_format == "mp4"
        assert req.transparent is False


# ---------------------------------------------------------------------------
# Overlay-safe template allow-list
# ---------------------------------------------------------------------------


class TestOverlaySafeTemplates:
    def test_lower_third_is_overlay_safe(self):
        assert "lower_third" in OVERLAY_SAFE_TEMPLATES

    def test_every_entry_is_a_registered_template(self):
        assert set(TEMPLATES) >= OVERLAY_SAFE_TEMPLATES

    def test_full_frame_cards_are_excluded(self):
        # These paint their own opaque stage; alpha capture cannot help them.
        for tid in ("animated_title_card", "animated_slide", "social_card", "ken_burns"):
            assert tid not in OVERLAY_SAFE_TEMPLATES


# ---------------------------------------------------------------------------
# Backdrop override
# ---------------------------------------------------------------------------


class TestInjectTransparentStyle:
    def test_override_is_last_in_head(self):
        html = "<html><head><style>body{background:#0B1437}</style></head><body></body></html>"
        out = _inject_transparent_style(html)
        # Later in source order than the template's own rule, and !important on
        # top — either alone would be enough, both together are unambiguous.
        assert out.index("background:#0B1437") < out.index("background:transparent !important")
        assert out.index("background:transparent !important") < out.index("</head>")
        assert "background-image:none !important" in out

    def test_prepends_when_no_head(self):
        out = _inject_transparent_style("<div>bare</div>")
        assert out.startswith("<style>html,body{background:transparent")


# ---------------------------------------------------------------------------
# Capture path: omit_background must track the flag.
# ---------------------------------------------------------------------------


class _FakePage:
    def __init__(self) -> None:
        self.screenshots: list[dict] = []
        self.document = ""

    async def set_content(self, html, **_kwargs) -> None:
        self.document = html

    async def evaluate(self, _script, *_args) -> None:
        return None

    async def screenshot(self, **kwargs) -> None:
        self.screenshots.append(kwargs)
        Path(kwargs["path"]).write_bytes(b"png")


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def new_page(self) -> _FakePage:
        return self._page

    async def close(self) -> None:
        return None


class _FakeBrowser:
    def __init__(self, context: _FakeContext) -> None:
        self._context = context

    async def new_context(self, **_kwargs) -> _FakeContext:
        return self._context

    async def close(self) -> None:
        return None


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser) -> None:
        self.chromium = MagicMock()
        self.chromium.launch = AsyncMock(return_value=browser)

    async def __aenter__(self) -> _FakePlaywright:
        return self

    async def __aexit__(self, *_exc) -> None:
        return None


async def _capture_once(*, transparent: bool, tmp_path: Path) -> _FakePage:
    page = _FakePage()
    fake = _FakePlaywright(_FakeBrowser(_FakeContext(page)))
    recorder = _make_recorder()
    with (
        patch("playwright.async_api.async_playwright", return_value=fake),
        patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
    ):
        await recorder._capture_frames(
            "job-t",
            tmp_path,
            "<html><head><style>body{background:#000}</style></head><body></body></html>",
            {"width": 320, "height": 240, "headless": True, "color_scheme": "light"},
            1,
            1000,
            None,
            None,
            transparent=transparent,
        )
    return page


class TestCaptureFramesTransparency:
    @pytest.mark.asyncio
    async def test_transparent_omits_background_and_clears_css(self, tmp_path):
        page = await _capture_once(transparent=True, tmp_path=tmp_path)
        assert page.screenshots
        assert all(s["omit_background"] is True for s in page.screenshots)
        # Chromium's white base alone is not enough — the document's own paint
        # must be overridden too, or the frames come back opaque anyway.
        assert "background:transparent !important" in page.document

    @pytest.mark.asyncio
    async def test_opaque_keeps_playwright_default_and_original_css(self, tmp_path):
        page = await _capture_once(transparent=False, tmp_path=tmp_path)
        assert all(s["omit_background"] is False for s in page.screenshots)
        assert "background:transparent !important" not in page.document


# ---------------------------------------------------------------------------
# Capability probe — the alpha-dropping ffmpeg build.
# ---------------------------------------------------------------------------


def _fake_measurement(alpha_mode: str, roundtrip_alpha: int):
    """Patch the probe's measurement step with a canned (tag, alpha) reading.

    The measurement itself shells out to ffmpeg twice; these tests are about
    the VERDICT the readings produce, which is where the silent-degradation
    bug would live.
    """
    return patch.object(
        alpha_probe_mod, "_measure_alpha_capability", return_value=(alpha_mode, roundtrip_alpha)
    )


class TestVp9AlphaCapabilityProbe:
    def test_reports_unsupported_when_alpha_is_silently_dropped(self, caplog):
        """The exact trap: encode exits 0, no tag, frame decodes back opaque."""
        with _fake_measurement("", 255), caplog.at_level("ERROR"):
            assert alpha_probe_mod.probe_vp9_alpha_support() is False
        assert "VP9 ALPHA UNSUPPORTED" in caplog.text
        assert "alpha=255" in caplog.text

    def test_tag_without_a_real_payload_is_still_unsupported(self):
        """A build that writes alpha_mode but decodes opaque does not pass."""
        with _fake_measurement("1", 255):
            assert alpha_probe_mod.probe_vp9_alpha_support() is False

    def test_payload_without_the_tag_is_still_unsupported(self):
        with _fake_measurement("", 129):
            assert alpha_probe_mod.probe_vp9_alpha_support() is False

    def test_reports_supported_on_a_real_alpha_roundtrip(self):
        # Both signals as measured against a genuine libvpx alpha encode.
        with _fake_measurement("1", 129):
            assert alpha_probe_mod.probe_vp9_alpha_support() is True

    def test_probe_failure_is_not_optimistic(self, caplog):
        """ffmpeg missing / encode blowing up must read as "no", never "probably"."""
        with (
            patch.object(
                alpha_probe_mod, "_measure_alpha_capability", side_effect=RuntimeError("no ffmpeg")
            ),
            caplog.at_level("ERROR"),
        ):
            assert alpha_probe_mod.probe_vp9_alpha_support() is False
        assert "could not complete" in caplog.text

    def test_result_is_cached(self):
        """The probe shells out to ffmpeg twice; once per process is the contract.

        The call count is the point here and stays — but on its own it is also
        satisfied by a cache that memoises the WRONG answer, so the returned
        verdict is asserted alongside it.
        """
        with _fake_measurement("1", 129) as m:
            first = alpha_probe_mod.probe_vp9_alpha_support()
            second = alpha_probe_mod.probe_vp9_alpha_support()
        assert m.call_count == 1
        assert first is True
        assert second is first

    def test_require_raises_with_a_readable_reason(self):
        with patch.object(alpha_probe_mod, "probe_vp9_alpha_support", return_value=False):
            with pytest.raises(alpha_probe_mod.AlphaCapabilityError, match="drops the alpha"):
                alpha_probe_mod.require_vp9_alpha_support()

    def test_require_is_quiet_when_supported(self):
        with patch.object(alpha_probe_mod, "probe_vp9_alpha_support", return_value=True):
            alpha_probe_mod.require_vp9_alpha_support()


# ---------------------------------------------------------------------------
# Real ffmpeg: the alpha actually survives the encode/decode round trip.
# Skipped where ffmpeg is absent locally; FAILS in CI (see `require_ffmpeg` in
# conftest.py), mirroring TestRealFrameAssembly in test_animation_recorder.py.
#
# This is the test that would catch a base-image regression: everything above
# is faked, so only this one touches the binary the product ships with.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("require_ffmpeg")
class TestRealVp9AlphaRoundTrip:
    def test_transparent_encode_carries_alpha(self, tmp_path):
        from PIL import Image

        for i in range(2):
            Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(tmp_path / f"frame_{i:06d}.png")
        out = FFmpegOps.frames_to_webm(tmp_path, tmp_path / "a.webm", fps=2, transparent=True)
        # VP9 stores alpha as WebM side data, so the STREAM's pix_fmt is
        # yuv420p even here — the container tag is the signal.
        assert FFmpegOps.probe_stream_tag(out, "alpha_mode") == "1"

    def test_opaque_encode_carries_no_alpha(self, tmp_path):
        """The control: without the flag there is no alpha payload at all."""
        from PIL import Image

        for i in range(2):
            Image.new("RGBA", (16, 16), (255, 0, 0, 128)).save(tmp_path / f"frame_{i:06d}.png")
        out = FFmpegOps.frames_to_webm(tmp_path, tmp_path / "b.webm", fps=2)
        assert FFmpegOps.probe_stream_tag(out, "alpha_mode") == ""

    def test_capability_probe_agrees_with_this_build(self):
        # End-to-end: real encode, real ffprobe, real libvpx decode.
        assert alpha_probe_mod.probe_vp9_alpha_support(force=True) is True


# ---------------------------------------------------------------------------
# record_animation end-to-end gating (browser + encode mocked).
# ---------------------------------------------------------------------------


class TestRecordAnimationTransparentGates:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("fmt", ["mp4", "gif"])
    async def test_direct_caller_cannot_get_a_silently_opaque_clip(self, fmt, _alpha_supported):
        """The compiler builds animation configs by hand; the gate holds there too."""
        recorder = _make_recorder()
        with patch.object(recorder, "_capture_frames", new=AsyncMock()) as capture:
            result = await recorder.record_animation(
                "job-x", "<html></html>", {"output_format": fmt, "transparent": True}
            )
        assert result["success"] is False
        assert "output_format='webm'" in result["error"]
        # Refused BEFORE the browser ran — no wasted capture.
        capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_incapable_build_refuses_before_capture(self, tmp_path):
        recorder = _make_recorder()
        with (
            patch.object(alpha_probe_mod, "probe_vp9_alpha_support", return_value=False),
            patch.object(recorder, "_capture_frames", new=AsyncMock()) as capture,
        ):
            result = await recorder.record_animation(
                "job-y", "<html></html>", {"output_format": "webm", "transparent": True}
            )
        assert result["success"] is False
        assert "transparent rendering is unavailable" in result["error"]
        capture.assert_not_called()

    @pytest.mark.asyncio
    async def test_transparent_webm_reaches_the_alpha_encode(self, tmp_path, _alpha_supported):
        recorder = _make_recorder()
        out = tmp_path / "out.webm"
        out.write_bytes(b"\x00" * 32)
        probe = MagicMock(duration=1.0, width=320, height=240, has_audio=False)
        with (
            patch.object(
                recorder,
                "_capture_frames",
                new=AsyncMock(return_value=(tmp_path, 30, 1000.0, 0.4)),
            ),
            patch.object(recorder, "_assemble", return_value=out) as assemble,
            patch.object(FFmpegOps, "probe_all", return_value=probe),
        ):
            result = await recorder.record_animation(
                "job-z", "<html></html>", {"output_format": "webm", "transparent": True}
            )
        assert result["success"] is True
        # _assemble is called positionally via asyncio.to_thread.
        assert assemble.call_args.args[-1] is True
