"""Unit tests for video color correction and overlay filters."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

_PATCH_FFMPEG_RUN = "src.services.compilation.video_filters.FFmpegOps.run"
_PATCH_FFMPEG_PROBE_HAS_AUDIO = "src.services.compilation.video_filters.FFmpegOps.probe_has_audio"


class TestColorCorrectionDataclass:
    def test_default_values(self):
        from src.services.compilation.video_filters import ColorCorrection

        cc = ColorCorrection()
        assert cc.brightness == 0.0
        assert cc.contrast == 1.0
        assert cc.saturation == 1.0
        assert cc.gamma == 1.0
        assert cc.temperature == 0.0

    def test_custom_values(self):
        from src.services.compilation.video_filters import ColorCorrection

        cc = ColorCorrection(
            brightness=0.1, contrast=1.2, saturation=0.8, gamma=0.9, temperature=0.5
        )
        assert cc.brightness == 0.1
        assert cc.contrast == 1.2
        assert cc.saturation == 0.8
        assert cc.gamma == 0.9
        assert cc.temperature == 0.5


class TestApplyColorCorrection:
    @patch(_PATCH_FFMPEG_RUN)
    def test_eq_filter_contains_all_parameters(self, mock_run):
        from src.services.compilation.video_filters import ColorCorrection, apply_color_correction

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output" / "corrected.mp4"

            cc = ColorCorrection(brightness=0.2, contrast=1.5, saturation=0.9, gamma=1.1)
            apply_color_correction(input_path, output_path, cc)

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            vf_index = args.index("-vf")
            vf_value = args[vf_index + 1]

            assert "brightness=0.2" in vf_value
            assert "contrast=1.5" in vf_value
            assert "saturation=0.9" in vf_value
            assert "gamma=1.1" in vf_value

    @patch(_PATCH_FFMPEG_RUN)
    def test_colorbalance_appended_when_temperature_nonzero(self, mock_run):
        from src.services.compilation.video_filters import ColorCorrection, apply_color_correction

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output" / "corrected.mp4"

            cc = ColorCorrection(temperature=0.5)
            apply_color_correction(input_path, output_path, cc)

            args = mock_run.call_args[0][0]
            vf_index = args.index("-vf")
            vf_value = args[vf_index + 1]

            assert "colorbalance" in vf_value
            # temperature=0.5 -> temp = 0.5 * 0.3 = 0.15
            assert "rs=0.15" in vf_value
            assert "bs=-0.15" in vf_value

    @patch(_PATCH_FFMPEG_RUN)
    def test_no_colorbalance_at_zero_temperature(self, mock_run):
        from src.services.compilation.video_filters import ColorCorrection, apply_color_correction

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "output" / "corrected.mp4"

            cc = ColorCorrection(temperature=0.0)
            apply_color_correction(input_path, output_path, cc)

            args = mock_run.call_args[0][0]
            vf_index = args.index("-vf")
            vf_value = args[vf_index + 1]

            assert "colorbalance" not in vf_value

    @patch(_PATCH_FFMPEG_RUN)
    def test_output_directory_created(self, mock_run):
        from src.services.compilation.video_filters import ColorCorrection, apply_color_correction

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.mp4"
            output_path = Path(tmpdir) / "deep" / "nested" / "output.mp4"

            cc = ColorCorrection()
            result = apply_color_correction(input_path, output_path, cc)

            assert output_path.parent.exists()
            assert result == output_path


class TestOverlayImage:
    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=True)
    @patch(_PATCH_FFMPEG_RUN)
    def test_filter_complex_built_correctly(self, mock_run, mock_probe):
        from src.services.compilation.video_filters import overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(
                input_path, overlay_path, output_path, position="center", opacity=0.5, scale=0.2
            )

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            fc_index = args.index("-filter_complex")
            fc_value = args[fc_index + 1]

            assert "scale=iw*0.2:-2:flags=lanczos" in fc_value
            assert "colorchannelmixer=aa=0.5" in fc_value
            assert "overlay=" in fc_value

    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=True)
    @patch(_PATCH_FFMPEG_RUN)
    def test_scale_uses_even_height_not_odd(self, mock_run, mock_probe):
        """``-1`` (odd-height-capable) breaks yuv420p; must be ``-2``."""
        from src.services.compilation.video_filters import overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(input_path, overlay_path, output_path, scale=0.2)

            args = mock_run.call_args[0][0]
            fc_value = args[args.index("-filter_complex") + 1]

            assert "scale=iw*0.2:-1," not in fc_value
            assert "flags=lanczos+accurate_rnd+full_chroma_int" in fc_value

    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=True)
    @patch(_PATCH_FFMPEG_RUN)
    def test_position_map_used_for_known_position(self, mock_run, mock_probe):
        from src.services.compilation.video_filters import POSITION_MAP, overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(input_path, overlay_path, output_path, position="top-left")

            args = mock_run.call_args[0][0]
            fc_index = args.index("-filter_complex")
            fc_value = args[fc_index + 1]

            assert f"overlay={POSITION_MAP['top-left']}" in fc_value

    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=True)
    @patch(_PATCH_FFMPEG_RUN)
    def test_fallback_to_bottom_right_for_unknown_position(self, mock_run, mock_probe):
        from src.services.compilation.video_filters import POSITION_MAP, overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(input_path, overlay_path, output_path, position="nonexistent-position")

            args = mock_run.call_args[0][0]
            fc_index = args.index("-filter_complex")
            fc_value = args[fc_index + 1]

            assert f"overlay={POSITION_MAP['bottom-right']}" in fc_value

    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=True)
    @patch(_PATCH_FFMPEG_RUN)
    def test_audio_copy_args_when_has_audio(self, mock_run, mock_probe):
        from src.services.compilation.video_filters import overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(input_path, overlay_path, output_path)

            args = mock_run.call_args[0][0]
            assert "-map" in args
            assert "0:a" in args
            assert "-c:a" in args
            assert "copy" in args

    @patch(_PATCH_FFMPEG_PROBE_HAS_AUDIO, return_value=False)
    @patch(_PATCH_FFMPEG_RUN)
    def test_no_audio_args_when_no_audio(self, mock_run, mock_probe):
        from src.services.compilation.video_filters import overlay_image

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "video.mp4"
            overlay_path = Path(tmpdir) / "watermark.png"
            output_path = Path(tmpdir) / "output" / "watermarked.mp4"

            overlay_image(input_path, overlay_path, output_path)

            args = mock_run.call_args[0][0]
            assert "0:a" not in args
            assert "-c:a" not in args


class TestPositionMap:
    def test_all_five_positions_exist(self):
        from src.services.compilation.video_filters import POSITION_MAP

        expected_keys = {"top-left", "top-right", "center", "bottom-left", "bottom-right"}
        assert set(POSITION_MAP.keys()) == expected_keys

    def test_position_values_are_non_empty_strings(self):
        from src.services.compilation.video_filters import POSITION_MAP

        for _key, value in POSITION_MAP.items():
            assert isinstance(value, str)
            assert len(value) > 0
