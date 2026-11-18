"""Unit tests for compiler service -- real behavior, not mock echoes.

Replaces the old unittest.TestCase-based file that mixed concerns from
ProbeResult, FFmpegOps, title cards, video editor, and storage into a
single file with mostly trivial assertions.  Each class below tests ONE
thing with assertions that would break if the production code changed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.compilation.title_card_generator import _HEX_COLOR_RE, _hex_to_rgb
from src.services.compilation.video_editor import VideoEditor
from src.services.ffmpeg import FFmpegOps, ProbeResult

# ==========================================================================
# ProbeResult -- verify dataclass fields and edge cases
# ==========================================================================


class TestProbeResult:
    def test_fields_assigned_correctly(self):
        result = ProbeResult(duration=10.5, width=1920, height=1080, has_audio=True)
        assert result.duration == 10.5
        assert result.width == 1920
        assert result.height == 1080
        assert result.has_audio is True

    def test_zero_values_are_valid(self):
        result = ProbeResult(duration=0.0, width=0, height=0, has_audio=False)
        assert result.duration == 0.0
        assert result.width == 0
        assert result.height == 0
        assert result.has_audio is False

    def test_equality(self):
        a = ProbeResult(duration=5.0, width=640, height=480, has_audio=True)
        b = ProbeResult(duration=5.0, width=640, height=480, has_audio=True)
        assert a == b

    def test_inequality_on_different_values(self):
        a = ProbeResult(duration=5.0, width=640, height=480, has_audio=True)
        b = ProbeResult(duration=5.0, width=640, height=480, has_audio=False)
        assert a != b


# ==========================================================================
# FFmpegOps._build_atempo_chain -- verify actual chain output
# ==========================================================================


class TestBuildAtempoChain:
    def test_identity_factor_returns_literal(self):
        assert FFmpegOps._build_atempo_chain(1.0) == "atempo=1.0"

    def test_normal_2x_single_filter(self):
        chain = FFmpegOps._build_atempo_chain(2.0)
        assert chain == "atempo=2.0"

    def test_half_speed_single_filter(self):
        chain = FFmpegOps._build_atempo_chain(0.5)
        assert chain == "atempo=0.5"

    def test_very_slow_factor_chains_multiple_filters(self):
        """Factor 0.25 requires two atempo=0.5 filters (0.5 * 0.5 = 0.25)."""
        chain = FFmpegOps._build_atempo_chain(0.25)
        parts = chain.split(",")
        assert len(parts) == 2
        assert all(p.startswith("atempo=") for p in parts)
        # Verify the product of factors approximately equals 0.25
        product = 1.0
        for p in parts:
            product *= float(p.split("=")[1])
        assert abs(product - 0.25) < 0.01

    def test_very_fast_factor_chains_multiple_filters(self):
        """Factor 4.0 requires: atempo=2.0,atempo=2.0 (since max per step is 100)."""
        chain = FFmpegOps._build_atempo_chain(4.0)
        parts = chain.split(",")
        product = 1.0
        for p in parts:
            product *= float(p.split("=")[1])
        assert abs(product - 4.0) < 0.01

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            FFmpegOps._build_atempo_chain(0)

    def test_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            FFmpegOps._build_atempo_chain(-1.5)

    def test_very_small_factor_clamped_to_0_25(self):
        """Factor < 0.25 should be clamped, producing the same chain as 0.25."""
        result_clamped = FFmpegOps._build_atempo_chain(0.1)
        result_min = FFmpegOps._build_atempo_chain(0.25)
        assert result_clamped == result_min


# ==========================================================================
# _hex_to_rgb -- validate color parsing produces correct RGB tuples
# ==========================================================================


class TestHexToRgb:
    def test_six_digit_primary_colors(self):
        assert _hex_to_rgb("#FF0000") == (255, 0, 0)
        assert _hex_to_rgb("#00FF00") == (0, 255, 0)
        assert _hex_to_rgb("#0000FF") == (0, 0, 255)

    def test_three_digit_expands_correctly(self):
        assert _hex_to_rgb("#FFF") == (255, 255, 255)
        assert _hex_to_rgb("#000") == (0, 0, 0)
        assert _hex_to_rgb("#ABC") == (0xAA, 0xBB, 0xCC)

    def test_without_hash_prefix(self):
        assert _hex_to_rgb("FF0000") == (255, 0, 0)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _hex_to_rgb("")

    def test_garbage_input_raises(self):
        with pytest.raises(ValueError):
            _hex_to_rgb("not-a-color")


class TestHexColorRegex:
    def test_matches_valid_hex_patterns(self):
        for pattern in ("#FFF", "#FFFFFF", "ABC", "AABBCC"):
            assert _HEX_COLOR_RE.match(pattern), f"Expected match for {pattern}"

    def test_rejects_invalid_hex_characters(self):
        assert _HEX_COLOR_RE.match("#GGG") is None
        assert _HEX_COLOR_RE.match("XYZXYZ") is None


# ==========================================================================
# TitleCardGenerator -- verify image dimensions and file output
# ==========================================================================


class TestTitleCardGenerator:
    @patch("src.config.get_settings")
    def test_output_path_writes_valid_png(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        from src.services.compilation.title_card_generator import TitleCardGenerator

        gen = TitleCardGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "title.png"
            gen.generate_image("Test", output_path=out, width=320, height=180)
            assert out.exists()
            assert out.stat().st_size > 0
            # Verify it's valid by re-opening with PIL
            from PIL import Image

            reopened = Image.open(out)
            assert reopened.size == (320, 180)


# ==========================================================================
# VideoEditor -- verify path computation and trim thresholds
# ==========================================================================


class TestVideoEditor:
    def test_edited_path_produces_correct_stem_and_parent(self):
        editor = VideoEditor(output_dir=Path("/tmp/test_edit"))
        result = editor._edited_path(Path("/input/video.mp4"), "trimmed")
        assert result.name == "video-trimmed.mp4"
        assert result.parent == Path("/tmp/test_edit").resolve()

    def test_edited_path_preserves_suffix(self):
        editor = VideoEditor(output_dir=Path("/tmp/test_edit"))
        result = editor._edited_path(Path("/input/clip.webm"), "speed2")
        assert result.suffix == ".webm"
        assert result.stem == "clip-speed2"

    def test_default_crf_is_20(self):
        editor = VideoEditor()
        assert editor.crf == 20

    def test_custom_crf_is_stored(self):
        editor = VideoEditor(crf=18)
        assert editor.crf == 18


# ==========================================================================
# StorageService -- real async store/retrieve with filesystem
# ==========================================================================
