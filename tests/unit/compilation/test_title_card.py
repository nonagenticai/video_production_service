"""Unit tests for title card generator."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from src.services.compilation.title_card_generator import (
    _BRAND_FONT_BOLD,
    _BRAND_FONT_REGULAR,
    TitleCardGenerator,
    _get_font,
    _hex_to_rgb,
    _wrap_text,
)

# ---------------------------------------------------------------------------
# _get_font tests
# ---------------------------------------------------------------------------


class TestGetFont:
    @patch("src.config.get_settings")
    def test_get_font_default(self, mock_settings):
        mock_settings.return_value = MagicMock(theme=MagicMock(font_path="", font_bold_path=""))
        font = _get_font(24)
        assert font is not None

    @patch("src.config.get_settings")
    def test_get_font_bold(self, mock_settings):
        mock_settings.return_value = MagicMock(theme=MagicMock(font_path="", font_bold_path=""))
        font = _get_font(24, bold=True)
        assert font is not None

    @patch("src.config.get_settings")
    def test_get_font_configured_path_missing(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                font_path="/nonexistent/font.ttf",
                font_bold_path="/nonexistent/bold.ttf",
            )
        )
        font = _get_font(24)
        # Should fall back to system fonts or default
        assert font is not None


class TestBrandFont:
    """The PIL title-card path must render in the vendored Fraunces brand face."""

    def test_vendored_ttf_files_exist_and_load(self):
        from PIL import ImageFont

        for path in (_BRAND_FONT_REGULAR, _BRAND_FONT_BOLD):
            assert Path(path).is_file(), f"missing vendored brand font: {path}"
            font = ImageFont.truetype(path, 48)
            family = font.getname()[0]
            assert family is not None and family.startswith("Fraunces"), family

    @patch("src.config.get_settings")
    def test_get_font_defaults_to_fraunces(self, mock_settings):
        """With no THEME_FONT_* config, _get_font resolves the brand face, not
        the generic platform fallback or PIL's bitmap default."""
        from PIL import ImageFont

        mock_settings.return_value = MagicMock(theme=MagicMock(font_path="", font_bold_path=""))
        for bold in (False, True):
            font = _get_font(48, bold=bold)
            assert isinstance(font, ImageFont.FreeTypeFont)
            family = font.getname()[0]
            assert family is not None and family.startswith("Fraunces"), (bold, family)

    @patch("src.config.get_settings")
    def test_configured_path_still_overrides_brand(self, mock_settings):
        """An explicit THEME_FONT_PATH takes precedence over the vendored face."""
        from PIL import ImageFont

        mock_settings.return_value = MagicMock(
            theme=MagicMock(font_path=_BRAND_FONT_BOLD, font_bold_path=_BRAND_FONT_BOLD)
        )
        font = _get_font(48)
        assert isinstance(font, ImageFont.FreeTypeFont)
        family = font.getname()[0]
        assert family is not None and family.startswith("Fraunces")


# ---------------------------------------------------------------------------
# _hex_to_rgb tests (additional edge cases)
# ---------------------------------------------------------------------------


class TestHexToRgbEdgeCases:
    def test_8_digit_hex(self):
        # 8 hex digits include alpha channel, but only first 6 are used
        result = _hex_to_rgb("#FF0000FF")
        assert result == (255, 0, 0)

    def test_whitespace_handling(self):
        result = _hex_to_rgb("  #FF0000  ")
        assert result == (255, 0, 0)

    def test_none_raises(self):
        with pytest.raises(ValueError):
            _hex_to_rgb(cast(str, None))

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            _hex_to_rgb(cast(str, 123))


# ---------------------------------------------------------------------------
# _wrap_text tests
# ---------------------------------------------------------------------------


class TestWrapText:
    @patch("src.config.get_settings")
    def test_short_text_no_wrap(self, mock_settings):
        mock_settings.return_value = MagicMock(theme=MagicMock(font_path="", font_bold_path=""))
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (800, 600))
        draw = ImageDraw.Draw(img)
        font = _get_font(20)
        result = _wrap_text(draw, "Hello", font, 780)
        assert "\n" not in result

    @patch("src.config.get_settings")
    def test_long_text_wraps(self, mock_settings):
        mock_settings.return_value = MagicMock(theme=MagicMock(font_path="", font_bold_path=""))
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (200, 100))
        draw = ImageDraw.Draw(img)
        font = _get_font(20)
        long_text = "This is a very long title text that should definitely wrap because it is too wide for a 200px image"
        result = _wrap_text(draw, long_text, font, 100)
        assert "\n" in result


# ---------------------------------------------------------------------------
# TitleCardGenerator tests
# ---------------------------------------------------------------------------


class TestTitleCardGeneratorInit:
    @patch("src.config.get_settings")
    def test_default_init(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        assert gen._primary_color is not None
        assert gen._background_color is not None

    @patch("src.config.get_settings")
    def test_init_with_theme_overrides(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator(
            theme_config={"primary_color": "#FF0000", "background_color": "#000000"}
        )
        assert gen._primary_color == "#FF0000"
        assert gen._background_color == "#000000"


class TestTitleCardGeneratorImage:
    @patch("src.config.get_settings")
    def test_generate_image_with_bullet_points(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        img = gen.generate_image(
            "Test Title",
            subtitle="Sub",
            bullet_points=["Point 1", "Point 2"],
            width=640,
            height=360,
        )
        assert img.size == (640, 360)

    @patch("src.config.get_settings")
    def test_generate_image_saves_to_file(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "title.png"
            gen.generate_image("Title", output_path=out_path, width=320, height=180)
            assert out_path.exists()


class TestTitleCardGeneratorVideo:
    @patch("src.config.get_settings")
    def test_generate_video_requires_output_path(self, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        with pytest.raises(ValueError, match="output_path is required"):
            gen.generate_video("Title", output_path=None)

    @patch("src.config.get_settings")
    @patch("src.services.compilation.title_card_generator.FFmpegOps.image_to_video")
    def test_generate_video_success(self, mock_ffmpeg, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "title.mp4"
            mock_ffmpeg.return_value = out_path

            result = gen.generate_video("Title", output_path=out_path, width=320, height=180)
            assert result == out_path
            mock_ffmpeg.assert_called_once()

    @patch("src.config.get_settings")
    @patch("src.services.compilation.title_card_generator.FFmpegOps.image_to_video")
    @patch("src.services.compilation.title_card_generator.FFmpegOps.run")
    def test_generate_video_with_fade_in_animation(self, mock_run, mock_ffmpeg, mock_settings):
        mock_settings.return_value = MagicMock(
            theme=MagicMock(
                primary_color="#3B82F6",
                background_color="#1E293B",
                text_color="#F1F5F9",
                subtitle_color="#94A3B8",
                gradient_end="#0F172A",
            )
        )
        gen = TitleCardGenerator()
        gen._animation = None

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "title.mp4"
            # Create the file so exists() returns True
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"fake video data")
            mock_ffmpeg.return_value = out_path

            result = gen.generate_video(
                "Title", output_path=out_path, width=320, height=180, animation="fade_in"
            )
            assert result == out_path
