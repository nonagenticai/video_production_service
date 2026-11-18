"""Generate title card images using Pillow and convert to video clips."""

from __future__ import annotations

import contextlib
import logging
import re
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.services.ffmpeg import COLOR_TAG_ARGS, FFmpegOps

logger = logging.getLogger(__name__)

# Vendored brand display face (Fraunces, OFL-1.1). Pillow cannot load the woff2
# used by the HTML motion cards, so real .ttf static instances live in
# assets/fonts/ (see assets/fonts/FRAUNCES-PROVENANCE.md). These make the
# PIL/FFmpeg title cards render on-brand by default, with no THEME_FONT_* config.
# Two locations are probed: the repo asset dir (local dev / tests) and the
# container install path (Dockerfile: COPY assets/fonts/*.ttf -> here).
_BRAND_FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
_BRAND_FONTS_INSTALLED = Path("/usr/share/fonts/truetype/brand")


def _brand_faces(bold: bool) -> list[str]:
    fname = "Fraunces-SemiBold.ttf" if bold else "Fraunces-Regular.ttf"
    return [str(_BRAND_FONTS_DIR / fname), str(_BRAND_FONTS_INSTALLED / fname)]


# Exposed for tests / callers that want the primary (repo) brand-face paths.
_BRAND_FONT_REGULAR = _brand_faces(bold=False)[0]
_BRAND_FONT_BOLD = _brand_faces(bold=True)[0]


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from src.config import get_settings

    settings = get_settings()
    # Explicit THEME_FONT_* config wins over everything.
    configured = settings.theme.font_bold_path if bold else settings.theme.font_path
    if configured:
        try:
            return ImageFont.truetype(configured, size)
        except OSError:
            pass
    # Vendored brand face first, then platform fallbacks.
    if bold:
        font_names = [
            *_brand_faces(bold=True),
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    else:
        font_names = [
            *_brand_faces(bold=False),
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


_HEX_COLOR_RE = re.compile(r"^#?(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    if not isinstance(hex_color, str) or not hex_color.strip():
        raise ValueError(f"Invalid color: {hex_color!r}")
    hex_color = hex_color.strip()
    if not _HEX_COLOR_RE.match(hex_color):
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
) -> str:
    bbox = draw.textbbox((0, 0), text, font=font)
    if (bbox[2] - bbox[0]) <= max_width:
        return text
    avg_char_w = max(1, (bbox[2] - bbox[0]) / max(1, len(text)))
    est_chars = max(1, int(max_width / avg_char_w))
    for chars in range(est_chars, 4, -1):
        lines = textwrap.wrap(text, width=chars)
        if all(
            (draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0])
            <= max_width
            for line in lines
        ):
            return "\n".join(lines)
    return "\n".join(textwrap.wrap(text, width=max(4, est_chars)))


class TitleCardGenerator:
    """Generate title card images and convert to video clips."""

    def __init__(
        self,
        theme_name: str | None = None,
        theme_config: dict | None = None,
    ) -> None:
        from src.config import get_settings
        from src.services.compilation.themes import resolve_theme_config

        settings = get_settings()

        # Merge preset (if named) with per-request overrides
        tc = resolve_theme_config(theme_name, theme_config)

        # Fall back to global settings for any field not provided
        self._primary_color = tc.get("primary_color", settings.theme.primary_color)
        self._background_color = tc.get("background_color", settings.theme.background_color)
        self._text_color = tc.get("text_color", settings.theme.text_color)
        self._subtitle_color = tc.get("subtitle_color", settings.theme.subtitle_color)
        self._gradient_end = tc.get("gradient_end", settings.theme.gradient_end)

        # WP-3: Enhanced theme config
        self._background_image = tc.get("background_image")
        self._logo_path = tc.get("logo_path")
        self._logo_position = tc.get("logo_position", "top-right")
        self._logo_scale = tc.get("logo_scale", 0.1)
        self._animation = tc.get("animation")

    def _build_background(
        self,
        background_image: str | None,
        bg_color: tuple[int, int, int],
        grad_end: tuple[int, int, int],
        width: int,
        height: int,
    ) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        bg_img_path = background_image or self._background_image
        if bg_img_path:
            try:
                img = Image.open(bg_img_path).resize((width, height), Image.Resampling.LANCZOS)
                img = img.convert("RGB")
                return img, ImageDraw.Draw(img)
            except OSError, ValueError:
                logger.warning("Failed to load background image %s, using gradient", bg_img_path)
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        self._draw_gradient(img, draw, bg_color, grad_end, width, height)
        return img, ImageDraw.Draw(img)

    def generate_image(
        self,
        title: str,
        subtitle: str = "",
        chapter: int | None = None,
        output_path: Path | None = None,
        width: int = 1920,
        height: int = 1080,
        background_image: str | None = None,
        logo_path: str | None = None,
        logo_position: str | None = None,
        logo_scale: float | None = None,
        bullet_points: list[str] | None = None,
    ) -> Image.Image:
        """Render a title card image with gradient background, text, and optional logo.

        Args:
            title: Main title text.
            subtitle: Subtitle text below the title.
            chapter: Optional chapter number displayed above the title.
            output_path: If given, save the image to this path.
            width: Image width in pixels.
            height: Image height in pixels.
            background_image: Optional background image path.
            logo_path: Optional logo image path to overlay.
            logo_position: Logo placement (e.g. 'top-right').
            logo_scale: Logo size relative to image width.
            bullet_points: Optional bullet points rendered below the subtitle.

        Returns:
            The generated PIL Image.
        """
        bg_color = _hex_to_rgb(self._background_color)
        text_color = _hex_to_rgb(self._text_color)
        primary_color = _hex_to_rgb(self._primary_color)
        subtitle_color = _hex_to_rgb(self._subtitle_color)
        grad_end = _hex_to_rgb(self._gradient_end)

        img, draw = self._build_background(background_image, bg_color, grad_end, width, height)

        effective_logo = logo_path or self._logo_path
        if effective_logo:
            try:
                self._place_logo(
                    img,
                    effective_logo,
                    position=logo_position or self._logo_position,
                    scale=logo_scale or self._logo_scale,
                    width=width,
                    height=height,
                )
                draw = ImageDraw.Draw(img)
            except OSError, ValueError:
                logger.warning("Failed to place logo %s", effective_logo)

        if chapter is not None:
            ch_font = _get_font(18)
            ch_text = f"CHAPTER {chapter}"
            ch_bbox = draw.textbbox((0, 0), ch_text, font=ch_font)
            ch_w = ch_bbox[2] - ch_bbox[0]
            draw.text(
                ((width - ch_w) // 2, height // 2 - 80), ch_text, fill=primary_color, font=ch_font
            )

        title_font = _get_font(52, bold=True)
        max_text_width = int(width * 0.85)
        try:
            wrapped_title = _wrap_text(draw, title, title_font, max_text_width)
        except TypeError, AttributeError:
            wrapped_title = title
        title_lines = wrapped_title.split("\n")
        line_height = 62
        total_title_height = line_height * len(title_lines)
        title_y_start = height // 2 - 30 - (total_title_height - line_height) // 2
        for line_idx, line in enumerate(title_lines):
            line_bbox = draw.textbbox((0, 0), line, font=title_font)
            line_w = line_bbox[2] - line_bbox[0]
            draw.text(
                ((width - line_w) // 2, title_y_start + line_idx * line_height),
                line,
                fill=text_color,
                font=title_font,
            )

        if subtitle:
            sub_font = _get_font(22)
            sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            sub_w = sub_bbox[2] - sub_bbox[0]
            subtitle_y = title_y_start + total_title_height + 20
            draw.text(
                ((width - sub_w) // 2, subtitle_y), subtitle, fill=subtitle_color, font=sub_font
            )

        # WP-3: Bullet points
        if bullet_points:
            bullet_font = _get_font(20)
            bullet_y = title_y_start + total_title_height + (60 if subtitle else 30)
            for bp in bullet_points:
                bp_text = f"  \u2022  {bp}"
                bp_bbox = draw.textbbox((0, 0), bp_text, font=bullet_font)
                bp_w = bp_bbox[2] - bp_bbox[0]
                draw.text(
                    ((width - bp_w) // 2, bullet_y), bp_text, fill=subtitle_color, font=bullet_font
                )
                bullet_y += 32

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(output_path))

        return img

    def _draw_gradient(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        bg_color: tuple,
        grad_end: tuple,
        width: int,
        height: int,
    ) -> None:
        """Draw gradient background onto image."""
        try:
            import numpy as np

            ratios = np.linspace(0.0, 1.0, height, dtype=np.float32).reshape(-1, 1)
            bg_arr = np.array(bg_color, dtype=np.float32)
            end_arr = np.array(grad_end, dtype=np.float32)
            gradient = (bg_arr * (1.0 - ratios) + end_arr * ratios).astype(np.uint8)
            gradient_full = np.broadcast_to(gradient[:, np.newaxis, :], (height, width, 3)).copy()
            new_img = Image.fromarray(gradient_full, "RGB")
            img.paste(new_img)
        except ImportError:
            for y in range(height):
                ratio = y / height
                r = int(bg_color[0] * (1 - ratio) + grad_end[0] * ratio)
                g = int(bg_color[1] * (1 - ratio) + grad_end[1] * ratio)
                b = int(bg_color[2] * (1 - ratio) + grad_end[2] * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

    @staticmethod
    def _place_logo(
        img: Image.Image,
        logo_path: str,
        position: str = "top-right",
        scale: float = 0.1,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Place a logo on the image at the specified position (WP-3)."""
        logo = Image.open(logo_path).convert("RGBA")
        logo_w = int(width * scale)
        aspect = logo.height / max(logo.width, 1)
        logo_h = int(logo_w * aspect)
        logo = logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        margin = 20
        positions = {
            "top-left": (margin, margin),
            "top-right": (width - logo_w - margin, margin),
            "center": ((width - logo_w) // 2, (height - logo_h) // 2),
            "bottom-left": (margin, height - logo_h - margin),
            "bottom-right": (width - logo_w - margin, height - logo_h - margin),
        }
        x, y = positions.get(position, positions["top-right"])
        img.paste(logo, (x, y), logo)

    def generate_video(
        self,
        title: str,
        subtitle: str = "",
        chapter: int | None = None,
        output_path: Path | None = None,
        duration: float = 5.0,
        width: int = 1920,
        height: int = 1080,
        background_image: str | None = None,
        logo_path: str | None = None,
        logo_position: str | None = None,
        logo_scale: float | None = None,
        animation: str | None = None,
        bullet_points: list[str] | None = None,
    ) -> Path:
        """Generate a title card as a video clip with optional animation.

        Args:
            title: Main title text.
            subtitle: Subtitle text.
            chapter: Optional chapter number.
            output_path: Output MP4 path (required).
            duration: Video duration in seconds.
            width: Video width in pixels.
            height: Video height in pixels.
            background_image: Optional background image path.
            logo_path: Optional logo overlay path.
            logo_position: Logo placement position.
            logo_scale: Logo size relative to video width.
            animation: Animation effect ('fade_in' or 'slide_in').
            bullet_points: Bullet point text lines.

        Returns:
            The output_path of the generated video.

        Raises:
            ValueError: If output_path is not provided.
        """
        if output_path is None:
            raise ValueError("output_path is required")
        img_path = output_path.with_suffix(".png")
        self.generate_image(
            title,
            subtitle,
            chapter,
            img_path,
            width,
            height,
            background_image=background_image,
            logo_path=logo_path,
            logo_position=logo_position,
            logo_scale=logo_scale,
            bullet_points=bullet_points,
        )
        FFmpegOps.image_to_video(img_path, output_path, duration, width=width, height=height)

        # WP-3: Animation effects
        effective_animation = animation or self._animation
        if effective_animation and output_path.exists():
            animated_path = output_path.with_name(output_path.stem + "_anim.mp4")
            try:
                if effective_animation == "fade_in":
                    FFmpegOps.run(
                        [
                            "-i",
                            str(output_path),
                            "-vf",
                            f"fade=t=in:d={min(1.5, duration / 2)}",
                            "-c:v",
                            "libx264",
                            "-crf",
                            "18",
                            *COLOR_TAG_ARGS,
                            "-c:a",
                            "copy",
                            str(animated_path),
                        ]
                    )
                elif effective_animation == "slide_in":
                    FFmpegOps.run(
                        [
                            "-i",
                            str(output_path),
                            "-vf",
                            (f"crop=iw:ih:0:0,fade=t=in:d={min(1.0, duration / 3)}"),
                            "-c:v",
                            "libx264",
                            "-crf",
                            "18",
                            *COLOR_TAG_ARGS,
                            "-c:a",
                            "copy",
                            str(animated_path),
                        ]
                    )
                if animated_path.exists():
                    import shutil

                    shutil.move(str(animated_path), str(output_path))
            except OSError, ValueError, RuntimeError:
                logger.warning(
                    "Animation effect %s failed, using static title card", effective_animation
                )

        with contextlib.suppress(OSError):
            img_path.unlink()
        return output_path
