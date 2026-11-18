"""Color correction and watermark overlay filters for video processing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.services.ffmpeg import COLOR_TAG_ARGS, SCALE_FLAGS, FFmpegOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Position presets for overlay placement
# ---------------------------------------------------------------------------

POSITION_MAP: dict[str, str] = {
    "top-left": "10:10",
    "top-right": "W-w-10:10",
    "center": "(W-w)/2:(H-h)/2",
    "bottom-left": "10:H-h-10",
    "bottom-right": "W-w-10:H-h-10",
}


# ---------------------------------------------------------------------------
# Color correction data
# ---------------------------------------------------------------------------


@dataclass
class ColorCorrection:
    """Parameters for FFmpeg eq / colorbalance filters.

    Attributes:
        brightness: Additive brightness adjustment (default 0.0).
        contrast:   Multiplicative contrast (default 1.0, no change).
        saturation: Multiplicative saturation (default 1.0, no change).
        gamma:      Gamma curve exponent (default 1.0, no change).
        temperature: Warm/cool shift in the range -1.0 (cool) to 1.0 (warm).
    """

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    temperature: float = 0.0  # range -1.0 .. 1.0


# ---------------------------------------------------------------------------
# Color correction
# ---------------------------------------------------------------------------


def apply_color_correction(
    input_path: Path,
    output_path: Path,
    correction: ColorCorrection,
) -> Path:
    """Apply brightness / contrast / saturation / gamma / temperature adjustments.

    Builds an FFmpeg ``eq`` filter and, when *temperature* is non-zero, appends a
    ``colorbalance`` filter to shift the red/blue channels.

    Returns:
        *output_path* after a successful FFmpeg run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    eq_filter = (
        f"eq=brightness={correction.brightness}"
        f":contrast={correction.contrast}"
        f":saturation={correction.saturation}"
        f":gamma={correction.gamma}"
    )

    filters: list[str] = [eq_filter]

    if correction.temperature != 0.0:
        temp = correction.temperature * 0.3
        filters.append(f"colorbalance=rs={temp}:gs=0:bs={-temp}")

    vf = ",".join(filters)
    logger.info(
        "Applying color correction to %s -> %s  vf=%s",
        input_path,
        output_path,
        vf,
    )

    FFmpegOps.run(
        [
            "-i",
            str(input_path),
            "-vf",
            vf,
            *COLOR_TAG_ARGS,
            "-c:a",
            "copy",
            str(output_path),
        ]
    )

    return output_path


# ---------------------------------------------------------------------------
# Image overlay (watermark)
# ---------------------------------------------------------------------------


def overlay_image(
    input_path: Path,
    overlay_path: Path,
    output_path: Path,
    position: str = "bottom-right",
    opacity: float = 0.7,
    scale: float = 0.15,
) -> Path:
    """Overlay a semi-transparent image (e.g. watermark) on a video.

    The overlay is scaled relative to the main video width, alpha-blended to
    *opacity*, and placed according to *position* (one of ``POSITION_MAP`` keys).

    Audio is copied through when present in the source video.

    Returns:
        *output_path* after a successful FFmpeg run.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = POSITION_MAP.get(position, POSITION_MAP["bottom-right"])

    # ``-2`` (not ``-1``): the derived height must stay even. An odd watermark
    # height makes the overlay output odd-dimensioned, which yuv420p cannot
    # encode, and the failure surfaces only at ffmpeg run time.
    filter_complex = (
        f"[1:v]scale=iw*{scale}:-2:{SCALE_FLAGS},format=rgba,"
        f"colorchannelmixer=aa={opacity}[wm];"
        f"[0:v][wm]overlay={pos}[v]"
    )

    has_audio = FFmpegOps.probe_has_audio(input_path)

    args: list[str] = [
        "-i",
        str(input_path),
        "-i",
        str(overlay_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
    ]

    if has_audio:
        args.extend(["-map", "0:a", "-c:a", "copy"])

    args.extend([*COLOR_TAG_ARGS, "-max_muxing_queue_size", "4096"])
    args.append(str(output_path))

    logger.info(
        "Overlaying %s on %s (pos=%s, opacity=%.2f, scale=%.2f) -> %s",
        overlay_path,
        input_path,
        position,
        opacity,
        scale,
        output_path,
    )

    FFmpegOps.run(args)

    return output_path
