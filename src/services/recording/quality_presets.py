"""Quality preset definitions for recording configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityPreset:
    """Encapsulates viewport dimensions and encoding parameters for a quality level."""

    viewport_width: int
    viewport_height: int
    crf: int  # Constant Rate Factor for H.264 (lower = higher quality)
    video_bitrate: str  # e.g. "2M", "5M", "8M"
    label: str


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "preview": QualityPreset(
        viewport_width=1280,
        viewport_height=720,
        crf=32,
        video_bitrate="1M",
        label="Preview (720p, low bitrate)",
    ),
    "standard": QualityPreset(
        viewport_width=1920,
        viewport_height=1080,
        crf=23,
        video_bitrate="5M",
        label="Standard (1080p)",
    ),
    "high_quality": QualityPreset(
        viewport_width=1920,
        viewport_height=1080,
        crf=18,
        video_bitrate="8M",
        label="High Quality (1080p, high CRF)",
    ),
    "2k": QualityPreset(
        # Matches the orchestrator's default resolution tier (2k). 2560-wide is a
        # common 1440p desktop width that real sites lay out cleanly, so browser
        # recordings stay sharp at this tier without a dramatic reflow.
        viewport_width=2560,
        viewport_height=1440,
        crf=18,
        video_bitrate="10M",
        label="2K (1440p)",
    ),
    "4k": QualityPreset(
        viewport_width=3840,
        viewport_height=2160,
        crf=18,
        video_bitrate="15M",
        label="4K (2160p)",
    ),
}


def get_quality_preset(name: str) -> QualityPreset:
    """Return a QualityPreset by name, raising ValueError for unknown names."""
    preset = QUALITY_PRESETS.get(name.lower().strip())
    if preset is None:
        valid = ", ".join(sorted(QUALITY_PRESETS.keys()))
        raise ValueError(f"Unknown quality preset '{name}'. Valid presets: {valid}")
    return preset
