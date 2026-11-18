"""Compilation preset definitions and resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompilationPreset:
    """A named set of compilation defaults."""

    name: str
    description: str
    target_width: int
    target_height: int
    crf: int
    fps: int
    transition_type: str
    crossfade_duration: float


PRESETS: dict[str, CompilationPreset] = {
    "default": CompilationPreset(
        name="default",
        description="Default 1080p compilation",
        target_width=1920,
        target_height=1080,
        crf=20,
        fps=30,
        transition_type="smart",
        crossfade_duration=0.5,
    ),
    "quick-preview": CompilationPreset(
        name="quick-preview",
        description="Fast preview at 480p",
        target_width=854,
        target_height=480,
        crf=28,
        fps=24,
        transition_type="cut",
        crossfade_duration=0.0,
    ),
    "standard": CompilationPreset(
        name="standard",
        description="Standard 1080p compilation",
        target_width=1920,
        target_height=1080,
        crf=20,
        fps=30,
        transition_type="smart",
        crossfade_duration=0.5,
    ),
    "high-quality": CompilationPreset(
        name="high-quality",
        description="High quality 1080p with fade transitions",
        target_width=1920,
        target_height=1080,
        crf=16,
        fps=30,
        transition_type="fade",
        crossfade_duration=0.7,
    ),
    "4k": CompilationPreset(
        name="4k",
        description="4K UHD compilation",
        target_width=3840,
        target_height=2160,
        crf=18,
        fps=30,
        transition_type="smart",
        crossfade_duration=0.5,
    ),
}


def get_preset(name: str) -> CompilationPreset | None:
    """Get a preset by name, returns None if not found."""
    return PRESETS.get(name)


def list_presets() -> list[CompilationPreset]:
    """Return all available presets."""
    return list(PRESETS.values())


def resolve_preset_config(
    preset_name: str | None,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    crf: int | None = None,
    fps: int | None = None,
    transition_type: str | None = None,
    crossfade_duration: float | None = None,
) -> dict:
    """Resolve compilation config from preset + explicit overrides.

    Explicit per-field values always take precedence over preset defaults.
    Returns dict with keys: target_width, target_height, crf, fps,
    transition_type, crossfade_duration.
    """
    base: dict = {}
    if preset_name:
        preset = get_preset(preset_name)
        if preset is None:
            raise ValueError(f"Unknown preset: {preset_name!r}. Available: {list(PRESETS.keys())}")
        base = {
            "target_width": preset.target_width,
            "target_height": preset.target_height,
            "crf": preset.crf,
            "fps": preset.fps,
            "transition_type": preset.transition_type,
            "crossfade_duration": preset.crossfade_duration,
        }

    # Explicit overrides take precedence
    if target_width is not None:
        base["target_width"] = target_width
    if target_height is not None:
        base["target_height"] = target_height
    if crf is not None:
        base["crf"] = crf
    if fps is not None:
        base["fps"] = fps
    if transition_type is not None:
        base["transition_type"] = transition_type
    if crossfade_duration is not None:
        base["crossfade_duration"] = crossfade_duration

    return base
