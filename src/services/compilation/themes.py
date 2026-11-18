"""Theme presets and resolution for video compilation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGB_RE = re.compile(
    r"^rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}" r"\s*(?:,\s*(?:0|1|0?\.\d+)\s*)?\)$"
)


def validate_color(value: str, field_name: str = "color") -> str:
    """Validate a color string is hex or rgb/rgba format."""
    if _HEX_RE.match(value) or _RGB_RE.match(value):
        return value
    raise ValueError(
        f"Invalid color for {field_name}: {value!r}. "
        "Colors must be hex (#RGB, #RRGGBB, #RRGGBBAA) "
        "or rgb(r,g,b) / rgba(r,g,b,a)."
    )


@dataclass(frozen=True)
class ThemePreset:
    """A complete theme preset with all color definitions."""

    name: str
    primary_color: str
    background_color: str
    text_color: str
    accent_color: str
    subtitle_color: str
    gradient_start: str
    gradient_end: str

    def __post_init__(self) -> None:
        for field_name in [
            "primary_color",
            "background_color",
            "text_color",
            "accent_color",
            "subtitle_color",
            "gradient_start",
            "gradient_end",
        ]:
            validate_color(getattr(self, field_name), field_name)


THEME_PRESETS: dict[str, ThemePreset] = {
    "default": ThemePreset(
        name="default",
        primary_color="#6366f1",
        background_color="#0f172a",
        text_color="#f8fafc",
        accent_color="#818cf8",
        subtitle_color="#94a3b8",
        gradient_start="#0f172a",
        gradient_end="#1e293b",
    ),
    "dark": ThemePreset(
        name="dark",
        primary_color="#10b981",
        background_color="#111827",
        text_color="#f9fafb",
        accent_color="#34d399",
        subtitle_color="#9ca3af",
        gradient_start="#111827",
        gradient_end="#1f2937",
    ),
    "light": ThemePreset(
        name="light",
        primary_color="#3b82f6",
        background_color="#ffffff",
        text_color="#1e293b",
        accent_color="#60a5fa",
        subtitle_color="#64748b",
        gradient_start="#f8fafc",
        gradient_end="#e2e8f0",
    ),
    "corporate": ThemePreset(
        name="corporate",
        primary_color="#1e40af",
        background_color="#1e293b",
        text_color="#f1f5f9",
        accent_color="#3b82f6",
        subtitle_color="#94a3b8",
        gradient_start="#1e293b",
        gradient_end="#334155",
    ),
    "creative": ThemePreset(
        name="creative",
        primary_color="#8b5cf6",
        background_color="#18181b",
        text_color="#fafafa",
        accent_color="#a78bfa",
        subtitle_color="#a1a1aa",
        gradient_start="#18181b",
        gradient_end="#27272a",
    ),
    "minimal": ThemePreset(
        name="minimal",
        primary_color="#171717",
        background_color="#fafafa",
        text_color="#171717",
        accent_color="#525252",
        subtitle_color="#737373",
        gradient_start="#fafafa",
        gradient_end="#f5f5f5",
    ),
}


def get_theme(name: str) -> ThemePreset:
    """Get a theme preset by name, falling back to default."""
    return THEME_PRESETS.get(name, THEME_PRESETS["default"])


def list_themes() -> list[ThemePreset]:
    """Return all available theme presets."""
    return list(THEME_PRESETS.values())


def resolve_theme_config(
    theme_name: str | None,
    overrides: dict | None,
) -> dict:
    """Resolve a complete theme config dict from preset + overrides.

    Base colors come from the named preset (if given), then per-field
    overrides are layered on top with validation.
    """
    if theme_name:
        preset = get_theme(theme_name)
        base: dict = {
            "primary_color": preset.primary_color,
            "background_color": preset.background_color,
            "text_color": preset.text_color,
            "accent_color": preset.accent_color,
            "subtitle_color": preset.subtitle_color,
            "gradient_start": preset.gradient_start,
            "gradient_end": preset.gradient_end,
        }
    else:
        base = {}

    if overrides:
        for key, val in overrides.items():
            # Skip non-color keys (e.g. theme_name stored in the dict)
            if key == "theme_name":
                continue
            if val is not None:
                validate_color(val, key)
                base[key] = val

    return base
