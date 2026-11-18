"""The canonical still-image motion vocabulary.

One list, three consumers:

* ``src/services/ffmpeg/transcode.py`` — the ffmpeg render path
  (``image_to_motion_video``).
* ``src/services/recording/templates/ken_burns.html`` — the browser render
  path (documented in ``src/services/recording/templates/__init__.py``).
* ``src/api/schemas/compilation.py`` — the ingress that used to type the field
  as a bare ``str | None``, so an unknown preset was forwarded verbatim and
  silently degraded to a completely static clip.

Keep the three in sync. A value that is not in here must FAIL LOUD.
"""

from __future__ import annotations

from typing import Final

MOTION_PRESETS: Final[tuple[str, ...]] = (
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
)

DEFAULT_MOTION_PRESET: Final[str] = "zoom_in"


def normalize_motion_preset(value: str | None) -> str | None:
    """Trim + lowercase a preset, returning ``None`` for empty input.

    Raises ``ValueError`` for a non-empty value outside :data:`MOTION_PRESETS`.
    """
    if value is None:
        return None
    preset = str(value).strip().lower()
    if not preset:
        return None
    if preset not in MOTION_PRESETS:
        raise ValueError(
            f"unknown motion_preset {value!r}; expected one of {', '.join(MOTION_PRESETS)}"
        )
    return preset
