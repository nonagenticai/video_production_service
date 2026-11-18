"""Startup capability probe: does THIS ffmpeg build round-trip VP9 alpha?

Transparent motion graphics (lower-thirds, logo bugs composited over drone
footage) are only worth encoding if the alpha channel actually survives. The
failure mode that makes this probe necessary is entirely silent: an ffmpeg
build can accept ``-pix_fmt yuva420p``, exit 0, print no warning, and write a
WebM with no alpha in it at all. Downstream, the compositor lays what it
believes is an overlay over the footage and gets an opaque box covering the
shot. Nothing errors — not the encode, not the mux, not the compile — so the
only way to know is to encode, read the bytes back, and look.

HOW VP9 ALPHA IS ACTUALLY STORED (verified against a real encode, and the
reason this module does NOT do the obvious thing):

* VP9 does **not** carry alpha as a fourth plane in the bitstream. libvpx
  encodes the alpha as a SECOND VP9 stream muxed into WebM as per-block
  ``BlockAdditional`` side data, flagged by the container tag
  ``alpha_mode=1``.
* Consequently ``ffprobe -show_entries stream=pix_fmt`` reports **yuv420p** on
  a perfectly good alpha file. Asserting ``pix_fmt == yuva420p`` on the encoded
  file — the intuitive check — fails on EVERY build and would disable the
  feature everywhere. It is a false alarm, not a safety net.
* ffmpeg's built-in ``vp9`` decoder IGNORES that side data and hands back
  opaque frames. Only the external ``libvpx-vp9`` decoder honours it. Anything
  that later composites one of these clips must therefore select that decoder
  explicitly, or it will composite an opaque box — the same failure, one stage
  further down.

So the probe checks the two signals that genuinely discriminate, measured
against both a transparent and an opaque control encode:

1. the ``alpha_mode`` container tag is present (the alpha payload was muxed);
2. decoding one frame back through ``-c:v libvpx-vp9`` yields a pixel whose
   alpha is NOT fully opaque (the payload is real and readable).

Signal 2 is the load-bearing one: it inspects the actual decoded bytes, so a
build that writes the tag but drops the payload still fails.

The result is cached process-wide: it is a property of the binary on PATH, and
re-encoding per request would add two subprocesses to every job.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from src.services.ffmpeg.ops import FFmpegOps

logger = logging.getLogger(__name__)

# Deliberately tiny and even-dimensioned: this is a capability question, not a
# quality one, and it runs on the startup path of every replica.
_PROBE_SIZE = (16, 16)
_PROBE_FRAMES = 2
_PROBE_FPS = 2
# Half-transparent red. A real intermediate alpha value, not 0 or 255, so a
# build that flattens onto a matte lands visibly outside the tolerance below.
_PROBE_RGBA = (255, 0, 0, 128)
# The round-trip is lossy (VP9 is not a lossless codec at crf 32) so the
# readback is compared against "clearly not opaque" rather than against 128
# exactly; a real encode measured 129.
_MAX_ROUNDTRIP_ALPHA = 250

_cached_support: bool | None = None


class AlphaCapabilityError(RuntimeError):
    """Raised when a transparent render is requested on a build that cannot do it.

    Subclasses ``RuntimeError`` so the animation recorder's existing
    ``except (OSError, RuntimeError, ...)`` turns it into a failed job carrying
    a readable reason, rather than a 500 or — far worse — an opaque clip
    reported as a success.
    """


def _write_probe_frames(frames_dir: Path) -> None:
    """Write the semi-transparent PNG sequence the probe encodes."""
    from PIL import Image

    for i in range(_PROBE_FRAMES):
        Image.new("RGBA", _PROBE_SIZE, _PROBE_RGBA).save(frames_dir / f"frame_{i:06d}.png")


def _roundtrip_alpha(webm_path: Path, png_path: Path) -> int:
    """Decode one frame with libvpx-vp9 and return the centre pixel's alpha.

    ``-c:v libvpx-vp9`` is an INPUT option and must precede ``-i``: ffmpeg's
    native vp9 decoder silently discards WebM alpha side data, so decoding with
    the default decoder would report every build as broken.
    """
    from PIL import Image

    FFmpegOps.run(
        [
            "-c:v",
            "libvpx-vp9",
            "-i",
            str(webm_path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgba",
            str(png_path),
        ]
    )
    with Image.open(png_path) as img:
        rgba = img.convert("RGBA")
        centre = (_PROBE_SIZE[0] // 2, _PROBE_SIZE[1] // 2)
        pixel = rgba.getpixel(centre)
    if not isinstance(pixel, tuple) or len(pixel) < 4:
        # A decode that produced no fourth band is a decode with no alpha —
        # report the fully-opaque value so the caller's verdict is "unsupported"
        # rather than a crash on an unexpected shape.
        return 255
    return int(pixel[3])


def _measure_alpha_capability() -> tuple[str, int]:
    """Encode the probe sequence and return ``(alpha_mode_tag, roundtrip_alpha)``."""
    with tempfile.TemporaryDirectory(prefix="vp9_alpha_probe_") as tmp:
        frames_dir = Path(tmp)
        _write_probe_frames(frames_dir)
        # Encoded through the SAME helper a real overlay render uses — a probe
        # that exercises a different command line proves nothing about the
        # command line we ship.
        out = FFmpegOps.frames_to_webm(
            frames_dir,
            frames_dir / "probe.webm",
            fps=_PROBE_FPS,
            transparent=True,
        )
        alpha_mode = FFmpegOps.probe_stream_tag(out, "alpha_mode")
        return alpha_mode, _roundtrip_alpha(out, frames_dir / "probe_readback.png")


def probe_vp9_alpha_support(*, force: bool = False) -> bool:
    """Return True iff this ffmpeg build round-trips alpha through a VP9 encode.

    Cached after the first call. ``force`` re-runs it (used by tests; there is
    no runtime reason to re-probe a binary that cannot change under a running
    process).
    """
    # Module-level memo: the answer is a property of the binary on PATH, which
    # cannot change under a running process.
    global _cached_support
    if _cached_support is not None and not force:
        return _cached_support

    try:
        alpha_mode, roundtrip_alpha = _measure_alpha_capability()
    except (OSError, RuntimeError, ValueError) as exc:
        # An encode or decode that cannot even run (no ffmpeg on PATH, a libvpx
        # that rejects yuva420p outright, a timeout) answers the question just
        # as definitively as one that runs and drops the plane — and it is the
        # FRIENDLY branch of the two, because it is loud by itself. Cache the
        # negative so a transparent-heavy tenant does not respawn a doomed
        # subprocess on every request.
        logger.error(
            "VP9 alpha capability probe could not complete: %s. Transparent "
            "overlay renders are unavailable on this replica.",
            exc,
        )
        _cached_support = False
        return False

    supported = bool(alpha_mode) and roundtrip_alpha <= _MAX_ROUNDTRIP_ALPHA
    if not supported:
        # Loud and specific: state both measurements, what they mean, and what
        # to do. An empty alpha_mode renders as '' — the honest rendering of
        # "the container carries no alpha payload at all".
        logger.error(
            "VP9 ALPHA UNSUPPORTED by this ffmpeg build: encoded a "
            "half-transparent PNG sequence with -pix_fmt yuva420p; the muxed "
            "file reports alpha_mode='%s' and a frame decoded back through "
            "libvpx-vp9 has alpha=%d (fully opaque is 255). The encode "
            "SUCCEEDED and silently discarded the alpha channel — a clip "
            "produced this way composites as an opaque box over the footage, "
            "with no error anywhere in the pipeline. Transparent overlay "
            "renders are therefore REFUSED on this replica; rebuild the image "
            "with a libvpx/ffmpeg that supports VP9 alpha to enable them.",
            alpha_mode,
            roundtrip_alpha,
        )
    _cached_support = supported
    return supported


def require_vp9_alpha_support() -> None:
    """Raise :class:`AlphaCapabilityError` unless transparent encoding is usable.

    Called on the request path before a transparent job spends a browser and N
    frames. Refusing here is the whole point of the capability probe: the
    alternative is not "a slightly worse clip", it is a clip that looks correct
    in isolation and destroys the shot it is composited into.
    """
    if not probe_vp9_alpha_support():
        msg = (
            "transparent rendering is unavailable on this replica: the ffmpeg "
            "build drops the alpha channel during VP9 encoding (startup "
            "capability probe failed)"
        )
        raise AlphaCapabilityError(msg)


def reset_cache_for_tests() -> None:
    """Clear the memoised capability so a test can re-probe with a fake ffmpeg."""
    global _cached_support
    _cached_support = None
