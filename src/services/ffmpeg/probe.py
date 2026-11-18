"""ffprobe wrappers + probe result cache."""

from __future__ import annotations

import contextlib
import json as _json
import logging
import subprocess  # nosec B404
from pathlib import Path

from src.services.ffmpeg.base import COPY_COLOR_TAG_ARGS, FFmpegOpsBase, ProbeResult

logger = logging.getLogger(__name__)


class ProbeMixin(FFmpegOpsBase):
    """ffprobe operations.

    Inherits from :class:`FFmpegOpsBase` so the shared
    ``_probe_cache`` / ``_probe_lock`` / ``FFPROBE_TIMEOUT`` class state
    is visible to type checkers without ``# type: ignore`` noise.  At
    runtime the diamond is collapsed by ``FFmpegOps``' MRO.
    """

    @classmethod
    def _probe(cls, args: list[str]) -> str:
        try:
            result = subprocess.run(  # nosec B603  # noqa: S603 -- ffprobe with controlled args
                ["ffprobe", "-v", "error", *args],  # noqa: S607  # PATH lookup ok
                capture_output=True,
                text=True,
                timeout=cls.FFPROBE_TIMEOUT,
            )
        except FileNotFoundError:
            raise RuntimeError("ffprobe not found on PATH") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ffprobe timed out after {cls.FFPROBE_TIMEOUT}s") from None
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[:200]}")
        return result.stdout.strip()

    @classmethod
    def probe_all(cls, path: Path) -> ProbeResult:
        """Probe a media file for duration, dimensions, and audio presence (cached)."""
        resolved = path.resolve()
        try:
            mtime = resolved.stat().st_mtime
        except OSError:
            mtime = 0.0
        key = f"{resolved}@{mtime}"
        with cls._probe_lock:
            if key in cls._probe_cache:
                return cls._probe_cache[key]

        try:
            output = cls._probe(["-show_format", "-show_streams", "-of", "json", str(path)])
            data = _json.loads(output)
        except (RuntimeError, _json.JSONDecodeError, ValueError) as e:
            logger.warning("Probe failed for %s: %s", path, e)
            # NOT cached: a transient probe failure (timeout under load, a
            # muxer mid-flush) must not permanently poison the cache for this
            # path -- a later, successful probe of the same file should still
            # be reachable. ``probed=False`` is the signal a caller MUST check
            # before treating ``duration=0.0`` as a real measurement.
            return ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=False)

        duration = 0.0
        dur_str = data.get("format", {}).get("duration", "")
        if dur_str:
            with contextlib.suppress(ValueError):
                duration = float(dur_str)

        width, height = 0, 0
        has_audio = False
        has_video = False
        video_frames = 0
        for stream in data.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video" and not has_video:
                has_video = True
                width = int(stream.get("width", 0))
                height = int(stream.get("height", 0))
                nb_frames = str(stream.get("nb_frames", "")).strip()
                if nb_frames.isdigit():
                    video_frames = int(nb_frames)
            elif codec_type == "audio":
                has_audio = True

        result = ProbeResult(
            duration=duration,
            width=width,
            height=height,
            has_audio=has_audio,
            has_video=has_video,
            video_frames=video_frames,
        )
        with cls._probe_lock:
            cls._probe_cache[key] = result
        return result

    @classmethod
    def probe_pix_fmt(cls, path: Path) -> str:
        """Return the first video stream's pixel format, or ``""`` if unreadable.

        Deliberately NOT served from ``_probe_cache``: that cache stores a
        :class:`ProbeResult`, which has no pix_fmt field, and the one caller
        that needs this (the VP9-alpha capability probe) runs once at startup
        against a file it just wrote. Adding a field to the cached dataclass
        would make every existing probe parse and store a value nothing reads.

        Unlike :meth:`probe_all` this does NOT swallow the failure into a
        zero-valued result: an unreadable pixel format is the whole answer the
        caller is asking for, so it is returned as an explicit empty string
        that the caller treats as "no alpha", never as "probably fine".
        """
        try:
            return cls._probe(
                [
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=pix_fmt",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ]
            )
        except RuntimeError as e:
            logger.warning("pix_fmt probe failed for %s: %s", path, e)
            return ""

    @classmethod
    def probe_stream_tag(cls, path: Path, tag: str) -> str:
        """Return a container-level tag on the first video stream, or ``""``.

        Some properties live on the CONTAINER, not in the codec bitstream —
        WebM's ``alpha_mode`` being the case this exists for (VP9 stores alpha
        as a per-block side payload, so the stream's own pix_fmt never reveals
        it). Uncached and non-raising for the same reasons as
        :meth:`probe_pix_fmt`.
        """
        try:
            return cls._probe(
                [
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    f"stream_tags={tag}",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ]
            )
        except RuntimeError as e:
            logger.warning("stream tag '%s' probe failed for %s: %s", tag, path, e)
            return ""

    @staticmethod
    def _require_probed(path: Path, result: ProbeResult) -> ProbeResult:
        """Refuse to hand back a zero-valued result that was never measured.

        ``probe_all`` swallows an ffprobe failure (crash, non-zero exit, or the
        10s ``FFPROBE_TIMEOUT`` elapsing under load) into
        ``ProbeResult(duration=0.0, width=0, height=0, has_audio=False,
        probed=False)``. Its own docstring says ``probed`` "is the signal a
        caller MUST check", and exactly one caller in the service did
        (``recorder_service._verify_artifact``). Every accessor below dropped
        the flag, so an UNMEASURED probe reached the compiler as a real
        measurement and degraded the output SILENTLY -- clean-green, no ERROR
        line, in at least six distinct ways from one timeout:

        * ``_assembly._compute_chapter_timestamps`` -- a 0.0 for one segment
          shifts every later chapter start early, and the crossfade offset it
          subtracts first can drive ``start_seconds`` NEGATIVE.
        * ``_finalize._finalize_sync`` -- persists and webhooks
          ``total_duration=0.0`` for a real video.
        * ``_segment_builders._mix_per_segment_narration`` -- ``audio_dur <= 0``
          returns early, so the segment ships with NO NARRATION.
        * ``_effects._speed_adjust_to_narration`` -- skips the sync entirely.
        * ``video_editor.remove_dead_time`` / ``auto_trim`` -- returns the clip
          untouched, so dead time survives.
        * ``compose`` / ``filters`` / ``transcode`` / ``video_filters`` --
          ``has_audio=False`` builds a video-only filtergraph, dropping AUDIO.

        None of those can tell "0.0 because the file is empty" from "0.0
        because we never read the file". So the gap is made loud here, once, at
        the single point that knows the difference. A raised ``RuntimeError``
        matches ``_probe``'s own contract for the same condition and fails the
        compile attributably instead of delivering a corrupted master.
        """
        if not result.probed:
            raise RuntimeError(
                f"ffprobe could not read {path}: this is an UNMEASURED probe, not a "
                f"measurement -- duration/dimensions/audio are unknown, not zero"
            )
        return result

    @classmethod
    def probe_duration(cls, path: Path) -> float:
        """Return the duration in seconds of a media file.

        Raises:
            RuntimeError: if ffprobe could not read the file. See
                :meth:`_require_probed` -- a silent 0.0 here corrupts chapter
                timestamps and the reported total duration.
        """
        return cls._require_probed(path, cls.probe_all(path)).duration

    @classmethod
    def probe_dimensions(cls, path: Path) -> tuple[int, int]:
        """Return (width, height) of the first video stream.

        Raises:
            RuntimeError: if ffprobe could not read the file (see
                :meth:`_require_probed`).
        """
        r = cls._require_probed(path, cls.probe_all(path))
        return r.width, r.height

    @classmethod
    def probe_has_audio(cls, path: Path) -> bool:
        """Return True if the file contains an audio stream.

        Raises:
            RuntimeError: if ffprobe could not read the file (see
                :meth:`_require_probed`). A silent ``False`` here builds a
                video-only filtergraph and drops the audio track.
        """
        return cls._require_probed(path, cls.probe_all(path)).has_audio

    @classmethod
    def probe_video_codec(cls, path: Path) -> str:
        """Return the first video stream's codec name, or ``""`` if unreadable.

        Uncached and non-raising for the same reasons as
        :meth:`probe_pix_fmt`. The one thing callers do with this is decide
        whether an H.264-only bitstream filter may be spliced in, and the safe
        answer to "I could not tell" is "do not splice", never "probably fine".
        """
        try:
            return cls._probe(
                [
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ]
            ).strip()
        except RuntimeError as e:
            logger.warning("codec_name probe failed for %s: %s", path, e)
            return ""

    @classmethod
    def copy_color_tag_args(cls, path: Path) -> list[str]:
        """BT.709 tag args for a ``-c:v copy`` of ``path`` -- or ``[]``.

        ``COPY_COLOR_TAG_ARGS`` is the ``h264_metadata`` bitstream filter, and
        ffmpeg HARD-ERRORS when a bsf is applied to a stream of the wrong
        codec. This service really does emit VP9/WebM
        (``transcode.frames_to_webm``), so splicing it unconditionally into
        every stream-copy would turn a working webm path into a stage failure.
        The tag is therefore gated on the SOURCE stream's actual codec: H.264
        gets the VUI rewrite, anything else (including an unreadable probe)
        gets nothing and stays as untagged as it is today. VP9 carries its
        colour description in the container/bitstream already and has no
        equivalent metadata bsf in ffmpeg.
        """
        if cls.probe_video_codec(path) == "h264":
            return list(COPY_COLOR_TAG_ARGS)
        return []
