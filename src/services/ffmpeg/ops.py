"""Final FFmpegOps assembly: all five operation mixins + shared base."""

from __future__ import annotations

from src.services.ffmpeg.base import FFmpegOpsBase
from src.services.ffmpeg.compose import ComposeMixin
from src.services.ffmpeg.filters import FiltersMixin
from src.services.ffmpeg.probe import ProbeMixin
from src.services.ffmpeg.subtitles import SubtitlesMixin
from src.services.ffmpeg.transcode import TranscodeMixin

__all__ = ["FFmpegOps"]


class FFmpegOps(
    TranscodeMixin,
    ComposeMixin,
    FiltersMixin,
    SubtitlesMixin,
    ProbeMixin,
    FFmpegOpsBase,
):
    """Thin wrapper around ffmpeg CLI commands.

    Composed from per-bucket mixins; all shared class state
    (``_default_timeout``, ``_default_preset``, ``FFPROBE_TIMEOUT``,
    ``_probe_cache``, ``_probe_lock``) lives on
    :class:`FFmpegOpsBase` and is reachable via ``cls.<attr>`` from
    every mixin thanks to MRO.
    """
