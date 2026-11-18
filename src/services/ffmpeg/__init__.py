"""FFmpeg operations package.

Public surface:

- :class:`FFmpegOps` -- the composed operations class.
- :class:`FFmpegError` -- raised on non-zero ffmpeg exits.
- :class:`ProbeResult` -- ffprobe result dataclass.
- ``COLOR_TAG_ARGS`` / ``COPY_COLOR_TAG_ARGS`` -- BT.709 tagging for every
  encoded / stream-copied output.
- ``SCALE_FLAGS`` -- swscale algorithm/accuracy flags for every real resize.
"""

from src.services.ffmpeg.base import (
    COLOR_TAG_ARGS,
    COPY_COLOR_TAG_ARGS,
    SCALE_FLAGS,
    FFmpegError,
    ProbeResult,
)
from src.services.ffmpeg.ops import FFmpegOps

__all__ = [
    "COLOR_TAG_ARGS",
    "COPY_COLOR_TAG_ARGS",
    "SCALE_FLAGS",
    "FFmpegError",
    "FFmpegOps",
    "ProbeResult",
]
