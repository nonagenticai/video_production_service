"""FFmpeg base: error type, probe result, and FFmpegOpsBase with run/configure."""

from __future__ import annotations

import logging
import subprocess  # nosec B404
import threading
from dataclasses import dataclass
from typing import ClassVar

from src.config import get_settings

logger = logging.getLogger(__name__)

# Every video this service emits must be tagged BT.709 (primaries / transfer /
# matrix). Untagged H.264 leaves the decoder to guess, and players disagree —
# the delivered master then reads as "colorspace_ok = false" downstream. The
# video_compiler already does this (``mastering.colorspace_tag_args``); this
# service was the sole producer of untagged streams.
#
# Encoder-side form: valid for libx264 AND libvpx-vp9 (these are generic
# AVCodecContext options, not encoder-private ones), so one constant covers
# both. Splice it into every OUTPUT arg list, never into input args.
# swscale algorithm + accuracy flags for every REAL resize this service emits.
#
# A bare ``scale=`` leaves the algorithm to swscale's default (bilinear), which
# softens both the downscale of captured screen pixels and — worse — the 1.15x
# UPSCALE the Ken Burns motion presets perform on a generated still before
# zoompan crops back into it. ``accurate_rnd`` drops swscale's fast rounding
# and ``full_chroma_int`` interpolates chroma at full resolution instead of
# reusing the subsampled plane, which is what keeps coloured labels and caption
# text from bleeding. Append with a leading ``:`` to a scale filter's options.
SCALE_FLAGS: str = "flags=lanczos+accurate_rnd+full_chroma_int"

COLOR_TAG_ARGS: list[str] = [
    "-color_primaries",
    "bt709",
    "-color_trc",
    "bt709",
    "-colorspace",
    "bt709",
]

# Stream-copy form. ``-c:v copy`` hands the bitstream through untouched, so no
# encoder flag can reach it — the tag has to be rewritten in the H.264 SPS VUI
# by a bitstream filter. The numeric codes are the BT.709 entries of the H.264
# spec's colour-description tables (Table E-3/E-4/E-5): 1 = BT.709 for all
# three, plus limited range. NOTE the BRITISH spelling of ``colour_primaries``
# — that is what the ``h264_metadata`` bsf actually names the option.
# Only valid on an H.264 video stream; do not splice it into a copy whose
# source codec is unknown (a VP9/WebM recording would hard-error).
COPY_COLOR_TAG_ARGS: list[str] = [
    "-bsf:v",
    (
        "h264_metadata=colour_primaries=1:transfer_characteristics=1"
        ":matrix_coefficients=1:video_full_range_flag=0"
    ),
]


class FFmpegError(RuntimeError):
    """Custom exception for FFmpeg failures with structured error info."""

    def __init__(
        self,
        message: str,
        command: str = "",
        return_code: int = -1,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.command = command
        self.return_code = return_code
        self.stderr = stderr


@dataclass
class ProbeResult:
    """Cached ffprobe result for a single media file."""

    duration: float
    width: int
    height: int
    has_audio: bool
    # Defaulted so the many existing ``ProbeResult(...)`` call sites (and test
    # doubles) keep working. ``video_frames`` is 0 when ffprobe reports no
    # ``nb_frames`` for the container, so a 0 must never be read as "no frames".
    has_video: bool = False
    video_frames: int = 0
    # True unless ``probe_all`` swallowed a probe failure (ffprobe crashed,
    # timed out, or returned unparseable output) into this zero-valued
    # result. Defaulted to True for the same reason as the fields above: the
    # many existing ``ProbeResult(...)`` call sites and test doubles build a
    # result for a file they DID successfully probe (or don't care), and must
    # keep working unchanged. A caller that bills off ``duration`` must check
    # this first -- ``duration=0.0, probed=True`` is a genuinely empty file;
    # ``probed=False`` means the file was never actually read and the 0.0 is
    # not a measurement.
    probed: bool = True


class FFmpegOpsBase:
    """Shared class state + ffmpeg subprocess driver for all FFmpegOps mixins."""

    _default_timeout: int = 300
    _default_preset: str = "medium"
    FFPROBE_TIMEOUT: int = 10

    # Shared probe cache + lock.  These are defined ONCE on the base so every
    # mixin that does ``cls._probe_cache`` resolves to the same object.
    _probe_cache: ClassVar[dict[str, ProbeResult]] = {}
    _probe_lock: ClassVar[threading.Lock] = threading.Lock()

    # Process-wide gate on concurrent ffmpeg subprocesses. Compilation runs each
    # ffmpeg step on a worker thread (asyncio.to_thread), so multiple steps —
    # across multiple jobs — can hit ``run`` at once; without a cap their
    # combined full-resolution decode buffers OOM-kill ffmpeg (exit=-9) under
    # load. A threading (not asyncio) primitive is correct because ``run`` is
    # synchronous and called from those worker threads. Built lazily so ``run``
    # works in tests that never call ``configure_from_settings``; rebuilt to the
    # configured size when settings load.
    _max_concurrent_subprocesses: ClassVar[int] = 2
    _run_semaphore: ClassVar[threading.BoundedSemaphore | None] = None
    _run_semaphore_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _get_run_semaphore(cls) -> threading.BoundedSemaphore:
        sem = cls._run_semaphore
        if sem is None:
            with cls._run_semaphore_lock:
                if cls._run_semaphore is None:
                    cls._run_semaphore = threading.BoundedSemaphore(
                        cls._max_concurrent_subprocesses
                    )
                sem = cls._run_semaphore
        return sem

    @classmethod
    def configure_from_settings(cls) -> None:
        """Load FFmpeg defaults from application settings."""
        settings = get_settings()
        cls._default_timeout = settings.ffmpeg.timeout
        cls._default_preset = settings.ffmpeg.preset
        cls.FFPROBE_TIMEOUT = settings.ffmpeg.ffprobe_timeout
        cls._max_concurrent_subprocesses = settings.compilation.max_concurrent_ffmpeg_subprocesses
        # Rebuild the gate to the configured size (config may differ from the
        # lazy default if a step already ran before startup wiring).
        with cls._run_semaphore_lock:
            cls._run_semaphore = threading.BoundedSemaphore(cls._max_concurrent_subprocesses)

    @classmethod
    def run(cls, args: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
        """Execute an ffmpeg command with the given arguments.

        Args:
            args: FFmpeg CLI arguments (without the leading 'ffmpeg -y').
            timeout: Override timeout in seconds; defaults to configured timeout.

        Returns:
            The completed subprocess result.

        Raises:
            FFmpegError: If ffmpeg returns a non-zero exit code, times out, or
                is not found on PATH.
        """
        effective_timeout = timeout if timeout is not None else cls._default_timeout
        cmd = ["ffmpeg", "-y", *args]
        logger.debug("Running: %s", " ".join(cmd))
        # Hold the gate only around the actual exec (the memory-heavy part), so
        # the cheap error-formatting below doesn't occupy a slot. The context
        # manager releases the slot even if subprocess.run raises (e.g.
        # TimeoutExpired).
        try:
            with cls._get_run_semaphore():
                result = subprocess.run(  # nosec B603  # noqa: S603 -- ffmpeg/ffprobe with controlled args
                    cmd, capture_output=True, text=True, timeout=effective_timeout
                )
        except FileNotFoundError:
            raise FFmpegError("ffmpeg not found on PATH", command=" ".join(cmd)) from None
        except subprocess.TimeoutExpired as exc:
            # subprocess.run has already killed and reaped the child, so there is
            # nothing to clean up here — but the raw TimeoutExpired must not leak
            # out of a paid stage: every caller in filters/compose only handles
            # FFmpegError. Mirrors the normalization in probe.py.
            #
            # Carry ffmpeg's partial stderr the same way the non-zero-exit path
            # below does. subprocess.run drains the pipes after killing the
            # child, so whatever ffmpeg managed to say (the filtergraph error it
            # was stuck on, the last frame= progress line) is on the exception —
            # dropping it leaves the operator with a bare "timed out" and no clue
            # WHERE. text=True makes these str, but typeshed types them bytes and
            # a caller could pass text=False, so decode defensively.
            partial: object = exc.stderr
            if isinstance(partial, bytes | bytearray):
                partial = partial.decode("utf-8", errors="replace")
            stderr_tail = str(partial or "")[-800:]
            logger.error(
                "ffmpeg timed out after %ss:\n  cmd: %s\n  partial stderr: %s",
                effective_timeout,
                " ".join(cmd),
                stderr_tail,
            )
            # The message stays free of the command line and of the tail on
            # purpose: it is persisted to the job's ``error`` column and shipped
            # to the tenant webhook. The tail rides on ``.stderr`` for operators.
            raise FFmpegError(
                f"ffmpeg timed out after {effective_timeout}s",
                command=" ".join(cmd),
                stderr=stderr_tail,
            ) from None
        if result.returncode != 0:
            # FFmpeg writes version/config at the start of stderr and the actual
            # error at the end.  Log generously and put the *tail* into the
            # exception message so the real reason is visible in poll responses.
            stderr_full = result.stderr
            cmd_str = " ".join(cmd)
            logger.error(
                "ffmpeg error (rc=%d):\n  cmd: %s\n  stderr: %s",
                result.returncode,
                cmd_str,
                stderr_full[-3000:],
            )
            stderr_tail = stderr_full[-800:].strip()
            raise FFmpegError(
                f"ffmpeg failed (rc={result.returncode}): {stderr_tail}",
                command=cmd_str,
                return_code=result.returncode,
                stderr=stderr_full[-4000:],
            )
        return result
