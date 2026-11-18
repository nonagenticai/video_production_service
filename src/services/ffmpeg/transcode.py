"""Transcode + per-clip operations: convert, speed, trim, image-to-video, audio extract/strip."""

from __future__ import annotations

import logging
from pathlib import Path

from src.services.ffmpeg.base import COLOR_TAG_ARGS, SCALE_FLAGS
from src.services.ffmpeg.probe import ProbeMixin
from src.utils.motion_presets import DEFAULT_MOTION_PRESET, MOTION_PRESETS

logger = logging.getLogger(__name__)


class TranscodeMixin(ProbeMixin):
    """Single-input transcode + clip-level helpers.

    Inherits from :class:`ProbeMixin` so ``probe_has_audio`` is in scope
    for type checking without redefining the shared base state.
    """

    @classmethod
    def convert_to_mp4(
        cls,
        input_path: Path,
        output_path: Path,
        crf: int = 20,
        start_offset_s: float = 0.0,
    ) -> Path:
        """Re-encode a video file to H.264 MP4 with a silent audio track if needed.

        ``start_offset_s`` (>0) drops that many seconds from the front via input
        seeking before ``-i``; because this path re-encodes, the seek is frame
        accurate. Used to trim a blank page-load lead-in from URL recordings.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        has_audio = cls.probe_has_audio(input_path)
        seek = ["-ss", str(start_offset_s)] if start_offset_s > 0 else []
        if has_audio:
            cls.run(
                [
                    *seek,
                    "-i",
                    str(input_path),
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-crf",
                    str(crf),
                    "-preset",
                    cls._default_preset,
                    "-pix_fmt",
                    "yuv420p",
                    *COLOR_TAG_ARGS,
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ]
            )
        else:
            cls.run(
                [
                    *seek,
                    "-i",
                    str(input_path),
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-crf",
                    str(crf),
                    "-preset",
                    cls._default_preset,
                    "-pix_fmt",
                    "yuv420p",
                    *COLOR_TAG_ARGS,
                    "-movflags",
                    "+faststart",
                    "-shortest",
                    str(output_path),
                ]
            )
        return output_path

    @classmethod
    def speed_adjust(cls, input_path: Path, output_path: Path, factor: float) -> Path:
        """Speed up or slow down a video by the given factor (>1 = faster)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pts = 1.0 / factor
        has_audio = cls.probe_has_audio(input_path)
        codec_args = [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            cls._default_preset,
            "-pix_fmt",
            "yuv420p",
            *COLOR_TAG_ARGS,
        ]
        if has_audio:
            atempo_filters = cls._build_atempo_chain(factor)
            cls.run(
                [
                    "-i",
                    str(input_path),
                    "-filter_complex",
                    f"[0:v]setpts={pts}*PTS,fps=30[v];[0:a]{atempo_filters}[a]",
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    *codec_args,
                    "-c:a",
                    "aac",
                    "-max_muxing_queue_size",
                    "4096",
                    str(output_path),
                ]
            )
        else:
            cls.run(
                [
                    "-i",
                    str(input_path),
                    "-filter:v",
                    f"setpts={pts}*PTS,fps=30",
                    "-an",
                    *codec_args,
                    str(output_path),
                ]
            )
        return output_path

    @staticmethod
    def _build_atempo_chain(factor: float) -> str:
        if factor <= 0:
            raise ValueError(f"Speed factor must be positive, got {factor}")
        if factor < 0.25:
            logger.warning("Speed factor %.4f clamped to minimum 0.25", factor)
            factor = 0.25
        if abs(factor - 1.0) < 1e-6:
            return "atempo=1.0"
        filters: list[str] = []
        remaining = factor
        iterations = 0
        while remaining > 100.0 and iterations < 10:
            filters.append("atempo=100.0")
            remaining /= 100.0
            iterations += 1
        while remaining < 0.5 and iterations < 10:
            filters.append("atempo=0.5")
            remaining /= 0.5
            iterations += 1
        remaining = max(0.5, min(remaining, 100.0))
        filters.append(f"atempo={remaining}")
        return ",".join(filters)

    @classmethod
    def trim_reencode(
        cls,
        input_path: Path,
        output_path: Path,
        start_s: float,
        duration_s: float,
        crf: int = 20,
    ) -> Path:
        """Frame-accurate trim to an H.264 MP4 clip of ``duration_s`` starting at
        ``start_s``.

        Unlike :meth:`trim` (stream copy, keyframe-bounded and container-locked),
        this re-encodes so the start/length are exact and the output is a clean
        H.264 MP4 regardless of the source codec — needed to slice per-screen
        interaction clips out of a single continuous WebM walk, where the cut
        must land right before the click. Input-seek (``-ss`` before ``-i``)
        with re-encode is frame-accurate. Video only: the source walk has no
        audio and the compiler synthesises silence when muxing.
        """
        if duration_s <= 0:
            raise ValueError(f"trim_reencode() duration_s must be > 0 (got {duration_s})")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        seek = ["-ss", str(start_s)] if start_s > 0 else []
        cls.run(
            [
                *seek,
                "-i",
                str(input_path),
                "-t",
                str(duration_s),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def trim(cls, input_path: Path, output_path: Path, start_s: float, end_s: float) -> Path:
        """Trim a video to the given start/end timestamps using stream copy."""
        if end_s <= start_s:
            raise ValueError(f"trim() end_s ({end_s}) must be > start_s ({start_s})")
        duration = end_s - start_s
        cls.run(
            [
                "-ss",
                str(start_s),
                "-i",
                str(input_path),
                "-t",
                str(duration),
                "-c",
                "copy",
                *cls.copy_color_tag_args(input_path),
                str(output_path),
            ]
        )
        return output_path

    # Stills and Ken Burns clips were the only encodes in this service that
    # passed ``-preset`` without ``-crf``, i.e. libx264's default 23 — the exact
    # bottleneck already removed from the capture transcode. 18 matches what the
    # master is graded against, on the one source (a generated image) that has
    # no camera noise to hide quantisation in.
    _STILL_CRF: int = 18

    @classmethod
    def image_to_video(
        cls,
        image_path: Path,
        output_path: Path,
        duration: float,
        width: int = 1920,
        height: int = 1080,
    ) -> Path:
        """Convert a static image to a video clip with the given duration and silent audio."""
        cls.run(
            [
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-t",
                str(duration),
                "-crf",
                str(cls._STILL_CRF),
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-vf",
                f"scale={width}:{height}:{SCALE_FLAGS}",
                "-shortest",
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def image_to_motion_video(
        cls,
        image_path: Path,
        output_path: Path,
        duration: float,
        width: int = 1920,
        height: int = 1080,
        motion_preset: str = DEFAULT_MOTION_PRESET,
    ) -> Path:
        """Convert a static image to a video clip with a simple motion preset."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_duration = max(float(duration), 1.0)
        fps = 30
        total_frames = max(round(safe_duration * fps), fps)
        zoomed_w = max(round(width * 1.15), width + 2)
        zoomed_h = max(round(height * 1.15), height + 2)
        preset = (motion_preset or DEFAULT_MOTION_PRESET).strip().lower()

        if preset == "zoom_in":
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"zoompan=z='min(1+0.15*(on/{total_frames}),1.15)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={width}x{height}:fps={fps}"
            )
        elif preset == "zoom_out":
            # Mirror of zoom_in: start at the 1.15 over-scale and settle back to
            # frame-fill. Clamped with max() so the last frame lands exactly on
            # 1.0 instead of drifting below it (which would expose an edge).
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"zoompan=z='max(1.15-0.15*(on/{total_frames}),1.0)':"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"d={total_frames}:s={width}x{height}:fps={fps}"
            )
        elif preset == "pan_left":
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"crop={width}:{height}:"
                f"x='(in_w-out_w)*(1-min(t/{safe_duration},1))':"
                f"y='(in_h-out_h)/2',fps={fps}"
            )
        elif preset == "pan_right":
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"crop={width}:{height}:"
                f"x='(in_w-out_w)*min(t/{safe_duration},1)':"
                f"y='(in_h-out_h)/2',fps={fps}"
            )
        elif preset == "pan_up":
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"crop={width}:{height}:"
                f"x='(in_w-out_w)/2':"
                f"y='(in_h-out_h)*(1-min(t/{safe_duration},1))',fps={fps}"
            )
        elif preset == "pan_down":
            vf = (
                f"scale={zoomed_w}:{zoomed_h}:{SCALE_FLAGS},"
                f"crop={width}:{height}:"
                f"x='(in_w-out_w)/2':"
                f"y='(in_h-out_h)*min(t/{safe_duration},1)',fps={fps}"
            )
        else:
            # FAIL LOUD. Falling back to ``image_to_video`` here used to emit a
            # bare ``scale=`` filter, i.e. a 100%-static clip, while the job
            # went green — a direct contributor to the duplicate-frame class.
            raise ValueError(
                f"unknown motion_preset {motion_preset!r}; "
                f"expected one of {', '.join(MOTION_PRESETS)}"
            )

        cls.run(
            [
                "-loop",
                "1",
                "-i",
                str(image_path),
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-t",
                str(safe_duration),
                "-crf",
                str(cls._STILL_CRF),
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-vf",
                vf,
                "-shortest",
                str(output_path),
            ]
        )
        return output_path

    # ------------------------------------------------------------------
    # Frame-sequence assembly (deterministic motion-graphics capture).
    #
    # The animation recorder screenshots one PNG per timeline frame
    # (``frame_000000.png`` …) and these helpers assemble that sequence into a
    # clip at a fixed fps — so the output is identical regardless of machine
    # load, unlike Playwright's wall-clock ``record_video_dir`` path.
    # ------------------------------------------------------------------

    _FRAME_GLOB: str = "frame_%06d.png"
    # Force even output dimensions: yuv420p (H.264 baseline) requires both
    # width and height divisible by 2, and a screenshot of an odd-sized
    # template would otherwise fail the encode.
    # NO resampler flags here, deliberately: this filter only rounds each dimension
    # DOWN to the nearest even number (at most a 1px change, required by yuv420p).
    # It is not a resize, so a resampler choice cannot improve it -- and on the alpha
    # twin below, full_chroma_int would touch chroma handling on a path that carries
    # transparency. Lanczos belongs on the real resizes (still->video, Ken Burns).
    _EVEN_DIMS_VF: str = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    # Alpha-preserving twin of the above, for transparent overlay renders.
    #
    # The scale expression itself does NOT destroy alpha: swscale carries the
    # fourth plane through whenever the negotiated output format has one, and
    # there is no alphaextract / alphamerge / overlay-onto-opaque anywhere in
    # this chain. What it DOES do is leave the pixel format to filtergraph
    # negotiation — and every alpha-capable encoder we use also advertises a
    # non-alpha format, so the graph is free to settle on plain yuv420p and
    # drop the plane without a word. Pinning ``format=yuva420p`` as the last
    # link removes that freedom INSIDE the graph, rather than merely asking the
    # encoder for it via ``-pix_fmt`` after the fact.
    _EVEN_DIMS_ALPHA_VF: str = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuva420p"

    @classmethod
    def frames_to_mp4(
        cls,
        frames_dir: Path,
        output_path: Path,
        *,
        fps: int,
        crf: int = 20,
        frame_glob: str | None = None,
        timeout: int | None = None,
    ) -> Path:
        """Assemble a ``frame_%06d.png`` sequence into an H.264 MP4 at ``fps``.

        A silent stereo audio track is muxed in (mirroring
        :meth:`image_to_video`) so the clip is a valid standalone video and
        slots into the compiler's shot list exactly like a recorded clip.

        ``timeout`` overrides the ffmpeg subprocess timeout (default:
        ``cls._default_timeout``) -- the animation recorder passes its own
        per-job encode budget so a stuck encode is killed at the OS level
        within that budget rather than the class-wide default.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / (frame_glob or cls._FRAME_GLOB))
        cls.run(
            [
                "-framerate",
                str(fps),
                "-i",
                pattern,
                "-f",
                "lavfi",
                "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-crf",
                str(crf),
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-vf",
                cls._EVEN_DIMS_VF,
                "-r",
                str(fps),
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            timeout=timeout,
        )
        return output_path

    @classmethod
    def frames_to_webm(
        cls,
        frames_dir: Path,
        output_path: Path,
        *,
        fps: int,
        crf: int = 32,
        frame_glob: str | None = None,
        transparent: bool = False,
        timeout: int | None = None,
    ) -> Path:
        """Assemble a frame sequence into a VP9 WebM at ``fps`` (no audio track).

        ``transparent=True`` keeps the PNG sequence's alpha channel in the
        encoded stream (``yuva420p``), so the clip can be composited OVER other
        footage as a lower-third or logo bug. This is the only encode path in
        this class that can carry alpha at all — H.264 has no alpha plane and
        GIF's is a single bit (hard-keyed edges, useless for an anti-aliased
        overlay) — which is why the animation recorder forces this container for
        a transparent render instead of quietly degrading to an opaque MP4.

        The default (``False``) emits the pre-alpha argv verbatim. ``timeout``
        overrides the ffmpeg subprocess timeout -- see :meth:`frames_to_mp4`.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / (frame_glob or cls._FRAME_GLOB))
        args = [
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p" if transparent else "yuv420p",
            "-b:v",
            "0",
            "-crf",
            str(crf),
            *COLOR_TAG_ARGS,
            "-vf",
            cls._EVEN_DIMS_ALPHA_VF if transparent else cls._EVEN_DIMS_VF,
            "-r",
            str(fps),
        ]
        if transparent:
            # libvpx carries VP9 alpha as a second per-frame payload muxed
            # alongside the colour frame. Alt-ref (hidden/golden) frames are
            # synthesised by the encoder and have no such counterpart, so a
            # build that elects to use them can emit frames whose alpha is
            # undefined. Disabling alt-ref and the lookahead that feeds it
            # costs a little compression on a short overlay clip and buys a
            # well-defined alpha plane on every single frame.
            args += ["-auto-alt-ref", "0", "-lag-in-frames", "0"]
        args.append(str(output_path))
        cls.run(args, timeout=timeout)
        return output_path

    @classmethod
    def frames_to_gif(
        cls,
        frames_dir: Path,
        output_path: Path,
        *,
        fps: int,
        frame_glob: str | None = None,
        timeout: int | None = None,
    ) -> Path:
        """Assemble a frame sequence into a high-quality looping GIF at ``fps``.

        Two-pass palettegen/paletteuse — a single global palette plus Bayer
        dithering keeps gradients clean at a fraction of the size of a naive
        256-colour quantise. ``timeout`` overrides the ffmpeg subprocess
        timeout for EACH pass -- see :meth:`frames_to_mp4`.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / (frame_glob or cls._FRAME_GLOB))
        palette = output_path.with_suffix(".palette.png")
        try:
            cls.run(
                [
                    "-framerate",
                    str(fps),
                    "-i",
                    pattern,
                    "-vf",
                    "palettegen=stats_mode=diff",
                    str(palette),
                ],
                timeout=timeout,
            )
            cls.run(
                [
                    "-framerate",
                    str(fps),
                    "-i",
                    pattern,
                    "-i",
                    str(palette),
                    "-lavfi",
                    "paletteuse=dither=bayer:bayer_scale=3",
                    "-r",
                    str(fps),
                    str(output_path),
                ],
                timeout=timeout,
            )
        finally:
            palette.unlink(missing_ok=True)
        return output_path

    @classmethod
    def strip_audio(cls, input_path: Path, output_path: Path) -> Path:
        """Remove audio track from a video file (WP-1)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cls.run(
            [
                "-i",
                str(input_path),
                "-an",
                "-c:v",
                "copy",
                *cls.copy_color_tag_args(input_path),
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def extract_audio(
        cls, video_path: Path, output_path: Path, audio_format: str = "mp3", bitrate: str = "192k"
    ) -> Path:
        """Extract audio track from a video file (WP-10)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec_map = {"mp3": "libmp3lame", "wav": "pcm_s16le", "flac": "flac"}
        codec = codec_map.get(audio_format, "libmp3lame")
        args = ["-i", str(video_path), "-vn", "-acodec", codec]
        if audio_format == "mp3":
            args.extend(["-b:a", bitrate])
        args.append(str(output_path))
        cls.run(args)
        return output_path
