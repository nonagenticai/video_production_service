"""Multi-input compose ops: concat, transitions, audio mixing, chapters."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import ClassVar

from src.services.ffmpeg.base import COLOR_TAG_ARGS
from src.services.ffmpeg.probe import ProbeMixin

logger = logging.getLogger(__name__)


class ComposeMixin(ProbeMixin):
    """Multi-input ffmpeg operations (concat, transitions, mixing)."""

    # Valid FFmpeg xfade transition names (FFmpeg 5.x+)
    XFADE_TRANSITIONS: ClassVar[set[str]] = {
        "fade",
        "wipeleft",
        "wiperight",
        "wipeup",
        "wipedown",
        "slideleft",
        "slideright",
        "slideup",
        "slidedown",
        "circlecrop",
        "rectcrop",
        "distance",
        "fadeblack",
        "fadewhite",
        "radial",
        "smoothleft",
        "smoothright",
        "smoothup",
        "smoothdown",
        "circleopen",
        "circleclose",
        "vertopen",
        "vertclose",
        "horzopen",
        "horzclose",
        "dissolve",
        "pixelize",
        "diagtl",
        "diagtr",
        "diagbl",
        "diagbr",
        "hlslice",
        "hrslice",
        "vuslice",
        "vdslice",
        "hblur",
        "fadegrays",
        "squeezev",
        "squeezeh",
        "zoomin",
        "coverup",
        "coverdown",
        "coverleft",
        "coverright",
        "revealup",
        "revealdown",
        "revealleft",
        "revealright",
    }

    @classmethod
    def concat(
        cls,
        file_list_path: Path,
        output_path: Path,
        crf: int = 20,
        fps: int = 30,
        cfr: bool = False,
    ) -> Path:
        """Concatenate video segments listed in a concat file.

        When ``cfr`` is True, emit ``-fps_mode cfr -r {fps}`` so the concatenated
        output is true constant-frame-rate. Defaults to False → byte-identical.
        """
        cfr_args = ["-fps_mode", "cfr", "-r", str(fps)] if cfr else []
        cls.run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(file_list_path),
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
                *cfr_args,
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def crossfade(
        cls,
        input1: Path,
        input2: Path,
        output: Path,
        fade_duration: float = 0.5,
        transition_type: str = "fade",
        fps: int = 30,
        cfr: bool = False,
    ) -> Path:
        """Join two videos with an xfade crossfade transition.

        When ``cfr`` is True, emit ``-fps_mode cfr -r {fps}``. Defaults to False
        with ``fps=30`` → byte-identical to the pre-CFR behaviour.
        """
        cfr_args = ["-fps_mode", "cfr", "-r", str(fps)] if cfr else []
        d1 = cls.probe_duration(input1)
        d2 = cls.probe_duration(input2)
        # An unbounded ``crossfade_duration`` drove ``offset`` to 0, which makes
        # the xfade start at t=0 and erases clip 1 entirely. Never spend more
        # than half of either clip on the transition.
        max_fade = min(d for d in (d1, d2) if d > 0) / 2 if (d1 > 0 or d2 > 0) else fade_duration
        if fade_duration > max_fade:
            logger.warning(
                "crossfade %.3fs exceeds half the shorter clip (%.3fs/%.3fs); clamping to %.3fs",
                fade_duration,
                d1,
                d2,
                max_fade,
            )
            fade_duration = max_fade
        offset = max(0, d1 - fade_duration)
        w1, h1 = cls.probe_dimensions(input1)
        w2, h2 = cls.probe_dimensions(input2)
        target_w = max(w1, w2) if w1 > 0 and w2 > 0 else (w1 or w2 or 1920)
        target_h = max(h1, h2) if h1 > 0 and h2 > 0 else (h1 or h2 or 1080)
        has_audio1 = cls.probe_has_audio(input1)
        has_audio2 = cls.probe_has_audio(input2)

        # Validate and default to "fade" if the transition name is unrecognised
        xfade_name = transition_type if transition_type in cls.XFADE_TRANSITIONS else "fade"

        vfilter = (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v0];"
            f"[1:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v1];"
            f"[v0][v1]xfade=transition={xfade_name}:duration={fade_duration}:offset={offset}[v]"
        )
        if has_audio1 and has_audio2:
            delay_ms = int(offset * 1000)
            afilter = (
                f";[0:a]apad=whole_dur={d1}[a0];"
                f"[1:a]adelay={delay_ms}|{delay_ms}[a1];"
                f"[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]"
            )
            maps = ["-map", "[v]", "-map", "[a]"]
        elif has_audio1:
            afilter = ";[0:a]anull[a]"
            maps = ["-map", "[v]", "-map", "[a]"]
        elif has_audio2:
            # Offset input2's audio to start at the crossfade point so it
            # aligns with the xfade video transition instead of playing from
            # the very beginning of the output.
            delay_ms = int(offset * 1000)
            afilter = f";[1:a]adelay={delay_ms}|{delay_ms}[a]"
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            afilter = ""
            maps = ["-map", "[v]"]

        cls.run(
            [
                "-i",
                str(input1),
                "-i",
                str(input2),
                "-filter_complex",
                vfilter + afilter,
                *maps,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-max_muxing_queue_size",
                "4096",
                *cfr_args,
                str(output),
            ]
        )
        return output

    @classmethod
    def hstack(
        cls,
        left: Path,
        right: Path,
        output: Path,
        target_width: int = 1920,
        target_height: int = 1080,
        audio_from: int | None = 0,
        fps: int = 30,
        cfr: bool = False,
    ) -> Path:
        """Place two videos side-by-side into a single output.

        Each input is scaled+padded to ``target_width // 2`` x ``target_height``
        and combined with FFmpeg's ``hstack`` filter so the result matches
        ``target_width`` x ``target_height``. Audio is taken from the input
        indicated by ``audio_from`` (0=left, 1=right, ``None`` to drop audio).
        Used by the AA trilogy recordings to compose fr/ + n8n side-by-side.
        """
        if target_width % 2 != 0:
            raise ValueError(f"target_width must be even, got {target_width}")
        if audio_from not in (None, 0, 1):
            raise ValueError(f"audio_from must be None, 0 or 1, got {audio_from}")
        cfr_args = ["-fps_mode", "cfr", "-r", str(fps)] if cfr else []
        half_w = target_width // 2
        vfilter = (
            f"[0:v]scale={half_w}:{target_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={half_w}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v0];"
            f"[1:v]scale={half_w}:{target_height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={half_w}:{target_height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v1];"
            f"[v0][v1]hstack=inputs=2[v]"
        )
        maps = ["-map", "[v]"]
        if audio_from is not None and cls.probe_has_audio(left if audio_from == 0 else right):
            maps += ["-map", f"{audio_from}:a"]
        cls.run(
            [
                "-i",
                str(left),
                "-i",
                str(right),
                "-filter_complex",
                vfilter,
                *maps,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-max_muxing_queue_size",
                "4096",
                *cfr_args,
                str(output),
            ]
        )
        return output

    @classmethod
    def fade_to_black(
        cls,
        input1: Path,
        input2: Path,
        output: Path,
        fade_duration: float = 0.5,
        fps: int = 30,
        cfr: bool = False,
    ) -> Path:
        """Join two videos with a fade-to-black transition.

        When ``cfr`` is True, emit ``-fps_mode cfr -r {fps}``. Defaults to False
        with ``fps=30`` → byte-identical to the pre-CFR behaviour.
        """
        cfr_args = ["-fps_mode", "cfr", "-r", str(fps)] if cfr else []
        d1 = cls.probe_duration(input1)
        fade_out_start = max(0, d1 - fade_duration)
        w1, h1 = cls.probe_dimensions(input1)
        w2, h2 = cls.probe_dimensions(input2)
        target_w = max(w1, w2) if w1 > 0 and w2 > 0 else (w1 or w2 or 1920)
        target_h = max(h1, h2) if h1 > 0 and h2 > 0 else (h1 or h2 or 1080)
        has_audio1 = cls.probe_has_audio(input1)
        has_audio2 = cls.probe_has_audio(input2)

        vfilter = (
            f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"fade=t=out:st={fade_out_start}:d={fade_duration}[v0];"
            f"[1:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"fade=t=in:d={fade_duration}[v1];"
            f"[v0][v1]concat=n=2:v=1:a=0[v]"
        )
        if has_audio1 and has_audio2:
            afilter = (
                f";[0:a]afade=t=out:st={fade_out_start}:d={fade_duration}[a0];"
                f"[1:a]afade=t=in:d={fade_duration}[a1];"
                f"[a0][a1]concat=n=2:v=0:a=1[a]"
            )
            maps = ["-map", "[v]", "-map", "[a]"]
        elif has_audio1:
            afilter = f";[0:a]afade=t=out:st={fade_out_start}:d={fade_duration}[a]"
            maps = ["-map", "[v]", "-map", "[a]"]
        elif has_audio2:
            afilter = f";[1:a]afade=t=in:d={fade_duration}[a]"
            maps = ["-map", "[v]", "-map", "[a]"]
        else:
            afilter = ""
            maps = ["-map", "[v]"]

        cls.run(
            [
                "-i",
                str(input1),
                "-i",
                str(input2),
                "-filter_complex",
                vfilter + afilter,
                *maps,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                cls._default_preset,
                "-pix_fmt",
                "yuv420p",
                *COLOR_TAG_ARGS,
                "-max_muxing_queue_size",
                "4096",
                *cfr_args,
                str(output),
            ]
        )
        return output

    @classmethod
    def mix_voiceover(
        cls,
        video_path: Path,
        voiceover_path: Path,
        output_path: Path,
        music_path: Path | None = None,
        voiceover_volume: float = 0.9,
        music_volume: float = 0.1,
    ) -> Path:
        """Mix a voiceover track into a video, optionally ducking background music."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        video_duration = cls.probe_duration(video_path)
        vo_dur = cls.probe_duration(voiceover_path)

        # When the voiceover runs LONGER than the assembled video, freeze the
        # last video frame (tpad clone) so the narration tail plays over a held
        # frame instead of a dead/absent video tail. tpad forces a video
        # re-encode, so the `-c:v copy` fast-path is kept only when the video
        # already covers the full mixed audio. Mirrors the no-audio branch.
        freeze_video = bool(video_duration) and vo_dur > video_duration
        freeze_extra = vo_dur - video_duration + 0.5 if freeze_video else 0.0
        video_prefix = (
            f"[0:v]tpad=stop_mode=clone:stop_duration={freeze_extra:.3f}[v];"
            if freeze_video
            else ""
        )
        video_map = ["-map", "[v]"] if freeze_video else ["-map", "0:v"]
        # The freeze branch re-encodes (tpad), so encoder colour flags reach it.
        # The ``-c:v copy`` branch cannot be tagged by any encoder flag — its
        # BT.709 description has to be rewritten in the H.264 SPS VUI by the
        # h264_metadata bitstream filter -- which HARD-ERRORS on a non-H.264
        # stream, so it is gated on the source's probed codec
        # (``copy_color_tag_args``) rather than assumed.
        video_codec: list[str] = (
            [*COLOR_TAG_ARGS]
            if freeze_video
            else ["-c:v", "copy", *cls.copy_color_tag_args(video_path)]
        )

        if music_path and music_path.exists():
            vo_pad = f"apad=whole_dur={video_duration}," if video_duration else ""
            cls.run(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(voiceover_path),
                    "-i",
                    str(music_path),
                    "-filter_complex",
                    (
                        f"{video_prefix}"
                        f"[1:a]volume={voiceover_volume},{vo_pad}asplit=2[vo_mix][vo_sc];"
                        f"[2:a]volume={music_volume}[music];"
                        f"[music][vo_sc]sidechaincompress=threshold=0.125:ratio=9:attack=200:release=1000[ducked];"
                        f"[vo_mix][ducked]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
                    ),
                    *video_map,
                    "-map",
                    "[aout]",
                    *video_codec,
                    "-max_muxing_queue_size",
                    "4096",
                    str(output_path),
                ]
            )
        else:
            has_video_audio = cls.probe_has_audio(video_path)
            if has_video_audio:
                vo_pad = f"apad=whole_dur={video_duration}," if video_duration else ""
                cls.run(
                    [
                        "-i",
                        str(video_path),
                        "-i",
                        str(voiceover_path),
                        "-filter_complex",
                        (
                            f"{video_prefix}"
                            f"[1:a]volume={voiceover_volume},{vo_pad}asetpts=PTS-STARTPTS[vo];"
                            f"[0:a][vo]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
                        ),
                        *video_map,
                        "-map",
                        "[aout]",
                        *video_codec,
                        "-max_muxing_queue_size",
                        "4096",
                        str(output_path),
                    ]
                )
            else:
                if freeze_video:
                    # Voiceover longer than video: freeze last frame
                    cls.run(
                        [
                            "-i",
                            str(video_path),
                            "-i",
                            str(voiceover_path),
                            "-filter_complex",
                            f"[0:v]tpad=stop_mode=clone:stop_duration={freeze_extra:.3f}[v];"
                            f"[1:a]volume={voiceover_volume},asetpts=PTS-STARTPTS[vo]",
                            "-map",
                            "[v]",
                            "-map",
                            "[vo]",
                            *COLOR_TAG_ARGS,
                            "-max_muxing_queue_size",
                            "4096",
                            str(output_path),
                        ]
                    )
                else:
                    # Voiceover shorter or equal: pad audio with silence
                    vo_pad = (
                        f"apad=whole_dur={video_duration},"
                        if video_duration and vo_dur < video_duration
                        else ""
                    )
                    cls.run(
                        [
                            "-i",
                            str(video_path),
                            "-i",
                            str(voiceover_path),
                            "-filter_complex",
                            f"[1:a]volume={voiceover_volume},{vo_pad}asetpts=PTS-STARTPTS[vo]",
                            "-map",
                            "0:v",
                            "-map",
                            "[vo]",
                            "-c:v",
                            "copy",
                            *cls.copy_color_tag_args(video_path),
                            "-max_muxing_queue_size",
                            "4096",
                            str(output_path),
                        ]
                    )
        return output_path

    @classmethod
    def concat_audio(
        cls, file_list_path: Path, output_path: Path, audio_format: str = "mp3"
    ) -> Path:
        """Concatenate audio files (WP-10)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec_map = {"mp3": "libmp3lame", "wav": "pcm_s16le", "flac": "flac"}
        codec = codec_map.get(audio_format, "libmp3lame")
        cls.run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(file_list_path),
                "-acodec",
                codec,
                str(output_path),
            ]
        )
        return output_path

    @classmethod
    def embed_chapters(
        cls,
        video_path: Path,
        output_path: Path,
        chapters: list[dict],
    ) -> Path:
        """Embed chapter markers into an MP4 container via ffmetadata.

        Each chapter dict must have: chapter_number, title, start_seconds,
        duration_seconds.  Optionally subtitle.

        Creates a temporary FFMETADATA1 file and remuxes the video with
        chapter metadata using ``-map_metadata 1``.
        """
        if not chapters:
            return video_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build FFMETADATA1 content
        lines = [";FFMETADATA1"]
        for ch in chapters:
            start_ms = int(ch["start_seconds"] * 1000)
            end_ms = start_ms + int(ch["duration_seconds"] * 1000)
            title = ch.get("title", f"Chapter {ch['chapter_number']}")
            lines.append("[CHAPTER]")
            lines.append("TIMEBASE=1/1000")
            lines.append(f"START={start_ms}")
            lines.append(f"END={end_ms}")
            lines.append(f"title={title}")

        metadata_path = video_path.parent / f"{video_path.stem}_chapters.txt"
        metadata_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cls.run(
            [
                "-i",
                str(video_path),
                "-i",
                str(metadata_path),
                "-map_metadata",
                "1",
                "-map",
                "0:v",
                "-map",
                "0:a?",
                "-c",
                "copy",
                *cls.copy_color_tag_args(video_path),
                str(output_path),
            ]
        )

        # Clean up temp metadata file
        with contextlib.suppress(OSError):
            metadata_path.unlink()

        return output_path

    @classmethod
    def mix_background_music(
        cls,
        video_path: Path,
        music_path: Path,
        output_path: Path,
        volume: float = 0.1,
        fade_in_s: float = 2.0,
        fade_out_s: float = 3.0,
    ) -> Path:
        """Mix background music into a video with volume and fade controls."""
        duration = cls.probe_duration(video_path)
        if duration <= 0:
            import shutil

            shutil.copy2(video_path, output_path)
            return output_path
        fade_out_start = max(0, duration - fade_out_s)
        has_audio = cls.probe_has_audio(video_path)
        music_filter = (
            f"[1:a]volume={volume},"
            f"afade=t=in:d={fade_in_s},"
            f"afade=t=out:st={fade_out_start}:d={fade_out_s}[music]"
        )
        if has_audio:
            cls.run(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(music_path),
                    "-filter_complex",
                    f"{music_filter};[0:a][music]amix=inputs=2:duration=first[aout]",
                    "-map",
                    "0:v",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    *cls.copy_color_tag_args(video_path),
                    "-max_muxing_queue_size",
                    "4096",
                    str(output_path),
                ]
            )
        else:
            cls.run(
                [
                    "-i",
                    str(video_path),
                    "-i",
                    str(music_path),
                    "-filter_complex",
                    music_filter,
                    "-map",
                    "0:v",
                    "-map",
                    "[music]",
                    "-c:v",
                    "copy",
                    *cls.copy_color_tag_args(video_path),
                    "-shortest",
                    "-max_muxing_queue_size",
                    "4096",
                    str(output_path),
                ]
            )
        return output_path
