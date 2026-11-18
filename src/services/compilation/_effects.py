"""Effects stage helpers (mixin)."""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.services.compilation._compiler_context import _JobContext
from src.utils.error_formatting import format_error
from src.utils.url_guard import resolve_local_asset_path

if TYPE_CHECKING:
    from src.services.ffmpeg import FFmpegOps as _FFmpegOpsType


def _ffmpeg() -> type[_FFmpegOpsType]:
    """Resolve FFmpegOps from the compiler_service module (late-bound)."""
    from src.services.compilation import compiler_service

    return compiler_service.FFmpegOps


logger = logging.getLogger(__name__)


class _EffectsMixin:
    """Mixin providing color correction, watermark, audio mixing, subtitles."""

    @staticmethod
    def _apply_color_correction(ctx: _JobContext, cc_cfg: dict) -> None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        job_id = ctx.job_id
        tmp_output = output_path.with_suffix(".tmp")
        try:
            from src.services.compilation.video_filters import (
                ColorCorrection,
                apply_color_correction,
            )

            correction = ColorCorrection(**cc_cfg)
            cc_output = work_dir / "demo_cc.mp4"
            apply_color_correction(output_path, cc_output, correction)
            if cc_output.exists():
                shutil.move(str(cc_output), str(tmp_output))
                tmp_output.replace(output_path)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            tmp_output.unlink(missing_ok=True)
            logger.warning("Color correction failed for job %s", job_id, exc_info=True)
            ctx.record_degradation("color_correction", "Colour correction failed", exc)

    @staticmethod
    def _apply_watermark(ctx: _JobContext, wm_cfg: dict) -> None:
        """Overlay the tenant's watermark image onto the finished video.

        LFI sink: ``image_path`` is tenant-controlled and ffmpeg composites
        whatever it names into a video the tenant downloads, so the reference is
        confined to the storage root by ``resolve_local_asset_path`` — the same
        helper and the same boundary as every other local asset ref on the
        compile path. A refusal skips the watermark rather than failing the job,
        matching the existing behaviour when the overlay errors.
        """
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        job_id = ctx.job_id
        wm_path = resolve_local_asset_path(str(wm_cfg.get("image_path") or ""))
        if wm_path is None:
            logger.warning(
                "Watermark image for job %s refused by the local-path guard; skipping overlay",
                job_id,
            )
            ctx.record_degradation("watermark", "Watermark image refused by the local-path guard")
            return
        tmp_output = output_path.with_suffix(".tmp")
        try:
            from src.services.compilation.video_filters import overlay_image

            wm_output = work_dir / "demo_wm.mp4"
            overlay_image(
                output_path,
                wm_path,
                wm_output,
                position=wm_cfg.get("position", "bottom-right"),
                opacity=wm_cfg.get("opacity", 0.7),
                scale=wm_cfg.get("scale", 0.15),
            )
            if wm_output.exists():
                shutil.move(str(wm_output), str(tmp_output))
                tmp_output.replace(output_path)
        except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
            tmp_output.unlink(missing_ok=True)
            logger.warning("Watermark overlay failed for job %s", job_id, exc_info=True)
            ctx.record_degradation("watermark", "Watermark overlay failed", exc)

    @staticmethod
    def _resolve_narration_audio(ctx: _JobContext) -> Path | None:
        if not ctx.narration_audio_dir:
            return None
        has_per_segment = ctx.segments_with_narration > 0
        narr_full = ctx.narration_audio_dir / "narration_full.mp3"
        narration_audio: Path | None = (
            narr_full if narr_full.exists() and not has_per_segment else None
        )
        if narration_audio and narration_audio.exists():
            normed = narration_audio.parent / "narration_normed.mp3"
            try:
                _ffmpeg().loudnorm_audio(narration_audio, normed)
                if normed.exists():
                    narration_audio = normed
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                logger.warning("Narration loudnorm failed, using original audio", exc_info=True)
                ctx.record_degradation(
                    "narration_loudnorm", "Narration loudness normalisation failed", exc
                )
        return narration_audio

    @staticmethod
    def _speed_adjust_to_narration(ctx: _JobContext, narration_audio: Path) -> None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        if not (narration_audio and output_path.exists()):
            return
        narration_dur = _ffmpeg().probe_duration(narration_audio)
        video_dur = _ffmpeg().probe_duration(output_path)
        if narration_dur > 0 and video_dur > narration_dur * 1.05:
            speed_factor = min(video_dur / narration_dur, 2.0)
            sped_path = work_dir / "demo_speed_adjusted.mp4"
            _ffmpeg().speed_adjust(output_path, sped_path, speed_factor)
            if sped_path.exists():
                shutil.move(str(sped_path), str(output_path))

    @staticmethod
    def _mix_global_narration(
        ctx: _JobContext, narration_audio: Path, bg_music_path: str | None
    ) -> None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        with_narration = work_dir / "demo_with_narration.mp4"
        music_for_duck = Path(bg_music_path) if bg_music_path else None
        _ffmpeg().mix_voiceover(
            output_path,
            narration_audio,
            with_narration,
            music_path=music_for_duck if music_for_duck and music_for_duck.exists() else None,
        )
        if with_narration.exists():
            tmp_output = output_path.with_suffix(".tmp")
            try:
                shutil.move(str(with_narration), str(tmp_output))
                tmp_output.replace(output_path)
            except OSError, RuntimeError, ValueError, TimeoutError:
                tmp_output.unlink(missing_ok=True)
                raise

    @staticmethod
    def _mix_background_music(ctx: _JobContext, bg_music_path: str) -> None:
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        settings = ctx.settings
        music_path = Path(bg_music_path)
        if not music_path.exists():
            # Was a bare return with no log at all -- the most invisible drop in
            # the service: background music was configured, never mixed, and
            # nothing anywhere recorded it.
            logger.warning(
                "Background music file for job %s does not exist; skipping mix", ctx.job_id
            )
            ctx.record_degradation("background_music", "Configured background music file not found")
            return
        with_music = work_dir / "demo_with_music.mp4"
        _ffmpeg().mix_background_music(
            output_path,
            music_path,
            with_music,
            volume=settings.background_music.volume,
            fade_in_s=settings.background_music.fade_in_s,
            fade_out_s=settings.background_music.fade_out_s,
        )
        if with_music.exists():
            shutil.move(str(with_music), str(output_path))

    @classmethod
    def _run_audio_effects(
        cls, ctx: _JobContext, narration_audio: Path | None, has_per_segment: bool
    ) -> dict[str, Any]:
        if narration_audio is not None:
            cls._speed_adjust_to_narration(ctx, narration_audio)
        bg_music_path = ctx.settings.background_music.path
        details: dict[str, Any] = {}
        if narration_audio:
            cls._mix_global_narration(ctx, narration_audio, bg_music_path)
            details["narration"] = "global"
            details["segments_with_narration"] = ctx.segments_with_narration
        elif bg_music_path:
            cls._mix_background_music(ctx, bg_music_path)
            if has_per_segment:
                details["narration"] = "per_segment"
                details["segments_with_narration"] = ctx.segments_with_narration
        return details

    @staticmethod
    def _apply_subtitles(ctx: _JobContext, sub_cfg: dict) -> None:
        """Burn or embed the tenant's subtitle file into the finished video.

        The sharpest LFI sink on the compile path: with ``mode='burn'`` ffmpeg
        renders the file's *text* into the frames, so an unconfined
        ``file_path`` would hand the contents of any readable file back to the
        caller as pixels. Confined to the storage root by
        ``resolve_local_asset_path``.

        The refusal message deliberately does not echo the rejected path: the
        existing "file not found" branch below reports the path back through
        ``ctx.subtitle_error``, which would turn a refusal into a path-existence
        oracle over the whole filesystem.
        """
        output_path = ctx.output_path
        work_dir = ctx.work_dir
        job_id = ctx.job_id
        sub_path = resolve_local_asset_path(str(sub_cfg.get("file_path") or ""))
        if sub_path is None:
            ctx.subtitle_error = "Subtitle file refused by the local-path guard"
            logger.error("Subtitle path for job %s refused by the local-path guard", job_id)
            return
        try:
            if sub_path.exists():
                sub_output = work_dir / "demo_sub.mp4"
                # Only an explicit "burn" hardcodes; anything else is a soft track.
                if sub_cfg.get("mode", "embed") != "burn":
                    _ffmpeg().embed_subtitles(output_path, sub_path, sub_output)
                else:
                    _ffmpeg().burn_subtitles(
                        output_path,
                        sub_path,
                        sub_output,
                        font_size=sub_cfg.get("font_size", 24),
                        font_color=sub_cfg.get("font_color", "white"),
                        position=sub_cfg.get("position", "bottom"),
                    )
                if sub_output.exists():
                    sub_output.replace(output_path)
            else:
                ctx.subtitle_error = f"Subtitle file not found: {sub_path}"
                logger.error("Subtitle file missing for job %s: %s", job_id, sub_path)
        except Exception as sub_exc:
            ctx.subtitle_error = format_error(sub_exc)
            logger.error(
                "Subtitle processing failed for job %s: %s",
                job_id,
                ctx.subtitle_error,
                exc_info=True,
            )

    async def _apply_effects(self, ctx: _JobContext, progress_callback: Any | None) -> None:
        """Apply color correction, watermark, narration, music, and subtitles."""
        output_path = ctx.output_path
        filter_cfg = ctx.job.filter_config or {}

        cc_cfg = filter_cfg.get("color_correction")
        if cc_cfg and output_path.exists():
            stage_t = time.time()
            await asyncio.to_thread(self._apply_color_correction, ctx, cc_cfg)
            ctx.record_stage("color_correction", stage_t)

        wm_cfg = filter_cfg.get("watermark")
        if wm_cfg and output_path.exists():
            stage_t = time.time()
            await asyncio.to_thread(self._apply_watermark, ctx, wm_cfg)
            ctx.record_stage("watermark", stage_t)

        has_per_segment = ctx.segments_with_narration > 0
        narration_audio = self._resolve_narration_audio(ctx)

        stage_t = time.time()
        audio_mixing_details = await asyncio.to_thread(
            self._run_audio_effects, ctx, narration_audio, has_per_segment
        )
        ctx.record_stage("mix_audio", stage_t)

        if progress_callback:
            await progress_callback.report("mixing_audio", 92.0, "Mixing audio")

        # Subtitles
        sub_cfg = ctx.job.subtitles
        if sub_cfg and output_path.exists():
            stage_t = time.time()
            await asyncio.to_thread(self._apply_subtitles, ctx, sub_cfg)
            ctx.record_stage("subtitles", stage_t)

        # Store audio details for finalize
        ctx.audio_mixing_details = audio_mixing_details
