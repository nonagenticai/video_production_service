"""Per-segment building stage helpers (mixin)."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from w2v_common.net import safe_get, safe_get_sync

from src.services.compilation._compiler_context import _JobContext
from src.utils.url_guard import (
    ARTIFACT_IMAGE_MAX_BYTES,
    ARTIFACT_RECORDING_MAX_BYTES,
    artifact_allow_hosts,
    is_local_file_ref,
    resolve_local_asset_path,
)

if TYPE_CHECKING:
    from src.services.ffmpeg import FFmpegOps as _FFmpegOpsType
    from src.services.storage_service import StorageService


def _ffmpeg() -> type[_FFmpegOpsType]:
    """Resolve FFmpegOps from the compiler_service module.

    Late-bound so that ``mock.patch("...compiler_service.FFmpegOps")`` in tests
    redirects the helper modules' lookups too.
    """
    from src.services.compilation import compiler_service

    return compiler_service.FFmpegOps


logger = logging.getLogger(__name__)


class _SegmentBuildersMixin:
    """Mixin providing per-segment build/edit/normalize logic."""

    _storage: StorageService

    async def _process_segments(self, ctx: _JobContext, progress_callback: Any | None) -> None:
        """Convert, edit, normalize, and narrate each part into segments."""
        stage_t = time.time()

        for idx, part_id in enumerate(ctx.recording_order):
            part = ctx.parts_by_id.get(part_id)
            if not part:
                ctx.segments_excluded += 1
                continue

            part_type = part.get("part_type", "browser_recording")

            if part_type == "title_card":
                # Opt-in animated intro/outro card — per-part ``motion`` flag OR a
                # job-level ``title_card_config.motion`` switch. Soft-falls back to
                # the static Pillow card. Slide-image title cards keep their path.
                if self._title_card_motion_on(ctx, part) and not (
                    part.get("slide_image_url") or part.get("image_url")
                ):
                    await self._process_title_card_motion(ctx, part_id, part)
                else:
                    await asyncio.to_thread(self._process_title_card, ctx, part_id, part)
            elif part_type in {"slide_preview", "generated_image", "comparison_frame"}:
                await asyncio.to_thread(self._process_visual_asset_part, ctx, part_id, part)
            elif part_id in ctx.recording_map:
                await self._process_recording(ctx, part_id, part)
            elif part_type == "ai_generated" and ctx.clips_dir:
                await asyncio.to_thread(self._process_ai_clip, ctx, part_id, part)
            else:
                logger.warning("No recording for part %s, skipping", part_id)
                ctx.segments_excluded += 1

            # The per-part tick. This loop is where a compile spends most of its
            # wall time, so a stage-granularity signal alone would look SILENT for
            # the whole of it and the reaper would fail a perfectly healthy job.
            pct = 10.0 + (70.0 * (idx + 1) / len(ctx.recording_order))
            ctx.mark_progress(stage="processing", percent=pct)

            if progress_callback:
                await progress_callback.report(
                    "processing",
                    pct,
                    f"Processing part {idx + 1}/{len(ctx.recording_order)}",
                    current_part=idx + 1,
                    total_parts=len(ctx.recording_order),
                )

        ctx.record_stage("convert_normalize", stage_t, segments=len(ctx.segments))

    def _segment_with_optional_narration(
        self,
        ctx: _JobContext,
        part_id: str,
        input_path: Path,
    ) -> Path:
        norm_path = ctx.normalized_dir / f"{part_id}.mp4"
        _ffmpeg().normalize_stream(
            input_path,
            norm_path,
            ctx.target_w,
            ctx.target_h,
            fps=ctx.target_fps,
            crf=ctx.crf,
            cfr=ctx.cfr_mode,
        )

        if ctx.narration_audio_dir:
            full_narr = ctx.narration_audio_dir / "narration_full.mp3"
            if not full_narr.exists():
                narr_audio = ctx.narration_audio_dir / f"{part_id}.mp3"
                if narr_audio.exists():
                    mixed = self._mix_per_segment_narration(
                        ctx, norm_path, narr_audio, ctx.normalized_dir, part_id
                    )
                    if mixed != norm_path:
                        ctx.segments_with_narration += 1
                    norm_path = mixed

        return norm_path

    @staticmethod
    def _guarded_local_image_path(part_id: str, ref: str) -> Path | None:
        """Resolve a ``file://`` / bare-path image ref through the LFI guard.

        ``ref`` is tenant-controlled and whatever it names is read off the pod
        and composited into a **downloadable** video, so an unguarded
        ``Path(ref)`` turns a compile request into an arbitrary file read.
        ``resolve_local_asset_path`` confines the reference to the configured
        storage root unless ``compilation.allow_local_file_inputs`` is on.

        Returns:
            The path when it is both permitted and present, else ``None`` —
            which the caller already treats as "asset unavailable" and skips.
        """
        local_path = resolve_local_asset_path(ref)
        if local_path is None:
            logger.warning("Local image ref for %s refused by the LFI guard: %s", part_id, ref)
            return None
        return local_path if local_path.exists() else None

    def _materialize_image_ref(
        self,
        ctx: _JobContext,
        part_id: str,
        image_ref: str | None,
        *,
        suffix_hint: str = ".png",
    ) -> Path | None:
        """Resolve an image ref (data:, http(s):, s3:, or local path) to a local file.

        Deliberately fail-soft: one unreachable image must not abort a whole
        compilation, so every failure mode returns ``None`` and the caller drops
        that part instead of raising.

        Returns:
            Path to the materialized local file, or ``None`` if the ref could not
            be fetched, decoded, or resolved.
        """
        ref = str(image_ref or "").strip()
        if not ref:
            return None

        cached = ctx.asset_cache.get(ref)
        if cached and cached.exists():
            return cached

        def _cache(path: Path) -> Path:
            ctx.asset_cache[ref] = path
            return path

        try:
            if ref.startswith("data:image/"):
                header, payload = ref.split(",", 1)
                media_type = header.split(";", 1)[0].replace("data:", "", 1)
                ext = mimetypes.guess_extension(media_type) or suffix_hint
                local_path = ctx.title_cards_dir / f"{part_id}_asset{ext}"
                local_path.write_bytes(base64.b64decode(payload))
                return _cache(local_path)

            if ref.startswith(("http://", "https://")):
                suffix = Path(ref.split("?", 1)[0]).suffix or suffix_hint
                local_path = ctx.title_cards_dir / f"{part_id}_asset{suffix}"
                # SSRF sink: ``ref`` is tenant-controlled and this fetch used to
                # follow redirects blindly, so a schema-approved public host
                # could 302 into the tailnet. ``safe_get_sync`` validates the
                # initial URL AND every redirect hop, and trusts only the
                # configured MinIO artifact hosts.
                # ``max_fetch_bytes`` is explicit because the ``except`` below
                # catches ``ValueError`` and ``ResponseTooLargeError`` is one:
                # inheriting w2v_common's 64 MiB default would turn an oversized
                # image into a silently skipped part, not an error.
                with httpx.Client(timeout=60.0) as client:
                    resp = safe_get_sync(
                        client,
                        ref,
                        allow_hosts=artifact_allow_hosts(),
                        max_fetch_bytes=ARTIFACT_IMAGE_MAX_BYTES,
                    )
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
                return _cache(local_path)

            if ref.startswith("s3://"):
                # Bucket-allowlist sink: ``ref`` is tenant-controlled and carries
                # an arbitrary bucket, so an unguarded read would serve any
                # bucket the service account can reach. ``tenant_id`` is passed
                # for the audit log; see StorageService._parse_and_authorize_uri
                # for why confinement stops at the bucket.
                data = self._storage.retrieve_uri_sync(ref, tenant_id=ctx.tenant_id)
                if data is None:
                    return None
                suffix = Path(ref).suffix or suffix_hint
                local_path = ctx.title_cards_dir / f"{part_id}_asset{suffix}"
                local_path.write_bytes(data)
                return _cache(local_path)

            if (local_path := self._guarded_local_image_path(part_id, ref)) is not None:
                return _cache(local_path)
        # ``httpx.HTTPError`` is the load-bearing entry: ``resp.raise_for_status()``
        # raises ``httpx.HTTPStatusError``, and a connect/read failure raises
        # ``httpx.RequestError`` — neither derives from ``OSError``, so without the
        # httpx base a single 404 on one image ref escaped this handler and
        # hard-failed the entire compile instead of skipping the part. The two
        # sibling handlers in this file (the background-image download and
        # ``_resolve_recording_video_path``) already list it; this one did not.
        # The builtins stay: a socket-level failure under Pillow, base64 decode, or
        # the storage backend still surfaces as OSError/ValueError/TimeoutError.
        except (OSError, RuntimeError, ValueError, TimeoutError, httpx.HTTPError) as exc:
            logger.warning("Failed to materialize image ref for %s from %s: %s", part_id, ref, exc)
            return None

        logger.warning("Image ref for %s is not accessible: %s", part_id, ref)
        return None

    def _process_visual_asset_part(self, ctx: _JobContext, part_id: str, part: dict) -> None:
        approval_status = str(part.get("approval_status") or "").strip().lower()
        if approval_status in {"rejected", "blocked", "failed"}:
            logger.info(
                "Skipping visual asset segment %s due to approval_status=%s",
                part_id,
                approval_status,
            )
            ctx.segments_excluded += 1
            return

        image_ref = (
            str(part.get("slide_image_url") or "").strip()
            or str(part.get("image_url") or "").strip()
            or str(part.get("background_image") or "").strip()
        )
        image_path = self._materialize_image_ref(ctx, part_id, image_ref)
        if image_path is None:
            logger.warning("No image available for visual asset part %s, skipping", part_id)
            ctx.segments_excluded += 1
            return

        segment_path = ctx.title_cards_dir / f"{part_id}.mp4"
        duration = max(float(part.get("estimated_duration_seconds") or 5.0), 1.0)
        motion_preset = str(part.get("motion_preset") or "").strip()

        if motion_preset:
            _ffmpeg().image_to_motion_video(
                image_path=image_path,
                output_path=segment_path,
                duration=duration,
                width=ctx.target_w,
                height=ctx.target_h,
                motion_preset=motion_preset,
            )
        else:
            _ffmpeg().image_to_video(
                image_path=image_path,
                output_path=segment_path,
                duration=duration,
                width=ctx.target_w,
                height=ctx.target_h,
            )

        ctx.segments.append(self._segment_with_optional_narration(ctx, part_id, segment_path))

    def _process_title_card(self, ctx: _JobContext, part_id: str, part: dict) -> None:
        if part.get("slide_image_url") or part.get("image_url"):
            self._process_visual_asset_part(ctx, part_id, part)
            return

        tc_path = ctx.title_cards_dir / f"{part_id}.mp4"

        # Download remote background images so Pillow can open them.
        bg_image = part.get("background_image")
        if bg_image and bg_image.startswith(("http://", "https://")):
            local_bg = ctx.title_cards_dir / f"{part_id}_bg.jpg"
            try:
                # This used ``urllib.request.urlretrieve``, which has no SSRF
                # guard and follows redirects on its own — the "URL validated as
                # http/https above" comment it carried only ever checked the
                # scheme. Same sink, same treatment as every other asset fetch:
                # ``safe_get_sync`` validates the target, pins the resolved IP
                # and re-validates each hop.
                # Explicit ceiling, same reason as the image-ref sink above: the
                # handler below swallows ``ValueError``, so an oversized
                # background would quietly become no background at all.
                with httpx.Client(timeout=60.0) as client:
                    resp = safe_get_sync(
                        client,
                        bg_image,
                        allow_hosts=artifact_allow_hosts(),
                        max_fetch_bytes=ARTIFACT_IMAGE_MAX_BYTES,
                    )
                    resp.raise_for_status()
                    local_bg.write_bytes(resp.content)
                bg_image = str(local_bg)
            except (OSError, RuntimeError, ValueError, TimeoutError, httpx.HTTPError) as exc:
                logger.warning("Failed to download background image %s", bg_image)
                bg_image = None
                ctx.record_degradation(
                    "background_image", "Background image could not be downloaded", exc
                )
        elif bg_image and is_local_file_ref(bg_image):
            # A local background image is the same LFI sink as any other asset
            # ref; Pillow will happily open whatever path it is handed. ``s3://``
            # and ``data:`` values are left alone — they name no local file.
            resolved_bg = resolve_local_asset_path(bg_image)
            if resolved_bg is None:
                logger.warning(
                    "Local background image for %s refused by the LFI guard: %s", part_id, bg_image
                )
                ctx.record_degradation(
                    "background_image", "Background image refused by the local-path guard"
                )
            bg_image = str(resolved_bg) if resolved_bg is not None else None

        logo_ref = str(part.get("logo_path") or "").strip()
        logo_path: str | None = logo_ref or None
        if logo_ref and is_local_file_ref(logo_ref):
            # Identical sink to ``background_image`` one branch up — both end at
            # ``Image.open`` inside ``TitleCardGenerator`` and both get composited
            # into a downloadable video — so it gets the identical guard rather
            # than a second mechanism. ``logo_path`` has no download branch: the
            # generator only ever opens it locally, so an http(s) value is simply
            # left for Pillow to reject as it does today.
            resolved_logo = resolve_local_asset_path(logo_ref)
            if resolved_logo is None:
                logger.warning("Local logo for %s refused by the LFI guard: %s", part_id, logo_ref)
                ctx.record_degradation("logo", "Logo image refused by the local-path guard")
            logo_path = str(resolved_logo) if resolved_logo is not None else None

        ctx.title_gen.generate_video(
            title=part.get("title", ""),
            subtitle=part.get("subtitle", ""),
            chapter=part.get("chapter"),
            output_path=tc_path,
            duration=part.get("estimated_duration_seconds", 5.0),
            width=ctx.target_w,
            height=ctx.target_h,
            background_image=bg_image,
            logo_path=logo_path,
            logo_position=part.get("logo_position"),
            logo_scale=part.get("logo_scale"),
            animation=part.get("animation"),
            bullet_points=part.get("bullet_points"),
        )
        ctx.segments.append(self._segment_with_optional_narration(ctx, part_id, tc_path))

    @staticmethod
    def _title_card_motion_on(ctx: _JobContext, part: dict) -> bool:
        """Whether to render this title card via the motion template.

        Opt-in: a per-part ``motion`` flag, or a job-level
        ``title_card_config.motion`` switch (so any compile caller can turn it on
        without stamping every part).
        """
        if part.get("motion"):
            return True
        job_cfg = getattr(ctx.job, "title_card_config", None) or {}
        return bool(job_cfg.get("motion"))

    def _motion_title_params(self, ctx: _JobContext, part: dict) -> dict[str, Any]:
        """Map a title-card part + resolved theme onto animated_title_card PARAMS."""
        params: dict[str, Any] = {
            "title": part.get("title", ""),
            "subtitle": part.get("subtitle", ""),
        }
        if (eyebrow := part.get("eyebrow")) is not None:
            params["eyebrow"] = str(eyebrow)
        if (chapter := part.get("chapter")) is not None:
            params["chapter"] = f"Chapter {chapter}" if isinstance(chapter, int) else str(chapter)
        # Brand the template with the job's resolved theme colors (only truthy keys —
        # the template falls back to its on-brand defaults for anything omitted).
        theme = ctx.theme_config or {}
        for tmpl_key, theme_keys in (
            ("bg", ("gradient_start", "background_color")),
            ("bg2", ("gradient_end",)),
            ("accent", ("accent_color", "primary_color")),
            ("text", ("text_color",)),
            ("muted", ("subtitle_color",)),
        ):
            for tk in theme_keys:
                if theme.get(tk):
                    params[tmpl_key] = theme[tk]
                    break
        return params

    async def _process_title_card_motion(self, ctx: _JobContext, part_id: str, part: dict) -> None:
        """Render the intro/outro card via the animated_title_card motion template.

        Soft-fails to the static Pillow card so a bad clip never fails the compile.
        """
        from playwright.async_api import Error as PlaywrightError

        from src.services.recording.animation_recorder import AnimationRecorder
        from src.services.recording.templates import get_template

        try:
            html = get_template("animated_title_card")
            duration_ms = int(float(part.get("estimated_duration_seconds", 5.0)) * 1000)
            result = await AnimationRecorder().record_animation(
                job_id=f"titlecard-{ctx.job_id}-{part_id}",
                html_content=html,
                animation_config={
                    "params": self._motion_title_params(ctx, part),
                    "duration_ms": duration_ms,
                    "output_format": "mp4",
                },
                recording_config={
                    "viewport_width": ctx.target_w,
                    "viewport_height": ctx.target_h,
                },
            )
            if not result.get("success"):
                raise RuntimeError(result.get("error") or "animation render failed")
            tc_path = ctx.title_cards_dir / f"{part_id}.mp4"
            src_path = Path(result["video_path"])
            if src_path != tc_path:
                shutil.move(str(src_path), str(tc_path))
            ctx.segments.append(self._segment_with_optional_narration(ctx, part_id, tc_path))
        except (OSError, RuntimeError, ValueError, TimeoutError, PlaywrightError) as exc:
            logger.warning(
                "Motion title card failed for %s; using static Pillow card: %s", part_id, exc
            )
            ctx.record_degradation(
                "motion_title_card", "Animated title card fell back to a static card", exc
            )
            await asyncio.to_thread(self._process_title_card, ctx, part_id, part)

    async def _resolve_recording_video_path(
        self, ctx: _JobContext, part_id: str, video_path_str: str
    ) -> Path | None:
        """Resolve recording video into a local Path. Returns None if unavailable.

        The http(s) branch is an SSRF sink on a tenant-controlled ``video_path``,
        so it goes through ``safe_get``, which validates the target and every
        redirect hop. A blocked target is treated exactly like any other failed
        download — logged and skipped — because this resolver is deliberately
        fail-soft: one bad part must not abort a whole compilation. ``ValueError``
        is therefore in the except tuple — ``SSRFError`` subclasses it, as does
        the malformed-URL error — because without it a guard rejection would
        escape as a hard error and change that contract.

        That same fail-soft contract is why ``max_fetch_bytes`` is stated here
        rather than inherited. ``ResponseTooLargeError`` is also a
        ``ValueError``, so w2v_common 0.55.0's 64 MiB default would land in the
        handler below and turn any recording longer than about a minute of
        1080p into a silently skipped part. This is the one genuinely unbounded
        source in the service, so it gets
        :data:`~src.utils.url_guard.ARTIFACT_RECORDING_MAX_BYTES` (1 GiB) — see
        that constant for why the ceiling sits there and not higher.
        """
        if video_path_str.startswith(("http://", "https://")):
            local_path = ctx.edited_dir / f"{part_id}_src.mp4"
            try:
                async with httpx.AsyncClient(timeout=120.0) as http_client:
                    resp = await safe_get(
                        http_client,
                        video_path_str,
                        allow_hosts=artifact_allow_hosts(),
                        max_fetch_bytes=ARTIFACT_RECORDING_MAX_BYTES,
                    )
                    resp.raise_for_status()
                    local_path.write_bytes(resp.content)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                logger.warning(
                    "Could not download video for part %s from %s: %s, skipping",
                    part_id,
                    video_path_str,
                    exc,
                )
                return None
            return local_path
        if video_path_str.startswith("s3://"):
            # Bucket-allowlist sink — same rationale as the image-ref branch above.
            data = await self._storage.retrieve_uri(video_path_str, tenant_id=ctx.tenant_id)
            if data is None:
                logger.warning(
                    "Could not download video for part %s from %s, skipping",
                    part_id,
                    video_path_str,
                )
                return None
            local_path = ctx.edited_dir / f"{part_id}_src.mp4"
            local_path.write_bytes(data)
            return local_path
        # LFI sink. Unlike the image ref, a bare local path here is a LEGITIMATE
        # production value: ``GET /api/record/jobs/{id}/result`` publishes
        # ``{STORAGE_LOCAL_PATH}/{job_id}/{part_id}.mp4`` and re-submitting it on
        # ``recordings[].video_path`` is the documented record -> compile round
        # trip (prod runs the local backend with /app/data on a PVC shared by
        # both replicas). So this is confinement, not denial:
        # ``resolve_local_asset_path`` accepts anything under the storage root
        # and refuses ``/etc/passwd``, ``/app/.env`` and friends.
        local_video = resolve_local_asset_path(video_path_str)
        if local_video is None:
            logger.warning(
                "Video path for part %s refused by the LFI guard (outside the storage root): %s",
                part_id,
                video_path_str,
            )
            return None
        if not local_video.exists():
            logger.warning("Video file missing for part %s, skipping", part_id)
            return None
        return local_video

    async def _process_recording(self, ctx: _JobContext, part_id: str, part: dict) -> None:
        rec = ctx.recording_map[part_id]
        video_path_str = rec["video_path"]
        video_path = await self._resolve_recording_video_path(ctx, part_id, video_path_str)
        if video_path is None:
            ctx.segments_excluded += 1
            return

        def _ffmpeg_work() -> Path:
            mp4_path = ctx.edited_dir / f"{part_id}.mp4"
            _ffmpeg().convert_to_mp4(video_path, mp4_path, ctx.crf)

            cur = mp4_path
            if part_id in ctx.edit_decisions:
                cur = self._apply_edit_decisions(ctx, part_id, cur)

            trimmed = ctx.editor.auto_trim(cur)
            if trimmed != cur and trimmed.exists():
                cur = trimmed

            norm_path = ctx.normalized_dir / f"{part_id}.mp4"
            _ffmpeg().normalize_stream(
                cur,
                norm_path,
                ctx.target_w,
                ctx.target_h,
                fps=ctx.target_fps,
                crf=ctx.crf,
                cfr=ctx.cfr_mode,
            )

            if ctx.narration_audio_dir:
                full_narr = ctx.narration_audio_dir / "narration_full.mp3"
                if not full_narr.exists():
                    narr_audio = ctx.narration_audio_dir / f"{part_id}.mp3"
                    if narr_audio.exists():
                        mixed = self._mix_per_segment_narration(
                            ctx, norm_path, narr_audio, ctx.normalized_dir, part_id
                        )
                        if mixed != norm_path:
                            ctx.segments_with_narration += 1
                        norm_path = mixed
                    else:
                        logger.warning(
                            "Job %s: narration file not found: %s", ctx.job_id, narr_audio
                        )

            return norm_path

        result_path = await asyncio.to_thread(_ffmpeg_work)
        ctx.segments.append(result_path)

    def _apply_edit_decisions(self, ctx: _JobContext, part_id: str, mp4_path: Path) -> Path:
        decision = ctx.edit_decisions[part_id]
        if decision.get("speed_factor", 1.0) != 1.0:
            speed_result = ctx.editor.speed_adjust(mp4_path, decision["speed_factor"])
            if speed_result.exists():
                mp4_path = speed_result
        if decision.get("trim_start_s", 0) > 0 or decision.get("trim_end_s", 0) > 0:
            duration = _ffmpeg().probe_duration(mp4_path)
            start_time = decision.get("trim_start_s", 0)
            end_time = duration - decision.get("trim_end_s", 0)
            if end_time <= start_time:
                logger.warning(
                    "Trim bounds invalid for part %s: start=%.2f end=%.2f duration=%.2f, skipping",
                    part_id,
                    start_time,
                    end_time,
                    duration,
                )
            else:
                trim_path = ctx.edited_dir / f"{part_id}_trim.mp4"
                _ffmpeg().trim(mp4_path, trim_path, start_time, end_time)
                if trim_path.exists():
                    mp4_path = trim_path
        if decision.get("remove_dead_time"):
            dead_trimmed = ctx.editor.remove_dead_time(
                mp4_path, fps=ctx.target_fps, cfr=ctx.cfr_mode
            )
            if dead_trimmed != mp4_path and dead_trimmed.exists():
                mp4_path = dead_trimmed
        return mp4_path

    def _process_ai_clip(self, ctx: _JobContext, part_id: str, part: dict) -> None:
        if ctx.clips_dir is None:
            logger.warning("No clips_dir set for part %s, skipping", part_id)
            ctx.segments_excluded += 1
            return
        ai_clip = ctx.clips_dir / f"{part_id}.mp4"
        if not ai_clip.exists():
            logger.warning("AI clip file missing for part %s, skipping gracefully", part_id)
            ctx.segments_excluded += 1
            return

        processed_clip = ai_clip
        if part.get("strip_ai_audio", True):
            stripped = ctx.edited_dir / f"{part_id}_noaudio.mp4"
            try:
                _ffmpeg().strip_audio(ai_clip, stripped)
                if stripped.exists():
                    processed_clip = stripped
            except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
                logger.warning("Failed to strip audio from AI clip %s", part_id)
                # The AI clip's own audio stays in the deliverable.
                ctx.record_degradation("ai_audio_strip", "Could not strip AI clip audio", exc)

        norm_path = ctx.normalized_dir / f"{part_id}.mp4"
        _ffmpeg().normalize_stream(
            processed_clip,
            norm_path,
            ctx.target_w,
            ctx.target_h,
            fps=ctx.target_fps,
            crf=ctx.crf,
            cfr=ctx.cfr_mode,
        )

        if ctx.narration_audio_dir:
            full_narr = ctx.narration_audio_dir / "narration_full.mp3"
            if not full_narr.exists():
                narr_audio = ctx.narration_audio_dir / f"{part_id}.mp3"
                if narr_audio.exists():
                    mixed = self._mix_per_segment_narration(
                        ctx, norm_path, narr_audio, ctx.normalized_dir, part_id
                    )
                    if mixed != norm_path:
                        ctx.segments_with_narration += 1
                    norm_path = mixed

        ctx.segments.append(norm_path)

    def _mix_per_segment_narration(
        self, ctx: _JobContext, video_path: Path, audio_path: Path, work_dir: Path, part_id: str
    ) -> Path:
        """Mix narration audio into a single video segment, speed-adjusting if needed."""
        try:
            audio_dur = _ffmpeg().probe_duration(audio_path)
            video_dur = _ffmpeg().probe_duration(video_path)
            if audio_dur <= 0 or video_dur <= 0:
                return video_path

            result_path = video_path
            if video_dur > audio_dur * 1.05:
                speed_factor = min(video_dur / audio_dur, 2.0)
                sped_path = work_dir / f"{part_id}_sped.mp4"
                _ffmpeg().speed_adjust(video_path, sped_path, speed_factor)
                if sped_path.exists():
                    result_path = sped_path

            mixed_path = work_dir / f"{part_id}_narrated.mp4"
            _ffmpeg().mix_voiceover(result_path, audio_path, mixed_path)
            if mixed_path.exists():
                # Restore the original part_id stem so downstream lookups
                # (_compute_chapter_timestamps, _assemble_smart) can match by
                # seg.stem == part_id.
                final_path = work_dir / f"{part_id}.mp4"
                if final_path != mixed_path:
                    mixed_path.replace(final_path)
                return final_path
            return result_path
        except (OSError, RuntimeError) as exc:
            logger.error("Per-segment narration mixing failed for %s", part_id, exc_info=True)
            # Returns the un-narrated segment; the deliverable is missing the
            # narration for this part and nothing else records that.
            ctx.record_degradation("segment_narration", "Per-segment narration mixing failed", exc)
            return video_path
