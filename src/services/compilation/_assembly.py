"""Video assembly stage helpers (mixin)."""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.services.compilation._compiler_context import _JobContext

if TYPE_CHECKING:
    from src.services.ffmpeg import FFmpegOps as _FFmpegOpsType


def _ffmpeg() -> type[_FFmpegOpsType]:
    """Resolve FFmpegOps from the compiler_service module (late-bound)."""
    from src.services.compilation import compiler_service

    return compiler_service.FFmpegOps


logger = logging.getLogger(__name__)


class _AssemblyMixin:
    """Mixin providing video assembly + chapter computation logic."""

    def _assemble_video(self, ctx: _JobContext) -> None:
        """Assemble normalized segments into a single video with transitions."""
        stage_t = time.time()
        ctx.output_path = ctx.work_dir / ctx.settings.compilation.output_filename
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)

        filter_cfg_pre = ctx.job.filter_config or {}
        xfade_transition = filter_cfg_pre.get("xfade_transition", "fade")

        if len(ctx.segments) <= 1:
            if ctx.segments:
                tmp_output = ctx.output_path.with_suffix(".tmp")
                try:
                    shutil.copy2(ctx.segments[0], tmp_output)
                    tmp_output.replace(ctx.output_path)
                except OSError, RuntimeError, ValueError, TimeoutError:
                    tmp_output.unlink(missing_ok=True)
                    raise
        elif ctx.transition_type == "cut":
            concat_file = ctx.edited_dir / "concat_all.txt"
            with concat_file.open("w") as f:
                for seg in ctx.segments:
                    f.write(f"file '{seg.resolve()}'\n")
            _ffmpeg().concat(concat_file, ctx.output_path, fps=ctx.target_fps, cfr=ctx.cfr_mode)
        elif ctx.transition_type == "fade":
            # ``transitions_used`` is what the caller reports in
            # ``compilation_report``. It used to be a hard-coded
            # ``len(segments) - 1``, so a run where every xfade raised and the
            # assembler silently fell back to plain concat still claimed a full
            # set of transitions the file does not contain.
            ctx.transitions_used = self._assemble_with_xfade(
                ctx.segments,
                ctx.output_path,
                ctx.edited_dir,
                ctx.crossfade_dur,
                xfade_transition,
                fps=ctx.target_fps,
                cfr=ctx.cfr_mode,
            )
        else:
            ctx.transitions_used = self._assemble_smart(
                ctx,
                ctx.segments,
                ctx.parts_by_id,
                ctx.recording_order,
                ctx.output_path,
                ctx.edited_dir,
                ctx.crossfade_dur,
                xfade_transition=xfade_transition,
                fps=ctx.target_fps,
                cfr=ctx.cfr_mode,
            )

        # The xfade assemblers fall back to a hard concat when a step fails and
        # RETURN the count actually applied -- which is why ``transitions_used``
        # is trustworthy. But a caller cannot tell "8 because 8 were wanted"
        # from "8 because the 9th died and the tail was concatenated": that
        # needs the REQUESTED count. In ``fade`` mode that is exactly one per
        # segment gap and is knowable here, so the check lives at the call site.
        # ``smart`` crossfades between CHAPTERS, not segments, so its requested
        # count is not knowable here and it records from inside
        # ``_progressive_xfade_merge`` instead -- using segments there would
        # false-positive on every correctly-grouped smart compile.
        if (
            ctx.transition_type == "fade"
            and len(ctx.segments) > 1
            and ctx.transitions_used < len(ctx.segments) - 1
        ):
            ctx.record_degradation(
                "transitions",
                "Some crossfade transitions failed and fell back to hard cuts",
            )

        ctx.record_stage("assemble", stage_t, transition_type=ctx.transition_type)

    @staticmethod
    def _crossfade_offset_for_segment(
        idx: int,
        seg: Path,
        segments: list[Path],
        parts_by_id: dict,
        transition_type: str,
        crossfade_dur: float,
    ) -> float:
        if idx == 0 or crossfade_dur <= 0:
            return 0.0
        if transition_type == "fade":
            return crossfade_dur
        if transition_type == "smart":
            prev_part = parts_by_id.get(segments[idx - 1].stem)
            cur_part = parts_by_id.get(seg.stem)
            if (
                prev_part
                and cur_part
                and (
                    prev_part.get("chapter") != cur_part.get("chapter")
                    or prev_part.get("part_type") == "title_card"
                )
            ):
                return crossfade_dur
        return 0.0

    def _compute_chapter_timestamps(
        self,
        segments: list[Path],
        parts_by_id: dict,
        transition_type: str = "cut",
        crossfade_dur: float = 0.0,
    ) -> list[dict]:
        """Compute chapter timestamp metadata from assembled segments."""
        chapter_data: dict[int, dict] = {}
        cumulative = 0.0
        for idx, seg in enumerate(segments):
            cumulative -= self._crossfade_offset_for_segment(
                idx, seg, segments, parts_by_id, transition_type, crossfade_dur
            )
            dur = _ffmpeg().probe_duration(seg) if seg.exists() else 0.0
            part = parts_by_id.get(seg.stem)
            if part and part.get("chapter") is not None:
                ch_num = part["chapter"]
                if not isinstance(ch_num, int):
                    try:
                        ch_num = int(ch_num)
                    except TypeError, ValueError:
                        cumulative += dur
                        continue
                if ch_num not in chapter_data:
                    chapter_data[ch_num] = {
                        "title": part.get("title", ""),
                        "subtitle": part.get("subtitle", ""),
                        "start": cumulative,
                        "duration": 0.0,
                    }
                chapter_data[ch_num]["duration"] += dur
            cumulative += dur

        return [
            {
                "chapter_number": ch_num,
                "title": info["title"],
                "subtitle": info["subtitle"],
                "start_seconds": round(info["start"], 3),
                "duration_seconds": round(info["duration"], 3),
            }
            for ch_num, info in sorted(chapter_data.items())
        ]

    def _assemble_smart(
        self,
        ctx: _JobContext,
        segments: list[Path],
        parts_by_id: dict,
        recording_order: list[str],
        output_path: Path,
        work_dir: Path,
        crossfade_dur: float,
        xfade_transition: str = "fade",
        fps: int = 30,
        cfr: bool = False,
    ) -> int:
        """Assemble segments: concat within chapters, xfade between chapters.

        Returns the number of xfade transitions actually applied (0 when
        everything was concatenated).

        Two-phase approach to avoid OOM from progressive xfade on long
        accumulated videos:
          Phase 1 -- Group consecutive same-chapter segments and concat each
                    group (cheap, single re-encode per group).
          Phase 2 -- Progressive xfade merge between the short chapter-level
                    videos.  Falls back to concat if any xfade step is killed
                    (SIGKILL / OOM).
        """
        segment_part_ids = [seg.stem for seg in segments]
        groups = self._group_segments_by_chapter(segments, segment_part_ids, parts_by_id)
        chapter_videos = self._concat_chapter_groups(segments, groups, work_dir, fps=fps, cfr=cfr)

        if len(chapter_videos) <= 1:
            if chapter_videos:
                shutil.copy2(chapter_videos[0], output_path)
            return 0

        return self._progressive_xfade_merge(
            ctx,
            chapter_videos,
            output_path,
            work_dir,
            crossfade_dur,
            xfade_transition,
            fps=fps,
            cfr=cfr,
        )

    @staticmethod
    def _group_segments_by_chapter(
        segments: list[Path], segment_part_ids: list[str], parts_by_id: dict
    ) -> list[list[int]]:
        groups: list[list[int]] = []
        current_group: list[int] = [0]
        for i in range(1, len(segments)):
            prev_part = parts_by_id.get(segment_part_ids[i - 1])
            curr_part = parts_by_id.get(segment_part_ids[i])
            same_chapter = (
                prev_part
                and curr_part
                and prev_part.get("chapter") == curr_part.get("chapter")
                and curr_part.get("part_type") != "title_card"
            )
            if same_chapter:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        groups.append(current_group)
        return groups

    @staticmethod
    def _concat_chapter_groups(
        segments: list[Path],
        groups: list[list[int]],
        work_dir: Path,
        fps: int = 30,
        cfr: bool = False,
    ) -> list[Path]:
        chapter_videos: list[Path] = []
        for gi, group in enumerate(groups):
            if len(group) == 1:
                chapter_videos.append(segments[group[0]])
            else:
                concat_tmp = work_dir / f"chapter_concat_{gi}.txt"
                with concat_tmp.open("w") as f:
                    for idx in group:
                        f.write(f"file '{segments[idx].resolve()}'\n")
                chapter_out = work_dir / f"chapter_{gi}.mp4"
                _ffmpeg().concat(concat_tmp, chapter_out, fps=fps, cfr=cfr)
                chapter_videos.append(chapter_out)
        return chapter_videos

    @staticmethod
    def _progressive_xfade_merge(
        ctx: _JobContext,
        chapter_videos: list[Path],
        output_path: Path,
        work_dir: Path,
        crossfade_dur: float,
        xfade_transition: str,
        fps: int = 30,
        cfr: bool = False,
    ) -> int:
        """Return the number of xfades applied before any concat fallback."""
        current = chapter_videos[0]
        for i in range(1, len(chapter_videos)):
            temp_out = work_dir / f"merged_{i}.mp4"
            try:
                _ffmpeg().crossfade(
                    current,
                    chapter_videos[i],
                    temp_out,
                    fade_duration=crossfade_dur,
                    transition_type=xfade_transition,
                    fps=fps,
                    cfr=cfr,
                )
            except (OSError, RuntimeError) as exc:
                logger.warning(
                    "xfade merge failed at step %d, falling back to concat: %s",
                    i,
                    exc,
                )
                ctx.record_degradation(
                    "transitions",
                    "Some crossfade transitions failed and fell back to hard cuts",
                    exc,
                )
                concat_tmp = work_dir / "fallback_concat.txt"
                remaining = [current, *chapter_videos[i:]]
                with concat_tmp.open("w") as f:
                    for seg in remaining:
                        f.write(f"file '{seg.resolve()}'\n")
                _ffmpeg().concat(concat_tmp, output_path, fps=fps, cfr=cfr)
                return i - 1
            if current != chapter_videos[0] and current.exists():
                with contextlib.suppress(OSError):
                    current.unlink()
            current = temp_out

        shutil.copy2(current, output_path)
        return len(chapter_videos) - 1

    def _assemble_with_fade(
        self,
        segments: list[Path],
        output_path: Path,
        work_dir: Path,
        crossfade_dur: float,
        fps: int = 30,
        cfr: bool = False,
    ) -> None:
        """Legacy fade-to-black assembly (used when no xfade_transition is specified)."""
        current = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            temp_out = work_dir / f"fade_{i}.mp4"
            _ffmpeg().fade_to_black(
                current, next_seg, temp_out, fade_duration=crossfade_dur, fps=fps, cfr=cfr
            )
            if current != segments[0] and current.exists():
                with contextlib.suppress(OSError):
                    current.unlink()
            current = temp_out
        shutil.copy2(current, output_path)

    def _assemble_with_xfade(
        self,
        segments: list[Path],
        output_path: Path,
        work_dir: Path,
        crossfade_dur: float,
        xfade_transition: str = "fade",
        fps: int = 30,
        cfr: bool = False,
    ) -> int:
        """Assemble segments using FFmpeg xfade transitions.

        Falls back to simple concat if any xfade step fails (e.g. OOM/SIGKILL).
        Returns the number of xfades actually applied before that fallback —
        the concat tail contains none, and the compilation report must not
        claim them.
        """
        current = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            temp_out = work_dir / f"xfade_{i}.mp4"
            try:
                _ffmpeg().crossfade(
                    current,
                    next_seg,
                    temp_out,
                    fade_duration=crossfade_dur,
                    transition_type=xfade_transition,
                    fps=fps,
                    cfr=cfr,
                )
            except (OSError, RuntimeError) as exc:
                logger.info(
                    "xfade failed at step %d, falling back to concat: %s",
                    i,
                    exc,
                )
                concat_tmp = work_dir / "fallback_concat.txt"
                remaining = [current, *segments[i:]]
                with concat_tmp.open("w") as f:
                    for seg in remaining:
                        f.write(f"file '{seg.resolve()}'\n")
                _ffmpeg().concat(concat_tmp, output_path, fps=fps, cfr=cfr)
                return i - 1
            if current != segments[0] and current.exists():
                with contextlib.suppress(OSError):
                    current.unlink()
            current = temp_out
        shutil.copy2(current, output_path)
        return len(segments) - 1
