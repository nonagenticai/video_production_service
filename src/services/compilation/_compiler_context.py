"""Shared mutable context for compilation stages."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Module-level set tracking fire-and-forget webhook tasks (prevents GC).
_webhook_tasks: set[asyncio.Task[None]] = set()


@dataclass
class _JobContext:
    """Mutable state shared across compilation stages."""

    job_id: str
    tenant_id: uuid.UUID
    job: Any
    settings: Any
    work_dir: Path = field(default_factory=lambda: Path())
    edited_dir: Path = field(default_factory=lambda: Path())
    title_cards_dir: Path = field(default_factory=lambda: Path())
    normalized_dir: Path = field(default_factory=lambda: Path())
    output_path: Path = field(default_factory=lambda: Path())
    target_w: int = 1920
    target_h: int = 1080
    target_fps: int = 30
    cfr_mode: bool = True
    crf: int = 23
    transition_type: str = "cut"
    crossfade_dur: float = 0.5
    parts_by_id: dict = field(default_factory=dict)
    recording_order: list = field(default_factory=list)
    recording_map: dict = field(default_factory=dict)
    edit_decisions: dict = field(default_factory=dict)
    narration_audio_dir: Path | None = None
    clips_dir: Path | None = None
    title_gen: Any = None
    theme_config: dict | None = None
    editor: Any = None
    input_duration: float = 0.0
    segments: list[Path] = field(default_factory=list)
    segments_excluded: int = 0
    segments_with_narration: int = 0
    transitions_used: int = 0
    stage_timings: list[dict] = field(default_factory=list)
    subtitle_error: str | None = None
    degradations: list[dict[str, str]] = field(default_factory=list)
    audio_mixing_details: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    asset_cache: dict[str, Path] = field(default_factory=dict)
    # Monotonic count of FORWARD PROGRESS events, and the stage that last made
    # one. This is the compilation path's liveness signal, and it did not exist:
    # `updated_at` was written only on STATE TRANSITIONS (`mark_processing`,
    # `mark_completed`, `mark_failed`) -- there is no `update_progress` on this
    # domain at all, unlike `recording/job_manager.py`. So the stale reaper's
    # `updated_at <= now() - 600s` predicate could only ever measure AGE, and
    # reaped every compile that ran longer than ten minutes. p50 compile is
    # ~7.6 min, so that fired on ordinary healthy jobs.
    #
    # A COUNTER rather than a timer, deliberately. A heartbeat that beats on a
    # clock alone would refresh a genuinely wedged compile forever -- the same
    # leaked-lease-renewed-forever shape that hid the nine-hour recording wedge.
    # The heartbeat only writes when this number has MOVED, so silence is what
    # ages the row: a progressing compile is never reaped however long it runs,
    # and a wedged one still is.
    progress_ticks: int = 0
    progress_stage: str = "starting"
    progress_percent: float = 0.0
    # True ONLY after `mark_completed` actually wrote the completed row. False
    # when the transition lost to a terminal state written by somebody else -- the
    # stale reaper, or a cancel. `process_job` used to fire a "completed" webhook
    # unconditionally after `_finalize` returned, and `_finalize` swallows that
    # conflict and returns normally, so a job the reaper had just failed still
    # told its consumer it had completed -- with an `output_path` that was never
    # persisted. A false failure and a false success out of one defect.
    completed: bool = False

    def mark_progress(self, stage: str | None = None, percent: float | None = None) -> None:
        """Record one unit of forward progress. See ``progress_ticks``."""
        self.progress_ticks += 1
        if stage is not None:
            self.progress_stage = stage
        if percent is not None:
            self.progress_percent = percent

    def record_degradation(
        self, feature: str, reason: str, exc: BaseException | None = None
    ) -> None:
        """Record that *feature* was requested but dropped to keep the video.

        A stage on the compile path that swallows a failure SHOULD call this.
        The compilation report derives its ``status`` from whether this list is
        empty, so a swallow that does not record here is reported to the caller
        as a fully-successful compile.

        ⚠️ Coverage is PARTIAL and this list is not a complete inventory of what
        a compile can drop -- ``title_card_generator.py`` swallows three drops
        with no call here at all. ``tests/unit/test_degradation_coverage.py``
        catches only one swallow shape (``except`` + ``logger.warning`` + no
        re-raise, in five modules) and can be silenced by adding an allowlist
        entry, so it is a partial ratchet, not a gate. Treat an empty list as
        "nothing covered fired", never as "nothing went wrong".

        ``reason`` MUST be a fixed literal chosen by the call site -- never an
        exception message, path, URL, or anything else derived from runtime
        data. It is returned verbatim to the API caller, and the failures being
        described are exactly the ones carrying sensitive detail: a MinIO
        ``S3Error`` names the bucket, object key, host and request id; an httpx
        error names the internal endpoint; a filesystem error names a server
        path. ``_apply_subtitles`` already refuses to echo a rejected path to
        avoid a path-existence oracle, and this channel must not reintroduce
        what that docstring rules out. Pass *exc* to attach ``error_type`` --
        the exception CLASS name only, which is diagnostic but carries no
        runtime data; the full detail belongs in the logs.
        """
        entry = {"feature": feature, "reason": reason}
        if exc is not None:
            entry["error_type"] = type(exc).__name__
        self.degradations.append(entry)

    def record_stage(self, name: str, start_t: float, **details: Any) -> None:
        self.stage_timings.append(
            {
                "stage": name,
                "duration_seconds": round(time.time() - start_t, 3),
                "details": details or None,
            }
        )
        # Every completed stage is forward progress, so the coarse ticks come for
        # free at all seven `record_stage` sites. The per-part loop in
        # `_segment_builders` ticks separately, because it is the long one and a
        # stage-granularity signal alone would go silent inside it.
        self.mark_progress(stage=name)
