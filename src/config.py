"""Configuration management for video_production_service using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import APISettings as BaseAPISettings
from w2v_common.config import AuthSettings as AuthSettings
from w2v_common.config import DatabaseSettings as BaseDatabaseSettings
from w2v_common.config import FFmpegSettings as FFmpegSettings
from w2v_common.config import IdempotencySettings as IdempotencySettings
from w2v_common.config import LoggingSettings as LoggingSettings
from w2v_common.config import RedisSettings as BaseRedisSettings
from w2v_common.config import StorageSettings as BaseStorageSettings


class DatabaseSettings(BaseDatabaseSettings):
    """Database settings — DATABASE_URL required env (no localhost fallback)."""


class RedisSettings(BaseRedisSettings):
    """Redis settings — REDIS_URL required env (no localhost fallback)."""


class APISettings(BaseAPISettings):
    """Extended API settings for video_production_service."""

    port: int = Field(default=6308, description="API server port")
    rate_limit_requests: int = Field(default=500, description="Max requests per window per tenant")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")
    max_page_size: int = Field(default=100, description="Maximum page size for list endpoints")
    # Hard ceiling on ONE SSE subscription (`GET /events/{tenant_id}`). An SSE
    # stream is not a request: it is a connection that pins a Redis pubsub, a
    # worker task and a socket for as long as the client keeps it, and
    # `Request.is_disconnected()` is the only thing that used to end it — which a
    # half-open TCP connection never trips. So a stream had NO upper bound on
    # lifetime, and a licence revoked after the stream opened never reached it
    # (the entitlement gate runs at connect time, once). The cap turns "forever"
    # into "at most one window", after which the client must reconnect THROUGH
    # the gate. 1h is chosen to be far longer than any compile job and far
    # shorter than a leaked connection's natural lifetime.
    sse_max_stream_seconds: float = Field(
        default=3600.0,
        gt=0,
        description="Max lifetime of a single SSE subscription before the server closes it",
    )


class StorageSettings(BaseStorageSettings):
    """Extended storage settings for video_production_service."""

    minio_bucket: str = Field(default="w2v-video-production", description="MinIO bucket name")
    minio_external_endpoint: str | None = Field(
        default=None,
        description="External MinIO endpoint for presigned URLs (replaces internal host when set)",
    )
    cross_bucket_allowlist: str = Field(
        default="",
        description=(
            "Comma-separated buckets, BESIDES this service's own `minio_bucket`, "
            "that `StorageService.retrieve_uri{,_sync}` may read. Those two take "
            "a whole `s3://<bucket>/<key>` URI off a tenant-supplied compile "
            "request, so without this list any bucket the service account can "
            "reach is readable and composited into a downloadable video — e.g. "
            "`s3://w2v-audiobooks/<anything>`. FAIL-CLOSED: empty (the default) "
            "means NO cross-bucket read at all, own bucket only. A deployment "
            "that legitimately compiles assets out of a sibling bucket must name "
            "those buckets in STORAGE_CROSS_BUCKET_ALLOWLIST; a refused read logs "
            "an error naming this env var so the ops fix is one grep away. "
            "Default empty in CODE, not merely in gitops, for the same reason "
            "`COMPILATION_ALLOW_LOCAL_FILE_INPUTS` is: 'only the gitops half "
            "landed' is how a sibling repo's identical guard sat disarmed."
        ),
    )
    retention_hours: int = Field(default=168, description="Job retention in hours (7 days)")
    cleanup_interval_seconds: int = Field(default=3600, description="Cleanup check interval")
    stale_reap_interval_seconds: int = Field(
        default=60,
        description=(
            "Seconds between STALE-reaper sweeps (`_reap_stale_jobs` only), run on a "
            "SEPARATE loop from `cleanup_interval_seconds`'s hourly expired-job sweep. "
            "Measured: a job stale at +stale_job_timeout_seconds (600s) was left "
            "un-reaped until the next HOURLY `_run_loop` tick -- up to +3596s late, ~6x "
            "the window it was meant to enforce, silently holding a fleet concurrency "
            "slot the whole time. The expired-job sweep does real storage-delete work "
            "and stays hourly; the stale sweep is one advisory-lock probe plus a small "
            "indexed SELECT, cheap enough to run every minute."
        ),
    )
    stale_job_timeout_seconds: int = Field(
        default=600,
        description=(
            "Seconds without PROGRESS before a job that is actually RUNNING "
            "(status='processing') is reaped. Measured from `updated_at`, which "
            "`mark_processing` and `record_progress` bump — i.e. the clock starts when "
            "work began, not when the job was created."
        ),
    )
    queued_job_timeout_seconds: int = Field(
        default=7200,
        description=(
            "Seconds a job may sit PRE-DISPATCH -- status='pending' (INSERTed, not yet "
            "handed to the queue) or status='queued' (waiting for a fleet-wide recording "
            "slot, doing nothing wrong) -- before it is treated as an ORPHAN and reaped. "
            "Both statuses share this one window because they are the same failure (the "
            "owning pod died before the recording started) measured from the same "
            "non-advancing `updated_at`. Before 2026-09-04 'pending' was covered by NO "
            "clock at all and such a row leaked forever. "
            "⚠️ This is NOT a queue-wait SLA — `recording.queue_wait_timeout_seconds` is, and "
            "it fails the job loudly and quickly. This one exists only to garbage-collect a "
            "row whose owning pod died between INSERT and dispatch, so it MUST stay far "
            "above both that SLA and every client-side timeout.\n\n"
            "Why it is separate at all (measured 2026-09-03/04): a single window covered both "
            "statuses at 600s, which EQUALLED the orchestrator's client timeout, so the reaper "
            "failed jobs that had never been given a slot. 282 jobs were accepted between "
            "21:40 and 01:30 and 276 of them logged 'Job lease NOT renewed … no progress' — "
            "they were WAITING, never RUNNING. Zero completions in 10 hours. A job that has "
            "not started is not stale; only a job that started and went silent is.\n\n"
            "Where the clock actually starts (checked 2026-09-04): `mark_queued` stamps "
            "`updated_at=now()` on the pending -> queued transition, so for a queued row "
            "this window runs from DISPATCH-to-queue, not from job creation; only a "
            "'pending' row still measures from creation. That makes the effective deadline "
            "LATER than the naive reading, i.e. strictly more headroom, so 7200 needs no "
            "adjustment -- it stays ~24x the 300s queue-wait SLA and 12x the 600s client "
            "timeout on either clock."
        ),
    )
    queued_orphan_timeout_seconds: int = Field(
        default=450,
        description=(
            "Accelerated staleness window for 'pending'/'queued' recording jobs, used "
            "by an ADDITIONAL reap sweep alongside (never instead of) "
            "`queued_job_timeout_seconds` -- and only when a JobLease is configured, "
            "i.e. only when liveness can be positively CONFIRMED rather than merely "
            "guessed from age.\n\n"
            "A job still owned by a live pod holds a live Redis lease for the WHOLE "
            "`RecorderService.process_job` body, including the queued wait -- see "
            "`_job_liveness` -- and that lease keeps being renewed by the heartbeat for "
            "as long as `stalled_recording_job_ids` (unchanged, still keyed on "
            "`queued_job_timeout_seconds`) does not call the row stale. So a row this "
            "sweep selects whose lease is ALSO absent cannot be a live pod's job still "
            "within its normal queue wait -- it can only be a pod that died before "
            "`RecordingConcurrencyManager.acquire`'s own `queue_wait_timeout_seconds` "
            "(300s) loop could fail it out on its own. Must stay safely above that "
            "300s SLA and above `job.lease_ttl_seconds` (120s) + "
            "`job.lease_heartbeat_seconds` (45s) so a momentarily-unrenewed-but-alive "
            "lease read is never mistaken for a dead one; 450s clears both with "
            "margin. `queued_job_timeout_seconds` (7200s) remains the ONLY guard when "
            "no lease is configured (Redis absent) -- there liveness cannot be "
            "confirmed at all, so the wide age-only window stays load-bearing on its "
            "own and is deliberately left untouched by this setting."
        ),
    )
    compilation_pending_stale_seconds: int = Field(
        default=21600,
        description=(
            "Seconds a COMPILATION job may sit at status='pending' before it is treated "
            "as an ORPHAN and reaped. `POST /api/compile` commits the row at 'pending' "
            "and only then spawns the background task, so nothing bumps `updated_at` "
            "while the job waits: this clock runs from job CREATION and measures queue "
            "wait, not silence.\n\n"
            "⚠️ This MUST stay far above `stale_job_timeout_seconds` (600) and must NEVER "
            "be collapsed into it. 'pending' is a LEGITIMATE queued state, so reaping it "
            "on the no-progress window would fail live queued work — the exact 2026-09-03 "
            "recording outage (276 of 282 jobs reaped while still waiting, zero "
            "completions in 10 hours), reproduced in the compilation domain.\n\n"
            "Why the window exists at all: before 2026-09-05 the compilation reaper "
            "matched `status IN ('processing','queued')`, and 'queued' is never written "
            "for compilation_jobs — so a pod that took SIGTERM between the commit and the "
            "first `mark_processing` write left the row at 'pending' with NO clock able to "
            "reach it, and the caller's poller waited forever. 6h is ~36x the 600s client "
            "timeout: long enough that no real queue wait is ever mistaken for an orphan, "
            "short enough that a crashed dispatch is collected the same day."
        ),
    )
    compilation_batch_stale_seconds: int = Field(
        default=21600,
        description=(
            "WALL-CLOCK CEILING on a whole compilation BATCH, measured from the last "
            "COMMITTED write to the row. Past it, a batch still at 'pending' or "
            "'processing' is eligible to be failed as an orphan.\n\n"
            "The only writer that ever advances a batch is the request-spawned "
            "`BatchService.process_batch` task. If the pod restarts, OOMs, or that task "
            "raises, nothing else in the system will ever touch the row again — before "
            "2026-09-06 there was no reaper AND no prune for `compilation_batches` at "
            "all, so the row was immortal and a client polling batch status waited "
            "forever.\n\n"
            "⚠️ THIS IS NOT A NO-PROGRESS WINDOW, however much it looks like the job "
            "clocks above. `process_batch` opens ONE transaction for the batch's whole "
            "life and does not commit between the 'processing' flip and the final "
            "status write — every per-job progress write is a bare flush() — so from "
            "any OTHER connection, including the reaper's, `updated_at` is frozen at "
            "batch start no matter how much progress the batch is making. Reading this "
            "knob as 'silence' is wrong and was the original mis-specification.\n\n"
            "What protects live work is the ROW LOCK, not this clock: from the first "
            "child-job flush the owning transaction holds the row, and the reaper uses "
            "SKIP LOCKED, so an in-flight batch is passed over. The residual exposure "
            "is a batch whose FIRST child job has not finished inside this window — it "
            "is reaped mid-flight, and `process_batch` then strips the reaper's error "
            "entry if it finishes anyway. Closing that residue needs a COMMITTED "
            "heartbeat on the batch row, which is a change to `process_batch`'s "
            "transaction shape and has NOT been made.\n\n"
            "Size it as the longest batch you are willing to let run, not as a silence "
            "threshold: 6h, i.e. 36x the 600s per-job no-progress window. It must NEVER "
            "be collapsed onto `stale_job_timeout_seconds`, which would cap every batch "
            "at 10 minutes of total runtime."
        ),
    )


class RecordingSettings(BaseSettings):
    """Playwright recording configuration."""

    viewport_width: int = Field(default=1920, description="Recording viewport width")
    viewport_height: int = Field(default=1080, description="Recording viewport height")
    timeout_ms: int = Field(default=120000, description="Playwright test timeout in ms")
    slow_mo_ms: int = Field(default=0, description="Slow motion delay between actions")
    timeout_buffer_s: int = Field(
        default=30, description="Extra seconds beyond test timeout for subprocess"
    )
    min_video_size_bytes: int = Field(
        default=10000, description="Minimum video size to accept (bytes)"
    )
    color_scheme: str = Field(default="light", description="Color scheme: light or dark")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    max_concurrent_recordings: int = Field(
        default=6,
        description=(
            "Max concurrent recording subprocesses, FLEET-WIDE (not per-pod). "
            "Raised 3 → 6 on 2026-09-04. Justification, from measurement rather than taste: "
            "real recording work is 1-20s (the `video=` field of the completion line), while "
            "wall time climbed 15.4s → 121.7s → 621.3s → 928.9s → 1212.3s purely as queue "
            "wait. Demand is bounded and known: the orchestrator's `ken_burns` stage fans out "
            "at concurrency=2 and the E2E scenario matrix runs 4 shards, so the peak is 8 "
            "concurrent recordings. 6 is the largest cap that is ALWAYS actually reachable: "
            "the HPA is min 2 / max 3 replicas and the per-pod sub-limit below is 3, so at the "
            "guaranteed floor of 2 pods the fleet can genuinely run 6. Setting it to 8 would "
            "mean a cap that 2 pods cannot fill (they would exceed the per-pod memory ceiling "
            "first), i.e. a number that lies during a rollout. The residual queue of 2 at peak "
            "drains in one 20s work unit — three orders of magnitude inside the 600s client "
            "timeout. Override with RECORDING_MAX_CONCURRENT_RECORDINGS."
        ),
    )
    max_concurrent_recordings_per_pod: int = Field(
        default=3,
        description=(
            "Per-pod sub-limit, held BELOW the fleet-wide cap. Kept at 3 deliberately: 3 is the "
            "concurrency the current 8Gi/2000m pod envelope is known to survive (it OOM-killed "
            "at 2Gi and again at 4Gi under nightly-e2e render load — Chromium + ffmpeg is "
            "~2.7Gi of headroom per recording). Raising the FLEET cap adds throughput by "
            "spreading across replicas; raising THIS would add throughput by stacking Chromium "
            "processes inside one memory limit, which is how the pod gets OOM-killed. Change "
            "the pod's memory limit first, not this. Override with "
            "RECORDING_MAX_CONCURRENT_RECORDINGS_PER_POD."
        ),
    )
    queue_wait_timeout_seconds: float = Field(
        default=300.0,
        description=(
            "Ceiling on how long `acquire()` waits for a recording slot before failing the job "
            "with a QUEUE_SATURATED error that is textually distinct from a recording failure. "
            "0 disables the bound (the old, unbounded behaviour). 300s is deliberately BELOW "
            "the orchestrator's 600s client timeout: an unbounded wait turns a capacity problem "
            "into a silent client-side timeout with the job still nominally alive, which is "
            "exactly how the 2026-09-03 collapse presented. Failing at 300s makes saturation "
            "surface as a loud, attributable error with ~300s of slack left for the caller to "
            "react. It is also far above the 20s worst-case work unit, so it can only fire on "
            "genuine sustained oversubscription, never on normal queueing."
        ),
    )
    global_concurrency_enabled: bool = Field(
        default=True,
        description=(
            "Enforce max_concurrent_recordings as a SINGLE cap across all replicas via a "
            "Redis distributed semaphore. When Redis is unavailable the limiter degrades to "
            "a per-pod asyncio.Semaphore (the pre-multi-replica behaviour).\n\n"
            "⚠️ ACCEPTED DEGRADATION, stated rather than left to be rediscovered: with Redis "
            "down the fleet-wide cap is UNENFORCED, so at the HPA ceiling of 3 replicas the "
            "fleet can run 3 x max_concurrent_recordings_per_pod = 9 concurrent recordings, "
            "above the cap of 6. That is deliberate, because the two limits protect different "
            "things. The per-pod sub-limit is the SAFETY bound -- it is the concurrency the "
            "8Gi pod envelope is known to survive -- and it is enforced locally, so it holds "
            "with or without Redis; no pod can be OOM-killed by this degradation. The fleet "
            "cap is a throughput/fairness bound on shared downstream capacity, and briefly "
            "overshooting it by 3 during a Redis outage is a far better failure than the "
            "alternatives: failing closed would turn a cache outage into a total recording "
            "outage, and that is the same class of over-eager rejection as the 2026-09-03 "
            "collapse this setting's neighbours exist to prevent."
        ),
    )
    global_slot_ttl_seconds: int = Field(
        default=900,
        description=(
            "TTL on a held global recording slot. A crashed holder's slot auto-releases once "
            "this elapses so a dead pod can never leak the fleet-wide cap. Heartbeated at "
            "TTL/2 while a recording runs; must exceed the longest single recording."
        ),
    )
    global_slot_poll_seconds: float = Field(
        default=0.5, description="Poll interval while waiting for a free global recording slot"
    )
    executor_stall_threshold_seconds: float = Field(
        default=1800.0,
        description=(
            "How long the recording wait queue may be non-empty with ZERO slot admissions or "
            "releases before /health/live reports the executor wedged and the kubelet restarts "
            "the pod. Must exceed the longest legitimate slot hold (global_slot_ttl_seconds) "
            "with headroom -- an idle service is never unhealthy, so this only fires when work "
            "is demonstrably waiting and nothing is moving. Set to 0 to disable the check."
        ),
    )
    global_slot_redis_key: str = Field(
        default="video_production:recording:global_slots",
        description="Redis sorted-set key backing the fleet-wide recording slot semaphore",
    )
    max_retries: int = Field(default=2, description="Max retry attempts for failed recordings")
    retry_base_delay_s: float = Field(
        default=5.0, description="Base delay between retries in seconds"
    )
    retryable_error_patterns: list[str] = Field(
        default=["timed out", "browser disconnected", "target closed", "connection refused"],
        description="Error patterns that trigger retry",
    )
    scripts_base_dir: str = Field(
        default="/app/scripts", description="Allowed base directory for scripts"
    )
    output_format: str = Field(
        default="mp4",
        description="Output video format: 'mp4' (H.264) or 'webm'.",
    )
    default_quality_preset: str = Field(
        default="standard",
        description="Default quality preset: preview, standard, high_quality, 2k, 4k",
    )
    error_output_max_length: int = Field(
        default=500, description="Max chars for error output in recording results"
    )
    cancel_wait_timeout_s: float = Field(
        default=5.0, description="Seconds to wait for subprocess termination on cancel"
    )
    animation_render_timeout_floor_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Hard floor on AnimationRecorder.record_animation's per-job execution "
            "timeout (browser frame-step capture + ffmpeg encode), which is otherwise "
            "sized from the job's OWN frame count/fps. Guarantees even a 1-frame job "
            "gets enough time to actually launch Chromium and run ffmpeg once."
        ),
    )
    animation_render_timeout_ceiling_seconds: float = Field(
        default=18500.0,
        gt=0,
        description=(
            "Absolute BACKSTOP on the animation execution timeout -- a sanity bound, "
            "not the primary hang detector (that is the per-frame stall watchdog in "
            "_capture_frames, sized by animation_render_stall_window_seconds). "
            "CORRECTED 2026-09-17: the previous 480.0 default was picked to sit under "
            "the orchestrator's 600s /api/record/jobs poll, but that poll only applies "
            "to title-card callers -- other callers render far longer, and a flat "
            "wall-clock ceiling cannot distinguish a stuck render from a legitimately "
            "big one. Replaying it against production `recording_jobs` (n=2105 "
            "completed animation jobs, p99 recording_time_seconds=606.5s, "
            "max=767.06s) showed it would have killed 80 of 2106 successful jobs "
            "(3.8%) mid-render -- it saturates for any job over ~260 frames. This "
            "default is sized for the worst case the frame-count formula can produce "
            "(_MAX_FRAMES=3600 frames * (animation_render_seconds_per_frame=4.5 + "
            "animation_render_ffmpeg_seconds_per_frame=0.5) + "
            "animation_render_encode_reserve_seconds=120 = 18120s), rounded up with "
            "margin -- so it can never truncate a per-frame-scaled budget within the "
            "max frame count, and never kills a job that historically succeeded."
        ),
    )
    animation_render_stall_window_seconds: float = Field(
        default=90.0,
        gt=0,
        description=(
            "PRIMARY hang detector for the frame-step capture loop: each frame's "
            "`page.evaluate` (timeline seek) + `page.screenshot` is wrapped in its "
            "own `asyncio.wait_for` at this timeout, so a render fails as soon as NO "
            "SINGLE FRAME makes progress within the window -- not after the whole job "
            "runs long. It bounds ONLY that one call; it cannot fire during page load "
            "(`page.set_content(..., timeout=30000)`, a separate fixed bound) or "
            "during ffmpeg encoding (a separate subprocess timeout sized by "
            "animation_render_ffmpeg_seconds_per_frame / "
            "animation_render_encode_reserve_seconds) -- neither phase can be killed "
            "by this setting no matter how long it legitimately runs. "
            ""
            "CORRECTED 2026-09-17: this was previously justified as a 'per-frame "
            "capture time distribution' (p50=0.547s ... max=5.859s) measured against "
            "2053 production jobs, but that is false -- the source numbers are "
            "`recording_time_seconds / estimated_frame_count`, a per-JOB AVERAGE, not "
            "per-frame telemetry. No per-frame timing existed anywhere in this "
            "service before this fix (see `max_frame_seconds`, added alongside this "
            "correction, which finally records one). An average cannot bound a single "
            "frame: a 500-frame job can absorb one 30s outlier and still report an "
            "unremarkable mean, so 'max observed average' says nothing about the "
            "worst single frame anyone has ever actually seen. Independently "
            "reproducing that same average-based calculation gives max ~4.687s, not "
            "5.859s -- the number itself was also never re-verified. "
            ""
            "Absent real per-frame data, this value is a deliberate ENGINEERING "
            "MARGIN, not a measured bound: `page.evaluate` (pause+seek a handful of "
            "Web Animations) and `page.screenshot` are both bounded, in-process "
            "Chromium operations with no network or long-running work inside them, so "
            "a legitimate one should complete in well under a second even under load; "
            "the ~4.7s worst-case JOB AVERAGE is the only real evidence available "
            "that per-frame cost can run into multiple seconds under contention. "
            "Raised from 60 -> 90 here specifically because 60s could not be defended "
            "against an average that already reaches 4.7s with an unknown, possibly "
            "much larger, single-frame tail hiding inside it -- 90s keeps ~19x margin "
            "over that average while still failing far below the outer per-job "
            "capture-timeout backstop (capture_timeout_s in record_animation, which "
            "scales with frame count and reaches into the thousands of seconds for a "
            "full-length job). Tune this from `max_frame_seconds` once enough real "
            "per-frame data has accumulated, rather than by intuition again."
        ),
    )
    animation_render_launch_timeout_seconds: float = Field(
        default=45.0,
        gt=0,
        description=(
            "Bound on Chromium launch + new_context + new_page during "
            "_capture_frames' setup, BEFORE the frame-step loop starts. These calls "
            "used to run ahead of the try/finally that calls _safe_close, so a hang "
            "during startup leaked the browser/context with no teardown at all -- "
            "fixed 2026-09-17 by moving them inside the guard. 45s is generous "
            "headroom over a healthy launch (low single-digit seconds); a launch that "
            "does not complete in this window is presumed wedged, not slow."
        ),
    )
    animation_render_seconds_per_frame: float = Field(
        default=4.5,
        gt=0,
        description=(
            "Per-frame CAPTURE time budget (JS timeline seek + screenshot) used to "
            "size the overall capture-phase timeout from the job's own frame "
            "count/fps rather than a flat number. This is a BACKSTOP behind the "
            "stall watchdog (animation_render_stall_window_seconds), not itself the "
            "hang detector -- it only has to bound the SUM of frame times, not each "
            "one, which is exactly what a per-job AVERAGE is fit to do (unlike the "
            "stall window, which needs a single-frame bound an average cannot give --"
            " see that field's docstring for the correction). Raised 2026-09-17 from "
            "1.5 (the value that caused the 480s-ceiling kills): measured against "
            "2053 completed production animation jobs, `recording_time_seconds / "
            "estimated_frame_count` runs p99=1.731s, p99.9=3.598s, max=5.859s -- 4.5 "
            "sits above p99.9 with margin so a job whose average frame cost genuinely "
            "runs high, but never stalls on any single frame, is never killed by this "
            "budget alone."
        ),
    )
    animation_render_ffmpeg_seconds_per_frame: float = Field(
        default=0.5,
        gt=0,
        description=(
            "Per-frame ffmpeg budget (lanczos downscale pass + final encode), added "
            "to animation_render_encode_reserve_seconds to size the OS-level "
            "subprocess timeout for ffmpeg. NEW 2026-09-17 -- previously the ffmpeg "
            "timeout was the flat animation_render_encode_reserve_seconds regardless "
            "of frame count, which did not scale for large clips. Production "
            "recording_jobs has no separate encode-only telemetry (recording_time_"
            "seconds covers capture+encode together), so this is a deliberately "
            "generous fraction of the measured per-frame capture rate -- ffmpeg's "
            "own per-frame cost (PNG decode + encode) is well under Chromium's "
            "per-frame JS-seek-and-screenshot cost on a healthy pod."
        ),
    )
    animation_render_encode_reserve_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Fixed reserve, on top of the per-frame budgets above, for browser "
            "launch/teardown overhead and other fixed per-job cost that does not "
            "scale with frame count. Raised 2026-09-17 from 90 to 120: replaying the "
            "new (capture 4.5 + ffmpeg 0.5) = 5.0s/frame budget against all 2053 "
            "completed production animation jobs, the tightest historical job (48 "
            "frames, 281.21s total -- a low-frame job dominated by fixed overhead, "
            "not sustained per-frame cost) only required a fixed reserve of 41.21s "
            "to fit; 120 leaves ~3x margin on that boundary. Also passed straight "
            "through, plus the frame-scaled ffmpeg term, as the ffmpeg subprocess "
            "`timeout=` (see animation_render_ffmpeg_seconds_per_frame), so a stuck "
            "ffmpeg is killed at the OS level within that window."
        ),
    )

    @field_validator("output_format", mode="before")
    @classmethod
    def validate_output_format(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.lower().strip()
            if v not in ("mp4", "webm"):
                raise ValueError(f"Invalid output_format: {v}. Must be 'mp4' or 'webm'.")
        return str(v)

    @field_validator("default_quality_preset", mode="before")
    @classmethod
    def validate_quality_preset(cls, v: Any) -> str:
        if isinstance(v, str):
            v = v.lower().strip()
            valid = ("preview", "standard", "high_quality", "2k", "4k")
            if v not in valid:
                raise ValueError(f"Invalid quality preset: {v}. Must be one of {valid}.")
        return str(v)

    model_config = SettingsConfigDict(
        env_prefix="RECORDING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class JobSettings(BaseSettings):
    """Job lifecycle configuration."""

    max_error_length: int = Field(default=4000, description="Max error length stored in DB")
    log_error_truncate_length: int = Field(
        default=120, description="Max error length in log messages"
    )
    expired_batch_limit: int = Field(default=100, description="Max expired jobs per cleanup cycle")
    lease_ttl_seconds: int = Field(
        default=120,
        description=(
            "TTL on a per-job Redis liveness lease held by the pod running a recording/"
            "compilation job. The lease-gated reapers skip any job whose lease is still live "
            "(running on another pod); a crashed pod's lease TTL-expires so the reaper can then "
            "safely fail the orphaned job. Must exceed the reaper's heartbeat interval."
        ),
    )
    lease_heartbeat_seconds: int = Field(
        default=45, description="Interval at which a running job refreshes its Redis liveness lease"
    )

    model_config = SettingsConfigDict(
        env_prefix="JOB_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class WebhookSettings(BaseSettings):
    """Webhook delivery configuration."""

    max_retries: int = Field(default=3, description="Max webhook delivery attempts")
    backoff_base: float = Field(default=2.0, description="Exponential backoff base")
    timeout: float = Field(default=10.0, description="HTTP request timeout in seconds")
    enabled: bool = Field(default=True, description="Enable webhook delivery")

    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CompilationSettings(BaseSettings):
    """Video compilation configuration."""

    output_filename: str = Field(default="demo.mp4", description="Output video filename")
    crf: int = Field(default=20, description="H.264 CRF quality (lower = better)")
    target_width: int = Field(default=1920, description="Target video width")
    target_height: int = Field(default=1080, description="Target video height")
    target_fps: int = Field(default=30, description="Target frame rate")
    cfr_mode: bool = Field(
        default=True,
        description="Emit constant-frame-rate output (-fps_mode cfr -r target_fps) in compile ops",
    )
    transition_type: str = Field(default="smart", description="Transition type: cut, fade, smart")
    crossfade_duration: float = Field(default=0.5, description="Crossfade duration in seconds")
    cleanup_intermediates: bool = Field(
        default=True, description="Remove temp files after compilation"
    )
    allow_local_file_inputs: bool = Field(
        default=False,
        description=(
            "Read `file://` and bare on-disk paths from ANYWHERE on the pod, "
            "instead of only from inside STORAGE_LOCAL_PATH. Applies to the "
            "tenant-supplied asset references on a compile request — "
            "`shot_list.parts[].image_url` / `slide_image_url` / "
            "`background_image`, `recordings[].video_path` and the "
            "findings-video equivalents. Defaults FALSE (secure-by-default): a "
            "hostile `slide_image_url='file:///etc/passwd'` would otherwise be "
            "read off the pod and composited into a downloadable video (LFI). "
            "The default is FALSE in CODE, not merely in gitops, because 'only "
            "the gitops half landed' is exactly how a sibling repo's identical "
            "guard sat disarmed in prod. FALSE does NOT break the record -> "
            "compile round trip: `GET /api/record/jobs/{id}/result` publishes a "
            "path under STORAGE_LOCAL_PATH, which stays inside the confinement "
            "boundary. Only tests and local dev whose fixtures live in a tmpdir "
            "need COMPILATION_ALLOW_LOCAL_FILE_INPUTS=true."
        ),
    )
    max_concurrent_ffmpeg_subprocesses: int = Field(
        default=2,
        ge=1,
        description=(
            "Process-wide ceiling on simultaneously-running ffmpeg subprocesses. "
            "Job-level limits (max_concurrent batch, per-tenant) bound the number "
            "of compilation JOBS, but each job spawns many sequential ffmpeg "
            "processes (one per xfade/compose step); under concurrent load their "
            "combined full-resolution decode buffers can exceed the pod memory "
            "limit and the kernel SIGKILLs ffmpeg (exit=-9). This gate caps total "
            "concurrent ffmpeg memory regardless of how many jobs are in flight. "
            "Default 2 is conservative for a 1080p workload on an 8Gi pod."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="COMPILATION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ThemeSettings(BaseSettings):
    """Title card and visual theme configuration."""

    primary_color: str = Field(default="#2563EB", description="Primary color")
    background_color: str = Field(default="#0F172A", description="Background color")
    text_color: str = Field(default="#F8FAFC", description="Text color")
    accent_color: str = Field(default="#F59E0B", description="Accent color")
    subtitle_color: str = Field(default="#94A3B8", description="Subtitle color")
    gradient_start: str = Field(default="#0F172A", description="Gradient start color")
    gradient_end: str = Field(default="#1E293B", description="Gradient end color")
    font_path: str = Field(default="", description="Path to regular font file")
    font_bold_path: str = Field(default="", description="Path to bold font file")

    model_config = SettingsConfigDict(
        env_prefix="THEME_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class BackgroundMusicSettings(BaseSettings):
    """Background music configuration."""

    path: str = Field(default="", description="Path to background music file")
    volume: float = Field(default=0.1, description="Music volume (0.0-1.0)")
    fade_in_s: float = Field(default=2.0, description="Fade in duration in seconds")
    fade_out_s: float = Field(default=3.0, description="Fade out duration in seconds")

    model_config = SettingsConfigDict(
        env_prefix="MUSIC_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CleanupSettings(BaseSettings):
    """Job cleanup configuration."""

    enabled: bool = Field(default=True, description="Enable automatic cleanup")
    interval_seconds: int = Field(default=3600, description="Cleanup check interval")
    completed_ttl_days: int = Field(default=30, description="Days to keep completed jobs")
    failed_ttl_days: int = Field(default=7, description="Days to keep failed jobs")

    model_config = SettingsConfigDict(
        env_prefix="CLEANUP_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class WatermarkSettings(BaseSettings):
    """Default watermark configuration."""

    image_path: str = Field(default="", description="Default watermark image path")
    position: str = Field(default="bottom-right", description="Default position")
    opacity: float = Field(default=0.7, description="Default opacity (0.0-1.0)")
    scale: float = Field(default=0.15, description="Default scale relative to video")

    model_config = SettingsConfigDict(
        env_prefix="WATERMARK_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    enabled: bool = Field(default=True, description="Enable rate limiting")
    max_concurrent_jobs: int = Field(default=3, description="Max concurrent jobs per tenant")
    daily_quota: int = Field(default=100, description="Max compilations per tenant per day")
    requests_per_minute: int = Field(default=30, description="Max API requests per minute")

    model_config = SettingsConfigDict(
        env_prefix="RATELIMIT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class OTelSettings(BaseSettings):
    """OpenTelemetry observability configuration."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry instrumentation")
    service_name: str = Field(default="video-production-service", description="OTel service name")
    # Empty default — OTel is opt-in. Production sets OTEL_OTLP_ENDPOINT via env;
    # exporter never fires when the endpoint is blank (and OTEL_ENABLED is False).
    otlp_endpoint: str = Field(default="", description="OTLP gRPC exporter endpoint")

    model_config = SettingsConfigDict(
        env_prefix="OTEL_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""

    app_name: str = Field(default="Video Production Service", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development", description="Environment (development, staging, production)"
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    ffmpeg: FFmpegSettings = Field(default_factory=FFmpegSettings)
    job: JobSettings = Field(default_factory=JobSettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)
    compilation: CompilationSettings = Field(default_factory=CompilationSettings)
    theme: ThemeSettings = Field(default_factory=ThemeSettings)
    background_music: BackgroundMusicSettings = Field(default_factory=BackgroundMusicSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    watermark: WatermarkSettings = Field(default_factory=WatermarkSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    otel: OTelSettings = Field(default_factory=OTelSettings)
    # Canonical Idempotency-Key block read by w2v_common.asgi.create_app. The
    # orchestrator sends ``Idempotency-Key: {run_id}:{stage}:{media_id}`` on
    # ``POST /api/record/animation``; without this block the middleware is never
    # registered and a resumed run (orchestrator pod roll) re-POSTs and pays for a
    # SECOND recording. Forced on (same shape as w2v_orchestrator): a dedup guard
    # that an env typo can switch off is not a guard. Store = Redis on
    # ``app.state.redis`` (falls open when Redis is down). TTL 24h (default) is
    # well above the orchestrator's 7200s resume window.
    idempotency: IdempotencySettings = Field(default_factory=IdempotencySettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @model_validator(mode="after")
    def _wire_idempotency(self) -> Settings:
        """Always register the canonical Idempotency-Key middleware."""
        self.idempotency.enabled = True
        return self

    def get_database_url(self) -> str:
        """Return the configured database connection URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True if environment is 'production'."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True if environment is 'development'."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance (useful for testing)."""
    get_settings.cache_clear()
