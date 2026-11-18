"""SQLAlchemy ORM models for video_production_service."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID  # noqa: N811
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


# ── Recording Domain ─────────────────────────────────────────────


class RecordingJob(Base):
    """A single video recording job for one shot list part."""

    __tablename__ = "recording_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")

    # Recording context
    part_id: Mapped[str] = mapped_column(String(128), nullable=False)
    part_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scripts_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    script_source: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Recording config overrides (JSONB)
    recording_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    auth_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Results
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The raster the recorder ACTUALLY captured at, before the lanczos downscale
    # to width x height. AnimationRecorder supersamples (_CAPTURE_SCALE = 2) and
    # _resolve_capture_scale steps that factor DOWN silently when the surface
    # would exceed the raster budget -- a 4k order captures at 1x. That decision
    # was invisible outside the recorder, so a caller sizing source stills to the
    # capture could only guess, or hardcode the private constant and regress the
    # day it changes. Stored as absolute dimensions (not the factor) so the
    # consumer does no arithmetic: capture_width == width IS the clamp-down.
    # NULL on rows written before this column existed and on every recorder that
    # does not supersample; readers must treat NULL as "not reported" and fall
    # back to width/height, which is exactly the pre-feature behaviour.
    capture_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    capture_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Slowest single frame's own capture time (JS timeline seek + screenshot)
    # during AnimationRecorder's frame-step loop -- see
    # Settings.animation_render_stall_window_seconds for why this exists: the
    # stall watchdog was sized from job-level AVERAGES, and this is the real
    # per-frame telemetry that lets it eventually be tuned from data instead.
    # NULL on every row written before this column existed and on every
    # non-animation recording job (content/url recorders don't frame-step).
    max_frame_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_audio: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    # WHAT the stored artifact is -- "video" (the default) or "screens_zip".
    # See src/services/recording/artifacts.py. The producer declares this and the
    # completion path already dispatched on it, but it was never PERSISTED, so the
    # knowledge died at completion and the download route had to guess: it mapped
    # the stored suffix through a video-only table and served a screens .zip as
    # `Content-Type: video/webm`. Stored so the serving path can dispatch on the
    # same declared fact the verification path does.
    # NULL on every row written before this column existed; readers must treat
    # NULL as "video", which is what those rows overwhelmingly are and what the
    # strict default already assumes.
    artifact_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # EXECUTION time only -- the span after a fleet recording slot was acquired.
    # It included the slot wait until #156; see queue_wait_seconds below.
    recording_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Seconds spent waiting for a fleet recording slot (Redis semaphore, cap 3).
    # Kept rather than discarded: it is the number a capacity question needs, and
    # folding it into recording_time_seconds is exactly the defect #156 fixed.
    # NULL on every row written before #156 -- there is no honest backfill, since
    # the split was never recorded.
    queue_wait_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Progress tracking
    progress: Mapped[float | None] = mapped_column(Float, nullable=True, default=0.0)
    progress_stage: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Retry support
    retry_count: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    max_retries: Mapped[int | None] = mapped_column(Integer, nullable=True, default=2)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Batch support
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    priority: Mapped[int | None] = mapped_column(Integer, nullable=True, default=0)
    batch_order: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Expiry
    expired: Mapped[bool | None] = mapped_column(Boolean, nullable=True, default=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Webhook
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (Index("ix_recording_jobs_tenant_status", "tenant_id", "status"),)


class RecordingUsage(Base):
    """Daily usage aggregation per tenant for recordings."""

    __tablename__ = "recording_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    date: Mapped[str] = mapped_column(String(10), nullable=False)
    recordings_started: Mapped[int] = mapped_column(Integer, default=0)
    recordings_completed: Mapped[int] = mapped_column(Integer, default=0)
    recordings_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    total_file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (UniqueConstraint("tenant_id", "date", name="uq_recording_usage_tenant_date"),)


# ── Compilation Domain ───────────────────────────────────────────


class CompilationJob(Base):
    """A video compilation job."""

    __tablename__ = "compilation_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, default=lambda: str(uuid.uuid7()))
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")

    # Compilation inputs
    shot_list: Mapped[dict | None] = mapped_column(JSONB)
    recordings: Mapped[list | None] = mapped_column(JSONB)
    narration_audio_dir: Mapped[str | None] = mapped_column(Text)
    clips_dir: Mapped[str | None] = mapped_column(Text)

    # Configuration overrides
    preset: Mapped[str | None] = mapped_column(String(32))
    transition_type: Mapped[str | None] = mapped_column(String(32))
    crossfade_duration: Mapped[float | None] = mapped_column(Float)
    target_width: Mapped[int | None] = mapped_column(Integer)
    target_height: Mapped[int | None] = mapped_column(Integer)
    crf: Mapped[int | None] = mapped_column(Integer)
    theme_config: Mapped[dict | None] = mapped_column(JSONB)
    edit_decisions: Mapped[dict | None] = mapped_column(JSONB)

    # Results
    output_path: Mapped[str | None] = mapped_column(Text)
    total_duration_seconds: Mapped[float | None] = mapped_column(Float)
    total_parts: Mapped[int | None] = mapped_column(Integer)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer)
    chapter_timestamps: Mapped[list | None] = mapped_column(JSONB)
    thumbnail_path: Mapped[str | None] = mapped_column(Text)

    # Metadata
    error_message: Mapped[str | None] = mapped_column(Text)
    compilation_time_seconds: Mapped[float | None] = mapped_column(Float)
    progress: Mapped[float | None] = mapped_column(Float)
    current_stage: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Enhancement columns
    compilation_report: Mapped[dict | None] = mapped_column(JSONB)
    webhook_url: Mapped[str | None] = mapped_column(Text)
    webhook_secret: Mapped[str | None] = mapped_column(Text)
    quality_metrics: Mapped[dict | None] = mapped_column(JSONB)
    filter_config: Mapped[dict | None] = mapped_column(JSONB)
    watermark_config: Mapped[dict | None] = mapped_column(JSONB)
    subtitles: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (Index("ix_compilation_jobs_tenant_status", "tenant_id", "status"),)


class CustomPreset(Base):
    """User-defined compilation presets."""

    __tablename__ = "custom_presets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    preset_id: Mapped[str] = mapped_column(
        String(64), unique=True, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    extends: Mapped[str | None] = mapped_column(String(32))
    target_width: Mapped[int | None] = mapped_column(Integer)
    target_height: Mapped[int | None] = mapped_column(Integer)
    crf: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[int | None] = mapped_column(Integer)
    transition_type: Mapped[str | None] = mapped_column(String(32))
    crossfade_duration: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_custom_presets_tenant_name"),)


class CompilationTemplate(Base):
    """Compilation templates combining preset + theme + config."""

    __tablename__ = "compilation_templates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    template_id: Mapped[str] = mapped_column(
        String(64), unique=True, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), index=True)
    name: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(Text)
    preset_name: Mapped[str | None] = mapped_column(String(128))
    theme_name: Mapped[str | None] = mapped_column(String(128))
    theme_config: Mapped[dict | None] = mapped_column(JSONB)
    music_config: Mapped[dict | None] = mapped_column(JSONB)
    title_card_config: Mapped[dict | None] = mapped_column(JSONB)
    transition_type: Mapped[str | None] = mapped_column(String(32))
    crossfade_duration: Mapped[float | None] = mapped_column(Float)
    is_builtin: Mapped[str] = mapped_column(String(5), default="false")

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_compilation_templates_tenant_name"),
    )


class CompilationBatch(Base):
    """Batch compilation job tracking."""

    __tablename__ = "compilation_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(
        String(64), unique=True, default=lambda: str(uuid.uuid7())
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    total_jobs: Mapped[int] = mapped_column(Integer, default=0)
    completed_jobs: Mapped[int] = mapped_column(Integer, default=0)
    failed_jobs: Mapped[int] = mapped_column(Integer, default=0)
    job_ids: Mapped[list[str] | None] = mapped_column(JSONB)
    errors: Mapped[list[dict[str, str]] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CompilationUsage(Base):
    """Daily usage counters per tenant for compilations."""

    __tablename__ = "compilation_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True))
    date: Mapped[str] = mapped_column(String(10))
    compilations_started: Mapped[int] = mapped_column(Integer, default=0)
    compilations_completed: Mapped[int] = mapped_column(Integer, default=0)
    compilations_failed: Mapped[int] = mapped_column(Integer, default=0)
    total_output_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    total_output_size_bytes: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        UniqueConstraint("tenant_id", "date", name="uq_compilation_usage_tenant_date"),
    )


# Phase D RLS: strict per-table policies on all 7 tenant_id tables (w2v_common) +
# the SECURITY DEFINER cross-tenant cleanup READ resolvers. Registered as after_create
# events so tests' create_all matches the 002_rls migration. Inert until the runtime
# connects as restricted w2v_video_production_app with TENANT_RLS_MODE=enforce.
register_rls_ddl(Base.metadata)

from src.database.rls_resolvers import register_resolvers  # noqa: E402

register_resolvers(Base.metadata)
