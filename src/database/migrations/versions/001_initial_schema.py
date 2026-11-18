"""Initial schema for video_production_service.

Revision ID: 001
Revises:
Create Date: 2026-03-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Recording Domain ─────────────────────────────────────────────

    op.create_table(
        "recording_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("part_id", sa.String(128), nullable=False),
        sa.Column("part_type", sa.String(64), nullable=True),
        sa.Column("scripts_dir", sa.Text(), nullable=True),
        sa.Column("script_source", sa.Text(), nullable=True),
        sa.Column("recording_config", postgresql.JSONB(), nullable=True),
        sa.Column("auth_config", postgresql.JSONB(), nullable=True),
        sa.Column("video_path", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("has_audio", sa.Boolean(), nullable=True, server_default="false"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("recording_time_seconds", sa.Float(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=True, server_default="0.0"),
        sa.Column("progress_stage", sa.String(64), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("max_retries", sa.Integer(), nullable=True, server_default="2"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("batch_id", sa.String(64), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("batch_order", sa.Integer(), nullable=True),
        sa.Column("expired", sa.Boolean(), nullable=True, server_default="false"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_recording_jobs_tenant_id", "recording_jobs", ["tenant_id"])
    op.create_index("ix_recording_jobs_batch_id", "recording_jobs", ["batch_id"])
    op.create_index("ix_recording_jobs_tenant_status", "recording_jobs", ["tenant_id", "status"])

    op.create_table(
        "recording_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("recordings_started", sa.Integer(), server_default="0"),
        sa.Column("recordings_completed", sa.Integer(), server_default="0"),
        sa.Column("recordings_failed", sa.Integer(), server_default="0"),
        sa.Column("total_duration_seconds", sa.Float(), server_default="0.0"),
        sa.Column("total_file_size_bytes", sa.Integer(), server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "date", name="uq_recording_usage_tenant_date"),
    )

    # ── Compilation Domain ───────────────────────────────────────────

    op.create_table(
        "compilation_jobs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("shot_list", postgresql.JSONB(), nullable=True),
        sa.Column("recordings", postgresql.JSONB(), nullable=True),
        sa.Column("narration_audio_dir", sa.Text(), nullable=True),
        sa.Column("clips_dir", sa.Text(), nullable=True),
        sa.Column("preset", sa.String(32), nullable=True),
        sa.Column("transition_type", sa.String(32), nullable=True),
        sa.Column("crossfade_duration", sa.Float(), nullable=True),
        sa.Column("target_width", sa.Integer(), nullable=True),
        sa.Column("target_height", sa.Integer(), nullable=True),
        sa.Column("crf", sa.Integer(), nullable=True),
        sa.Column("theme_config", postgresql.JSONB(), nullable=True),
        sa.Column("edit_decisions", postgresql.JSONB(), nullable=True),
        sa.Column("output_path", sa.Text(), nullable=True),
        sa.Column("total_duration_seconds", sa.Float(), nullable=True),
        sa.Column("total_parts", sa.Integer(), nullable=True),
        sa.Column("file_size_bytes", sa.Integer(), nullable=True),
        sa.Column("chapter_timestamps", postgresql.JSONB(), nullable=True),
        sa.Column("thumbnail_path", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("compilation_time_seconds", sa.Float(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=True),
        sa.Column("current_stage", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("compilation_report", postgresql.JSONB(), nullable=True),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("webhook_secret", sa.Text(), nullable=True),
        sa.Column("quality_metrics", postgresql.JSONB(), nullable=True),
        sa.Column("filter_config", postgresql.JSONB(), nullable=True),
        sa.Column("watermark_config", postgresql.JSONB(), nullable=True),
        sa.Column("subtitles", postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id"),
    )
    op.create_index("ix_compilation_jobs_tenant_id", "compilation_jobs", ["tenant_id"])
    op.create_index(
        "ix_compilation_jobs_tenant_status", "compilation_jobs", ["tenant_id", "status"]
    )

    op.create_table(
        "custom_presets",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("preset_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("extends", sa.String(32), nullable=True),
        sa.Column("target_width", sa.Integer(), nullable=True),
        sa.Column("target_height", sa.Integer(), nullable=True),
        sa.Column("crf", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Integer(), nullable=True),
        sa.Column("transition_type", sa.String(32), nullable=True),
        sa.Column("crossfade_duration", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("preset_id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_custom_presets_tenant_name"),
    )
    op.create_index("ix_custom_presets_tenant_id", "custom_presets", ["tenant_id"])

    op.create_table(
        "compilation_templates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("template_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("preset_name", sa.String(128), nullable=True),
        sa.Column("theme_name", sa.String(128), nullable=True),
        sa.Column("theme_config", postgresql.JSONB(), nullable=True),
        sa.Column("music_config", postgresql.JSONB(), nullable=True),
        sa.Column("title_card_config", postgresql.JSONB(), nullable=True),
        sa.Column("transition_type", sa.String(32), nullable=True),
        sa.Column("crossfade_duration", sa.Float(), nullable=True),
        sa.Column("is_builtin", sa.String(5), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("template_id"),
        sa.UniqueConstraint("tenant_id", "name", name="uq_compilation_templates_tenant_name"),
    )
    op.create_index("ix_compilation_templates_tenant_id", "compilation_templates", ["tenant_id"])

    op.create_table(
        "compilation_batches",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("batch_id", sa.String(64), nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("total_jobs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_jobs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_jobs", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("job_ids", postgresql.JSONB(), nullable=True),
        sa.Column("errors", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("batch_id"),
    )
    op.create_index("ix_compilation_batches_tenant_id", "compilation_batches", ["tenant_id"])

    op.create_table(
        "compilation_usage",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(128), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),
        sa.Column("compilations_started", sa.Integer(), server_default="0"),
        sa.Column("compilations_completed", sa.Integer(), server_default="0"),
        sa.Column("compilations_failed", sa.Integer(), server_default="0"),
        sa.Column("total_output_duration_seconds", sa.Float(), server_default="0.0"),
        sa.Column("total_output_size_bytes", sa.Integer(), server_default="0"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "date", name="uq_compilation_usage_tenant_date"),
    )


def downgrade() -> None:
    op.drop_table("compilation_usage")
    op.drop_table("compilation_batches")
    op.drop_table("compilation_templates")
    op.drop_table("custom_presets")
    op.drop_table("compilation_jobs")
    op.drop_table("recording_usage")
    op.drop_table("recording_jobs")
