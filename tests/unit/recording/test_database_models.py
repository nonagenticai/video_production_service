"""Tests for src/database/models.py -- ORM model structure and defaults."""

from __future__ import annotations

from typing import cast

from sqlalchemy import Table

from src.database.models import Base, RecordingJob, RecordingUsage

# ── RecordingJob ──────────────────────────────────────────────────


class TestRecordingJob:
    def test_table_name(self):
        assert RecordingJob.__tablename__ == "recording_jobs"

    def test_column_names(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        expected = {
            "id",
            "job_id",
            "tenant_id",
            "status",
            "part_id",
            "part_type",
            "scripts_dir",
            "script_source",
            "recording_config",
            "auth_config",
            "video_path",
            "duration_seconds",
            "file_size_bytes",
            "width",
            "height",
            "has_audio",
            "error_message",
            "recording_time_seconds",
            # Execution/queue-wait split (#156): recording_time_seconds means
            # EXECUTION only, and the slot wait is stored separately here.
            "queue_wait_seconds",
            "created_at",
            "updated_at",
            "completed_at",
            # Progress & retry (WS-4)
            "progress",
            "progress_stage",
            "retry_count",
            "max_retries",
            "last_error",
            # Batch (WS-5)
            "batch_id",
            "priority",
            "batch_order",
            # Expiry (WS-7)
            "expired",
            "expires_at",
            # Webhook (WS-8)
            "webhook_url",
        }
        assert expected.issubset(col_names)

    def test_progress_column_exists(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        assert "progress" in col_names

    def test_retry_columns_exist(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        assert "retry_count" in col_names
        assert "max_retries" in col_names

    def test_batch_columns_exist(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        assert "batch_id" in col_names
        assert "priority" in col_names

    def test_expiry_columns_exist(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        assert "expired" in col_names
        assert "expires_at" in col_names

    def test_webhook_column_exists(self):
        col_names = {c.name for c in RecordingJob.__table__.columns}
        assert "webhook_url" in col_names

    def test_indexes(self):
        index_names = {idx.name for idx in cast(Table, RecordingJob.__table__).indexes}
        assert "ix_recording_jobs_tenant_status" in index_names
        assert "ix_recording_jobs_batch_id" in index_names


# ── RecordingUsage ────────────────────────────────────────────────


class TestRecordingUsage:
    def test_table_name(self):
        assert RecordingUsage.__tablename__ == "recording_usage"

    def test_unique_constraint(self):
        constraints = cast(Table, RecordingUsage.__table__).constraints
        constraint_names = {c.name for c in constraints if c.name}
        assert "uq_recording_usage_tenant_date" in constraint_names


# ── Base ──────────────────────────────────────────────────────────


class TestBase:
    def test_base_exists(self):
        assert Base is not None
        assert hasattr(Base, "metadata")
