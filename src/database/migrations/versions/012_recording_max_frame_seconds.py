"""Persist the slowest single frame's capture time for an animation render.

The frame-step capture loop's stall watchdog
(``animation_render_stall_window_seconds``) was sized from job-level AVERAGES
(``recording_time_seconds / estimated_frame_count``), not real per-frame
telemetry -- there was no way to know how slow the single worst frame in a job
actually was, only the mean across the whole job. An average cannot bound a
single outlier: a job with hundreds of frames can absorb one very slow frame
and still report an unremarkable mean.

``AnimationRecorder._capture_frames`` now times every frame's own capture
(JS timeline seek + screenshot) and keeps the maximum. This column persists it
so the stall window (and the per-frame budget it is justified from) can
eventually be tuned from real distributions instead of averages. See
``Settings.animation_render_stall_window_seconds`` for the full reasoning.

Nullable, no default, no backfill: no recorder before this measured a
per-frame max, so every existing row's true value is unknown, not zero. NULL
also covers every non-animation recording job (content/url recorders don't do
frame-step capture at all).

Adding a nullable column with no default is a metadata-only ``ALTER TABLE`` on
Postgres 11+ -- no table rewrite, safe on the live table.

Revision ID: 012_recording_max_frame_seconds
Revises: 011_recording_capture_raster
"""

import sqlalchemy as sa
from alembic import op

revision = "012_recording_max_frame_seconds"
down_revision = "011_recording_capture_raster"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("recording_jobs", sa.Column("max_frame_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("recording_jobs", "max_frame_seconds")
