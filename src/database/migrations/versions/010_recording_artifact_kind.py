"""Persist the artifact KIND a recorder declared, so the serving path can dispatch on it.

Every recorder returns its output under ``video_path``, but two of the five
producers (``create_screens_job`` -> ``UrlRecorder.capture_screens`` /
``capture_screen_clips``) put a **zip** of per-screen stills/clips there. #160
made the completion path honest about that: the producer declares
``artifact_kind`` and ``_verify_artifact`` dispatches on it, verifying a zip as
a zip instead of vetoing it with ffprobe.

But the declaration was never written to the row, so it died at completion. The
download route (``GET /jobs/{id}/video``, documented in
``schemas/recording.py`` as the screens download path) had nothing to dispatch
on and fell back to mapping the stored file suffix through a video-only table:
``{"mp4", "webm", "gif"}.get(ext, "video/webm")`` has no ``"zip"`` key, so a
screens archive was served to the client as ``Content-Type: video/webm``.

A suffix sniff at the route would paper over that; this column fixes it, because
"what is this artifact" is a fact the PRODUCER knows and no consumer should be
re-deriving. Same shape as #157/#160, now closed on the serving side too.

Nullable, no default, no backfill. NULL means "written before this column
existed". Readers must resolve NULL to ``"video"`` -- keeping #160's principle
that the strict/video case is the default, so a producer (or a historical row)
that declares nothing is unaffected.

Adding a nullable column with no default is a metadata-only ``ALTER TABLE`` on
Postgres 11+ -- no table rewrite, safe on the live table.

Revision ID: 010_recording_artifact_kind
Revises: 009_compilation_batch_reap
"""

import sqlalchemy as sa
from alembic import op

revision = "010_recording_artifact_kind"
down_revision = "009_compilation_batch_reap"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "recording_jobs",
        sa.Column("artifact_kind", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("recording_jobs", "artifact_kind")
