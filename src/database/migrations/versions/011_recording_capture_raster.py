"""Persist the raster the recorder actually captured at, not just the output size.

``AnimationRecorder`` supersamples the Chromium capture (``_CAPTURE_SCALE = 2``)
and downscales the lossless PNG sequence with lanczos before the encode, so the
stored ``width``/``height`` are the ORDERED output size and say nothing about the
surface the glyphs were rasterised on.

``_resolve_capture_scale`` then steps that factor DOWN, silently, whenever
``area * scale**2`` would exceed ``_MAX_CAPTURE_RASTER_PIXELS`` -- a 4k order
captures at 1x. Nothing outside the recorder module could observe either fact.
A caller that wants to size its source assets to the real raster (the
orchestrator's Ken Burns stage upscales stills before recording them, and
measured +75% laplacian variance / +1.9 dB doing so) therefore had exactly two
options: guess, or hardcode this service's PRIVATE constant and take a silent
quality regression the day the constant moves.

Same shape as #157/#160/010: the fact belongs to the PRODUCER, so the producer
declares it and it is persisted, rather than every consumer re-deriving it.

Stored as absolute dimensions rather than the scale factor: the consumer does no
arithmetic, cannot multiply against the wrong width, and the clamp-down is
readable directly as ``capture_width == width``.

Nullable, no default, no backfill. NULL means "not reported" -- every row written
before this column existed, plus every recorder that does not supersample. Readers
must fall back to ``width``/``height``, which is the pre-feature behaviour exactly.

Adding a nullable column with no default is a metadata-only ``ALTER TABLE`` on
Postgres 11+ -- no table rewrite, safe on the live table.

Revision ID: 011_recording_capture_raster
Revises: 010_recording_artifact_kind
"""

import sqlalchemy as sa
from alembic import op

revision = "011_recording_capture_raster"
down_revision = "010_recording_artifact_kind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("recording_jobs", sa.Column("capture_width", sa.Integer(), nullable=True))
    op.add_column("recording_jobs", sa.Column("capture_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("recording_jobs", "capture_height")
    op.drop_column("recording_jobs", "capture_width")
