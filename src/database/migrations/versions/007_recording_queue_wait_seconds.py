"""Split queue-wait out of ``recording_time_seconds`` with a new column.

``recording_time_seconds`` was measured from a ``time.time()`` taken BEFORE
``RecordingConcurrencyManager.acquire()`` at all five recording entry points, so
the stored "execution time" was really ``queue-wait + execution``. The fleet cap
is 3 (a Redis semaphore), so a 10-minute bucket can contain at most 1800
slot-seconds of true execution; observed reported sums in single buckets were
7885s (438% of physically possible) and 8538s (474%). Bounding the 08:10 bucket
-- 12 jobs, mean reported 711s, mean true execute <=150s -- puts >=79% of the
reported figure in queue-wait.

The application fix moves the clock to after ``acquire()`` returns. This column
is the other half: the wait is genuinely useful (it is the only thing that
distinguishes a starved fleet from a slow recorder) and dropping it would answer
the performance question by making the capacity question unanswerable.

Nullable with NO default and NO backfill. Every row written before this revision
has an inflated ``recording_time_seconds`` and no recorded split, so any
backfilled value would be fabricated -- and a fabricated 0.0 is worse than NULL,
because it reads as "this job never waited". NULL means "unmeasured", which is
the truth. Consumers must treat NULL as unknown, not zero.

Adding a nullable column with no default is a metadata-only ``ALTER TABLE`` on
Postgres 11+ -- no table rewrite, no lock held for the scan, safe on the live
table.

Revision ID: 007_recording_queue_wait_seconds
Revises: 006_split_stale_recording_clock
"""

import sqlalchemy as sa
from alembic import op

revision = "007_recording_queue_wait_seconds"
down_revision = "006_split_stale_recording_clock"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "recording_jobs",
        sa.Column("queue_wait_seconds", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("recording_jobs", "queue_wait_seconds")
