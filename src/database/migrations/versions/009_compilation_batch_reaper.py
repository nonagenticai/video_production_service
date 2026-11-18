"""Make ``compilation_batches`` reapable and prunable — the aggregate row leaked.

Revision 008 gave the CHILD ``compilation_jobs`` rows a clock that can reach every
status a writer actually produces. The AGGREGATE row was left out entirely, and it
leaks worse: ``compilation_batches`` had NO resolver, NO reaper and NO prune.

The only writer that ever advances a batch is the request-spawned
``BatchService.process_batch`` task (``src/services/compilation/batch_service.py``):
it flips the row to ``'processing'`` and, at the very end of the gather, to
``completed`` / ``partial`` / ``failed``. If the pod restarts, OOMs, or that task
raises anything the caller's handler does not catch, nothing else in the system will
ever touch the row again — and if the pod died BEFORE ``process_batch`` ran, the row
sits at ``'pending'`` instead. Either way the batch is immortal: a client polling
batch status waits forever, no webhook fires, and the failure is never accounted.

Two new SECURITY DEFINER cross-tenant READ resolvers, mirroring the compilation-job
pair and the ``reaper_service`` shape used elsewhere in the estate (reclaim +
prune backstop):

* ``stale_compilation_batches(p_max_age_s, p_limit)`` — non-terminal rows
  (``pending``/``processing``) with no batch-level progress inside the window. The
  cleanup loop then flips each to ``failed`` with a recorded reason, tenant-bound.
* ``expired_compilation_batches(p_completed_before, p_failed_before, p_limit)`` —
  terminal rows past their TTL, deleted by the existing expired sweep, so the table
  stops growing without bound. ``partial`` shares the FAILED ttl.

⚠️ ONE stale window, not the two the JOB resolvers carry. For a job, ``pending``
and ``processing`` are genuinely different measurements (queue wait from creation vs
silence since the last progress write) with different legitimate magnitudes, and
collapsing them is the 2026-09-03 outage. For a BATCH they are the same measurement:
``updated_at`` starts at creation and is bumped only when a child job finishes, so
both states mean "no batch-level progress since T" and both can only be resolved by
the single owning task. The window is therefore sized for the slowest LEGITIMATE
batch and gets its own generous knob
(``STORAGE_COMPILATION_BATCH_STALE_SECONDS``, default 21600 = 6h) rather than
borrowing the 600s no-progress window. A second knob here would be a second name for
one number and one more way for the two to drift.

⚠️ Both functions are BRAND NEW objects with fresh, EMPTY ACLs, so their EXECUTE
grants ship IN-BAND via ``pg_roles``-guarded DO blocks. Revision 005 exists because a
resolver's grant was left to an out-of-band gitops step that was never applied, and
the reaper logged "permission denied for function" hourly while looking healthy — CI
never sees it, because CI runs as the owner. Not repeating that.

There is deliberately NO compatibility overload for either signature, unlike 008:
neither function has ever existed, so no still-serving old pod can be mid-call on an
earlier arity. The rollout hazard 008 documents simply does not apply to a function
whose first version this is.

Revision ID: 009_compilation_batch_reap
Revises: 008_reclaim_pending_compile
"""

from alembic import op

from src.database.rls_resolvers import (
    EXPIRED_COMPILATION_BATCHES_SQL,
    GRANT_EXPIRED_COMPILATION_BATCHES_SQL,
    GRANT_STALE_COMPILATION_BATCHES_SQL,
    STALE_COMPILATION_BATCHES_SQL,
)

# NOTE: alembic_version.version_num is varchar(32); a longer id fails the UPDATE
# only at the END of a successful upgrade, after every DDL has already run.
revision = "009_compilation_batch_reap"
down_revision = "008_reclaim_pending_compile"
branch_labels = None
depends_on = None

# Module-level so the test can apply the REAL migration statements rather than a
# hand-copied paraphrase of them.
UPGRADE_STATEMENTS: tuple[str, ...] = (
    STALE_COMPILATION_BATCHES_SQL,
    "REVOKE EXECUTE ON FUNCTION stale_compilation_batches(int, int) FROM PUBLIC",
    GRANT_STALE_COMPILATION_BATCHES_SQL,
    EXPIRED_COMPILATION_BATCHES_SQL,
    "REVOKE EXECUTE ON FUNCTION expired_compilation_batches(timestamptz, timestamptz, int) "
    "FROM PUBLIC",
    GRANT_EXPIRED_COMPILATION_BATCHES_SQL,
)


def upgrade() -> None:
    for stmt in UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    # Downgrading to pre-009 code means restoring the leak — which is what
    # "downgrade" means here. Nothing else references these functions.
    for stmt in (
        "DROP FUNCTION IF EXISTS stale_compilation_batches(int, int)",
        "DROP FUNCTION IF EXISTS expired_compilation_batches(timestamptz, timestamptz, int)",
    ):
        op.execute(stmt)
