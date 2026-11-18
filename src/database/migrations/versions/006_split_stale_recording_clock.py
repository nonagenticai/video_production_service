"""Give the recording reaper a SEPARATE clock for queued vs processing jobs.

Measured 2026-09-03/04 from Loki. ``stale_recording_jobs(p_max_age_s, p_limit)``
applied ONE window to ``status IN ('processing', 'queued')`` keyed on
``updated_at``. That is correct for a running job — ``mark_processing`` and
``record_progress`` bump ``updated_at``, so the window really does mean "no
progress for N seconds". It is wrong for a queued one: nothing bumps
``updated_at`` while a job waits for a fleet-wide recording slot, so the value
still holds the job's CREATION time and the window measures queue wait.

``storage.stale_job_timeout_seconds`` is 600, which EQUALS the orchestrator's
client timeout, so the reaper failed jobs at exactly the moment the caller gave
up — before they had ever been given a slot:

    21:46  Reaped 58 stale job(s)
    22:46  Reaped 53 stale job(s)
    23:46  Reaped 93 stale job(s)
    00:46  Reaped 97 stale job(s)

Of 282 jobs accepted between 21:40 and 01:30, 276 logged "Job lease NOT renewed
… no progress within the reaper's staleness window" — they were WAITING, never
RUNNING. Real recording work in the same period was 1-20s. Zero completions in
10 hours. Raising the concurrency cap alone does not fix this: any burst large
enough to build a >600s queue reproduces it exactly.

So the function takes a second age and applies it to the PRE-DISPATCH statuses:

    processing        → p_max_age_s        (no-progress window, unchanged)
    pending / queued  → p_queued_max_age_s (orphan window; owner died pre-dispatch)

``'pending'`` is in the second branch because it was in NEITHER before: jobs are
INSERTed at ``'pending'`` and ``mark_queued`` moves them on, so a pod that dies
between the INSERT and the dispatch call left a row no clock could ever reach.

⚠️ Signature change, so this is a NEW function with a fresh, empty ACL, and
EXECUTE is re-granted to the restricted runtime role in-band. Revision 005 exists
precisely because a resolver's grant was left to an out-of-band gitops step that
was never applied, and the reaper logged "permission denied for function
stale_recording_jobs" hourly while looking healthy. Not repeating that.

⚠️ The two-arg overload is KEPT, as a shim delegating to the new function. Alembic
runs in an initContainer (``src/asgi.py``), so this migration lands while the OLD
replicas (HPA floor 2) are still serving. Dropping it would give every one of them
UndefinedFunction on each reaper cycle and each heartbeat probe — and per 005,
a raising probe makes JobLease fail OPEN and renew every lease, re-arming this very
bug for the whole rollout and permanently after a code-only rollback. Two Postgres
overloads of DIFFERENT ARITY are never ambiguous, so keeping both costs nothing.
The shim can be dropped one release after this one, when no running or
rollback-reachable pod issues the two-arg call.

Revision ID: 006_split_stale_recording_clock
Revises: 005_grant_reaper_resolvers
"""

from alembic import op

from src.database.rls_resolvers import (
    COMPAT_STALE_RECORDING_2ARG_SQL,
    GRANT_COMPAT_STALE_RECORDING_2ARG_SQL,
    GRANT_STALE_RECORDING_SQL,
    STALE_RECORDING_SQL,
)

revision = "006_split_stale_recording_clock"
down_revision = "005_grant_reaper_resolvers"
branch_labels = None
depends_on = None

_REVOKE = "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int, int) FROM PUBLIC"

# On downgrade the three-arg function goes away, so the shim (which calls it)
# cannot stay — the previous two-arg body is restored verbatim in its place. That
# re-creates the outage, which is what a downgrade to pre-006 code means; leaving
# NO two-arg function would be worse still.
_OLD_STALE_RECORDING_SQL = """
CREATE OR REPLACE FUNCTION stale_recording_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id, rj.status FROM recording_jobs rj
    WHERE rj.status IN ('processing', 'queued')
      AND rj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""
_OLD_GRANT = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_recording_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""


def upgrade() -> None:
    for stmt in (
        # Order matters: the shim's body calls the three-arg function, so that must
        # exist first. CREATE OR REPLACE overwrites the old two-arg body in place —
        # same signature, so the ACL granted at 002/005 survives, and the extra
        # GRANT below is belt-and-braces for a database that never had it.
        STALE_RECORDING_SQL,
        _REVOKE,
        GRANT_STALE_RECORDING_SQL,
        COMPAT_STALE_RECORDING_2ARG_SQL,
        "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int) FROM PUBLIC",
        GRANT_COMPAT_STALE_RECORDING_2ARG_SQL,
    ):
        op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        # The shim first (it depends on the three-arg function), then the function.
        "DROP FUNCTION IF EXISTS stale_recording_jobs(int, int)",
        "DROP FUNCTION IF EXISTS stale_recording_jobs(int, int, int)",
        _OLD_STALE_RECORDING_SQL,
        "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int) FROM PUBLIC",
        _OLD_GRANT,
    ):
        op.execute(stmt)
