"""Give the compilation reaper a clock that can reach ``pending``, and expire ``cancelled``.

Two rows leaked, both because a status the writers actually produce appeared in
NO resolver predicate.

1. ``stale_compilation_jobs(p_max_age_s, p_limit)`` matched
   ``status IN ('processing', 'queued')``. ``'queued'`` is **never written** for
   ``compilation_jobs`` -- ``src/services/compilation/job_manager.py`` writes
   ``pending`` (:73), ``processing`` (:96), ``completed`` (:126), ``failed``
   (:175) and ``cancelled`` (:310), and nothing else. So the second half of that
   predicate matched nothing and ``'pending'`` was reachable by no clock at all.

   The window that strands one is small and real: ``src/api/routes/compilation.py``
   commits the row at ``pending`` (:212) and only then starts the background task
   (:214), whose first write is ``mark_processing``
   (``compiler_service.py``:289). A SIGTERM in between kills the task with
   ``asyncio.CancelledError`` -- a ``BaseException``, so ``process_job``'s
   ``except (OSError, RuntimeError, ValueError, TimeoutError)`` never sees it --
   and ``_shutdown_services`` (``src/asgi.py``:309-321) does not drain or fail
   registered compile tasks. The row stays ``pending`` forever and the caller's
   poller waits forever.

2. ``expired_compilation_jobs`` deletes ``completed`` and ``failed`` only, so
   every ``cancelled`` row and its storage directory were immortal. They now
   expire on the FAILED ttl: like a failure, a cancelled job's output is not a
   deliverable anyone returns for.

⚠️ The two stale windows are SEPARATE and must stay that way. ``pending`` is a
LEGITIMATE queued state whose ``updated_at`` never advances while it waits, so
that clock measures QUEUE WAIT from creation, not silence. Collapsing it onto
the 600s processing window would fail out live queued work -- exactly the
2026-09-03 recording outage (276 of 282 accepted jobs reaped while still
waiting, zero completions in 10 hours) reproduced in the compilation domain.
Revision 006 split those clocks for recordings; this is the same split for
compilations, with its own knob
(``STORAGE_COMPILATION_PENDING_STALE_SECONDS``, default 21600 = 6h).

⚠️ A changed argument list is a NEW function with a fresh, EMPTY ACL, and
``CREATE OR REPLACE`` cannot change an argument list. So EXECUTE for the new
arity is granted IN-BAND by a ``pg_roles``-guarded DO block, and the OLD arity's
grant is re-applied rather than revoked. Revision 005 exists precisely because a
resolver's grant was left to an out-of-band gitops step that was never applied,
and the reaper logged "permission denied for function" hourly while looking
healthy -- CI never sees it, because CI runs as the owner. Not repeating that.

⚠️⚠️ THE TWO-ARG OVERLOAD IS KEPT RESIDENT. DROPPING IT HERE IS AN OUTAGE.

An earlier draft of this revision opened with
``DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int)``. That is wrong, and
the reason is the live rollout spec (read from the cluster, not assumed)::

    video-production-service: replicas=1, maxSurge=25%, maxUnavailable=25%

Kubernetes rounds ``maxSurge`` UP (25% of 1 -> 1) and ``maxUnavailable`` DOWN
(25% of 1 -> 0), so the OLD pod keeps serving until the NEW pod is Ready: the two
versions OVERLAP. Alembic runs in the new pod's initContainer (``src/asgi.py``),
so this migration lands while the old pod's reaper loop is still issuing
``SELECT job_id, tenant_id, status FROM stale_compilation_jobs(:age, :lim)``
(origin/main ``src/services/compilation/cleanup_service.py``:59) every cycle.
Dropping the signature makes every one of those calls raise
``function stale_compilation_jobs(integer, integer) does not exist`` for the
length of the rollout -- a self-inflicted outage of the very reclaim path this
revision exists to restore, and a permanent one after a code-only rollback.

So both overloads are created. There is NO ambiguity hazard: Postgres resolves
overloads by ARITY first, and two candidates of different arity are never
ambiguous -- ``stale_compilation_jobs(1, 2)`` can only be the two-arg form.
Proven, not assumed, in
``tests/unit/compilation/test_reclaim_pending_compilation_jobs.py``.

The two-arg body is the ORIGINAL pre-008 one, byte for byte: one window,
``status IN ('processing', 'queued')``. Its only caller is a transient old pod
and the new pod uses the three-arg form, so re-pointing it at the new semantics
mid-rollout would be an unrequested behaviour change.

DROPPING IT IS A SAFE FOLLOW-UP PR, NOT PART OF THIS ONE: once no running pod
calls the two-arg form -- one release after this, when every replica is on the
three-arg call and no rollback target predates it -- a NEW migration can drop the
function and delete ``COMPAT_STALE_COMPILATION_2ARG_SQL``, its GRANT, and its
``_ALL_SQL`` / ``_REVOKE`` entries. Free then; an outage now.

Revision ID: 008_reclaim_pending_compile
Revises: 007_recording_queue_wait_seconds
"""

from alembic import op

from src.database.rls_resolvers import (
    COMPAT_STALE_COMPILATION_2ARG_SQL,
    EXPIRED_COMPILATION_SQL,
    GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL,
    GRANT_STALE_COMPILATION_SQL,
    STALE_COMPILATION_SQL,
)

# NOTE: alembic_version.version_num is varchar(32); a longer id fails the UPDATE
# only at the END of a successful upgrade, after every DDL has already run.
revision = "008_reclaim_pending_compile"
down_revision = "007_recording_queue_wait_seconds"
branch_labels = None
depends_on = None

# The pre-008 bodies, restored verbatim on downgrade. A downgrade to pre-008 code
# means re-creating the leak, which is what "downgrade" means here.
#
# ``_OLD_STALE_COMPILATION_SQL`` is now byte-identical to
# ``rls_resolvers.COMPAT_STALE_COMPILATION_2ARG_SQL`` -- because the compatibility
# overload IS the pre-008 body -- so on downgrade it is an idempotent no-op
# re-CREATE rather than a resurrection. It is spelled out here anyway, pinned like
# 002/003/004: a downgrade must render era-correct DDL and must not follow a live
# module that later revisions are free to change.
_OLD_STALE_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION stale_compilation_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id, cj.status FROM compilation_jobs cj
    WHERE cj.status IN ('processing', 'queued')
      AND cj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""
_OLD_EXPIRED_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION expired_compilation_jobs(
    p_completed_before timestamptz, p_failed_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id FROM compilation_jobs cj
    WHERE (cj.status = 'completed' AND cj.updated_at < p_completed_before)
       OR (cj.status = 'failed' AND cj.updated_at < p_failed_before)
    LIMIT p_limit;
  $fn$
"""
_OLD_GRANT = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""


# ⚠️ NO ``DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int)`` in this list.
# Adding one back reintroduces the rollout outage described in the module
# docstring; ``tests/unit/compilation/test_reclaim_pending_compilation_jobs.py``
# executes exactly these statements and fails if either arity goes missing.
#
# Module-level so the test can apply the REAL migration statements rather than a
# hand-copied paraphrase of them.
UPGRADE_STATEMENTS: tuple[str, ...] = (
    # The three-arg form is a NEW function object (an argument list cannot be
    # changed by CREATE OR REPLACE), so this adds an overload; it does not
    # supersede anything in place.
    STALE_COMPILATION_SQL,
    "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int, int) FROM PUBLIC",
    # New signature => fresh empty ACL => the grant MUST ship in-band (see 005).
    GRANT_STALE_COMPILATION_SQL,
    # The two-arg form stays RESIDENT for the rollout overlap. CREATE OR REPLACE so
    # the statement is idempotent, and re-grant because the old overload's EXECUTE
    # must not be lost: it is what the still-serving old pod's reaper needs.
    COMPAT_STALE_COMPILATION_2ARG_SQL,
    "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC",
    GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL,
    # Same signature, so this is a genuine in-place REPLACE and the ACL granted
    # at 002/005 survives untouched.
    EXPIRED_COMPILATION_SQL,
    "REVOKE EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int) "
    "FROM PUBLIC",
)


def upgrade() -> None:
    for stmt in UPGRADE_STATEMENTS:
        op.execute(stmt)


def downgrade() -> None:
    for stmt in (
        "DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int, int)",
        _OLD_STALE_COMPILATION_SQL,
        "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC",
        _OLD_GRANT,
        _OLD_EXPIRED_COMPILATION_SQL,
        "REVOKE EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int) "
        "FROM PUBLIC",
    ):
        op.execute(stmt)
