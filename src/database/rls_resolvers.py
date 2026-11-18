"""SECURITY DEFINER cross-tenant READ resolvers for the cleanup loops (single source).

Both cleanup services (recording + compilation) run at app boot with no tenant
context and sweep jobs across ALL tenants. Under Phase D RLS enforce the runtime is a
non-bypass role with no ``app.current_tenant`` bound, so their SELECTs would
fail-closed to 0 rows and nothing would ever be cleaned.

Rather than re-implement the nuanced ``mark_expired`` / ``mark_failed`` / ``delete_jobs``
mutation logic in SQL, these resolvers only do the cross-tenant *read* — owned by the
table owner so they bypass ENABLE — returning ``(job_id, tenant_id)``. The cleanup
service then binds the GUC per tenant (``set_identity_context`` + the after_begin
listener) and reuses the existing, tested JobManager mutation methods, which run
tenant-scoped and therefore work under enforce.

EXECUTE is REVOKEd from PUBLIC here and GRANTed to the app role out-of-band in Stage A
(the app role does not exist in CI). Single-sourced so BOTH the migration (prod) and
the ``create_all`` ``after_create`` event (tests) create identical functions.
"""

from __future__ import annotations

from sqlalchemy import Connection, MetaData, event, text

# Recording: completed/failed/cancelled jobs past expiry, not yet flagged expired.
EXPIRED_RECORDING_SQL = """
CREATE OR REPLACE FUNCTION expired_recording_jobs(p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id uuid)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id FROM recording_jobs rj
    WHERE rj.expires_at <= now()
      AND rj.expired IS false
      AND rj.status IN ('completed', 'failed', 'cancelled')
    LIMIT p_limit;
  $fn$
"""

# Recording: jobs stuck past their max_age -- with a SEPARATE clock per status.
#
# ⚠️ The two windows are NOT interchangeable, and collapsing them back into one is
# the 2026-09-03 outage. `updated_at` means "last progress" for a job that is
# RUNNING (mark_processing and record_progress bump it). For a job that is merely
# WAITING it means something else entirely: `mark_queued` stamps `updated_at=now()`
# at the pending -> queued transition and then NOTHING bumps it again while the job
# waits for a fleet-wide slot, so the value is "the moment this job was handed to
# the queue" (for 'pending' rows, which mark_queued has not reached, it is still
# the creation time). Either way the clock does not advance while the job waits, so
# reaping waiters on the no-progress window fails jobs for the crime of queueing:
# 276 of 282 jobs accepted in one 4-hour window were reaped while still waiting,
# and the service completed nothing for 10 hours.
#
# So: 'processing' keeps the tight no-progress window (p_max_age_s), and the two
# PRE-DISPATCH statuses share their own, far longer orphan window
# (p_queued_max_age_s) whose only job is to collect rows whose owning pod died
# before the recording ever started. Fail-fast on saturation is the concurrency
# manager's queue_wait_timeout, not this.
#
# 'pending' is in that second branch DELIBERATELY. `create_job` inserts at
# status='pending' and `mark_queued` moves it on; a pod that dies between the
# INSERT and the dispatch call leaves the row at 'pending' forever, and before this
# it was covered by NO clock at all -- a genuine orphan leak, and the exact row
# `storage.queued_job_timeout_seconds` claims to collect. It shares the queued
# window rather than getting a third one because it is the same failure (owner died
# pre-dispatch) measured from the same non-advancing timestamp; a separate knob
# would be a second name for one number and one more way for the two to drift.
STALE_RECORDING_SQL = """
CREATE OR REPLACE FUNCTION stale_recording_jobs(
    p_max_age_s int, p_queued_max_age_s int, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id, rj.status FROM recording_jobs rj
    WHERE (
            rj.status = 'processing'
            AND rj.updated_at <= now() - make_interval(secs => p_max_age_s)
          )
       OR (
            rj.status IN ('pending', 'queued')
            AND rj.updated_at <= now() - make_interval(secs => p_queued_max_age_s)
          )
    LIMIT p_limit;
  $fn$
"""

# ⚠️ COMPATIBILITY SHIM -- the pre-2026-09-04 two-argument overload, kept ALIVE.
#
# Alembic runs in an initContainer (src/asgi.py), so the NEW pod migrates the
# database while the OLD replicas are still serving (the HPA floor is 2). Dropping
# the two-arg overload would make every one of those pods raise UndefinedFunction
# on every reaper cycle AND on every `stalled_recording_job_ids` heartbeat probe --
# and per revision 005, when that probe raises, JobLease fails OPEN and renews
# every lease, re-arming the precise bug this change exists to kill, for the whole
# rollout and permanently after a code-only rollback.
#
# There is no ambiguity hazard in keeping both: Postgres resolves overloads by
# ARITY first, and two candidates of different arity are never ambiguous. The
# earlier claim that they would be is simply wrong.
#
# The shim delegates to the three-arg function and supplies the queued window from
# the same 7200s default the config carries, so an old pod calling it behaves
# strictly BETTER than it did before (it stops reaping waiters) and never worse.
#
# WHEN TO DROP: once no running pod calls the two-arg form -- i.e. one release
# after this one, when every replica is on the three-arg call and rollback targets
# no longer predate it. Then delete this block, its GRANT, and its `_ALL_SQL` /
# `_REVOKE` / `_DROP` entries in a new migration.
#
# The 7200 below is written out LITERALLY rather than interpolated from
# `StorageSettings.queued_job_timeout_seconds`: an f-string here trips ruff S608
# (string-based query construction), and suppressing S608 on DDL would train the
# eye to ignore it where it does matter. Migration 005 makes the same call. If that
# default ever moves, this constant is dead code by then anyway -- see WHEN TO DROP.
COMPAT_STALE_RECORDING_2ARG_SQL = """
CREATE OR REPLACE FUNCTION stale_recording_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT * FROM stale_recording_jobs(p_max_age_s, 7200, p_limit);
  $fn$
"""

# The shim is a distinct function object, so it needs its own EXECUTE grant -- the
# three-arg grant does not cover it, and an ungranted shim is just a slower way of
# reproducing the permission-denied outage that revision 005 exists to fix.
GRANT_COMPAT_STALE_RECORDING_2ARG_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_recording_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""

# Self-contained conditional grant for the re-signatured recording reaper resolver.
# Migration 005 exists because three resolvers relied on an out-of-band gitops grant
# that was never applied in prod, and the reaper hit "permission denied for function"
# hourly. A NEW signature is a NEW function with a fresh, empty ACL, so it would
# reproduce that bug exactly unless the grant ships in-band. It does, here.
GRANT_STALE_RECORDING_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_recording_jobs(int, int, int) TO w2v_video_production_app;
  END IF;
END $$
"""

# Compilation: terminal jobs past their TTL.
#
# 'cancelled' shares the FAILED ttl. It is a terminal state written by
# ``CompilationJobManager.cancel_job`` and, before 2026-09-05, it appeared in NO
# branch of this function — so a cancelled row and its storage directory were
# immortal, accumulating for the lifetime of the database. It belongs on the
# failed clock rather than the completed one because, like a failure, its output
# is not a deliverable anybody will come back for; only ``completed`` carries the
# long retention because that artifact is the product.
EXPIRED_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION expired_compilation_jobs(
    p_completed_before timestamptz, p_failed_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id FROM compilation_jobs cj
    WHERE (cj.status = 'completed' AND cj.updated_at < p_completed_before)
       OR (cj.status IN ('failed', 'cancelled') AND cj.updated_at < p_failed_before)
    LIMIT p_limit;
  $fn$
"""

# Compilation: jobs stuck past max_age -- with a SEPARATE clock per status.
#
# ⚠️ The two windows are NOT interchangeable, and collapsing them into one is a
# REGRESSION, not a simplification. It is the same defect the recording resolver
# above carries a long comment about, and it would fail out live queued work.
#
# 'processing' keeps the tight no-progress window (p_max_age_s): ``mark_processing``
# and the progress writes bump ``updated_at``, so there the window really does mean
# "this job started and then went silent".
#
# 'pending' gets its own, far longer orphan window (p_pending_max_age_s). It is a
# LEGITIMATE queued state -- ``POST /api/compile`` commits the row at 'pending'
# (src/api/routes/compilation.py) and only then hands it to a background task, and
# nothing bumps ``updated_at`` while it waits, so this clock runs from CREATION and
# measures queue wait, not silence. Reaping it on the processing window would fail
# jobs for the crime of queueing.
#
# 'pending' is here at all because before 2026-09-05 it was covered by NO clock:
# the predicate was ``status IN ('processing', 'queued')`` and 'queued' is NEVER
# written for compilation_jobs (the writers are pending/processing/completed/
# failed/cancelled). A pod that took SIGTERM in the window between the commit and
# the first ``mark_processing`` write left the row at 'pending' FOREVER --
# ``asyncio.CancelledError`` is a BaseException, so the background task's
# ``except (OSError, RuntimeError, ValueError, TimeoutError)`` never saw it, and
# ``_shutdown_services`` does not drain or fail registered compile tasks. The
# caller's poller then waited on a job nothing would ever move again.
#
# 'queued' stays in the second branch purely defensively: it costs nothing and
# means a future writer of that status is covered from day one rather than
# leaking exactly like 'pending' did.
STALE_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION stale_compilation_jobs(
    p_max_age_s int, p_pending_max_age_s int, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id, cj.status FROM compilation_jobs cj
    WHERE (
            cj.status = 'processing'
            AND cj.updated_at <= now() - make_interval(secs => p_max_age_s)
          )
       OR (
            cj.status IN ('pending', 'queued')
            AND cj.updated_at <= now() - make_interval(secs => p_pending_max_age_s)
          )
    LIMIT p_limit;
  $fn$
"""

# ⚠️ COMPATIBILITY OVERLOAD -- the pre-2026-09-05 TWO-argument form, kept RESIDENT
# ALONGSIDE the three-arg one. Dropping it is a self-inflicted outage during every
# rollout, and the measured reason is the live rollout spec, not a hypothetical:
#
#     video-production-service: replicas=1, maxSurge=25%, maxUnavailable=25%
#
# Kubernetes rounds maxSurge UP (25% of 1 -> 1) and maxUnavailable DOWN (25% of 1
# -> 0), so the OLD pod keeps serving until the NEW pod is Ready: the two versions
# OVERLAP for the whole rollout. Alembic runs in the new pod's initContainer
# (src/asgi.py), so the migration lands while the old pod's reaper is still issuing
#
#     SELECT job_id, tenant_id, status FROM stale_compilation_jobs(:age, :lim)
#
# on its periodic loop (origin/main src/services/compilation/cleanup_service.py:59).
# A DROP would make every one of those calls raise
# ``function stale_compilation_jobs(integer, integer) does not exist`` for the
# length of the rollout -- taking down the very reclaim path this revision exists
# to fix, and permanently so after a code-only rollback.
#
# There is NO ambiguity hazard in keeping both. Postgres resolves overloads by
# ARITY first, and two candidates with different numbers of arguments are never
# ambiguous: ``stale_compilation_jobs(1, 2)`` can only be the two-arg form and
# ``stale_compilation_jobs(1, 2, 3)`` only the three-arg one. (Proven by calling
# both in tests/unit/compilation/test_reclaim_pending_compilation_jobs.py.)
#
# The body is the ORIGINAL pre-008 body, unchanged: one window, matching
# ``processing``/``queued`` only. That is deliberate -- the only caller is a
# transient old pod, the new pod uses the three-arg form, and re-pointing the old
# call site at new semantics mid-rollout would be a behaviour change nobody asked
# for. ``queued`` is never written for compilation_jobs, so in practice it reaps
# exactly what it always reaped: silent ``processing`` rows.
#
# WHEN TO DROP -- a SAFE FOLLOW-UP PR: once no running pod calls the two-arg form,
# i.e. one release after this one, when every replica is on the three-arg call and
# any rollback target already predates nothing. Then delete this block, its GRANT,
# and its ``_ALL_SQL`` / ``_REVOKE`` entries in a NEW migration. Doing it in THIS
# migration is the outage; doing it in the next one is free.
COMPAT_STALE_COMPILATION_2ARG_SQL = """
CREATE OR REPLACE FUNCTION stale_compilation_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id, cj.status FROM compilation_jobs cj
    WHERE cj.status IN ('processing', 'queued')
      AND cj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""

# The compatibility overload is a DISTINCT function object with its OWN ACL, so it
# needs its own EXECUTE grant: the three-arg grant does not cover it. Note this is
# not merely "re-granting what was already there" -- migration 004 dropped and
# re-created the compilation resolvers for the uuid type flip, so this two-arg
# function is itself young. An ungranted overload is a slower way of reproducing
# revision 005's "permission denied for function" outage, invisible in CI (which
# runs as the DB owner) and hourly in prod.
GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""

# Self-contained conditional grant for the NEW compilation reaper resolver.
# The runtime connects as the restricted, non-owner ``w2v_video_production_app``
# role under TENANT_RLS_MODE=enforce; without EXECUTE it hits "permission denied
# for function" and the reaper silently no-ops (CI is blind — it runs as the DB
# owner, which can always execute its own SECURITY DEFINER function). We do NOT
# rely on the gitops stage-a grant file for this one — the DO block grants in-band
# when the role exists (prod) and no-ops when it is absent (CI). Mirrors
# slide_service's reaper_sql pattern.
#
# ⚠️ The arity here MUST track STALE_COMPILATION_SQL. A re-signatured function is
# a NEW function object with a fresh, EMPTY ACL: granting the old two-arg form
# would leave the three-arg one ungranted and reproduce revision 005's
# "permission denied for function" outage, invisibly, because CI runs as owner.
# This grant is IN ADDITION TO, never INSTEAD OF,
# GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL -- both overloads are resident during a
# rollout and each needs EXECUTE.
GRANT_STALE_COMPILATION_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_jobs(int, int, int) TO w2v_video_production_app;
  END IF;
END $$
"""

# Compilation BATCHES: the aggregate row, stuck past its orphan window.
#
# ``compilation_batches`` had NO resolver, NO reaper and NO prune at all: the only
# writer that ever advances a batch past 'processing' is the request-spawned
# ``BatchService.process_batch`` task (src/services/compilation/batch_service.py).
# If the pod restarts, OOMs, or that task raises anything the caller's handler does
# not catch, the row stays 'processing' — or 'pending', if the pod died before
# ``process_batch`` ran at all — FOREVER. A client polling batch status then waits
# forever, no webhook fires, and the failure is never accounted. That is the same
# class of leak revision 008 closed for the CHILD ``compilation_jobs`` rows; the
# aggregate was simply left out.
#
# ⚠️ ONE window here, not the two the JOB resolvers carry — and it is a WALL-CLOCK
# CEILING, not a no-progress window. ``process_batch`` never commits between the
# 'processing' flip and the final status write (per-job progress is a bare
# ``flush()``), so from THIS function's connection ``updated_at`` is frozen at batch
# start however much progress the batch is making: this predicate cannot distinguish
# a progressing batch from a dead one, and one window is all the data supports.
# Live work is protected by the ROW LOCK instead — the caller reaps with SKIP LOCKED
# and the owning transaction holds the row from its first child flush onward. The
# knob (``STORAGE_COMPILATION_BATCH_STALE_SECONDS``, default 21600 = 6h) is therefore
# "the longest batch we allow", not a silence threshold. Real no-progress detection
# would need a COMMITTED heartbeat on the row; that has NOT been built.
STALE_COMPILATION_BATCHES_SQL = """
CREATE OR REPLACE FUNCTION stale_compilation_batches(p_max_age_s int, p_limit int)
  RETURNS TABLE(batch_id varchar, tenant_id uuid, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cb.batch_id, cb.tenant_id, cb.status FROM compilation_batches cb
    WHERE cb.status IN ('pending', 'processing')
      AND cb.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""

# Compilation BATCHES: terminal rows past their TTL — the prune backstop.
#
# 'partial' shares the FAILED ttl: like 'failed' it records that something in the
# batch did not produce a deliverable, and only a clean 'completed' earns the long
# retention. Without this the batch table grows without bound even once the reaper
# above stops rows being immortal in a NON-terminal state.
EXPIRED_COMPILATION_BATCHES_SQL = """
CREATE OR REPLACE FUNCTION expired_compilation_batches(
    p_completed_before timestamptz, p_failed_before timestamptz, p_limit int
) RETURNS TABLE(batch_id varchar, tenant_id uuid)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cb.batch_id, cb.tenant_id FROM compilation_batches cb
    WHERE (cb.status = 'completed' AND cb.updated_at < p_completed_before)
       OR (cb.status IN ('failed', 'partial') AND cb.updated_at < p_failed_before)
    LIMIT p_limit;
  $fn$
"""

# Both batch resolvers are BRAND NEW function objects with fresh, EMPTY ACLs, so
# their EXECUTE grants must ship in-band exactly like revision 005's — the runtime
# connects as the non-owner ``w2v_video_production_app`` under
# TENANT_RLS_MODE=enforce, and CI is blind to a missing grant because CI runs as
# the owner. There is no compatibility overload to keep resident for either: no
# released pod has ever called a function that did not exist.
GRANT_STALE_COMPILATION_BATCHES_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_batches(int, int)
      TO w2v_video_production_app;
  END IF;
END $$
"""

GRANT_EXPIRED_COMPILATION_BATCHES_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION
      expired_compilation_batches(timestamptz, timestamptz, int)
      TO w2v_video_production_app;
  END IF;
END $$
"""

_ALL_SQL = (
    EXPIRED_RECORDING_SQL,
    STALE_RECORDING_SQL,
    # MUST follow STALE_RECORDING_SQL: the shim's body calls the three-arg function.
    COMPAT_STALE_RECORDING_2ARG_SQL,
    EXPIRED_COMPILATION_SQL,
    STALE_COMPILATION_SQL,
    # Resident ALONGSIDE the three-arg form for rollout overlap -- see the block
    # above. Unlike the recording shim this one does not delegate, so its position
    # here carries no ordering constraint.
    COMPAT_STALE_COMPILATION_2ARG_SQL,
    # The batch aggregate resolvers (new 2026-09-06). No compat overload for
    # either: neither signature has ever existed before, so no running pod can be
    # mid-call on an older one.
    STALE_COMPILATION_BATCHES_SQL,
    EXPIRED_COMPILATION_BATCHES_SQL,
)
_DROP = (
    "DROP FUNCTION IF EXISTS expired_recording_jobs(int)",
    # The two-arg compatibility shim depends on the three-arg function, so it is
    # dropped FIRST. (Both signatures coexist by design -- see the shim above.)
    "DROP FUNCTION IF EXISTS stale_recording_jobs(int, int)",
    "DROP FUNCTION IF EXISTS stale_recording_jobs(int, int, int)",
    "DROP FUNCTION IF EXISTS expired_compilation_jobs(timestamptz, timestamptz, int)",
    # Both compilation arities: the pre-2026-09-05 two-arg form and the current
    # three-arg one, so a downgrade never leaves an orphaned overload resident.
    "DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int)",
    "DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int, int)",
    "DROP FUNCTION IF EXISTS stale_compilation_batches(int, int)",
    "DROP FUNCTION IF EXISTS expired_compilation_batches(timestamptz, timestamptz, int)",
)
_REVOKE = (
    "REVOKE EXECUTE ON FUNCTION expired_recording_jobs(int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_compilation_batches(int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION expired_compilation_batches(timestamptz, timestamptz, int) "
    "FROM PUBLIC",
)


def upgrade_statements() -> list[str]:
    return [*_ALL_SQL, *_REVOKE]


def downgrade_statements() -> list[str]:
    return list(_DROP)


def stale_compilation_statements() -> list[str]:
    """Single source for migration 003 AND the test after_create path: create +
    revoke-from-PUBLIC + the pg_roles-guarded GRANT to the restricted runtime role.

    Self-contained (does NOT depend on the gitops stage-a grant file) so the
    compilation reaper can actually EXECUTE the resolver in prod under enforce.

    ⚠️ There is deliberately NO ``DROP FUNCTION ... (int, int)`` here. The two
    overloads are resident TOGETHER so an old replica can keep calling the two-arg
    form for the length of a rollout -- see COMPAT_STALE_COMPILATION_2ARG_SQL for
    the measured rollout-spec reason and for when it is safe to drop.
    """
    return [
        # CREATE OR REPLACE cannot change an argument list, so the three-arg form is
        # a genuinely NEW function object rather than a replacement of the two-arg
        # one -- which is exactly why both can, and must, coexist here.
        STALE_COMPILATION_SQL,
        "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int, int) FROM PUBLIC",
        GRANT_STALE_COMPILATION_SQL,
        COMPAT_STALE_COMPILATION_2ARG_SQL,
        "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC",
        GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL,
    ]


def register_resolvers(metadata: MetaData) -> None:
    """Create the read resolvers after the tables exist (test create_all path).

    Applies the base creates for every resolver plus the self-contained guarded
    GRANT for ``stale_compilation_jobs`` (the DO block no-ops in CI where the
    ``w2v_video_production_app`` role is absent), so the test path applies the
    same new-function DDL as migration 003.
    """

    def _apply(target: object, connection: Connection, **kw: object) -> None:
        for sql in (
            # No DROP of the two-arg compilation overload: BOTH arities are created,
            # because both must be resident during a rollout (see
            # COMPAT_STALE_COMPILATION_2ARG_SQL). ``_ALL_SQL`` creates both.
            *_ALL_SQL,
            GRANT_STALE_COMPILATION_SQL,
            GRANT_COMPAT_STALE_COMPILATION_2ARG_SQL,
            GRANT_STALE_RECORDING_SQL,
            GRANT_COMPAT_STALE_RECORDING_2ARG_SQL,
            GRANT_STALE_COMPILATION_BATCHES_SQL,
            GRANT_EXPIRED_COMPILATION_BATCHES_SQL,
        ):
            connection.execute(text(sql))

    event.listen(metadata, "after_create", _apply)
