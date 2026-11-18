"""GRANT EXECUTE on the three cleanup resolvers the app role could not call.

Revision 003 fixed this for ``stale_compilation_jobs`` and its docstring states the
failure mode exactly: without an in-band grant the reaper, "which runs as the
non-owner app role under ``TENANT_RLS_MODE=enforce``, would hit 'permission denied
for function' and silently no-op in prod".

The other three resolvers were left relying on an out-of-band gitops stage-a grant.
That grant is NOT present in production. Measured 2026-08-26 on the write primary:

    proname                   | proacl                          | app can execute
    stale_compilation_jobs    | ...,w2v_video_production_app=X  | t   <- granted in 003
    stale_recording_jobs      | w2v_service=X/w2v_service       | f
    expired_recording_jobs    | w2v_service=X/w2v_service       | f
    expired_compilation_jobs  | w2v_service=X/w2v_service       | f

and the live pods logged, hourly:

    ERROR - Stale job reaper cycle failed
    psycopg.errors.InsufficientPrivilege: permission denied for function stale_recording_jobs

So the recording stale-job reaper, the recording expiry sweep and the compilation
expiry sweep have all been aborting every cycle before reclaiming anything. The one
resolver whose GRANT lives in code works; the three that depended on someone
remembering a manual step do not.

This also silently disarmed the heartbeat fix shipped in #139: its progress probe
calls ``stale_recording_jobs``, the InsufficientPrivilege propagates, and JobLease
fails OPEN — renewing every lease exactly as before. The fix was live and inert.

GRANT only; no CREATE. The functions were re-created from the live module at
revision 004, so the era-signature hazard documented in 003 does not apply here.
Each grant is wrapped in a ``pg_roles``-guarded DO block, as 003 is, so it grants in
prod and no-ops in CI, where the role is absent and the owner executes anyway — and,
because that makes the statement one CI never executes, in a signature guard as well.
See the comment above the grants for why both are mandatory.
Re-running is harmless: GRANT is idempotent.

Revision ID: 005_grant_reaper_resolvers
Revises: 004_tenant_id_uuid
"""

from alembic import op

revision = "005_grant_reaper_resolvers"
down_revision = "004_tenant_id_uuid"
branch_labels = None
depends_on = None

# Written out literally rather than interpolated from a list. Migration 003 does the
# same, and a formatted SQL string trips ruff S608 (string-based query construction) —
# suppressing that on a GRANT would train the eye to ignore it where it matters.
# ⚠️ WHY EVERY GRANT BELOW CARRIES **TWO** GUARDS — role AND signature.
#
# A role-guarded DDL statement is an UNTESTED statement. CI's ephemeral postgres has
# no ``w2v_video_production_app`` role, so the whole DO body is skipped and no test
# in this repo ever executes the GRANT. It runs for the first time in production.
# ``content_service`` shipped exactly this shape and the grant aborted deterministically
# in prod with "function ... does not exist": 85 init-container restarts and a
# six-hour un-rollable service, because an aborted statement rolls back the entire
# migration transaction. Any GRANT on a signature that could drift MUST therefore also
# assert the signature exists, so a drifted arity degrades to a skipped grant (loud,
# recoverable) instead of a wedged upgrade.
#
# The drift path is real, not hypothetical. 004's UPGRADE renders the resolvers from
# the LIVE module (``rls_resolvers.upgrade_statements()``), and ``register_resolvers``
# does the same on ``create_all``. Both produce whatever signatures the module carries
# TODAY, not the ones this revision was written against. 006 already added a third
# argument to the stale resolvers; the expiry resolvers can move the same way.
#
# ⚠️ The guard predicate is ``to_regprocedure(...) IS NOT NULL``, NOT a comparison
# against ``pg_get_function_identity_arguments``. That function renders argument
# NAMES as well as types — the real value for the two-arg stale resolver is
# ``'p_max_age_s integer, p_limit integer'``, never ``'integer, integer'`` — so the
# string comparison this revision originally used was ALWAYS FALSE and silently
# suppressed its own grant, reproducing the permission-denied bug the revision exists
# to fix. Verified on postgres 18.6 against the live resolver set.
# ``to_regprocedure`` parses a signature by TYPE, is name-agnostic, discriminates
# arity, and returns NULL (never raises) when nothing matches.
_GRANT_STALE_RECORDING = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app')
     AND to_regprocedure('stale_recording_jobs(int, int)') IS NOT NULL THEN
    GRANT EXECUTE ON FUNCTION stale_recording_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""

_GRANT_EXPIRED_RECORDING = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app')
     AND to_regprocedure('expired_recording_jobs(int)') IS NOT NULL THEN
    GRANT EXECUTE ON FUNCTION expired_recording_jobs(int) TO w2v_video_production_app;
  END IF;
END $$
"""

_GRANT_EXPIRED_COMPILATION = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app')
     AND to_regprocedure('expired_compilation_jobs(timestamptz, timestamptz, int)')
         IS NOT NULL THEN
    GRANT EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int)
      TO w2v_video_production_app;
  END IF;
END $$
"""


def upgrade() -> None:
    for stmt in (
        _GRANT_STALE_RECORDING,
        _GRANT_EXPIRED_RECORDING,
        _GRANT_EXPIRED_COMPILATION,
    ):
        op.execute(stmt)


def downgrade() -> None:
    # Revoking would re-break the reaper, which is the bug this revision exists to
    # fix. Downgrade is deliberately a no-op: the grant is the safe state, and
    # nothing above this revision depends on its absence.
    pass
