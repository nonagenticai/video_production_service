"""Add the ``stale_compilation_jobs`` SECURITY DEFINER cross-tenant read resolver.

The compilation cleanup loop gains a lease-gated stale-processing reaper (mirroring
the recording one) so a compilation job orphaned by a crashed pod is safely failed
instead of hanging in ``processing`` forever. Like the other cleanup resolvers it is
owner-owned (bypasses ENABLE) and does the cross-tenant READ only; the reaper then
mutates per-tenant via the existing ``CompilationJobManager.mark_failed``.

EXECUTE is REVOKEd from PUBLIC and then GRANTed **in-band** to the restricted runtime
role ``w2v_video_production_app`` via a ``pg_roles``-guarded DO block. This is
self-contained (does NOT rely on the gitops stage-a grant file): without it the
reaper — which runs as the non-owner app role under ``TENANT_RLS_MODE=enforce`` —
would hit "permission denied for function" and silently no-op in prod. The guard
no-ops in CI, where the role is absent and the owner executes the function anyway.

################################################################################
# PINNED — DO NOT RE-DERIVE THIS DDL FROM LIVE CODE.
#
# This migration used to call ``src.database.rls_resolvers.stale_compilation_
# statements()`` — the LIVE module. Same historical-migration defect as 002: after
# revision 004 flipped the model to ``uuid``, that module declares
# ``RETURNS TABLE(..., tenant_id uuid)``, and a ``LANGUAGE sql`` body is validated
# at CREATE time — so on a FRESH database, where ``compilation_jobs.tenant_id`` is
# still ``varchar(128)`` at this revision, the CREATE fails. Static gates (ruff, ty,
# ``alembic heads``) are all green and blind to it; only ``alembic upgrade head``
# on an EMPTY database sees it.
#
# The era-correct signature at revision 003 is ``tenant_id varchar``. Revision 004
# DROPs this function and re-creates it from the live module, which is correct
# there because at 004 the live metadata IS the current era.
################################################################################

Revision ID: 003_stale_compilation_resolver
Revises: 002_rls
"""

from alembic import op

revision = "003_stale_compilation_resolver"
down_revision = "002_rls"
branch_labels = None
depends_on = None

# Era-correct: ``compilation_jobs.tenant_id`` is ``varchar(128)`` at this revision.
_STALE_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION stale_compilation_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id, cj.status FROM compilation_jobs cj
    WHERE cj.status IN ('processing', 'queued')
      AND cj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""

_REVOKE_SQL = "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC"

_GRANT_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""


def upgrade() -> None:
    for stmt in (_STALE_COMPILATION_SQL, _REVOKE_SQL, _GRANT_SQL):
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int)")
