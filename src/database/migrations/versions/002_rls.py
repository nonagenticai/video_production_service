"""Phase D RLS — enable row-level security on the tenant-scoped tables.

(1) Strict per-table policies on all 7 ``tenant_id`` tables:
    recording_jobs, recording_usage, compilation_jobs, custom_presets,
    compilation_templates, compilation_batches, compilation_usage.
(2) SECURITY DEFINER cross-tenant READ resolvers (expired_recording_jobs /
    stale_recording_jobs / expired_compilation_jobs) so the boot-time cleanup loops
    can still find jobs across all tenants under enforce (a non-bypass role with no
    GUC would see 0 rows); the cleanup service then mutates per-tenant via the
    existing JobManager methods. EXECUTE is REVOKEd from PUBLIC here and GRANTed to
    the app role out-of-band in Stage A (the app role does not exist in CI).

ENABLE-not-FORCE → inert until the runtime connects as the restricted
``w2v_video_production_app`` role with ``TENANT_RLS_MODE=enforce``. Fully reversible.

################################################################################
# PINNED — DO NOT RE-DERIVE THIS DDL FROM LIVE CODE.
#
# This migration used to render its policies from
# ``w2v_common.database.rls.upgrade_statements(Base.metadata)`` and its resolvers
# from ``src.database.rls_resolvers`` — i.e. from the LIVE model / LIVE module,
# not from the schema as it exists at revision 002. That is a historical-migration
# defect: the helper picks its cast from the *current* SQLAlchemy column type, so
# the moment ``models.py`` flipped ``tenant_id`` from ``String(128)`` to ``uuid``
# (revision 004), a FRESH database died right here at 002 with
#
#     (psycopg.errors.UndefinedFunction) operator does not exist:
#     character varying = uuid
#
# because 002 still creates the column as ``varchar(128)`` while the live metadata
# now renders ``NULLIF(current_setting(...), '')::uuid``. Identically, the live
# ``rls_resolvers`` module now declares ``RETURNS TABLE(..., tenant_id uuid)``,
# which a ``LANGUAGE sql`` function body validated at CREATE time rejects against a
# varchar column.
#
# Every static gate (ruff, ty, ``alembic heads``) is green and blind to this. Only
# ``alembic upgrade head`` against an EMPTY database sees it.
#
# A historical migration must therefore describe the era it belongs to LITERALLY.
# The era-correct predicate here is the text comparison against a varchar column;
# revision 004 is the one that flips it to ``::uuid``, and 004 is the only place
# allowed to render from live metadata (there, live metadata IS the correct era).
#
# The downgrade drops EVERY policy name the generator could ever have emitted
# (``tenant_isolation`` plus the nullable "global-or-own" variants ``_read``,
# ``_ins``, ``_upd``, ``_del``) so a database provisioned by an older code path
# still reverses cleanly.
################################################################################

Revision ID: 002_rls
Revises: 001
"""

from alembic import op

revision = "002_rls"
down_revision = "001"
branch_labels = None
depends_on = None

# The 7 tenant-scoped tables at this revision. All carry ``tenant_id varchar(128)
# NOT NULL`` here → strict equality (no global-or-own read policy).
TENANT_TABLES = (
    "recording_jobs",
    "recording_usage",
    "compilation_jobs",
    "compilation_batches",
    "compilation_templates",
    "compilation_usage",
    "custom_presets",
)

POLICY_NAME = "tenant_isolation"

# Every policy name the w2v_common generator can emit, so the downgrade reverses a
# database built by any code path (strict OR the nullable global-or-own variant).
ALL_POLICY_NAMES = (
    POLICY_NAME,
    f"{POLICY_NAME}_read",
    f"{POLICY_NAME}_ins",
    f"{POLICY_NAME}_upd",
    f"{POLICY_NAME}_del",
)

# Era-correct predicate: text-to-text against a ``varchar(128)`` column. Postgres
# stores it as ``((tenant_id)::text = NULLIF(current_setting('app.current_tenant'::text,
# true), ''::text))``. Casting the GUC ``::uuid`` here would raise
# "operator does not exist: character varying = uuid".
_TEXT_PREDICATE = "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')"

# The three cross-tenant READ resolvers that existed at THIS revision, pinned with
# their era-correct ``tenant_id varchar`` return column. (``stale_compilation_jobs``
# arrives in 003.) A ``LANGUAGE sql`` body is validated at CREATE time, so declaring
# ``uuid`` here against a varchar column fails outright.
_EXPIRED_RECORDING_SQL = """
CREATE OR REPLACE FUNCTION expired_recording_jobs(p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id FROM recording_jobs rj
    WHERE rj.expires_at <= now()
      AND rj.expired IS false
      AND rj.status IN ('completed', 'failed', 'cancelled')
    LIMIT p_limit;
  $fn$
"""

_STALE_RECORDING_SQL = """
CREATE OR REPLACE FUNCTION stale_recording_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id, rj.status FROM recording_jobs rj
    WHERE rj.status IN ('processing', 'queued')
      AND rj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
"""

_EXPIRED_COMPILATION_SQL = """
CREATE OR REPLACE FUNCTION expired_compilation_jobs(
    p_completed_before timestamptz, p_failed_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id FROM compilation_jobs cj
    WHERE (cj.status = 'completed' AND cj.updated_at < p_completed_before)
       OR (cj.status = 'failed' AND cj.updated_at < p_failed_before)
    LIMIT p_limit;
  $fn$
"""

_RESOLVER_SQL = (
    _EXPIRED_RECORDING_SQL,
    _STALE_RECORDING_SQL,
    _EXPIRED_COMPILATION_SQL,
)

_RESOLVER_REVOKE = (
    "REVOKE EXECUTE ON FUNCTION expired_recording_jobs(int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int) "
    "FROM PUBLIC",
)

# Drop every resolver this repo has ever created, including the 003 one, so the
# downgrade reverses a DB built by any code path.
_RESOLVER_DROP = (
    "DROP FUNCTION IF EXISTS expired_recording_jobs(int)",
    "DROP FUNCTION IF EXISTS stale_recording_jobs(int, int)",
    "DROP FUNCTION IF EXISTS expired_compilation_jobs(timestamptz, timestamptz, int)",
    "DROP FUNCTION IF EXISTS stale_compilation_jobs(int, int)",
)


def upgrade() -> None:
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}")
        op.execute(
            f"CREATE POLICY {POLICY_NAME} ON {table} "
            f"USING ({_TEXT_PREDICATE}) WITH CHECK ({_TEXT_PREDICATE})"
        )
    for stmt in (*_RESOLVER_SQL, *_RESOLVER_REVOKE):
        op.execute(stmt)


def downgrade() -> None:
    for stmt in _RESOLVER_DROP:
        op.execute(stmt)
    for table in TENANT_TABLES:
        for policy in ALL_POLICY_NAMES:
            op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
