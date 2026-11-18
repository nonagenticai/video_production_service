"""Flip all 7 ``tenant_id`` columns from ``varchar(128)`` to native ``uuid``.

Fleet-wide ruling: ``tenant_id`` is a real uuid, never text, never a sentinel.

The only dirty value in this database is the literal string ``n8n-dev`` — which is
NOT a sentinel, it is a REGISTERED TENANT NAME whose uuid is
``019de842-c92a-7e37-8fdd-eb0d8c5396f9`` (verified live in ``licenses.tenants``
2026-08-26). So this migration is a name→uuid backfill followed by a type flip.

Measured live rows carrying it: ``compilation_usage`` 1, ``recording_jobs`` 9,
``recording_usage`` 1. Every other value already parses as a uuid.

Ordering is load-bearing — four things must happen in this exact sequence:

1. **Backfill while the column is still text.** Once it is ``uuid`` the string
   ``n8n-dev`` can no longer be matched or written, so the rename must land first.
2. **Fail loud on anything still unparseable.** A ``DO`` block counts rows that are
   NOT NULL and do not match the uuid regex and ``RAISE EXCEPTION`` with that count.
   We deliberately do NOT hide them behind a ``WHERE tenant_id ~ ...`` on the ALTER —
   a silent skip would either drop rows on the floor or leave the migration
   half-applied. If this fires, a new dirty value appeared after the 2026-08-26
   measurement and a human must map it to a tenant uuid before retrying.
3. **Drop the RLS policies and the resolver functions BEFORE the ALTER.**
   * Postgres refuses ``ALTER COLUMN ... TYPE`` outright on a column referenced by a
     policy expression ("cannot alter type of a column used in a policy definition"),
     so the ALTER cannot even run while the ``tenant_isolation`` policies exist.
   * The policy predicate is stored as parsed SQL. The w2v_common generator
     (``rls._predicate``) picks its cast from the SQLAlchemy column type: a ``String``
     column compares text-to-text, anything else casts the GUC ``::uuid``. Because
     ``models.py`` now declares ``PgUUID``, re-running ``upgrade_statements`` after the
     ALTER re-creates every policy with ``NULLIF(current_setting(...), '')::uuid`` —
     single-sourced, so the migration and the ``create_all`` test path cannot drift.
   * The four SECURITY DEFINER cross-tenant read resolvers declare
     ``RETURNS TABLE(..., tenant_id varchar)``. After the ALTER the body returns
     ``uuid`` and every call dies with "structure of query does not match function
     result type" — which would silently kill BOTH cleanup/reaper loops. A
     ``CREATE OR REPLACE`` cannot change a return type, so they are DROPped here and
     re-created from ``rls_resolvers`` (now declaring ``tenant_id uuid``).
4. **ALTER, then re-create policies + resolvers.**

Downgrade returns the columns to ``varchar(128)`` and restores the text-comparison
policies. NOTE: the name→uuid backfill is deliberately NOT reversed — the uuid is the
correct, canonical identifier for that tenant and re-introducing the string
``n8n-dev`` would only recreate the defect. The downgrade also cannot reuse
``w2v_common.upgrade_statements``: it renders from the *current* model, which is uuid,
so it would emit a ``::uuid`` predicate against a varchar column and fail with
"operator does not exist: character varying = uuid". The text predicate is therefore
spelled out explicitly below, matching what ``002_rls`` originally produced.

Revision ID: 004_tenant_id_uuid
Revises: 003_stale_compilation_resolver
"""

import sqlalchemy as sa
from alembic import op
from w2v_common.database.rls import (
    downgrade_statements as rls_downgrade_statements,
)
from w2v_common.database.rls import (
    upgrade_statements as rls_upgrade_statements,
)

from src.database import rls_resolvers
from src.database.models import Base

revision = "004_tenant_id_uuid"
down_revision = "003_stale_compilation_resolver"
branch_labels = None
depends_on = None

# The 7 tenant-scoped tables, all ``tenant_id varchar(128) NOT NULL`` before this.
TENANT_TABLES = (
    "recording_jobs",
    "recording_usage",
    "compilation_jobs",
    "compilation_batches",
    "compilation_templates",
    "compilation_usage",
    "custom_presets",
)

# Registered tenant name → its uuid (licenses.tenants, verified live 2026-08-26).
LEGACY_TENANT_NAME = "n8n-dev"
LEGACY_TENANT_UUID = "019de842-c92a-7e37-8fdd-eb0d8c5396f9"

_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

# What ``002_rls`` emitted for a NOT NULL varchar tenant_id (strict equality, text
# vs text). Re-created verbatim on downgrade. See the module docstring for why this
# cannot be regenerated from w2v_common at downgrade time.
_TEXT_PREDICATE = "tenant_id = NULLIF(current_setting('app.current_tenant', true), '')"

# Era-correct (pre-004) resolver signatures, spelled out literally for the downgrade.
_LEGACY_RESOLVER_SQL = (
    """
CREATE OR REPLACE FUNCTION expired_recording_jobs(p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id FROM recording_jobs rj
    WHERE rj.expires_at <= now()
      AND rj.expired IS false
      AND rj.status IN ('completed', 'failed', 'cancelled')
    LIMIT p_limit;
  $fn$
""",
    """
CREATE OR REPLACE FUNCTION stale_recording_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT rj.job_id, rj.tenant_id, rj.status FROM recording_jobs rj
    WHERE rj.status IN ('processing', 'queued')
      AND rj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
""",
    """
CREATE OR REPLACE FUNCTION expired_compilation_jobs(
    p_completed_before timestamptz, p_failed_before timestamptz, p_limit int
) RETURNS TABLE(job_id varchar, tenant_id varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id FROM compilation_jobs cj
    WHERE (cj.status = 'completed' AND cj.updated_at < p_completed_before)
       OR (cj.status = 'failed' AND cj.updated_at < p_failed_before)
    LIMIT p_limit;
  $fn$
""",
    """
CREATE OR REPLACE FUNCTION stale_compilation_jobs(p_max_age_s int, p_limit int)
  RETURNS TABLE(job_id varchar, tenant_id varchar, status varchar)
  LANGUAGE sql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
    SELECT cj.job_id, cj.tenant_id, cj.status FROM compilation_jobs cj
    WHERE cj.status IN ('processing', 'queued')
      AND cj.updated_at <= now() - make_interval(secs => p_max_age_s)
    LIMIT p_limit;
  $fn$
""",
)

_LEGACY_RESOLVER_REVOKE = (
    "REVOKE EXECUTE ON FUNCTION expired_recording_jobs(int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_recording_jobs(int, int) FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION expired_compilation_jobs(timestamptz, timestamptz, int) "
    "FROM PUBLIC",
    "REVOKE EXECUTE ON FUNCTION stale_compilation_jobs(int, int) FROM PUBLIC",
)

_LEGACY_GRANT_STALE_COMPILATION_SQL = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_video_production_app') THEN
    GRANT EXECUTE ON FUNCTION stale_compilation_jobs(int, int) TO w2v_video_production_app;
  END IF;
END $$
"""


# Guard template. Interpolated only with names from TENANT_TABLES (module-level
# constants) and the uuid regex — never with anything caller- or row-derived.
_GUARD_TEMPLATE = """
DO $$
DECLARE
  bad_count bigint;
  sample text;
BEGIN
  SELECT count(*), min(tenant_id) INTO bad_count, sample
    FROM {table}
   WHERE tenant_id IS NOT NULL AND tenant_id !~ '{uuid_re}';
  IF bad_count > 0 THEN
    RAISE EXCEPTION
      '{table}.tenant_id: % row(s) are not a parseable uuid (e.g. %). Refusing to '
      'ALTER to uuid. Map each value to its tenant uuid in licenses.tenants and '
      'add an explicit backfill above, then re-run. Do NOT skip them with a WHERE.',
      bad_count, sample;
  END IF;
END $$
"""


def _guard_sql(table: str) -> str:
    """FAIL LOUD if any non-NULL ``tenant_id`` is still not a parseable uuid."""
    return _GUARD_TEMPLATE.format(table=table, uuid_re=_UUID_RE)


def upgrade() -> None:
    # 1. Backfill the registered tenant NAME to its uuid, while still text.
    for table in TENANT_TABLES:
        # The VALUES are bound, so nothing user-supplied is interpolated. Only
        # the table IDENTIFIER is, and it cannot be a bind parameter -- it comes
        # from TENANT_TABLES, a module-level literal tuple. bandit still flags
        # the f-string shape, so it needs `nosec`: `noqa: S608` silences ruff
        # ONLY, which is why this line read as handled while CI stayed red.
        op.execute(  # nosec B608
            sa.text(
                f"UPDATE {table} SET tenant_id = :new_owner "  # noqa: S608
                "WHERE tenant_id = :legacy_name"
            ).bindparams(new_owner=LEGACY_TENANT_UUID, legacy_name=LEGACY_TENANT_NAME),
        )

    # 2. Fail loud on anything still unparseable — never silently skipped.
    for table in TENANT_TABLES:
        op.execute(_guard_sql(table))

    # 3. Drop the policies (Postgres refuses the ALTER while they reference the
    #    column) and the resolvers (their varchar return type would break).
    for stmt in rls_resolvers.downgrade_statements():
        op.execute(stmt)
    for stmt in rls_downgrade_statements(Base.metadata):
        op.execute(stmt)

    # 4. The type flip.
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")

    # 5. Re-create policies (now rendering a ``::uuid`` GUC cast, because models.py
    #    declares PgUUID) and the resolvers (now declaring ``tenant_id uuid``).
    for stmt in rls_upgrade_statements(Base.metadata):
        op.execute(stmt)
    for stmt in rls_resolvers.upgrade_statements():
        op.execute(stmt)
    for stmt in rls_resolvers.stale_compilation_statements():
        op.execute(stmt)


def downgrade() -> None:
    # Drop uuid-shaped policies + resolvers before touching the column type.
    for stmt in rls_resolvers.downgrade_statements():
        op.execute(stmt)
    for stmt in rls_downgrade_statements(Base.metadata):
        op.execute(stmt)

    for table in TENANT_TABLES:
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE varchar(128) USING tenant_id::text"
        )

    # Restore the text-comparison policies 002_rls originally created. All 7
    # columns are NOT NULL → strict equality (no global-or-own read policy).
    for table in TENANT_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_TEXT_PREDICATE}) WITH CHECK ({_TEXT_PREDICATE})"
        )

    # Resolvers must go back to their ``tenant_id varchar`` signature. A downgrade
    # can NEVER render from the live module (it declares ``uuid`` now, and a
    # ``LANGUAGE sql`` body is validated at CREATE time against the varchar column),
    # so the era-correct DDL is spelled out literally — same reason 002/003 are pinned.
    for stmt in rls_resolvers.downgrade_statements():
        op.execute(stmt)
    for sql in _LEGACY_RESOLVER_SQL:
        op.execute(sql)
    for stmt in _LEGACY_RESOLVER_REVOKE:
        op.execute(stmt)
    op.execute(_LEGACY_GRANT_STALE_COMPILATION_SQL)
