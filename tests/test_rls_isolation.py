"""Proves the Phase D RLS policies ISOLATE tenants for video_production_service.

* ``recording_jobs`` — strict own-only (representative of all 7 strict tenant tables).
* ``expired_recording_jobs`` — the SECURITY DEFINER cross-tenant cleanup READ resolver
  still finds jobs across tenants (owner bypasses ENABLE).

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role (the runtime posture under
enforce). Needs a superuser test DB; skips loudly in CI where it can't CREATE ROLE.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

# tenant_id is native ``uuid`` since migration 004 — a slug no longer casts.
from tests.conftest import TENANT_A, TENANT_B

_ROLE = "w2v_video_production_app_rls_test"
_INS = "INSERT INTO recording_jobs (job_id, tenant_id, part_id, status) VALUES (:id, :t, 'p1', :st)"


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        await conn.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON recording_jobs TO "{_ROLE}"')
        )

        await conn.execute(text(_INS), {"id": "job-a", "t": TENANT_A, "st": "completed"})
        await conn.execute(text(_INS), {"id": "job-b", "t": TENANT_B, "st": "completed"})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            n = (await conn.execute(text("SELECT count(*) FROM recording_jobs"))).scalar()
            assert n == 0, f"leaked {n} rows with no GUC"

            await conn.execute(
                # ``set_config`` is (text, text, bool): the GUC is a STRING even though
                # ``tenant_id`` is a ``uuid`` column, so the uuid is rendered here.
                text("SELECT set_config('app.current_tenant', :t, true)"),
                {"t": str(TENANT_A)},
            )
            rows = (
                (await conn.execute(text("SELECT tenant_id FROM recording_jobs"))).scalars().all()
            )
            assert list(rows) == [TENANT_A], f"not isolated under GUC=A: {rows}"

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS), {"id": "job-x", "t": TENANT_B, "st": "completed"})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))


@pytest.mark.asyncio
async def test_cleanup_read_resolver_is_cross_tenant(db_engine) -> None:
    """expired_recording_jobs returns expired jobs across tenants (SECURITY DEFINER)."""
    async with db_engine.begin() as conn:
        # Two expired completed jobs in different tenants (expires_at in the past).
        for jid, tid in (("exp-a", TENANT_A), ("exp-b", TENANT_B)):
            await conn.execute(
                text(
                    "INSERT INTO recording_jobs "
                    "(job_id, tenant_id, part_id, status, expired, expires_at) "
                    "VALUES (:id, :t, 'p1', 'completed', false, now() - interval '1 day')"
                ),
                {"id": jid, "t": tid},
            )
        rows = (
            await conn.execute(text("SELECT job_id, tenant_id FROM expired_recording_jobs(100)"))
        ).all()
        # ``tenant_id`` is a native ``uuid`` column since migration 004, so psycopg
        # hands back ``uuid.UUID`` and the conftest constants ARE ``uuid.UUID``.
        # Stringifying one side only would make this comparison silently false.
        found = {(r.job_id, r.tenant_id) for r in rows}
        assert ("exp-a", TENANT_A) in found and ("exp-b", TENANT_B) in found, found
