"""Lease-gated compilation stale-reaper tests (real Postgres + fakeredis).

Mirrors the recording reaper: a compilation job stuck in ``processing`` past the
age threshold is failed ONLY when no live lease protects it (crashed pod), and
is SKIPPED while its lease is present (running on another pod). Exercises the new
``stale_compilation_jobs`` SECURITY DEFINER resolver.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import fakeredis
import fakeredis.aioredis
from sqlalchemy import select

from src.database.models import CompilationJob
from src.services.compilation.cleanup_service import CleanupService
from src.services.job_lease import JobLease
from tests.conftest import TENANT_A


async def _seed(db, job_id, status="processing", *, minutes_old=60):
    ts = datetime.now(UTC) - timedelta(minutes=minutes_old)
    db.add(
        CompilationJob(
            job_id=job_id,
            tenant_id=TENANT_A,
            status=status,
            created_at=ts,
            updated_at=ts,
        )
    )
    await db.commit()


async def _status_of(db, job_id):
    return (
        await db.execute(select(CompilationJob).where(CompilationJob.job_id == job_id))
    ).scalar_one_or_none()


def _make_service(job_lease, stale_timeout=600):
    with patch("src.services.compilation.cleanup_service.get_settings") as mock_gs:
        mock_gs.return_value = MagicMock(
            cleanup=MagicMock(
                enabled=True, interval_seconds=3600, completed_ttl_days=30, failed_ttl_days=7
            ),
            storage=MagicMock(
                stale_job_timeout_seconds=stale_timeout,
                # 'pending' has its own, far longer clock since 008; these tests
                # all seed 'processing', so the value only has to be a real int.
                compilation_pending_stale_seconds=21600,
            ),
        )
        return CleanupService(job_lease=job_lease)


def _lease() -> JobLease:
    fake = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return JobLease(fake, prefix="compilation", instance_id="pod-under-test", ttl_seconds=60)


class TestCompilationReaper:
    async def test_reaps_orphaned_stale_job(self, db_session):
        await _seed(db_session, "comp-orphan", status="processing", minutes_old=60)
        lease = _lease()  # no acquire → not present
        svc = _make_service(lease)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        job = await _status_of(db_session, "comp-orphan")
        assert job.status == "failed"
        assert "reaped" in (job.error_message or "").lower()
        await lease.release_all()

    async def test_skips_job_with_live_lease(self, db_session):
        await _seed(db_session, "comp-live", status="processing", minutes_old=60)
        lease = _lease()
        await lease.acquire("comp-live")
        svc = _make_service(lease)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        assert (await _status_of(db_session, "comp-live")).status == "processing"
        await lease.release_all()

    async def test_leaves_fresh_job_alone(self, db_session):
        await _seed(db_session, "comp-fresh", status="processing", minutes_old=0)
        svc = _make_service(_lease())
        await svc._reap_stale_jobs()
        db_session.expire_all()
        assert (await _status_of(db_session, "comp-fresh")).status == "processing"


class TestGuardedGrant:
    async def test_guarded_grant_block_applies_without_error(self, db_session):
        """The self-contained create + revoke + pg_roles-guarded GRANT statements
        apply cleanly. The DO block no-ops here (the w2v_video_production_app role
        is absent in CI), which is exactly the prod-safe posture."""
        from sqlalchemy import text

        from src.database.rls_resolvers import stale_compilation_statements

        for stmt in stale_compilation_statements():
            await db_session.execute(text(stmt))
        await db_session.commit()

        # Function present, and PUBLIC has no EXECUTE (revoked).
        #
        # n == 2, not 1: the two-arg compatibility overload is resident ALONGSIDE
        # the three-arg form so the old pod can keep reaping through a rollout.
        # See test_reclaim_pending_compilation_jobs.py (d) for the arity assertion.
        row = (
            await db_session.execute(
                text(
                    "SELECT has_function_privilege('public', "
                    "'stale_compilation_jobs(int, int, int)', 'EXECUTE') AS pub, "
                    "count(*) OVER () AS n "
                    "FROM pg_proc WHERE proname = 'stale_compilation_jobs'"
                )
            )
        ).all()[0]
        assert row.n == 2
        assert row.pub is False
