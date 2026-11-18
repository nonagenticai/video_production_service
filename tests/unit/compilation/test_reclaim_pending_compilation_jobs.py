"""The compilation reaper must reach ``pending`` -- on its OWN clock -- and expire ``cancelled``.

Real Postgres 18 + psycopg3 (the SECURITY DEFINER resolvers are the thing under
test, so a fake session would test nothing) + fakeredis for the lease.

Three properties, one per leak closed:

(a) a row stranded at ``pending`` past the PENDING window is reclaimed at all.
    Before this change ``stale_compilation_jobs`` matched
    ``status IN ('processing', 'queued')`` and ``'queued'`` is never written for
    compilation jobs, so ``'pending'`` was reachable by NO clock. A SIGTERM
    between the route's ``db.commit()`` and the background task's first
    ``mark_processing`` stranded the row forever.

(b) ⚠️ THE REGRESSION GUARD. A ``pending`` row that is old by the PROCESSING
    window but young by the PENDING window must be SPARED. ``pending`` is a
    legitimate queued state and its ``updated_at`` does not advance while it
    waits, so that clock measures queue wait, not silence. Anyone who later
    "simplifies" the resolver by passing one age for both -- or by collapsing the
    two disjuncts -- fails live queued work, which is the 2026-09-03 recording
    outage in the compilation domain. This test is what stops them.

(c) ``cancelled`` rows expire on the failed TTL. They appeared in no branch of
    ``expired_compilation_jobs``, so they and their storage dirs were immortal.

(d) ⚠️ THE ROLLOUT-SAFETY GUARD. After the migration BOTH the two-arg and the
    three-arg ``stale_compilation_jobs`` overloads exist and are callable. The
    deployment overlaps versions (replicas=1, maxSurge 25% -> 1, maxUnavailable
    25% -> 0), so the old pod keeps calling the two-arg form while the new pod's
    initContainer runs alembic. Dropping that signature is a self-inflicted
    outage of this very reclaim path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import fakeredis.aioredis
from sqlalchemy import select

from src.database.models import CompilationJob
from src.services.compilation.cleanup_service import CleanupService
from src.services.job_lease import JobLease
from tests.conftest import TENANT_A

# The two windows, deliberately far apart so "old by one, young by the other" is
# an expressible state -- which is the whole point of test (b).
PROCESSING_WINDOW_S = 600  # 10 min: no-progress window for a RUNNING job
PENDING_WINDOW_S = 21600  # 6 h: orphan window for a job that never started


async def _seed(db, job_id: str, status: str, *, minutes_old: float) -> None:
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


async def _row(db, job_id: str):
    return (
        await db.execute(select(CompilationJob).where(CompilationJob.job_id == job_id))
    ).scalar_one_or_none()


def _lease() -> JobLease:
    fake = fakeredis.aioredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=True)
    return JobLease(fake, prefix="compilation", instance_id="pod-under-test", ttl_seconds=60)


def _make_service(
    job_lease: JobLease | None = None,
    *,
    stale_timeout: int = PROCESSING_WINDOW_S,
    pending_timeout: int = PENDING_WINDOW_S,
    completed_ttl_days: int = 30,
    failed_ttl_days: int = 7,
) -> CleanupService:
    with patch("src.services.compilation.cleanup_service.get_settings") as mock_gs:
        mock_gs.return_value = MagicMock(
            cleanup=MagicMock(
                enabled=True,
                interval_seconds=3600,
                completed_ttl_days=completed_ttl_days,
                failed_ttl_days=failed_ttl_days,
            ),
            storage=MagicMock(
                stale_job_timeout_seconds=stale_timeout,
                compilation_pending_stale_seconds=pending_timeout,
            ),
        )
        return CleanupService(job_lease=job_lease)


class TestPendingIsReclaimed:
    async def test_pending_older_than_pending_window_is_reaped(self, db_session):
        """(a) The strand itself: 7h at 'pending' with nothing to move it."""
        await _seed(db_session, "comp-stranded", "pending", minutes_old=7 * 60)
        lease = _lease()  # never acquired -> no live lease -> genuinely orphaned
        svc = _make_service(lease)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        job = await _row(db_session, "comp-stranded")
        assert job is not None
        assert job.status == "failed", (
            "a 'pending' row 7h past the 6h pending window was NOT reclaimed -- "
            "the reaper's predicate cannot reach status='pending', so a job "
            "stranded by a SIGTERM between the route commit and mark_processing "
            "stays pending forever and the caller's poller never terminates"
        )
        assert "reaped" in (job.error_message or "").lower()
        # The message must name the window that actually fired, not the 600s one.
        assert str(PENDING_WINDOW_S) in (job.error_message or "")
        await lease.release_all()

    async def test_live_lease_still_protects_a_pending_row(self, db_session):
        """Reaching 'pending' must not bypass the liveness gate."""
        await _seed(db_session, "comp-pending-live", "pending", minutes_old=7 * 60)
        lease = _lease()
        await lease.acquire("comp-pending-live")
        svc = _make_service(lease)

        await svc._reap_stale_jobs()

        db_session.expire_all()
        assert (await _row(db_session, "comp-pending-live")).status == "pending"
        await lease.release_all()


class TestPendingWindowIsSeparate:
    async def test_pending_old_by_processing_window_but_young_by_pending_is_spared(
        self, db_session
    ):
        """(b) ⚠️ REGRESSION GUARD -- do not collapse the two cutoffs.

        30 minutes old: 3x past the 600s PROCESSING window, but only 1/12th of the
        way through the 6h PENDING window. This is an ordinary job waiting its
        turn. If the pending disjunct is ever keyed on ``p_max_age_s`` -- or the
        caller passes ``stale_job_timeout_seconds`` for both -- this row is failed
        and so is every legitimately queued compilation on the fleet.
        """
        await _seed(db_session, "comp-just-queued", "pending", minutes_old=30)
        svc = _make_service(_lease())

        await svc._reap_stale_jobs()

        db_session.expire_all()
        job = await _row(db_session, "comp-just-queued")
        assert job.status == "pending", (
            f"a 'pending' row {30 * 60}s old was reaped: that is past the "
            f"{PROCESSING_WINDOW_S}s PROCESSING window but well inside the "
            f"{PENDING_WINDOW_S}s PENDING window. The two clocks have been "
            "collapsed, and this fails out live queued work -- the 2026-09-03 "
            "recording outage reproduced for compilations"
        )
        assert job.error_message is None

    async def test_processing_keeps_the_tight_window(self, db_session):
        """The split must not have loosened 'processing' onto the 6h clock."""
        await _seed(db_session, "comp-silent", "processing", minutes_old=30)
        svc = _make_service(_lease())

        await svc._reap_stale_jobs()

        db_session.expire_all()
        assert (await _row(db_session, "comp-silent")).status == "failed", (
            "a 'processing' row silent for 30 min was not reaped -- the processing "
            "window has been widened onto the pending clock"
        )


class TestCancelledExpires:
    async def test_cancelled_rows_are_deleted_on_the_failed_ttl(self, db_session):
        """(c) Cancelled rows were in no branch of expired_compilation_jobs."""
        # failed_ttl_days=7, so 10 days old is past it; the completed TTL (30d) is not.
        await _seed(db_session, "comp-cancelled", "cancelled", minutes_old=10 * 24 * 60)
        await _seed(db_session, "comp-cancelled-fresh", "cancelled", minutes_old=60)
        svc = _make_service()
        svc._storage = MagicMock(delete_job_dir=AsyncMock())

        await svc._cleanup_cycle()

        db_session.expire_all()
        assert await _row(db_session, "comp-cancelled") is None, (
            "a 'cancelled' compilation job 10 days old survived the expiry sweep "
            "(failed TTL is 7 days) -- 'cancelled' is missing from "
            "expired_compilation_jobs, so cancelled rows and their storage "
            "directories accumulate forever"
        )
        assert (await _row(db_session, "comp-cancelled-fresh")) is not None, (
            "a 1-hour-old cancelled job was deleted -- the TTL is not being applied"
        )
        svc._storage.delete_job_dir.assert_awaited_once_with("comp-cancelled")


def _migration_008_upgrade_statements() -> tuple[str, ...]:
    """The REAL statement list from ``008_reclaim_pending_compile.py``.

    Loaded by path because the module name starts with a digit and so is not
    importable. Deliberately not a hand-copied paraphrase: test (d) exists to
    catch a ``DROP FUNCTION`` reappearing in THAT file, and a paraphrase would
    stay green while the migration shipped the outage.
    """
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "database"
        / "migrations"
        / "versions"
        / "008_reclaim_pending_compile.py"
    )
    spec = importlib.util.spec_from_file_location("migration_008_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UPGRADE_STATEMENTS


class TestResolverSignature:
    async def test_both_arities_are_resident_and_callable_after_the_migration(self, db_session):
        """(d) ⚠️ ROLLOUT-SAFETY GUARD -- do not drop the two-arg overload.

        The live rollout spec is ``replicas=1, maxSurge=25%, maxUnavailable=25%``.
        Kubernetes rounds maxSurge UP to 1 and maxUnavailable DOWN to 0, so the OLD
        pod keeps serving until the NEW pod is Ready -- the two versions OVERLAP.
        Alembic runs in the new pod's initContainer, so this migration lands while
        the old pod's reaper is still calling the TWO-arg form every cycle
        (``cleanup_service.py``:59 on origin/main). Dropping that signature raises
        ``function stale_compilation_jobs(integer, integer) does not exist`` on
        every one of those calls, for the whole rollout, and permanently after a
        code-only rollback: a self-inflicted outage of the reclaim path this PR
        exists to fix.

        Both overloads must therefore exist AND be callable, and the call must be
        UNAMBIGUOUS -- which is why both are actually invoked below rather than
        merely counted in ``pg_proc``.
        """
        from sqlalchemy import text

        for stmt in _migration_008_upgrade_statements():
            await db_session.execute(text(stmt))
        await db_session.commit()

        args = sorted(
            r.args
            for r in (
                await db_session.execute(
                    text(
                        "SELECT pg_get_function_identity_arguments(oid) AS args FROM pg_proc "
                        "WHERE proname = 'stale_compilation_jobs'"
                    )
                )
            ).all()
        )
        assert args == [
            "p_max_age_s integer, p_limit integer",
            "p_max_age_s integer, p_pending_max_age_s integer, p_limit integer",
        ], (
            "after `alembic upgrade head` BOTH stale_compilation_jobs overloads must "
            "be resident -- the three-arg form for the new pod and the TWO-ARG form "
            f"(p_max_age_s integer, p_limit integer) for the old pod that keeps "
            f"serving through the rollout. Found: {args}. A missing two-arg signature "
            "means every reaper cycle on the old replica raises UndefinedFunction "
            "until the new pod is Ready"
        )

        # Both are CALLABLE, and neither call is ambiguous. Arity differs (2 vs 3),
        # so Postgres can always pick -- but assert it rather than assume it.
        await _seed(db_session, "comp-overload-pending", "pending", minutes_old=7 * 60)
        await _seed(db_session, "comp-overload-processing", "processing", minutes_old=30)

        two = (
            (
                await db_session.execute(
                    text("SELECT job_id FROM stale_compilation_jobs(:age, :lim)"),
                    {"age": PROCESSING_WINDOW_S, "lim": 50},
                )
            )
            .scalars()
            .all()
        )
        three = (
            (
                await db_session.execute(
                    text("SELECT job_id FROM stale_compilation_jobs(:age, :pending, :lim)"),
                    {"age": PROCESSING_WINDOW_S, "pending": PENDING_WINDOW_S, "lim": 50},
                )
            )
            .scalars()
            .all()
        )

        # The two-arg overload keeps its ORIGINAL behaviour: processing/queued only.
        assert "comp-overload-processing" in two
        assert "comp-overload-pending" not in two, (
            "the two-arg compatibility overload must keep its ORIGINAL body "
            "(status IN ('processing','queued') on ONE window) -- the old pod that "
            "calls it did not ask for new semantics mid-rollout"
        )
        # The three-arg form is the fix, and reaches 'pending' on its own clock.
        assert {"comp-overload-processing", "comp-overload-pending"} <= set(three)

        pub = (
            await db_session.execute(
                text(
                    "SELECT has_function_privilege('public', "
                    "'stale_compilation_jobs(int, int, int)', 'EXECUTE') AS three, "
                    "has_function_privilege('public', "
                    "'stale_compilation_jobs(int, int)', 'EXECUTE') AS two"
                )
            )
        ).one()
        assert pub.three is False
        assert pub.two is False
