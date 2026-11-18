"""``_process_batch_bg`` must mark the batch failed when the DB raises.

The guard around ``batch_svc.process_batch(...)`` used to read
``except (OSError, ValueError, RuntimeError, TimeoutError)``. ``SQLAlchemyError``
inherits from NONE of those, so the single most likely failure of a function whose
whole body is database work escaped BOTH handlers: the batch was never marked
``failed`` and the row stayed 'processing' with the client polling it forever.

These tests assert the resulting ROW STATE in a real Postgres, not a mock's call
count — a call-count assertion would pass against the broken guard too.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from sqlalchemy import select
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from src.api.routes._compilation_helpers import _process_batch_bg
from src.database.models import CompilationBatch
from tests.conftest import TENANT_A


async def _seed(db, batch_id):
    now = datetime.now(UTC)
    db.add(
        CompilationBatch(
            batch_id=batch_id,
            tenant_id=TENANT_A,
            status="processing",
            total_jobs=1,
            completed_jobs=0,
            failed_jobs=0,
            job_ids=[],
            errors=[],
            created_at=now,
            updated_at=now,
        )
    )
    await db.commit()


async def _row(db, batch_id):
    db.expire_all()
    return (
        await db.execute(select(CompilationBatch).where(CompilationBatch.batch_id == batch_id))
    ).scalar_one_or_none()


def _raising_batch_service(exc: BaseException) -> MagicMock:
    svc = MagicMock()

    async def _boom(*_args, **_kwargs):
        raise exc

    svc.process_batch = _boom
    return svc


async def _run(db_session_factory, batch_id, exc):
    with patch("src.api.routes.compilation.get_db_session", db_session_factory):
        await _process_batch_bg(_raising_batch_service(exc), batch_id, TENANT_A, 3)


class TestBatchBackgroundTaskFailurePaths:
    async def test_sqlalchemy_error_marks_the_batch_failed(self, db_session, db_session_factory):
        bid = f"batch-sqlerr-{uuid.uuid4()}"
        await _seed(db_session, bid)

        await _run(db_session_factory, bid, SQLAlchemyError("connection reset"))

        row = await _row(db_session, bid)
        assert row.status == "failed"
        assert "connection reset" in str(row.errors)

    async def test_operational_error_subclass_marks_the_batch_failed(
        self, db_session, db_session_factory
    ):
        """The concrete shape a dead pgbouncer connection takes."""
        bid = f"batch-operr-{uuid.uuid4()}"
        await _seed(db_session, bid)

        await _run(
            db_session_factory,
            bid,
            OperationalError("SELECT 1", {}, Exception("server closed the connection")),
        )

        assert (await _row(db_session, bid)).status == "failed"

    async def test_value_error_still_marks_the_batch_failed(self, db_session, db_session_factory):
        """Regression guard for the errors the old curated tuple DID cover."""
        bid = f"batch-valerr-{uuid.uuid4()}"
        await _seed(db_session, bid)

        await _run(db_session_factory, bid, ValueError("bad config"))

        assert (await _row(db_session, bid)).status == "failed"
