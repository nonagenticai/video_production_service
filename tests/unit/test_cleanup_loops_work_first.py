"""Both cleanup loops must run a cycle BEFORE their first sleep.

Regression guard. Each ``_run_loop`` used to ``await asyncio.sleep(self._interval)``
as the first statement of its body. The interval defaults to 3600s while these
pods roll far more often than hourly, so neither loop ever completed a single
cycle in production -- which is why the recording reaper reported zero cycles and
zero reaps even after its heartbeat (#139) and function-ACL (#141) fixes landed.
Those two fixes were inert until the loop could actually reach them.

Each test drives the real loop and asserts on an ordered log of what executed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ServerDisconnectedError
from miniopy_async.error import S3Error
from sqlalchemy.exc import OperationalError

if TYPE_CHECKING:
    from aiohttp import ClientResponse

from src.services.compilation.cleanup_service import CleanupService as CompilationCleanupService
from src.services.recording.cleanup_service import CleanupService as RecordingCleanupService


class _LoopService(Protocol):
    """The only surface these tests drive: the periodic loop itself."""

    async def _run_loop(self) -> None: ...


async def _drive(service: _LoopService, module: str, work_attrs: tuple[str, ...]) -> list[str]:
    """Run one pass of ``service._run_loop``, recording what it executed, in order.

    The patched sleep raises ``CancelledError``, so anything recorded before the
    ``sleep:`` entry genuinely ran ahead of it -- the property under test.
    """
    events: list[str] = []

    def _recorder(name: str) -> AsyncMock:
        async def _record() -> None:
            events.append(name)

        return AsyncMock(side_effect=_record)

    for attr in work_attrs:
        setattr(service, attr, _recorder(attr))

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with patch(f"{module}.asyncio.sleep", fake_sleep), contextlib.suppress(asyncio.CancelledError):
        await service._run_loop()
    return events


async def test_recording_cleanup_runs_a_cycle_before_sleeping() -> None:
    with patch("src.services.recording.cleanup_service.get_settings") as settings:
        settings.return_value.storage.cleanup_interval_seconds = 3600
        settings.return_value.storage.stale_job_timeout_seconds = 900
        service = RecordingCleanupService(storage=MagicMock())

    events = await _drive(
        service,
        "src.services.recording.cleanup_service",
        ("_cleanup_expired", "_reap_stale_jobs"),
    )

    assert events == ["_cleanup_expired", "_reap_stale_jobs", "sleep:3600"]


async def test_compilation_cleanup_runs_a_cycle_before_sleeping() -> None:
    with patch("src.services.compilation.cleanup_service.get_settings") as settings:
        settings.return_value.cleanup.enabled = True
        settings.return_value.cleanup.interval_seconds = 3600
        settings.return_value.cleanup.completed_ttl_days = 7
        settings.return_value.cleanup.failed_ttl_days = 3
        settings.return_value.storage.stale_job_timeout_seconds = 900
        service = CompilationCleanupService()

    events = await _drive(
        service,
        "src.services.compilation.cleanup_service",
        ("_cleanup_cycle", "_reap_stale_jobs"),
    )

    assert events == ["_cleanup_cycle", "_reap_stale_jobs", "sleep:3600"]


async def test_recording_loop_survives_a_failing_cycle_and_still_waits() -> None:
    """A raising cycle must be swallowed AND still wait out the interval.

    If the sleep were skipped on the error path the loop would spin hot against
    a failing database instead of backing off.
    """
    with patch("src.services.recording.cleanup_service.get_settings") as settings:
        settings.return_value.storage.cleanup_interval_seconds = 3600
        settings.return_value.storage.stale_job_timeout_seconds = 900
        service = RecordingCleanupService(storage=MagicMock())

    events: list[str] = []

    async def boom() -> None:
        events.append("_cleanup_expired")
        raise RuntimeError("db down")

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch.object(service, "_cleanup_expired", AsyncMock(side_effect=boom)),
        patch.object(service, "_reap_stale_jobs", AsyncMock()),
        patch("src.services.recording.cleanup_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._run_loop()

    assert events == ["_cleanup_expired", "sleep:3600"]


def _compilation_service() -> CompilationCleanupService:
    with patch("src.services.compilation.cleanup_service.get_settings") as settings:
        settings.return_value.cleanup.enabled = True
        settings.return_value.cleanup.interval_seconds = 3600
        settings.return_value.cleanup.completed_ttl_days = 30
        settings.return_value.cleanup.failed_ttl_days = 7
        settings.return_value.storage.stale_job_timeout_seconds = 900
        return CompilationCleanupService()


async def test_compilation_loop_survives_a_cycle_error_outside_the_old_tuple() -> None:
    """The compilation loop must survive ANY cycle failure, not three builtins.

    Regression guard for the defect #144 made worse. ``_run_loop`` used to catch
    only ``OSError, RuntimeError, ValueError``; a storage or SQLAlchemy error out
    of ``delete_job_dir`` escaped it, killed the asyncio task, and took the
    compilation stale-reaper with it. Because ``self._task`` pins a reference for
    the pod's lifetime the task is never GC'd, so not even "Task exception was
    never retrieved" is logged -- the death is TOTALLY silent. Sleep-first used to
    defer it by an interval; work-first reaches it ~200ms after pod start.

    ``OperationalError`` is deliberately chosen: it is a ``SQLAlchemyError``,
    which is OUTSIDE the old tuple, so this test is only green because the guard
    widened. It must also still reach its sleep -- a loop that skipped the sleep
    on the error path would hot-spin against a failing database.
    """
    service = _compilation_service()

    events: list[str] = []

    async def boom() -> None:
        events.append("_cleanup_cycle")
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

    async def reap() -> None:
        events.append("_reap_stale_jobs")

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch.object(service, "_cleanup_cycle", AsyncMock(side_effect=boom)),
        patch.object(service, "_reap_stale_jobs", AsyncMock(side_effect=reap)),
        patch("src.services.compilation.cleanup_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._run_loop()

    # Survived the raise and BACKED OFF rather than spinning or dying.
    assert events == ["_cleanup_cycle", "sleep:3600"]


async def test_compilation_cycle_swallows_its_own_failure_so_the_reaper_still_runs() -> None:
    """A failing cleanup sweep must NOT cost that cycle its stale-reaper sweep.

    This is why the guard lives inside ``_cleanup_cycle`` and not only in
    ``_run_loop``: a loop-level catch alone would skip ``_reap_stale_jobs`` for
    the whole cycle. Driven through the REAL ``_cleanup_cycle``, with the
    database scan raising, so the assertion covers the handler's own line.
    """
    service = _compilation_service()

    events: list[str] = []

    def exploding_session() -> object:
        events.append("db_scan")
        raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))

    async def reap() -> None:
        events.append("_reap_stale_jobs")

    async def fake_sleep(seconds: float) -> None:
        events.append(f"sleep:{seconds}")
        raise asyncio.CancelledError

    with (
        patch(
            "src.services.compilation.cleanup_service.get_db_session",
            side_effect=exploding_session,
        ),
        patch.object(service, "_reap_stale_jobs", AsyncMock(side_effect=reap)),
        patch("src.services.compilation.cleanup_service.asyncio.sleep", fake_sleep),
        contextlib.suppress(asyncio.CancelledError),
    ):
        await service._run_loop()

    # The reaper ran DESPITE the cleanup sweep blowing up in the same cycle.
    # Four sweeps per cycle now (job cleanup, job reap, batch reap, batch prune);
    # the three db-backed ones each open their own session, so the exploding
    # session fires three times. That every one of them is caught and the cycle
    # still reaches its sleep is the point: no sweep can cost its successors.
    assert events == ["db_scan", "_reap_stale_jobs", "db_scan", "db_scan", "sleep:3600"]


class _ScanResult:
    """Minimal stand-in for the SQLAlchemy ``Result`` of the expired-job scan."""

    def __init__(self, scalar: object = None, rows: list[tuple[str, str]] | None = None) -> None:
        self._scalar = scalar
        self._rows = rows or []

    def scalar(self) -> object:
        return self._scalar

    def all(self) -> list[tuple[str, str]]:
        return self._rows


class _ScanSession:
    """Async-context session that grants the advisory lock, then returns ``rows``."""

    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    async def execute(self, statement: object, params: dict | None = None) -> _ScanResult:
        if params is not None and "k" in params:
            return _ScanResult(scalar=True)
        return _ScanResult(rows=self._rows)

    async def __aenter__(self) -> _ScanSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _s3_error(job_id: str) -> S3Error:
    """A MinIO server-side error, as ``miniopy_async`` raises it."""
    return S3Error(
        code="InternalError",
        message="We encountered an internal error, please try again.",
        resource=f"/video-production/{job_id}",
        request_id="17F1B4C2A9E0",
        host_id="minio-0",
        response=cast("ClientResponse", None),
        bucket_name="video-production",
        object_name=job_id,
    )


def _transport_error(job_id: str) -> ServerDisconnectedError:
    """The aiohttp transport failure that escapes w2v-common's OWN MinIO guard."""
    return ServerDisconnectedError(f"Server disconnected while deleting {job_id}/")


@pytest.mark.parametrize("make_error", [_s3_error, _transport_error])
async def test_compilation_cycle_finishes_the_batch_when_storage_rejects_one_job(
    make_error: Callable[[str], Exception],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One job's storage failure must not abandon the rest of the 200-row batch.

    ``delete_job_dir`` reaches MinIO through ``w2v_common.storage`` over the
    ``miniopy-async``/``aiohttp`` client, so what it raises is NEVER one of three
    builtins. Under the old per-job ``except OSError, RuntimeError, ValueError``
    tuple such an error escaped the per-job ``try`` into the cycle-level guard:
    the failing job's DB ROW SURVIVED (immortal job, retried every hour forever),
    every later row of the scan was never visited, and the summary line was never
    emitted -- the batch truncated silently at the first bad job with the loop
    itself still alive. That is the original incident in a milder form, one layer
    below the ``_run_loop`` guard, and it is why this handler must be bare
    ``except Exception``.

    Both parameters are concrete real classes, not a bare ``Exception``, so the
    test still fails if the handler is ever re-narrowed around a builtin:

    * ``miniopy_async.error.S3Error`` -- a ``MinioException(Exception)``; the
      class the audit named.
    * ``aiohttp.ServerDisconnectedError`` -- a ``ClientError(Exception)``. This
      one is the sharper guard: w2v-common's own ``_delete_minio_prefix`` catches
      ``(OSError, S3Error, MinioException)`` and degrades, so a transport error is
      what actually reaches this caller today.

    The REAL ``_cleanup_cycle`` is driven (``get_db_session`` patched, as in the
    sibling test above) so the assertions cover the handler's own line.
    """
    service = _compilation_service()

    rows = [
        ("job-a", "11111111-1111-4111-8111-111111111111"),
        ("job-b", "22222222-2222-4222-8222-222222222222"),
        ("job-c", "33333333-3333-4333-8333-333333333333"),
    ]
    events: list[str] = []

    async def delete_job_dir(job_id: str) -> None:
        events.append(f"storage:{job_id}")
        if job_id == "job-b":
            raise make_error(job_id)

    async def delete_jobs(db: object, ids: list[str]) -> None:
        events.extend(f"db_delete:{job_id}" for job_id in ids)

    service._storage = MagicMock(delete_job_dir=AsyncMock(side_effect=delete_job_dir))

    with (
        patch(
            "src.services.compilation.cleanup_service.get_db_session",
            lambda: _ScanSession(rows),
        ),
        patch(
            "src.services.compilation.job_manager.CompilationJobManager.delete_jobs",
            AsyncMock(side_effect=delete_jobs),
        ),
        caplog.at_level(logging.INFO, logger="src.services.compilation.cleanup_service"),
    ):
        await service._cleanup_cycle()

    # Every job was visited in order, and the storage casualty's DB row went
    # anyway: the batch was not abandoned at job-b, and job-b is not immortal.
    assert events == [
        "storage:job-a",
        "db_delete:job-a",
        "storage:job-b",
        "db_delete:job-b",
        "storage:job-c",
        "db_delete:job-c",
    ]
    # deleted == len(rows): the summary line was reached and counted every row.
    assert f"Cleanup: deleted {len(rows)} expired compilation jobs" in caplog.messages
