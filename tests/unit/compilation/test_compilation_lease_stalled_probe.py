"""The compilation liveness lease must stop vouching for a progress-dead job.

``CleanupService._reap_stale_jobs`` is gated on ``JobLease.present()``: a job whose
lease is live is assumed to be running on another pod and is SKIPPED. That gate is
only safe while something can stop the heartbeat from renewing a lease forever, and
the only such thing is ``JobLease(stalled_probe=...)``.

The recording lease has carried that probe since the 2026-08-25 incident (18 wedged
recording jobs; ``stale_recording_jobs`` selected all 18 correctly and the reaper
skipped all 18, because the heartbeat kept renewing their leases). The COMPILATION
lease -- same class, same reaper shape, constructed 50 lines further down the same
module -- was built without one. A compile wedged on a pod that stays ALIVE (a
stalled MinIO read on the uncancellable ``retrieve_uri_sync`` worker thread, or an
ffmpeg step inside ``asyncio.to_thread`` that never returns) therefore sat in
``processing`` permanently: never failed, never webhooked, no ERROR line, caller
polling to its own deadline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

from src.asgi import _init_compilation_services
from src.services.compilation.cleanup_service import (
    _STALE_COMPILATION_LIMIT,
    _STALE_COMPILATION_QUERY,
    stalled_compilation_job_ids,
)
from src.services.job_lease import JobLease

pytestmark = pytest.mark.asyncio


def _stub_app() -> FastAPI:
    """A real FastAPI so `app.state` behaves exactly as it does at boot."""
    app = FastAPI()
    app.state.redis_client = None
    app.state.instance_id = "pod-under-test"
    app.state.storage_service = object()
    return app


class TestTheCompilationLeaseIsWiredWithAProbe:
    async def test_compilation_lease_carries_the_stalled_probe(self) -> None:
        app = _stub_app()
        await _init_compilation_services(app)
        lease = app.state.compilation_job_lease
        assert lease._stalled_probe is stalled_compilation_job_ids, (
            "the compilation lease was constructed without a stalled_probe, so its "
            "heartbeat renews every held lease forever and the lease-gated reaper "
            "can never fail a compile wedged on a live pod"
        )

    async def test_the_probe_asks_the_reaper_s_own_question(self) -> None:
        """Probe and reaper must not diverge.

        A probe with a SHORTER window would mass-expire healthy long compiles; a
        LONGER one restores the bug. Sharing the literal query and limit is what
        makes divergence impossible, so pin that they are shared.
        """
        captured: dict = {}

        class _Result:
            @staticmethod
            def all() -> list[tuple[str]]:
                return [("wedged-compile",)]

        class _DB:
            async def execute(self, stmt, params):
                captured["sql"] = str(stmt)
                captured["params"] = params
                return _Result()

        class _Ctx:
            async def __aenter__(self):
                return _DB()

            async def __aexit__(self, *exc):
                return False

        with patch("src.services.compilation.cleanup_service.get_db_session", return_value=_Ctx()):
            stalled = await stalled_compilation_job_ids()

        assert stalled == {"wedged-compile"}
        assert captured["sql"] == _STALE_COMPILATION_QUERY
        assert captured["params"]["lim"] == _STALE_COMPILATION_LIMIT


class TestTheProbeActuallyStopsTheRenewal:
    async def test_a_stalled_compilation_lease_is_dropped_not_renewed(self) -> None:
        """The consequence the wiring buys, asserted end-to-end on the lease.

        Without the probe this loop calls ``raw.set`` for the wedged job on every
        tick, so ``present()`` stays True and the reaper skips it forever.
        """
        raw = SimpleNamespace(set=AsyncMock(), delete=AsyncMock())
        lease = JobLease(
            raw,
            prefix="compilation",
            ttl_seconds=1,
            heartbeat_seconds=0.01,
            stalled_probe=AsyncMock(return_value={"wedged-compile"}),
        )
        await lease.acquire("wedged-compile")
        await lease.acquire("healthy-compile")
        raw.set.reset_mock()

        # Drive exactly one heartbeat tick.
        import asyncio

        await asyncio.sleep(0.05)

        renewed = {call.args[0] for call in raw.set.call_args_list}
        assert lease.key("wedged-compile") not in renewed, (
            "the wedged job's lease was renewed; the reaper will keep skipping it"
        )
        assert lease.key("healthy-compile") in renewed
        await lease.release_all()
