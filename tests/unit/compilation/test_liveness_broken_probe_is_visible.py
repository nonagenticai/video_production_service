""" "Cannot measure the executor" must not be served as a clean, disabled "ok".

``/health/live`` exists to catch a WEDGED recording executor (the 2026-09-04
9-hour outage: leaked fleet slots, every job stuck ``queued``, ``/healthz``
green throughout). The check is skipped in four situations, and before this
change all four produced the byte-identical body ``{"status": "ok"}``:

* the operator disabled it (``threshold <= 0``) — a CHOICE;
* ``concurrency_manager`` is absent from ``app.state`` (startup ordering);
* ``stall_status`` was renamed by a refactor;
* ``stall_status()`` raised.

The last three mean the wedge detector is silently GONE and liveness has
degraded to a bare process-alive check — forever, with nothing to alert on.
These tests pin that the broken cases are now DISTINGUISHABLE from the
disabled one, and that none of them fails the probe (a crashloop over a probe
bug is worse than a blind probe).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse

from src.api.health import health_live
from src.config import get_settings


async def _live(concurrency_manager=None) -> dict:
    """Call the probe and assert it did NOT fail liveness, returning the 200 body.

    A probe bug must never restart the pod, so every case here must come back as
    a plain report rather than a 503 ``JSONResponse``.
    """
    result = await health_live(_request(concurrency_manager=concurrency_manager))
    assert not isinstance(result, JSONResponse), "a probe bug must never fail liveness"
    return dict(result)


def _request(concurrency_manager=None):
    request = MagicMock()
    request.app.state.concurrency_manager = concurrency_manager
    return request


@pytest.fixture
def threshold_enabled():
    """Force the stall check ON regardless of ambient settings."""
    settings = get_settings()
    original = settings.recording.executor_stall_threshold_seconds
    settings.recording.executor_stall_threshold_seconds = 1800.0
    yield
    settings.recording.executor_stall_threshold_seconds = original


class TestBrokenProbeIsVisible:
    async def test_missing_manager_is_reported_not_silently_ok(self, threshold_enabled):
        result = await _live(concurrency_manager=None)

        assert result["status"] == "ok"
        assert result["executor_check"] == "manager_missing"

    async def test_renamed_stall_status_is_reported(self, threshold_enabled):
        manager = MagicMock(spec=[])  # no stall_status attribute at all
        result = await _live(concurrency_manager=manager)

        assert result["status"] == "ok"
        assert result["executor_check"] == "stall_status_unavailable"

    async def test_raising_stall_status_is_reported(self, threshold_enabled):
        manager = MagicMock()
        manager.stall_status.side_effect = RuntimeError("probe bug")
        result = await _live(concurrency_manager=manager)

        assert result["status"] == "ok"
        assert result["executor_check"] == "stall_status_error"

    async def test_disabled_is_distinguishable_from_broken(self, threshold_enabled):
        """The whole point: the two bodies must not be equal."""
        settings = get_settings()
        settings.recording.executor_stall_threshold_seconds = 0.0
        disabled = await _live(concurrency_manager=None)
        settings.recording.executor_stall_threshold_seconds = 1800.0
        broken = await _live(concurrency_manager=None)

        assert disabled["executor_check"] == "disabled"
        assert broken["executor_check"] == "manager_missing"
        assert disabled != broken, (
            "a broken wedge detector is being served as an operator's deliberate opt-out"
        )

    async def test_working_check_still_reports_the_executor(self, threshold_enabled):
        manager = MagicMock()
        manager.stall_status.return_value = {
            "stalled": False,
            "queued": 0,
            "seconds_since_progress": None,
            "threshold_seconds": 1800.0,
        }
        result = await _live(concurrency_manager=manager)

        assert result["executor"]["stalled"] is False
        assert "executor_check" not in result
