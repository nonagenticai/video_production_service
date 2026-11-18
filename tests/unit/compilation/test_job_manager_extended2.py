"""JobManager.get_usage_daily coerces NULL aggregate columns to 0 on billing usage rows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.services.compilation.job_manager import CompilationJobManager as JobManager
from tests.conftest import TENANT_U4


def _mock_db():
    """Create a mock async DB session."""
    return AsyncMock()


class TestJobManagerGetUsageDaily:
    """Tests for JobManager.get_usage_daily (lines 337-345)."""

    async def test_usage_daily_with_none_values(self):
        """Daily rows with None values default to 0."""
        db = _mock_db()
        row = MagicMock()
        row.date = "2026-02-01"
        row.compilations_started = None
        row.compilations_completed = None
        row.compilations_failed = None
        row.total_output_duration_seconds = None
        row.total_output_size_bytes = None

        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[row]))
        db.execute.return_value = mock_result

        result = await JobManager.get_usage_daily(db, tenant_id=TENANT_U4)

        assert len(result) == 1
        assert result[0].compilations_started == 0
        assert result[0].compilations_completed == 0
        assert result[0].compilations_failed == 0
        assert result[0].total_output_duration_seconds == 0.0
        assert result[0].total_output_size_bytes == 0
