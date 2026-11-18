"""Tests for rate_limiter Redis paths and daily quota Redis failure.

Covers:
  - _check_rate_redis (lines 121-149)
  - _redis_client helper (lines 176-178)
  - check_daily_quota Redis failure fallback (lines 212-217)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.config import RateLimitSettings
from src.services.compilation.rate_limiter import (
    RateLimitMiddleware,
    check_daily_quota,
)


class TestCheckRateRedis:
    """Tests for _check_rate_redis with mocked Redis pipeline."""

    def _make_middleware(self, rpm: int = 30) -> RateLimitMiddleware:
        settings = RateLimitSettings(enabled=True, requests_per_minute=rpm)
        return RateLimitMiddleware(app=MagicMock(), settings=settings)

    @patch("w2v_common.cache.redis_client.get_redis_client")
    async def test_under_limit(self, mock_get_redis):
        """Under the limit: returns (False, 0)."""
        middleware = self._make_middleware(rpm=10)

        # The redis wrapper's _ensure_connected() is a sync method returning
        # the underlying redis client. The client's pipeline() is also sync,
        # but pipe.execute() is async.
        mock_client = MagicMock()
        mock_redis_wrapper = MagicMock()
        mock_redis_wrapper._ensure_connected.return_value = mock_client
        mock_get_redis.return_value = mock_redis_wrapper

        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(
            return_value=[None, None, 5, None]
        )  # count=5, under limit of 10
        mock_client.pipeline.return_value = mock_pipe

        exceeded, retry_after = await middleware._check_rate_redis("tenant-test")
        assert exceeded is False
        assert retry_after == 0

    @patch("w2v_common.cache.redis_client.get_redis_client")
    async def test_over_limit_with_oldest(self, mock_get_redis):
        """Over limit with oldest entry: returns (True, calculated_retry_after)."""
        middleware = self._make_middleware(rpm=5)

        import time

        now = time.time()

        mock_client = MagicMock()
        mock_redis_wrapper = MagicMock()
        mock_redis_wrapper._ensure_connected.return_value = mock_client
        mock_get_redis.return_value = mock_redis_wrapper

        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(
            return_value=[None, None, 6, None]
        )  # count=6, over limit of 5
        mock_client.pipeline.return_value = mock_pipe

        # oldest entry was 30 seconds ago
        mock_client.zrange = AsyncMock(return_value=[("entry", now - 30)])

        exceeded, retry_after = await middleware._check_rate_redis("tenant-test")
        assert exceeded is True
        assert 28 <= retry_after <= 31  # ~30 seconds, with timing tolerance

    @patch("w2v_common.cache.redis_client.get_redis_client")
    async def test_over_limit_no_oldest(self, mock_get_redis):
        """Over limit with no oldest entry: returns (True, 60)."""
        middleware = self._make_middleware(rpm=5)

        mock_client = MagicMock()
        mock_redis_wrapper = MagicMock()
        mock_redis_wrapper._ensure_connected.return_value = mock_client
        mock_get_redis.return_value = mock_redis_wrapper

        mock_pipe = MagicMock()
        mock_pipe.execute = AsyncMock(return_value=[None, None, 6, None])
        mock_client.pipeline.return_value = mock_pipe

        # No oldest entry
        mock_client.zrange = AsyncMock(return_value=[])

        exceeded, retry_after = await middleware._check_rate_redis("tenant-test")
        assert exceeded is True
        assert retry_after == 60


class TestRedisClientHelper:
    """Tests for the _redis_client helper function (lines 174-178)."""

    @patch("w2v_common.cache.redis_client.get_redis_client")
    async def test_redis_client_returns_connected_client(self, mock_get_redis):
        from src.services.compilation.rate_limiter import _redis_client

        mock_inner = MagicMock()
        mock_wrapper = MagicMock()
        mock_wrapper._ensure_connected.return_value = mock_inner
        mock_get_redis.return_value = mock_wrapper

        result = await _redis_client()
        assert result is mock_inner


class TestCheckDailyQuotaRedisFailure:
    """Test check_daily_quota when Redis fails (lines 212-217)."""

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_redis_failure_returns_true(self, mock_redis_fn):
        """When Redis is unavailable, check_daily_quota fails open (True)."""
        mock_redis_fn.side_effect = ConnectionError("redis down")

        result = await check_daily_quota("tenant-x", daily_quota=100)
        assert result is True

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_none_value_returns_true(self, mock_redis_fn):
        """When Redis returns None for the key, current=0 which is under quota."""
        mock_client = AsyncMock()
        mock_client.get.return_value = None
        mock_redis_fn.return_value = mock_client

        result = await check_daily_quota("tenant-x", daily_quota=100)
        assert result is True
