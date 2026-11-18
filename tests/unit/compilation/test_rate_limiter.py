"""Unit tests for rate limiting middleware and helpers."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import RateLimitSettings
from src.services.compilation.rate_limiter import (
    _HEALTH_PATHS,
    RateLimitMiddleware,
    _inmemory_timestamps,
    check_concurrent_limit,
    check_daily_quota,
)

# ---------------------------------------------------------------------------
# TestRateLimitMiddlewareDispatch
# ---------------------------------------------------------------------------


class TestRateLimitMiddlewareDispatch:
    """Tests for RateLimitMiddleware.dispatch behaviour."""

    def _make_middleware(self, *, enabled: bool = True, rpm: int = 5) -> RateLimitMiddleware:
        settings = RateLimitSettings(enabled=enabled, requests_per_minute=rpm)
        return RateLimitMiddleware(app=MagicMock(), settings=settings)

    def _make_request(self, path: str = "/api/compile", tenant: str = "test-tenant"):
        request = MagicMock()
        request.url.path = path
        request.state.tenant_id = tenant
        return request

    async def test_disabled_passes_through(self):
        """When rate limiting is disabled, call_next is invoked without checking rate."""
        middleware = self._make_middleware(enabled=False)
        request = self._make_request()
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert response.status_code == 200

    async def test_health_path_bypasses_rate_limiting(self):
        """Health-check paths skip rate limiting entirely."""
        middleware = self._make_middleware(enabled=True)
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        for path in _HEALTH_PATHS:
            request = self._make_request(path=path)
            response = await middleware.dispatch(request, call_next)
            assert response.status_code == 200

        assert call_next.await_count == len(_HEALTH_PATHS)

    async def test_429_returned_when_rate_exceeded(self):
        """A 429 JSON response is returned when the rate check says exceeded."""
        middleware = self._make_middleware(enabled=True)
        request = self._make_request()
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch.object(
            middleware, "_check_rate", new_callable=AsyncMock, return_value=(True, 42)
        ):
            response = await middleware.dispatch(request, call_next)

        assert response.status_code == 429
        assert response.headers["Retry-After"] == "42"
        call_next.assert_not_awaited()

    async def test_under_limit_passes_through(self):
        """When under the limit, the request passes to call_next."""
        middleware = self._make_middleware(enabled=True)
        request = self._make_request()
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch.object(
            middleware, "_check_rate", new_callable=AsyncMock, return_value=(False, 0)
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once_with(request)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestCheckRateInmemory
# ---------------------------------------------------------------------------


class TestCheckRateInmemory:
    """Tests for the in-memory rate check fallback."""

    def setup_method(self):
        _inmemory_timestamps.clear()

    def _make_middleware(self, rpm: int = 5) -> RateLimitMiddleware:
        settings = RateLimitSettings(enabled=True, requests_per_minute=rpm)
        return RateLimitMiddleware(app=MagicMock(), settings=settings)

    def test_under_limit_returns_false(self):
        """Below the RPM limit, exceeded is False and retry_after is 0."""
        middleware = self._make_middleware(rpm=5)
        exceeded, retry_after = middleware._check_rate_inmemory("tenant-a")
        assert exceeded is False
        assert retry_after == 0

    def test_over_limit_returns_true(self):
        """At or over the RPM limit, exceeded is True with a positive retry_after."""
        middleware = self._make_middleware(rpm=3)
        now = time.time()
        _inmemory_timestamps["tenant-b"] = [now - 10, now - 5, now - 1]

        exceeded, retry_after = middleware._check_rate_inmemory("tenant-b")
        assert exceeded is True
        assert retry_after > 0

    def test_window_expiry_removes_old_timestamps(self):
        """Timestamps older than 60 seconds are pruned and do not count."""
        middleware = self._make_middleware(rpm=3)
        old_time = time.time() - 120  # 2 minutes ago -- well outside the window
        _inmemory_timestamps["tenant-c"] = [old_time, old_time + 1, old_time + 2]

        exceeded, retry_after = middleware._check_rate_inmemory("tenant-c")
        assert exceeded is False
        assert retry_after == 0


# ---------------------------------------------------------------------------
# TestConcurrentLimit
# ---------------------------------------------------------------------------


class TestConcurrentLimit:
    """Tests for check_concurrent_limit helper."""

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_under_limit_returns_true(self, mock_redis_fn):
        """Under the concurrent limit returns True."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "1"
        mock_redis_fn.return_value = mock_client

        result = await check_concurrent_limit("tenant-x", max_concurrent=3)
        assert result is True

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_at_limit_returns_false(self, mock_redis_fn):
        """At or over the concurrent limit returns False."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "3"
        mock_redis_fn.return_value = mock_client

        result = await check_concurrent_limit("tenant-x", max_concurrent=3)
        assert result is False

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_redis_failure_returns_true(self, mock_redis_fn):
        """Redis failure is fail-open -- returns True."""
        mock_redis_fn.side_effect = ConnectionError("redis down")

        result = await check_concurrent_limit("tenant-x", max_concurrent=3)
        assert result is True


# ---------------------------------------------------------------------------
# TestDailyQuota
# ---------------------------------------------------------------------------


class TestDailyQuota:
    """Tests for check_daily_quota helper."""

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_under_quota_returns_true(self, mock_redis_fn):
        """Under the daily quota returns True."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "50"
        mock_redis_fn.return_value = mock_client

        result = await check_daily_quota("tenant-x", daily_quota=100)
        assert result is True

    @patch("src.services.compilation.rate_limiter._redis_client", new_callable=AsyncMock)
    async def test_at_quota_returns_false(self, mock_redis_fn):
        """At or over the daily quota returns False."""
        mock_client = AsyncMock()
        mock_client.get.return_value = "100"
        mock_redis_fn.return_value = mock_client

        result = await check_daily_quota("tenant-x", daily_quota=100)
        assert result is False
