"""Tests for src/middleware/rate_limiter.py -- Redis-based rate limiting middleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.middleware.rate_limiter import RateLimitMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(path: str = "/api/record/part", tenant: str = "test-tenant"):
    """Create a mock Request."""
    request = MagicMock()
    request.url.path = path
    request.state.tenant_id = tenant
    request.app.state.redis_client = None
    return request


def _make_settings(max_requests: int = 60, window_seconds: int = 60):
    """Create mock settings for rate limiter."""
    settings = MagicMock()
    settings.api.rate_limit_requests = max_requests
    settings.api.rate_limit_window_seconds = window_seconds
    return settings


# ---------------------------------------------------------------------------
# TestRateLimitMiddleware
# ---------------------------------------------------------------------------


class TestRateLimitMiddleware:
    """Tests for the middleware's dispatch method."""

    async def test_non_api_path_passes_through(self):
        """Non-rate-limited paths bypass the middleware entirely."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/health")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        response = await middleware.dispatch(request, call_next)
        call_next.assert_awaited_once_with(request)
        assert response.status_code == 200

    async def test_api_record_path_is_rate_limited(self):
        """Paths starting with /api/record are subject to rate limiting."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/record/part")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        # No Redis client -> graceful degradation
        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200

    async def test_api_compile_path_is_rate_limited(self):
        """Paths starting with /api/compile are subject to rate limiting."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/compile")
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200

    async def test_graceful_degradation_without_redis(self):
        """When redis_client is None, requests pass through without rate limiting."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/record/batch")
        request.app.state.redis_client = None
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200

    async def test_rate_exceeded_returns_429(self):
        """When the sliding window count is at or above max_requests, return 429."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/record/part")

        mock_redis = MagicMock()
        mock_pipe = AsyncMock()
        # zremrangebyscore result, zcard result (at limit), zadd, expire
        mock_pipe.execute = AsyncMock(return_value=[0, 60, True, True])
        mock_redis._ensure_connected.return_value.pipeline.return_value = mock_pipe

        request.app.state.redis_client = mock_redis
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(max_requests=60, window_seconds=60),
        ):
            response = await middleware.dispatch(request, call_next)

        assert response.status_code == 429
        call_next.assert_not_awaited()

    async def test_under_limit_passes_through(self):
        """When the sliding window count is under max_requests, pass through."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/record/part")

        mock_redis = MagicMock()
        mock_pipe = AsyncMock()
        # zcard result is 5 (well under limit of 60)
        mock_pipe.execute = AsyncMock(return_value=[0, 5, True, True])
        mock_redis._ensure_connected.return_value.pipeline.return_value = mock_pipe

        request.app.state.redis_client = mock_redis
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(max_requests=60, window_seconds=60),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200

    async def test_redis_error_allows_request(self):
        """When Redis pipeline raises, the request passes through (fail-open)."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/record/part")

        mock_redis = MagicMock()
        mock_redis._ensure_connected.side_effect = ConnectionError("redis down")

        request.app.state.redis_client = mock_redis
        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200

    async def test_anonymous_tenant_when_no_state(self):
        """When request.state has no tenant_id, defaults to 'anonymous'."""
        middleware = RateLimitMiddleware(app=MagicMock())
        request = _make_request(path="/api/compile/batch")
        del request.state.tenant_id  # remove tenant_id attr

        call_next = AsyncMock(return_value=MagicMock(status_code=200))

        with patch(
            "src.middleware.rate_limiter.get_settings",
            return_value=_make_settings(),
        ):
            response = await middleware.dispatch(request, call_next)

        call_next.assert_awaited_once()
        assert response.status_code == 200
