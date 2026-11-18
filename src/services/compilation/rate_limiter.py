"""Rate limiting middleware and helpers for per-tenant request throttling.

Uses Redis sorted sets for sliding-window request-per-minute tracking with
an in-memory fallback when Redis is unavailable.  Helper functions manage
concurrent-job and daily-quota counters stored in Redis.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import Request
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.config import RateLimitSettings, get_settings

logger = logging.getLogger(__name__)

# Every degrade path in this module (in-memory fallback, "allowing request",
# skipped counter update) is reached from an ``except`` clause, so the clause has
# to name what redis-py actually raises. redis-py declares its own
# ``ConnectionError``/``TimeoutError`` under ``RedisError(Exception)`` — same
# names as the builtins, unrelated classes, and no ``OSError`` ancestry. Catching
# only the builtins made all six handlers unreachable, turning the documented
# fallback into a hard failure of the request. The builtins remain for
# socket-level faults raised beneath the connection pool.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RedisError)

# Paths that bypass rate limiting entirely.
_HEALTH_PATHS: frozenset[str] = frozenset(
    {
        "/health",
        "/health/live",
        "/health/ready",
    }
)

# ---------------------------------------------------------------------------
# In-memory fallback state (used when Redis is unreachable)
# ---------------------------------------------------------------------------
_inmemory_timestamps: defaultdict[str, list[float]] = defaultdict(list)


def _prune_inmemory(tenant_id: str, window: float = 60.0) -> None:
    """Remove timestamps older than *window* seconds for *tenant_id*."""
    cutoff = time.time() - window
    _inmemory_timestamps[tenant_id] = [ts for ts in _inmemory_timestamps[tenant_id] if ts > cutoff]


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces per-tenant request-per-minute limits.

    When Redis is available the middleware uses a sorted-set sliding window
    (``ratelimit:{tenant_id}:rpm``).  If Redis cannot be reached it falls
    back to a process-local ``defaultdict[str, list[float]]`` so the service
    stays available (with degraded accuracy across workers).
    """

    def __init__(
        self,
        app: ASGIApp,
        settings: RateLimitSettings | None = None,
    ) -> None:
        super().__init__(app)
        cfg = settings or get_settings().rate_limit
        self._enabled: bool = cfg.enabled
        self._requests_per_minute: int = cfg.requests_per_minute

    # ------------------------------------------------------------------

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._enabled:
            return await call_next(request)

        path = request.url.path.rstrip("/") or "/"
        if path in _HEALTH_PATHS:
            return await call_next(request)

        tenant_id: str = getattr(request.state, "tenant_id", "anonymous")

        exceeded, retry_after = await self._check_rate(tenant_id)
        if exceeded:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)

    # ------------------------------------------------------------------

    async def _check_rate(
        self,
        tenant_id: str,
    ) -> tuple[bool, int]:
        """Return ``(exceeded, retry_after_seconds)``.

        Tries Redis first; falls back to in-memory tracking on failure.
        """
        try:
            return await self._check_rate_redis(tenant_id)
        except _REDIS_ERRORS:
            logger.warning(
                "Redis unavailable for rate limiting - "
                "falling back to in-memory tracking (tenant=%s)",
                tenant_id,
            )
            return self._check_rate_inmemory(tenant_id)

    # -- Redis path -----------------------------------------------------

    async def _check_rate_redis(
        self,
        tenant_id: str,
    ) -> tuple[bool, int]:
        from w2v_common.cache.redis_client import get_redis_client

        redis_client = get_redis_client()
        client = redis_client._ensure_connected()

        key = f"ratelimit:{tenant_id}:rpm"
        now = time.time()
        window_start = now - 60.0

        pipe = client.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, 120)
        results = await pipe.execute()

        count: int = results[2]

        if count > self._requests_per_minute:
            # Determine how long until the oldest entry in the window
            # expires so the caller knows when to retry.
            oldest = await client.zrange(key, 0, 0, withscores=True)
            retry_after = max(1, int(60 - (now - float(oldest[0][1])))) if oldest else 60
            return True, retry_after

        return False, 0

    # -- In-memory path -------------------------------------------------

    def _check_rate_inmemory(
        self,
        tenant_id: str,
    ) -> tuple[bool, int]:
        now = time.time()
        _prune_inmemory(tenant_id)

        if len(_inmemory_timestamps[tenant_id]) >= self._requests_per_minute:
            oldest = _inmemory_timestamps[tenant_id][0]
            retry_after = max(1, int(60 - (now - oldest)))
            return True, retry_after

        _inmemory_timestamps[tenant_id].append(now)
        return False, 0


# ---------------------------------------------------------------------------
# Helper functions — concurrent jobs & daily quota
# ---------------------------------------------------------------------------


async def _redis_client() -> Any:
    """Return the raw ``redis.asyncio.Redis`` instance."""
    from w2v_common.cache.redis_client import get_redis_client

    return get_redis_client()._ensure_connected()


async def check_concurrent_limit(
    tenant_id: str,
    max_concurrent: int,
) -> bool:
    """Return ``True`` if *tenant_id* has fewer than *max_concurrent* active jobs."""
    try:
        client = await _redis_client()
        key = f"ratelimit:{tenant_id}:concurrent"
        value = await client.get(key)
        current = int(value) if value is not None else 0
        return current < max_concurrent
    except _REDIS_ERRORS:
        logger.warning(
            "Redis unavailable for concurrent-limit check (tenant=%s) - allowing request",
            tenant_id,
        )
        return True


async def check_daily_quota(
    tenant_id: str,
    daily_quota: int,
) -> bool:
    """Return ``True`` if *tenant_id* is under the daily compilation quota."""
    try:
        client = await _redis_client()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        key = f"ratelimit:{tenant_id}:daily:{today}"
        value = await client.get(key)
        current = int(value) if value is not None else 0
        return current < daily_quota
    except _REDIS_ERRORS:
        logger.warning(
            "Redis unavailable for daily-quota check (tenant=%s) - allowing request",
            tenant_id,
        )
        return True


async def increment_concurrent(tenant_id: str) -> None:
    """Increment the concurrent-job counter for *tenant_id*."""
    try:
        client = await _redis_client()
        key = f"ratelimit:{tenant_id}:concurrent"
        await client.incr(key)
    except _REDIS_ERRORS:
        logger.warning(
            "Redis unavailable - could not increment concurrent counter (tenant=%s)",
            tenant_id,
        )


async def decrement_concurrent(tenant_id: str) -> None:
    """Decrement the concurrent-job counter for *tenant_id* (minimum 0)."""
    try:
        client = await _redis_client()
        key = f"ratelimit:{tenant_id}:concurrent"
        value = await client.decr(key)
        # Guard against going negative (e.g. after a restart).
        if int(value) < 0:
            await client.set(key, 0)
    except _REDIS_ERRORS:
        logger.warning(
            "Redis unavailable - could not decrement concurrent counter (tenant=%s)",
            tenant_id,
        )


async def increment_daily_quota(tenant_id: str) -> None:
    """Increment the daily compilation counter for *tenant_id*.

    The key auto-expires after 86 400 seconds (24 h).
    """
    try:
        client = await _redis_client()
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        key = f"ratelimit:{tenant_id}:daily:{today}"
        await client.incr(key)
        await client.expire(key, 86400)
    except _REDIS_ERRORS:
        logger.warning(
            "Redis unavailable - could not increment daily quota (tenant=%s)",
            tenant_id,
        )
