"""Redis-based sliding window rate limiter middleware."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.config import get_settings

logger = logging.getLogger(__name__)

# redis-py raises its OWN ``ConnectionError``/``TimeoutError`` off
# ``RedisError(Exception)`` — same names as the builtins, different classes, and
# neither derives from ``OSError``. A builtin-only tuple is therefore dead code
# against a redis client. ``RedisError`` is the base and covers every redis-py
# failure; the builtins stay because a socket-level fault under the connection
# pool can still surface as ``OSError``/``TimeoutError``. Mirrors the
# ``_REDIS_ERRORS`` constants in ``src/services/job_lease.py`` and
# ``src/services/recording/concurrency.py``.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RedisError)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant rate limiting using Redis sliding window."""

    # `/events` and `/api/analytics` added with the SSE resource ceiling: the
    # limiter covered only the two job-creating prefixes, so the one route in the
    # service that opens an unbounded long-lived connection — `GET
    # /events/{tenant_id}` — was the one route with no ingress ceiling at all.
    # Paired with `APISettings.sse_max_stream_seconds`, this is what bounds
    # CONCURRENT streams: a tenant can open at most `rate_limit_requests` per
    # window, and each stream now dies after the cap, so concurrency is bounded
    # by roughly (rate x lifetime) instead of by nothing.
    #
    # An explicit per-tenant CONNECTION counter was considered and rejected: it
    # needs a Redis increment on open and a decrement on close, and the decrement
    # is exactly what a pod kill, an OOM or a half-open socket skips — the
    # counter then ratchets up and locks a paying tenant out of their own event
    # stream with no operator-visible cause. A rate limit plus a hard stream
    # lifetime bounds the same resource with no state that can leak.
    RATE_LIMITED_PREFIXES = ("/api/record", "/api/compile", "/api/analytics", "/events")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Apply sliding-window rate limiting on recording and compilation API endpoints.

        This middleware **fails open**. It runs on every request to the rate-limited
        prefixes, so if it raised on a Redis fault it would convert a degraded cache
        into a total outage of the recording and compilation APIs. A rate limiter
        that cannot reach its store must let traffic through, not become the outage
        it exists to prevent: any Redis failure is logged and the request proceeds
        unmetered.

        Args:
            request: Incoming HTTP request.
            call_next: Downstream handler.

        Returns:
            Downstream response, or 429 JSON on rate limit exceeded. On Redis
            failure, the downstream response (fail-open).
        """
        path = request.url.path

        # Only rate-limit recording and compilation API endpoints
        if not any(path.startswith(prefix) for prefix in self.RATE_LIMITED_PREFIXES):
            return await call_next(request)

        settings = get_settings()
        max_requests = settings.api.rate_limit_requests
        window_seconds = settings.api.rate_limit_window_seconds

        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        redis_client = getattr(request.app.state, "redis_client", None)

        if not redis_client:
            # Graceful degradation: allow requests if Redis is down
            return await call_next(request)

        key = f"ratelimit:{tenant_id}"
        now = time.time()
        window_start = now - window_seconds

        try:
            client = redis_client._ensure_connected()
            pipe = client.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(key, 0, window_start)
            # Count current entries
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {f"{now}": now})
            # Set TTL on the key
            pipe.expire(key, window_seconds)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= max_requests:
                retry_after = int(window_seconds - (now - window_start))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        except _REDIS_ERRORS as e:
            logger.warning("Rate limiter error (allowing request): %s", e)

        return await call_next(request)
