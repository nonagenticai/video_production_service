"""Redis pub/sub event publisher for job status transitions.

Publishes structured JSON events to per-tenant Redis channels for
real-time SSE streaming to connected clients.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# redis-py defines its own ``ConnectionError``/``TimeoutError`` off
# ``RedisError(Exception)``: same names as the builtins, different classes, and
# none of them derive from ``OSError``. Catching only the builtins made these
# handlers dead code, so a Redis blip raised out of ``publish`` and aborted the
# recording that was merely trying to announce its progress. Event delivery is
# best-effort — SSE clients reconcile via polling — so the publisher swallows
# and logs. The builtins stay for socket-level faults under the connection pool.
_REDIS_ERRORS = (OSError, ConnectionError, TimeoutError, RedisError)


class EventPublisher:
    """Publish job status events to Redis pub/sub channels.

    Attributes:
        _redis: The async Redis client used for pub/sub.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    async def publish_status(
        self,
        tenant_id: uuid.UUID,
        job_id: str,
        status: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Publish a job status change event.

        Best-effort: a publish failure is logged and swallowed so that losing an
        SSE notification never fails the recording job that emitted it.

        Args:
            tenant_id: Tenant whose channel receives the event.
            job_id: Recording job the status belongs to.
            status: New job status.
            extra: Optional additional fields merged into the payload.
        """
        channel = f"recording:jobs:{tenant_id}"
        payload = {
            "event": "status_changed",
            "job_id": job_id,
            "status": status,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if extra:
            payload.update(extra)

        try:
            await self._redis.publish(channel, json.dumps(payload))
        except _REDIS_ERRORS as e:
            logger.warning("Failed to publish event: %s", e)

    async def publish(self, channel: str, data: dict[str, Any]) -> None:
        """Publish arbitrary data to a Redis channel.

        Best-effort, for the same reason as :meth:`publish_status`: a dropped
        event must not propagate as a failure into the caller's workflow.

        Args:
            channel: Redis pub/sub channel name.
            data: JSON-serializable payload.
        """
        try:
            await self._redis.publish(channel, json.dumps(data))
        except _REDIS_ERRORS as e:
            logger.warning("Failed to publish to %s: %s", channel, e)
