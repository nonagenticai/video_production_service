"""WebSocket connection manager for real-time progress streaming.

Multi-replica note: the in-memory ``ConnectionManager`` below is pod-affine — a
client only sees progress if it happens to connect to the pod running the job.
Under ``replicas>1`` that breaks. The fix (see ``job_progress_ws`` in
``routes/_compilation_jobs.py``) mirrors the SSE path in ``api/events.py``:
:meth:`ConnectionManager.broadcast` ALSO publishes each progress message to the
per-job Redis channel ``compilation:job:{job_id}``, and the WS endpoint
subscribes to that channel, so ANY pod delivers progress regardless of which pod
runs the compilation. The in-memory path is retained as a same-pod fast path /
Redis-down fallback (and for the existing unit tests).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import WebSocket
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# redis-py exceptions don't subclass the builtins, so a mid-flight publish failure
# (ConnectionError / TimeoutError / ResponseError) would slip past a builtin-only
# tuple and escape broadcast() → fail a live compile job. RedisError is their base.
_REDIS_PUBLISH_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError)


def compilation_job_channel(job_id: str) -> str:
    """Redis pub/sub channel carrying progress for one compilation job."""
    return f"compilation:job:{job_id}"


class MessageBuffer:
    """In-memory ring buffer for WebSocket message persistence (WP-7).

    Stores recent messages per job so late-connecting clients can replay
    missed progress updates.
    """

    def __init__(self, max_messages: int = 50, ttl_seconds: float = 3600.0) -> None:
        self._max_messages = max_messages
        self._ttl = ttl_seconds
        self._buffers: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=max_messages))
        self._timestamps: dict[str, float] = {}

    def store(self, job_id: str, message: dict) -> None:
        """Store a message in the buffer for the given job."""
        self._buffers[job_id].append(message)
        self._timestamps[job_id] = time.monotonic()

    def get_messages(self, job_id: str) -> list[dict]:
        """Get all buffered messages for a job."""
        self._expire(job_id)
        return list(self._buffers.get(job_id, []))

    def clear(self, job_id: str) -> None:
        """Clear the buffer for a job."""
        self._buffers.pop(job_id, None)
        self._timestamps.pop(job_id, None)

    def _expire(self, job_id: str) -> None:
        """Remove buffer if TTL has elapsed since last message."""
        ts = self._timestamps.get(job_id)
        if ts and (time.monotonic() - ts) > self._ttl:
            self.clear(job_id)


class ConnectionManager:
    """Manages WebSocket connections grouped by job_id."""

    def __init__(
        self,
        message_buffer: MessageBuffer | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._buffer = message_buffer or MessageBuffer()
        self._redis = redis_client

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        """Accept a WebSocket connection and replay any buffered messages."""
        await websocket.accept()
        async with self._lock:
            self._connections[job_id].append(websocket)
        # WP-7: Replay buffered messages for late-connecting clients
        for msg in self._buffer.get_messages(job_id):
            try:
                await websocket.send_json(msg)
            except OSError, ConnectionError, TimeoutError, RuntimeError:
                break
        logger.debug("WebSocket connected for job %s", job_id)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        """Remove a WebSocket connection from the job's connection list."""
        async with self._lock:
            conns = self._connections.get(job_id, [])
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                self._connections.pop(job_id, None)
        logger.debug("WebSocket disconnected for job %s", job_id)

    async def broadcast(self, job_id: str, data: dict) -> None:
        """Send a message to all connections watching a job.

        Publishes to the per-job Redis channel first so a client connected to a
        DIFFERENT replica (subscribed via ``job_progress_ws``) still receives the
        update, then delivers to any locally-connected sockets (same-pod fast
        path + Redis-down fallback).
        """
        # WP-7: Store in buffer for replay
        self._buffer.store(job_id, data)
        await self._publish_redis(job_id, data)
        async with self._lock:
            conns = list(self._connections.get(job_id, []))
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except OSError, ConnectionError, TimeoutError, RuntimeError:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    conns_list = self._connections.get(job_id, [])
                    if ws in conns_list:
                        conns_list.remove(ws)

    async def _publish_redis(self, job_id: str, data: dict) -> None:
        """Best-effort publish of a progress message to the per-job Redis channel."""
        if self._redis is None:
            return
        try:
            await self._redis.publish(compilation_job_channel(job_id), json.dumps(data))
        except _REDIS_PUBLISH_ERRORS as exc:
            # Degrade to local-only delivery on any Redis blip; NEVER let a publish
            # failure escape broadcast() and fail the live compilation job.
            logger.warning("Failed to publish WS progress to Redis for job %s: %s", job_id, exc)


class ProgressReporter:
    """Wraps ConnectionManager to emit structured progress for a specific job."""

    def __init__(self, manager: ConnectionManager, job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    async def report(
        self,
        stage: str,
        progress_pct: float,
        message: str = "",
        current_part: int | None = None,
        total_parts: int | None = None,
    ) -> None:
        """Broadcast a progress update for the tracked job."""
        data: dict = {
            "event": "progress",
            "job_id": self._job_id,
            "stage": stage,
            "progress_pct": round(progress_pct, 1),
            "message": message,
        }
        if current_part is not None:
            data["current_part"] = current_part
        if total_parts is not None:
            data["total_parts"] = total_parts
        await self._manager.broadcast(self._job_id, data)

    async def complete(self) -> None:
        """Broadcast a completion event for the tracked job."""
        await self._manager.broadcast(
            self._job_id,
            {
                "event": "complete",
                "job_id": self._job_id,
                "stage": "complete",
                "progress_pct": 100.0,
            },
        )

    async def error(self, message: str) -> None:
        """Broadcast an error event for the tracked job."""
        await self._manager.broadcast(
            self._job_id,
            {
                "event": "error",
                "job_id": self._job_id,
                "stage": "error",
                "message": message,
            },
        )
