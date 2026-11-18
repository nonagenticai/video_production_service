"""Tests for src/api/events.py -- SSE event streaming endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from starlette.responses import StreamingResponse

from src.api.events import stream_events

#: The path parameter and the caller tenant must both be REAL uuids now: w2v-common
#: 0.67.0 made tenant a ``uuid.UUID`` and ``stream_events`` parses both sides
#: before comparing, so a slug 403s regardless of whether the two sides match.
_TENANT = "019de900-0000-7000-8000-0000000000d1"

# ── Helpers ────────────────────────────────────────────────────────


def _make_request(redis_client=None, tenant_id: str = _TENANT):
    """Build a mock Request with optional redis_client on app.state.

    ``state.tenant_id`` must be a real string: ``stream_events`` now compares it
    against the path tenant and 403s on mismatch, and a bare ``MagicMock``
    attribute never equals the requested id. The cross-tenant guard itself is
    covered in ``tests/unit/test_events_tenant_guard.py``; these tests are the
    same-tenant (streaming) path.
    """
    request = MagicMock()
    request.app.state = MagicMock()
    request.state.tenant_id = tenant_id
    if redis_client is not None:
        request.app.state.redis_client = redis_client
    else:
        request.app.state.redis_client = None
    request.is_disconnected = AsyncMock(return_value=False)
    return request


# ── stream_events ────────────────────────────────────────────────


class TestStreamEvents:
    async def test_no_redis_returns_error_stream(self):
        request = _make_request(redis_client=None)
        response = await stream_events(_TENANT, request)
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        # Consume the iterator to check the content
        body_parts = []
        async for chunk in response.body_iterator:
            body_parts.append(chunk)
        body = "".join(body_parts)
        assert "error" in body
        assert "Redis not available" in body

    async def test_with_redis_returns_streaming_response(self):
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.reset = AsyncMock()

        # Simulate one message then client disconnects
        mock_pubsub.get_message = AsyncMock(
            side_effect=[
                {"type": "message", "data": b'{"job_id": "j1", "status": "completed"}'},
                None,  # keepalive
            ]
        )

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis._ensure_connected = MagicMock(return_value=mock_client)

        request = _make_request(redis_client=mock_redis)
        # After getting one message and one keepalive, simulate disconnect
        request.is_disconnected = AsyncMock(side_effect=[False, False, True])

        response = await stream_events(_TENANT, request)
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

        # Collect the events
        body_parts = []
        async for chunk in response.body_iterator:
            body_parts.append(chunk)

        body = "".join(body_parts)
        assert "connected" in body
        assert "j1" in body

    async def test_response_headers(self):
        # Use a redis client so the full StreamingResponse with custom headers is created
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.reset = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis._ensure_connected = MagicMock(return_value=mock_client)

        request = _make_request(redis_client=mock_redis)
        request.is_disconnected = AsyncMock(return_value=True)

        response = await stream_events(_TENANT, request)

        # SSE responses should have no-cache and keep-alive headers
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("connection") == "keep-alive"
        assert response.headers.get("x-accel-buffering") == "no"

    async def test_handles_stream_error_gracefully(self):
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.reset = AsyncMock()
        mock_pubsub.get_message = AsyncMock(side_effect=RuntimeError("Redis disconnected"))

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis._ensure_connected = MagicMock(return_value=mock_client)

        request = _make_request(redis_client=mock_redis)
        request.is_disconnected = AsyncMock(return_value=False)

        response = await stream_events(_TENANT, request)
        body_parts = []
        async for chunk in response.body_iterator:
            body_parts.append(chunk)

        body = "".join(body_parts)
        assert "error" in body
        assert "Redis disconnected" in body

    async def test_string_message_data(self):
        """Verify string data (not bytes) passes through correctly."""
        mock_redis = MagicMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.reset = AsyncMock()

        mock_pubsub.get_message = AsyncMock(
            side_effect=[
                {"type": "message", "data": '{"job_id": "j2", "status": "failed"}'},
            ]
        )

        mock_client = MagicMock()
        mock_client.pubsub.return_value = mock_pubsub
        mock_redis._ensure_connected = MagicMock(return_value=mock_client)

        request = _make_request(redis_client=mock_redis)
        request.is_disconnected = AsyncMock(side_effect=[False, True])

        response = await stream_events(_TENANT, request)
        body_parts = []
        async for chunk in response.body_iterator:
            body_parts.append(chunk)

        body = "".join(body_parts)
        assert "j2" in body
