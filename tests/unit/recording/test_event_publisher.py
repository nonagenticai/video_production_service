"""Tests for src/services/event_publisher.py -- EventPublisher."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from src.services.recording.event_publisher import EventPublisher
from tests.conftest import TENANT_U1, TENANT_U4

# ── Helpers ────────────────────────────────────────────────────────


def _make_publisher(redis_mock=None):
    redis = redis_mock or AsyncMock()
    return EventPublisher(redis_client=redis), redis


# ── publish_status ────────────────────────────────────────────────


class TestPublishStatus:
    async def test_publishes_to_correct_channel(self):
        publisher, redis = _make_publisher()
        await publisher.publish_status(TENANT_U4, "j1", "completed")

        redis.publish.assert_awaited_once()
        call_args = redis.publish.call_args
        channel = call_args[0][0]
        assert channel == f"recording:jobs:{TENANT_U4}"

        payload = json.loads(call_args[0][1])
        assert payload["event"] == "status_changed"
        assert payload["job_id"] == "j1"
        assert payload["status"] == "completed"

    async def test_handles_redis_error_gracefully(self):
        redis = AsyncMock()
        redis.publish.side_effect = ConnectionError("Redis down")
        publisher = EventPublisher(redis_client=redis)

        # Should not raise
        await publisher.publish_status(TENANT_U1, "j1", "failed")

    async def test_payload_includes_timestamp(self):
        publisher, redis = _make_publisher()
        await publisher.publish_status(TENANT_U1, "j1", "processing")

        payload = json.loads(redis.publish.call_args[0][1])
        assert "timestamp" in payload


# ── publish ──────────────────────────────────────────────────────


class TestPublish:
    async def test_arbitrary_channel_and_data(self):
        publisher, redis = _make_publisher()
        await publisher.publish("custom:channel", {"foo": "bar"})

        redis.publish.assert_awaited_once()
        channel = redis.publish.call_args[0][0]
        assert channel == "custom:channel"

        payload = json.loads(redis.publish.call_args[0][1])
        assert payload["foo"] == "bar"
