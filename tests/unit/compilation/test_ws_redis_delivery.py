"""Cross-pod compilation progress delivery via Redis pub/sub.

The compilation WS is Redis-backed so progress reaches a client regardless of
which replica runs the job: the pod running the job publishes to
``compilation:job:{job_id}`` and a client connected to ANY other pod (subscribed
to that channel) receives it. This test proves the publish → subscribe hop
across two Redis clients sharing one server (== two pods).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis
import fakeredis.aioredis
from fastapi import WebSocketDisconnect

from src.api.ws import ConnectionManager, ProgressReporter, compilation_job_channel
from tests.conftest import TENANT_U1


def _shared_pair() -> tuple:
    server = fakeredis.FakeServer()
    return (
        fakeredis.aioredis.FakeRedis(server=server, decode_responses=True),
        fakeredis.aioredis.FakeRedis(server=server, decode_responses=True),
    )


@contextlib.contextmanager
def _socket_owns_its_job(mock_ws, tenant=TENANT_U1):
    """Give the mock socket a real tenant string and a job it legitimately owns.

    ``job_progress_ws`` authorizes the job before subscribing: it seeds
    ``set_identity_context`` from ``websocket.state.tenant_id`` and asks
    ``CompilationJobManager.get_job`` (RLS decides) whether the job is visible.
    An ``AsyncMock`` answers every attribute with a child mock, so the tenant was
    an ``AsyncMock`` and ``set_identity_context`` refused it — by design; it
    rejects a non-string tenant so a spoofable identity cannot reach the GUC.
    Handing it a real uuid keeps this test about the CROSS-POD RELAY, which is
    what it was written to prove, instead of about the ownership gate.
    """
    mock_ws.state.tenant_id = str(tenant)
    mock_ws.state.subject_id = "user-1"

    @contextlib.asynccontextmanager
    async def _session():
        yield object()

    with (
        patch("src.api.routes.compilation.get_db_session", _session),
        patch(
            "src.api.routes._compilation_jobs.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
    ):
        yield mock_ws


async def _read_message(pubsub, tries: int = 30) -> dict | None:
    for _ in range(tries):
        m = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
        if m and m["type"] == "message":
            return m
    return None


class TestCrossPodDelivery:
    async def test_progress_published_reaches_a_client_on_another_pod(self):
        producer_redis, consumer_redis = _shared_pair()

        # Consumer pod subscribes to the per-job channel (what job_progress_ws does).
        pubsub = consumer_redis.pubsub()
        await pubsub.subscribe(compilation_job_channel("job-42"))

        # Producer pod runs the job and reports progress.
        mgr = ConnectionManager(redis_client=producer_redis)
        reporter = ProgressReporter(mgr, "job-42")
        await reporter.report("encoding", 50.0, "half way")

        msg = await _read_message(pubsub)
        assert msg is not None, "progress was not delivered across pods"
        data = json.loads(msg["data"])
        assert data["job_id"] == "job-42"
        assert data["progress_pct"] == 50.0
        assert data["stage"] == "encoding"

        await pubsub.unsubscribe()
        await pubsub.aclose()

    async def test_complete_event_published(self):
        producer_redis, consumer_redis = _shared_pair()
        pubsub = consumer_redis.pubsub()
        await pubsub.subscribe(compilation_job_channel("job-done"))

        mgr = ConnectionManager(redis_client=producer_redis)
        await ProgressReporter(mgr, "job-done").complete()

        msg = await _read_message(pubsub)
        assert msg is not None
        data = json.loads(msg["data"])
        assert data["event"] == "complete"
        assert data["progress_pct"] == 100.0

        await pubsub.unsubscribe()
        await pubsub.aclose()

    async def test_no_redis_broadcast_is_still_in_memory(self):
        """Without Redis, broadcast still works (in-memory, pod-affine) and does
        not raise."""
        mgr = ConnectionManager()  # no redis_client
        await ProgressReporter(mgr, "job-local").report("x", 10.0)
        # Buffered for replay to a same-pod late joiner.
        assert mgr._buffer.get_messages("job-local")

    async def test_publish_redis_error_does_not_fail_the_job(self):
        """A redis-py ConnectionError raised MID-PUBLISH (not a builtin subclass)
        must NOT escape broadcast() — it degrades to local-only delivery."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        class _FailPublish:
            async def publish(self, *a, **k):
                raise RedisConnectionError("dropped mid-flight")

        mgr = ConnectionManager(redis_client=_FailPublish())
        # broadcast() (via report) must complete without raising.
        await ProgressReporter(mgr, "job-blip").report("encoding", 5.0)
        # In-memory buffer still populated for a same-pod client.
        assert mgr._buffer.get_messages("job-blip")

    async def test_ws_handler_relays_redis_messages_to_socket(self):
        """The WS handler (Redis path) subscribes and relays a published message
        to the connected socket regardless of which pod published it."""
        from src.api.routes.compilation import job_progress_ws

        producer_redis, consumer_redis = _shared_pair()

        mock_ws = AsyncMock()
        mock_ws.app.state.redis_client = consumer_redis
        # First relayed message ends the loop (as a disconnected client would).
        mock_ws.send_text.side_effect = WebSocketDisconnect()

        with _socket_owns_its_job(mock_ws):
            handler = asyncio.create_task(job_progress_ws(mock_ws, "job-relay"))
            await asyncio.sleep(0.05)  # let the handler authorize + subscribe
            await producer_redis.publish(
                compilation_job_channel("job-relay"), json.dumps({"event": "progress", "pct": 25})
            )
            await asyncio.wait_for(handler, timeout=3.0)

        mock_ws.accept.assert_awaited_once()
        mock_ws.send_text.assert_awaited()
        sent = mock_ws.send_text.await_args.args[0]
        assert json.loads(sent)["pct"] == 25
