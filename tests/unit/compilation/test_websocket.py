"""Unit tests for WebSocket connection manager."""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.api.ws import ConnectionManager, ProgressReporter


class TestConnectionManager:
    async def test_connect_and_broadcast(self):
        manager = ConnectionManager()
        ws = AsyncMock()
        await manager.connect("job-1", ws)
        ws.accept.assert_called_once()
        await manager.broadcast("job-1", {"event": "test"})
        ws.send_json.assert_called_once_with({"event": "test"})

    async def test_disconnect(self):
        manager = ConnectionManager()
        ws = AsyncMock()
        await manager.connect("job-1", ws)
        await manager.disconnect("job-1", ws)
        # Broadcast should not call anything after disconnect
        ws.send_json.reset_mock()
        await manager.broadcast("job-1", {"event": "test"})
        ws.send_json.assert_not_called()

    async def test_dead_connection_cleanup(self):
        """A socket that raises is dropped from the job, and the healthy peer keeps its feed.

        Two behaviours, both invisible to a call count on the first broadcast:
        one dead socket must not swallow the message the *other* client is
        waiting for, and the dead socket must be de-registered so every later
        broadcast is not spent re-raising on a connection that is gone.
        """
        manager = ConnectionManager()
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send_json.side_effect = RuntimeError("connection lost")
        # Dead socket FIRST on purpose: if the broadcast loop aborted on the
        # first failure instead of continuing, ws_good would never be reached.
        await manager.connect("job-1", ws_bad)
        await manager.connect("job-1", ws_good)

        await manager.broadcast("job-1", {"event": "first"})
        ws_good.send_json.assert_awaited_once_with({"event": "first"})
        assert ws_bad not in manager._connections["job-1"]
        assert ws_good in manager._connections["job-1"]

        await manager.broadcast("job-1", {"event": "second"})
        assert [c.args[0] for c in ws_good.send_json.await_args_list] == [
            {"event": "first"},
            {"event": "second"},
        ]
        # The dead socket is never touched again.
        ws_bad.send_json.assert_awaited_once()

    async def test_multiple_connections_same_job(self):
        manager = ConnectionManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await manager.connect("job-1", ws1)
        await manager.connect("job-1", ws2)
        await manager.broadcast("job-1", {"event": "test"})
        ws1.send_json.assert_called_once_with({"event": "test"})
        ws2.send_json.assert_called_once_with({"event": "test"})

    async def test_broadcast_to_nonexistent_job(self):
        manager = ConnectionManager()
        # Should not raise an error
        await manager.broadcast("nonexistent-job", {"event": "test"})

    async def test_disconnect_nonexistent_connection(self):
        manager = ConnectionManager()
        ws = AsyncMock()
        # Should not raise an error
        await manager.disconnect("nonexistent-job", ws)


class TestProgressReporter:
    async def test_report(self):
        manager = AsyncMock(spec=ConnectionManager)
        reporter = ProgressReporter(manager, "job-1")
        await reporter.report("processing", 50.0, "Halfway done")
        manager.broadcast.assert_called_once()
        call_args = manager.broadcast.call_args
        assert call_args[0][0] == "job-1"
        data = call_args[0][1]
        assert data["event"] == "progress"
        assert data["progress_pct"] == 50.0
        assert data["stage"] == "processing"
        assert data["message"] == "Halfway done"

    async def test_report_with_part_info(self):
        manager = AsyncMock(spec=ConnectionManager)
        reporter = ProgressReporter(manager, "job-1")
        await reporter.report("encoding", 75.0, "Part 3/4", current_part=3, total_parts=4)
        data = manager.broadcast.call_args[0][1]
        assert data["current_part"] == 3
        assert data["total_parts"] == 4

    async def test_complete(self):
        manager = AsyncMock(spec=ConnectionManager)
        reporter = ProgressReporter(manager, "job-1")
        await reporter.complete()
        data = manager.broadcast.call_args[0][1]
        assert data["event"] == "complete"
        assert data["progress_pct"] == 100.0
        assert data["stage"] == "complete"
        assert data["job_id"] == "job-1"

    async def test_error(self):
        manager = AsyncMock(spec=ConnectionManager)
        reporter = ProgressReporter(manager, "job-1")
        await reporter.error("something failed")
        data = manager.broadcast.call_args[0][1]
        assert data["event"] == "error"
        assert data["message"] == "something failed"
        assert data["stage"] == "error"
        assert data["job_id"] == "job-1"

    async def test_report_rounds_progress(self):
        manager = AsyncMock(spec=ConnectionManager)
        reporter = ProgressReporter(manager, "job-1")
        await reporter.report("processing", 33.333333, "One third")
        data = manager.broadcast.call_args[0][1]
        assert data["progress_pct"] == 33.3


class TestMessageBuffer:
    def test_store_and_get_returns_messages(self):
        from src.api.ws import MessageBuffer

        buf = MessageBuffer()
        buf.store("job-1", {"event": "progress", "pct": 10})
        buf.store("job-1", {"event": "progress", "pct": 50})
        messages = buf.get_messages("job-1")
        assert len(messages) == 2
        assert messages[0]["pct"] == 10
        assert messages[1]["pct"] == 50

    def test_get_messages_unknown_job_returns_empty(self):
        from src.api.ws import MessageBuffer

        buf = MessageBuffer()
        assert buf.get_messages("nonexistent") == []

    def test_clear_removes_messages(self):
        from src.api.ws import MessageBuffer

        buf = MessageBuffer()
        buf.store("job-1", {"event": "progress"})
        assert len(buf.get_messages("job-1")) == 1
        buf.clear("job-1")
        assert buf.get_messages("job-1") == []

    def test_ttl_expiry_removes_old_messages(self):
        import time
        from unittest.mock import patch

        from src.api.ws import MessageBuffer

        buf = MessageBuffer(ttl_seconds=60.0)
        # Store a message at t=1000
        with patch.object(time, "monotonic", return_value=1000.0):
            buf.store("job-1", {"event": "progress"})
        # Retrieve before TTL expires -- message should still be there
        with patch.object(time, "monotonic", return_value=1050.0):
            assert len(buf.get_messages("job-1")) == 1
        # Retrieve after TTL expires -- message should be gone
        with patch.object(time, "monotonic", return_value=1061.0):
            assert buf.get_messages("job-1") == []


class TestConnectionManagerWithBuffer:
    async def test_broadcast_stores_in_buffer(self):
        from src.api.ws import MessageBuffer

        buf = MessageBuffer()
        manager = ConnectionManager(message_buffer=buf)
        ws = AsyncMock()
        await manager.connect("job-1", ws)
        await manager.broadcast("job-1", {"event": "progress", "pct": 25})
        messages = buf.get_messages("job-1")
        assert len(messages) == 1
        assert messages[0]["event"] == "progress"
        assert messages[0]["pct"] == 25

    async def test_late_connect_replays_buffered_messages(self):
        from src.api.ws import MessageBuffer

        buf = MessageBuffer()
        manager = ConnectionManager(message_buffer=buf)

        # First client connects and receives broadcasts
        ws1 = AsyncMock()
        await manager.connect("job-1", ws1)
        await manager.broadcast("job-1", {"event": "progress", "pct": 25})
        await manager.broadcast("job-1", {"event": "progress", "pct": 75})

        # Late client connects -- should receive replayed messages
        ws2 = AsyncMock()
        await manager.connect("job-1", ws2)
        replay_calls = ws2.send_json.call_args_list
        # First call is from accept, replayed messages follow
        assert len(replay_calls) == 2
        assert replay_calls[0][0][0] == {"event": "progress", "pct": 25}
        assert replay_calls[1][0][0] == {"event": "progress", "pct": 75}
