"""Tests for src/services/webhook_service.py -- WebhookService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from src.services.recording.webhook_service import WebhookService

# ── Helpers ────────────────────────────────────────────────────────


def _make_service():
    return WebhookService()


# ── notify ────────────────────────────────────────────────────────


class TestNotify:
    @patch("src.services.recording.webhook_service.safe_post")
    async def test_success_on_first_try(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_post.return_value = mock_resp

        svc = _make_service()
        result = await svc.notify("http://hook.test/cb", "j1", "completed")
        assert result is True
        mock_post.assert_awaited_once()

    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("src.services.recording.webhook_service.safe_post")
    async def test_retries_on_failure_then_succeeds(self, mock_post, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        mock_post.side_effect = [fail_resp, ok_resp]

        svc = _make_service()
        result = await svc.notify("http://hook.test/cb", "j1", "completed")
        assert result is True
        assert mock_post.call_count == 2

    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("src.services.recording.webhook_service.safe_post")
    async def test_exhausts_retries(self, mock_post, mock_sleep):
        fail_resp = MagicMock()
        fail_resp.status_code = 500

        mock_post.return_value = fail_resp

        svc = _make_service()
        result = await svc.notify("http://hook.test/cb", "j1", "failed")
        assert result is False
        assert mock_post.call_count == svc._max_retries

    @patch("src.services.recording.webhook_service.safe_post")
    async def test_correct_payload_structure(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_post.return_value = mock_resp

        svc = _make_service()
        await svc.notify(
            "http://hook.test/cb",
            "j1",
            "completed",
            result={"duration_seconds": 10.0},
            error=None,
        )

        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["event"] == "recording.status_changed"
        assert payload["job_id"] == "j1"
        assert payload["status"] == "completed"
        assert payload["result"]["duration_seconds"] == 10.0

    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("src.services.recording.webhook_service.safe_post")
    async def test_handles_httpx_exception(self, mock_post, mock_sleep):
        mock_post.side_effect = OSError("Connection refused")

        svc = _make_service()
        result = await svc.notify("http://hook.test/cb", "j1", "failed", error="boom")
        assert result is False

    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("src.services.recording.webhook_service.safe_post")
    async def test_connecterror_retries_then_returns_false(self, mock_post, mock_sleep):
        """httpx.ConnectError (not an OSError subclass) must be caught: the loop
        retries/backs off and returns False after exhaustion -- it must NOT
        escape (callers fire-and-forget via asyncio.create_task)."""
        mock_post.side_effect = httpx.ConnectError("Connection refused")

        svc = _make_service()
        result = await svc.notify("http://hook.test/cb", "j1", "failed", error="boom")
        assert result is False
        assert mock_post.call_count == svc._max_retries
