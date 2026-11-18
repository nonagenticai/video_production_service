"""Unit tests for webhook delivery service."""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

_PATCH_SETTINGS = "src.services.compilation.webhook_service.get_settings"
_PATCH_CLIENT = "src.services.compilation.webhook_service.httpx.AsyncClient"
_PATCH_SLEEP = "src.services.compilation.webhook_service.asyncio.sleep"
# Delivery goes through the shared SSRF-safe sender, not a bare
# ``client.post``, so the transport double moved here. ``_PATCH_CLIENT``
# stays patched alongside it so no test can build a real client.
_PATCH_SAFE_POST = "src.services.compilation.webhook_service.safe_post"


def _make_settings(*, enabled=True, max_retries=3, initial_backoff=1.0, timeout=5.0):
    return MagicMock(
        webhook=MagicMock(
            enabled=enabled,
            max_retries=max_retries,
            backoff_base=initial_backoff,
            timeout=timeout,
        )
    )


def _mock_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


class TestWebhookDelivery:
    @patch(_PATCH_SETTINGS)
    async def test_disabled_returns_false_immediately(self, mock_settings):
        mock_settings.return_value = _make_settings(enabled=False)

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.deliver("https://example.com/hook", "secret", {"key": "val"})
        assert result is False

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_success_200_returns_true(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings()

        mock_client = AsyncMock()
        mock_safe_post.return_value = _mock_response(200)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.deliver("https://example.com/hook", None, {"event": "done"})
        assert result is True
        mock_safe_post.assert_called_once()

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_hmac_signature_correctness(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings()

        mock_client = AsyncMock()
        mock_safe_post.return_value = _mock_response(200)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        secret = "my-secret-key"  # pragma: allowlist secret
        payload = {"event": "completed", "job_id": "abc"}

        svc = WebhookService()
        await svc.deliver("https://example.com/hook", secret, payload)

        # Verify the call was made with the correct HMAC signature
        call_kwargs = mock_safe_post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        content = call_kwargs.kwargs.get("content") or call_kwargs[1].get("content")

        expected_body = json.dumps(payload, separators=(",", ":")).encode()
        expected_sig = hmac.new(secret.encode(), expected_body, hashlib.sha256).hexdigest()

        assert headers["X-Webhook-Signature"] == expected_sig
        assert content == expected_body

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_retry_on_non_2xx_then_success(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings(max_retries=3)

        mock_client = AsyncMock()
        # First call returns 500, second returns 200
        mock_safe_post.side_effect = [
            _mock_response(500),
            _mock_response(200),
        ]
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.deliver("https://example.com/hook", None, {"data": "test"})
        assert result is True
        assert mock_safe_post.call_count == 2

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_retry_on_http_error_then_success(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings(max_retries=3)

        mock_client = AsyncMock()
        mock_safe_post.side_effect = [
            httpx.ConnectError("connection refused"),
            _mock_response(200),
        ]
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.deliver("https://example.com/hook", None, {"data": "test"})
        assert result is True
        assert mock_safe_post.call_count == 2

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_max_retries_exhausted_returns_false(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings(max_retries=3)

        mock_client = AsyncMock()
        mock_safe_post.return_value = _mock_response(500)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        result = await svc.deliver("https://example.com/hook", None, {"data": "test"})
        assert result is False
        assert mock_safe_post.call_count == 3

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_exponential_backoff_sleep_values(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings(max_retries=4, initial_backoff=2.0)

        mock_client = AsyncMock()
        mock_safe_post.return_value = _mock_response(500)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        from src.services.compilation.webhook_service import WebhookService

        svc = WebhookService()
        await svc.deliver("https://example.com/hook", None, {"data": "test"})

        # Backoff formula: initial_backoff * (2 ** attempt)
        # attempt 0: 2.0 * 1 = 2.0 (sleep)
        # attempt 1: 2.0 * 2 = 4.0 (sleep)
        # attempt 2: 2.0 * 4 = 8.0 (sleep)
        # attempt 3: final attempt, sleep skipped
        assert mock_sleep.call_count == 3
        mock_sleep.assert_any_await(2.0)
        mock_sleep.assert_any_await(4.0)
        mock_sleep.assert_any_await(8.0)


class TestFireWebhook:
    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_payload_structure(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings()

        captured_payload = {}

        mock_client = AsyncMock()
        mock_safe_post.return_value = _mock_response(200)
        mock_client_cls.return_value.__aenter__.return_value = mock_client

        async def _capture_deliver(self, url, secret, payload):
            captured_payload.update(payload)
            return True

        from src.services.compilation.webhook_service import WebhookService, fire_webhook

        with patch.object(WebhookService, "deliver", _capture_deliver):
            await fire_webhook(
                url="https://example.com/hook",
                secret="secret",  # pragma: allowlist secret
                event="job.completed",
                job_id="job-123",
                data={"output": "file.mp4"},
            )

        assert captured_payload["event"] == "job.completed"
        assert captured_payload["job_id"] == "job-123"
        assert captured_payload["data"] == {"output": "file.mp4"}
        assert "timestamp" in captured_payload

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_calls_webhook_service_deliver(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings()

        mock_deliver = AsyncMock(return_value=True)

        from src.services.compilation.webhook_service import WebhookService, fire_webhook

        with patch.object(WebhookService, "deliver", mock_deliver):
            await fire_webhook(
                url="https://example.com/hook",
                secret="secret",  # pragma: allowlist secret
                event="job.started",
                job_id="job-456",
                data={},
            )

        mock_deliver.assert_awaited_once()
        call_args = mock_deliver.call_args
        assert call_args[0][0] == "https://example.com/hook"
        assert call_args[0][1] == "secret"

    @patch(_PATCH_SLEEP, new_callable=AsyncMock)
    @patch(_PATCH_SAFE_POST, new_callable=AsyncMock)
    @patch(_PATCH_CLIENT)
    @patch(_PATCH_SETTINGS)
    async def test_timestamp_is_utc_iso_format(
        self, mock_settings, mock_client_cls, mock_safe_post, mock_sleep
    ):
        mock_settings.return_value = _make_settings()

        captured_payload = {}

        async def _capture_deliver(self, url, secret, payload):
            captured_payload.update(payload)
            return True

        from src.services.compilation.webhook_service import WebhookService, fire_webhook

        with patch.object(WebhookService, "deliver", _capture_deliver):
            await fire_webhook(
                url="https://example.com/hook",
                secret=None,
                event="test",
                job_id="j1",
                data={},
            )

        ts = captured_payload["timestamp"]
        # UTC ISO format contains a timezone indicator ('+00:00' or 'Z')
        assert "+00:00" in ts or "Z" in ts
        # Should be parseable as ISO format
        from datetime import datetime

        parsed = datetime.fromisoformat(ts)
        assert parsed is not None
