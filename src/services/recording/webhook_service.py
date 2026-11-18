"""Webhook notification service for job status changes.

Delivers JSON payloads to external webhook URLs with exponential-backoff retry.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from w2v_common.net import SSRFError, safe_post

from src.config import get_settings

logger = logging.getLogger(__name__)


class WebhookService:
    """POST JSON payloads to webhook URLs on job completion/failure."""

    def __init__(self) -> None:
        settings = get_settings()
        self._max_retries = settings.webhook.max_retries
        self._backoff_base = settings.webhook.backoff_base
        self._timeout = settings.webhook.timeout

    async def notify(
        self,
        webhook_url: str,
        job_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> bool:
        """Send webhook notification with retry.

        Args:
            webhook_url: Target URL to POST the payload to.
            job_id: Recording job identifier.
            status: Current job status (completed, failed, etc.).
            result: Optional result metadata dict.
            error: Optional error message string.

        Returns:
            True if delivery succeeded, False after all retries exhausted.
        """
        logger.info(
            "Webhook delivery starting: job=%s status=%s url=%s", job_id, status, webhook_url
        )
        t0 = time.monotonic()
        payload: dict[str, Any] = {
            "event": "recording.status_changed",
            "job_id": job_id,
            "status": status,
        }
        if result:
            payload["result"] = result
        if error:
            payload["error"] = error

        for attempt in range(self._max_retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    # ``webhook_url`` is caller-chosen, so this POST is both an
                    # SSRF sink and a data-exfiltration sink. The schema
                    # validator rejects an internal target at admission; this
                    # re-checks at delivery time, which is the one that counts
                    # -- admission and delivery resolve independently, and a
                    # queued job can be delivered long after it was accepted.
                    # ``max_redirects=0`` because a job payload must never be
                    # replayed to a redirect target the caller chose.
                    resp = await safe_post(client, webhook_url, json=payload, max_redirects=0)
                    if resp.status_code < 400:
                        logger.info(
                            "Webhook delivered: job=%s status=%s url=%s",
                            job_id,
                            status,
                            webhook_url,
                        )
                        return True
                    logger.warning(
                        "Webhook returned %d: job=%s url=%s (attempt %d/%d)",
                        resp.status_code,
                        job_id,
                        webhook_url,
                        attempt + 1,
                        self._max_retries,
                    )
            except SSRFError as e:
                # A refused target is a DETERMINISTIC verdict about where this
                # URL points -- retrying it three times with backoff only delays
                # the same answer while re-running the resolver. Give up now.
                logger.error(
                    "Webhook refused as an internal/reserved target: job=%s url=%s error=%s",
                    job_id,
                    webhook_url,
                    e,
                )
                return False
            except (httpx.HTTPError, OSError, TimeoutError, ValueError, RuntimeError) as e:
                logger.warning(
                    "Webhook failed: job=%s url=%s error=%s (attempt %d/%d)",
                    job_id,
                    webhook_url,
                    e,
                    attempt + 1,
                    self._max_retries,
                )

            if attempt < self._max_retries - 1:
                await asyncio.sleep(self._backoff_base**attempt)

        logger.error(
            "Webhook exhausted retries: job=%s url=%s elapsed=%.1fs",
            job_id,
            webhook_url,
            time.monotonic() - t0,
        )
        return False
