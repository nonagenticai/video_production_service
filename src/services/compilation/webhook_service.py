"""Webhook delivery service for notifying external systems of job events.

Delivers JSON payloads to caller-supplied URLs with optional HMAC-SHA256
signing, automatic retries with exponential backoff, and per-attempt timeouts.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime

import httpx
from w2v_common.net import SSRFError, safe_post

from src.config import get_settings

logger = logging.getLogger(__name__)


class WebhookService:
    """Delivers webhook payloads with retries and optional HMAC signing."""

    def __init__(self) -> None:
        cfg = get_settings().webhook
        self._max_retries: int = cfg.max_retries
        self._initial_backoff: float = cfg.backoff_base
        self._timeout: float = cfg.timeout
        self._enabled: bool = cfg.enabled

    async def deliver(
        self,
        url: str,
        secret: str | None,
        payload: dict,
    ) -> bool:
        """Deliver *payload* to *url* as a JSON POST.

        If *secret* is provided the raw JSON body is signed with
        HMAC-SHA256 and the hex digest is sent in the
        ``X-Webhook-Signature`` header.

        Returns ``True`` on a 2xx response, ``False`` after all retries
        are exhausted or when delivery is disabled.
        """
        if not self._enabled:
            logger.info("Webhook delivery disabled, skipping POST to %s", url)
            return False

        body: bytes = json.dumps(payload, separators=(",", ":")).encode()
        headers: dict[str, str] = {"Content-Type": "application/json"}

        if secret is not None:
            signature = hmac.new(
                secret.encode(),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Webhook-Signature"] = signature

        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            backoff = self._initial_backoff * (2**attempt)
            try:
                logger.info(
                    "Webhook attempt %d/%d  POST %s",
                    attempt + 1,
                    self._max_retries,
                    url,
                )
                async with httpx.AsyncClient(
                    timeout=self._timeout,
                ) as client:
                    # Sink-layer SSRF guard. ``url`` is tenant-supplied on the
                    # compile body and this POST carries the whole job payload,
                    # so a raw ``client.post`` is both an SSRF primitive and an
                    # exfiltration channel. ``src/utils/url_guard`` guards this
                    # field at the pydantic boundary, but that layer alone does
                    # not cover a DB row replayed after admission (or one
                    # poisoned since), which is why this module documents both
                    # layers as required.
                    #
                    # ``max_redirects=0``: ``body`` may carry an HMAC signature
                    # in ``X-Webhook-Signature``. Following a redirect would
                    # replay that signed payload to a target the tenant's
                    # webhook host chose, so a redirect is refused outright.
                    #
                    # No ``allow_hosts``: a webhook is by definition an external
                    # callback. The MinIO artifact allow-list exists for the
                    # service's own presigned storage URLs and has no business
                    # widening an outbound tenant callback.
                    response = await safe_post(
                        client,
                        url,
                        content=body,
                        headers=headers,
                        max_redirects=0,
                    )

                if 200 <= response.status_code < 300:
                    logger.info(
                        "Webhook delivered successfully to %s (status=%d, attempt=%d)",
                        url,
                        response.status_code,
                        attempt + 1,
                    )
                    return True

                logger.warning(
                    "Webhook delivery to %s returned status %d (attempt %d/%d, backoff=%.1fs)",
                    url,
                    response.status_code,
                    attempt + 1,
                    self._max_retries,
                    backoff,
                )
            except ValueError as exc:
                # PERMANENT rejections, never retried. ``SSRFError`` (a
                # ``ValueError`` subclass) means the target is blocked by
                # policy, not transiently unreachable — retrying re-attempts the
                # same exfiltration and burns the whole backoff ladder on every
                # job. A bare ``ValueError`` from ``safe_post`` is a malformed
                # URL, which is equally permanent; catching it here also keeps
                # it from escaping this coroutine as an unhandled background
                # task exception, which is the degrade-path contract PR #105
                # established for this module.
                logger.error(
                    "Webhook delivery to %s refused (%s), not retrying: %s",
                    url,
                    "SSRF guard" if isinstance(exc, SSRFError) else "invalid URL",
                    exc,
                )
                return False
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "Webhook delivery to %s failed with %s: %s (attempt %d/%d, backoff=%.1fs)",
                    url,
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    self._max_retries,
                    backoff,
                )

            # Sleep before retrying (skip sleep after the final attempt)
            if attempt < self._max_retries - 1:
                await asyncio.sleep(backoff)

        logger.error(
            "Webhook delivery to %s failed after %d attempts (last_error=%s)",
            url,
            self._max_retries,
            last_exc,
        )
        return False


async def fire_webhook(
    url: str,
    secret: str | None,
    event: str,
    job_id: str,
    data: dict,
) -> None:
    """Convenience helper that builds a standard payload and delivers it.

    The payload schema is::

        {
            "event": "<event>",
            "job_id": "<job_id>",
            "data": { ... },
            "timestamp": "<ISO-8601 UTC>"
        }
    """
    payload: dict = {
        "event": event,
        "job_id": job_id,
        "data": data,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    service = WebhookService()
    result = await service.deliver(url, secret, payload)
    if result:
        logger.info(
            "fire_webhook: event=%s job_id=%s delivered to %s",
            event,
            job_id,
            url,
        )
    else:
        logger.warning(
            "fire_webhook: event=%s job_id=%s delivery to %s failed",
            event,
            job_id,
            url,
        )
