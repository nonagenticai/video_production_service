"""Sink-layer SSRF/LFI proof for the compile path's outbound calls.

Why a SINK-layer file exists on top of the admission-layer one
-------------------------------------------------------------
``tests/unit/test_tenant_url_ssrf_schemas.py`` (PR #109) proves the *pydantic*
layer refuses an internal target, and ``src/utils/url_guard`` documents why that
layer alone is not enough: a job row is persisted and replayed later, so a row
poisoned or loosened after admission still reaches the sink. This file drives
the sinks themselves.

⚠️ TEST SAFETY — these tests must never become the exploit they describe.
Every sink is driven through an ``httpx.MockTransport`` that RECORDS requests
and answers ``200``, so:

* no real outbound connection is made, even with the guard removed; and
* the assertions are non-vacuous. Because the fake transport answers ``200``,
  UNGUARDED code SUCCEEDS (fetches the bytes / reports the webhook delivered)
  while guarded code refuses. ``assert recorded == []`` therefore fails the
  moment a guard is reverted, whereas a lone "returned None" / "is False"
  assertion would also be satisfied by unrelated breakage.

Targets are IP literals only, so ``validate_public_url``'s ``getaddrinfo``
resolves locally and no DNS query leaves the machine or depends on the runner's
resolver. (Name-based blocking is covered by the admission-layer file, which
stubs DNS.)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from tests.conftest import TENANT_U1

# --- Blocked targets (IP literals only — no DNS leaves the machine) ----------

METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
RFC1918_URL = "http://10.0.0.5/internal-only"
LOOPBACK_URL = "http://127.0.0.1:6308/api/compile/jobs"
# RFC 6598 CGNAT — this fleet's Tailscale tailnet. ``is_private`` is False for
# all of it, which is exactly why a hand-rolled guard waves it through.
CGNAT_URL = "http://100.64.0.5:5432/"  # worker4, the bare-metal prod Postgres

BLOCKED_TARGETS = [
    pytest.param(METADATA_URL, id="cloud-metadata-169.254.169.254"),
    pytest.param(RFC1918_URL, id="rfc1918-10.0.0.5"),
    pytest.param(LOOPBACK_URL, id="loopback-127.0.0.1"),
    pytest.param(CGNAT_URL, id="cgnat-tailnet-100.64.0.5"),
]

# A public IP literal used as the FIRST hop of the redirect tests.
PUBLIC_IP_URL = "http://93.184.216.34/asset.png"


class _Recorder:
    """An ``httpx`` transport that records every request and always answers 200.

    Answering 200 is the point: it makes unguarded code *succeed*, so a test
    that still passes after the guard is reverted is impossible.
    """

    def __init__(self, redirect_to: str | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._redirect_to = redirect_to

    @property
    def hosts(self) -> list[str | None]:
        return [r.url.host for r in self.requests]

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._redirect_to is not None and len(self.requests) == 1:
            return httpx.Response(302, headers={"Location": self._redirect_to})
        return httpx.Response(200, content=b"\x89PNG\r\n\x1a\n-pretend-bytes")

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


def _patch_client(target: str, rec: _Recorder, ctor_kwargs: list[dict] | None = None):
    """Force every client constructed at ``target`` onto the recorder transport."""
    real = httpx.Client if target.endswith("httpx.Client") else httpx.AsyncClient

    def _factory(*args, **kwargs):
        if ctor_kwargs is not None:
            ctor_kwargs.append(dict(kwargs))
        kwargs["transport"] = rec.transport()
        return real(*args, **kwargs)

    return patch(target, _factory)


_SEG_SYNC = "src.services.compilation._segment_builders.httpx.Client"
_SEG_ASYNC = "src.services.compilation._segment_builders.httpx.AsyncClient"
_WEBHOOK_ASYNC = "src.services.compilation.webhook_service.httpx.AsyncClient"
_WEBHOOK_SETTINGS = "src.services.compilation.webhook_service.get_settings"
_WEBHOOK_SLEEP = "src.services.compilation.webhook_service.asyncio.sleep"


def _ctx(tmp_path: Path) -> _JobContext:
    ctx = _JobContext(job_id="job-1", tenant_id=TENANT_U1, job=MagicMock(), settings=MagicMock())
    ctx.title_cards_dir = tmp_path / "title_cards"
    ctx.edited_dir = tmp_path / "edited"
    ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
    ctx.edited_dir.mkdir(parents=True, exist_ok=True)
    return ctx


class _Builder(_SegmentBuildersMixin):
    """The segment-builder mixin with only the collaborators the sinks touch."""

    def __init__(self) -> None:
        self._storage = MagicMock()
        self._storage.retrieve_uri_sync = MagicMock(return_value=None)
        self._storage.retrieve_uri = AsyncMock(return_value=None)


# --- Sink: shot image ref (_materialize_image_ref) ---------------------------


class TestImageRefFetchSink:
    """``ShotListPartInput.slide_image_url`` / ``image_url`` → sync GET."""

    @pytest.mark.parametrize("url", BLOCKED_TARGETS)
    def test_blocked_target_is_never_connected_to(self, url: str, tmp_path: Path) -> None:
        rec = _Recorder()
        with _patch_client(_SEG_SYNC, rec):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", url)

        assert rec.requests == [], f"outbound request was made to {url}: {rec.hosts}"
        assert result is None

    def test_redirect_into_the_tailnet_is_refused_at_the_hop(self, tmp_path: Path) -> None:
        """The 302-into-tailnet trap — the reason the sink layer is not redundant.

        A schema-approved public host can redirect straight into the estate. The
        guard follows redirects itself and re-validates every hop, so the second
        hop must never be dialled.
        """
        rec = _Recorder(redirect_to=CGNAT_URL)
        with _patch_client(_SEG_SYNC, rec):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert "100.64.0.5" not in rec.hosts, f"redirect into the tailnet followed: {rec.hosts}"
        assert len(rec.requests) == 1, f"redirect was followed: {rec.hosts}"
        assert result is None

    def test_client_is_not_built_with_auto_follow_redirects(self, tmp_path: Path) -> None:
        """Guards the mechanism, not one blocked example.

        Redirect-following must belong to the guard (manual, re-validated per
        hop), never to httpx — ``follow_redirects=True`` would walk into the
        estate before any per-hop check could run.
        """
        rec = _Recorder()
        ctor_kwargs: list[dict] = []
        with _patch_client(_SEG_SYNC, rec, ctor_kwargs):
            _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert ctor_kwargs, "no httpx.Client was constructed — the test would be vacuous"
        assert ctor_kwargs[0].get("follow_redirects") is not True

    def test_a_public_asset_is_still_fetched(self, tmp_path: Path) -> None:
        """Control: the guard must not break a legitimate remote asset."""
        rec = _Recorder()
        with _patch_client(_SEG_SYNC, rec):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert len(rec.requests) == 1
        assert result is not None
        assert result.read_bytes().startswith(b"\x89PNG")


# --- Sink: RecordingInput.video_path -----------------------------------------


class TestRecordingVideoPathSink:
    """``RecordingInput.video_path`` → async GET."""

    @pytest.mark.parametrize("url", BLOCKED_TARGETS)
    async def test_blocked_target_is_never_connected_to(self, url: str, tmp_path: Path) -> None:
        rec = _Recorder()
        with _patch_client(_SEG_ASYNC, rec):
            result = await _Builder()._resolve_recording_video_path(_ctx(tmp_path), "p1", url)

        assert rec.requests == [], f"outbound request was made to {url}: {rec.hosts}"
        assert result is None

    async def test_rejection_does_not_escape_as_an_unhandled_error(self, tmp_path: Path) -> None:
        """``SSRFError`` subclasses ``ValueError``; this resolver is fail-soft.

        A blocked part must be skipped, not abort the whole compilation.
        """
        rec = _Recorder()
        with _patch_client(_SEG_ASYNC, rec):
            assert (
                await _Builder()._resolve_recording_video_path(_ctx(tmp_path), "p1", CGNAT_URL)
                is None
            )
        assert rec.requests == []

    async def test_redirect_into_the_tailnet_is_refused_at_the_hop(self, tmp_path: Path) -> None:
        rec = _Recorder(redirect_to=CGNAT_URL)
        with _patch_client(_SEG_ASYNC, rec):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", PUBLIC_IP_URL
            )

        assert "100.64.0.5" not in rec.hosts, f"redirect into the tailnet followed: {rec.hosts}"
        assert len(rec.requests) == 1
        assert result is None


# --- Sink: CompileRequest.webhook_url ----------------------------------------


def _webhook_settings() -> MagicMock:
    return MagicMock(webhook=MagicMock(enabled=True, max_retries=3, backoff_base=0.0, timeout=1.0))


class TestWebhookPostSink:
    """``CompileRequest.webhook_url`` → outbound POST carrying the job payload.

    The worst sink of the set: it does not merely *read* an internal address, it
    *writes* the job payload to whatever host the tenant named. PR #109 guarded
    this field at the schema and left the sink a bare ``client.post``.
    """

    @pytest.mark.parametrize("url", BLOCKED_TARGETS)
    async def test_payload_is_never_posted_to_a_blocked_target(self, url: str) -> None:
        from src.services.compilation.webhook_service import WebhookService

        rec = _Recorder()
        payload = {"event": "completed", "job_id": "j1", "output": "s3://bucket/final.mp4"}

        with (
            patch(_WEBHOOK_SETTINGS, _webhook_settings),
            _patch_client(_WEBHOOK_ASYNC, rec),
            patch(_WEBHOOK_SLEEP, new_callable=AsyncMock),
        ):
            result = await WebhookService().deliver(url, "sig-secret", payload)

        assert rec.requests == [], f"job payload was POSTed to {url}: {rec.hosts}"
        assert result is False

    async def test_blocked_target_is_not_retried(self) -> None:
        """A guard rejection is permanent; retrying re-attempts the exfiltration."""
        from src.services.compilation.webhook_service import WebhookService

        rec = _Recorder()
        sleep_mock = AsyncMock()
        with (
            patch(_WEBHOOK_SETTINGS, _webhook_settings),
            _patch_client(_WEBHOOK_ASYNC, rec),
            patch(_WEBHOOK_SLEEP, sleep_mock),
        ):
            await WebhookService().deliver(CGNAT_URL, None, {"e": "x"})

        # Both assertions are required. "No sleep" ALONE is vacuous: with the
        # guard reverted the fake transport answers 200 on the first attempt, so
        # nothing sleeps either — for entirely the wrong reason. Pinning
        # ``requests == []`` is what makes this fail when the guard is removed.
        assert rec.requests == [], f"job payload was POSTed to {CGNAT_URL}: {rec.hosts}"
        assert sleep_mock.await_count == 0, "an SSRF-blocked webhook was retried"

    async def test_signed_payload_is_not_replayed_to_a_redirect_target(self) -> None:
        """``max_redirects=0``: a signed body must never follow a 302.

        The body carries an HMAC in ``X-Webhook-Signature``. A redirect-following
        POST hands that signed payload — signature intact — to a host the
        tenant's webhook endpoint chose, which turns a legitimate-looking
        callback into an arbitrary-destination exfiltration primitive. It also
        re-opens the SSRF the schema closed, since the redirect target is not on
        the request body at all.
        """
        from src.services.compilation.webhook_service import WebhookService

        rec = _Recorder(redirect_to=CGNAT_URL)
        with (
            patch(_WEBHOOK_SETTINGS, _webhook_settings),
            _patch_client(_WEBHOOK_ASYNC, rec),
            patch(_WEBHOOK_SLEEP, new_callable=AsyncMock),
        ):
            result = await WebhookService().deliver(PUBLIC_IP_URL, "sig-secret", {"e": "x"})

        assert len(rec.requests) == 1, f"a redirect was followed: {rec.hosts}"
        assert "100.64.0.5" not in rec.hosts
        assert result is False

    async def test_a_public_target_still_delivers_with_its_signature(self) -> None:
        """Control: the guard must not break a legitimate external webhook."""
        from src.services.compilation.webhook_service import WebhookService

        rec = _Recorder()
        with (
            patch(_WEBHOOK_SETTINGS, _webhook_settings),
            _patch_client(_WEBHOOK_ASYNC, rec),
            patch(_WEBHOOK_SLEEP, new_callable=AsyncMock),
        ):
            result = await WebhookService().deliver(
                "http://93.184.216.34/hook", "secret", {"event": "done"}
            )

        assert result is True
        assert len(rec.requests) == 1
        assert rec.requests[0].method == "POST"
        assert json.loads(rec.requests[0].content)["event"] == "done"
        assert "X-Webhook-Signature" in rec.requests[0].headers
