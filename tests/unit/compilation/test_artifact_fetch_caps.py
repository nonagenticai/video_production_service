"""Per-site response-body ceilings for the allow-listed artifact fetches.

Why this file exists
--------------------
w2v_common 0.55.0 gave ``safe_get``/``safe_get_sync`` a 64 MiB **default**
``max_fetch_bytes`` where ``None`` previously meant an unbounded read. Inheriting
that default here is a DATA-LOSS bug, not a hardening win, because every artifact
fetch in this service is deliberately fail-soft:

    except (httpx.HTTPError, OSError, ValueError) as exc:
        logger.warning(...); return None

``ResponseTooLargeError`` subclasses ``ValueError``. So an artifact over the cap
does not raise and does not fail the job — it silently becomes a missing
segment, a missing background, or a video with no narration. The only trace is a
warning line. A one-minute 1080p screencast already passes 64 MiB, so the
inherited default would have skipped essentially every real recording.

The four sites therefore state their own ceiling (``src/utils/url_guard``,
``ARTIFACT_*_MAX_BYTES``). This file proves that at the sites themselves.

What the tests actually exercise
--------------------------------
The REAL ``w2v_common`` guard runs. Nothing here fakes ``safe_get`` — the
existing ``_patch_client`` harness swaps only the httpx *transport*, so the
production call path, the real cap arithmetic and the real
``ResponseTooLargeError`` are all live. (A fake ``safe_get`` narrower than the
real signature could not fail if a site stopped passing the kwarg at all, which
is precisely the defect being guarded.)

Two response shapes are used, matching the guard's two enforcement points:

* a **declared** ``Content-Length`` over the cap — rejected before any body is
  read, which keeps the oversize cases cheap (5 real bytes, not 2 GiB); and
* a **real** oversized body, which drives the streaming accumulator. One test
  uses real bytes at 64 MiB + 1 so the accumulator path is genuinely covered
  and not merely assumed.

The ACCEPT cases are the load-bearing half. A test that only proves "oversize is
refused" would still pass if someone capped the recording site at 1 MiB and
broke every compile in production.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from w2v_common.net.safe_fetch import DEFAULT_MAX_FETCH_BYTES, ResponseTooLargeError

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from src.utils.url_guard import (
    ARTIFACT_AUDIO_MAX_BYTES,
    ARTIFACT_IMAGE_MAX_BYTES,
    ARTIFACT_RECORDING_MAX_BYTES,
)
from tests.conftest import TENANT_U1

MIB = 1024 * 1024

# A public IP literal: ``validate_public_url`` resolves it locally, so no DNS
# query leaves the machine and no real connection is ever made (MockTransport).
PUBLIC_IP_URL = "http://93.184.216.34/artifact.bin"

_SEG_SYNC = "src.services.compilation._segment_builders.httpx.Client"
_SEG_ASYNC = "src.services.compilation._segment_builders.httpx.AsyncClient"


class _SizedResponder:
    """A transport answering 200 with a chosen real body and declared length.

    ``declared_length`` forges ``Content-Length`` independently of the body so an
    over-cap case costs a handful of bytes instead of gigabytes. It exercises the
    guard's header pre-check; ``body`` exercises its streaming accumulator.
    """

    def __init__(self, body: bytes, declared_length: int | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._body = body
        self._declared = declared_length

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        headers = {} if self._declared is None else {"Content-Length": str(self._declared)}
        return httpx.Response(200, content=self._body, headers=headers)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)


def _patch_client(target: str, rec: _SizedResponder):
    real = httpx.Client if target.endswith("httpx.Client") else httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = rec.transport
        return real(*args, **kwargs)

    return patch(target, _factory)


def _ctx(tmp_path: Path) -> _JobContext:
    ctx = _JobContext(job_id="job-1", tenant_id=TENANT_U1, job=MagicMock(), settings=MagicMock())
    ctx.title_cards_dir = tmp_path / "title_cards"
    ctx.edited_dir = tmp_path / "edited"
    ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
    ctx.edited_dir.mkdir(parents=True, exist_ok=True)
    return ctx


class _Builder(_SegmentBuildersMixin):
    def __init__(self) -> None:
        self._storage = MagicMock()
        self._storage.retrieve_uri_sync = MagicMock(return_value=None)
        self._storage.retrieve_uri = AsyncMock(return_value=None)


# --- Non-vacuity controls ----------------------------------------------------


class TestTheDefaultIsTheThreatBeingGuarded:
    """Pins WHY the explicit ceilings exist. If these fail the premise moved."""

    def test_installed_w2v_common_defaults_to_64_mib(self) -> None:
        """The inherited default. Every accept test below is meaningless if this
        is unbounded — a site that dropped its kwarg would still pass."""
        assert DEFAULT_MAX_FETCH_BYTES == 64 * MIB

    def test_response_too_large_is_swallowed_by_the_fail_soft_handlers(self) -> None:
        """The mechanism that makes an over-cap artifact SILENT rather than loud.

        Every artifact sink catches ``ValueError``. This is the single fact that
        turns a size cap into data loss instead of an error.
        """
        assert issubclass(ResponseTooLargeError, ValueError)

    def test_each_site_ceiling_clears_the_inherited_default(self) -> None:
        """Sizing sanity: the two sites that genuinely outgrow 64 MiB say so, and
        the image sites are pinned rather than left to drift with upstream."""
        assert ARTIFACT_RECORDING_MAX_BYTES > DEFAULT_MAX_FETCH_BYTES
        assert ARTIFACT_AUDIO_MAX_BYTES > DEFAULT_MAX_FETCH_BYTES
        assert ARTIFACT_IMAGE_MAX_BYTES == DEFAULT_MAX_FETCH_BYTES


# --- Site 3: the source recording (the unbounded one) ------------------------


class TestRecordingVideoPathCap:
    """``_resolve_recording_video_path`` — the trap's principal victim."""

    async def test_a_recording_over_the_inherited_default_still_downloads(
        self, tmp_path: Path
    ) -> None:
        """THE accept case. 200 MiB is over the 64 MiB default and under the
        site's 1 GiB ceiling, i.e. an ordinary screen recording.

        Reverting ``max_fetch_bytes=ARTIFACT_RECORDING_MAX_BYTES`` makes this
        return None — a silently skipped part, exactly the production breakage.
        """
        rec = _SizedResponder(b"\x00\x00\x00\x18ftypmp42", declared_length=200 * MIB)
        with _patch_client(_SEG_ASYNC, rec):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", PUBLIC_IP_URL
            )

        assert len(rec.requests) == 1, "no fetch was attempted — the test would be vacuous"
        assert result is not None, (
            "a 200 MiB recording was dropped: the site is inheriting the 64 MiB default"
        )
        assert result.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")

    async def test_a_real_body_over_the_default_streams_through(self, tmp_path: Path) -> None:
        """Same accept case with REAL bytes, so the streaming accumulator — not
        just the Content-Length pre-check — is proven at this site."""
        body = b"\x00\x00\x00\x18ftypmp42" + b"\xab" * (64 * MIB + 1)
        rec = _SizedResponder(body)
        with _patch_client(_SEG_ASYNC, rec):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", PUBLIC_IP_URL
            )

        assert len(rec.requests) == 1
        assert result is not None, (
            "a 64 MiB + 1 body was dropped by the accumulator: default cap in force"
        )
        assert result.stat().st_size == len(body)

    async def test_the_site_is_still_bounded(self, tmp_path: Path) -> None:
        """The reject half: 1 GiB is a ceiling, not an opt-out. A body past it is
        refused, and — per this resolver's fail-soft contract — refused as a
        skipped part rather than an escaping error."""
        rec = _SizedResponder(b"x", declared_length=ARTIFACT_RECORDING_MAX_BYTES + 1)
        with _patch_client(_SEG_ASYNC, rec):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", PUBLIC_IP_URL
            )

        assert result is None


# --- Sites 1 & 2: title-card / visual-asset images ---------------------------


class TestImageRefCap:
    """``_materialize_image_ref`` — a still handed to Pillow."""

    def test_an_ordinary_image_still_downloads(self, tmp_path: Path) -> None:
        """Control: the ceiling must not break a legitimate asset."""
        rec = _SizedResponder(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4096)
        with _patch_client(_SEG_SYNC, rec):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert len(rec.requests) == 1
        assert result is not None
        assert result.read_bytes().startswith(b"\x89PNG")

    def test_an_image_past_the_ceiling_is_refused(self, tmp_path: Path) -> None:
        rec = _SizedResponder(b"\x89PNG", declared_length=ARTIFACT_IMAGE_MAX_BYTES + 1)
        with _patch_client(_SEG_SYNC, rec):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert result is None

    def test_the_image_site_passes_its_own_ceiling_not_the_inherited_one(
        self, tmp_path: Path
    ) -> None:
        """The image ceiling currently equals the upstream default, so a
        behavioural test alone cannot tell "stated" from "inherited". This
        asserts the kwarg is actually handed over, which is what keeps the value
        pinned at the site if upstream ever retunes its default."""
        captured: dict = {}
        import src.services.compilation._segment_builders as sb

        real = sb.safe_get_sync

        def _spy(client, url, **kwargs):
            captured.update(kwargs)
            return real(client, url, **kwargs)

        rec = _SizedResponder(b"\x89PNG")
        with _patch_client(_SEG_SYNC, rec), patch.object(sb, "safe_get_sync", _spy):
            _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", PUBLIC_IP_URL)

        assert captured.get("max_fetch_bytes") == ARTIFACT_IMAGE_MAX_BYTES


class TestBackgroundImageCap:
    """The title-card ``background_image`` download."""

    def test_background_image_passes_its_own_ceiling(self, tmp_path: Path) -> None:
        captured: dict = {}
        import src.services.compilation._segment_builders as sb

        real = sb.safe_get_sync

        def _spy(client, url, **kwargs):
            captured.update(kwargs)
            return real(client, url, **kwargs)

        ctx = _ctx(tmp_path)
        # Card rendering itself is out of scope here — only the fetch is.
        ctx.title_gen = MagicMock()
        rec = _SizedResponder(b"\x89PNG")
        part = {"background_image": PUBLIC_IP_URL}
        with (
            _patch_client(_SEG_SYNC, rec),
            patch.object(sb, "safe_get_sync", _spy),
            patch.object(_Builder, "_segment_with_optional_narration", MagicMock()),
        ):
            _Builder()._process_title_card(ctx, "p1", part)

        assert len(rec.requests) == 1, "no fetch attempted — the test would be vacuous"
        assert captured.get("max_fetch_bytes") == ARTIFACT_IMAGE_MAX_BYTES


# --- Site 4: the narration track ---------------------------------------------


class TestNarrationAudioCap:
    """``compiler_service._initialize`` — the ``narration_audio_dir`` URL branch.

    Losing this one is the quietest failure of the four: the handler sets
    ``narration_audio_dir = None`` and the compilation proceeds to produce a
    SILENT video, with the job still reported successful.
    """

    async def test_a_narration_over_the_inherited_default_still_downloads(
        self, tmp_path: Path
    ) -> None:
        """Accept case: 128 MiB of mp3 — past the 64 MiB default (~70 min at
        128 kbps) and inside the site's 256 MiB ceiling."""
        import src.services.compilation.compiler_service as cs

        rec = _SizedResponder(b"ID3\x03\x00\x00\x00", declared_length=128 * MIB)
        ctx = _JobContext(
            job_id="job-1",
            tenant_id=TENANT_U1,
            job=MagicMock(narration_audio_dir=PUBLIC_IP_URL),
            settings=MagicMock(),
        )
        ctx.work_dir = tmp_path / "work"
        ctx.work_dir.mkdir(parents=True, exist_ok=True)

        with _patch_client("src.services.compilation.compiler_service.httpx.AsyncClient", rec):
            await cs._download_narration_audio(ctx, PUBLIC_IP_URL)

        assert len(rec.requests) == 1, "no fetch attempted — vacuous"
        assert ctx.narration_audio_dir is not None, (
            "a 128 MiB narration was dropped: the site is inheriting the 64 MiB "
            "default and the compile would render SILENT"
        )

    async def test_the_narration_site_is_still_bounded(self, tmp_path: Path) -> None:
        import src.services.compilation.compiler_service as cs

        rec = _SizedResponder(b"ID3", declared_length=ARTIFACT_AUDIO_MAX_BYTES + 1)
        ctx = _JobContext(
            job_id="job-1", tenant_id=TENANT_U1, job=MagicMock(), settings=MagicMock()
        )
        ctx.work_dir = tmp_path / "work"
        ctx.work_dir.mkdir(parents=True, exist_ok=True)

        with _patch_client("src.services.compilation.compiler_service.httpx.AsyncClient", rec):
            await cs._download_narration_audio(ctx, PUBLIC_IP_URL)

        assert ctx.narration_audio_dir is None
