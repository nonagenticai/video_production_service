"""The EFFECTIVE capture raster must reach the caller, not die inside the recorder.

``AnimationRecorder`` supersamples the Chromium capture (``_CAPTURE_SCALE = 2``)
and lanczos-downscales the frames before the encode, so the reported
``width``/``height`` are the ORDERED output size and disclose nothing about the
surface the glyphs were rasterised on. Worse, ``_resolve_capture_scale`` steps
that factor DOWN silently whenever the surface would exceed
``_MAX_CAPTURE_RASTER_PIXELS`` -- a 4k order captures at 1x -- and no consumer
could tell.

A caller that wants its source assets to survive the capture at full detail (the
orchestrator's Ken Burns stage upscales stills before recording them) had to
either guess or hardcode this service's private constant, which buys a silent
quality regression the day the constant moves.

These tests pin the whole chain, because a declared-but-never-populated response
field is the same defect wearing a nicer hat:

    _resolve_capture_scale -> record_animation result dict
                           -> RecorderService._finalize_result
                           -> mark_completed kwargs (persisted columns)
                           -> GET /jobs/{id}/result  and  GET /jobs/{id}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.services.ffmpeg.base import ProbeResult
from src.services.recording.animation_recorder import (
    _CAPTURE_SCALE,
    _MAX_CAPTURE_RASTER_PIXELS,
    AnimationRecorder,
    _resolve_capture_scale,
)
from tests.conftest import TENANT_U1, make_job
from tests.unit.recording.test_recorder_service import (
    _completed_kwargs,
    _FakeRecorder,
    _make_jm,
    _make_service,
    _success_result,
)
from tests.unit.recording.test_routes import _make_client, _make_mock_jm

# ── The producer ───────────────────────────────────────────────────


async def _record(recorder: AnimationRecorder, tmp_path: Path, **rec_config: Any) -> dict[str, Any]:
    """Drive ``record_animation`` with the browser and ffmpeg halves stubbed out.

    Everything the assertions care about -- config resolution, the raster
    decision, the result dict -- is real; only Chromium and the encode are not.
    """
    out = tmp_path / "out.mp4"
    out.write_bytes(b"encoded")

    frames = tmp_path / "frames"
    frames.mkdir(exist_ok=True)

    with (
        patch.object(
            AnimationRecorder,
            "_capture_frames",
            new=AsyncMock(return_value=(frames, 60, 2000.0, 0.6)),
        ),
        patch.object(AnimationRecorder, "_assemble", return_value=out),
        patch(
            "src.services.recording.animation_recorder.FFmpegOps.probe_all",
            return_value=ProbeResult(
                # 0 is what the recorder treats as 'probe told me nothing',
                # so the width/height in the result fall back to the resolved config.
                duration=2.0,
                width=0,
                height=0,
                has_audio=False,
                probed=True,
            ),
        ),
    ):
        return await recorder.record_animation(
            job_id="anim-1",
            html_content="<html></html>",
            animation_config={"fps": 30},
            recording_config=rec_config or None,
        )


class TestTheRecorderReportsTheRasterItCaptured:
    async def test_a_supersampled_capture_reports_the_supersampled_raster(
        self, tmp_path: Path
    ) -> None:
        """The whole point: the caller learns the raster, not the output size."""
        result = await _record(
            AnimationRecorder(), tmp_path, viewport_width=1280, viewport_height=720
        )

        assert result["width"] == 1280
        assert result["height"] == 720
        assert result["capture_width"] == 1280 * _CAPTURE_SCALE
        assert result["capture_height"] == 720 * _CAPTURE_SCALE
        # And it is bigger than the output -- the field is not an alias.
        assert result["capture_width"] > result["width"]

    async def test_a_clamped_down_capture_reports_the_clamp_not_the_constant(
        self, tmp_path: Path
    ) -> None:
        """The silent step-down becomes visible as ``capture_width == width``.

        This is the case that made hardcoding ``_CAPTURE_SCALE`` caller-side a
        latent regression: a 4k order never gets 2x, so a caller that assumed it
        would upscale its stills to twice the raster actually used.
        """
        assert 3840 * 2160 * _CAPTURE_SCALE**2 > _MAX_CAPTURE_RASTER_PIXELS
        assert _resolve_capture_scale(3840, 2160) == 1

        result = await _record(
            AnimationRecorder(), tmp_path, viewport_width=3840, viewport_height=2160
        )

        assert result["capture_width"] == 3840, (
            "a 4k order captures at 1x -- reporting anything larger here is "
            "exactly the lie the caller would have told itself"
        )
        assert result["capture_width"] == result["width"]


# ── The persistence hand-off ───────────────────────────────────────


@pytest.fixture
def _successful_reprobe():
    """``_handle_success`` re-probes the stored artifact before completing it.

    The shared success fixtures write throwaway bytes, not a real container, so
    without this the completion guard trips for a reason unrelated to these
    tests (same rationale as ``test_recorder_service._default_successful_probe``).
    """
    with patch(
        "src.services.recording.recorder_service.FFmpegOps.probe_all",
        return_value=ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=True),
    ):
        yield


@pytest.mark.usefixtures("_successful_reprobe")
class TestTheServicePersistsTheReportedRaster:
    async def test_the_recorders_raster_reaches_mark_completed(self) -> None:
        """``_finalize_result`` must forward it, or the value dies at completion."""
        result = _success_result()
        result["capture_width"] = 2560
        result["capture_height"] = 1440

        service = _make_service(recorder=_FakeRecorder(results=[result]))
        jm, fake = _make_jm(make_job(job_id="cap-1"))

        await service.process_job(jm, "cap-1", TENANT_U1)

        kw = _completed_kwargs(fake)
        assert kw["capture_width"] == 2560
        assert kw["capture_height"] == 1440

    async def test_a_recorder_that_reports_nothing_persists_null(self) -> None:
        """Pre-feature behaviour, unchanged: not reported is not invented."""
        service = _make_service(recorder=_FakeRecorder(results=[_success_result()]))
        jm, fake = _make_jm(make_job(job_id="cap-2"))

        await service.process_job(jm, "cap-2", TENANT_U1)

        kw = _completed_kwargs(fake)
        assert kw["capture_width"] is None
        assert kw["capture_height"] is None


# ── The wire ───────────────────────────────────────────────────────


class TestTheRoutesSerialiseTheRaster:
    def test_get_result_returns_the_capture_raster(self) -> None:
        job = make_job(
            job_id="j1",
            status="completed",
            video_path="/v.mp4",
            duration_seconds=10.0,
            file_size_bytes=5000,
            width=1280,
            height=720,
            capture_width=2560,
            capture_height=1440,
            has_audio=False,
        )
        client, _ = _make_client(mock_jm=_make_mock_jm(get_job=AsyncMock(return_value=job)))

        body = client.get("/api/record/jobs/j1/result").json()

        assert body["width"] == 1280
        assert body["capture_width"] == 2560, (
            "the field is declared on RecordingResultResponse but the route is "
            "not populating it -- declared-but-never-populated is the defect"
        )
        assert body["capture_height"] == 1440

    def test_get_status_returns_the_capture_raster(self) -> None:
        job = make_job(
            job_id="j1",
            status="completed",
            width=1280,
            height=720,
            capture_width=2560,
            capture_height=1440,
        )
        client, _ = _make_client(mock_jm=_make_mock_jm(get_job=AsyncMock(return_value=job)))

        body = client.get("/api/record/jobs/j1").json()

        assert body["capture_width"] == 2560
        assert body["capture_height"] == 1440

    def test_an_older_row_reports_null_not_a_guess(self) -> None:
        """NULL means "not reported"; the route must not synthesise width."""
        job = make_job(job_id="j1", status="completed", video_path="/v.mp4", width=1280, height=720)
        client, _ = _make_client(mock_jm=_make_mock_jm(get_job=AsyncMock(return_value=job)))

        body = client.get("/api/record/jobs/j1/result").json()

        assert body["capture_width"] is None
        assert body["capture_height"] is None


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1280, 720, _CAPTURE_SCALE),
        (2560, 1440, _CAPTURE_SCALE),
        (3840, 2160, 1),
    ],
)
def test_the_raster_budget_decides_the_factor(width: int, height: int, expected: int) -> None:
    """Pins the arithmetic the reported raster is derived from."""
    assert _resolve_capture_scale(width, height) == expected
    assert width * expected * height * expected <= _MAX_CAPTURE_RASTER_PIXELS
