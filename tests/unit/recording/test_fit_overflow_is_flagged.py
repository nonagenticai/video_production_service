"""Text a template cannot fit is flagged, never silently clamped (render 01a1179d).

The orchestrator had cut a card's second line to a character count ("…an audit
trail you can") because nothing told it what the template could show.
``MotionLib.fit`` now reports text still over its line budget at the smallest
type in ``window.FIT_OVERFLOW``; the recorder logs it and returns it as
``fit_overflow`` in the recording result.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.services.recording.animation_recorder import AnimationRecorder, FitOverflow, RenderFlags
from src.services.recording.templates import get_template


class _Page:
    def __init__(self, value: Any) -> None:
        self.value = value

    async def evaluate(self, expression: str) -> Any:
        assert expression == "window.FIT_OVERFLOW || null"
        return self.value


async def test_an_overflow_is_logged_and_carried_in_the_flags(
    caplog: pytest.LogCaptureFixture,
) -> None:
    flags = RenderFlags()
    overflow = {"emph": {"lines": 3, "max_lines": 2}}

    with caplog.at_level(logging.WARNING):
        await AnimationRecorder._flag_fit_overflow(_Page(overflow), "job-1", flags)

    assert flags.fit_overflow == {"emph": FitOverflow(lines=3, max_lines=2)}
    assert "overflows its template" in caplog.text and "job-1" in caplog.text


@pytest.mark.parametrize("value", [None, {}, "not-a-dict"])
async def test_text_that_fits_flags_nothing(value: Any) -> None:
    flags = RenderFlags()
    await AnimationRecorder._flag_fit_overflow(_Page(value), "job-1", flags)
    assert flags.fit_overflow == {}


def test_fit_reports_instead_of_clamping() -> None:
    lib = get_template("two_beat")
    assert "window.FIT_OVERFLOW" in lib
    assert "reportFit(el," in lib


@pytest.mark.parametrize("overflow", [{"emph": {"lines": 3, "max_lines": 2}}, None])
async def test_the_recording_result_carries_the_overflow(tmp_path: Any, overflow: Any) -> None:
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    from src.services.ffmpeg.base import ProbeResult

    out = Path(tmp_path) / "out.mp4"
    out.write_bytes(b"encoded")
    frames = Path(tmp_path) / "frames"
    frames.mkdir(exist_ok=True)

    async def _capture(*_args: Any, flags: RenderFlags | None = None, **_kw: Any) -> Any:
        if overflow and flags is not None:
            flags.fit_overflow = RenderFlags.model_validate({"fit_overflow": overflow}).fit_overflow
        return frames, 60, 2000.0, 0.6

    with (
        patch.object(AnimationRecorder, "_capture_frames", new=AsyncMock(side_effect=_capture)),
        patch.object(AnimationRecorder, "_assemble", return_value=out),
        patch(
            "src.services.recording.animation_recorder.FFmpegOps.probe_all",
            return_value=ProbeResult(duration=2.0, width=0, height=0, has_audio=False, probed=True),
        ),
    ):
        result = await AnimationRecorder().record_animation(
            job_id="anim-1", html_content="<html></html>", animation_config={"fps": 30}
        )

    assert result["success"] is True
    assert result.get("fit_overflow") == overflow


async def test_capture_frames_reads_the_template_report(tmp_path: Any) -> None:
    """The real _capture_frames asks the page for its fit report after the fonts load."""
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    from tests.unit.recording.test_animation_transparency import (
        _FakeBrowser,
        _FakeContext,
        _FakePage,
        _FakePlaywright,
        _make_recorder,
    )

    overflow = {"title": {"lines": 4, "max_lines": 3}}

    page = _FakePage()

    async def _evaluate(script: Any, *_args: Any) -> Any:
        return overflow if script == "window.FIT_OVERFLOW || null" else None

    setattr(page, "evaluate", _evaluate)  # noqa: B010 -- a fake page answering the fit report
    flags = RenderFlags()
    with (
        patch(
            "playwright.async_api.async_playwright",
            return_value=_FakePlaywright(_FakeBrowser(_FakeContext(page))),
        ),
        patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
    ):
        await _make_recorder()._capture_frames(
            "job-f",
            Path(tmp_path),
            "<html><body></body></html>",
            {"width": 320, "height": 240, "headless": True, "color_scheme": "light"},
            1,
            1000,
            None,
            None,
            flags=flags,
        )

    assert flags.fit_overflow == {"title": FitOverflow(lines=4, max_lines=3)}
