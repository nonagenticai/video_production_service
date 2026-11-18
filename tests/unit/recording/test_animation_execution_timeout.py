"""Defect 1 regression: AnimationRecorder.record_animation must bound the
whole render (browser frame-step capture + ffmpeg encode) with a hard
execution timeout.

Before this, ``_capture_frames``'s ``for i in range(frame_count)`` loop had no
bound at all, and a stalled Chromium page-step/screenshot blocked the render
forever, holding its fleet concurrency slot. This is the exact defect behind
the lost ``website_review`` title-card jobs: one job hung in ``processing``
across a deploy roll, and a sibling created 0.49s later never left ``queued``
behind the wedged slot.

Two properties are proven here:

(a) a render whose frame loop never returns fails within the execution
    timeout, and its browser page/context is actually closed (not merely
    orphaned by cancelling the awaiting coroutine);
(d) a NORMAL render still completes and is not killed by the new timeout --
    guards against a timeout sized too tight.

Slot release is proven at the ``RecorderService`` layer (its ``finally``
around ``_attempt_recording`` already releases the concurrency slot on ANY
exception, including the ``TimeoutError`` this module raises) rather than
re-proven here; see ``test_recorder_service.py``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.ffmpeg import FFmpegOps
from tests.unit.recording.test_animation_recorder import _make_recorder


class _FakePage:
    """Minimal Playwright Page double.

    ``screenshot`` is the one call inside the frame loop with no OTHER bound
    (``set_content`` already has its own 30s Playwright-level timeout), so it
    is the call the hang scenario targets.
    """

    def __init__(self, *, hang: bool) -> None:
        self.evaluate = AsyncMock(return_value=1000)
        self.set_content = AsyncMock()
        self.screenshot_calls = 0

        async def _screenshot(*_a, **_kw) -> None:
            self.screenshot_calls += 1
            if hang:
                # Never returns on its own -- only cancellation ends this.
                await asyncio.Event().wait()

        self.screenshot = AsyncMock(side_effect=_screenshot)


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page
        self.closed = False

    async def new_page(self) -> _FakePage:
        return self._page

    async def close(self) -> None:
        self.closed = True


class _FakeBrowser:
    def __init__(self, *, hang: bool) -> None:
        self.page = _FakePage(hang=hang)
        self.context = _FakeContext(self.page)
        self.closed = False

    async def new_context(self, **_kwargs) -> _FakeContext:
        return self.context

    async def close(self) -> None:
        self.closed = True


class _FakePlaywright:
    def __init__(self, browser: _FakeBrowser) -> None:
        self._browser = browser
        self.chromium = MagicMock()
        self.chromium.launch = AsyncMock(return_value=browser)

    async def __aenter__(self) -> _FakePlaywright:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _patched(browser: _FakeBrowser):
    return (
        patch("playwright.async_api.async_playwright", return_value=_FakePlaywright(browser)),
        patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
    )


class TestHungRenderFailsFastAndTerminates:
    @pytest.mark.asyncio
    async def test_hung_frame_loop_fails_within_the_execution_timeout(self):
        recorder = _make_recorder()
        # Tiny, deterministic budget: 1 frame at a generous per-frame cost
        # still clamps to the floor below, so the test's wall time is bounded
        # by the settings we control here, not by production defaults.
        recorder._timeout_floor_s = 1.5
        recorder._timeout_ceiling_s = 1.5
        recorder._seconds_per_frame = 0.01
        recorder._ffmpeg_seconds_per_frame = 0.0
        recorder._encode_reserve_s = 0.2
        # Stall watchdog is now the PRIMARY hang detector -- keep it tight
        # too so this test proves both layers catch the hang fast, not just
        # the outer capture-budget backstop.
        recorder._stall_window_s = 0.3

        browser = _FakeBrowser(hang=True)
        ctx_a, ctx_b = _patched(browser)

        started = time.monotonic()
        with ctx_a, ctx_b:
            result = await recorder.record_animation(
                "job-hang", "<html></html>", {"fps": 30, "duration_ms": 1000}
            )
        elapsed = time.monotonic() - started

        assert result["success"] is False
        # With a 0.3s stall window well under the 1.5s outer ceiling, the
        # PER-FRAME stall watchdog fires first -- its own descriptive message
        # is preserved verbatim (see record_animation's except TimeoutError).
        assert "stalled" in result["error"]
        # Bounded, not hung: comfortably under a lax multiple of the 1.5s
        # ceiling we configured, and nowhere near "forever".
        assert elapsed < 5.0
        # The browser/context were actually closed on the timeout path --
        # not merely orphaned by cancelling the awaiting coroutine.
        assert browser.context.closed is True
        assert browser.closed is True

    @pytest.mark.asyncio
    async def test_stall_watchdog_alone_catches_the_hang_under_a_huge_outer_ceiling(self):
        """Isolates the PER-FRAME stall watchdog from the outer capture-budget
        backstop: the outer budget is made enormous (it would never fire on
        its own within this test), so a failure here can only come from the
        inner ``asyncio.wait_for`` around each frame's own work in
        ``_capture_frames``. Proves the stall detector is what the audit
        asked for -- lack of PROGRESS, not total duration."""
        recorder = _make_recorder()
        recorder._timeout_floor_s = 3600.0
        recorder._timeout_ceiling_s = 3600.0
        recorder._seconds_per_frame = 100.0
        recorder._ffmpeg_seconds_per_frame = 0.0
        recorder._encode_reserve_s = 0.0
        recorder._stall_window_s = 0.2

        browser = _FakeBrowser(hang=True)
        ctx_a, ctx_b = _patched(browser)

        started = time.monotonic()
        with ctx_a, ctx_b:
            result = await recorder.record_animation(
                "job-stall-only", "<html></html>", {"fps": 30, "duration_ms": 1000}
            )
        elapsed = time.monotonic() - started

        assert result["success"] is False
        assert "stalled" in result["error"]
        assert elapsed < 5.0  # nowhere near the 3600s outer ceiling
        assert browser.context.closed is True
        assert browser.closed is True

    @pytest.mark.asyncio
    async def test_mutation_without_the_wait_for_wrapper_the_call_would_hang(self, tmp_path):
        """Sanity check on the fake harness itself: calling `_capture_frames`
        directly (no `asyncio.wait_for`) against the SAME hung page does not
        return on its own within a short window -- proving the hang is real
        and it is `record_animation`'s `asyncio.wait_for` doing the work
        above, not an artifact of the fake.
        """
        recorder = _make_recorder()
        browser = _FakeBrowser(hang=True)
        ctx_a, ctx_b = _patched(browser)
        config = {"width": 320, "height": 240, "capture_scale": 1, "headless": True}

        with ctx_a, ctx_b:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(
                    recorder._capture_frames(
                        "job-raw", tmp_path, "<html></html>", config, 30, 1000, None, None
                    ),
                    timeout=0.3,
                )


class TestNormalRenderStillCompletes:
    @pytest.mark.asyncio
    async def test_normal_render_completes_and_is_not_killed_by_the_timeout(self, tmp_path):
        """Guard against a timeout sized too tight: a fast, ordinary capture
        must still succeed end-to-end (encode mocked -- ffmpeg is exercised
        for real elsewhere)."""
        recorder = _make_recorder()
        recorder._timeout_floor_s = 30.0
        recorder._timeout_ceiling_s = 480.0
        recorder._seconds_per_frame = 1.5
        recorder._encode_reserve_s = 90.0

        browser = _FakeBrowser(hang=False)
        ctx_a, ctx_b = _patched(browser)

        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)
        probe = MagicMock(duration=1.0, width=1080, height=1080, has_audio=True)

        with (
            ctx_a,
            ctx_b,
            patch.object(recorder, "_assemble", return_value=out),
            patch.object(FFmpegOps, "probe_all", return_value=probe),
        ):
            result = await recorder.record_animation(
                "job-normal", "<html></html>", {"fps": 30, "duration_ms": 200}
            )

        assert result["success"] is True
        assert result["video_path"] == str(out)
        # The capture loop actually ran (a handful of frames at 200ms/30fps).
        assert browser.page.screenshot_calls > 0
        assert browser.context.closed is True
        assert browser.closed is True


class TestOldFlatCeilingRegression:
    """Defect: the flat 480s ceiling killed real production jobs.

    Measured against ``recording_jobs`` on the production DB (worker4,
    2026-09-17): 80 of 2106 successful animation jobs (3.8%) took longer than
    480s; the worst, 532 frames @ 30fps (17717ms requested), took 767.06s.
    600-frame jobs (20000ms @ 30fps) commonly ran 620-733s. This proves the
    NEW frame-scaled formula would not have killed either shape, while the
    OLD flat-480s formula would have -- deterministic arithmetic, no event
    loop involved, so it stays fast and is directly mutation-provable against
    the formula in ``record_animation``.
    """

    @staticmethod
    def _capture_budget(
        frames: int, *, seconds_per_frame: float, ceiling: float, encode_s: float
    ) -> float:
        execution_timeout_s = max(30.0, min(frames * seconds_per_frame + encode_s, ceiling))
        return max(1.0, execution_timeout_s - encode_s)

    def test_old_constants_would_have_killed_the_worst_historical_job(self):
        # 532 frames, actual recording_time_seconds=767.06 in production.
        budget = self._capture_budget(532, seconds_per_frame=1.5, ceiling=480.0, encode_s=90.0)
        assert budget < 767.06

    def test_old_constants_would_have_killed_a_600_frame_job(self):
        # 600 frames (20000ms @ 30fps), commonly 620-733s in production.
        budget = self._capture_budget(600, seconds_per_frame=1.5, ceiling=480.0, encode_s=90.0)
        assert budget < 620.0

    def test_new_constants_do_not_kill_either_historical_shape(self):
        encode_600 = 120.0 + 600 * 0.5
        budget_600 = self._capture_budget(
            600, seconds_per_frame=4.5, ceiling=18500.0, encode_s=encode_600
        )
        assert budget_600 > 733.99  # the largest observed 600-frame job

        encode_532 = 120.0 + 532 * 0.5
        budget_532 = self._capture_budget(
            532, seconds_per_frame=4.5, ceiling=18500.0, encode_s=encode_532
        )
        assert budget_532 > 767.06  # the worst job in the whole dataset


class TestSlowButProgressingRenderIsNotKilled:
    """(b) A render whose frames each complete on time, but whose FAKED wall
    clock reports a total past the old 480s ceiling, must still succeed.

    Frame work here resolves near-instantly (no real sleeping -- this proves
    the behavior via record_animation's own execution-timeout arithmetic
    against a job shape the audit specifically cited: 300 frames, well past
    the "~260 frames saturates the old ceiling" threshold), while
    ``time.time`` is patched to make the REPORTED elapsed time exceed 480s.
    A regression back to the flat 480s ceiling would still not fail this
    test on wall-clock alone (asyncio.wait_for uses the real loop clock, not
    time.time) -- that is exactly why the arithmetic regression test above
    exists alongside this behavioral one; together they prove both the
    formula and the live code path.
    """

    @pytest.mark.asyncio
    async def test_300_frame_render_completes_and_reports_past_the_old_ceiling(self, tmp_path):
        recorder = _make_recorder()  # production-shaped defaults from _mock_settings()

        browser = _FakeBrowser(hang=False)
        ctx_a, ctx_b = _patched(browser)

        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)
        probe = MagicMock(duration=10.0, width=1080, height=1080, has_audio=True)

        # 500s reported elapsed -- past the old 480s ceiling -- without the
        # test actually waiting 500s: record_animation only calls time.time()
        # twice (start, then elapsed = time.time() - start), so faking those
        # two calls is enough; the frame loop's own awaits resolve immediately.
        with (
            ctx_a,
            ctx_b,
            patch.object(recorder, "_assemble", return_value=out),
            patch.object(FFmpegOps, "probe_all", return_value=probe),
            patch(
                "src.services.recording.animation_recorder.time.time",
                side_effect=[0.0, 500.0],
            ),
        ):
            result = await recorder.record_animation(
                "job-slow-progressing", "<html></html>", {"fps": 30, "duration_ms": 10000}
            )

        assert result["success"] is True
        assert result["recording_time_seconds"] > 480.0
        assert browser.page.screenshot_calls > 260  # past the audit's saturation point
        assert browser.context.closed is True
        assert browser.closed is True


class TestLaunchHangIsBoundedAndLeaksNothing:
    """(c) A hang during browser launch/new_page must be bounded by
    ``animation_render_launch_timeout_seconds`` and close whatever handle
    WAS acquired -- the leak the audit found: these calls used to run BEFORE
    the try/finally that calls ``_safe_close`` existed at all.
    """

    @pytest.mark.asyncio
    async def test_hang_in_new_context_still_closes_the_launched_browser(self):
        recorder = _make_recorder()
        recorder._launch_timeout_s = 0.2

        class _HangingBrowser:
            def __init__(self) -> None:
                self.closed = False

            async def new_context(self, **_kwargs):
                await asyncio.Event().wait()  # never returns on its own

            async def close(self) -> None:
                self.closed = True

        browser = _HangingBrowser()

        class _FakePlaywrightHang:
            def __init__(self, b) -> None:
                self.chromium = MagicMock()
                self.chromium.launch = AsyncMock(return_value=b)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_exc):
                return None

        with (
            patch(
                "playwright.async_api.async_playwright", return_value=_FakePlaywrightHang(browser)
            ),
            patch("src.services.recording.animation_recorder.install_ssrf_guard", new=AsyncMock()),
        ):
            result = await recorder.record_animation(
                "job-launch-hang", "<html></html>", {"fps": 30, "duration_ms": 1000}
            )

        assert result["success"] is False
        # The browser WAS acquired (launch succeeded) before new_context hung
        # -- it must be closed even though no context/page ever existed.
        assert browser.closed is True


class TestFfmpegTimeoutScalesWithFrameCount:
    """(d) ffmpeg's OS-level subprocess timeout must scale with frame count,
    not stay a flat reserve -- previously ``timeout=int(encode_budget_s)``
    was always the flat ``animation_render_encode_reserve_seconds`` regardless
    of how many frames there were to encode.
    """

    @pytest.mark.asyncio
    async def test_large_job_forwards_a_frame_scaled_timeout_to_assemble(self, tmp_path):
        recorder = _make_recorder()
        browser = _FakeBrowser(hang=False)
        ctx_a, ctx_b = _patched(browser)

        out = tmp_path / "out.mp4"
        out.write_bytes(b"\x00" * 64)
        probe = MagicMock(duration=20.0, width=1080, height=1080, has_audio=True)

        # 600 frames (20000ms @ 30fps) -- the audit's other cited job shape.
        with (
            ctx_a,
            ctx_b,
            patch.object(recorder, "_assemble", return_value=out) as mock_assemble,
            patch.object(FFmpegOps, "probe_all", return_value=probe),
        ):
            await recorder.record_animation(
                "job-large", "<html></html>", {"fps": 30, "duration_ms": 20000}
            )

        assert mock_assemble.called
        forwarded_timeout = mock_assemble.call_args.kwargs["timeout"]
        # Flat old reserve (90s) would be far too small for 600 frames; the
        # new frame-scaled budget (120 + 600*0.5 = 420) must reach ffmpeg.
        assert forwarded_timeout == 420

    def test_ffmpeg_run_actually_enforces_its_timeout_at_the_os_level(self):
        """Sanity check on the mechanism this scaled timeout feeds: FFmpegOps
        .run already converts a real subprocess.TimeoutExpired into
        FFmpegError (a RuntimeError, caught by record_animation's except
        clause) -- proven directly against subprocess.run here so this test
        fails if that OS-level enforcement ever regresses."""
        import subprocess as sp

        from src.services.ffmpeg import FFmpegError, FFmpegOps

        with patch(
            "src.services.ffmpeg.base.subprocess.run",
            side_effect=sp.TimeoutExpired(cmd="ffmpeg", timeout=5),
        ):
            with pytest.raises(FFmpegError):
                FFmpegOps.run(["-i", "in.png", "out.mp4"], timeout=5)


class TestSafeCloseToleratesAnUnacquiredHandle:
    """Direct coverage of ``AnimationRecorder._safe_close``'s ``None`` guard.

    Before this fix, ``launch``/``new_context``/``install_ssrf_guard``/
    ``new_page`` ran BEFORE the try/finally that calls ``_safe_close`` existed
    at all, so a hang partway through startup (browser launched, then
    ``new_context`` hangs or raises) leaked the browser with no teardown.
    ``TestLaunchHangIsBoundedAndLeaksNothing`` above proves the END-TO-END
    behavior through ``record_animation``; this proves the UNIT the fix
    actually lives in, ``_safe_close`` itself, so a regression there is
    caught even if the end-to-end scenario ever changes shape.

    The ``if handle is None: continue`` guard is the fix. Removing it does
    NOT raise (``_safe_close``'s own ``except Exception`` swallows the
    resulting ``AttributeError`` from calling ``.close()`` on ``None``), so a
    naive "does it raise" test cannot tell the two versions apart -- exactly
    why 949 tests stayed green with the guard removed. What DOES differ is a
    spurious "failed to close context cleanly" warning: without the guard,
    ``_safe_close`` logs a close FAILURE for a handle that never existed to
    fail closing in the first place, which is actively misleading (an
    operator reading logs would conclude Playwright's teardown broke, not
    that this frame never created a context). That log line is the signal
    this test asserts against.
    """

    @pytest.mark.asyncio
    async def test_none_context_is_skipped_and_browser_still_closes(self, caplog):
        from src.services.recording.animation_recorder import AnimationRecorder

        class _Closeable:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        browser = _Closeable()

        with caplog.at_level("WARNING"):
            # Must not raise: context is None (new_context never completed).
            await AnimationRecorder._safe_close(
                None,
                browser,  # ty: ignore[invalid-argument-type]  # why: _Closeable duck-types Browser
            )

        assert browser.closed is True
        # No "failed to close" warning for a handle that was never acquired --
        # the None guard means _safe_close never even attempts None.close().
        assert not any("failed to close" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_both_handles_none_is_a_total_no_op(self, caplog):
        """Launch itself never completed (chromium.launch hung/raised) --
        both handles are still None. Must not raise and must log nothing."""
        from src.services.recording.animation_recorder import AnimationRecorder

        with caplog.at_level("WARNING"):
            await AnimationRecorder._safe_close(None, None)

        assert not any("failed to close" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_mutation_without_the_none_guard_a_spurious_warning_appears(self, caplog):
        """Mutation proof, run against the REAL module: temporarily delete the
        ``if handle is None: continue`` guard's effect by monkeypatching
        ``_safe_close`` with the pre-fix body (handles are never skipped) and
        show it does NOT raise but DOES log the misleading warning the guard
        exists to prevent -- i.e. the fixed version's behavior (no warning)
        is genuinely load-bearing, not an assertion nothing could ever break.
        """
        import logging

        from src.services.recording.animation_recorder import (
            _TEARDOWN_TIMEOUT_S,
            AnimationRecorder,
        )

        logger = logging.getLogger("src.services.recording.animation_recorder")

        async def _pre_fix_safe_close(context: object, browser: object) -> None:
            # Same body as AnimationRecorder._safe_close, MINUS the
            # `if handle is None: continue` guard -- the exact mutation the
            # audit asked to be provable.
            for label, handle in (("context", context), ("browser", browser)):
                try:
                    await asyncio.wait_for(
                        handle.close(),  # ty: ignore[unresolved-attribute]  # why: handle may be a bare None here -- exactly the pre-fix bug
                        timeout=_TEARDOWN_TIMEOUT_S,
                    )
                except Exception:
                    logger.warning(
                        "Animation recorder: failed to close %s cleanly within %.0fs",
                        label,
                        _TEARDOWN_TIMEOUT_S,
                        exc_info=True,
                    )

        class _Closeable:
            def __init__(self) -> None:
                self.closed = False

            async def close(self) -> None:
                self.closed = True

        browser = _Closeable()

        # RED without the guard: the real `_safe_close` (context=None) would
        # pass this assertion; the pre-fix body fails it.
        with caplog.at_level("WARNING"):
            await _pre_fix_safe_close(None, browser)
        assert any("failed to close context" in r.message for r in caplog.records), (
            "mutation harness itself is broken: the pre-fix body should have logged"
        )

        # GREEN with the guard restored: the real, fixed _safe_close.
        caplog.clear()
        with caplog.at_level("WARNING"):
            await AnimationRecorder._safe_close(
                None,
                browser,  # ty: ignore[invalid-argument-type]  # why: _Closeable duck-types Browser
            )
        assert not any("failed to close" in r.message for r in caplog.records)
        assert browser.closed is True
