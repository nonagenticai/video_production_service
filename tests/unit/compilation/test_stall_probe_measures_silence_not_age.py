"""The compilation stall probe must measure SILENCE, not elapsed time.

``storage.stale_job_timeout_seconds`` documents itself as "seconds without
PROGRESS", and ``stale_compilation_jobs`` correctly asks
``updated_at <= now() - interval``. But on the compilation domain nothing ever
wrote ``updated_at`` between ``mark_processing`` and a terminal transition --
``update_progress`` exists only on ``recording/job_manager.py`` -- so the
predicate degenerated to "started more than 600s ago". p50 compile is ~7.6
minutes (456s), so the reaper failed ORDINARY healthy compiles, not edge cases.

Worse, it did so while the compile kept running: ``_finalize`` then lost its
``mark_completed`` to the reaper's terminal write, swallowed the
``JobStateConflictError`` and returned normally, and ``process_job`` fired a
``"completed"`` webhook anyway -- carrying an ``output_path`` that was never
persisted. One defect producing both a false failure and a false success.

The fix is a heartbeat that beats on a COUNTER, not a clock: it writes only when
``ctx.progress_ticks`` has moved. That is what separates the two tests that
matter here -- a progressing compile must survive past the threshold, and a
wedged one must still be reaped. A test that only exercised a stalled job could
not tell the fix from the bug.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation.compiler_service import CompilerService
from tests.conftest import TENANT_U4


def _ctx(**kw) -> _JobContext:
    return _JobContext(
        job_id=kw.pop("job_id", "job-hb"),
        tenant_id=TENANT_U4,
        job=MagicMock(),
        settings=MagicMock(),
        **kw,
    )


class TestProgressHeartbeat:
    """The probe's two halves, tested against each other."""

    async def test_a_progressing_compile_keeps_bumping_updated_at(self):
        """A healthy long compile is NEVER reaped, however long it runs.

        This is the half the old code got wrong. The job here runs well past the
        threshold and is doing real work the whole time.
        """
        service = CompilerService(MagicMock())
        ctx = _ctx()
        writes: list[tuple[float | None, str | None]] = []

        async def _record(db, job_id, *, progress=None, stage=None):
            writes.append((progress, stage))
            return True

        with (
            patch("src.services.compilation.compiler_service.get_db_session") as mock_session,
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_progress",
                new=_record,
            ),
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            hb = asyncio.create_task(service._progress_heartbeat(ctx, 0.02))
            # Simulate a compile that keeps advancing across several beats.
            for i in range(5):
                ctx.mark_progress(stage="processing", percent=float(i * 10))
                await asyncio.sleep(0.05)
            hb.cancel()
            with pytest.raises(asyncio.CancelledError):
                await hb

        assert len(writes) >= 3, (
            f"a compile that was actively progressing wrote updated_at only "
            f"{len(writes)} time(s); the reaper would age it out and fail it mid-run"
        )
        assert writes[-1][1] == "processing"

    async def test_a_wedged_compile_writes_nothing_and_is_left_to_the_reaper(self):
        """The half a naive timer-heartbeat would have destroyed.

        A wedged compile (stalled MinIO read, an ffmpeg step in
        ``asyncio.to_thread`` that never returns) leaves the event loop perfectly
        responsive, so the beats keep firing. They must find no movement and write
        NOTHING, so the row ages and the reaper collects it. A heartbeat that beat
        on the clock alone would renew a wedged job forever -- the
        leaked-lease-renewed-forever shape that hid the nine-hour recording wedge.
        """
        service = CompilerService(MagicMock())
        ctx = _ctx()
        writes: list = []

        async def _record(db, job_id, *, progress=None, stage=None):
            writes.append(stage)
            return True

        with (
            patch("src.services.compilation.compiler_service.get_db_session") as mock_session,
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_progress",
                new=_record,
            ),
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            ctx.mark_progress(stage="processing", percent=10.0)  # one tick, then WEDGE
            hb = asyncio.create_task(service._progress_heartbeat(ctx, 0.02))
            await asyncio.sleep(0.25)  # ~12 beats, zero progress
            hb.cancel()
            with pytest.raises(asyncio.CancelledError):
                await hb

        assert len(writes) == 1, (
            f"a WEDGED compile wrote updated_at {len(writes)} times; a heartbeat "
            f"that renews a stalled job makes it unreapable forever"
        )

    async def test_it_stops_beating_once_the_row_leaves_processing(self):
        """A heartbeat must not argue with a terminal state written by somebody else."""
        service = CompilerService(MagicMock())
        ctx = _ctx()
        calls = 0

        async def _record(db, job_id, *, progress=None, stage=None):
            nonlocal calls
            calls += 1
            return False  # cancelled / already reaped

        with (
            patch("src.services.compilation.compiler_service.get_db_session") as mock_session,
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_progress",
                new=_record,
            ),
        ):
            mock_session.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)

            ctx.mark_progress(stage="processing")
            hb = asyncio.create_task(service._progress_heartbeat(ctx, 0.02))
            await asyncio.sleep(0.15)
            for i in range(3):
                ctx.mark_progress(percent=float(i))
                await asyncio.sleep(0.03)

        assert hb.done(), "the heartbeat kept running after the row left 'processing'"
        assert calls == 1

    def test_the_interval_beats_several_times_inside_the_reaper_window(self):
        """Sizing, so a single slow beat can never by itself trip the reaper."""
        assert CompilerService._heartbeat_interval(600) == 30.0
        assert CompilerService._heartbeat_interval(60) == 15.0
        assert CompilerService._heartbeat_interval(8) == 5.0  # floor
        assert CompilerService._heartbeat_interval(100000) == 30.0  # ceiling


class TestProgressTicks:
    """The counter the heartbeat reads. If nothing ticks, the fix is inert."""

    def test_every_completed_stage_is_a_tick(self):
        ctx = _ctx()
        before = ctx.progress_ticks
        ctx.record_stage("assemble", 0.0)
        assert ctx.progress_ticks == before + 1
        assert ctx.progress_stage == "assemble"

    def test_the_per_part_loop_ticks(self):
        """The long stage must tick per PART, not once for the whole stage.

        Stage-granularity alone would look silent for the whole of the segment
        loop -- which is where a compile spends most of its wall time -- and the
        reaper would fail a perfectly healthy job.
        """
        src = Path("src/services/compilation/_segment_builders.py").read_text()
        assert "ctx.mark_progress(" in src, (
            "the per-part segment loop does not report progress; the heartbeat "
            "would go silent for the longest stage of every compile"
        )


class TestReapedJobDoesNotAnnounceCompletion:
    """The false-success half: a job reaped as failed must not emit `completed`."""

    async def test_a_job_that_lost_mark_completed_fires_failed_not_completed(self):
        """`_finalize` swallows the conflict and returns normally -- so the webhook
        gate cannot be "did `_finalize` return", it must be "did it actually store"."""
        from src.services.compilation.job_manager import JobStateConflictError

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = MagicMock()
            storage.get_job_dir.return_value = work_dir
            storage.store_file = AsyncMock(side_effect=lambda j, p, f: str(work_dir / f))
            storage.presign_url = AsyncMock(return_value=None)
            storage.delete_job_dir = AsyncMock()
            storage.retrieve_uri = AsyncMock(return_value=None)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            from tests.unit.compilation.test_compiler_service_process_job import (
                _base_job,
                _make_settings_mock,
            )

            job = _base_job(
                webhook_url="https://example.com/hook",
                webhook_secret="s3cr3t",  # pragma: allowlist secret
                shot_list={
                    "parts": [{"id": "p1", "part_type": "browser_recording"}],
                    "recording_order": ["p1"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid),
                        "duration_seconds": 5.0,
                    }
                ],
            )

            with (
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.get_job",
                    new_callable=AsyncMock,
                    return_value=job,
                ),
                patch(
                    "src.services.compilation.compiler_service."
                    "CompilationJobManager.mark_processing",
                    new_callable=AsyncMock,
                ),
                # THE REAPER GOT THERE FIRST: the row is already 'failed'.
                patch(
                    "src.services.compilation.job_manager.CompilationJobManager.mark_completed",
                    new_callable=AsyncMock,
                    side_effect=JobStateConflictError("not processing"),
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=_make_settings_mock()),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
                patch.object(CompilerService, "_fire_webhook", new_callable=AsyncMock) as mock_hook,
            ):
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.extract_thumbnail = MagicMock()
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                (work_dir / "demo.mp4").write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)
                await asyncio.sleep(0)

            mock_hook.assert_awaited_once()
            call = mock_hook.await_args
            assert call is not None
            event = call.args[3]
            assert event == "failed", (
                f"a job that was reaped/cancelled before its result could be stored "
                f"announced {event!r} to its consumer -- a completion with no artifact"
            )
