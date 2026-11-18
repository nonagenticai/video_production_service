"""Extended unit tests for compiler service."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation.compiler_service import CompilerService
from tests.conftest import TENANT_U1, TENANT_U4


def _mixctx() -> _JobContext:
    """Context for _mix_per_segment_narration, used only to record a degradation
    when the mix fails and the un-narrated segment is returned."""
    return _JobContext(job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None)


def _assembly_ctx() -> _JobContext:
    """Minimal context for the assembly helpers, which use it only to record
    degradations when an xfade step falls back to a hard concat."""
    return _JobContext(job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None)


def _mock_storage():
    """Create a mock StorageService with async methods properly mocked."""
    storage = MagicMock()
    storage.store_file = AsyncMock(return_value="stored.mp4")
    storage.presign_url = AsyncMock(return_value="https://example.com/presigned")
    storage.delete_job_dir = AsyncMock()
    storage.retrieve_uri = AsyncMock(return_value=None)
    storage.store_bytes = AsyncMock(return_value="stored")
    storage.health_check = AsyncMock(return_value=True)
    storage.initialize = AsyncMock()
    return storage


class TestCompilerServiceInit:
    def test_init(self):
        storage = _mock_storage()
        service = CompilerService(storage)
        assert service._storage is storage
        assert service._ws_manager is None

    def test_init_with_ws_manager(self):
        storage = _mock_storage()
        ws = MagicMock()
        service = CompilerService(storage, ws_manager=ws)
        assert service._ws_manager is ws


class TestCompilerServiceTaskManagement:
    def test_register_task(self):
        service = CompilerService(_mock_storage())
        task = MagicMock()
        service.register_task("job-1", task)
        assert "job-1" in service._active_tasks

    def test_unregister_task(self):
        service = CompilerService(_mock_storage())
        task = MagicMock()
        service._active_tasks["job-1"] = task
        service.unregister_task("job-1")
        assert "job-1" not in service._active_tasks

    def test_unregister_nonexistent_task(self):
        service = CompilerService(_mock_storage())
        # Should not raise
        service.unregister_task("nonexistent")


class TestCompilerServiceCancelJob:
    async def test_cancel_with_active_task(self):
        service = CompilerService(_mock_storage())
        task = MagicMock()
        task.done.return_value = False
        service._active_tasks["job-1"] = task

        db = AsyncMock()
        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_cancelled",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await service.cancel_job(db, "job-1")

        assert result is True
        task.cancel.assert_called_once()
        assert "job-1" not in service._active_tasks

    async def test_cancel_with_done_task(self):
        service = CompilerService(_mock_storage())
        task = MagicMock()
        task.done.return_value = True
        service._active_tasks["job-1"] = task

        db = AsyncMock()
        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_cancelled",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await service.cancel_job(db, "job-1")

        assert result is True
        task.cancel.assert_not_called()

    async def test_cancel_no_active_task(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_cancelled",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await service.cancel_job(db, "job-1")

        assert result is True

    async def test_cancel_returns_false(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_cancelled",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await service.cancel_job(db, "job-1")

        assert result is False


class TestCompilerServiceCreateJob:
    async def test_create_job_basic(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "new-job-id"

        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            result = await service.create_job(
                db,
                tenant_id=TENANT_U1,
                shot_list={"parts": []},
                recordings=[],
            )

        assert result == "new-job-id"
        mock_create.assert_awaited_once()

    async def test_create_job_with_theme_name(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "job-with-theme"

        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            await service.create_job(
                db,
                tenant_id=TENANT_U1,
                shot_list={"parts": []},
                recordings=[],
                theme_name="corporate",
                theme_config={"primary_color": "#FFF"},
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["theme_config"]["theme_name"] == "corporate"
        assert call_kwargs["theme_config"]["primary_color"] == "#FFF"

    async def test_create_job_with_filters(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "job-with-filters"

        with patch(
            "src.services.compilation.compiler_service.CompilationJobManager.create_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ) as mock_create:
            await service.create_job(
                db,
                tenant_id=TENANT_U1,
                shot_list={"parts": []},
                recordings=[],
                color_correction={"brightness": 1.2},
                watermark={"image_path": "/logo.png"},
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["filter_config"]["color_correction"]["brightness"] == 1.2
        assert call_kwargs["filter_config"]["watermark"]["image_path"] == "/logo.png"


class TestCompilerServiceProcessJob:
    async def test_process_job_not_found(self):
        service = CompilerService(_mock_storage())
        db = AsyncMock()

        with (
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(ValueError, match="Job not found"),
        ):
            await service.process_job(db, "missing-job", TENANT_U4)

    async def test_process_job_no_segments(self):
        """When there are no segments, job should be marked failed with no-segments reason."""
        storage = _mock_storage()
        service = CompilerService(storage)
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "empty-job"
        mock_job.shot_list = {"parts": [], "recording_order": []}
        mock_job.recordings = []
        mock_job.edit_decisions = None
        mock_job.theme_config = None
        mock_job.narration_audio_dir = None
        mock_job.clips_dir = None
        mock_job.preset = None
        mock_job.target_width = None
        mock_job.target_height = None
        mock_job.crf = None
        mock_job.transition_type = None
        mock_job.crossfade_duration = None
        mock_job.filter_config = None
        mock_job.subtitles = None
        mock_job.webhook_url = None
        mock_job.webhook_secret = None

        with tempfile.TemporaryDirectory() as tmpdir:
            storage.get_job_dir.return_value = Path(tmpdir)

            with (
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.get_job",
                    new_callable=AsyncMock,
                    return_value=mock_job,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_processing",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                    new_callable=AsyncMock,
                ) as mock_failed,
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ) as mock_usage,
            ):
                await service.process_job(db, "empty-job", TENANT_U4)

            mock_failed.assert_awaited_once()
            assert "No segments to compile" in mock_failed.call_args[0][2]
            mock_usage.assert_awaited_once()
            assert mock_usage.call_args[1]["is_error"] is True


class TestCompilerServiceComputeChapters:
    def test_empty_segments(self):
        service = CompilerService(_mock_storage())
        result = service._compute_chapter_timestamps([], {})
        assert result == []

    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration", return_value=10.0)
    def test_segments_with_chapters(self, mock_dur):
        service = CompilerService(_mock_storage())
        seg1 = MagicMock(spec=Path)
        seg1.stem = "part1"
        seg1.exists.return_value = True
        seg2 = MagicMock(spec=Path)
        seg2.stem = "part2"
        seg2.exists.return_value = True

        parts_by_id = {
            "part1": {"title": "Intro", "subtitle": "", "chapter": 1},
            "part2": {"title": "Main", "subtitle": "Details", "chapter": 2},
        }
        result = service._compute_chapter_timestamps([seg1, seg2], parts_by_id)
        assert len(result) == 2
        assert result[0]["chapter_number"] == 1
        assert result[0]["title"] == "Intro"
        assert result[0]["start_seconds"] == 0.0
        assert result[1]["chapter_number"] == 2
        assert result[1]["start_seconds"] == 10.0

    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration", return_value=5.0)
    def test_segments_without_chapters(self, mock_dur):
        service = CompilerService(_mock_storage())
        seg1 = MagicMock(spec=Path)
        seg1.stem = "part1"
        seg1.exists.return_value = True

        parts_by_id = {
            "part1": {"title": "Intro", "subtitle": ""},
        }
        # No chapter field in part, so no chapter timestamps
        result = service._compute_chapter_timestamps([seg1], parts_by_id)
        assert result == []


class TestCompilerServiceAssemble:
    @patch("src.services.compilation.compiler_service.FFmpegOps.fade_to_black")
    def test_assemble_with_fade(self, mock_fade):
        service = CompilerService(_mock_storage())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "seg1.mp4"
            seg2 = work_dir / "seg2.mp4"
            seg1.write_bytes(b"fake1")
            seg2.write_bytes(b"fake2")
            output = work_dir / "output.mp4"

            mock_fade.return_value = work_dir / "fade_1.mp4"
            # Create the fade output so copy2 works
            (work_dir / "fade_1.mp4").write_bytes(b"faded")

            service._assemble_with_fade([seg1, seg2], output, work_dir, 0.5)
            mock_fade.assert_called_once()
            assert output.exists()

    @patch("src.services.compilation.compiler_service.FFmpegOps.concat")
    def test_assemble_smart_cut_transition(self, mock_concat):
        service = CompilerService(_mock_storage())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "part1.mp4"
            seg2 = work_dir / "part2.mp4"
            seg1.write_bytes(b"fake1")
            seg2.write_bytes(b"fake2")
            output = work_dir / "output.mp4"

            # Same chapter, not title card => grouped and concat'd (no crossfade)
            parts = {
                "part1": {"chapter": 1, "part_type": "browser_recording"},
                "part2": {"chapter": 1, "part_type": "browser_recording"},
            }

            chapter_out = work_dir / "chapter_0.mp4"
            mock_concat.return_value = chapter_out
            chapter_out.write_bytes(b"merged")

            service._assemble_smart(
                _assembly_ctx(), [seg1, seg2], parts, ["part1", "part2"], output, work_dir, 0.5
            )
            mock_concat.assert_called_once()

    @patch("src.services.compilation.compiler_service.FFmpegOps.crossfade")
    def test_assemble_smart_crossfade(self, mock_crossfade):
        service = CompilerService(_mock_storage())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "part1.mp4"
            seg2 = work_dir / "part2.mp4"
            seg1.write_bytes(b"fake1")
            seg2.write_bytes(b"fake2")
            output = work_dir / "output.mp4"

            # Different chapters => each is its own group, xfade between them
            parts = {
                "part1": {"chapter": 1, "part_type": "browser_recording"},
                "part2": {"chapter": 2, "part_type": "browser_recording"},
            }

            mock_crossfade.return_value = work_dir / "merged_1.mp4"
            (work_dir / "merged_1.mp4").write_bytes(b"crossfaded")

            service._assemble_smart(
                _assembly_ctx(), [seg1, seg2], parts, ["part1", "part2"], output, work_dir, 0.5
            )
            mock_crossfade.assert_called_once()


class TestCompilerServiceMixNarration:
    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration", return_value=0.0)
    def test_mix_narration_zero_duration_returns_original(self, mock_dur):
        service = CompilerService(_mock_storage())
        result = service._mix_per_segment_narration(
            _mixctx(), Path("/video.mp4"), Path("/audio.mp3"), Path("/work"), "part1"
        )
        assert result == Path("/video.mp4")

    @patch("src.services.compilation.compiler_service.FFmpegOps.mix_voiceover")
    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration")
    def test_mix_narration_matching_duration(self, mock_dur, mock_mix):
        mock_dur.side_effect = [10.0, 10.0]  # audio_dur, video_dur
        mock_mix.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            mixed_path = work / "part1_narrated.mp4"
            mixed_path.write_bytes(b"mixed")

            service = CompilerService(_mock_storage())
            result = service._mix_per_segment_narration(
                _mixctx(), Path("/video.mp4"), Path("/audio.mp3"), work, "part1"
            )
            # Bug 1 fix: _narrated.mp4 is renamed to {part_id}.mp4 so
            # downstream lookups by seg.stem match the part_id.
            assert result == work / "part1.mp4"

    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration")
    def test_mix_narration_error_returns_original(self, mock_dur):
        mock_dur.side_effect = OSError("file not found")

        service = CompilerService(_mock_storage())
        result = service._mix_per_segment_narration(
            _mixctx(), Path("/video.mp4"), Path("/audio.mp3"), Path("/work"), "part1"
        )
        assert result == Path("/video.mp4")


class TestCompilerServiceWebhook:
    async def test_fire_webhook_success(self):
        service = CompilerService(_mock_storage())
        with patch(
            "src.services.compilation.webhook_service.fire_webhook",
            new_callable=AsyncMock,
        ) as mock_fire:
            await service._fire_webhook(
                "https://example.com/hook", "secret", "job-1", "completed", {"key": "value"}
            )
            # NOTE the deliberate re-ordering: this wrapper takes
            # (url, secret, job_id, event, data) and the transport takes
            # (url, secret, event, job_id, data). Both middle values are plain
            # strings, so swapping them raises nothing and delivers a webhook
            # announcing event "job-1" for job "completed" — invisible to a
            # call-count assertion, and only visible to the receiver.
            mock_fire.assert_awaited_once_with(
                "https://example.com/hook",
                "secret",  # pragma: allowlist secret
                "completed",
                "job-1",
                {"key": "value"},
            )

    async def test_fire_webhook_failure_does_not_raise(self):
        service = CompilerService(_mock_storage())
        with patch(
            "src.services.compilation.webhook_service.fire_webhook",
            new_callable=AsyncMock,
            side_effect=OSError("network error"),
        ):
            # Should not raise
            await service._fire_webhook(
                "https://example.com/hook", "secret", "job-1", "completed", {}
            )
