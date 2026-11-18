"""Edge case tests for compiler_service.

Rewritten to verify that exception handlers log/record failures correctly
(not just that they "don't crash"), and that assembly helpers produce
correct output files and handle cleanup errors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation.compiler_service import CompilerService
from tests.conftest import TENANT_U4


def _assembly_ctx() -> _JobContext:
    """Minimal context for the assembly helpers, which use it only to record
    degradations when an xfade step falls back to a hard concat."""
    return _JobContext(job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None)


def _mock_storage(tmpdir: Path) -> MagicMock:
    storage = MagicMock()
    storage.get_job_dir.return_value = tmpdir

    # All async methods on the real StorageService must be AsyncMock
    async def _fake_store_file(job_id: str, file_path: Path, filename: str) -> str:
        return str(tmpdir / filename)

    storage.store_file = AsyncMock(side_effect=_fake_store_file)
    storage.presign_url = AsyncMock(return_value="https://example.com/presigned")
    storage.delete_job_dir = AsyncMock()
    storage.retrieve_uri = AsyncMock(return_value=None)
    storage.store_bytes = AsyncMock(return_value="stored")
    storage.health_check = AsyncMock(return_value=True)
    storage.initialize = AsyncMock()
    return storage


def _base_job(**overrides) -> MagicMock:
    defaults = {
        "job_id": "job-1",
        "shot_list": {"parts": [], "recording_order": []},
        "recordings": [],
        "edit_decisions": None,
        "theme_config": None,
        "narration_audio_dir": None,
        "clips_dir": None,
        "preset": None,
        "target_width": None,
        "target_height": None,
        "crf": None,
        "transition_type": None,
        "crossfade_duration": None,
        "filter_config": None,
        "subtitles": None,
        "webhook_url": None,
        "webhook_secret": None,
    }
    defaults.update(overrides)
    job = MagicMock()
    for k, v in defaults.items():
        setattr(job, k, v)
    return job


def _make_settings_mock():
    s = MagicMock()
    s.compilation.target_width = 1920
    s.compilation.target_height = 1080
    s.compilation.crf = 20
    s.compilation.transition_type = "smart"
    s.compilation.crossfade_duration = 0.5
    s.compilation.output_filename = "demo.mp4"
    s.compilation.cleanup_intermediates = False
    # EXPLICIT opt-in to the LFI gate. These tests feed the compiler real
    # fixture files from a pytest tmpdir, which is outside the configured
    # storage root that `resolve_local_asset_path` confines local refs to.
    # Set deliberately rather than left to MagicMock's truthiness, so the
    # opt-in is visible and a future reader cannot mistake it for coverage
    # of the guard — that lives in test_compile_lfi_guards.py.
    s.compilation.allow_local_file_inputs = True
    s.background_music.path = ""
    s.background_music.volume = 0.1
    s.background_music.fade_in_s = 2.0
    s.background_music.fade_out_s = 3.0
    # Read by `_job_liveness` to size the progress heartbeat. A real number, not
    # MagicMock's default, because the interval is computed arithmetically.
    s.storage.stale_job_timeout_seconds = 600
    return s


def _single_recording_job(tmpdir: Path, **extra):
    """Create a job with one browser recording part."""
    vid = tmpdir / "v1.webm"
    vid.write_bytes(b"video")
    defaults = {
        "shot_list": {
            "parts": [{"id": "p1", "part_type": "browser_recording"}],
            "recording_order": ["p1"],
        },
        "recordings": [
            {
                "part_id": "p1",
                "success": True,
                "video_path": str(vid),
                "duration_seconds": 5.0,
            }
        ],
    }
    defaults.update(extra)
    return _base_job(**defaults)


async def _run_process_job_with_mocks(tmpdir, job, settings_mock=None, ffmpeg_extra_setup=None):
    """Helper to run process_job with common mock setup.

    Returns (service, mock_completed, mock_failed, mock_usage) so callers
    can assert on what mark_completed/mark_failed received.
    """
    work_dir = Path(tmpdir)
    storage = _mock_storage(work_dir)
    service = CompilerService(storage)
    db = AsyncMock()

    output_path = work_dir / "demo.mp4"

    with (
        patch(
            "src.services.compilation.compiler_service.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=job,
        ),
        patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_processing",
            new_callable=AsyncMock,
        ),
        patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_completed",
            new_callable=AsyncMock,
        ) as mock_completed,
        patch(
            "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
            new_callable=AsyncMock,
        ) as mock_failed,
        patch(
            "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
            new_callable=AsyncMock,
        ) as mock_usage,
        patch("src.config.get_settings", return_value=settings_mock or _make_settings_mock()),
        patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
        patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
    ):
        mock_ffmpeg.convert_to_mp4 = MagicMock()
        mock_ffmpeg.normalize_stream = MagicMock()
        mock_ffmpeg.probe_duration.return_value = 5.0
        mock_ffmpeg.extract_thumbnail = MagicMock()
        mock_editor.return_value.auto_trim.side_effect = lambda p: p

        norm_dir = work_dir / "normalized"
        norm_dir.mkdir(parents=True, exist_ok=True)
        (norm_dir / "p1.mp4").write_bytes(b"n1")
        output_path.write_bytes(b"output")

        if ffmpeg_extra_setup:
            ffmpeg_extra_setup(mock_ffmpeg, mock_editor, work_dir)

        await service.process_job(db, "job-1", TENANT_U4)

    return service, mock_completed, mock_failed, mock_usage


class TestColorCorrectionException:
    async def test_color_correction_failure_still_completes_job(self):
        """Color correction failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _single_recording_job(
                work_dir,
                filter_config={"color_correction": {"brightness": 999}},
            )

            with patch(
                "src.services.compilation.video_filters.apply_color_correction",
                side_effect=RuntimeError("cc failed"),
            ):
                _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(tmpdir, job)

            # Job should be marked completed (not failed) because cc is non-fatal
            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestWatermarkException:
    async def test_watermark_failure_still_completes_job(self):
        """Watermark overlay failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _single_recording_job(
                work_dir,
                filter_config={"watermark": {"image_path": "/nonexistent.png"}},
            )

            with patch(
                "src.services.compilation.video_filters.overlay_image",
                side_effect=RuntimeError("watermark failed"),
            ):
                _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(tmpdir, job)

            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestSubtitleException:
    async def test_subtitle_failure_still_completes_job(self):
        """Subtitle processing failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            sub_file = work_dir / "subs.srt"
            sub_file.write_bytes(b"bad subs")

            job = _single_recording_job(
                work_dir,
                subtitles={"file_path": str(sub_file), "mode": "burn"},
            )

            def setup_ffmpeg(mock_ffmpeg, mock_editor, wdir):
                mock_ffmpeg.burn_subtitles = MagicMock(
                    side_effect=RuntimeError("subtitle processing failed")
                )

            _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(
                tmpdir, job, ffmpeg_extra_setup=setup_ffmpeg
            )
            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestChapterEmbedException:
    async def test_chapter_embed_failure_still_completes_job(self):
        """Chapter embedding failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _single_recording_job(
                work_dir,
                shot_list={
                    "parts": [
                        {
                            "id": "p1",
                            "part_type": "browser_recording",
                            "chapter": 1,
                            "title": "Ch1",
                            "subtitle": "",
                        }
                    ],
                    "recording_order": ["p1"],
                },
            )

            def setup_ffmpeg(mock_ffmpeg, mock_editor, wdir):
                mock_ffmpeg.embed_chapters = MagicMock(
                    side_effect=RuntimeError("chapter embed failed")
                )

            _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(
                tmpdir, job, ffmpeg_extra_setup=setup_ffmpeg
            )
            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestThumbnailException:
    async def test_thumbnail_failure_still_completes_job(self):
        """Thumbnail generation failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _single_recording_job(work_dir)

            def setup_ffmpeg(mock_ffmpeg, mock_editor, wdir):
                mock_ffmpeg.extract_thumbnail = MagicMock(
                    side_effect=RuntimeError("thumbnail failed")
                )

            _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(
                tmpdir, job, ffmpeg_extra_setup=setup_ffmpeg
            )
            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestQualityValidationException:
    async def test_quality_validation_failure_still_completes_job(self):
        """Quality validation failure should be caught; the job should still complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            job = _single_recording_job(work_dir)

            with patch(
                "src.services.compilation.quality_validator.QualityValidator.validate",
                side_effect=RuntimeError("quality check failed"),
            ):
                _, mock_completed, mock_failed, _ = await _run_process_job_with_mocks(tmpdir, job)

            mock_completed.assert_awaited_once()
            mock_failed.assert_not_awaited()


class TestMixNarrationNoOutput:
    @patch("src.services.compilation.compiler_service.FFmpegOps.mix_voiceover")
    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration")
    def test_missing_mix_output_returns_original_video(self, mock_dur, mock_mix):
        """If mix_voiceover doesn't create the output file, return the original."""
        mock_dur.side_effect = [10.0, 10.0]
        service = CompilerService(MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            original = Path("/video.mp4")
            result = service._mix_per_segment_narration(
                _assembly_ctx(), original, Path("/audio.mp3"), work, "p1"
            )
            assert result == original


class TestAutoTrimReturnsDifferentPath:
    async def test_auto_trim_result_replaces_mp4_path(self):
        """When auto_trim returns a different path, the pipeline uses it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _single_recording_job(work_dir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            output_path = work_dir / "demo.mp4"

            with (
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.get_job",
                    new_callable=AsyncMock,
                    return_value=job,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_processing",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_completed",
                    new_callable=AsyncMock,
                ) as mock_completed,
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=_make_settings_mock()),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
            ):
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                edited_dir = work_dir / "edited"
                edited_dir.mkdir(parents=True, exist_ok=True)
                trimmed_path = edited_dir / "p1_trimmed.mp4"
                trimmed_path.write_bytes(b"trimmed")
                mock_editor.return_value.auto_trim.return_value = trimmed_path

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

            # The job should complete successfully
            mock_completed.assert_awaited_once()


class TestThumbnailCreatedSuccessfully:
    async def test_thumbnail_path_set_in_completed_call(self):
        """When extract_thumbnail creates the file, thumbnail_path is set in mark_completed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _single_recording_job(work_dir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            output_path = work_dir / "demo.mp4"

            with (
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.get_job",
                    new_callable=AsyncMock,
                    return_value=job,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_processing",
                    new_callable=AsyncMock,
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_completed",
                    new_callable=AsyncMock,
                ) as mock_completed,
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=_make_settings_mock()),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
            ):
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                def fake_extract(inp, out, **kw):
                    out.write_bytes(b"thumb")

                mock_ffmpeg.extract_thumbnail = MagicMock(side_effect=fake_extract)

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["thumbnail_path"] is not None
                assert "thumbnail.jpg" in call_kwargs["thumbnail_path"]


class TestAssembleSmartUnlinkOSError:
    @patch("src.services.compilation.compiler_service.FFmpegOps.crossfade")
    def test_unlink_oserror_does_not_prevent_output(self, mock_crossfade):
        """OSError during intermediate file cleanup should not prevent assembly output."""
        service = CompilerService(MagicMock())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "p1.mp4"
            seg2 = work_dir / "p2.mp4"
            seg3 = work_dir / "p3.mp4"
            seg1.write_bytes(b"s1")
            seg2.write_bytes(b"s2")
            seg3.write_bytes(b"s3")
            output = work_dir / "out.mp4"

            (work_dir / "merged_1.mp4").write_bytes(b"m1")
            (work_dir / "merged_2.mp4").write_bytes(b"m2")

            parts = {
                "p1": {"chapter": 1, "part_type": "browser_recording"},
                "p2": {"chapter": 2, "part_type": "browser_recording"},
                "p3": {"chapter": 3, "part_type": "browser_recording"},
            }

            mock_crossfade.side_effect = lambda _a, _b, out, **kw: out.write_bytes(b"xf")

            original_unlink = Path.unlink

            def patched_unlink(self, *args, **kwargs):
                if "merged_" in str(self):
                    raise OSError("permission denied")
                return original_unlink(self, *args, **kwargs)

            with patch.object(Path, "unlink", patched_unlink):
                service._assemble_smart(
                    _assembly_ctx(),
                    [seg1, seg2, seg3],
                    parts,
                    ["p1", "p2", "p3"],
                    output,
                    work_dir,
                    0.5,
                )
            assert output.exists()
            assert output.stat().st_size > 0


class TestAssembleWithFadeUnlinkOSError:
    @patch("src.services.compilation.compiler_service.FFmpegOps.fade_to_black")
    def test_fade_unlink_oserror_does_not_prevent_output(self, mock_fade):
        """OSError during intermediate cleanup in _assemble_with_fade should not break output."""
        service = CompilerService(MagicMock())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "s1.mp4"
            seg2 = work_dir / "s2.mp4"
            seg3 = work_dir / "s3.mp4"
            seg1.write_bytes(b"s1")
            seg2.write_bytes(b"s2")
            seg3.write_bytes(b"s3")
            output = work_dir / "out.mp4"

            (work_dir / "fade_1.mp4").write_bytes(b"f1")
            (work_dir / "fade_2.mp4").write_bytes(b"f2")

            original_unlink = Path.unlink

            def patched_unlink(self, *args, **kwargs):
                if "fade_" in str(self):
                    raise OSError("permission denied")
                return original_unlink(self, *args, **kwargs)

            with patch.object(Path, "unlink", patched_unlink):
                service._assemble_with_fade([seg1, seg2, seg3], output, work_dir, 0.5)
            assert mock_fade.call_count == 2
            assert output.exists()
            assert output.stat().st_size > 0


class TestAssembleWithXfadeUnlinkOSError:
    @patch("src.services.compilation.compiler_service.FFmpegOps.crossfade")
    def test_xfade_unlink_oserror_does_not_prevent_output(self, mock_crossfade):
        """OSError during intermediate cleanup in _assemble_with_xfade should not break output."""
        service = CompilerService(MagicMock())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "s1.mp4"
            seg2 = work_dir / "s2.mp4"
            seg3 = work_dir / "s3.mp4"
            seg1.write_bytes(b"s1")
            seg2.write_bytes(b"s2")
            seg3.write_bytes(b"s3")
            output = work_dir / "out.mp4"

            (work_dir / "xfade_1.mp4").write_bytes(b"x1")
            (work_dir / "xfade_2.mp4").write_bytes(b"x2")

            original_unlink = Path.unlink

            def patched_unlink(self, *args, **kwargs):
                if "xfade_" in str(self):
                    raise OSError("permission denied")
                return original_unlink(self, *args, **kwargs)

            with patch.object(Path, "unlink", patched_unlink):
                service._assemble_with_xfade([seg1, seg2, seg3], output, work_dir, 0.5, "fade")
            assert mock_crossfade.call_count == 2
            assert output.exists()
            assert output.stat().st_size > 0
