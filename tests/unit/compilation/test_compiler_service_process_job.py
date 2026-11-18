"""Comprehensive tests for CompilerService.process_job and assembly helpers.

Covers the core compilation loop (lines 225-651) including title cards,
browser recordings with edit decisions, AI clips, assembly methods, color
correction, watermark, narration, background music, subtitles, chapter
embedding, thumbnail generation, quality validation, cleanup, webhooks,
and error handling.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.exceptions import ArtifactNotStoredError
from src.services.compilation.compiler_service import CompilerService, _JobContext
from tests.conftest import TENANT_U1, TENANT_U4


def _mixctx() -> _JobContext:
    """Context for _mix_per_segment_narration, used only to record a degradation
    when the mix fails and the un-narrated segment is returned."""
    return _JobContext(job_id="job-1", tenant_id=TENANT_U4, job=None, settings=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Create a mock CompilationJob with all attributes needed by process_job."""
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
    """Return a settings mock with all attributes process_job reads."""
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


# ---------------------------------------------------------------------------
# process_job: title card segment
# ---------------------------------------------------------------------------


class TestProcessJobTitleCard:
    async def test_title_card_segment_processed(self):
        """Title card parts generate a video and normalize it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                shot_list={
                    "parts": [
                        {
                            "id": "tc1",
                            "part_type": "title_card",
                            "title": "Intro",
                            "subtitle": "Welcome",
                            "chapter": 1,
                            "estimated_duration_seconds": 3.0,
                        }
                    ],
                    "recording_order": ["tc1"],
                },
            )

            # The output file must exist so the downstream stat() works
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
                patch(
                    "src.services.compilation.compiler_service.TitleCardGenerator"
                ) as mock_title_gen,
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
            ):
                mock_title_gen.return_value.generate_video = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 3.0

                # Create the normalized file so shutil.copy2 works
                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "tc1.mp4").write_bytes(b"normalized")

                # Single segment => shutil.copy2
                output_path.write_bytes(b"output")

                # Mock stat and exists on the output
                mock_ffmpeg.extract_thumbnail = MagicMock()

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()
                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["total_parts"] == 1

    async def test_narration_url_download_failure_degrades(self):
        """A narration-URL download raising httpx.HTTPStatusError (which is NOT
        an OSError/RuntimeError/ValueError/TimeoutError) must be caught so the
        compile degrades to no-narration and still COMPLETES, rather than the
        error escaping _initialize + process_job and hard-failing the job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                narration_audio_dir="https://cdn.test/narration",
                shot_list={
                    "parts": [
                        {
                            "id": "tc1",
                            "part_type": "title_card",
                            "title": "Intro",
                            "subtitle": "Welcome",
                            "chapter": 1,
                            "estimated_duration_seconds": 3.0,
                        }
                    ],
                    "recording_order": ["tc1"],
                },
            )

            output_path = work_dir / "demo.mp4"

            # The narration fetch now goes through ``w2v_common.net.safe_get``
            # (SSRF-gated, IP-pinned, per-hop revalidated) rather than a bare
            # ``client.get``. Patch that seam directly so this test keeps
            # asserting the thing it was written for -- an ``HTTPStatusError``
            # from the download degrading instead of hard-failing the job --
            # without depending on DNS or on the guard's internals. The guard's
            # own reject-path is covered separately below.
            failing_safe_get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "500 Server Error",
                    request=httpx.Request("GET", "https://cdn.test/narration"),
                    response=httpx.Response(500),
                )
            )
            failing_client = AsyncMock()
            failing_client.__aenter__ = AsyncMock(return_value=failing_client)
            failing_client.__aexit__ = AsyncMock(return_value=False)

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
                patch("httpx.AsyncClient", return_value=failing_client),
                patch("w2v_common.net.safe_get", failing_safe_get),
                patch(
                    "src.services.compilation.compiler_service.TitleCardGenerator"
                ) as mock_title_gen,
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
            ):
                mock_title_gen.return_value.generate_video = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 3.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "tc1.mp4").write_bytes(b"normalized")
                output_path.write_bytes(b"output")

                # Must NOT raise -- the httpx failure degrades gracefully.
                await service.process_job(db, "job-1", TENANT_U4)

                # Narration download was attempted and failed...
                failing_safe_get.assert_awaited_once()
                # ...but the job still completed WITHOUT narration.
                mock_completed.assert_awaited_once()

    async def test_narration_url_ssrf_block_degrades(self):
        """A narration URL pointing into the estate must be refused, and the
        compile must still COMPLETE without narration.

        ``narration_audio_dir`` is tenant-supplied and was fetched with no guard
        at all. RFC 6598 ``100.64.0.0/10`` is this fleet's tailnet and
        ``ipaddress.is_private`` is False for it, so ``100.64.0.5`` (the
        bare-metal prod Postgres primary) was a reachable target. ``safe_get``
        now raises ``SSRFError`` -- a ``ValueError`` -- which the existing
        except tuple absorbs, so the block degrades rather than hard-failing the
        job.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                narration_audio_dir="http://100.64.0.5:5432/narration",
                shot_list={
                    "parts": [
                        {
                            "id": "tc1",
                            "part_type": "title_card",
                            "title": "Intro",
                            "subtitle": "Welcome",
                            "chapter": 1,
                            "estimated_duration_seconds": 3.0,
                        }
                    ],
                    "recording_order": ["tc1"],
                },
            )

            output_path = work_dir / "demo.mp4"

            # A real client whose transport would succeed if it were ever
            # reached -- so a passing test proves the guard refused, not that
            # the request happened to fail.
            reachable_client = AsyncMock()
            reachable_client.__aenter__ = AsyncMock(return_value=reachable_client)
            reachable_client.__aexit__ = AsyncMock(return_value=False)

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
                patch("httpx.AsyncClient", return_value=reachable_client),
                patch(
                    "src.services.compilation.compiler_service.TitleCardGenerator"
                ) as mock_title_gen,
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
            ):
                mock_title_gen.return_value.generate_video = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 3.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "tc1.mp4").write_bytes(b"normalized")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                # The tailnet host was never dialled...
                reachable_client.stream.assert_not_awaited()
                reachable_client.get.assert_not_awaited()
                # ...and the job still completed.
                mock_completed.assert_awaited_once()

    def test_title_card_with_slide_image_uses_motion_video(self):
        """Title cards with slide_image_url bypass TitleCardGenerator and render from the image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)

            ctx = _JobContext(
                job_id="job-1",
                tenant_id=TENANT_U4,
                job=_base_job(),
                settings=_make_settings_mock(),
                work_dir=work_dir,
                edited_dir=work_dir / "edited",
                title_cards_dir=work_dir / "title_cards",
                normalized_dir=work_dir / "normalized",
                title_gen=MagicMock(),
            )
            ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
            ctx.normalized_dir.mkdir(parents=True, exist_ok=True)

            part = {
                "id": "slide-1",
                "part_type": "title_card",
                "slide_image_url": "data:image/png;base64,aGVsbG8=",
                "motion_preset": "zoom_in",
                "estimated_duration_seconds": 4.0,
            }

            with patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg:
                mock_ffmpeg.image_to_motion_video = MagicMock()
                mock_ffmpeg.image_to_video = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()

                service._process_title_card(ctx, "slide-1", part)

                ctx.title_gen.generate_video.assert_not_called()
                mock_ffmpeg.image_to_motion_video.assert_called_once()
                mock_ffmpeg.image_to_video.assert_not_called()
                assert ctx.segments == [work_dir / "normalized" / "slide-1.mp4"]

    @pytest.mark.asyncio
    async def test_process_segments_skips_rejected_generated_image(self):
        """Rejected AI image segments are excluded before any rendering work starts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)

            ctx = _JobContext(
                job_id="job-1",
                tenant_id=TENANT_U4,
                job=_base_job(),
                settings=_make_settings_mock(),
                work_dir=work_dir,
                edited_dir=work_dir / "edited",
                title_cards_dir=work_dir / "title_cards",
                normalized_dir=work_dir / "normalized",
                parts_by_id={
                    "asset-1": {
                        "id": "asset-1",
                        "part_type": "generated_image",
                        "image_url": "https://cdn.example/asset.png",
                        "approval_status": "rejected",
                    }
                },
                recording_order=["asset-1"],
                title_gen=MagicMock(),
            )
            ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
            ctx.normalized_dir.mkdir(parents=True, exist_ok=True)

            await service._process_segments(ctx, None)

            assert ctx.segments == []
            assert ctx.segments_excluded == 1


# ---------------------------------------------------------------------------
# process_job: browser recording with edit decisions
# ---------------------------------------------------------------------------


class TestProcessJobBrowserRecording:
    async def test_browser_recording_with_edit_decisions(self):
        """Browser recording parts apply speed, trim, dead time, auto-trim, normalize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            # Create a fake video file
            video_file = work_dir / "recording.webm"
            video_file.write_bytes(b"fake-video")

            job = _base_job(
                shot_list={
                    "parts": [
                        {
                            "id": "part1",
                            "part_type": "browser_recording",
                        }
                    ],
                    "recording_order": ["part1"],
                },
                recordings=[
                    {
                        "part_id": "part1",
                        "success": True,
                        "video_path": str(video_file),
                        "duration_seconds": 30.0,
                    }
                ],
                edit_decisions=[
                    {
                        "part_id": "part1",
                        "speed_factor": 1.5,
                        "trim_start_s": 2.0,
                        "trim_end_s": 3.0,
                        "remove_dead_time": True,
                    }
                ],
            )

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
                # Setup mocks
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 10.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                editor_instance = mock_editor.return_value

                # Speed adjust returns a new path
                speed_path = work_dir / "edited" / "part1_speed.mp4"
                speed_path.parent.mkdir(parents=True, exist_ok=True)
                speed_path.write_bytes(b"speed")
                editor_instance.speed_adjust.return_value = speed_path

                # Trim path
                trim_path = work_dir / "edited" / "part1_trim.mp4"
                trim_path.write_bytes(b"trimmed")
                mock_ffmpeg.trim = MagicMock()

                # dead time removal returns same path
                editor_instance.remove_dead_time.return_value = speed_path

                # Auto-trim returns same path (no change)
                editor_instance.auto_trim.return_value = speed_path

                # Create normalized output
                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "part1.mp4").write_bytes(b"normalized")

                # Output file
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()
                editor_instance.speed_adjust.assert_called_once()
                editor_instance.remove_dead_time.assert_called_once()
                editor_instance.auto_trim.assert_called_once()

    async def test_recording_skipped_when_video_missing(self):
        """If video_path doesn't exist, the part is skipped and job fails (no segments)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                shot_list={
                    "parts": [{"id": "part1", "part_type": "browser_recording"}],
                    "recording_order": ["part1"],
                },
                recordings=[
                    {
                        "part_id": "part1",
                        "success": True,
                        "video_path": "/nonexistent/path.webm",
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
                patch("src.config.get_settings", return_value=_make_settings_mock()),
            ):
                await service.process_job(db, "job-1", TENANT_U4)
                mock_failed.assert_awaited_once()
                assert "No segments to compile" in mock_failed.call_args[0][2]
                mock_usage.assert_awaited_once()
                assert mock_usage.call_args[1]["is_error"] is True

    async def test_recording_skipped_when_not_success(self):
        """Recordings with success=False are excluded from recording_map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                shot_list={
                    "parts": [{"id": "part1", "part_type": "browser_recording"}],
                    "recording_order": ["part1"],
                },
                recordings=[
                    {
                        "part_id": "part1",
                        "success": False,
                        "video_path": "/some/path.webm",
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
                patch("src.config.get_settings", return_value=_make_settings_mock()),
            ):
                await service.process_job(db, "job-1", TENANT_U4)
                mock_failed.assert_awaited_once()
                assert "No segments to compile" in mock_failed.call_args[0][2]
                mock_usage.assert_awaited_once()
                assert mock_usage.call_args[1]["is_error"] is True

    async def test_part_not_found_skipped(self):
        """Parts in recording_order not in parts list are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                shot_list={
                    "parts": [],
                    "recording_order": ["missing-part"],
                },
            )

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
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                    new_callable=AsyncMock,
                ) as mock_failed,
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ) as mock_usage,
                patch("src.config.get_settings", return_value=_make_settings_mock()),
            ):
                await service.process_job(db, "job-1", TENANT_U4)
                mock_failed.assert_awaited_once()
                assert "No segments to compile" in mock_failed.call_args[0][2]
                mock_usage.assert_awaited_once()
                assert mock_usage.call_args[1]["is_error"] is True


# ---------------------------------------------------------------------------
# process_job: AI-generated clips
# ---------------------------------------------------------------------------


class TestProcessJobAIClip:
    async def test_ai_clip_processed_with_strip_audio(self):
        """AI clips are processed: strip audio, normalize, and add to segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            clips_dir = work_dir / "clips"
            clips_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            ai_clip = clips_dir / "ai1.mp4"
            ai_clip.write_bytes(b"ai-clip")

            job = _base_job(
                clips_dir=str(clips_dir),
                shot_list={
                    "parts": [
                        {
                            "id": "ai1",
                            "part_type": "ai_generated",
                            "strip_ai_audio": True,
                        }
                    ],
                    "recording_order": ["ai1"],
                },
            )

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
            ):
                # strip_audio creates the stripped file
                stripped_path = work_dir / "edited" / "ai1_noaudio.mp4"
                stripped_path.parent.mkdir(parents=True, exist_ok=True)
                stripped_path.write_bytes(b"stripped")

                def fake_strip(inp, out):
                    out.write_bytes(b"stripped")

                mock_ffmpeg.strip_audio = MagicMock(side_effect=fake_strip)
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                # Create normalized output
                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "ai1.mp4").write_bytes(b"normalized")

                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()
                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["total_parts"] == 1
                mock_ffmpeg.strip_audio.assert_called_once()

    async def test_ai_clip_missing_skipped(self):
        """If AI clip file doesn't exist, the part is skipped and job fails (no segments)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            clips_dir = work_dir / "clips"
            clips_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            job = _base_job(
                clips_dir=str(clips_dir),
                shot_list={
                    "parts": [{"id": "ai1", "part_type": "ai_generated"}],
                    "recording_order": ["ai1"],
                },
            )

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
                    "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                    new_callable=AsyncMock,
                ) as mock_failed,
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ) as mock_usage,
                patch("src.config.get_settings", return_value=_make_settings_mock()),
            ):
                await service.process_job(db, "job-1", TENANT_U4)
                mock_failed.assert_awaited_once()
                assert "No segments to compile" in mock_failed.call_args[0][2]
                mock_usage.assert_awaited_once()
                assert mock_usage.call_args[1]["is_error"] is True

    async def test_ai_clip_strip_audio_failure_graceful(self):
        """If strip_audio fails, we continue with the original clip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            clips_dir = work_dir / "clips"
            clips_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            ai_clip = clips_dir / "ai1.mp4"
            ai_clip.write_bytes(b"ai-clip")

            job = _base_job(
                clips_dir=str(clips_dir),
                shot_list={
                    "parts": [
                        {
                            "id": "ai1",
                            "part_type": "ai_generated",
                            "strip_ai_audio": True,
                        }
                    ],
                    "recording_order": ["ai1"],
                },
            )

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
            ):
                mock_ffmpeg.strip_audio = MagicMock(side_effect=RuntimeError("strip failed"))
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "ai1.mp4").write_bytes(b"normalized")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()
                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["total_parts"] == 1

    async def test_ai_clip_with_per_segment_narration(self):
        """AI clips can have per-segment narration mixed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            clips_dir = work_dir / "clips"
            clips_dir.mkdir()
            narr_dir = work_dir / "narration"
            narr_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            ai_clip = clips_dir / "ai1.mp4"
            ai_clip.write_bytes(b"ai-clip")
            narr_audio = narr_dir / "ai1.mp3"
            narr_audio.write_bytes(b"narration")

            job = _base_job(
                clips_dir=str(clips_dir),
                narration_audio_dir=str(narr_dir),
                shot_list={
                    "parts": [
                        {
                            "id": "ai1",
                            "part_type": "ai_generated",
                            "strip_ai_audio": False,
                        }
                    ],
                    "recording_order": ["ai1"],
                },
            )

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
            ):
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.mix_voiceover = MagicMock()
                mock_ffmpeg.speed_adjust = MagicMock()
                mock_ffmpeg.extract_thumbnail = MagicMock()

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "ai1.mp4").write_bytes(b"normalized")
                # Narrated output
                (norm_dir / "ai1_narrated.mp4").write_bytes(b"narrated")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()


# ---------------------------------------------------------------------------
# process_job: assembly transitions
# ---------------------------------------------------------------------------


class TestProcessJobAssembly:
    async def _run_assembly_test(self, transition_type: str):
        """Helper to run assembly with two browser recording segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid1 = work_dir / "v1.webm"
            vid2 = work_dir / "v2.webm"
            vid1.write_bytes(b"video1")
            vid2.write_bytes(b"video2")

            job = _base_job(
                transition_type=transition_type,
                shot_list={
                    "parts": [
                        {"id": "p1", "part_type": "browser_recording", "chapter": 1},
                        {"id": "p2", "part_type": "browser_recording", "chapter": 2},
                    ],
                    "recording_order": ["p1", "p2"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid1),
                        "duration_seconds": 10.0,
                    },
                    {
                        "part_id": "p2",
                        "success": True,
                        "video_path": str(vid2),
                        "duration_seconds": 10.0,
                    },
                ],
            )

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
                mock_ffmpeg.probe_duration.return_value = 10.0
                mock_ffmpeg.concat = MagicMock()
                mock_ffmpeg.crossfade = MagicMock()
                mock_ffmpeg.extract_thumbnail = MagicMock()

                editor_inst = mock_editor.return_value
                editor_inst.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                (norm_dir / "p2.mp4").write_bytes(b"n2")

                # Create edited dir for concat/crossfade temp files
                edited_dir = work_dir / "edited"
                edited_dir.mkdir(parents=True, exist_ok=True)

                # For 'cut' transition, concat is called
                # For 'fade' transition, crossfade is called
                # For 'smart' (default), _assemble_smart is called

                if transition_type == "cut":
                    # concat writes the output
                    def fake_concat(concat_file, out_path, **kwargs):
                        out_path.write_bytes(b"concatenated")

                    mock_ffmpeg.concat = MagicMock(side_effect=fake_concat)
                elif transition_type == "fade":

                    def fake_crossfade(
                        a, b, out, fade_duration=0.5, transition_type="fade", **kwargs
                    ):
                        out.write_bytes(b"crossfaded")

                    mock_ffmpeg.crossfade = MagicMock(side_effect=fake_crossfade)
                else:
                    # smart: crossfade for different chapters
                    def fake_crossfade(
                        a, b, out, fade_duration=0.5, transition_type="fade", **kwargs
                    ):
                        out.write_bytes(b"crossfaded")

                    mock_ffmpeg.crossfade = MagicMock(side_effect=fake_crossfade)

                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_completed.assert_awaited_once()
                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["total_parts"] == 2

    async def test_cut_transition(self):
        await self._run_assembly_test("cut")

    async def test_fade_transition(self):
        await self._run_assembly_test("fade")

    async def test_smart_transition(self):
        await self._run_assembly_test("smart")


# ---------------------------------------------------------------------------
# process_job: color correction + watermark + subtitles
# ---------------------------------------------------------------------------


class TestProcessJobFilters:
    async def test_color_correction_applied(self):
        """When color_correction is in filter_config, it is applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
                filter_config={"color_correction": {"brightness": 0.1, "contrast": 1.2}},
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
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=_make_settings_mock()),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
                patch("src.services.compilation.video_filters.apply_color_correction") as mock_cc,
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

                def fake_cc(inp, out, correction):
                    out.write_bytes(b"color-corrected")

                mock_cc.side_effect = fake_cc

                await service.process_job(db, "job-1", TENANT_U4)

                # The tenant's numbers must survive the trip into ColorCorrection…
                mock_cc.assert_called_once()
                cc_in, cc_out, correction = mock_cc.call_args.args
                assert Path(cc_in) == output_path
                assert (correction.brightness, correction.contrast) == (0.1, 1.2)
                # …and unspecified knobs must stay neutral, not zeroed.
                assert (correction.saturation, correction.gamma, correction.temperature) == (
                    1.0,
                    1.0,
                    0.0,
                )
                # …and the corrected file must REPLACE the delivered output. Producing
                # demo_cc.mp4 and shipping the uncorrected demo.mp4 keeps this mock
                # "called once" while the tenant gets the wrong video.
                assert output_path.read_bytes() == b"color-corrected"
                assert not cc_out.exists()
                assert not output_path.with_suffix(".tmp").exists()

    async def test_watermark_applied(self):
        """When watermark is in filter_config, overlay_image is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            wm_image = work_dir / "logo.png"
            wm_image.write_bytes(b"logo")

            job = _base_job(
                filter_config={"watermark": {"image_path": str(wm_image)}},
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
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=_make_settings_mock()),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
                patch("src.services.compilation.video_filters.overlay_image") as mock_overlay,
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

                def fake_overlay(inp, overlay, out, **kw):
                    out.write_bytes(b"watermarked")

                mock_overlay.side_effect = fake_overlay

                await service.process_job(db, "job-1", TENANT_U4)

                mock_overlay.assert_called_once()
                ov_in, ov_image, ov_out = mock_overlay.call_args.args
                assert Path(ov_in) == output_path
                # The configured image is what gets composited — not some other asset.
                assert Path(ov_image) == wm_image
                # Placement defaults are part of the contract: silently changing them
                # moves every existing tenant's logo.
                assert mock_overlay.call_args.kwargs == {
                    "position": "bottom-right",
                    "opacity": 0.7,
                    "scale": 0.15,
                }
                # The watermarked file must REPLACE the delivered output.
                assert output_path.read_bytes() == b"watermarked"
                assert not ov_out.exists()
                assert not output_path.with_suffix(".tmp").exists()

    async def test_subtitles_burn_mode(self):
        """Subtitles in burn mode call FFmpegOps.burn_subtitles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            sub_file = work_dir / "subs.srt"
            sub_file.write_bytes(b"1\n00:00:00,000 --> 00:00:05,000\nHello\n")

            job = _base_job(
                subtitles={"file_path": str(sub_file), "mode": "burn", "font_size": 30},
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
                ),
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

                def fake_burn(inp, sub, out, **kw):
                    out.write_bytes(b"subtitled")

                mock_ffmpeg.burn_subtitles = MagicMock(side_effect=fake_burn)
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                # mode="burn" must pick burn, never embed: an embed on a burn request
                # ships a video whose captions are invisible in most players.
                mock_ffmpeg.burn_subtitles.assert_called_once()
                mock_ffmpeg.embed_subtitles.assert_not_called()
                burn_in, burn_sub, _burn_out = mock_ffmpeg.burn_subtitles.call_args.args
                assert Path(burn_in) == output_path
                assert Path(burn_sub) == sub_file
                # The tenant set font_size=30; the other two stay on their defaults.
                assert mock_ffmpeg.burn_subtitles.call_args.kwargs == {
                    "font_size": 30,
                    "font_color": "white",
                    "position": "bottom",
                }
                # The subtitled file must REPLACE the delivered output.
                assert output_path.read_bytes() == b"subtitled"

    async def test_subtitles_embed_mode(self):
        """Subtitles in embed mode call FFmpegOps.embed_subtitles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            sub_file = work_dir / "subs.srt"
            sub_file.write_bytes(b"1\n00:00:00,000 --> 00:00:05,000\nHello\n")

            job = _base_job(
                subtitles={"file_path": str(sub_file), "mode": "embed"},
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
                ),
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

                def fake_embed(inp, sub, out):
                    out.write_bytes(b"embedded-subs")

                mock_ffmpeg.embed_subtitles = MagicMock(side_effect=fake_embed)
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                # mode="embed" must pick embed, never burn: a burn is irreversible —
                # the caption is baked into the pixels of the delivered video.
                mock_ffmpeg.embed_subtitles.assert_called_once()
                mock_ffmpeg.burn_subtitles.assert_not_called()
                emb_in, emb_sub, _emb_out = mock_ffmpeg.embed_subtitles.call_args.args
                assert Path(emb_in) == output_path
                assert Path(emb_sub) == sub_file
                assert output_path.read_bytes() == b"embedded-subs"


# ---------------------------------------------------------------------------
# process_job: narration and background music
# ---------------------------------------------------------------------------


class TestProcessJobNarration:
    async def test_global_narration_mixed(self):
        """Global narration (narration_full.mp3) triggers speed-adjust and mix_voiceover."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            narr_dir = work_dir / "narration"
            narr_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")
            narr_full = narr_dir / "narration_full.mp3"
            narr_full.write_bytes(b"narration")

            job = _base_job(
                narration_audio_dir=str(narr_dir),
                shot_list={
                    "parts": [{"id": "p1", "part_type": "browser_recording"}],
                    "recording_order": ["p1"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid),
                        "duration_seconds": 10.0,
                    }
                ],
            )

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
                ),
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
                # probe_duration: narration=8s, video=10s -> 10 > 8*1.05=8.4 => speed adjust
                mock_ffmpeg.probe_duration.side_effect = lambda p: (
                    8.0 if "narration" in str(p) else 10.0
                )
                mock_ffmpeg.extract_thumbnail = MagicMock()
                mock_ffmpeg.speed_adjust = MagicMock()
                mock_ffmpeg.mix_voiceover = MagicMock()

                def fake_speed(inp, out, factor):
                    out.write_bytes(b"sped")

                mock_ffmpeg.speed_adjust = MagicMock(side_effect=fake_speed)

                def fake_mix(inp, audio, out, **kw):
                    out.write_bytes(b"mixed")

                mock_ffmpeg.mix_voiceover = MagicMock(side_effect=fake_mix)
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

    async def test_background_music_added(self):
        """When bg music is configured and no narration, mix_background_music is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            music_file = work_dir / "music.mp3"
            music_file.write_bytes(b"music")

            job = _base_job(
                shot_list={
                    "parts": [{"id": "p1", "part_type": "browser_recording"}],
                    "recording_order": ["p1"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid),
                        "duration_seconds": 10.0,
                    }
                ],
            )

            output_path = work_dir / "demo.mp4"

            settings_mock = _make_settings_mock()
            settings_mock.background_music.path = str(music_file)

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
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=settings_mock),
                patch("src.services.compilation.compiler_service.FFmpegOps") as mock_ffmpeg,
                patch("src.services.compilation.compiler_service.VideoEditor") as mock_editor,
            ):
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 10.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                def fake_mix_bg(inp, music, out, **kw):
                    out.write_bytes(b"with-music")

                mock_ffmpeg.mix_background_music = MagicMock(side_effect=fake_mix_bg)
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_ffmpeg.mix_background_music.assert_called_once()

    async def test_per_segment_narration_for_recording(self):
        """Per-segment narration mixes audio for each browser recording part."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            narr_dir = work_dir / "narration"
            narr_dir.mkdir()
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")
            narr_audio = narr_dir / "p1.mp3"
            narr_audio.write_bytes(b"narration")

            job = _base_job(
                narration_audio_dir=str(narr_dir),
                shot_list={
                    "parts": [{"id": "p1", "part_type": "browser_recording"}],
                    "recording_order": ["p1"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid),
                        "duration_seconds": 10.0,
                    }
                ],
            )

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
                ),
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
                # Audio=5s, Video=10s -> speed adjust needed
                mock_ffmpeg.probe_duration.side_effect = lambda p: (
                    5.0 if "narration" in str(p) or ".mp3" in str(p) else 10.0
                )
                mock_ffmpeg.mix_voiceover = MagicMock()
                mock_ffmpeg.speed_adjust = MagicMock()
                mock_ffmpeg.extract_thumbnail = MagicMock()
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                (norm_dir / "p1_narrated.mp4").write_bytes(b"narrated")
                (norm_dir / "p1_sped.mp4").write_bytes(b"sped")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)


# ---------------------------------------------------------------------------
# process_job: error handling
# ---------------------------------------------------------------------------


class TestProcessJobErrorHandling:
    async def test_exception_marks_job_failed(self):
        """When process_job raises inside the try block, it marks the job as failed."""
        storage = MagicMock()
        storage.get_job_dir.side_effect = RuntimeError("storage error")
        service = CompilerService(storage)
        db = AsyncMock()

        job = _base_job()

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
                "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                new_callable=AsyncMock,
            ),
            patch("src.config.get_settings", return_value=_make_settings_mock()),
        ):
            with pytest.raises(RuntimeError, match="storage error"):
                await service.process_job(db, "job-1", TENANT_U4)

            mock_failed.assert_awaited_once()

    async def test_exception_with_progress_callback(self):
        """Progress callback.error() is called on failure."""
        storage = MagicMock()
        storage.get_job_dir.side_effect = RuntimeError("boom")
        service = CompilerService(storage)
        db = AsyncMock()

        job = _base_job()
        callback = AsyncMock()

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
                "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                new_callable=AsyncMock,
            ),
            patch("src.config.get_settings", return_value=_make_settings_mock()),
        ):
            with pytest.raises(RuntimeError):
                await service.process_job(db, "job-1", TENANT_U4, progress_callback=callback)

            callback.error.assert_awaited_once()

    async def test_record_usage_failure_does_not_mask_real_error(self):
        """A record_usage() blowup inside the except-handler must not replace
        the real stage exception.

        Regression test: ``_handle_process_job_exception`` used to call
        ``CompilationJobManager.record_usage`` unguarded from inside the
        ``except`` block in ``process_job``. If ``record_usage`` itself
        raised, that new exception silently replaced the real ffmpeg/stage
        error on the bare ``raise`` that follows -- every failed compilation
        job's actual cause was erased from the propagated exception (though
        the real cause was still logged via ``logger.error(..., exc_info=True)``
        a few lines above).
        """
        storage = MagicMock()
        storage.get_job_dir.side_effect = RuntimeError("real ffmpeg failure")
        service = CompilerService(storage)
        db = AsyncMock()

        job = _base_job()

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
                "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                new_callable=AsyncMock,
            ) as mock_failed,
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                new_callable=AsyncMock,
                side_effect=ConnectionError("usage-db pool exhausted"),
            ),
            patch("src.config.get_settings", return_value=_make_settings_mock()),
        ):
            with pytest.raises(RuntimeError, match="real ffmpeg failure") as exc_info:
                await service.process_job(db, "job-1", TENANT_U4)

            # The original stage error propagated, not the ConnectionError
            # from the usage-counter write.
            assert isinstance(exc_info.value, RuntimeError)
            assert not isinstance(exc_info.value, ConnectionError)

            # Second-order: mark_failed still ran and the session was still
            # committed despite record_usage blowing up.
            mock_failed.assert_awaited_once()
            db.commit.assert_awaited()


# ---------------------------------------------------------------------------
# process_job: progress callback
# ---------------------------------------------------------------------------


class TestProcessJobProgressCallback:
    async def test_progress_callback_called(self):
        """Progress callback is called at each stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
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

            output_path = work_dir / "demo.mp4"
            callback = AsyncMock()

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
                ),
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
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4, progress_callback=callback)

                # report called for: initializing, processing, assembling, mixing_audio
                assert callback.report.await_count >= 3
                callback.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# process_job: webhook firing
# ---------------------------------------------------------------------------


class TestProcessJobWebhook:
    async def test_webhook_fired_on_completion(self):
        """When webhook_url is set, a task is created to fire the webhook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
                webhook_url="https://example.com/hook",
                webhook_secret="secret123",  # pragma: allowlist secret
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
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)
                await asyncio.sleep(0)  # let the fire-and-forget webhook task run

                # Assert the EVENT, not merely that a task was created. The old
                # form (`create_task.assert_called_once()`) could not have caught
                # a job announcing "completed" for work it never stored.
                mock_hook.assert_awaited_once()
                call = mock_hook.await_args
                assert call is not None
                assert call.args[3] == "completed"


# ---------------------------------------------------------------------------
# process_job: cleanup intermediates
# ---------------------------------------------------------------------------


class TestProcessJobCleanup:
    async def test_cleanup_intermediates(self):
        """When cleanup_intermediates is True, temp dirs are removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
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

            output_path = work_dir / "demo.mp4"

            settings_mock = _make_settings_mock()
            settings_mock.compilation.cleanup_intermediates = True

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
                ),
                patch(
                    "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                    new_callable=AsyncMock,
                ),
                patch("src.config.get_settings", return_value=settings_mock),
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

                edited_dir = work_dir / "edited"
                edited_dir.mkdir(parents=True, exist_ok=True)
                (edited_dir / "temp.mp4").write_bytes(b"temp")

                title_dir = work_dir / "title_cards"
                title_dir.mkdir(parents=True, exist_ok=True)

                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                # After cleanup, the intermediate dirs should be removed
                assert not edited_dir.exists()
                assert not title_dir.exists()
                assert not norm_dir.exists()


# ---------------------------------------------------------------------------
# process_job: quality validation
# ---------------------------------------------------------------------------


class TestProcessJobQualityValidation:
    async def test_quality_validation_runs(self):
        """Quality validator is invoked and its result stored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
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
                patch(
                    "src.services.compilation.quality_validator.QualityValidator"
                ) as mock_validator,
            ):
                mock_ffmpeg.convert_to_mp4 = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 5.0
                mock_ffmpeg.extract_thumbnail = MagicMock()
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                qr = MagicMock()
                qr.valid = True
                qr.issues = []
                qr.file_size_bytes = 1000
                qr.duration_seconds = 5.0
                qr.resolution = "1920x1080"
                qr.audio_peak_db = -3.0
                qr.encoding_valid = True
                mock_validator.return_value.validate.return_value = qr

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                call_kwargs = mock_completed.call_args[1]
                assert call_kwargs["quality_metrics"]["valid"] is True


# ---------------------------------------------------------------------------
# process_job: chapter embedding
# ---------------------------------------------------------------------------


class TestProcessJobChapters:
    async def test_chapters_embedded_when_present(self):
        """When chapters are defined, embed_chapters is called."""
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            db = AsyncMock()

            vid = work_dir / "v1.webm"
            vid.write_bytes(b"video")

            job = _base_job(
                shot_list={
                    "parts": [
                        {
                            "id": "p1",
                            "part_type": "browser_recording",
                            "chapter": 1,
                            "title": "Ch1",
                            "subtitle": "",
                        },
                    ],
                    "recording_order": ["p1"],
                },
                recordings=[
                    {
                        "part_id": "p1",
                        "success": True,
                        "video_path": str(vid),
                        "duration_seconds": 10.0,
                    }
                ],
            )

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
                ),
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
                mock_ffmpeg.probe_duration.return_value = 10.0
                mock_ffmpeg.extract_thumbnail = MagicMock()

                def fake_embed_chapters(inp, out, chapters):
                    out.write_bytes(b"chaptered")

                mock_ffmpeg.embed_chapters = MagicMock(side_effect=fake_embed_chapters)
                mock_editor.return_value.auto_trim.side_effect = lambda p: p

                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "p1.mp4").write_bytes(b"n1")
                output_path.write_bytes(b"output")

                await service.process_job(db, "job-1", TENANT_U4)

                mock_ffmpeg.embed_chapters.assert_called_once()


# ---------------------------------------------------------------------------
# process_job: xfade_transition in filter_config
# ---------------------------------------------------------------------------


class TestProcessJobXfadeTransition:
    async def test_create_job_with_xfade_transition(self):
        """xfade_transition is stored in filter_config."""
        service = CompilerService(MagicMock())
        db = AsyncMock()

        mock_job = MagicMock()
        mock_job.job_id = "job-xfade"

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
                xfade_transition="wipeleft",
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["filter_config"]["xfade_transition"] == "wipeleft"


# ---------------------------------------------------------------------------
# _assemble_with_xfade
# ---------------------------------------------------------------------------


class TestAssembleWithXfade:
    @patch("src.services.compilation.compiler_service.FFmpegOps.crossfade")
    def test_xfade_two_segments(self, mock_crossfade):
        service = CompilerService(MagicMock())
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            seg1 = work_dir / "s1.mp4"
            seg2 = work_dir / "s2.mp4"
            seg1.write_bytes(b"s1")
            seg2.write_bytes(b"s2")
            output = work_dir / "out.mp4"

            (work_dir / "xfade_1.mp4").write_bytes(b"xfaded")

            service._assemble_with_xfade([seg1, seg2], output, work_dir, 0.5, "wipeleft")
            mock_crossfade.assert_called_once()
            assert output.exists()

    @patch("src.services.compilation.compiler_service.FFmpegOps.crossfade")
    def test_xfade_three_segments(self, mock_crossfade):
        """With 3 segments, crossfade is called twice and intermediates cleaned."""
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

            (work_dir / "xfade_1.mp4").write_bytes(b"xfade1")
            (work_dir / "xfade_2.mp4").write_bytes(b"xfade2")

            service._assemble_with_xfade([seg1, seg2, seg3], output, work_dir, 0.5, "fade")
            assert mock_crossfade.call_count == 2
            assert output.exists()


# ---------------------------------------------------------------------------
# _mix_per_segment_narration: speed-adjust path
# ---------------------------------------------------------------------------


class TestMixPerSegmentNarrationSpeedAdjust:
    @patch("src.services.compilation.compiler_service.FFmpegOps.mix_voiceover")
    @patch("src.services.compilation.compiler_service.FFmpegOps.speed_adjust")
    @patch("src.services.compilation.compiler_service.FFmpegOps.probe_duration")
    def test_speed_adjust_when_video_longer(self, mock_dur, mock_speed, mock_mix):
        """Video significantly longer than audio triggers speed adjustment."""
        mock_dur.side_effect = [5.0, 10.0]  # audio=5, video=10 -> 10 > 5*1.05
        service = CompilerService(MagicMock())

        with tempfile.TemporaryDirectory() as tmpdir:
            work = Path(tmpdir)
            sped_path = work / "p1_sped.mp4"
            sped_path.write_bytes(b"sped")
            mixed_path = work / "p1_narrated.mp4"
            mixed_path.write_bytes(b"mixed")

            result = service._mix_per_segment_narration(
                _mixctx(), Path("/video.mp4"), Path("/audio.mp3"), work, "p1"
            )
            # Bug 1 fix: _narrated.mp4 is renamed to {part_id}.mp4 so
            # downstream lookups by seg.stem match the part_id.
            assert result == work / "p1.mp4"
            mock_speed.assert_called_once()


# ---------------------------------------------------------------------------
# Billing safety: never meter a compilation whose video is not in storage
# ---------------------------------------------------------------------------


class _BackendUploadError(Exception):
    """Stand-in for a storage-backend error outside the old curated tuple.

    minio's ``S3Error`` is exactly this shape: not an ``OSError``,
    ``RuntimeError``, ``ValueError`` or ``TimeoutError``. The old handler in
    ``_upload_final_artifacts`` would not have caught it at all, and neither
    would ``process_job``'s identically-shaped handler -- the job would have
    been stranded in ``processing``. Using it here proves the widened
    ``except Exception`` both catches it and routes it to the failure path.
    """


class TestNoBillingWithoutStoredArtifact:
    """The final PUT to MinIO can fail (5xx, connection reset) on a compile that
    otherwise went fine. Before the fix ``_upload_final_artifacts`` swallowed
    that, logged a warning, and returned the pod-local ``work_dir`` path; the
    job was then marked ``completed`` with that path and
    ``record_usage(compilations_completed=1, output_duration, output_size)`` ran.
    ``GET /api/compile/jobs/{id}/video`` 404s on both storage backends for such a
    path, so the tenant paid for a video no consumer can fetch.
    """

    @staticmethod
    def _run_ctx(work_dir: Path, job, storage):
        """Patch set shared by the cases below."""
        return (
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
            ),
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.mark_failed",
                new_callable=AsyncMock,
            ),
            patch(
                "src.services.compilation.compiler_service.CompilationJobManager.record_usage",
                new_callable=AsyncMock,
            ),
            patch("src.config.get_settings", return_value=_make_settings_mock()),
            patch("src.services.compilation.compiler_service.TitleCardGenerator"),
            patch("src.services.compilation.compiler_service.FFmpegOps"),
        )

    @staticmethod
    def _title_card_job():
        return _base_job(
            shot_list={
                "parts": [
                    {
                        "id": "tc1",
                        "part_type": "title_card",
                        "title": "Intro",
                        "subtitle": "Welcome",
                        "chapter": 1,
                        "estimated_duration_seconds": 3.0,
                    }
                ],
                "recording_order": ["tc1"],
            },
        )

    @staticmethod
    def _assert_not_billed(mock_completed, mock_failed, mock_usage) -> None:
        mock_completed.assert_not_awaited()
        mock_failed.assert_awaited_once()
        assert mock_usage.await_count == 1, (
            f"expected exactly one usage write, got {mock_usage.await_args_list}"
        )
        usage_kwargs = mock_usage.await_args[1]
        assert usage_kwargs.get("is_error") is True, (
            f"the only usage write must be the error counter, got {usage_kwargs}"
        )
        assert "compilations_completed" not in usage_kwargs, (
            f"tenant was billed a completed compilation: {usage_kwargs}"
        )

    @pytest.mark.parametrize(
        "upload_error",
        [
            # In the old curated tuple: this is the case that silently fell back
            # to the pod-local path and billed the tenant for it.
            pytest.param(OSError("connection reset by peer"), id="in-old-tuple"),
            # NOT in the old curated tuple: this one escaped the handler AND
            # ``process_job``'s identical tuple, leaving the job in ``processing``
            # with no ``mark_failed`` at all.
            pytest.param(_BackendUploadError("503 Slow Down"), id="outside-old-tuple"),
        ],
    )
    async def test_final_upload_failure_fails_job_and_is_not_billed(self, upload_error):
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            storage.store_file = AsyncMock(side_effect=upload_error)
            service = CompilerService(storage)
            db = AsyncMock()
            job = self._title_card_job()

            (
                p_get,
                p_proc,
                p_completed,
                p_failed,
                p_usage,
                p_settings,
                p_title,
                p_ffmpeg,
            ) = self._run_ctx(work_dir, job, storage)

            with (
                p_get,
                p_proc,
                p_completed as mock_completed,
                p_failed as mock_failed,
                p_usage as mock_usage,
                p_settings,
                p_title as mock_title_gen,
                p_ffmpeg as mock_ffmpeg,
            ):
                mock_title_gen.return_value.generate_video = MagicMock()
                mock_ffmpeg.normalize_stream = MagicMock()
                mock_ffmpeg.probe_duration.return_value = 3.0
                mock_ffmpeg.extract_thumbnail = MagicMock()
                norm_dir = work_dir / "normalized"
                norm_dir.mkdir(parents=True, exist_ok=True)
                (norm_dir / "tc1.mp4").write_bytes(b"normalized")
                (work_dir / "demo.mp4").write_bytes(b"output")

                with pytest.raises(
                    ArtifactNotStoredError, match="Failed to upload compiled video to storage"
                ):
                    await service.process_job(db, "job-1", TENANT_U4)

                self._assert_not_billed(mock_completed, mock_failed, mock_usage)

    async def test_missing_output_file_is_never_returned_as_a_local_path(self):
        """Same class, other entry into the same function: the encode left no
        file. The old code skipped the upload entirely and returned
        ``str(output_path)`` -- the pod-local ``work_dir`` path -- which
        ``_finalize`` then handed to ``mark_completed`` before billing a 0-byte
        compilation. It must raise instead; ``ArtifactNotStoredError`` is a
        ``RuntimeError``, so ``process_job``'s handler routes it to
        ``mark_failed`` + ``record_usage(is_error=True)`` exactly as proven above.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            storage = _mock_storage(work_dir)
            service = CompilerService(storage)
            ctx = _JobContext(
                job_id="job-1",
                tenant_id=TENANT_U4,
                job=self._title_card_job(),
                settings=_make_settings_mock(),
                work_dir=work_dir,
                output_path=work_dir / "demo.mp4",  # deliberately never written
            )

            assert not ctx.output_path.exists()
            with pytest.raises(ArtifactNotStoredError, match="is missing at"):
                await service._upload_final_artifacts(ctx, None)

            storage.store_file.assert_not_awaited()
