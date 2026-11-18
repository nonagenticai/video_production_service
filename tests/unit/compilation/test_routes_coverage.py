"""Additional route tests to cover missing lines (template resolution, WebSocket, etc.)."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from tests.conftest import TENANT_U1, TENANT_U4


def _fake_create_task(coro, **kwargs):
    """Close the coroutine so it doesn't trigger 'coroutine was never awaited'."""
    coro.close()
    return MagicMock()


@contextlib.contextmanager
def _socket_owns_its_job(mock_ws, tenant=TENANT_U1, subject="user-1"):
    """Give an ``AsyncMock`` socket a REAL identity and an owned job.

    ``job_progress_ws`` now calls ``_authorize_job_for_socket`` first, which seeds
    ``set_identity_context`` from ``websocket.state.tenant_id`` and then asks
    ``CompilationJobManager.get_job`` (RLS decides ownership) whether the job is
    visible. Two things follow for a hand-rolled mock socket:

    * ``AsyncMock`` auto-creates a child mock for every attribute, so
      ``websocket.state.tenant_id`` was an ``AsyncMock`` and
      ``set_identity_context`` rejected it — correctly. It refuses a non-string
      tenant BY DESIGN (that strictness is what keeps a spoofable identity out),
      so the fix is to give the mock a real uuid string, never to soften the
      check.
    * The job lookup must resolve, or every one of these tests would be measuring
      the ownership refusal instead of what it was written to measure.

    This makes the socket a legitimate, entitled owner so each test still
    exercises its own subject: no-manager handling, disconnect handling, relay.
    """
    mock_ws.state.tenant_id = str(tenant)
    mock_ws.state.subject_id = subject

    @contextlib.asynccontextmanager
    async def _session():
        yield object()

    with (
        patch("src.api.routes.compilation.get_db_session", _session),
        patch(
            "src.api.routes._compilation_jobs.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
    ):
        yield mock_ws


# ---------------------------------------------------------------------------
# Template resolution in compile_video (lines 97-113)
# ---------------------------------------------------------------------------


class TestCompileVideoTemplateResolution:
    async def test_compile_video_with_template_not_found(self):
        """When template is specified but not found, raise 400."""
        from src.api.routes.compilation import compile_video
        from src.api.schemas.compilation import CompileRequest, ShotListInput

        body = CompileRequest(
            shot_list=ShotListInput(parts=[], recording_order=[]),
            recordings=[],
            template="nonexistent-template",
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="job-1")
        mock_service.register_task = MagicMock()
        db = AsyncMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch(
                "src.services.compilation.template_service.TemplateService.resolve_template",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await compile_video(body, MagicMock(), db)
            assert exc_info.value.status_code == 400
            assert "Template not found" in str(exc_info.value.detail)

    async def test_compile_video_with_template_applies_defaults(self):
        """When template is found, apply its values as defaults."""
        from src.api.routes.compilation import compile_video
        from src.api.schemas.compilation import CompileRequest, ShotListInput

        body = CompileRequest(
            shot_list=ShotListInput(parts=[], recording_order=[]),
            recordings=[],
            template="my-template",
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="templated-job")
        mock_service.register_task = MagicMock()

        tpl_config = {
            "preset_name": "cinematic",
            "theme_name": "dark",
            "transition_type": "fade",
            "crossfade_duration": 1.5,
        }

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch(
                "src.services.compilation.template_service.TemplateService.resolve_template",
                new_callable=AsyncMock,
                return_value=tpl_config,
            ),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await compile_video(body, MagicMock(), AsyncMock())

        assert result.job_id == "templated-job"
        # Verify the body was mutated with template defaults
        assert body.preset == "cinematic"
        assert body.theme_name == "dark"
        assert body.transition_type == "fade"
        assert body.crossfade_duration == 1.5


# ---------------------------------------------------------------------------
# _process_job_bg cancelled error (line 362-363, 368)
# ---------------------------------------------------------------------------


class TestProcessJobBgCancelled:
    async def test_process_job_bg_cancelled(self):
        """CancelledError should be caught and not propagate."""
        from src.api.routes.compilation import _process_job_bg

        service = MagicMock()
        service.process_job = AsyncMock(side_effect=asyncio.CancelledError())
        service._ws_manager = None
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        with patch("src.api.routes.compilation.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            # Should not raise
            await _process_job_bg(service, "cancelled-job", TENANT_U4)

        service.unregister_task.assert_called_once_with("cancelled-job")

    async def test_process_job_bg_ws_setup_failure(self):
        """If WebSocket progress reporter setup fails, job should still run."""
        from src.api.routes.compilation import _process_job_bg

        service = MagicMock()
        service.process_job = AsyncMock()
        service._ws_manager = MagicMock()
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        with (
            patch("src.api.routes.compilation.get_db_session") as mock_get_db,
            patch(
                "src.api.ws.ProgressReporter",
                side_effect=RuntimeError("ws setup failed"),
            ),
        ):
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_bg(service, "job-ws-fail", TENANT_U4)

        service.process_job.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_job_video (lines 548)
# ---------------------------------------------------------------------------


class TestGetJobVideo:
    async def test_get_job_video_no_video_data(self):
        """When storage.retrieve returns None, raise 404."""
        from src.api.routes.compilation import get_job_video

        mock_job = MagicMock()
        mock_job.output_path = "/path/to/video.mp4"

        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=None)

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_video("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404
            assert "not found on disk" in str(exc_info.value.detail)

    async def test_get_job_video_success(self):
        """When video data is available, return it."""
        from src.api.routes.compilation import get_job_video

        mock_job = MagicMock()
        mock_job.output_path = "/path/to/video.mp4"

        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=b"fake-video-data")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
        ):
            response = await get_job_video("job-1", MagicMock(), AsyncMock())

        assert response.body == b"fake-video-data"
        assert response.media_type == "video/mp4"


# ---------------------------------------------------------------------------
# get_job_video_url (lines 573, 575)
# ---------------------------------------------------------------------------


class TestGetJobVideoUrl:
    async def test_get_job_video_url_no_output_path(self):
        """Job with no output_path should return 404."""
        from src.api.routes.compilation import get_job_video_url

        mock_job = MagicMock()
        mock_job.output_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_video_url("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404

    async def test_get_job_video_url_success(self):
        """url is always None; consumers must use fallback_download_url
        (minio.test:8389 is unreachable from browsers)."""
        from src.api.routes.compilation import get_job_video_url

        mock_job = MagicMock()
        mock_job.output_path = "/path/to/video.mp4"

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            result = await get_job_video_url("job-1", MagicMock(), AsyncMock())

        assert result.url is None
        assert result.expires_in == 0
        assert result.fallback_download_url == "/api/compile/jobs/job-1/video"


# ---------------------------------------------------------------------------
# generate_thumbnail - video file not found (line 630)
# ---------------------------------------------------------------------------


class TestExtractThumbnailVideoNotFound:
    async def test_thumbnail_video_file_not_found(self, tmp_path):
        """When video file does not exist, raise 404."""
        from src.api.routes.compilation import generate_thumbnail
        from src.api.schemas.compilation import ThumbnailRequest

        body = ThumbnailRequest(timestamp=5.0, width=320, height=240)

        mock_job = MagicMock()
        mock_job.output_path = str(tmp_path / "nonexistent.mp4")

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_thumbnail("job-1", body, MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404
            assert "Video file not found" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# WebSocket endpoint (lines 664-685)
# ---------------------------------------------------------------------------


class TestJobProgressWebSocket:
    async def test_ws_no_manager_configured(self):
        """When ws_manager is None, close the WebSocket."""
        from src.api.routes.compilation import job_progress_ws

        mock_ws = AsyncMock()
        mock_ws.app = MagicMock()
        mock_ws.app.state = MagicMock(spec=[])  # No ws_manager attribute

        with _socket_owns_its_job(mock_ws):
            await job_progress_ws(mock_ws, "job-1")
        mock_ws.close.assert_awaited_once()

    async def test_ws_with_manager_disconnect(self):
        """WebSocket connects, then disconnects cleanly."""
        from fastapi import WebSocketDisconnect

        from src.api.routes.compilation import job_progress_ws

        mock_ws = AsyncMock()
        mock_manager = AsyncMock()
        # Redis absent → exercise the in-memory (pod-affine) fallback path.
        mock_ws.app.state.redis_client = None
        mock_ws.app.state.ws_manager = mock_manager
        mock_ws.receive_text.side_effect = WebSocketDisconnect()

        with _socket_owns_its_job(mock_ws):
            await job_progress_ws(mock_ws, "job-1")
        mock_manager.connect.assert_awaited_once_with("job-1", mock_ws)
        mock_manager.disconnect.assert_awaited_once_with("job-1", mock_ws)


# ---------------------------------------------------------------------------
# get_compilation_report - job not found (line 702)
# ---------------------------------------------------------------------------


class TestGetCompilationReport:
    async def test_report_job_not_found(self):
        """Requesting report for nonexistent job should 404."""
        from src.api.routes.compilation import get_compilation_report

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_compilation_report("missing-job", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# cleanup - storage deletion failure (lines 751-752)
# ---------------------------------------------------------------------------


class TestCleanupStorageFailure:
    async def test_cleanup_storage_delete_failure_logged(self):
        """When storage deletion fails for a job, it should be logged but not stop cleanup."""
        from src.api.routes.compilation import on_demand_cleanup
        from src.api.schemas.compilation import CleanupRequest

        body = CleanupRequest(min_age_hours=24, dry_run=False)

        mock_job1 = MagicMock()
        mock_job1.job_id = "old-job-1"
        mock_job2 = MagicMock()
        mock_job2.job_id = "old-job-2"

        mock_storage = MagicMock()
        mock_storage.delete_job_dir = AsyncMock(side_effect=OSError("disk error"))

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch(
                "src.api.routes.compilation.CompilationJobManager.find_jobs_by_criteria",
                new_callable=AsyncMock,
                return_value=[mock_job1, mock_job2],
            ),
            patch(
                "src.api.routes.compilation.CompilationJobManager.delete_jobs",
                new_callable=AsyncMock,
                return_value=2,
            ),
        ):
            result = await on_demand_cleanup(body, MagicMock(), AsyncMock())

        assert result.deleted_count == 2
        assert not result.dry_run


# ---------------------------------------------------------------------------
# Findings video theme fallback (line 963)
# ---------------------------------------------------------------------------


class TestFindingsVideoThemeFallback:
    async def test_findings_video_theme_from_theme_object(self):
        """When theme_name is not set, fallback to body.theme.theme_name."""
        from src.api.routes.compilation import create_findings_video
        from src.api.schemas.compilation import FindingsVideoRequest, ThemeInput

        body = FindingsVideoRequest(
            findings=[],
            title="Test",
            theme=ThemeInput(theme_name="corporate", primary_color="#000000"),
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="findings-job")
        mock_service.register_task = MagicMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await create_findings_video(body, MagicMock(), AsyncMock())

        assert result.job_id == "findings-job"
        call_kwargs = mock_service.create_job.call_args[1]
        assert call_kwargs["theme_name"] == "corporate"


# ---------------------------------------------------------------------------
# Audio export - video file not found (line 1074)
# ---------------------------------------------------------------------------


class TestAudioExportVideoNotFound:
    async def test_audio_export_video_file_missing(self, tmp_path):
        """When the video file doesn't exist on disk, raise 404."""
        from src.api.routes.compilation import audio_export
        from src.api.schemas.compilation import AudioExportRequest

        body = AudioExportRequest(job_id="job-1", format="mp3")

        mock_job = MagicMock()
        mock_job.output_path = str(tmp_path / "nonexistent.mp4")

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await audio_export(body, MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404
            assert "Video file not found" in str(exc_info.value.detail)


# ---------------------------------------------------------------------------
# Preview - video file not found (line 1121)
# ---------------------------------------------------------------------------


class TestPreviewVideoNotFound:
    async def test_preview_video_file_missing(self, tmp_path):
        """When the video file doesn't exist on disk, raise 404."""
        from src.api.routes.compilation import generate_preview
        from src.api.schemas.compilation import PreviewRequest

        body = PreviewRequest(job_id="job-1")

        mock_job = MagicMock()
        mock_job.output_path = str(tmp_path / "nonexistent.mp4")

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_preview(body, MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404
            assert "Video file not found" in str(exc_info.value.detail)
