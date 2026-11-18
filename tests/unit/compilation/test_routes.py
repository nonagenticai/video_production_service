"""Unit tests for API routes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.routes.compilation import (
    _get_compiler_service,
    _get_storage,
    _get_tenant_id,
    _process_job_bg,
)
from tests.conftest import TENANT_U4, TENANT_U5

# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestGetTenantId:
    def test_returns_tenant_id_from_request_state(self):
        request = MagicMock()
        # The auth middleware stashes ``str(tenant_id)``; the helper parses it.
        request.state.tenant_id = str(TENANT_U5)
        assert _get_tenant_id(request) == TENANT_U5

    def test_refuses_when_missing_instead_of_returning_anonymous(self):
        """Was ``assert result == "anonymous"``. See src/api/tenant.py.

        The sentinel is gone fleet-wide (w2v-common 0.67.0) and ``tenant_id`` is
        a ``uuid`` column, so the only honest answer to "who is this?" with no
        identity on the request is to refuse it.
        """
        request = MagicMock(spec=[])
        request.state = MagicMock(spec=[])
        with pytest.raises(HTTPException) as exc:
            _get_tenant_id(request)
        assert exc.value.status_code == 401

    def test_refuses_the_nil_uuid_by_value(self):
        """Rejected BY VALUE, so every spelling of the nil uuid is covered."""
        for spelling in (
            "00000000-0000-0000-0000-000000000000",
            "00000000000000000000000000000000",
            "{00000000-0000-0000-0000-000000000000}",
            "urn:uuid:00000000-0000-0000-0000-000000000000",
        ):
            request = MagicMock()
            request.state.tenant_id = spelling
            with pytest.raises(HTTPException) as exc:
                _get_tenant_id(request)
            assert exc.value.status_code == 401, spelling


class TestGetCompilerService:
    def test_returns_compiler_service(self):
        request = MagicMock()
        mock_service = MagicMock()
        request.app.state.compiler_service = mock_service
        assert _get_compiler_service(request) is mock_service


class TestGetStorage:
    def test_returns_storage_service(self):
        request = MagicMock()
        mock_storage = MagicMock()
        request.app.state.storage_service = mock_storage
        assert _get_storage(request) is mock_storage


# ---------------------------------------------------------------------------
# Background job processing tests
# ---------------------------------------------------------------------------


class TestProcessJobBg:
    async def test_process_job_bg_success(self):
        service = MagicMock()
        service.process_job = AsyncMock()
        service._ws_manager = None
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        with patch("src.api.routes.compilation.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_bg(service, "job-1", TENANT_U4)

        service.process_job.assert_awaited_once()
        service.unregister_task.assert_called_once_with("job-1")

    async def test_process_job_bg_with_ws_manager(self):
        service = MagicMock()
        service.process_job = AsyncMock()
        service._ws_manager = MagicMock()
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        with patch("src.api.routes.compilation.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_bg(service, "job-1", TENANT_U4)

        service.process_job.assert_awaited_once()

    async def test_process_job_bg_handles_exception(self):
        service = MagicMock()
        service.process_job = AsyncMock(side_effect=RuntimeError("ffmpeg crashed"))
        service._ws_manager = None
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        # Should not raise
        with patch("src.api.routes.compilation.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_bg(service, "job-1", TENANT_U4)
        service.unregister_task.assert_called_once_with("job-1")

    async def test_process_job_bg_handles_cancelled(self):
        service = MagicMock()
        service.process_job = AsyncMock(side_effect=asyncio.CancelledError())
        service._ws_manager = None
        service.unregister_task = MagicMock()

        mock_db = AsyncMock()
        with patch("src.api.routes.compilation.get_db_session") as mock_get_db:
            mock_get_db.return_value.__aenter__ = AsyncMock(return_value=mock_db)
            mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)
            await _process_job_bg(service, "job-1", TENANT_U4)
        service.unregister_task.assert_called_once_with("job-1")


# ---------------------------------------------------------------------------
# Route endpoint tests (testing the route functions directly)
# ---------------------------------------------------------------------------


class TestListThemes:
    async def test_list_themes_returns_response(self):
        from src.api.routes.compilation import list_themes

        result = await list_themes()
        assert hasattr(result, "themes")
        assert isinstance(result.themes, list)
        assert len(result.themes) > 0


class TestGetTheme:
    async def test_get_existing_theme(self):
        from src.api.routes.compilation import get_theme

        # "corporate" is one of the built-in themes
        result = await get_theme("corporate")
        assert result.name == "corporate"

    async def test_get_missing_theme(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_theme

        with pytest.raises(HTTPException) as exc_info:
            await get_theme("nonexistent-theme")
        assert exc_info.value.status_code == 404


class TestListPresets:
    async def test_list_presets_returns_response(self):
        from src.api.routes.compilation import list_presets

        result = await list_presets()
        assert hasattr(result, "presets")
        assert isinstance(result.presets, list)
        assert len(result.presets) > 0


class TestGetPreset:
    async def test_get_existing_preset(self):
        from src.api.routes.compilation import get_preset

        result = await get_preset("standard")
        assert result.name == "standard"

    async def test_get_missing_preset(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_preset

        with pytest.raises(HTTPException) as exc_info:
            await get_preset("nonexistent-preset")
        assert exc_info.value.status_code == 404


class TestGetJobStatus:
    async def test_job_found(self):
        from src.api.routes.compilation import get_job_status

        mock_job = MagicMock()
        mock_job.job_id = "job-123"
        mock_job.status = "completed"
        mock_job.created_at = datetime.now(UTC)
        mock_job.completed_at = datetime.now(UTC)
        mock_job.error_message = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            request = MagicMock()
            db = AsyncMock()
            result = await get_job_status("job-123", request, db)

        assert result.job_id == "job-123"
        assert result.status == "completed"
        assert result.result_url == "/api/compile/jobs/job-123/result"

    async def test_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_status

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            request = MagicMock()
            db = AsyncMock()
            with pytest.raises(HTTPException) as exc_info:
                await get_job_status("missing", request, db)
            assert exc_info.value.status_code == 404

    async def test_pending_job_no_result_url(self):
        from src.api.routes.compilation import get_job_status

        mock_job = MagicMock()
        mock_job.job_id = "job-pending"
        mock_job.status = "pending"
        mock_job.created_at = datetime.now(UTC)
        mock_job.completed_at = None
        mock_job.error_message = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            request = MagicMock()
            db = AsyncMock()
            result = await get_job_status("job-pending", request, db)

        assert result.result_url is None


class TestGetJobResult:
    async def test_completed_job_with_report_and_quality(self):
        from src.api.routes.compilation import get_job_result

        mock_job = MagicMock()
        mock_job.job_id = "job-done"
        mock_job.status = "completed"
        mock_job.output_path = "/output/demo.mp4"
        mock_job.total_duration_seconds = 120.5
        mock_job.total_parts = 5
        mock_job.file_size_bytes = 1048576
        mock_job.chapter_timestamps = [
            {"chapter_number": 1, "title": "Intro", "start_seconds": 0.0, "duration_seconds": 30.0}
        ]
        mock_job.compilation_time_seconds = 15.0
        mock_job.error_message = None
        mock_job.compilation_report = {
            "job_id": "job-done",
            "compilation_time_seconds": 15.0,
            "stages": [{"stage": "normalize", "duration_seconds": 5.0}],
            "segments_processed": 5,
            "segments_excluded": 0,
            "transitions_used": 4,
            "audio_mixing_details": None,
            "input_duration_seconds": 130.0,
            "output_duration_seconds": 120.5,
        }
        mock_job.quality_metrics = {
            "valid": True,
            "file_size_bytes": 1048576,
            "duration_seconds": 120.5,
            "resolution": "1920x1080",
            "audio_peak_db": -3.0,
            "encoding_valid": True,
        }

        mock_storage = MagicMock()
        mock_storage.presign_url = AsyncMock(return_value="https://example.com/presigned")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
        ):
            result = await get_job_result("job-done", MagicMock(), AsyncMock())

        assert result.job_id == "job-done"
        assert result.status == "completed"
        assert result.compilation_report is not None
        assert result.quality_metrics is not None
        assert result.chapter_timestamps is not None
        assert len(result.chapter_timestamps) == 1
        # This fixture's blob has no status key, so the API must NOT claim success.
        assert result.compilation_report.status == "unknown"
        assert result.compilation_report.degradations == []

    @staticmethod
    def _job_with_report(report_extra: dict) -> MagicMock:
        """A completed job whose compilation_report carries *report_extra*."""
        job = MagicMock()
        job.job_id = "job-done"
        job.status = "completed"
        job.output_path = "/output/demo.mp4"
        job.total_duration_seconds = 120.5
        job.total_parts = 5
        job.file_size_bytes = 1048576
        job.chapter_timestamps = None
        job.compilation_time_seconds = 15.0
        job.error_message = None
        job.quality_metrics = None
        job.compilation_report = {
            "job_id": "job-done",
            "compilation_time_seconds": 15.0,
            "stages": [],
            "segments_processed": 5,
            "segments_excluded": 0,
            "transitions_used": 4,
            "audio_mixing_details": None,
            "input_duration_seconds": 130.0,
            "output_duration_seconds": 120.5,
            **report_extra,
        }
        return job

    async def _result_for(self, job: MagicMock):
        from src.api.routes.compilation import get_job_result

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=job,
        ):
            return await get_job_result("job-done", MagicMock(), AsyncMock())

    async def test_dropped_subtitles_are_visible_in_the_response_body(self):
        """A compile that silently lost subtitles must be distinguishable from a clean one.

        Regression: _finalize sets status=partial_success on the report blob,
        but the response mapper dropped it and the job row reads 'completed'
        with error=None -- so the caller received a subtitle-less video with no
        field anywhere indicating anything went wrong.

        subtitle_error itself is deliberately NOT surfaced (it echoes a resolved
        server path); the drop reaches the caller through `degradations`.
        """
        clean = await self._result_for(self._job_with_report({"status": "success"}))
        degraded = await self._result_for(
            self._job_with_report(
                {
                    "status": "partial_success",
                    "subtitle_error": "burn-in failed: font 'DejaVu Sans' not found",
                    "degradations": [
                        {"feature": "subtitles", "reason": "Subtitles could not be applied"}
                    ],
                }
            )
        )

        # Both are terminally 'completed' -- the video exists and is billable.
        assert clean.status == "completed"
        assert degraded.status == "completed"
        # ...but the response body must still tell them apart.
        assert clean.compilation_report.status == "success"
        assert degraded.compilation_report.status == "partial_success"
        assert clean.compilation_report.status != degraded.compilation_report.status

        assert clean.compilation_report.degradations == []
        assert [d.feature for d in degraded.compilation_report.degradations] == ["subtitles"]

        # And it must survive serialization -- this is what the HTTP client sees.
        body = degraded.model_dump()
        assert body["compilation_report"]["status"] == "partial_success"
        # The raw ffmpeg text must NOT reach the caller.
        assert "font 'DejaVu Sans' not found" not in str(body["compilation_report"])
        assert body["compilation_report"]["degradations"][0]["feature"] == "subtitles"

    async def test_absent_status_is_unknown_never_success(self):
        """A report blob with no status key must not be reported as a clean compile.

        Only the producer knows whether a feature was swallowed. A blob that
        never carried the claim must read 'unknown' -- defaulting it to
        'success' would have the API invent a guarantee nobody made.
        """
        result = await self._result_for(self._job_with_report({}))
        assert result.compilation_report.status == "unknown"
        assert result.compilation_report.status != "success"
        assert result.model_dump()["compilation_report"]["status"] == "unknown"

    async def test_dropped_watermark_does_not_read_as_success(self):
        """A non-subtitle degradation must be just as visible as a subtitle one.

        _apply_watermark's LFI guard skips the overlay rather than failing the
        job -- a known silent drop. Before degradations existed it left no trace
        in the report at all, so a watermark-less video read as fully successful.
        """
        degraded = await self._result_for(
            self._job_with_report(
                {
                    "status": "partial_success",
                    "degradations": [
                        {
                            "feature": "watermark",
                            "reason": "Watermark image refused by the local-path guard",
                        }
                    ],
                }
            )
        )

        assert degraded.status == "completed"
        assert degraded.compilation_report.status == "partial_success"
        # No subtitle failure here -- the subtitle field must not be what carries it.
        assert [d.feature for d in degraded.compilation_report.degradations] == ["watermark"]
        assert [d.feature for d in degraded.compilation_report.degradations] == ["watermark"]
        body = degraded.model_dump()
        assert "local-path guard" in body["compilation_report"]["degradations"][0]["reason"]

    async def test_job_not_completed(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_result

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.status = "processing"

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_result("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 409

    async def test_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_result

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_result("missing", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404


class TestGetJobVideo:
    async def test_video_found(self):
        from src.api.routes.compilation import get_job_video

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.output_path = "/output/demo.mp4"

        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=b"video-data")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
        ):
            result = await get_job_video("job-1", MagicMock(), AsyncMock())

        assert result.body == b"video-data"
        assert result.media_type == "video/mp4"

    async def test_video_not_on_disk(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_video

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.output_path = "/output/demo.mp4"

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

    async def test_video_no_output_path(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_video

        mock_job = MagicMock()
        mock_job.output_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_video("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404


class TestGetJobVideoUrl:
    async def test_returns_stream_through_url_only(self):
        """get_job_video_url always returns url=None now; browsers can't reach
        minio.test:8389, so consumers MUST use the stream-through endpoint
        at fallback_download_url. Storage backend is never consulted."""
        from src.api.routes.compilation import get_job_video_url

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.output_path = "/output/demo.mp4"

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            result = await get_job_video_url("job-1", MagicMock(), AsyncMock())

        assert result.url is None
        assert result.expires_in == 0
        assert result.fallback_download_url == "/api/compile/jobs/job-1/video"


class TestGetJobThumbnail:
    async def test_thumbnail_not_available(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_thumbnail

        mock_job = MagicMock()
        mock_job.thumbnail_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_thumbnail("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404


class TestGetCompilationReport:
    async def test_report_available(self):
        from src.api.routes.compilation import get_compilation_report

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.compilation_report = {
            "job_id": "job-1",
            "compilation_time_seconds": 15.3,
            "stages": [],
            "segments_processed": 3,
            "segments_excluded": 0,
            "transitions_used": 2,
            "input_duration_seconds": 110,
            "output_duration_seconds": 100,
        }

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            result = await get_compilation_report("job-1", MagicMock(), AsyncMock())
        assert result.job_id == "job-1"

    async def test_no_report(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_compilation_report

        mock_job = MagicMock()
        mock_job.compilation_report = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_compilation_report("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404


class TestOnDemandCleanup:
    async def test_dry_run(self):
        from src.api.routes.compilation import on_demand_cleanup
        from src.api.schemas.compilation import CleanupRequest

        body = CleanupRequest(dry_run=True, status_filter=["completed"])

        mock_jobs = [MagicMock(job_id="j1"), MagicMock(job_id="j2")]
        request = MagicMock()
        request.state.tenant_id = TENANT_U4
        mock_storage = MagicMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch(
                "src.api.routes.compilation.CompilationJobManager.find_jobs_by_criteria",
                new_callable=AsyncMock,
                return_value=mock_jobs,
            ),
        ):
            result = await on_demand_cleanup(body, request, AsyncMock())

        assert result.dry_run is True
        assert result.deleted_count == 2
        assert result.job_ids == ["j1", "j2"]

    async def test_actual_cleanup(self):
        from src.api.routes.compilation import on_demand_cleanup
        from src.api.schemas.compilation import CleanupRequest

        body = CleanupRequest(dry_run=False, status_filter=["failed"])

        mock_jobs = [MagicMock(job_id="j1")]
        mock_storage = AsyncMock()
        mock_storage.delete_job_dir = AsyncMock()
        request = MagicMock()
        db = AsyncMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch(
                "src.api.routes.compilation.CompilationJobManager.find_jobs_by_criteria",
                new_callable=AsyncMock,
                return_value=mock_jobs,
            ),
            patch(
                "src.api.routes.compilation.CompilationJobManager.delete_jobs",
                new_callable=AsyncMock,
                return_value=1,
            ),
        ):
            result = await on_demand_cleanup(body, request, db)

        assert result.dry_run is False
        assert result.deleted_count == 1


class TestCancelJob:
    async def test_cancel_success(self):
        from src.api.routes.compilation import cancel_job

        mock_service = MagicMock()
        mock_service.cancel_job = AsyncMock(return_value=True)

        with patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service):
            result = await cancel_job("job-1", MagicMock(), AsyncMock())

        assert result.job_id == "job-1"
        assert result.status == "cancelled"

    async def test_cancel_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import cancel_job

        mock_service = MagicMock()
        mock_service.cancel_job = AsyncMock(return_value=False)

        with (
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cancel_job("missing", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 404

    async def test_cancel_already_completed(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import cancel_job

        mock_service = MagicMock()
        mock_service.cancel_job = AsyncMock(return_value=False)

        mock_job = MagicMock()
        mock_job.status = "completed"

        with (
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await cancel_job("job-1", MagicMock(), AsyncMock())
            assert exc_info.value.status_code == 409


class TestListJobs:
    async def test_list_jobs(self):
        from src.api.routes.compilation import list_jobs

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.status = "completed"
        mock_job.created_at = datetime.now(UTC)
        mock_job.completed_at = datetime.now(UTC)
        mock_job.preset = "standard"
        mock_job.error_message = None

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch(
                "src.api.routes.compilation.CompilationJobManager.list_jobs",
                new_callable=AsyncMock,
                return_value=([mock_job], 1),
            ),
        ):
            request = MagicMock()
            db = AsyncMock()
            result = await list_jobs(request, db)

        assert result.total == 1
        assert len(result.jobs) == 1
        assert result.jobs[0].job_id == "job-1"
