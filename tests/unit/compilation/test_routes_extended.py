"""Extended unit tests for API routes covering additional endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import TENANT_U1, TENANT_U2, TENANT_U4, TENANT_U8


def _fake_create_task(coro, **kwargs):
    """Close the coroutine so it doesn't trigger 'coroutine was never awaited'."""
    coro.close()
    return MagicMock()


def _mock_db() -> AsyncMock:
    """Create an AsyncMock DB session with synchronous SQLAlchemy methods patched."""
    db = AsyncMock()
    db.add = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    db.execute.return_value = mock_result
    return db


def _authed_request(tenant_id=TENANT_U1) -> MagicMock:
    """A request that carries a real tenant identity, like the auth middleware.

    A bare ``MagicMock()`` models an UNAUTHENTICATED caller since w2v-common
    0.67.0: ``request.state.tenant_id`` resolves to a ``MagicMock``, which
    ``parse_tenant_id`` rejects, so every handler under test 401s before it
    reaches the behaviour the test is about. The middleware writes
    ``str(tenant_id)``, so this does too — that keeps the route's own parse under
    test instead of handing it an already-parsed UUID.
    """
    request = MagicMock()
    request.state.tenant_id = str(tenant_id)
    return request


# ---------------------------------------------------------------------------
# compile_video endpoint
# ---------------------------------------------------------------------------


class TestCompileVideo:
    async def test_compile_video_basic(self):
        from src.api.routes.compilation import compile_video
        from src.api.schemas.compilation import CompileRequest, ShotListInput

        body = CompileRequest(
            shot_list=ShotListInput(parts=[], recording_order=[]),
            recordings=[],
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="new-job-123")
        mock_service.register_task = MagicMock()

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        request.state.tenant_id = str(TENANT_U4)
        request.app.state.compiler_service = mock_service

        db = _mock_db()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U4),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await compile_video(body, request, db)

        assert result.job_id == "new-job-123"
        assert result.status == "pending"
        mock_service.create_job.assert_awaited_once()
        mock_service.register_task.assert_called_once()

    async def test_compile_video_with_theme(self):
        from src.api.routes.compilation import compile_video
        from src.api.schemas.compilation import CompileRequest, ShotListInput, ThemeInput

        body = CompileRequest(
            shot_list=ShotListInput(parts=[], recording_order=[]),
            recordings=[],
            theme=ThemeInput(theme_name="corporate", primary_color="#FF0000"),
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="themed-job")
        mock_service.register_task = MagicMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await compile_video(body, MagicMock(), _mock_db())

        assert result.job_id == "themed-job"
        call_kwargs = mock_service.create_job.call_args[1]
        assert call_kwargs["theme_name"] == "corporate"

    async def test_compile_video_with_filters(self):
        from src.api.routes.compilation import compile_video
        from src.api.schemas.compilation import (
            ColorCorrectionInput,
            CompileRequest,
            ShotListInput,
            WatermarkInput,
        )

        body = CompileRequest(
            shot_list=ShotListInput(parts=[], recording_order=[]),
            recordings=[],
            color_correction=ColorCorrectionInput(brightness=1.2),
            watermark=WatermarkInput(image_path="/logo.png"),
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="filter-job")
        mock_service.register_task = MagicMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await compile_video(body, MagicMock(), _mock_db())

        assert result.job_id == "filter-job"


# ---------------------------------------------------------------------------
# generate_title_card endpoint
# ---------------------------------------------------------------------------


class TestGenerateTitleCard:
    async def test_generate_title_card(self):
        from src.api.routes.compilation import generate_title_card
        from src.api.schemas.compilation import TitleCardRequest

        body = TitleCardRequest(
            title="Test Title",
            subtitle="Subtitle",
            duration=3.0,
            width=640,
            height=360,
        )

        mock_storage = MagicMock()
        mock_storage.store_bytes = AsyncMock(return_value="/stored/title_card.mp4")

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the output file so read_bytes works
            title_path = Path(tmpdir) / "title_card.mp4"
            title_path.write_bytes(b"fake video")

            mock_gen = MagicMock()
            mock_gen.generate_video = MagicMock()

            with (
                patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
                patch(
                    "src.services.compilation.title_card_generator.TitleCardGenerator",
                    return_value=mock_gen,
                ),
                patch("src.api.routes.compilation.tempfile.TemporaryDirectory") as mock_tmpdir,
            ):
                mock_tmpdir.return_value.__enter__ = MagicMock(return_value=tmpdir)
                mock_tmpdir.return_value.__exit__ = MagicMock(return_value=False)

                result = await generate_title_card(body, request)

        assert result.duration == 3.0
        assert result.width == 640
        assert result.height == 360


# ---------------------------------------------------------------------------
# generate_edit_plan endpoint
# ---------------------------------------------------------------------------


class TestGenerateEditPlan:
    async def test_generate_edit_plan(self):
        from src.api.routes.compilation import generate_edit_plan
        from src.api.schemas.compilation import EditPlanRequest, ShotListInput

        body = EditPlanRequest(
            recordings=[],
            shot_list=ShotListInput(parts=[], recording_order=[]),
            target_duration_minutes=5.0,
        )

        mock_result = MagicMock()

        with (
            patch(
                "src.services.compilation.edit_plan_service.EditPlanService.__init__",
                return_value=None,
            ),
            patch(
                "src.services.compilation.edit_plan_service.EditPlanService.generate_edit_plan",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            result = await generate_edit_plan(body, MagicMock())

        assert result is mock_result


# ---------------------------------------------------------------------------
# generate_thumbnail endpoint
# ---------------------------------------------------------------------------


class TestGenerateThumbnail:
    async def test_generate_thumbnail_success(self):
        from src.api.routes.compilation import generate_thumbnail
        from src.api.schemas.compilation import ThumbnailRequest

        body = ThumbnailRequest(timestamp=5.0, width=640, height=360)

        mock_job = MagicMock()
        mock_job.job_id = "job-1"
        mock_job.output_path = "/output/demo.mp4"

        mock_storage = MagicMock()
        mock_storage.get_job_dir.return_value = Path("/tmp/job-1")

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        db = _mock_db()

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch("src.api.routes.compilation.FFmpegOps.extract_thumbnail"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            result = await generate_thumbnail("job-1", body, request, db)

        assert result.width == 640
        assert result.height == 360
        assert result.timestamp == 5.0

    async def test_generate_thumbnail_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import generate_thumbnail
        from src.api.schemas.compilation import ThumbnailRequest

        body = ThumbnailRequest(timestamp=5.0)

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_thumbnail("missing", body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404

    async def test_generate_thumbnail_no_video(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import generate_thumbnail
        from src.api.schemas.compilation import ThumbnailRequest

        body = ThumbnailRequest(timestamp=5.0)
        mock_job = MagicMock()
        mock_job.output_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_thumbnail("job-1", body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# get_job_thumbnail (with data)
# ---------------------------------------------------------------------------


class TestGetJobThumbnailFound:
    async def test_thumbnail_found(self):
        from src.api.routes.compilation import get_job_thumbnail

        mock_job = MagicMock()
        mock_job.thumbnail_path = "/thumbs/thumb.jpg"

        mock_storage = MagicMock()
        mock_storage.retrieve = AsyncMock(return_value=b"image-data")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
        ):
            result = await get_job_thumbnail("job-1", MagicMock(), _mock_db())

        assert result.body == b"image-data"
        assert result.media_type == "image/jpeg"

    async def test_thumbnail_file_missing(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_thumbnail

        mock_job = MagicMock()
        mock_job.thumbnail_path = "/thumbs/thumb.jpg"

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
                await get_job_thumbnail("job-1", MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404

    async def test_thumbnail_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_job_thumbnail

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_job_thumbnail("missing", MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# create_custom_preset endpoint
# ---------------------------------------------------------------------------


class TestCreateCustomPreset:
    async def test_create_custom_preset_success(self):
        from src.api.routes.compilation import create_custom_preset
        from src.api.schemas.compilation import CustomPresetCreateRequest

        body = CustomPresetCreateRequest(
            name="my-preset",
            description="A custom preset",
            target_width=1280,
            target_height=720,
            crf=23,
        )

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        db = _mock_db()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.services.compilation.presets.get_preset", return_value=None),
        ):
            result = await create_custom_preset(body, request, db)

        assert result.name == "my-preset"
        assert result.target_width == 1280
        db.add.assert_called_once()
        db.commit.assert_awaited_once()

    async def test_create_custom_preset_invalid_crf(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import create_custom_preset
        from src.api.schemas.compilation import CustomPresetCreateRequest

        body = CustomPresetCreateRequest(
            name="bad-preset",
            crf=99,
        )

        with patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1):
            with pytest.raises(HTTPException) as exc_info:
                await create_custom_preset(body, _authed_request(), _mock_db())
            assert exc_info.value.status_code == 400

    async def test_create_custom_preset_unknown_base(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import create_custom_preset
        from src.api.schemas.compilation import CustomPresetCreateRequest

        body = CustomPresetCreateRequest(
            name="extending-preset",
            extends="nonexistent",
        )

        with patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1):
            with pytest.raises(HTTPException) as exc_info:
                await create_custom_preset(body, _authed_request(), _mock_db())
            assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# list_custom_presets endpoint
# ---------------------------------------------------------------------------


class TestListCustomPresets:
    async def test_list_custom_presets(self):
        from src.api.routes.compilation import list_custom_presets

        mock_preset = MagicMock()
        mock_preset.preset_id = "p1"
        mock_preset.name = "my-preset"
        mock_preset.description = "desc"
        mock_preset.extends = None
        mock_preset.target_width = 1920
        mock_preset.target_height = 1080
        mock_preset.crf = 23
        mock_preset.fps = None
        mock_preset.transition_type = None
        mock_preset.crossfade_duration = None
        mock_preset.created_at = None

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_preset]

        db = _mock_db()
        db.execute = AsyncMock(return_value=mock_result)

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        with patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1):
            result = await list_custom_presets(request, db)

        assert len(result.presets) == 1
        assert result.presets[0].name == "my-preset"


# ---------------------------------------------------------------------------
# delete_custom_preset endpoint
# ---------------------------------------------------------------------------


class TestDeleteCustomPreset:
    async def test_delete_custom_preset_success(self):
        from src.api.routes.compilation import delete_custom_preset

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock()

        db = _mock_db()
        db.execute = AsyncMock(return_value=mock_result)

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        with patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1):
            result = await delete_custom_preset("p1", request, db)

        assert result["deleted"] is True

    async def test_delete_custom_preset_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import delete_custom_preset

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        db = _mock_db()
        db.execute = AsyncMock(return_value=mock_result)

        request = MagicMock()

        request.state.tenant_id = str(TENANT_U4)
        with patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1):
            with pytest.raises(HTTPException) as exc_info:
                await delete_custom_preset("missing", request, db)
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# create_template endpoint
# ---------------------------------------------------------------------------


class TestCreateTemplate:
    async def test_create_template_success(self):
        from src.api.routes.compilation import create_template
        from src.api.schemas.compilation import TemplateCreateRequest

        body = TemplateCreateRequest(
            name="my-template",
            description="A template",
        )

        mock_template = MagicMock()
        mock_template.template_id = "tmpl-1"
        mock_template.name = "my-template"
        mock_template.description = "A template"
        mock_template.preset_name = None
        mock_template.theme_name = None
        mock_template.theme_config = None
        mock_template.music_config = None
        mock_template.title_card_config = None
        mock_template.transition_type = None
        mock_template.crossfade_duration = None

        db = _mock_db()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch(
                "src.services.compilation.template_service.TemplateService.create_template",
                new_callable=AsyncMock,
                return_value=mock_template,
            ),
        ):
            result = await create_template(body, _authed_request(), db)

        assert result.id == "tmpl-1"
        assert result.name == "my-template"


# ---------------------------------------------------------------------------
# list_templates endpoint
# ---------------------------------------------------------------------------


class TestListTemplates:
    async def test_list_templates(self):
        from src.api.routes.compilation import list_templates

        mock_template = MagicMock()
        mock_template.template_id = "tmpl-1"
        mock_template.name = "my-template"
        mock_template.description = "desc"
        mock_template.preset_name = "standard"
        mock_template.theme_name = None
        mock_template.theme_config = None
        mock_template.music_config = None
        mock_template.title_card_config = None
        mock_template.transition_type = None
        mock_template.crossfade_duration = None
        mock_template.is_builtin = "false"
        mock_template.created_at = None

        db = _mock_db()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch(
                "src.services.compilation.template_service.TemplateService.list_templates",
                new_callable=AsyncMock,
                return_value=[mock_template],
            ),
        ):
            result = await list_templates(_authed_request(), db)

        assert len(result.templates) == 1
        assert result.templates[0].name == "my-template"


# ---------------------------------------------------------------------------
# delete_template endpoint
# ---------------------------------------------------------------------------


class TestDeleteTemplate:
    async def test_delete_template_success(self):
        from src.api.routes.compilation import delete_template

        db = _mock_db()
        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch(
                "src.services.compilation.template_service.TemplateService.delete_template",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await delete_template("tmpl-1", _authed_request(), db)

        assert result["deleted"] is True

    async def test_delete_template_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import delete_template

        db = _mock_db()
        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch(
                "src.services.compilation.template_service.TemplateService.delete_template",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await delete_template("missing", _authed_request(), db)
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# create_findings_video endpoint
# ---------------------------------------------------------------------------


class TestCreateFindingsVideo:
    async def test_create_findings_video(self):
        from src.api.routes.compilation import create_findings_video
        from src.api.schemas.compilation import FindingsVideoRequest

        body = FindingsVideoRequest(
            findings=[],
            title="Security Findings",
            subtitle="Q1 2026",
        )

        mock_service = MagicMock()
        mock_service.create_job = AsyncMock(return_value="findings-job")
        mock_service.register_task = MagicMock()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
            patch(
                "src.services.compilation.findings_video_service.FindingsVideoService.build_shot_list",
                return_value={"parts": [], "recording_order": []},
            ),
        ):
            result = await create_findings_video(body, MagicMock(), _mock_db())

        assert result.job_id == "findings-job"
        assert result.status == "pending"


# ---------------------------------------------------------------------------
# batch_compile endpoint
# ---------------------------------------------------------------------------


class TestBatchCompile:
    async def test_batch_compile(self):
        from src.api.routes.compilation import batch_compile
        from src.api.schemas.compilation import BatchCompileRequest

        body = BatchCompileRequest(
            configs=[],
            max_concurrent=2,
        )

        mock_service = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "pending"
        mock_batch.total_jobs = 0
        mock_batch.completed_jobs = 0
        mock_batch.failed_jobs = 0
        mock_batch.job_ids = []

        db = _mock_db()

        with (
            patch("src.api.routes.compilation._get_tenant_id", return_value=TENANT_U1),
            patch("src.api.routes.compilation._get_compiler_service", return_value=mock_service),
            patch("src.api.routes.compilation.get_db_session"),
            patch(
                "src.services.compilation.batch_service.BatchService.create_batch",
                new_callable=AsyncMock,
                return_value="batch-1",
            ),
            patch(
                "src.services.compilation.batch_service.BatchService.get_batch_status",
                new_callable=AsyncMock,
                return_value=mock_batch,
            ),
            patch("src.api.routes.compilation.asyncio.create_task", side_effect=_fake_create_task),
        ):
            result = await batch_compile(body, MagicMock(), db)

        assert result.batch_id == "batch-1"
        assert result.status == "pending"


# ---------------------------------------------------------------------------
# _process_batch_bg background wrapper
# ---------------------------------------------------------------------------


class TestProcessBatchBg:
    async def test_calls_process_batch_with_correct_args(self):
        """_process_batch_bg should open a DB session and delegate to batch_svc.process_batch."""
        from src.api.routes.compilation import _process_batch_bg

        mock_db = _mock_db()
        mock_batch_svc = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.api.routes.compilation.get_db_session", return_value=mock_ctx):
            await _process_batch_bg(
                mock_batch_svc, batch_id="b-42", tenant_id=TENANT_U8, max_concurrent=5
            )

        mock_batch_svc.process_batch.assert_awaited_once_with(mock_db, "b-42", TENANT_U8, 5)

    async def test_logs_exception_without_raising(self):
        """_process_batch_bg should swallow and log exceptions from process_batch."""
        from src.api.routes.compilation import _process_batch_bg

        mock_db = _mock_db()
        mock_batch_svc = AsyncMock()
        mock_batch_svc.process_batch = AsyncMock(side_effect=RuntimeError("db down"))

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_db)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.api.routes.compilation.get_db_session", return_value=mock_ctx):
            # Should not raise — the exception is caught and logged
            await _process_batch_bg(
                mock_batch_svc, batch_id="b-err", tenant_id=TENANT_U2, max_concurrent=3
            )

        mock_batch_svc.process_batch.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_batch_status endpoint
# ---------------------------------------------------------------------------


class TestGetBatchStatus:
    async def test_get_batch_status_found(self):
        from src.api.routes.compilation import get_batch_status

        mock_batch = MagicMock()
        mock_batch.batch_id = "batch-1"
        mock_batch.status = "completed"
        mock_batch.total_jobs = 3
        mock_batch.completed_jobs = 3
        mock_batch.failed_jobs = 0
        mock_batch.job_ids = ["j1", "j2", "j3"]
        mock_batch.errors = None
        mock_batch.created_at = None
        mock_batch.completed_at = None

        with patch(
            "src.services.compilation.batch_service.BatchService.get_batch_status",
            new_callable=AsyncMock,
            return_value=mock_batch,
        ):
            result = await get_batch_status("batch-1", _mock_db())

        assert result.batch_id == "batch-1"
        assert result.total_jobs == 3

    async def test_get_batch_status_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import get_batch_status

        with patch(
            "src.services.compilation.batch_service.BatchService.get_batch_status",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_batch_status("missing", _mock_db())
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# audio_export endpoint
# ---------------------------------------------------------------------------


class TestAudioExport:
    async def test_audio_export_success(self):
        from src.api.routes.compilation import audio_export
        from src.api.schemas.compilation import AudioExportRequest

        body = AudioExportRequest(job_id="job-1", format="mp3", bitrate="192k")

        mock_job = MagicMock()
        mock_job.output_path = "/output/demo.mp4"

        mock_storage = MagicMock()
        mock_storage.get_job_dir.return_value = Path("/tmp/job-1")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch("src.api.routes.compilation.FFmpegOps.extract_audio"),
            patch("src.api.routes.compilation.FFmpegOps.probe_duration", return_value=120.0),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024000)),
        ):
            result = await audio_export(body, MagicMock(), _mock_db())

        assert result.format == "mp3"
        assert result.duration_seconds == 120.0
        assert result.file_size_bytes == 1024000

    async def test_audio_export_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import audio_export
        from src.api.schemas.compilation import AudioExportRequest

        body = AudioExportRequest(job_id="missing")

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await audio_export(body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404

    async def test_audio_export_no_video(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import audio_export
        from src.api.schemas.compilation import AudioExportRequest

        body = AudioExportRequest(job_id="job-1")
        mock_job = MagicMock()
        mock_job.output_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await audio_export(body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# generate_preview endpoint
# ---------------------------------------------------------------------------


class TestGeneratePreview:
    async def test_generate_preview_success(self):
        from src.api.routes.compilation import generate_preview
        from src.api.schemas.compilation import PreviewRequest

        body = PreviewRequest(
            job_id="job-1",
            duration_seconds=10.0,
            width=640,
            height=360,
        )

        mock_job = MagicMock()
        mock_job.output_path = "/output/demo.mp4"

        mock_storage = MagicMock()
        mock_storage.get_job_dir.return_value = Path("/tmp/job-1")

        with (
            patch(
                "src.api.routes.compilation.CompilationJobManager.get_job",
                new_callable=AsyncMock,
                return_value=mock_job,
            ),
            patch("src.api.routes.compilation._get_storage", return_value=mock_storage),
            patch("src.api.routes.compilation.FFmpegOps.generate_preview"),
            patch("src.api.routes.compilation.FFmpegOps.probe_duration", return_value=10.0),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat", return_value=MagicMock(st_size=512000)),
        ):
            result = await generate_preview(body, MagicMock(), _mock_db())

        assert result.duration_seconds == 10.0
        assert result.width == 640
        assert result.file_size_bytes == 512000

    async def test_generate_preview_no_job_id_or_recordings(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import generate_preview
        from src.api.schemas.compilation import PreviewRequest

        body = PreviewRequest()

        with pytest.raises(HTTPException) as exc_info:
            await generate_preview(body, MagicMock(), _mock_db())
        assert exc_info.value.status_code == 400

    async def test_generate_preview_job_not_found(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import generate_preview
        from src.api.schemas.compilation import PreviewRequest

        body = PreviewRequest(job_id="missing")

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_preview(body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404

    async def test_generate_preview_no_video(self):
        from fastapi import HTTPException

        from src.api.routes.compilation import generate_preview
        from src.api.schemas.compilation import PreviewRequest

        body = PreviewRequest(job_id="job-1")
        mock_job = MagicMock()
        mock_job.output_path = None

        with patch(
            "src.api.routes.compilation.CompilationJobManager.get_job",
            new_callable=AsyncMock,
            return_value=mock_job,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_preview(body, MagicMock(), _mock_db())
            assert exc_info.value.status_code == 404
