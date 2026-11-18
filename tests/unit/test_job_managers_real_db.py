"""Real-database tests for RecordingJobManager and CompilationJobManager.

Uses the real PostgreSQL CI database via the conftest fixtures:
db_engine, db_session_factory, db_session.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.compilation.job_manager import (
    CompilationJobManager,
)
from src.services.compilation.job_manager import (
    JobStateConflictError as CompilationJobStateConflictError,
)
from src.services.recording.job_manager import (
    JobStateConflictError,
    RecordingJobManager,
)
from tests.conftest import TENANT_1, TENANT_2, TENANT_A, TENANT_ABSENT, TENANT_B, TENANT_EMPTY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_settings() -> MagicMock:
    """Return a fake Settings object sufficient for mark_failed / store_last_error."""
    s = MagicMock()
    s.job.max_error_length = 4000
    s.job.log_error_truncate_length = 120
    s.job.expired_batch_limit = 100
    return s


@pytest.fixture(autouse=True)
def _patch_settings():
    """Patch get_settings globally so mark_failed / store_last_error don't hit real config."""
    with patch("src.services.recording.job_manager.get_settings", _fake_settings):
        yield


# ===========================================================================
# RecordingJobManager
# ===========================================================================


class TestRecordingJobManager:
    """Tests for the instance-based RecordingJobManager against a real PostgreSQL DB."""

    # -- CRUD ---------------------------------------------------------------

    async def test_create_job_returns_recording_job(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        assert job.job_id is not None
        assert job.tenant_id == TENANT_1
        assert job.part_id == "p1"
        assert job.status == "pending"
        assert job.part_type == "browser_recording"
        assert job.progress == 0.0

    async def test_create_job_with_all_fields(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        cfg = {"width": 1280, "height": 720}
        auth = {"token": "abc123"}
        job = await mgr.create_job(
            tenant_id=TENANT_1,
            part_id="p2",
            part_type="content_recording",
            scripts_dir="/scripts",
            script_source="<html></html>",
            recording_config=cfg,
            auth_config=auth,
            webhook_url="https://hook.example.com",
            batch_id="batch-1",
            priority=5,
            batch_order=2,
            max_retries=3,
        )
        await mgr.commit()

        assert job.part_type == "content_recording"
        assert job.scripts_dir == "/scripts"
        assert job.script_source == "<html></html>"
        assert job.webhook_url == "https://hook.example.com"
        assert job.batch_id == "batch-1"
        assert job.priority == 5
        assert job.batch_order == 2
        assert job.max_retries == 3

    async def test_get_job_found(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        created = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        fetched = await mgr.get_job(created.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.job_id == created.job_id

    async def test_get_job_not_found(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        result = await mgr.get_job("nonexistent-id", tenant_id=TENANT_1)
        assert result is None

    async def test_get_job_other_tenant_returns_none(self, db_session: AsyncSession) -> None:
        """A job owned by tenant A is invisible to tenant B (app-layer guard)."""
        mgr = RecordingJobManager(db_session)
        created = await mgr.create_job(tenant_id=TENANT_A, part_id="p1")
        await mgr.commit()

        assert await mgr.get_job(created.job_id, tenant_id=TENANT_B) is None
        assert await mgr.get_job(created.job_id, tenant_id=TENANT_A) is not None

    # -- JSONB round-trip ---------------------------------------------------

    async def test_recording_config_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        config = {
            "width": 1920,
            "height": 1080,
            "fps": 30,
            "nested": {"key": [1, 2, 3]},
        }
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1", recording_config=config)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.recording_config is not None
        assert fetched.recording_config == config
        assert fetched.recording_config["nested"]["key"] == [1, 2, 3]

    async def test_auth_config_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        auth = {"username": "admin", "password": "secret", "roles": ["viewer"]}
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1", auth_config=auth)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.auth_config == auth

    # -- Status transitions -------------------------------------------------

    async def test_mark_processing(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.mark_processing(job.job_id)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.status == "processing"

    async def test_mark_queued(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.mark_queued(job.job_id)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.status == "queued"

    async def test_mark_completed_with_metadata(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        # mark_completed requires the job to be in 'processing' state.
        await mgr.mark_processing(job.job_id)
        await mgr.mark_completed(
            job.job_id,
            video_path="/output/video.mp4",
            duration_seconds=120.5,
            file_size_bytes=5_000_000,
            width=1920,
            height=1080,
            has_audio=True,
            recording_time_seconds=125.0,
        )

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.video_path == "/output/video.mp4"
        assert fetched.duration_seconds == 120.5
        assert fetched.file_size_bytes == 5_000_000
        assert fetched.width == 1920
        assert fetched.height == 1080
        assert fetched.has_audio is True
        assert fetched.recording_time_seconds == 125.0
        assert fetched.progress == 1.0
        assert fetched.progress_stage == "completed"
        assert fetched.completed_at is not None
        assert fetched.expires_at is not None

    async def test_mark_failed_stores_error(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.mark_failed(job.job_id, "Browser crashed unexpectedly")
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_message == "Browser crashed unexpectedly"
        assert fetched.last_error == "Browser crashed unexpectedly"
        assert fetched.completed_at is not None
        assert fetched.expires_at is not None

    async def test_mark_cancelled(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.mark_cancelled(job.job_id)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.status == "cancelled"
        assert fetched.completed_at is not None

    # -- Progress -----------------------------------------------------------

    async def test_update_progress(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.update_progress(job.job_id, 0.5, "recording")
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.progress == 0.5
        assert fetched.progress_stage == "recording"

    async def test_update_retry_count(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.update_retry_count(job.job_id, 2)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.retry_count == 2

    async def test_store_last_error(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.store_last_error(job.job_id, "Timeout on page load")
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.last_error == "Timeout on page load"
        # status should remain pending (store_last_error doesn't change status)
        assert fetched.status == "pending"

    # -- List / pagination --------------------------------------------------

    async def test_list_jobs_empty(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        jobs, total = await mgr.list_jobs(tenant_id=TENANT_EMPTY)
        assert jobs == []
        assert total == 0

    async def test_list_jobs_returns_correct_tenant(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.create_job(tenant_id=TENANT_1, part_id="p2")
        await mgr.create_job(tenant_id=TENANT_2, part_id="p3")
        await mgr.commit()

        jobs, total = await mgr.list_jobs(tenant_id=TENANT_1)
        assert total == 2
        assert len(jobs) == 2
        assert all(j.tenant_id == TENANT_1 for j in jobs)

    async def test_list_jobs_filter_by_status(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        j1 = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.create_job(tenant_id=TENANT_1, part_id="p2")
        await mgr.commit()

        await mgr.mark_processing(j1.job_id)
        await mgr.commit()

        jobs, total = await mgr.list_jobs(tenant_id=TENANT_1, status="processing")
        assert total == 1
        assert jobs[0].job_id == j1.job_id

    async def test_list_jobs_pagination(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        for i in range(5):
            await mgr.create_job(tenant_id=TENANT_1, part_id=f"p{i}")
        await mgr.commit()

        page1, total = await mgr.list_jobs(tenant_id=TENANT_1, offset=0, limit=2)
        assert total == 5
        assert len(page1) == 2

        page2, _ = await mgr.list_jobs(tenant_id=TENANT_1, offset=2, limit=2)
        assert len(page2) == 2

        page3, _ = await mgr.list_jobs(tenant_id=TENANT_1, offset=4, limit=2)
        assert len(page3) == 1

    async def test_list_jobs_sort_order_asc(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        j1 = await mgr.create_job(tenant_id=TENANT_1, part_id="first")
        j2 = await mgr.create_job(tenant_id=TENANT_1, part_id="second")
        await mgr.commit()

        jobs, _ = await mgr.list_jobs(tenant_id=TENANT_1, sort_order="asc")
        assert jobs[0].job_id == j1.job_id
        assert jobs[1].job_id == j2.job_id

    # -- Batch operations ---------------------------------------------------

    async def test_get_batch_jobs(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        await mgr.create_job(tenant_id=TENANT_1, part_id="p1", batch_id="b1", batch_order=1)
        await mgr.create_job(tenant_id=TENANT_1, part_id="p2", batch_id="b1", batch_order=0)
        await mgr.create_job(tenant_id=TENANT_1, part_id="p3", batch_id="b2", batch_order=0)
        await mgr.commit()

        batch_jobs = await mgr.get_batch_jobs("b1", TENANT_1)
        assert len(batch_jobs) == 2
        # Ordered by batch_order ascending
        assert batch_jobs[0].part_id == "p2"
        assert batch_jobs[1].part_id == "p1"

    async def test_get_batch_jobs_filters_by_tenant(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        await mgr.create_job(tenant_id=TENANT_1, part_id="p1", batch_id="b1", batch_order=0)
        await mgr.create_job(tenant_id=TENANT_2, part_id="p2", batch_id="b1", batch_order=0)
        await mgr.commit()

        batch_jobs = await mgr.get_batch_jobs("b1", TENANT_1)
        assert len(batch_jobs) == 1
        assert batch_jobs[0].tenant_id == TENANT_1

    async def test_cancel_batch_remaining(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        j1 = await mgr.create_job(tenant_id=TENANT_1, part_id="p1", batch_id="b1", batch_order=0)
        j2 = await mgr.create_job(tenant_id=TENANT_1, part_id="p2", batch_id="b1", batch_order=1)
        j3 = await mgr.create_job(tenant_id=TENANT_1, part_id="p3", batch_id="b1", batch_order=2)
        await mgr.commit()

        # Mark one as processing first (should NOT be cancelled)
        await mgr.mark_processing(j1.job_id)
        # Mark one as queued (should be cancelled)
        await mgr.mark_queued(j2.job_id)
        await mgr.commit()

        cancelled = await mgr.cancel_batch_remaining("b1")
        await mgr.commit()

        # j2 (queued) and j3 (pending) should be cancelled, j1 (processing) left alone
        assert cancelled == 2

        fetched_j1 = await mgr.get_job(j1.job_id, tenant_id=TENANT_1)
        assert fetched_j1 is not None
        assert fetched_j1.status == "processing"

        fetched_j2 = await mgr.get_job(j2.job_id, tenant_id=TENANT_1)
        assert fetched_j2 is not None
        assert fetched_j2.status == "cancelled"

        fetched_j3 = await mgr.get_job(j3.job_id, tenant_id=TENANT_1)
        assert fetched_j3 is not None
        assert fetched_j3.status == "cancelled"

    # -- Expiry / stale helpers ---------------------------------------------

    async def test_mark_expired(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        job = await mgr.create_job(tenant_id=TENANT_1, part_id="p1")
        await mgr.commit()

        await mgr.mark_expired(job.job_id)
        await mgr.commit()

        fetched = await mgr.get_job(job.job_id, tenant_id=TENANT_1)
        assert fetched is not None
        assert fetched.expired is True

    # -- Conflict on nonexistent / wrong state ------------------------------

    async def test_mark_processing_nonexistent_raises(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        with pytest.raises(JobStateConflictError):
            await mgr.mark_processing("nonexistent")

    async def test_mark_completed_nonexistent_raises(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        with pytest.raises(JobStateConflictError):
            await mgr.mark_completed("nonexistent", video_path="/nope.mp4")

    async def test_mark_failed_nonexistent_raises(self, db_session: AsyncSession) -> None:
        mgr = RecordingJobManager(db_session)
        with pytest.raises(JobStateConflictError):
            await mgr.mark_failed("nonexistent", "error")


# ===========================================================================
# CompilationJobManager
# ===========================================================================


class TestCompilationJobManager:
    """Tests for the static-method CompilationJobManager against a real PostgreSQL DB."""

    # -- CRUD ---------------------------------------------------------------

    async def test_create_job_returns_compilation_job(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        assert job.job_id is not None
        assert job.tenant_id == TENANT_1
        assert job.status == "pending"

    async def test_create_job_with_all_fields(self, db_session: AsyncSession) -> None:
        shot = {"parts": [{"id": "s1", "title": "Intro"}]}
        recs = [{"video_path": "/v1.mp4", "duration": 10.0}]
        theme = {"primary_color": "#ff0000", "font": "Arial"}
        edits = [{"type": "trim", "start": 0, "end": 5}]
        filters = {"brightness": 1.2}
        watermark = {"text": "DEMO", "position": "top-right"}
        subs = {"language": "en", "entries": []}

        job = await CompilationJobManager.create_job(
            db_session,
            tenant_id=TENANT_1,
            shot_list=shot,
            recordings=recs,
            narration_audio_dir="/audio",
            clips_dir="/clips",
            transition_type="crossfade",
            crossfade_duration=0.5,
            target_width=1920,
            target_height=1080,
            crf=23,
            theme_config=theme,
            edit_decisions=edits,
            preset="standard",
            webhook_url="https://hook.example.com",
            webhook_secret="s3cret",
            filter_config=filters,
            watermark_config=watermark,
            subtitles=subs,
        )
        await db_session.commit()

        assert job.transition_type == "crossfade"
        assert job.crossfade_duration == 0.5
        assert job.target_width == 1920
        assert job.target_height == 1080
        assert job.crf == 23
        assert job.preset == "standard"
        assert job.narration_audio_dir == "/audio"
        assert job.clips_dir == "/clips"
        assert job.webhook_url == "https://hook.example.com"
        assert job.webhook_secret == "s3cret"

    async def test_get_job_found(self, db_session: AsyncSession) -> None:
        created = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, created.job_id)
        assert fetched is not None
        assert fetched.job_id == created.job_id

    async def test_get_job_not_found(self, db_session: AsyncSession) -> None:
        result = await CompilationJobManager.get_job(db_session, "nonexistent-id")
        assert result is None

    # -- JSONB round-trip ---------------------------------------------------

    async def test_shot_list_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        shot = {
            "parts": [
                {"id": "s1", "title": "Intro", "nested": {"deep": True}},
                {"id": "s2", "title": "Main"},
            ]
        }
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1, shot_list=shot)
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.shot_list is not None
        assert fetched.shot_list == shot
        assert fetched.shot_list["parts"][0]["nested"]["deep"] is True

    async def test_recordings_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        recs = [
            {"video_path": "/v1.mp4", "duration": 10.0},
            {"video_path": "/v2.mp4", "duration": 20.0, "tags": ["hero"]},
        ]
        job = await CompilationJobManager.create_job(
            db_session, tenant_id=TENANT_1, recordings=recs
        )
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.recordings == recs

    async def test_theme_config_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        theme = {
            "primary_color": "#0000ff",
            "font_family": "Helvetica",
            "sizes": {"title": 48, "body": 16},
        }
        job = await CompilationJobManager.create_job(
            db_session, tenant_id=TENANT_1, theme_config=theme
        )
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.theme_config is not None
        assert fetched.theme_config == theme
        assert fetched.theme_config["sizes"]["title"] == 48

    async def test_edit_decisions_jsonb_roundtrip(self, db_session: AsyncSession) -> None:
        edits = [
            {"type": "trim", "start": 0, "end": 5},
            {"type": "speed", "factor": 2.0},
        ]
        job = await CompilationJobManager.create_job(
            db_session, tenant_id=TENANT_1, edit_decisions=edits
        )
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.edit_decisions == edits

    # -- Status transitions -------------------------------------------------

    async def test_mark_processing(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "processing"

    async def test_mark_completed_with_metadata(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()
        await CompilationJobManager.mark_processing(db_session, job.job_id)

        report = {"steps": 5, "warnings": []}
        metrics = {"psnr": 42.0, "bitrate_kbps": 5000}
        chapters = [{"title": "Intro", "start": 0.0, "end": 10.0}]

        await CompilationJobManager.mark_completed(
            db_session,
            job.job_id,
            output_path="/output/final.mp4",
            total_duration_seconds=300.5,
            total_parts=5,
            file_size_bytes=50_000_000,
            chapter_timestamps=chapters,
            compilation_time_seconds=45.2,
            thumbnail_path="/thumb.jpg",
            compilation_report=report,
            quality_metrics=metrics,
        )
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "completed"
        assert fetched.output_path == "/output/final.mp4"
        assert fetched.total_duration_seconds == 300.5
        assert fetched.total_parts == 5
        assert fetched.file_size_bytes == 50_000_000
        assert fetched.chapter_timestamps == chapters
        assert fetched.compilation_time_seconds == 45.2
        assert fetched.thumbnail_path == "/thumb.jpg"
        assert fetched.compilation_report == report
        assert fetched.quality_metrics == metrics
        assert fetched.completed_at is not None

    async def test_mark_failed_stores_error(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_failed(db_session, job.job_id, "FFmpeg exited with code 1")
        await db_session.commit()

        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "failed"
        assert fetched.error_message == "FFmpeg exited with code 1"
        assert fetched.completed_at is not None

    async def test_mark_cancelled_from_pending(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        result = await CompilationJobManager.mark_cancelled(db_session, job.job_id)
        await db_session.commit()

        assert result is True
        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "cancelled"

    async def test_mark_cancelled_from_processing(self, db_session: AsyncSession) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, job.job_id)
        await db_session.commit()

        result = await CompilationJobManager.mark_cancelled(db_session, job.job_id)
        await db_session.commit()

        assert result is True
        fetched = await CompilationJobManager.get_job(db_session, job.job_id)
        assert fetched is not None
        assert fetched.status == "cancelled"

    async def test_mark_cancelled_already_completed_returns_false(
        self, db_session: AsyncSession
    ) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, job.job_id)
        await CompilationJobManager.mark_completed(db_session, job.job_id)

        result = await CompilationJobManager.mark_cancelled(db_session, job.job_id)
        assert result is False

    async def test_mark_cancelled_already_cancelled_returns_false(
        self, db_session: AsyncSession
    ) -> None:
        job = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_cancelled(db_session, job.job_id)
        await db_session.commit()

        result = await CompilationJobManager.mark_cancelled(db_session, job.job_id)
        assert result is False

    async def test_mark_cancelled_nonexistent_returns_false(self, db_session: AsyncSession) -> None:
        result = await CompilationJobManager.mark_cancelled(db_session, "nonexistent-id")
        assert result is False

    # -- List / pagination --------------------------------------------------

    async def test_list_jobs_empty(self, db_session: AsyncSession) -> None:
        jobs, total = await CompilationJobManager.list_jobs(db_session, TENANT_EMPTY)
        assert jobs == []
        assert total == 0

    async def test_list_jobs_returns_correct_tenant(self, db_session: AsyncSession) -> None:
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_2)
        await db_session.commit()

        jobs, total = await CompilationJobManager.list_jobs(db_session, TENANT_1)
        assert total == 2
        assert len(jobs) == 2
        assert all(j.tenant_id == TENANT_1 for j in jobs)

    async def test_list_jobs_filter_by_status(self, db_session: AsyncSession) -> None:
        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, j1.job_id)
        await db_session.commit()

        jobs, total = await CompilationJobManager.list_jobs(
            db_session, TENANT_1, status="processing"
        )
        assert total == 1
        assert jobs[0].job_id == j1.job_id

    async def test_list_jobs_pagination(self, db_session: AsyncSession) -> None:
        for _ in range(5):
            await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        page1, total = await CompilationJobManager.list_jobs(
            db_session, TENANT_1, page=1, page_size=2
        )
        assert total == 5
        assert len(page1) == 2

        page2, _ = await CompilationJobManager.list_jobs(db_session, TENANT_1, page=2, page_size=2)
        assert len(page2) == 2

        page3, _ = await CompilationJobManager.list_jobs(db_session, TENANT_1, page=3, page_size=2)
        assert len(page3) == 1

    # -- Delete -------------------------------------------------------------

    async def test_delete_jobs(self, db_session: AsyncSession) -> None:
        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        j2 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        j3 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        deleted = await CompilationJobManager.delete_jobs(db_session, [j1.job_id, j2.job_id])
        await db_session.commit()

        assert deleted == 2

        # j3 should still exist
        remaining = await CompilationJobManager.get_job(db_session, j3.job_id)
        assert remaining is not None

        # j1, j2 should be gone
        assert await CompilationJobManager.get_job(db_session, j1.job_id) is None
        assert await CompilationJobManager.get_job(db_session, j2.job_id) is None

    async def test_delete_jobs_empty_list(self, db_session: AsyncSession) -> None:
        deleted = await CompilationJobManager.delete_jobs(db_session, [])
        assert deleted == 0

    async def test_delete_jobs_nonexistent_ids(self, db_session: AsyncSession) -> None:
        deleted = await CompilationJobManager.delete_jobs(db_session, ["nope-1", "nope-2"])
        assert deleted == 0

    # -- Count active -------------------------------------------------------

    async def test_count_active_jobs(self, db_session: AsyncSession) -> None:
        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)  # stays pending
        j3 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        # j1 -> processing, j3 -> processing -> completed (not active)
        await CompilationJobManager.mark_processing(db_session, j1.job_id)
        await CompilationJobManager.mark_processing(db_session, j3.job_id)
        await CompilationJobManager.mark_completed(db_session, j3.job_id)

        active = await CompilationJobManager.count_active_jobs(db_session, TENANT_1)
        # j1 (processing) + unnamed pending job = 2
        assert active == 2

    async def test_count_active_jobs_zero(self, db_session: AsyncSession) -> None:
        active = await CompilationJobManager.count_active_jobs(db_session, TENANT_ABSENT)
        assert active == 0

    # -- find_jobs_by_criteria ----------------------------------------------

    async def test_find_jobs_by_criteria_by_status(self, db_session: AsyncSession) -> None:
        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_failed(db_session, j1.job_id, "error")
        await db_session.commit()

        found = await CompilationJobManager.find_jobs_by_criteria(
            db_session, tenant_id=TENANT_1, status_filter=["failed"]
        )
        assert len(found) == 1
        assert found[0].job_id == j1.job_id

    async def test_find_jobs_by_criteria_by_job_ids(self, db_session: AsyncSession) -> None:
        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        j2 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        found = await CompilationJobManager.find_jobs_by_criteria(
            db_session, job_ids=[j1.job_id, j2.job_id]
        )
        assert len(found) == 2
        found_ids = {j.job_id for j in found}
        assert j1.job_id in found_ids
        assert j2.job_id in found_ids

    # -- find_expired_jobs --------------------------------------------------

    async def test_find_expired_jobs(self, db_session: AsyncSession) -> None:
        from datetime import UTC, datetime, timedelta

        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, j1.job_id)
        await CompilationJobManager.mark_completed(db_session, j1.job_id)

        # The job was just completed, so searching with a future cutoff should find it
        future = datetime.now(UTC) + timedelta(hours=1)
        found = await CompilationJobManager.find_expired_jobs(db_session, "completed", future)
        assert len(found) == 1
        assert found[0].job_id == j1.job_id

    async def test_find_expired_jobs_excludes_recent(self, db_session: AsyncSession) -> None:
        from datetime import UTC, datetime, timedelta

        j1 = await CompilationJobManager.create_job(db_session, tenant_id=TENANT_1)
        await db_session.commit()

        await CompilationJobManager.mark_processing(db_session, j1.job_id)
        await CompilationJobManager.mark_completed(db_session, j1.job_id)

        # Cutoff in the past should not find the just-completed job
        past = datetime.now(UTC) - timedelta(hours=1)
        found = await CompilationJobManager.find_expired_jobs(db_session, "completed", past)
        assert len(found) == 0

    # -- Conflict on nonexistent / wrong state ------------------------------

    async def test_mark_processing_nonexistent_raises(self, db_session: AsyncSession) -> None:
        with pytest.raises(CompilationJobStateConflictError):
            await CompilationJobManager.mark_processing(db_session, "nonexistent")

    async def test_mark_completed_nonexistent_raises(self, db_session: AsyncSession) -> None:
        with pytest.raises(CompilationJobStateConflictError):
            await CompilationJobManager.mark_completed(db_session, "nonexistent")

    async def test_mark_failed_nonexistent_raises(self, db_session: AsyncSession) -> None:
        with pytest.raises(CompilationJobStateConflictError):
            await CompilationJobManager.mark_failed(db_session, "nonexistent", "error")
