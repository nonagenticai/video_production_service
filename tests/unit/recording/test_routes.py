"""Tests for src/api/routes.py -- API endpoint tests with TestClient."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from w2v_common.auth import (
    clear_identity_context,
    get_correlation_id_from_context,
    get_stage_from_context,
    get_tenant_id_from_context,
    set_correlation_context,
    set_identity_context,
)

from src.api.dependencies import get_recording_job_manager
from src.api.routes import recording as recording_module
from src.api.routes.recording import router
from src.api.schemas.recording import RecordPartRequest
from tests.conftest import TENANT_U6, TENANT_U7, bind_test_tenant, make_job

# ── Helpers ────────────────────────────────────────────────────────


def _fake_create_task(coro, **kwargs):
    """Close the coroutine so it doesn't trigger 'coroutine was never awaited'."""
    coro.close()
    return MagicMock()


def _make_mock_jm(**method_overrides):
    """Create a mock JobManager with all async methods pre-wired."""
    jm = AsyncMock()
    jm.commit = AsyncMock()
    jm.rollback = AsyncMock()
    jm.get_job = AsyncMock(return_value=None)
    jm.list_jobs = AsyncMock(return_value=([], 0))
    jm.get_batch_jobs = AsyncMock(return_value=[])
    jm.mark_failed = AsyncMock()
    jm.mark_cancelled = AsyncMock()
    jm.mark_processing = AsyncMock()
    jm.mark_completed = AsyncMock()
    jm.update_progress = AsyncMock()
    jm.create_job = AsyncMock()
    for name, val in method_overrides.items():
        setattr(jm, name, val)
    return jm


def create_test_app(recorder_service=None, storage_service=None, tenant_id=TENANT_U7):
    app = FastAPI()
    app.include_router(router)
    app.state.recorder_service = recorder_service or AsyncMock()
    app.state.storage_service = storage_service or AsyncMock()
    if tenant_id is not None:
        bind_test_tenant(app, tenant_id)
    return app


def _make_client(
    recorder_service=None,
    storage_service=None,
    mock_jm=None,
):
    """Build a TestClient with mocked dependencies."""
    app = create_test_app(recorder_service, storage_service)
    if mock_jm is None:
        mock_jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: mock_jm
    client = TestClient(app, raise_server_exceptions=False)
    return client, mock_jm


# ── POST /api/record/part ──────────────────────────────────────


class TestRecordPart:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_success(self, mock_create_task):
        svc = AsyncMock()
        svc.create_job.return_value = "job-1"
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/part",
            json={"part_id": "p1", "scripts_dir": "/scripts"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["status"] == "pending"
        assert "poll_url" in body

    def test_missing_fields_422(self):
        client, _ = _make_client()
        resp = client.post("/api/record/part", json={})
        assert resp.status_code == 422


# ── POST /api/record/url ──────────────────────────────────────


class TestRecordUrl:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_success(self, mock_create_task):
        svc = AsyncMock()
        svc.create_url_job.return_value = "url-job-1"
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/url",
            json={
                "url": "https://w2v.nonagentic.ai",
                "part_id": "promo-bg",
                "auth_info": {"type": "bearer", "token": "t"},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "url-job-1"
        assert body["status"] == "pending"
        assert "poll_url" in body
        svc.create_url_job.assert_awaited_once()

    def test_non_http_url_422(self):
        client, _ = _make_client()
        resp = client.post("/api/record/url", json={"url": "ftp://x", "part_id": "p1"})
        assert resp.status_code == 422

    def test_missing_fields_422(self):
        client, _ = _make_client()
        resp = client.post("/api/record/url", json={})
        assert resp.status_code == 422


# ── POST /api/record/capture/screens ──────────────────────────


class TestCaptureScreens:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_success(self, mock_create_task):
        svc = AsyncMock()
        svc.create_screens_job.return_value = "screens-job-1"
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/capture/screens",
            json={
                "url": "https://commonpractice.nonagentic.ai/demo",
                "part_id": "demo-screens",
                "steps": [
                    {"selector": '.nav-item >> text="Home"', "capture_id": "home"},
                    {"selector": ".persona-switch"},
                    {"selector": '.nav-item >> text="Governance"', "capture_id": "gov"},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "screens-job-1"
        assert body["poll_url"] == "/api/record/jobs/screens-job-1"
        svc.create_screens_job.assert_awaited_once()

    def test_empty_steps_422(self):
        client, _ = _make_client()
        resp = client.post(
            "/api/record/capture/screens",
            json={"url": "https://x.com/demo", "part_id": "p1", "steps": []},
        )
        assert resp.status_code == 422

    def test_missing_fields_422(self):
        client, _ = _make_client()
        resp = client.post("/api/record/capture/screens", json={})
        assert resp.status_code == 422


# ── POST /api/record/batch ────────────────────────────────────


class TestRecordBatch:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    @patch("src.api.routes.recording.uuid.uuid7", return_value="batch-1")
    def test_success(self, mock_uuid, mock_create_task):
        svc = AsyncMock()
        call_count = 0

        async def _create_job(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"j{call_count}"

        svc.create_job.side_effect = _create_job
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/batch",
            json={"part_ids": ["a", "b"], "scripts_dir": "/s"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["batch_id"] == "batch-1"
        assert len(body["job_ids"]) == 2


# ── GET /api/record/jobs/{job_id} ─────────────────────────────


class TestGetJobStatus:
    def test_found(self):
        job = make_job(job_id="j1", status="processing")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"

    def test_not_found(self):
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=None))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/missing")
        assert resp.status_code == 404


# ── GET /api/record/jobs/{job_id}/result ───────────────────────


class TestGetJobResult:
    def test_completed(self):
        job = make_job(
            job_id="j1",
            status="completed",
            video_path="/v.webm",
            duration_seconds=10.0,
            file_size_bytes=5000,
            width=1920,
            height=1080,
            has_audio=True,
        )
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/result")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "completed"
        assert body["has_audio"] is True

    def test_not_completed_409(self):
        mock_jm = _make_mock_jm(
            get_job=AsyncMock(return_value=make_job(job_id="j1", status="processing"))
        )
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/result")
        assert resp.status_code == 409

    def test_not_found_404(self):
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=None))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/missing/result")
        assert resp.status_code == 404


# ── GET /api/record/jobs/{job_id}/video ────────────────────────


class TestGetJobVideo:
    def test_fallback_in_memory(self):
        """When local_path returns None and presigned URL is None, falls back to in-memory."""
        job = make_job(job_id="j1", status="completed", video_path="/v.webm")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        storage = AsyncMock()
        storage.backend_type = "local"
        storage.get_local_path = MagicMock(return_value=None)
        storage.get_presigned_url = AsyncMock(return_value=None)
        storage.retrieve = AsyncMock(return_value=b"video-bytes")
        client, _ = _make_client(storage_service=storage, mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 200
        assert resp.content == b"video-bytes"

    def test_local_file_response(self):
        """When local_path returns a real file, FileResponse is used."""
        import tempfile

        job = make_job(job_id="j1", status="completed", video_path="/v.mp4")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        storage = AsyncMock()
        storage.backend_type = "local"
        storage.get_presigned_url = AsyncMock(return_value=None)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"mp4-content")
            f.flush()
            from pathlib import Path

            storage.get_local_path = MagicMock(return_value=Path(f.name))
            client, _ = _make_client(storage_service=storage, mock_jm=mock_jm)

            resp = client.get("/api/record/jobs/j1/video")
            assert resp.status_code == 200
            assert b"mp4-content" in resp.content
            Path(f.name).unlink(missing_ok=True)

    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_minio_streams_bytes_no_presigned_redirect(self, mock_create_task):
        """MinIO backend streams the bytes through this service — no 302 to
        minio.test (which browsers can't reach). Matches the org-wide
        stream-through pattern (cf. artifact_service:download_artifact)."""
        job = make_job(job_id="j1", status="completed", video_path="s3://bucket/j1/p.mp4")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        storage = AsyncMock()
        storage.backend_type = "minio"
        storage.get_local_path = MagicMock(return_value=None)
        storage.retrieve = AsyncMock(return_value=b"streamed-video-bytes")
        # presign_url MUST NOT be called now; assert that below.
        storage.get_presigned_url = AsyncMock()
        client, _ = _make_client(storage_service=storage, mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/video", follow_redirects=False)
        assert resp.status_code == 200
        assert resp.content == b"streamed-video-bytes"
        storage.get_presigned_url.assert_not_called()

    def test_no_video_404(self):
        job = make_job(job_id="j1", status="completed")
        job.video_path = None
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 404

    def test_expired_410(self):
        job = make_job(job_id="j1", status="completed", video_path="/v.webm", expired=True)
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 410

    def test_not_found_404(self):
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=None))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/missing/video")
        assert resp.status_code == 404

    def test_video_not_on_disk_404(self):
        """When all strategies fail to find the file, return 404."""
        job = make_job(job_id="j1", status="completed", video_path="/missing.mp4")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        storage = AsyncMock()
        storage.backend_type = "local"
        storage.get_local_path = MagicMock(return_value=None)
        storage.get_presigned_url = AsyncMock(return_value=None)
        storage.retrieve = AsyncMock(return_value=None)
        client, _ = _make_client(storage_service=storage, mock_jm=mock_jm)

        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 404


# ── POST /api/record/jobs/{job_id}/cancel ─────────────────────


class TestCancelJob:
    def test_success(self):
        job = make_job(job_id="j1", status="pending")
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=job))
        svc = AsyncMock()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post("/api/record/jobs/j1/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_already_completed_409(self):
        mock_jm = _make_mock_jm(
            get_job=AsyncMock(return_value=make_job(job_id="j1", status="completed"))
        )
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.post("/api/record/jobs/j1/cancel")
        assert resp.status_code == 409

    def test_not_found_404(self):
        mock_jm = _make_mock_jm(get_job=AsyncMock(return_value=None))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.post("/api/record/jobs/missing/cancel")
        assert resp.status_code == 404


# ── Tenant scoping on the single-job routes ───────────────────
#
# Each route is asserted INDEPENDENTLY: a test that asserted two routes at once
# would still pass with one of them unguarded.

# Real uuids: ``tenant_id`` is a ``uuid`` column (migration 004) and a
# ``uuid.UUID`` in w2v-common 0.67.0, so slugs would only fail at the driver.
_OWNER = uuid.UUID("019de900-0000-7000-8000-0000000000a5")
_OTHER = uuid.UUID("019de900-0000-7000-8000-0000000000b5")


def _tenant_scoped_jm(job):
    """JobManager stand-in whose ``get_job`` honours the tenant filter, as the DB does."""
    jm = _make_mock_jm()

    async def _get_job(job_id: str, *, tenant_id: uuid.UUID):
        if job_id != job.job_id or tenant_id != job.tenant_id:
            return None
        return job

    jm.get_job = _get_job
    return jm


def _tenant_client(job, tenant_id, storage_service=None, recorder_service=None):
    """TestClient whose requests carry ``tenant_id`` on ``request.state``."""
    app = create_test_app(
        recorder_service or AsyncMock(),
        storage_service or AsyncMock(),
        tenant_id=tenant_id,
    )
    app.dependency_overrides[get_recording_job_manager] = lambda: _tenant_scoped_jm(job)
    return TestClient(app, raise_server_exceptions=False)


def _video_storage():
    storage = AsyncMock()
    storage.backend_type = "local"
    storage.get_local_path = MagicMock(return_value=None)
    storage.get_presigned_url = AsyncMock(return_value=None)
    storage.retrieve = AsyncMock(return_value=b"video-bytes")
    return storage


class TestJobRoutesAreTenantScoped:
    """Another tenant's job must read as 404 — never 403 (no enumeration oracle)."""

    def test_status_owner_gets_200(self):
        job = make_job(job_id="j1", status="processing", tenant_id=_OWNER)
        resp = _tenant_client(job, _OWNER).get("/api/record/jobs/j1")
        assert resp.status_code == 200

    def test_status_other_tenant_gets_404(self):
        job = make_job(job_id="j1", status="processing", tenant_id=_OWNER)
        resp = _tenant_client(job, _OTHER).get("/api/record/jobs/j1")
        assert resp.status_code == 404

    def test_result_owner_gets_200(self):
        job = make_job(job_id="j1", status="completed", tenant_id=_OWNER, video_path="/v.webm")
        resp = _tenant_client(job, _OWNER).get("/api/record/jobs/j1/result")
        assert resp.status_code == 200

    def test_result_other_tenant_gets_404(self):
        job = make_job(job_id="j1", status="completed", tenant_id=_OWNER, video_path="/v.webm")
        resp = _tenant_client(job, _OTHER).get("/api/record/jobs/j1/result")
        assert resp.status_code == 404

    def test_video_owner_gets_200(self):
        job = make_job(job_id="j1", status="completed", tenant_id=_OWNER, video_path="/v.webm")
        client = _tenant_client(job, _OWNER, storage_service=_video_storage())
        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 200
        assert resp.content == b"video-bytes"

    def test_video_other_tenant_gets_404(self):
        job = make_job(job_id="j1", status="completed", tenant_id=_OWNER, video_path="/v.webm")
        client = _tenant_client(job, _OTHER, storage_service=_video_storage())
        resp = client.get("/api/record/jobs/j1/video")
        assert resp.status_code == 404

    def test_cancel_owner_gets_200(self):
        job = make_job(job_id="j1", status="pending", tenant_id=_OWNER)
        client = _tenant_client(job, _OWNER, recorder_service=AsyncMock())
        resp = client.post("/api/record/jobs/j1/cancel")
        assert resp.status_code == 200

    def test_cancel_other_tenant_gets_404(self):
        """Cross-tenant WRITE: the cancel must not reach the recorder service."""
        job = make_job(job_id="j1", status="pending", tenant_id=_OWNER)
        svc = AsyncMock()
        client = _tenant_client(job, _OTHER, recorder_service=svc)
        resp = client.post("/api/record/jobs/j1/cancel")
        assert resp.status_code == 404
        svc.cancel_job.assert_not_awaited()


# ── GET /api/record/batch/{batch_id} ──────────────────────────


class TestGetBatchStatus:
    def test_success(self):
        jobs = [
            make_job(job_id="j1", status="completed", batch_id="b1"),
            make_job(job_id="j2", status="processing", batch_id="b1"),
        ]
        mock_jm = _make_mock_jm(get_batch_jobs=AsyncMock(return_value=jobs))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/batch/b1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert body["completed"] == 1
        assert body["jobs"][0]["result_url"] == "/api/record/jobs/j1/result"
        assert body["jobs"][1]["result_url"] is None

    def test_not_found(self):
        mock_jm = _make_mock_jm(get_batch_jobs=AsyncMock(return_value=[]))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/batch/missing")
        assert resp.status_code == 404


# ── GET /api/record/jobs (list) ───────────────────────────────


class TestListJobs:
    def test_pagination(self):
        mock_jm = _make_mock_jm(list_jobs=AsyncMock(return_value=([], 0)))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/jobs?offset=0&limit=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["offset"] == 0
        assert body["limit"] == 10


# ── Correlation/tenant propagation into the detached bg task ───────


class TestRecordPartCorrelationPropagation:
    """Regression guard for the create_task context-inheritance contract.

    A plain ``asyncio.create_task(coro)`` snapshots the current contextvars
    context at creation time, so the detached recording background task inherits
    the request's identity + cost-correlation ContextVars (tenant / correlation_id
    / stage). This is what makes the task's WM/IR cost events attribute to the
    run's tenant with the right correlation_id. If anyone moves this to a thread
    pool, run_in_executor, or passes a blank ``context=``, the context is dropped
    and correlation silently NULLs — this test would catch that.
    """

    async def test_bg_task_inherits_correlation_and_tenant(self):
        # ``get_tenant_id_from_context`` returns ``uuid.UUID | None`` since
        # w2v-common 0.67.0, so the spy dict is no longer str-only.
        seen: dict[str, object] = {}
        done = asyncio.Event()

        async def _spy_process_job_bg(service, job_id, tenant_id):
            seen["correlation_id"] = get_correlation_id_from_context()
            seen["stage"] = get_stage_from_context()
            seen["tenant_id"] = get_tenant_id_from_context()
            done.set()

        svc = AsyncMock()
        svc.create_job.return_value = "job-corr-1"
        jm = _make_mock_jm()

        # MagicMock (not SimpleNamespace) satisfies the Request type for the
        # static type checker (`ty`) while duck-typing the attributes the route
        # reads; a mypy-style inline ignore comment does NOT suppress `ty`.
        request = MagicMock()
        request.state.tenant_id = TENANT_U6
        request.app.state.recorder_service = svc
        body = RecordPartRequest(part_id="p1", scripts_dir="/scripts")

        # Establish the request-scoped identity + correlation context BEFORE the
        # route runs. The REAL asyncio.create_task must run (do not patch it), so
        # the scheduled task inherits this exact context snapshot.
        set_identity_context(TENANT_U6, "lic-1", "subj-1")
        set_correlation_context("run-xyz", "record_part")
        try:
            with patch.object(recording_module, "_process_job_bg", new=_spy_process_job_bg):
                resp = await recording_module.record_part(body, request, jm)
                assert resp.job_id == "job-corr-1"

                # The REAL create_task scheduled the detached bg task; yield until
                # the spy (which the task runs) signals completion. We wait on an
                # Event rather than the module-level _background_tasks set because
                # sibling tests that stub create_task leave mock objects in it.
                await asyncio.wait_for(done.wait(), timeout=5)
        finally:
            clear_identity_context()
            set_correlation_context(None, None)

        assert seen["correlation_id"] == "run-xyz"
        assert seen["stage"] == "record_part"
        assert seen["tenant_id"] == TENANT_U6
