"""Extended tests for src/api/routes/recording.py -- content endpoint and helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from w2v_common.auth.identity_context import get_identity_headers

from src.api.dependencies import get_recording_job_manager
from src.api.routes.recording import (
    _cancel_remaining_jobs,
    _mark_job_failed_safe,
    _process_batch_sequential,
    _process_content_job_bg,
    _process_job_bg,
    _process_screens_job_bg,
    _task_done_callback,
    router,
)
from src.exceptions import SiteLoginFailedError, VideoRecorderError
from tests.conftest import TENANT_U1, bind_test_tenant, make_job

# ── Helpers ────────────────────────────────────────────────────────


def _fake_create_task(coro, **kwargs):
    coro.close()
    return MagicMock()


def _make_mock_jm(**method_overrides):
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


def _make_client(recorder_service=None, storage_service=None, mock_jm=None):
    app = FastAPI()
    app.include_router(router)
    app.state.recorder_service = recorder_service or AsyncMock()
    app.state.storage_service = storage_service or AsyncMock()
    bind_test_tenant(app)
    if mock_jm is None:
        mock_jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: mock_jm
    return TestClient(app, raise_server_exceptions=False), mock_jm


# ── POST /api/record/content ──────────────────────────────────────


class TestRecordContent:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_success(self, mock_create_task):
        svc = AsyncMock()
        svc.create_content_job.return_value = "cj-1"
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/content",
            json={"html_content": "<h1>Hello</h1>", "part_id": "content-part-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "cj-1"
        assert body["status"] == "pending"
        assert body["poll_url"] == "/api/record/jobs/cj-1"

    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    def test_with_scroll_config(self, mock_create_task):
        svc = AsyncMock()
        svc.create_content_job.return_value = "cj-2"
        mock_jm = _make_mock_jm()
        client, _ = _make_client(recorder_service=svc, mock_jm=mock_jm)

        resp = client.post(
            "/api/record/content",
            json={
                "html_content": "<p>test</p>",
                "part_id": "scroll-test",
                "scroll_config": {
                    "enabled": True,
                    "speed_px_per_second": 500,
                    "start_delay_seconds": 0.5,
                    "end_delay_seconds": 1.0,
                },
            },
        )
        assert resp.status_code == 200

    def test_missing_html_content_422(self):
        client, _ = _make_client()
        resp = client.post(
            "/api/record/content",
            json={"part_id": "p1"},
        )
        assert resp.status_code == 422

    def test_missing_part_id_422(self):
        client, _ = _make_client()
        resp = client.post(
            "/api/record/content",
            json={"html_content": "<p>hi</p>"},
        )
        assert resp.status_code == 422


# ── POST /api/record/batch (sequential mode) ──────────────────────


class TestRecordBatchSequential:
    @patch("src.api.routes.recording.asyncio.create_task", side_effect=_fake_create_task)
    @patch("src.api.routes.recording.uuid.uuid7", return_value="batch-seq")
    def test_sequential_mode(self, mock_uuid, mock_create_task):
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
            json={
                "part_ids": ["a", "b"],
                "scripts_dir": "/scripts",
                "execution_mode": "sequential",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["batch_id"] == "batch-seq"


# ── GET /api/record/batch/{batch_id} status aggregation ───────────


class TestBatchStatusAggregation:
    def test_all_completed(self):
        jobs = [
            make_job(job_id="j1", status="completed", batch_id="b1"),
            make_job(job_id="j2", status="completed", batch_id="b1"),
        ]
        mock_jm = _make_mock_jm(get_batch_jobs=AsyncMock(return_value=jobs))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/batch/b1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_all_failed(self):
        jobs = [
            make_job(job_id="j1", status="failed", batch_id="b2"),
            make_job(job_id="j2", status="cancelled", batch_id="b2"),
        ]
        mock_jm = _make_mock_jm(get_batch_jobs=AsyncMock(return_value=jobs))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/batch/b2")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"

    def test_completed_with_errors(self):
        jobs = [
            make_job(job_id="j1", status="completed", batch_id="b3"),
            make_job(job_id="j2", status="failed", batch_id="b3"),
        ]
        mock_jm = _make_mock_jm(get_batch_jobs=AsyncMock(return_value=jobs))
        client, _ = _make_client(mock_jm=mock_jm)

        resp = client.get("/api/record/batch/b3")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed_with_errors"


# ── Helper functions ──────────────────────────────────────────────


class TestTaskDoneCallback:
    def test_cancelled_task_no_error(self):
        task = MagicMock()
        task.cancelled.return_value = True
        # Should not raise
        _task_done_callback(task)
        task.exception.assert_not_called()

    def test_task_with_exception_logs(self):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")
        task.get_name.return_value = "test-task"
        # Should not raise
        _task_done_callback(task)

    def test_task_without_exception(self):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        # Should not raise
        _task_done_callback(task)


class TestMarkJobFailedSafe:
    async def test_marks_job_failed(self):
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with patch(
            "src.api.routes.recording.get_db_session",
        ) as mock_get_db:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            with patch(
                "src.api.routes.recording.JobManager",
                return_value=mock_jm,
            ):
                await _mark_job_failed_safe("j-fail", "error msg")

            mock_jm.mark_failed.assert_awaited_once_with("j-fail", "error msg")
            mock_jm.commit.assert_awaited_once()

    async def test_swallows_db_errors(self):
        with patch(
            "src.api.routes.recording.get_db_session",
            side_effect=RuntimeError("db down"),
        ):
            # Should not raise
            await _mark_job_failed_safe("j-err", "some error")


class TestCancelRemainingJobs:
    async def test_cancels_jobs_from_index(self):
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with patch(
            "src.api.routes.recording.get_db_session",
        ) as mock_get_db:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            with patch(
                "src.api.routes.recording.JobManager",
                return_value=mock_jm,
            ):
                await _cancel_remaining_jobs(["j1", "j2", "j3"], from_index=1)

            # WHICH jobs get cancelled is the whole point: j1 is the one that
            # already ran, so cancelling it would overwrite a real terminal
            # status. A count of 2 is satisfied just as well by ["j1", "j2"].
            assert [c.args[0] for c in mock_jm.mark_cancelled.await_args_list] == ["j2", "j3"]
            # An uncommitted cancel is discarded when the session closes.
            assert mock_jm.commit.await_count == 2


class TestProcessJobBg:
    async def test_success(self):
        mock_service = AsyncMock()
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            captured: dict[str, str] = {}
            mock_service.process_job.side_effect = lambda *a: captured.update(
                get_identity_headers()
            )

            await _process_job_bg(mock_service, "j1", TENANT_U1)

        mock_service.process_job.assert_awaited_once_with(mock_jm, "j1", TENANT_U1)
        # Phase D RLS: the bg task has no request, so the tenant ContextVar must
        # be bound BEFORE the work runs — an unbound GUC makes every tenant-scoped
        # UPDATE silently affect 0 rows instead of raising.
        assert captured["X-Tenant-Id"] == str(TENANT_U1)

    async def test_video_recorder_error_handled(self):
        mock_service = AsyncMock()
        mock_service.process_job.side_effect = VideoRecorderError("recording failed")
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe") as mock_fail,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_job_bg(mock_service, "j1", TENANT_U1)

        # The job is persisted as failed with the recorder's own message —
        # a job left in "processing" is what strands a run forever.
        mock_fail.assert_awaited_once()
        job_id, error = mock_fail.await_args_list[0].args
        assert job_id == "j1"
        assert "recording failed" in error

    async def test_generic_exception_handled(self):
        mock_service = AsyncMock()
        mock_service.process_job.side_effect = RuntimeError("unexpected")
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe") as mock_fail,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_job_bg(mock_service, "j1", TENANT_U1)

        # An exception the recorder never declared must still land as a failed
        # job, not escape into the bg task's done-callback.
        mock_fail.assert_awaited_once()
        job_id, error = mock_fail.await_args_list[0].args
        assert job_id == "j1"
        assert "unexpected" in error


class TestProcessScreensJobBg:
    async def test_site_login_failure_marks_the_job_failed(self):
        """A sign-in that did not work fails the JOB, with the code the orchestrator
        matches. Recording on would ship a video of the sign-in page as footage."""
        mock_service = AsyncMock()
        mock_service.process_screens_job.side_effect = SiteLoginFailedError(
            "site_login_failed: still on the sign-in page after signing in"
        )
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe") as mock_fail,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_screens_job_bg(mock_service, "s-9", TENANT_U1)

        mock_fail.assert_awaited_once()
        job_id, error = mock_fail.await_args_list[0].args
        assert job_id == "s-9"
        # ``format_error`` prefixes the class name; the orchestrator matches the
        # CODE as a substring, so the prefix must not bury it.
        assert "site_login_failed" in error


class TestProcessContentJobBg:
    async def test_success(self):
        mock_service = AsyncMock()
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_content_job_bg(mock_service, "cj1", TENANT_U1)

        # The job id and tenant must arrive intact, against a JobManager built
        # on the bg task's OWN session (the request session is already closed).
        mock_service.process_content_job.assert_awaited_once_with(mock_jm, "cj1", TENANT_U1)

    async def test_exception_handled(self):
        mock_service = AsyncMock()
        mock_service.process_content_job.side_effect = RuntimeError("fail")
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe") as mock_fail,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_content_job_bg(mock_service, "cj1", TENANT_U1)

        mock_fail.assert_awaited_once()
        job_id, error = mock_fail.await_args_list[0].args
        assert job_id == "cj1"
        assert "fail" in error


class TestProcessBatchSequential:
    async def test_sequential_processing(self):
        mock_service = AsyncMock()
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_batch_sequential(mock_service, ["j1", "j2"], TENANT_U1)

        # Every job in the batch is processed, in submission order — a count of
        # 2 is also satisfied by processing j1 twice and never touching j2.
        assert [c.args[1] for c in mock_service.process_job.await_args_list] == ["j1", "j2"]

    async def test_fail_fast_cancels_remaining(self):
        mock_service = AsyncMock()
        mock_service.process_job.side_effect = [
            None,
            RuntimeError("fail"),
        ]
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe"),
            patch("src.api.routes.recording._cancel_remaining_jobs") as mock_cancel,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_batch_sequential(
                mock_service, ["j1", "j2", "j3"], TENANT_U1, failure_policy="fail_fast"
            )

        # j1 ran, j2 raised — so exactly the tail AFTER j2 is cancelled, and j3
        # is never started. Off-by-one here either cancels the job that just
        # failed (clobbering its error) or leaves j3 queued forever.
        mock_cancel.assert_awaited_once_with(["j1", "j2", "j3"], 2)
        assert [c.args[1] for c in mock_service.process_job.await_args_list] == ["j1", "j2"]

    async def test_continue_policy_processes_all(self):
        mock_service = AsyncMock()
        mock_service.process_job.side_effect = [
            RuntimeError("fail"),
            None,
        ]
        mock_jm = _make_mock_jm()
        mock_session = AsyncMock()

        with (
            patch("src.api.routes.recording.get_db_session") as mock_get_db,
            patch("src.api.routes.recording.JobManager", return_value=mock_jm),
            patch("src.api.routes.recording._mark_job_failed_safe"),
            patch("src.api.routes.recording._cancel_remaining_jobs") as mock_cancel,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_get_db.return_value = mock_ctx

            await _process_batch_sequential(
                mock_service, ["j1", "j2"], TENANT_U1, failure_policy="continue"
            )

        # j1 failed; j2 must still run, and nothing may be cancelled.
        assert [c.args[1] for c in mock_service.process_job.await_args_list] == ["j1", "j2"]
        mock_cancel.assert_not_awaited()
