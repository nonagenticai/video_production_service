"""POST /api/record/batch_from_artifacts must download each artifact, scaffold
a recording dir, and hand off to the existing batch recorder."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_recording_job_manager
from src.api.routes.recording import router
from tests.conftest import bind_test_tenant

PROJECT_ID = uuid4()


def _make_mock_jm():
    jm = AsyncMock()
    jm.commit = AsyncMock()
    jm.rollback = AsyncMock()
    return jm


def _make_app(tmp_path: Path, recorder_service=None) -> FastAPI:
    """Create a minimal FastAPI app with only the recording router — no DB lifespan."""
    os.environ["RECORDING_SCRIPTS_BASE_DIR"] = str(tmp_path)
    # Clear cached settings so the endpoint sees the overridden env var.
    from src.config import clear_settings_cache

    clear_settings_cache()

    app = FastAPI()
    app.include_router(router)
    app.state.recorder_service = recorder_service or AsyncMock()
    app.state.storage_service = AsyncMock()
    bind_test_tenant(app)
    return app


def test_batch_from_artifacts_downloads_and_dispatches(tmp_path: Path) -> None:
    artifact_a = uuid4()
    artifact_b = uuid4()

    async def fake_download(*, project_id, artifact_id, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"// {artifact_id}\n", encoding="utf-8")
        return dest

    fake_client = AsyncMock()
    fake_client.download_to_path = AsyncMock(side_effect=fake_download)

    fake_batch_id = "batch-" + str(uuid4())

    app = _make_app(tmp_path)
    mock_jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: mock_jm

    fake_job_ids = {"part-001": "job-1", "part-002": "job-2"}
    fake_dispatch = AsyncMock(return_value=(fake_batch_id, fake_job_ids))

    with (
        patch("src.api.routes.recording.get_artifact_client", return_value=fake_client),
        patch("src.api.routes.recording._dispatch_batch_job", fake_dispatch),
    ):
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.post(
            "/api/record/batch_from_artifacts",
            json={
                "project_id": str(PROJECT_ID),
                "parts": [
                    {"part_id": "part-001", "artifact_id": str(artifact_a)},
                    {"part_id": "part-002", "artifact_id": str(artifact_b)},
                ],
                "recording_config": None,
                "auth_config": None,
            },
            headers={"X-Tenant-Id": "Bearer t"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["batch_id"] == fake_batch_id
    assert body["job_ids"] == fake_job_ids
    assert body["scripts_dir"].startswith(str(tmp_path))
    assert body["poll_url"] == f"/api/record/batch/{body['batch_id']}"

    assert fake_client.download_to_path.await_count == 2

    written = sorted(tmp_path.rglob("*.demo.spec.ts"))
    assert len(written) == 2

    fake_dispatch.assert_awaited_once()


def test_batch_from_artifacts_rejects_traversal(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    mock_jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: mock_jm

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/record/batch_from_artifacts",
        json={
            "project_id": str(PROJECT_ID),
            "parts": [{"part_id": "../escape", "artifact_id": str(uuid4())}],
        },
        headers={"X-Tenant-Id": "Bearer t"},
    )
    assert resp.status_code == 422


def test_batch_from_artifacts_requires_at_least_one_part(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    mock_jm = _make_mock_jm()
    app.dependency_overrides[get_recording_job_manager] = lambda: mock_jm

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/api/record/batch_from_artifacts",
        json={"project_id": str(PROJECT_ID), "parts": []},
        headers={"X-Tenant-Id": "Bearer t"},
    )
    assert resp.status_code == 422
