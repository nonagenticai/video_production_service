"""Unit tests for ArtifactFetcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from src.services.recording.artifact_fetcher import ArtifactBatchError, ArtifactFetcher


@pytest.mark.asyncio
async def test_materialize_downloads_each_part(tmp_path: Path) -> None:
    job_id = uuid4()
    project_id = uuid4()

    async def fake_dl(*, project_id, artifact_id, dest: Path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"// {artifact_id}\n", encoding="utf-8")
        return dest

    artifact_client = AsyncMock()
    artifact_client.download_to_path = AsyncMock(side_effect=fake_dl)

    fetcher = ArtifactFetcher(artifact_client, scripts_base_dir=tmp_path)
    scripts_dir = await fetcher.materialize(
        job_id=job_id,
        project_id=project_id,
        parts=[("part-001", uuid4()), ("part-002", uuid4())],
    )
    assert scripts_dir == tmp_path / str(job_id)
    assert (scripts_dir / "parts" / "part-001.demo.spec.ts").exists()
    assert (scripts_dir / "parts" / "part-002.demo.spec.ts").exists()


@pytest.mark.asyncio
async def test_materialize_rejects_path_traversal_part_id(tmp_path: Path) -> None:
    artifact_client = AsyncMock()
    fetcher = ArtifactFetcher(artifact_client, scripts_base_dir=tmp_path)
    with pytest.raises(ValueError, match="unsafe part_id"):
        await fetcher.materialize(
            job_id=uuid4(),
            project_id=uuid4(),
            parts=[("../escape", uuid4())],
        )


@pytest.mark.asyncio
async def test_materialize_rejects_double_dot_part_id(tmp_path: Path) -> None:
    """Defence-in-depth: even if the schema is bypassed, fetcher rejects '..'."""
    artifact_client = AsyncMock()
    fetcher = ArtifactFetcher(artifact_client, scripts_base_dir=tmp_path)
    with pytest.raises(ValueError, match="unsafe part_id"):
        await fetcher.materialize(
            job_id=uuid4(),
            project_id=uuid4(),
            parts=[("..", uuid4())],
        )


@pytest.mark.asyncio
async def test_materialize_partial_failure_raises_structured_error(tmp_path: Path) -> None:
    """One bad part must not cancel its siblings, and the caller must learn which failed."""

    bad_aid = uuid4()

    async def fake_dl(*, project_id: UUID, artifact_id: UUID, dest: Path) -> Path:
        if artifact_id == bad_aid:
            raise RuntimeError("simulated download failure")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f"// {artifact_id}\n", encoding="utf-8")
        return dest

    artifact_client = AsyncMock()
    artifact_client.download_to_path = AsyncMock(side_effect=fake_dl)

    fetcher = ArtifactFetcher(artifact_client, scripts_base_dir=tmp_path)
    parts = [
        ("part-001", uuid4()),
        ("part-002", bad_aid),
        ("part-003", uuid4()),
    ]
    with pytest.raises(ArtifactBatchError) as excinfo:
        await fetcher.materialize(
            job_id=uuid4(),
            project_id=uuid4(),
            parts=parts,
        )

    err = excinfo.value
    assert [pid for pid, _ in err.failed] == ["part-002"]
    assert set(err.succeeded) == {"part-001", "part-003"}
    # The two good downloads completed before the exception was raised — sibling
    # tasks did not get cancelled mid-flight.
    assert artifact_client.download_to_path.await_count == 3
