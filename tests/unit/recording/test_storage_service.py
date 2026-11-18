"""Tests for src/services/storage_service.py — the facade-backed StorageService.

The wrapper is a thin ``(job_id, filename)`` domain layer over the shared
``w2v_common.storage.StorageService``. Local-backend tests run the real facade
against a tmp dir (true round-trips); minio-backend tests swap ``svc._storage``
for an AsyncMock to assert delegation + the object-key convention.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import StorageSettings
from src.services.storage_service import StorageService


def _local_service(local_path: str, cross_bucket_allowlist: str = "") -> StorageService:
    settings = MagicMock()
    settings.storage = StorageSettings(
        backend="local",
        local_path=local_path,
        cross_bucket_allowlist=cross_bucket_allowlist,
    )
    with patch("src.services.storage_service.get_settings", return_value=settings):
        return StorageService()


def _minio_service(cross_bucket_allowlist: str = "") -> tuple[StorageService, AsyncMock]:
    """A minio-backed StorageService with the facade swapped for an AsyncMock,
    so delegation + key convention can be asserted without a live MinIO. Returns
    ``(service, facade_mock)`` — assert on the mock (it is typed ``AsyncMock``).

    ``cross_bucket_allowlist`` defaults to empty so the fail-closed default is
    what tests see unless they opt in.
    """
    settings = MagicMock()
    settings.storage = StorageSettings(
        backend="minio",
        minio_endpoint="localhost:9000",
        minio_access_key="a",
        minio_secret_key="b",
        minio_bucket="test-bucket",
        minio_secure=False,
        cross_bucket_allowlist=cross_bucket_allowlist,
    )
    with patch("src.services.storage_service.get_settings", return_value=settings):
        svc = StorageService()
    facade = AsyncMock()
    svc._storage = facade
    return svc, facade


# ── local backend round-trips (real facade) ─────────────────────────


class TestLocalRoundTrip:
    async def test_store_bytes_key_convention_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            ref = await svc.store_bytes("job1", b"payload", "clip.bin")
            # local ref is the absolute path under local_path/{job_id}/{filename}
            assert ref == str(Path(tmp) / "job1" / "clip.bin")
            assert await svc.retrieve(ref) == b"payload"

    async def test_store_video_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            source = Path(tmp) / "source.webm"
            source.write_bytes(b"video-data")
            ref = await svc.store_video("job1", source, "part1")
            assert ref.endswith("job1/part1.webm")
            assert await svc.retrieve(ref) == b"video-data"

    async def test_store_file_in_place_no_samefile_error(self):
        """The compile pattern where the source already lives at the destination
        key must NOT raise SameFileError (the 0.39.0 facade guard)."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            job_dir = svc.get_job_dir("job1")  # == local_path/job1
            in_place = job_dir / "final.mp4"
            in_place.write_bytes(b"compiled")
            ref = await svc.store_file("job1", in_place, "final.mp4")
            assert ref == str(in_place)
            assert in_place.read_bytes() == b"compiled"

    async def test_get_job_dir_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            d = svc.get_job_dir("jobX")
            assert d.is_dir()
            assert d == Path(tmp) / "jobX"

    async def test_delete_job_dir_removes_objects_and_scratch(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            await svc.store_bytes("job1", b"a", "one.bin")
            assert (Path(tmp) / "job1" / "one.bin").exists()
            await svc.delete_job_dir("job1")
            assert not (Path(tmp) / "job1").exists()

    async def test_get_local_path_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            ref = await svc.store_bytes("job1", b"x", "f.bin")
            assert svc.get_local_path(ref) == Path(ref)
            assert svc.get_local_path(str(Path(tmp) / "job1" / "missing.bin")) is None

    async def test_retrieve_uri_plain_path_delegates_to_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            ref = await svc.store_bytes("job1", b"z", "f.bin")
            assert await svc.retrieve_uri(ref) == b"z"

    async def test_health_check_and_backend_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            assert await svc.health_check() is True
            assert svc.backend_type == "local"

    async def test_presign_local_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            await svc.initialize()
            assert await svc.presign_url("job1/f.bin") is None


# ── minio backend delegation (mocked facade) ────────────────────────


class TestMinioDelegation:
    async def test_store_bytes_uses_object_key(self):
        svc, facade = _minio_service()
        facade.put_bytes.return_value = "job1/clip.bin"
        ref = await svc.store_bytes("job1", b"d", "clip.bin")
        assert ref == "job1/clip.bin"
        facade.put_bytes.assert_awaited_once_with("job1/clip.bin", b"d")

    async def test_store_file_delegates_to_put_file(self):
        svc, facade = _minio_service()
        p = Path("/tmp/x.mp4")
        await svc.store_file("job1", p, "final.mp4")
        facade.put_file.assert_awaited_once_with("job1/final.mp4", p)

    async def test_retrieve_uri_cross_bucket_allowlisted(self):
        svc, facade = _minio_service(cross_bucket_allowlist="other-bucket")
        facade.get_bytes.return_value = b"clip"
        out = await svc.retrieve_uri("s3://other-bucket/path/to/clip.mp4")
        assert out == b"clip"
        facade.get_bytes.assert_awaited_once_with("path/to/clip.mp4", bucket="other-bucket")

    async def test_retrieve_uri_own_bucket_needs_no_allowlist(self):
        """The service's own bucket is not a *cross*-bucket read — always allowed."""
        svc, facade = _minio_service()
        facade.get_bytes.return_value = b"own"
        assert await svc.retrieve_uri("s3://test-bucket/job1/clip.mp4") == b"own"
        facade.get_bytes.assert_awaited_once_with("job1/clip.mp4", bucket="test-bucket")

    async def test_retrieve_uri_non_allowlisted_bucket_is_refused(self):
        """Fail-closed: an arbitrary bucket on a tenant-supplied URI is not read.

        The hole this closes: the URI carries a whole BUCKET, so key shape is
        irrelevant — ``s3://w2v-audiobooks/<anything>`` was readable with the
        service account's credentials and composited into a downloadable video.
        Asserting on ``get_bytes`` (not just the return value) proves the read was
        never issued, so a guard that filtered *after* fetching cannot pass.
        """
        svc, facade = _minio_service(cross_bucket_allowlist="other-bucket")
        assert await svc.retrieve_uri("s3://w2v-audiobooks/anything.mp3") is None
        facade.get_bytes.assert_not_awaited()

    async def test_retrieve_uri_refuses_everything_cross_bucket_when_unset(self):
        """Unset allowlist means no cross-bucket read at all, not 'allow all'."""
        svc, facade = _minio_service()
        assert await svc.retrieve_uri("s3://other-bucket/path/to/clip.mp4") is None
        facade.get_bytes.assert_not_awaited()

    async def test_retrieve_uri_malformed_returns_none(self):
        svc, facade = _minio_service()
        assert await svc.retrieve_uri("s3://only-bucket") is None
        facade.get_bytes.assert_not_awaited()

    async def test_delete_job_dir_calls_delete_prefix(self):
        svc, facade = _minio_service()
        await svc.delete_job_dir("job1")
        facade.delete_prefix.assert_awaited_once_with("job1/")

    async def test_presign_delegates_with_expiry(self):
        svc, facade = _minio_service()
        facade.presign_get.return_value = "https://minio/x?sig"
        assert await svc.get_presigned_url("job1/f.bin", 120) == "https://minio/x?sig"
        facade.presign_get.assert_awaited_once_with("job1/f.bin", expires=120)


# ── retrieve_uri_sync (worker-thread bridge) ────────────────────────


class TestRetrieveUriSync:
    async def test_raises_inside_running_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            with pytest.raises(RuntimeError, match="running event loop"):
                svc.retrieve_uri_sync("s3://b/k")

    def test_sync_plain_path_on_worker_thread(self):
        """From a thread with no running loop, a plain path resolves via a fresh
        asyncio.run against the (loop-safe, file-based) local facade."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            asyncio.run(svc.initialize())
            ref = asyncio.run(svc.store_bytes("job1", b"sync", "f.bin"))

            result: dict[str, bytes | None] = {}

            def worker() -> None:
                result["data"] = svc.retrieve_uri_sync(ref)

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            assert result["data"] == b"sync"

    def test_sync_s3_uses_sync_minio(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp, cross_bucket_allowlist="bucket")
            fake_resp = MagicMock()
            fake_resp.read.return_value = b"remote-clip"
            fake_client = MagicMock()
            fake_client.get_object.return_value = fake_resp
            with patch("src.services.storage_service.Minio", return_value=fake_client):
                out = svc.retrieve_uri_sync("s3://bucket/key.mp4")
            assert out == b"remote-clip"
            fake_client.get_object.assert_called_once_with("bucket", "key.mp4")
            fake_resp.close.assert_called_once()
            fake_resp.release_conn.assert_called_once()

    def test_sync_s3_malformed_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp)
            assert svc.retrieve_uri_sync("s3://onlybucket") is None

    def test_sync_s3_non_allowlisted_bucket_never_builds_a_client(self):
        """The sync bridge enforces the same allowlist — refused before any GET."""
        with tempfile.TemporaryDirectory() as tmp:
            svc = _local_service(tmp, cross_bucket_allowlist="allowed-bucket")
            with patch("src.services.storage_service.Minio") as minio_cls:
                assert svc.retrieve_uri_sync("s3://w2v-audiobooks/x.mp3") is None
            minio_cls.assert_not_called()
