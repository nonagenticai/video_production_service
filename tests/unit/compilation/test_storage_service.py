"""Storage-service tests specific to the compiler: path confinement (security)
and the sync→async worker-thread boundary (the exact ``asyncio.to_thread``
pattern ``_segment_builders.py`` uses).

Local round-trips + facade delegation live in
``tests/unit/recording/test_storage_service.py``; the backend internals are now
the shared ``w2v_common`` facade and are tested there.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import StorageSettings
from src.services.storage_service import StorageService
from tests.conftest import TENANT_U3


def _local_service(local_path: str, cross_bucket_allowlist: str = "") -> StorageService:
    settings = MagicMock()
    settings.storage = StorageSettings(
        backend="local",
        local_path=local_path,
        cross_bucket_allowlist=cross_bucket_allowlist,
    )
    with patch("src.services.storage_service.get_settings", return_value=settings):
        return StorageService()


class TestLocalBackendPathConfinement:
    """Defense-in-depth: ``retrieve`` must refuse paths outside the storage root
    even though the underlying facade would happily read an absolute path."""

    async def test_rejects_absolute_outside_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            root.mkdir()
            svc = _local_service(str(root))
            await svc.initialize()
            outside = Path(tmpdir) / "secret.txt"
            outside.write_bytes(b"sensitive")
            assert await svc.retrieve(str(outside)) is None
            assert outside.read_bytes() == b"sensitive"

    async def test_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            root.mkdir()
            svc = _local_service(str(root))
            await svc.initialize()
            outside = Path(tmpdir) / "secret.txt"
            outside.write_bytes(b"sensitive")
            assert await svc.retrieve("../secret.txt") is None
            assert outside.read_bytes() == b"sensitive"


class TestRetrieveUriSyncBoundary:
    """The single guarded sync→async boundary (see test_no_bare_asyncio_run)."""

    async def test_raises_when_called_from_running_loop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir)
            await svc.initialize()
            with pytest.raises(RuntimeError, match="running event loop"):
                svc.retrieve_uri_sync("s3://anything/x")

    async def test_to_thread_plain_path_returns_bytes(self):
        """Production pattern: an async caller offloads the sync wrapper onto a
        worker thread via ``asyncio.to_thread`` (as ``_segment_builders`` does).
        A plain local ref resolves through a fresh loop on that thread."""
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir)
            await svc.initialize()
            ref = await svc.store_bytes("job1", b"hello-world", "asset.bin")

            asyncio.get_running_loop()  # we are inside a running loop
            data = await asyncio.to_thread(svc.retrieve_uri_sync, ref)
            assert data == b"hello-world"

    async def test_to_thread_s3_uri_uses_sync_minio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir, cross_bucket_allowlist="bucket")
            fake_resp = MagicMock()
            fake_resp.read.return_value = b"remote"
            fake_client = MagicMock()
            fake_client.get_object.return_value = fake_resp
            with patch("src.services.storage_service.Minio", return_value=fake_client):
                data = await asyncio.to_thread(svc.retrieve_uri_sync, "s3://bucket/key.mp4")
            assert data == b"remote"
            fake_client.get_object.assert_called_once_with("bucket", "key.mp4")


class TestCrossBucketAllowlist:
    """The ``s3://`` branch of the compiler's asset resolution was the lone
    unconfined sink: its http(s) neighbour goes through ``safe_get_sync`` and its
    bare-path neighbour through ``_guarded_local_image_path``, but an ``s3://``
    ref named an arbitrary bucket that the service account could read."""

    async def test_non_allowlisted_bucket_is_refused_on_the_worker_thread(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir, cross_bucket_allowlist="allowed-bucket")
            with patch("src.services.storage_service.Minio") as minio_cls:
                data = await asyncio.to_thread(
                    svc.retrieve_uri_sync, "s3://w2v-audiobooks/book/ch1.mp3", TENANT_U3
                )
            assert data is None
            minio_cls.assert_not_called()

    async def test_allowlisted_bucket_still_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir, cross_bucket_allowlist="allowed-bucket,other")
            fake_resp = MagicMock()
            fake_resp.read.return_value = b"ok"
            fake_client = MagicMock()
            fake_client.get_object.return_value = fake_resp
            with patch("src.services.storage_service.Minio", return_value=fake_client):
                data = await asyncio.to_thread(
                    svc.retrieve_uri_sync, "s3://allowed-bucket/job1/clip.mp4", TENANT_U3
                )
            assert data == b"ok"
            fake_client.get_object.assert_called_once_with("allowed-bucket", "job1/clip.mp4")


class TestGetObjectSyncDegradesOnMinioFailure:
    """The MinIO family of exceptions (``minio.error.S3Error`` /
    ``MinioException``, and urllib3's connection-level ``HTTPError``) must
    degrade to ``None`` — a caller thread's caller (the compiler's segment
    builders) treats ``None`` as a skipped segment, not a hard-failed compile.
    Regression for the vacuous ``except OSError, ConnectionError, TimeoutError,
    ValueError`` that could never actually catch what minio raises."""

    async def test_s3_error_degrades_to_none(self):
        from minio.error import S3Error

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir, cross_bucket_allowlist="bucket")
            fake_client = MagicMock()
            fake_client.get_object.side_effect = S3Error(
                code="NoSuchKey",
                message="not found",
                resource="/bucket/key.mp4",
                request_id="req-1",
                host_id="host-1",
                response=MagicMock(),
                bucket_name="bucket",
                object_name="key.mp4",
            )
            with patch("src.services.storage_service.Minio", return_value=fake_client):
                data = await asyncio.to_thread(svc.retrieve_uri_sync, "s3://bucket/key.mp4")
            assert data is None

    async def test_urllib3_http_error_degrades_to_none(self):
        from urllib3.exceptions import MaxRetryError

        with tempfile.TemporaryDirectory() as tmpdir:
            svc = _local_service(tmpdir, cross_bucket_allowlist="bucket")
            fake_client = MagicMock()
            fake_client.get_object.side_effect = MaxRetryError(
                pool=MagicMock(), url="/bucket/key.mp4"
            )
            with patch("src.services.storage_service.Minio", return_value=fake_client):
                data = await asyncio.to_thread(svc.retrieve_uri_sync, "s3://bucket/key.mp4")
            assert data is None
