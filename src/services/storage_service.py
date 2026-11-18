"""File storage for video-production outputs.

A thin ``(job_id, filename)`` domain wrapper over the shared
:class:`w2v_common.storage.StorageService` facade. This service keeps its own
key convention (object names are ``{job_id}/{filename}``) and its own local
*job working directories* (``get_job_dir`` / ``get_local_path``), but delegates
every object-storage operation to the facade — no more hand-rolled
``StorageBackend`` ABC + Local/MinIO backends.

Reference formats are UNCHANGED so persisted DB rows round-trip identically:
- minio backend → the object key ``{job_id}/{filename}``,
- local backend → the absolute path ``{local_path}/{job_id}/{filename}``.

One intentional exception to the async facade: :meth:`retrieve_uri_sync`. The
compiler's segment builders run on worker threads (dispatched via
``asyncio.to_thread``) and need a *synchronous* cross-bucket read. The facade is
async-only over ``miniopy-async``, whose ``aiohttp`` session is cached bound to
the event loop of its first call, so it cannot be driven from a fresh
``asyncio.run`` loop on a worker thread. For that single path we keep a small
synchronous ``minio`` read (``_get_object_sync``) — the documented sync/async
impedance boundary.

Both ``retrieve_uri`` and ``retrieve_uri_sync`` take a whole ``s3://bucket/key``
URI off a tenant-supplied compile request, i.e. an arbitrary BUCKET, so they are
gated by ``_parse_and_authorize_uri`` against ``STORAGE_CROSS_BUCKET_ALLOWLIST``
(fail-closed: own bucket only when unset).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from w2v_common.storage import StorageService as CommonStorageService

from src.config import get_settings

logger = logging.getLogger(__name__)

# Conditional sync-MinIO import: only the retrieve_uri_sync worker-thread bridge
# uses it; a local-backend deployment that never fetches an ``s3://`` URI from a
# worker thread does not need the package installed.
try:
    from minio import Minio
    from minio.error import MinioException
    from urllib3.exceptions import HTTPError as Urllib3HTTPError

    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False


class StorageService:
    """Video-production storage over the shared w2v_common facade.

    Public API is unchanged from the previous hand-rolled implementation, so no
    call site changes: every route/recorder/compiler keeps calling the same
    methods and getting the same reference strings back.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        storage_settings = self._settings.storage
        self._backend_name = storage_settings.backend.lower()
        self._storage = CommonStorageService(storage_settings)

        # Local *working* directories for intermediate files. For the local
        # backend they coincide with the object store root (objects and scratch
        # live under local_path/{job_id}); for the minio backend they are a
        # throwaway temp root (objects live in MinIO, scratch stays on disk).
        if self._backend_name == "minio":
            self._work_root = Path(tempfile.gettempdir()) / "vps-minio"
        else:
            self._work_root = Path(storage_settings.local_path)

        logger.info("Storage backend selected: %s", self._backend_name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def backend(self) -> CommonStorageService:
        """The underlying shared facade (for advanced use-cases)."""
        return self._storage

    @property
    def backend_type(self) -> str:
        """Return the backend type name ('local' or 'minio')."""
        return self._storage.backend

    @staticmethod
    def _object_name(job_id: str, filename: str) -> str:
        return f"{job_id}/{filename}"

    # ------------------------------------------------------------------
    # Lifecycle + job working directories
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize the working root and the storage backend."""
        self._work_root.mkdir(parents=True, exist_ok=True)
        await self._storage.initialize()

    def get_job_dir(self, job_id: str) -> Path:
        """Return (and create) a writable local directory for job artifacts."""
        job_dir = self._work_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    # ------------------------------------------------------------------
    # Object store operations (delegated to the facade)
    # ------------------------------------------------------------------

    async def store_bytes(self, job_id: str, data: bytes, filename: str) -> str:
        """Store raw bytes and return a storage reference."""
        return await self._storage.put_bytes(self._object_name(job_id, filename), data)

    async def store_file(self, job_id: str, file_path: Path, filename: str) -> str:
        """Store a local file (streamed, never read into memory) and return a
        storage reference. Safe when the source already IS the destination —
        the facade no-ops the copy instead of raising ``SameFileError``.
        """
        return await self._storage.put_file(self._object_name(job_id, filename), file_path)

    async def store_video(self, job_id: str, source_path: Path, part_id: str) -> str:
        """Store a video file and return its storage reference.

        Convenience wrapper around :meth:`store_file` that derives the filename
        from the ``part_id`` and the source file extension.
        """
        ext = source_path.suffix or ".mp4"
        return await self.store_file(job_id, source_path, f"{part_id}{ext}")

    async def retrieve(self, path: str) -> bytes | None:
        """Retrieve file contents by storage reference (object key or path).

        For the local backend the reference is path-confined to the storage root
        (defense in depth against a stale/tampered DB row escaping via ``..`` or
        an absolute path outside the root); for minio it is an object key.
        """
        ref = path
        if self._backend_name != "minio":
            confined = self._confine_local(path)
            if confined is None:
                return None
            ref = confined
        return await self._storage.get_bytes(ref)

    def _confine_local(self, raw: str) -> str | None:
        """Resolve ``raw`` against the local root and reject anything outside it.

        ``store_*`` only ever emits paths under ``local_path``, so a legitimate
        DB row round-trips cleanly. A path that escapes the root (absolute
        outside, or ``..`` traversal) signals a stale/tampered row — return
        ``None`` so the caller treats it as "not found" rather than serving it.
        """
        local_root = self._settings.storage.local_path
        root = Path(local_root).resolve()
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(local_root) / raw
        try:
            resolved = candidate.resolve()
        except OSError, RuntimeError:
            logger.warning("Could not resolve storage path: %s", raw)
            return None
        if not resolved.is_relative_to(root):
            logger.warning("Rejected path outside storage root: %s", raw)
            return None
        return str(resolved)

    def _allowed_buckets(self) -> frozenset[str]:
        """Buckets ``retrieve_uri{,_sync}`` may read: own bucket + configured extras.

        Fail-closed: with ``STORAGE_CROSS_BUCKET_ALLOWLIST`` unset this is just the
        service's own bucket, i.e. no cross-bucket read at all.
        """
        s = self._settings.storage
        allowed = {(s.minio_bucket or "").strip()}
        allowed.update(b.strip() for b in (s.cross_bucket_allowlist or "").split(","))
        return frozenset(b for b in allowed if b)

    def _parse_and_authorize_uri(
        self, s3_uri: str, tenant_id: uuid.UUID | None
    ) -> tuple[str, str] | None:
        """Split ``s3://bucket/key`` and refuse buckets outside the allowlist.

        This is the confinement for the ``s3://`` branch of the compiler's asset
        resolution. Its neighbours are already confined — the http(s) branch by
        ``safe_get_sync(allow_hosts=artifact_allow_hosts())`` and the bare-path
        branch by ``_guarded_local_image_path`` — and this branch was the lone
        hole: the URI carries an arbitrary BUCKET, so it is readable with the
        service account's credentials whatever the key looks like.

        Confinement is at the BUCKET layer only, and that is a limit of the data,
        not a hedge: object keys in this bucket are ``{job_id}/{filename}`` and in
        the sibling w2v buckets they are ``{job_id}/…`` / ``{book_id}/…`` — no
        component of any of them encodes a tenant, and the ``job_id -> tenant``
        mapping lives in the OWNING service's database, which the compiler's
        worker thread has no session for. So there is no key prefix that can be
        derived from ``tenant_id`` and compared. ``tenant_id`` is still threaded
        through and logged, so a refused read is attributable and so that a future
        tenant-prefixed key convention has the value already at this seam.

        Args:
            s3_uri: The ``s3://bucket/key`` URI to authorize.
            tenant_id: Requesting tenant, for the audit log. ``None`` when the
                caller has no job context.

        Returns:
            ``(bucket, key)`` when the read is permitted, otherwise ``None``.
        """
        bucket, _, key = s3_uri[len("s3://") :].partition("/")
        if not bucket or not key:
            logger.error("Malformed S3 URI: %s", s3_uri)
            return None
        allowed = self._allowed_buckets()
        if bucket not in allowed:
            logger.error(
                "Refused cross-bucket read of s3://%s/%s (tenant=%s): bucket is not in the "
                "allowlist %s. Add it to STORAGE_CROSS_BUCKET_ALLOWLIST if this read is "
                "legitimate.",
                bucket,
                key,
                tenant_id or "unknown",
                sorted(allowed),
            )
            return None
        return bucket, key

    async def retrieve_uri(self, s3_uri: str, tenant_id: uuid.UUID | None = None) -> bytes | None:
        """Download from an S3 URI (cross-bucket) or a plain storage reference.

        Args:
            s3_uri: ``s3://bucket/key`` URI, or a plain storage reference.
            tenant_id: Requesting tenant, threaded through for the audit log on a
                refused cross-bucket read. See :meth:`_parse_and_authorize_uri`.

        Returns:
            The object bytes, or ``None`` if missing, malformed, or refused.
        """
        if not s3_uri.startswith("s3://"):
            return await self.retrieve(s3_uri)
        target = self._parse_and_authorize_uri(s3_uri, tenant_id)
        if target is None:
            return None
        bucket, key = target
        return await self._storage.get_bytes(key, bucket=bucket)

    def retrieve_uri_sync(self, s3_uri: str, tenant_id: uuid.UUID | None = None) -> bytes | None:
        """Synchronous cross-bucket read for compiler worker threads.

        Single intentional sync→async boundary. Callers running on a worker
        thread (e.g. compiler segment builders dispatched via
        ``asyncio.to_thread``) must use this rather than calling ``asyncio.run``
        themselves. For an ``s3://`` URI it uses a synchronous ``minio`` client
        (the async facade's ``miniopy-async`` session is loop-bound and cannot
        be reused across event loops); a plain path is a local file read, which
        is loop-safe on a fresh loop.

        Args:
            s3_uri: ``s3://bucket/key`` URI, or a plain storage reference.
            tenant_id: Requesting tenant, threaded through for the audit log on a
                refused cross-bucket read. See :meth:`_parse_and_authorize_uri`.

        Returns:
            The object bytes, or ``None`` if missing, malformed, or refused.

        Raises:
            RuntimeError: if invoked from inside a running event loop. The caller
                is then on the asyncio thread and should ``await retrieve_uri``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # No running loop on this thread — proceed.
        else:
            raise RuntimeError(
                "StorageService.retrieve_uri_sync() called from inside a running "
                "event loop. Use 'await storage.retrieve_uri(...)' instead."
            )

        if not s3_uri.startswith("s3://"):
            return asyncio.run(self.retrieve(s3_uri))
        target = self._parse_and_authorize_uri(s3_uri, tenant_id)
        if target is None:
            return None
        bucket, key = target
        return self._get_object_sync(bucket, key)

    def _get_object_sync(self, bucket: str, key: str) -> bytes | None:
        """Synchronous MinIO GET for the worker-thread bridge (see
        :meth:`retrieve_uri_sync`). Builds a short-lived sync client from the
        service's storage settings; no event loop involved.
        """
        if not HAS_MINIO:
            logger.error(
                "Cannot retrieve s3://%s/%s — the 'minio' package is not installed "
                "(required for the compiler's synchronous cross-bucket read).",
                bucket,
                key,
            )
            return None
        s = self._settings.storage
        try:
            client = Minio(
                s.minio_endpoint,
                access_key=s.minio_access_key,
                secret_key=s.minio_secret_key,
                secure=s.minio_secure,
            )
            response = client.get_object(bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        except OSError, ConnectionError, TimeoutError, ValueError, MinioException, Urllib3HTTPError:
            logger.exception("Failed to download s3://%s/%s via MinIO", bucket, key)
            return None

    async def delete_job_dir(self, job_id: str) -> None:
        """Delete all stored artefacts for a job (objects + local scratch)."""
        await self._storage.delete_prefix(f"{job_id}/")
        work_dir = self._work_root / job_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        logger.info("Deleted all artefacts for job: %s", job_id)

    async def cleanup_job(self, job_id: str) -> None:
        """Remove all files associated with a job (alias for delete_job_dir)."""
        await self.delete_job_dir(job_id)

    async def presign_url(self, path: str, expires_in: int = 3600) -> str | None:
        """Return a presigned download URL, or None if unsupported (local)."""
        return await self._storage.presign_get(path, expires=expires_in)

    async def get_presigned_url(self, path: str, expires_seconds: int = 3600) -> str | None:
        """Return a MinIO presigned URL for the given path, or None if unsupported."""
        return await self._storage.presign_get(path, expires=expires_seconds)

    async def health_check(self) -> bool:
        """Verify backend health."""
        return await self._storage.check_health()

    def get_local_path(self, path: str) -> Path | None:
        """Return a local filesystem path for ``path`` if one exists.

        For the local backend the reference is already an absolute path. For the
        minio backend, look for a local temp copy under a job working directory
        (best-effort; used by callers that prefer a file handle over bytes).
        """
        p = Path(path)
        if p.is_absolute() and p.exists():
            return p
        if not self._work_root.exists():
            return None
        tail = path.split("/", 1)[-1] if "/" in path else path
        for job_dir in self._work_root.iterdir():
            if not job_dir.is_dir():
                continue
            candidate = job_dir / tail
            if candidate.exists():
                return candidate
        return None
