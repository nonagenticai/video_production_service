"""Root test configuration and shared fixtures for video_production_service.

Merges fixtures from both video_recorder_service and video_compiler_service.
"""

from __future__ import annotations

import os

# Set test environment variables before any imports
os.environ.setdefault("ENVIRONMENT", "test")
# Port 5476, user ci_test, db video_production_service_ci — so conftest,
# scripts/ci-local.sh and docker-compose.yml all agree on one local test DB.
# Never default to 5432: that resolves to whatever postgres the dev happens to
# be running, which answers the TCP connect and THEN fails auth — disguising a
# config error as a connection error.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5476/video_production_service_ci",  # pragma: allowlist secret
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6398/1")
os.environ.setdefault("AUTH_ENABLED", "false")
os.environ.setdefault("STORAGE_BACKEND", "local")
# Required field on StorageSettings (no localhost fallback in w2v_common 0.15.0+).
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")
os.environ.setdefault("API_DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from w2v_common.cache.redis_client import RedisClient
from w2v_common.testing import inject_fakeredis

from src.database.models import RecordingJob

# Capture the test DB URL BEFORE the autouse _clear_settings fixture
# monkeypatches DATABASE_URL to its in-test default. CI sets DATABASE_URL
# to its video_production_service_ci postgres; locally falls back to the
# default set above (developer must run the compose test postgres on :5476 to
# execute db_engine tests).
_TEST_DATABASE_URL = os.environ["DATABASE_URL"]

# ---------------------------------------------------------------------------
# Tenant ids (migration 004: all 7 ``tenant_id`` columns are native ``uuid``)
# ---------------------------------------------------------------------------
# Every tenant value that reaches Postgres must be a real uuid — a readable slug
# like "t1" now raises InvalidTextRepresentation on INSERT. These are the shared
# constants for the real-database tests. They are DISTINCT on purpose: the
# isolation tests prove tenant A cannot see tenant B's rows, so collapsing them
# into one value would silently make those tests vacuous.
#
# They are ``uuid.UUID`` OBJECTS, not strings: w2v-common 0.67.0 made tenant a
# ``uuid.UUID`` end to end, so a str here would only be caught at the driver.
TENANT_1 = uuid.UUID("019de900-0000-7000-8000-000000000001")
TENANT_2 = uuid.UUID("019de900-0000-7000-8000-000000000002")
TENANT_A = uuid.UUID("019de900-0000-7000-8000-00000000000a")
TENANT_B = uuid.UUID("019de900-0000-7000-8000-00000000000b")
# Tenants that are never seeded — used to assert an empty result set.
TENANT_EMPTY = uuid.UUID("019de900-0000-7000-8000-0000000000e0")
TENANT_ABSENT = uuid.UUID("019de900-0000-7000-8000-0000000000f0")

# Mock-only tenants for the unit tests. These replace the readable slugs ("t1",
# "tenant-1", "my-tenant", …) those tests used to pass. Slugs never reached
# Postgres there (the session is a mock), so nothing caught them — but tenant is
# a ``uuid.UUID`` end to end since w2v-common 0.67.0, so a slug is now simply the
# wrong type. One distinct uuid PER distinct slug, so any test that relied on two
# tenants differing still sees them differ.
TENANT_U1 = uuid.UUID("019de900-0000-7000-8000-0000000000c1")
TENANT_U2 = uuid.UUID("019de900-0000-7000-8000-0000000000c2")
TENANT_U3 = uuid.UUID("019de900-0000-7000-8000-0000000000c3")
TENANT_U4 = uuid.UUID("019de900-0000-7000-8000-0000000000c4")
TENANT_U5 = uuid.UUID("019de900-0000-7000-8000-0000000000c5")
TENANT_U6 = uuid.UUID("019de900-0000-7000-8000-0000000000c6")
TENANT_U7 = uuid.UUID("019de900-0000-7000-8000-0000000000c7")
TENANT_U8 = uuid.UUID("019de900-0000-7000-8000-0000000000c8")


# ── Recording domain fixtures ──────────────────────────────────────────


@pytest.fixture
def mock_settings():
    """Create mock settings with sensible defaults for the recording domain."""
    settings = MagicMock()
    settings.database.url = "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5476/video_production_service_ci"  # pragma: allowlist secret
    settings.redis.url = "redis://localhost:6398/1"
    settings.auth.enabled = False
    settings.storage.backend = "local"
    settings.storage.local_path = "/tmp/test-recordings"
    settings.storage.retention_hours = 168
    settings.storage.cleanup_interval_seconds = 3600
    settings.recording.viewport_width = 1920
    settings.recording.viewport_height = 1080
    settings.recording.timeout_ms = 60000
    settings.recording.timeout_buffer_s = 30
    settings.recording.min_video_size_bytes = 10000
    settings.recording.slow_mo_ms = 0
    settings.recording.color_scheme = "light"
    settings.recording.headless = True
    settings.recording.max_concurrent_recordings = 3
    settings.recording.max_retries = 2
    settings.recording.retry_base_delay_s = 0.1  # Fast retries for tests
    settings.recording.retryable_error_patterns = [
        "timed out",
        "browser disconnected",
        "target closed",
    ]
    settings.api.host = "0.0.0.0"
    settings.api.port = 6308
    settings.api.debug = True
    settings.api.rate_limit_requests = 60
    settings.api.rate_limit_window_seconds = 60
    settings.api.cors_origins = ["*"]
    settings.api.cors_allow_credentials = True
    settings.api.cors_allow_methods = ["*"]
    settings.api.cors_allow_headers = ["*"]
    settings.recording.output_format = "mp4"
    settings.recording.default_quality_preset = "standard"
    settings.recording.error_output_max_length = 500
    settings.recording.cancel_wait_timeout_s = 5.0
    settings.recording.scripts_base_dir = "/app/scripts"
    settings.ffmpeg.timeout = 300
    settings.ffmpeg.preset = "medium"
    settings.ffmpeg.ffprobe_timeout = 10
    settings.ffmpeg.default_crf = 23
    settings.ffmpeg.error_truncate_length = 500
    settings.job.max_error_length = 4000
    settings.job.log_error_truncate_length = 120
    settings.job.expired_batch_limit = 100
    settings.webhook.max_retries = 3
    settings.webhook.backoff_base = 2.0
    settings.webhook.timeout = 10.0
    settings.auth.cache_maxsize = 1000
    settings.auth.cache_ttl = 300
    settings.auth.introspect_timeout = 2.0
    settings.api.max_page_size = 100
    settings.storage.stale_job_timeout_seconds = 600
    settings.storage.queued_job_timeout_seconds = 7200
    settings.logging.level = "WARNING"
    settings.app_name = "Video Production Service"
    settings.app_version = "0.1.0"
    settings.environment = "test"
    settings.is_production.return_value = False
    settings.is_development.return_value = False
    return settings


def bind_test_tenant(app, tenant_id: uuid.UUID = TENANT_U7) -> None:
    """Stamp ``request.state.tenant_id`` the way the real auth middleware does.

    ``w2v_common.auth.middleware`` is the ONLY producer of that attribute, and it
    always writes ``str(tenant_id)``. Since w2v-common 0.67.0 the routes refuse a
    request that carries no tenant (401) instead of inventing ``"anonymous"``, so
    a bare ``FastAPI()`` test app now models an UNAUTHENTICATED caller. Test apps
    that mean to model an authenticated one must say so — here — and the
    ``str(...)`` is deliberate: it keeps the route's own parse under test.
    """

    @app.middleware("http")
    async def _bind_tenant(request, call_next):  # pragma: no cover - trivial
        request.state.tenant_id = str(tenant_id)
        return await call_next(request)


def make_job(
    job_id: str | None = None,
    tenant_id: uuid.UUID = TENANT_U7,
    part_id: str = "test-part",
    status: str = "pending",
    **kwargs,
) -> RecordingJob:
    """Factory for creating test RecordingJob instances."""
    return RecordingJob(
        id=kwargs.pop("id", 1),
        job_id=job_id or str(uuid.uuid7()),
        tenant_id=tenant_id,
        part_id=part_id,
        status=status,
        part_type=kwargs.pop("part_type", "browser_recording"),
        scripts_dir=kwargs.pop("scripts_dir", "/test/scripts"),
        created_at=kwargs.pop("created_at", datetime.now(UTC)),
        updated_at=kwargs.pop("updated_at", datetime.now(UTC)),
        progress=kwargs.pop("progress", 0.0),
        retry_count=kwargs.pop("retry_count", 0),
        max_retries=kwargs.pop("max_retries", 2),
        **kwargs,
    )


# ── Compilation domain fixtures ────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove CI env vars that interfere with config-default tests, then disable auth."""
    for key in list(os.environ):
        if any(
            key.startswith(p)
            for p in (
                "DATABASE_",
                "AUTH_",
                "REDIS_",
                "API_",
                "STORAGE_",
                "LOG_",
                "CACHE_",
                "TRACING_",
                "LLM_",
                "FFMPEG_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    # Restore test-only env vars that were cleared above
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://ci_test:ci_test_password_secure@127.0.0.1:5476/video_production_service_ci",  # pragma: allowlist secret
    )
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6398/1")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    # Required field on StorageSettings (no localhost fallback in w2v_common 0.15.0+).
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "localhost:9000")
    monkeypatch.setenv("API_DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")


# ---------------------------------------------------------------------------
# Explicit trust lanes (w2v-common >= 0.89.0)
# ---------------------------------------------------------------------------

# w2v-common 0.89.0 made TRUSTED_CIDRS default to EMPTY: no peer is trusted
# unless the operator names the networks. Tests that exercise the trusted-
# network lane must configure it explicitly — they may not lean on a default.
# These are the lanes the tests were written against (the pre-0.89 default);
# 192.168.0.0/16 stays out (SSR-proxy bypass vector, 2026-04-18).
EXPLICIT_TRUSTED_CIDRS = "127.0.0.1/32,::1/128,172.16.0.0/12,10.0.0.0/8"


@pytest.fixture
def explicit_trusted_cidrs(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure TRUSTED_CIDRS explicitly for the duration of one test.

    ``TRUSTED_NETWORKS`` is parsed from the env once, at import, and
    ``_is_trusted_network`` / the dispatch lanes read that module global at
    call time — so set the env (for anything re-reading it) AND pin the parsed
    global, instead of reloading the module mid-suite.
    """
    import ipaddress

    import w2v_common.auth.middleware as auth_middleware

    monkeypatch.setenv("TRUSTED_CIDRS", EXPLICIT_TRUSTED_CIDRS)
    monkeypatch.setattr(
        auth_middleware,
        "TRUSTED_NETWORKS",
        [ipaddress.ip_network(c, strict=False) for c in EXPLICIT_TRUSTED_CIDRS.split(",")],
    )
    return EXPLICIT_TRUSTED_CIDRS


@pytest.fixture
def settings_override(monkeypatch: pytest.MonkeyPatch):
    """Helper to override settings via env vars."""

    def _override(**kwargs: str) -> None:
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)

    return _override


# ---------------------------------------------------------------------------
# Real database fixtures (postgres+psycopg, TRUNCATE-based isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, the video_production_service_test_postgres compose service on
    127.0.0.1:5476 locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.
    """
    from src.database.models import Base

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Real async_sessionmaker for video_production_service tests."""
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory):
    """Real AsyncSession for a single test."""
    async with db_session_factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Real Redis fixtures (fakeredis)
# ---------------------------------------------------------------------------


@pytest.fixture
def real_redis_client():
    """RedisClient backed by fakeredis (real commands, in-memory)."""
    client = RedisClient(redis_url="redis://localhost:6379/0")
    inject_fakeredis(client)
    return client
