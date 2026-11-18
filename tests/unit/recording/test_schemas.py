"""Tests for src/api/schemas.py -- Pydantic request/response models."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from src.api.schemas.recording import (
    AsyncJobResponse,
    AuthConfigInput,
    BatchJobResponse,
    BatchStatusResponse,
    JobStatusResponse,
    PaginatedJobsResponse,
    RecordBatchRequest,
    RecordingConfigInput,
    RecordingResultResponse,
    RecordPartRequest,
    RecordUrlRequest,
    ScrollConfig,
    UrlAuthInfo,
)

# The URL validators now resolve hostnames (SSRF guard, fail-closed on NXDOMAIN),
# so the placeholder hosts these tests use must resolve deterministically. No
# real DNS or network is touched.
_DNS = {
    "hook.test": "93.184.216.34",
    "example.com": "93.184.216.34",
    "www.example.com": "93.184.216.34",
    "test.com": "93.184.216.34",
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def fake_getaddrinfo(host: str, *_a: object, **_k: object) -> list:
        ip = _DNS.get(host, host)  # literal IPs pass through unchanged
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    yield


# ── RecordingConfigInput ───────────────────────────────────────────


class TestRecordingConfigInput:
    def test_all_optional(self):
        cfg = RecordingConfigInput()
        assert cfg.viewport_width is None
        assert cfg.headless is None
        assert cfg.max_retries is None
        assert cfg.output_format is None
        assert cfg.quality_preset is None

    def test_with_overrides(self):
        cfg = RecordingConfigInput(viewport_width=800, headless=False, max_retries=5)
        assert cfg.viewport_width == 800
        assert cfg.headless is False
        assert cfg.max_retries == 5

    def test_with_output_format(self):
        cfg = RecordingConfigInput(output_format="webm")
        assert cfg.output_format == "webm"

    def test_with_quality_preset(self):
        cfg = RecordingConfigInput(quality_preset="preview")
        assert cfg.quality_preset == "preview"

    def test_with_quality_preset_and_format(self):
        cfg = RecordingConfigInput(quality_preset="4k", output_format="mp4")
        assert cfg.quality_preset == "4k"
        assert cfg.output_format == "mp4"

    # -- viewport_width constraints --

    def test_viewport_width_too_small(self):
        with pytest.raises(ValidationError, match="viewport_width"):
            RecordingConfigInput(viewport_width=100)

    def test_viewport_width_too_large(self):
        with pytest.raises(ValidationError, match="viewport_width"):
            RecordingConfigInput(viewport_width=8000)

    def test_viewport_width_min_boundary(self):
        cfg = RecordingConfigInput(viewport_width=320)
        assert cfg.viewport_width == 320

    def test_viewport_width_max_boundary(self):
        cfg = RecordingConfigInput(viewport_width=7680)
        assert cfg.viewport_width == 7680

    # -- viewport_height constraints --

    def test_viewport_height_too_small(self):
        with pytest.raises(ValidationError, match="viewport_height"):
            RecordingConfigInput(viewport_height=100)

    def test_viewport_height_too_large(self):
        with pytest.raises(ValidationError, match="viewport_height"):
            RecordingConfigInput(viewport_height=5000)

    def test_viewport_height_min_boundary(self):
        cfg = RecordingConfigInput(viewport_height=240)
        assert cfg.viewport_height == 240

    def test_viewport_height_max_boundary(self):
        cfg = RecordingConfigInput(viewport_height=4320)
        assert cfg.viewport_height == 4320

    # -- timeout_ms constraints --

    def test_timeout_ms_too_small(self):
        with pytest.raises(ValidationError, match="timeout_ms"):
            RecordingConfigInput(timeout_ms=500)

    def test_timeout_ms_too_large(self):
        with pytest.raises(ValidationError, match="timeout_ms"):
            RecordingConfigInput(timeout_ms=700000)

    def test_timeout_ms_min_boundary(self):
        cfg = RecordingConfigInput(timeout_ms=1000)
        assert cfg.timeout_ms == 1000

    def test_timeout_ms_max_boundary(self):
        cfg = RecordingConfigInput(timeout_ms=600000)
        assert cfg.timeout_ms == 600000

    # -- slow_mo_ms constraints --

    def test_slow_mo_ms_negative(self):
        with pytest.raises(ValidationError, match="slow_mo_ms"):
            RecordingConfigInput(slow_mo_ms=-1)

    def test_slow_mo_ms_too_large(self):
        with pytest.raises(ValidationError, match="slow_mo_ms"):
            RecordingConfigInput(slow_mo_ms=20000)

    def test_slow_mo_ms_zero(self):
        cfg = RecordingConfigInput(slow_mo_ms=0)
        assert cfg.slow_mo_ms == 0

    def test_slow_mo_ms_max_boundary(self):
        cfg = RecordingConfigInput(slow_mo_ms=10000)
        assert cfg.slow_mo_ms == 10000

    # -- capture_scale constraints --

    def test_capture_scale_default_none(self):
        cfg = RecordingConfigInput()
        assert cfg.capture_scale is None

    def test_capture_scale_below_min(self):
        with pytest.raises(ValidationError, match="capture_scale"):
            RecordingConfigInput(capture_scale=0.5)

    def test_capture_scale_above_max(self):
        with pytest.raises(ValidationError, match="capture_scale"):
            RecordingConfigInput(capture_scale=3.5)

    def test_capture_scale_min_boundary(self):
        cfg = RecordingConfigInput(capture_scale=1.0)
        assert cfg.capture_scale == 1.0

    def test_capture_scale_max_boundary(self):
        cfg = RecordingConfigInput(capture_scale=3.0)
        assert cfg.capture_scale == 3.0

    def test_capture_scale_typical(self):
        cfg = RecordingConfigInput(capture_scale=2.0)
        assert cfg.capture_scale == 2.0

    # -- max_retries constraints --

    def test_max_retries_negative(self):
        with pytest.raises(ValidationError, match="max_retries"):
            RecordingConfigInput(max_retries=-1)

    def test_max_retries_too_large(self):
        with pytest.raises(ValidationError, match="max_retries"):
            RecordingConfigInput(max_retries=11)

    def test_max_retries_zero(self):
        cfg = RecordingConfigInput(max_retries=0)
        assert cfg.max_retries == 0

    def test_max_retries_max_boundary(self):
        cfg = RecordingConfigInput(max_retries=10)
        assert cfg.max_retries == 10

    # -- color_scheme validator --

    def test_color_scheme_light(self):
        cfg = RecordingConfigInput(color_scheme="light")
        assert cfg.color_scheme == "light"

    def test_color_scheme_dark(self):
        cfg = RecordingConfigInput(color_scheme="dark")
        assert cfg.color_scheme == "dark"

    def test_color_scheme_no_preference(self):
        cfg = RecordingConfigInput(color_scheme="no-preference")
        assert cfg.color_scheme == "no-preference"

    def test_color_scheme_invalid(self):
        with pytest.raises(ValidationError, match="color_scheme"):
            RecordingConfigInput(color_scheme="blue")

    def test_color_scheme_none(self):
        cfg = RecordingConfigInput(color_scheme=None)
        assert cfg.color_scheme is None

    # -- output_format validator --

    def test_output_format_mp4(self):
        cfg = RecordingConfigInput(output_format="mp4")
        assert cfg.output_format == "mp4"

    def test_output_format_invalid(self):
        with pytest.raises(ValidationError, match="output_format"):
            RecordingConfigInput(output_format="avi")

    def test_output_format_none(self):
        cfg = RecordingConfigInput(output_format=None)
        assert cfg.output_format is None

    # -- quality_preset validator --

    def test_quality_preset_standard(self):
        cfg = RecordingConfigInput(quality_preset="standard")
        assert cfg.quality_preset == "standard"

    def test_quality_preset_high_quality(self):
        cfg = RecordingConfigInput(quality_preset="high_quality")
        assert cfg.quality_preset == "high_quality"

    def test_quality_preset_invalid(self):
        with pytest.raises(ValidationError, match="quality_preset"):
            RecordingConfigInput(quality_preset="ultra")

    def test_quality_preset_none(self):
        cfg = RecordingConfigInput(quality_preset=None)
        assert cfg.quality_preset is None


# ── AuthConfigInput ────────────────────────────────────────────────


class TestAuthConfigInput:
    def test_defaults(self):
        auth = AuthConfigInput()
        assert auth.auth_type is None
        assert auth.env_vars is None

    def test_with_values(self):
        auth = AuthConfigInput(auth_type="bearer", env_vars={"TOKEN": "abc"})
        assert auth.auth_type == "bearer"
        assert auth.env_vars == {"TOKEN": "abc"}


# ── RecordPartRequest ──────────────────────────────────────────────


class TestRecordPartRequest:
    def test_valid(self):
        req = RecordPartRequest(part_id="p1", scripts_dir="/scripts")
        assert req.part_id == "p1"
        assert req.scripts_dir == "/scripts"
        assert req.part_type == "browser_recording"
        assert req.webhook_url is None
        assert req.recording_config is None
        assert req.auth_config is None

    def test_missing_part_id(self):
        with pytest.raises(ValidationError):
            RecordPartRequest(scripts_dir="/scripts")

    def test_missing_scripts_dir(self):
        with pytest.raises(ValidationError):
            RecordPartRequest(part_id="p1")

    def test_with_webhook(self):
        req = RecordPartRequest(
            part_id="p1", scripts_dir="/scripts", webhook_url="http://hook.test/cb"
        )
        assert req.webhook_url == "http://hook.test/cb"

    # -- part_id constraints --

    def test_part_id_empty(self):
        with pytest.raises(ValidationError, match="part_id"):
            RecordPartRequest(part_id="", scripts_dir="/scripts")

    def test_part_id_too_long(self):
        with pytest.raises(ValidationError, match="part_id"):
            RecordPartRequest(part_id="a" * 129, scripts_dir="/scripts")

    def test_part_id_max_length(self):
        req = RecordPartRequest(part_id="a" * 128, scripts_dir="/scripts")
        assert len(req.part_id) == 128

    def test_part_id_invalid_chars_space(self):
        with pytest.raises(ValidationError, match="part_id"):
            RecordPartRequest(part_id="part one", scripts_dir="/scripts")

    def test_part_id_invalid_chars_slash(self):
        with pytest.raises(ValidationError, match="part_id"):
            RecordPartRequest(part_id="part/one", scripts_dir="/scripts")

    def test_part_id_invalid_chars_special(self):
        with pytest.raises(ValidationError, match="part_id"):
            RecordPartRequest(part_id="part@one!", scripts_dir="/scripts")

    def test_part_id_valid_chars(self):
        req = RecordPartRequest(part_id="part-1_v2.0", scripts_dir="/scripts")
        assert req.part_id == "part-1_v2.0"

    # -- scripts_dir constraints --

    def test_scripts_dir_relative_path(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordPartRequest(part_id="p1", scripts_dir="relative/path")

    def test_scripts_dir_with_double_dots(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordPartRequest(part_id="p1", scripts_dir="/scripts/../etc")

    def test_scripts_dir_too_long(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordPartRequest(part_id="p1", scripts_dir="/" + "a" * 1024)

    def test_scripts_dir_empty(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordPartRequest(part_id="p1", scripts_dir="")

    # -- webhook_url constraints --

    def test_webhook_url_ftp_rejected(self):
        with pytest.raises(ValidationError, match="webhook_url"):
            RecordPartRequest(part_id="p1", scripts_dir="/s", webhook_url="ftp://bad.com/hook")

    def test_webhook_url_no_scheme_rejected(self):
        with pytest.raises(ValidationError, match="webhook_url"):
            RecordPartRequest(part_id="p1", scripts_dir="/s", webhook_url="hook.test/cb")

    def test_webhook_url_too_long(self):
        long_url = "https://example.com/" + "x" * 2040
        with pytest.raises(ValidationError, match="webhook_url"):
            RecordPartRequest(part_id="p1", scripts_dir="/s", webhook_url=long_url)

    def test_webhook_url_https(self):
        req = RecordPartRequest(part_id="p1", scripts_dir="/s", webhook_url="https://hook.test/cb")
        assert req.webhook_url == "https://hook.test/cb"

    def test_webhook_url_none_allowed(self):
        req = RecordPartRequest(part_id="p1", scripts_dir="/s", webhook_url=None)
        assert req.webhook_url is None


# ── RecordBatchRequest ─────────────────────────────────────────────


class TestRecordBatchRequest:
    def test_valid(self):
        req = RecordBatchRequest(part_ids=["a", "b"], scripts_dir="/s")
        assert req.part_ids == ["a", "b"]
        assert req.execution_mode == "parallel"
        assert req.failure_policy == "continue"
        assert req.priority == 0

    def test_missing_part_ids(self):
        with pytest.raises(ValidationError):
            RecordBatchRequest(scripts_dir="/s")

    # -- part_ids constraints --

    def test_part_ids_empty_list(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=[], scripts_dir="/s")

    def test_part_ids_too_many(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=[f"p{i}" for i in range(101)], scripts_dir="/s")

    def test_part_ids_max_count(self):
        req = RecordBatchRequest(part_ids=[f"p{i}" for i in range(100)], scripts_dir="/s")
        assert len(req.part_ids) == 100

    def test_part_ids_item_invalid_chars(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=["valid", "in valid"], scripts_dir="/s")

    def test_part_ids_item_too_long(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=["a" * 129], scripts_dir="/s")

    def test_part_ids_item_empty_string(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=[""], scripts_dir="/s")

    def test_part_ids_item_special_chars(self):
        with pytest.raises(ValidationError, match="part_ids"):
            RecordBatchRequest(part_ids=["ok", "not/ok"], scripts_dir="/s")

    # -- scripts_dir constraints --

    def test_scripts_dir_relative(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="relative")

    def test_scripts_dir_with_double_dots(self):
        with pytest.raises(ValidationError, match="scripts_dir"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/foo/../bar")

    # -- webhook_url constraints --

    def test_webhook_url_invalid_scheme(self):
        with pytest.raises(ValidationError, match="webhook_url"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/s", webhook_url="ftp://bad.com")

    def test_webhook_url_valid(self):
        req = RecordBatchRequest(
            part_ids=["a"], scripts_dir="/s", webhook_url="https://hook.test/cb"
        )
        assert req.webhook_url == "https://hook.test/cb"

    # -- execution_mode constraints --

    def test_execution_mode_sequential(self):
        req = RecordBatchRequest(part_ids=["a"], scripts_dir="/s", execution_mode="sequential")
        assert req.execution_mode == "sequential"

    def test_execution_mode_invalid(self):
        with pytest.raises(ValidationError, match="execution_mode"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/s", execution_mode="random")

    # -- failure_policy constraints --

    def test_failure_policy_fail_fast(self):
        req = RecordBatchRequest(part_ids=["a"], scripts_dir="/s", failure_policy="fail_fast")
        assert req.failure_policy == "fail_fast"

    def test_failure_policy_invalid(self):
        with pytest.raises(ValidationError, match="failure_policy"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/s", failure_policy="ignore")

    # -- priority constraints --

    def test_priority_too_low(self):
        with pytest.raises(ValidationError, match="priority"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/s", priority=-101)

    def test_priority_too_high(self):
        with pytest.raises(ValidationError, match="priority"):
            RecordBatchRequest(part_ids=["a"], scripts_dir="/s", priority=101)

    def test_priority_min_boundary(self):
        req = RecordBatchRequest(part_ids=["a"], scripts_dir="/s", priority=-100)
        assert req.priority == -100

    def test_priority_max_boundary(self):
        req = RecordBatchRequest(part_ids=["a"], scripts_dir="/s", priority=100)
        assert req.priority == 100


# ── AsyncJobResponse ──────────────────────────────────────────────


class TestAsyncJobResponse:
    def test_creation(self):
        resp = AsyncJobResponse(job_id="j1", poll_url="/poll/j1")
        assert resp.job_id == "j1"
        assert resp.status == "pending"


# ── BatchJobResponse ──────────────────────────────────────────────


class TestBatchJobResponse:
    def test_creation(self):
        resp = BatchJobResponse(batch_id="b1", job_ids={"p1": "j1", "p2": "j2"})
        assert resp.batch_id == "b1"
        assert resp.status == "pending"
        assert len(resp.job_ids) == 2


# ── JobStatusResponse ─────────────────────────────────────────────


class TestJobStatusResponse:
    def test_minimal(self):
        resp = JobStatusResponse(job_id="j1", status="pending")
        assert resp.progress is None
        assert resp.retry_count is None
        assert resp.duration_seconds is None
        assert resp.width is None
        assert resp.height is None
        assert resp.file_size_bytes is None
        assert resp.has_audio is None

    def test_with_progress(self):
        resp = JobStatusResponse(
            job_id="j1",
            status="processing",
            progress=0.5,
            progress_stage="recording",
            duration_seconds=12.5,
            width=1280,
            height=720,
            file_size_bytes=4096,
            has_audio=True,
        )
        assert resp.progress == 0.5
        assert resp.progress_stage == "recording"
        assert resp.duration_seconds == 12.5
        assert resp.width == 1280
        assert resp.height == 720
        assert resp.file_size_bytes == 4096
        assert resp.has_audio is True


# ── RecordingResultResponse ───────────────────────────────────────


class TestRecordingResultResponse:
    def test_all_optional(self):
        resp = RecordingResultResponse(job_id="j1", status="completed")
        assert resp.video_path is None
        assert resp.duration_seconds is None
        assert resp.has_audio is None

    def test_full(self):
        resp = RecordingResultResponse(
            job_id="j1",
            status="completed",
            part_id="p1",
            video_path="/v.webm",
            duration_seconds=10.0,
            file_size_bytes=5000,
            width=1920,
            height=1080,
            has_audio=True,
            recording_time_seconds=12.0,
            error=None,
        )
        assert resp.width == 1920
        assert resp.has_audio is True


# ── UrlAuthInfo ──────────────────────────────────────────────────


class TestUrlAuthInfo:
    def test_defaults_to_none_type(self):
        auth = UrlAuthInfo()
        assert auth.type == "none"
        assert auth.username is None
        assert auth.login_url is None

    def test_basic(self):
        auth = UrlAuthInfo(type="basic", username="u", password="p")
        assert auth.type == "basic"
        assert auth.username == "u"

    def test_bearer(self):
        auth = UrlAuthInfo(type="bearer", token="abc")
        assert auth.token == "abc"

    def test_cookies_header(self):
        auth = UrlAuthInfo(type="cookies", cookie_header="a=1; b=2")
        assert auth.cookie_header == "a=1; b=2"

    def test_form_login_fields(self):
        auth = UrlAuthInfo(
            type="none",
            login_url="https://auth/login",
            username="u",
            password="p",
            submit_selector="#go",
        )
        assert auth.login_url == "https://auth/login"
        assert auth.submit_selector == "#go"

    def test_invalid_type_rejected(self):
        with pytest.raises(ValidationError, match="auth_info type"):
            UrlAuthInfo(type="oauth")


# ── RecordUrlRequest ─────────────────────────────────────────────


class TestRecordUrlRequest:
    def test_minimal(self):
        req = RecordUrlRequest(url="https://w2v.nonagentic.ai", part_id="promo-bg")
        assert req.url == "https://w2v.nonagentic.ai"
        assert req.part_id == "promo-bg"
        assert req.auth_info is None

    def test_with_auth_and_scroll(self):
        req = RecordUrlRequest(
            url="https://example.com",
            part_id="p1",
            auth_info=UrlAuthInfo(type="bearer", token="t"),
            scroll_config=ScrollConfig(speed_px_per_second=500),
        )
        assert req.auth_info is not None
        assert req.auth_info.token == "t"
        assert req.scroll_config is not None
        assert req.scroll_config.speed_px_per_second == 500

    def test_non_http_url_rejected(self):
        with pytest.raises(ValidationError, match="url must start with"):
            RecordUrlRequest(url="ftp://example.com", part_id="p1")

    def test_bad_part_id_rejected(self):
        with pytest.raises(ValidationError):
            RecordUrlRequest(url="https://example.com", part_id="bad id!")

    def test_bad_webhook_rejected(self):
        with pytest.raises(ValidationError, match="webhook_url"):
            RecordUrlRequest(url="https://x.com", part_id="p1", webhook_url="ws://x")


# ── PaginatedJobsResponse ────────────────────────────────────────


class TestPaginatedJobsResponse:
    def test_creation(self):
        resp = PaginatedJobsResponse(jobs=[], total=0, offset=0, limit=20)
        assert resp.jobs == []
        assert resp.total == 0


# ── BatchStatusResponse ──────────────────────────────────────────


class TestBatchStatusResponse:
    def test_defaults(self):
        resp = BatchStatusResponse(batch_id="b1", total=3)
        assert resp.completed == 0
        assert resp.failed == 0
        assert resp.status == "processing"
