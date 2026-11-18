"""Security regression tests: subprocess env-var injection via auth_config.

Covers two layers:

1.  The Pydantic schema layer (`AuthConfigInput.env_vars`) — must reject
    dangerous keys with a `ValidationError` before the request even
    reaches the service.
2.  The subprocess sink (`PlaywrightRecorder._run_recording_subprocess`)
    — even if a poisoned/loose `auth_config` dict is replayed from the
    DB, the subprocess must NOT receive denylisted keys.

See `reports/video_production_service/scan.md` §S1 for the full call
chain and exploit primitives.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.schemas.recording import (
    AuthConfigInput,
    sanitize_auth_env_vars,
)
from src.services.recording.playwright_recorder import PlaywrightRecorder

# ── Schema layer ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "key",
    [
        "PATH",
        "path",  # case-insensitive denylist
        "Path",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "NODE_OPTIONS",
        "NODE_EXTRA_CA_CERTS",
        "PYTHONPATH",
        "PYTHONHOME",
        "NPM_CONFIG_PREFIX",
        "BROWSER",
        "CHROME_EXECUTABLE",
        "PLAYWRIGHT_BROWSERS_PATH",
    ],
)
def test_env_vars_denylist_keys_raise_validation_error(key: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        AuthConfigInput(env_vars={key: "value"})
    # Real assertion: error message must point at the key, not be a
    # generic dict-shape error.
    assert key in str(excinfo.value) or key.upper() in str(excinfo.value)


@pytest.mark.parametrize(
    "key",
    [
        "lowercase_key",  # not uppercase
        "1STARTS_WITH_DIGIT",
        "HAS-DASH",
        "HAS SPACE",
        "HAS.DOT",
        "HAS$DOLLAR",
        "",  # empty
    ],
)
def test_env_vars_invalid_shape_raises(key: str) -> None:
    with pytest.raises(ValidationError):
        AuthConfigInput(env_vars={key: "v"})


def test_env_vars_key_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        AuthConfigInput(env_vars={"A" * 65: "v"})


def test_env_vars_value_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        AuthConfigInput(env_vars={"OK_KEY": "x" * 4097})


def test_env_vars_non_string_value_raises() -> None:
    # Pydantic boundary must reject a non-string value; we go through
    # model_validate so the test does not depend on a static type ignore.
    with pytest.raises(ValidationError):
        AuthConfigInput.model_validate({"env_vars": {"OK_KEY": 123}})


def test_env_vars_happy_path_application_token() -> None:
    auth = AuthConfigInput(auth_type="bearer", env_vars={"MY_APP_TOKEN": "xyz"})
    assert auth.env_vars == {"MY_APP_TOKEN": "xyz"}


def test_env_vars_happy_path_multiple_keys() -> None:
    auth = AuthConfigInput(
        env_vars={
            "BEARER_TOKEN": "tok",
            "BASIC_AUTH_USER": "alice",
            "BASIC_AUTH_PASS": "secret",
            "TENANT_TOKEN": "ttt",
        }
    )
    assert auth.env_vars is not None
    assert auth.env_vars["BEARER_TOKEN"] == "tok"
    assert auth.env_vars["BASIC_AUTH_USER"] == "alice"


def test_env_vars_none_is_allowed() -> None:
    auth = AuthConfigInput(env_vars=None)
    assert auth.env_vars is None


# ── sanitize_auth_env_vars (sink helper, pure function) ───────────


def test_sanitize_drops_denylisted_silently() -> None:
    raw = {
        "PATH": "/tmp/evil",
        "LD_PRELOAD": "/x.so",
        "NODE_OPTIONS": "--inspect-brk",
        "BEARER_TOKEN": "good",
        "MY_APP_TOKEN": "good2",
    }
    safe = sanitize_auth_env_vars(raw)
    # Real assertion on KEYS, not just length.
    assert set(safe.keys()) == {"BEARER_TOKEN", "MY_APP_TOKEN"}
    assert safe["BEARER_TOKEN"] == "good"
    assert safe["MY_APP_TOKEN"] == "good2"


def test_sanitize_drops_invalid_shape() -> None:
    raw = {
        "lowercase": "v",
        "HAS-DASH": "v",
        "OK_KEY": "v",
    }
    safe = sanitize_auth_env_vars(raw)
    assert set(safe.keys()) == {"OK_KEY"}


def test_sanitize_drops_non_string_values() -> None:
    raw: dict[str, object] = {"OK_KEY": "v", "BAD_VAL": 42}
    safe = sanitize_auth_env_vars(raw)
    assert set(safe.keys()) == {"OK_KEY"}


def test_sanitize_non_dict_returns_empty() -> None:
    assert sanitize_auth_env_vars(None) == {}
    assert sanitize_auth_env_vars("not a dict") == {}
    assert sanitize_auth_env_vars([("K", "V")]) == {}


# ── Subprocess sink ────────────────────────────────────────────────


def _mock_settings() -> MagicMock:
    return MagicMock(
        recording=MagicMock(
            viewport_width=1920,
            viewport_height=1080,
            timeout_ms=60000,
            timeout_buffer_s=30,
            min_video_size_bytes=10000,
            slow_mo_ms=0,
            color_scheme="light",
            headless=True,
            output_format="webm",
            default_quality_preset="standard",
            error_output_max_length=500,
            cancel_wait_timeout_s=5.0,
        ),
        ffmpeg=MagicMock(preset="medium", timeout=300, default_crf=23),
    )


@patch("src.services.recording.playwright_recorder.get_settings")
def test_subprocess_env_drops_poisoned_keys_from_db(mock_gs: MagicMock) -> None:
    """If a poisoned auth_config row is replayed from the DB (bypassing
    the API schema), the subprocess sink must still strip the dangerous
    keys before merging into the subprocess `env=`.
    """
    mock_gs.return_value = _mock_settings()

    captured_env: dict[str, str] = {}

    async def fake_subproc_exec(*_args: object, **kwargs: object) -> AsyncMock:
        # Capture the env passed to the subprocess.
        env_arg = kwargs.get("env")
        assert isinstance(env_arg, dict)
        for k, v in env_arg.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            captured_env[k] = v
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"passed", b""))
        return proc

    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_path = Path(tmpdir)
            script_file = scripts_path / "part1.ts"
            script_file.write_text("test")

            recorder = PlaywrightRecorder()

            # Simulate a poisoned DB row: bypass the Pydantic boundary,
            # hand the recorder a dict that contains denylisted keys.
            poisoned_auth = {
                "env_vars": {
                    "PATH": "/tmp/evil",
                    "LD_PRELOAD": "/tmp/evil.so",
                    "NODE_OPTIONS": "--require=/tmp/x.js",
                    "BEARER_TOKEN": "legit-token",
                    "MY_APP_TOKEN": "also-legit",
                }
            }
            resolved_config = {
                "viewport_width": 1920,
                "viewport_height": 1080,
                "timeout_ms": 1000,
                "slow_mo_ms": 0,
                "color_scheme": "light",
                "headless": True,
            }

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=fake_subproc_exec,
            ):
                await recorder._run_recording_subprocess(
                    "part1",
                    script_file,
                    scripts_path,
                    resolved_config,
                    poisoned_auth,
                    progress_callback=None,
                )

    asyncio.run(run())

    # Real assertions on KEYS, not lengths.
    # Dangerous keys: must NOT have been overwritten by the attacker —
    # the subprocess sees the original (OS) values, not the injected ones.
    assert captured_env.get("PATH") == os.environ.get("PATH")
    assert "LD_PRELOAD" not in captured_env or captured_env["LD_PRELOAD"] != ("/tmp/evil.so")
    assert (
        "NODE_OPTIONS" not in captured_env or captured_env["NODE_OPTIONS"] != "--require=/tmp/x.js"
    )

    # Legitimate keys: must have been merged.
    assert captured_env.get("BEARER_TOKEN") == "legit-token"
    assert captured_env.get("MY_APP_TOKEN") == "also-legit"

    # The fixed RECORDING_* keys still get written (they come after
    # env.update and are not attacker-controlled).
    assert captured_env.get("RECORDING_VIEWPORT_WIDTH") == "1920"


@patch("src.services.recording.playwright_recorder.get_settings")
def test_subprocess_env_happy_path_clean_auth(mock_gs: MagicMock) -> None:
    """A clean auth_config (the kind the schema would accept) flows
    through end-to-end: the subprocess receives MY_APP_TOKEN.
    """
    mock_gs.return_value = _mock_settings()

    captured_env: dict[str, str] = {}

    async def fake_subproc_exec(*_args: object, **kwargs: object) -> AsyncMock:
        env_arg = kwargs.get("env")
        assert isinstance(env_arg, dict)
        for k, v in env_arg.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            captured_env[k] = v
        proc = AsyncMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"passed", b""))
        return proc

    async def run() -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_path = Path(tmpdir)
            script_file = scripts_path / "part1.ts"
            script_file.write_text("test")

            recorder = PlaywrightRecorder()
            # Validate via schema first to prove the happy-path is
            # representative of what the API actually accepts.
            auth = AuthConfigInput(auth_type="bearer", env_vars={"MY_APP_TOKEN": "xyz"})
            assert auth.env_vars == {"MY_APP_TOKEN": "xyz"}

            resolved_config = {
                "viewport_width": 1920,
                "viewport_height": 1080,
                "timeout_ms": 1000,
                "slow_mo_ms": 0,
                "color_scheme": "light",
                "headless": True,
            }

            with patch(
                "asyncio.create_subprocess_exec",
                side_effect=fake_subproc_exec,
            ):
                await recorder._run_recording_subprocess(
                    "part1",
                    script_file,
                    scripts_path,
                    resolved_config,
                    {"env_vars": {"MY_APP_TOKEN": "xyz"}},
                    progress_callback=None,
                )

    asyncio.run(run())

    assert captured_env.get("MY_APP_TOKEN") == "xyz"
