"""The sign-in rules are wired into EVERY recording path, not just defined.

Studio run 01a17e6f recorded a sign-in page as footage. The helpers that stop that
(``_sign_in``, ``_assert_signed_in``) only help if each recording path calls them,
and a username + password sign-in must never be offered to the RECORDED site as
HTTP Basic credentials (the password belongs to the sign-in page).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import SiteLoginFailedError
from src.services.recording.url_recorder import UrlRecorder

_SSO = {
    "type": "basic",
    "username": "u@example.com",
    "password": "p",  # pragma: allowlist secret
    "login_url": "https://auth.example/realms/app/protocol/openid-connect/auth",
}
_APP = "https://app.example/"


class _ReachedSignInCheckError(Exception):
    """Raised by the patched ``_assert_signed_in``: the path reached the check."""


def _settings() -> MagicMock:
    s = MagicMock()
    s.recording.viewport_width = 1280
    s.recording.viewport_height = 720
    s.recording.headless = True
    s.recording.output_format = "mp4"
    s.recording.default_quality_preset = "standard"
    s.ffmpeg.preset = "medium"
    s.ffmpeg.timeout = 300
    s.ffmpeg.default_crf = 23
    return s


def _fake_playwright() -> tuple[MagicMock, MagicMock]:
    page = MagicMock()
    page.goto = AsyncMock()
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    browser.close = AsyncMock()
    pw = MagicMock()
    pw.chromium.launch = AsyncMock(return_value=browser)

    @asynccontextmanager
    async def _cm():
        yield pw

    return MagicMock(side_effect=lambda: _cm()), browser


def _recorder() -> UrlRecorder:
    with patch("src.services.recording.url_recorder.get_settings", return_value=_settings()):
        return UrlRecorder()


async def _run_path(path: str, recorder: UrlRecorder, tmp_path: Path, auth=_SSO):
    config = {"viewport_width": 1280, "viewport_height": 720, "headless": True}
    if path == "screens":
        return await recorder._capture_with_playwright(_APP, auth, config, [], None)
    if path == "clips":
        return await recorder._capture_clips_with_playwright(
            "job", _APP, auth, config, [], 1000.0, None, tmp_path
        )
    return await recorder._record_with_playwright(
        "job", _APP, auth, config, {"enabled": False}, None, tmp_path
    )


@pytest.mark.parametrize("path", ["screens", "clips", "scroll"])
async def test_every_recording_path_signs_in_then_verifies_before_recording(
    path: str, tmp_path: Path
) -> None:
    """Swap ``_sign_in`` back to ``_form_login``, or drop ``_assert_signed_in`` from a
    path, and this fails: the form login would no longer be verified."""
    recorder = _recorder()
    fake, browser = _fake_playwright()
    with (
        patch("playwright.async_api.async_playwright", fake),
        patch("src.services.recording.url_recorder.install_ssrf_guard", AsyncMock()),
        patch.object(UrlRecorder, "_sign_in", AsyncMock()) as sign_in,
        patch.object(UrlRecorder, "_form_login", AsyncMock()) as form_login,
        patch.object(
            UrlRecorder, "_assert_signed_in", AsyncMock(side_effect=_ReachedSignInCheckError)
        ) as check,
        patch.object(UrlRecorder, "_settle_after_load", AsyncMock()),
        pytest.raises(_ReachedSignInCheckError),
    ):
        await _run_path(path, recorder, tmp_path)

    sign_in.assert_awaited_once()
    form_login.assert_not_awaited()  # only ever reached through _sign_in
    check.assert_awaited_once()
    assert check.await_args is not None
    assert check.await_args.args[2] == _APP  # verified against the RECORDED site
    # And the recorded site is never offered the sign-in password as HTTP Basic.
    assert "http_credentials" not in browser.new_context.await_args.kwargs


_SIGN_IN_PAGE = {
    "password": True,
    "heading": "Sign in",
    "title": "Acme",
    "controls": ["Sign in"],
    "auth_fields": 2,
}


class _RecordedPastTheCheckError(Exception):
    """Raised once a path starts capturing: it recorded without verifying."""


@pytest.mark.parametrize(
    "auth",
    [
        pytest.param({"type": "cookies", "cookies": [{"name": "s", "value": "v"}]}, id="cookies"),
        pytest.param({"type": "cookies", "cookie_header": "s=v"}, id="cookie_header"),
        pytest.param({"type": "bearer", "token": "t"}, id="bearer"),
        pytest.param(
            {"type": "basic", "username": "u", "password": "p"},  # pragma: allowlist secret
            id="basic",
        ),
        pytest.param(_SSO, id="sso_form"),
    ],
)
@pytest.mark.parametrize("path", ["screens", "clips", "scroll"])
async def test_every_path_fails_any_session_that_lands_on_a_sign_in_page(
    path: str, auth: dict, tmp_path: Path
) -> None:
    """The REAL check, every credential kind, every capture path. Gating the check on
    a form sign-in (the original "cookies were never verified" defect) lets a cookie,
    bearer or basic session record the sign-in page, and this fails."""
    recorder = _recorder()
    fake, browser = _fake_playwright()
    page = browser.new_context.return_value.new_page.return_value
    page.url = _APP  # the recorded site's own origin: only the PAGE gives it away
    page.evaluate = AsyncMock(return_value=_SIGN_IN_PAGE)
    with (
        patch("playwright.async_api.async_playwright", fake),
        patch("src.services.recording.url_recorder.install_ssrf_guard", AsyncMock()),
        patch.object(UrlRecorder, "_sign_in", AsyncMock()),
        patch.object(UrlRecorder, "_settle_after_load", AsyncMock()),
        patch.object(
            UrlRecorder, "_walk_and_capture", AsyncMock(side_effect=_RecordedPastTheCheckError)
        ),
        patch.object(
            UrlRecorder, "_walk_and_record_clips", AsyncMock(side_effect=_RecordedPastTheCheckError)
        ),
        patch.object(
            UrlRecorder, "_auto_scroll", AsyncMock(side_effect=_RecordedPastTheCheckError)
        ),
        pytest.raises(SiteLoginFailedError, match=r"^site_login_failed: the page still asks"),
    ):
        await _run_path(path, recorder, tmp_path, auth)
    page.evaluate.assert_awaited_once()


class TestNoBasicCredentialsForAFormSignIn:
    def test_an_sso_login_never_answers_the_apps_basic_challenge(self):
        assert UrlRecorder._context_auth_kwargs(_SSO, _APP) == {}

    def test_genuine_basic_without_a_sign_in_page_is_scoped_to_the_recorded_site(self):
        auth = {"type": "basic", "username": "u", "password": "p"}  # pragma: allowlist secret
        assert UrlRecorder._context_auth_kwargs(auth, "https://intranet.example/x") == {
            "http_credentials": {
                "username": "u",
                "password": "p",  # pragma: allowlist secret
                "origin": "https://intranet.example",
            }
        }
