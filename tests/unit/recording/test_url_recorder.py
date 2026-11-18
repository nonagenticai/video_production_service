"""Tests for src/services/recording/url_recorder.py -- UrlRecorder."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from playwright.async_api import Error as PlaywrightError

from src.exceptions import SiteLoginFailedError
from src.services.ffmpeg import FFmpegOps
from src.services.recording.artifacts import ARTIFACT_KIND_SCREENS_ZIP
from src.services.recording.url_recorder import (
    _DEFAULT_MIN_CLIP_SECONDS,
    _DEFAULT_PASSWORD_SELECTOR,
    _DEFAULT_SUBMIT_SELECTOR,
    _DEFAULT_USERNAME_SELECTOR,
    UrlRecorder,
)


def _png(blank: bool) -> bytes:
    """A tiny PNG: uniform white when blank, one dark pixel otherwise."""
    img = Image.new("RGB", (4, 4), (255, 255, 255))
    if not blank:
        img.putpixel((0, 0), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_settings():
    """Return mock settings suitable for UrlRecorder.__init__."""
    s = MagicMock()
    s.recording.viewport_width = 1920
    s.recording.viewport_height = 1080
    s.recording.headless = True
    s.recording.output_format = "mp4"
    s.recording.default_quality_preset = "standard"
    s.ffmpeg.preset = "medium"
    s.ffmpeg.timeout = 300
    s.ffmpeg.default_crf = 23
    return s


def _make_recorder():
    """Create a UrlRecorder with mocked settings."""
    with patch(
        "src.services.recording.url_recorder.get_settings",
        return_value=_mock_settings(),
    ):
        return UrlRecorder()


# ---------------------------------------------------------------------------
# _resolve_config / _resolve_scroll_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_defaults(self):
        recorder = _make_recorder()
        config = recorder._resolve_config(None)
        assert config["viewport_width"] == 1920
        assert config["viewport_height"] == 1080
        assert config["headless"] is True
        assert config["output_format"] == "mp4"

    def test_overrides(self):
        recorder = _make_recorder()
        config = recorder._resolve_config({"viewport_width": 1280, "headless": False})
        assert config["viewport_width"] == 1280
        assert config["headless"] is False

    def test_interaction_steps_pass_through(self):
        recorder = _make_recorder()
        steps = [{"action": "click", "selector": ".nav"}]
        config = recorder._resolve_config({"interaction_steps": steps})
        assert config["interaction_steps"] == steps

    def test_capture_scale_defaults_to_one(self):
        recorder = _make_recorder()
        assert recorder._resolve_config(None)["capture_scale"] == 1.0

    def test_capture_scale_override(self):
        recorder = _make_recorder()
        assert recorder._resolve_config({"capture_scale": 2.0})["capture_scale"] == 2.0


class TestRunInteractions:
    async def test_click_scroll_wait(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        page.locator = MagicMock(return_value=loc)
        page.mouse.wheel = AsyncMock()
        steps = [
            {"action": "click", "selector": '.nav-item >> text="Build"', "hold_ms": 0},
            {"action": "scroll", "pixels": 400, "hold_ms": 0},
            {"action": "wait", "hold_ms": 0},
        ]
        await UrlRecorder._run_interactions(page, steps)
        page.locator.assert_called_once_with('.nav-item >> text="Build"')
        loc.first.click.assert_awaited_once()
        page.mouse.wheel.assert_awaited_once_with(0, 400)

    async def test_failed_click_is_soft(self):
        """A missing click target must not abort the remaining interactions.

        "Soft" means the steps AFTER the failure still run — a swallow that also
        exits the loop leaves the recording frozen on the pre-click screen, and
        a call count on the click itself cannot tell the two apart.
        """
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock(side_effect=PlaywrightError("no element"))
        page.locator = MagicMock(return_value=loc)
        page.mouse.wheel = AsyncMock()

        await UrlRecorder._run_interactions(
            page,
            [
                {"action": "click", "selector": ".missing", "hold_ms": 0},
                {"action": "scroll", "pixels": 250, "hold_ms": 0},
            ],
        )

        loc.first.click.assert_awaited_once()
        page.mouse.wheel.assert_awaited_once_with(0, 250)


class TestResolveScrollConfig:
    def test_defaults(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config(None)
        assert scroll["enabled"] is True
        assert scroll["speed_px_per_second"] == 300

    def test_none_values_dropped(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config({"speed_px_per_second": None, "enabled": False})
        # None override is ignored; explicit False is kept
        assert scroll["speed_px_per_second"] == 300
        assert scroll["enabled"] is False


# ---------------------------------------------------------------------------
# context auth helpers
# ---------------------------------------------------------------------------


class TestContextAuthKwargs:
    def test_basic_credentials_are_restricted_to_the_bound_origin(self):
        kwargs = UrlRecorder._context_auth_kwargs(
            {"type": "basic", "username": "u", "password": "p"},  # pragma: allowlist secret
            "https://app.example.com/dash?x=1",
        )
        assert kwargs["http_credentials"] == {
            "username": "u",
            "password": "p",  # pragma: allowlist secret
            "origin": "https://app.example.com",
        }

    def test_bearer_is_never_a_context_wide_header(self):
        """extra_http_headers ride on EVERY request, every origin — never used."""
        auth = {"type": "bearer", "token": "abc"}  # pragma: allowlist secret
        assert UrlRecorder._context_auth_kwargs(auth, "https://x.com") == {}
        assert UrlRecorder._credential_scope(auth, "https://x.com/a") == {
            "credential_origin": "https://x.com",
            "credential_headers": {"Authorization": "Bearer abc"},
        }

    def test_none_is_empty(self):
        assert UrlRecorder._context_auth_kwargs({"type": "none"}, "https://x.com") == {}
        assert UrlRecorder._credential_scope({"type": "none"}, "https://x.com") == {}

    def test_basic_without_username_is_empty(self):
        assert UrlRecorder._context_auth_kwargs({"type": "basic"}, "https://x.com") == {}


class TestCookieScope:
    def test_cookie_header_becomes_a_bound_origin_header(self):
        scope = UrlRecorder._credential_scope(
            {"type": "cookies", "cookie_header": "session=xyz; theme=dark"},
            "https://example.com/app",
        )
        assert scope == {
            "credential_origin": "https://example.com",
            "credential_headers": {"Cookie": "session=xyz; theme=dark"},
        }

    def test_an_explicit_cookie_domain_is_not_honoured(self):
        jar = [{"name": "a", "value": "1", "domain": ".attacker.com", "path": "/"}]
        scope = UrlRecorder._credential_scope({"type": "cookies", "cookies": jar}, "https://x.com")
        assert scope == {
            "credential_origin": "https://x.com",
            "credential_headers": {"Cookie": "a=1"},
        }

    def test_basic_binds_an_origin_without_headers(self):
        scope = UrlRecorder._credential_scope(
            {"type": "basic", "username": "u", "password": "p"},  # pragma: allowlist secret
            "https://x.com:8443/a",
        )
        assert scope == {"credential_origin": "https://x.com:8443"}


class TestFormLogin:
    async def test_defaults_used_when_selectors_absent(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.goto = AsyncMock()
        page.fill = AsyncMock()
        page.click = AsyncMock()
        page.wait_for_load_state = AsyncMock()

        await UrlRecorder._form_login(
            page,
            {"login_url": "https://auth.example/login", "username": "bob", "password": "p"},
        )

        page.goto.assert_awaited_once()
        # username + password fields filled with the default selectors
        fill_selectors = [c.args[0] for c in page.fill.await_args_list]
        assert _DEFAULT_USERNAME_SELECTOR in fill_selectors
        assert _DEFAULT_PASSWORD_SELECTOR in fill_selectors
        page.click.assert_awaited_once()
        assert page.click.await_args is not None
        assert page.click.await_args.args[0] == _DEFAULT_SUBMIT_SELECTOR

    async def test_networkidle_timeout_is_swallowed(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.goto = AsyncMock()
        page.fill = AsyncMock()
        page.click = AsyncMock()
        page.wait_for_load_state = AsyncMock(side_effect=PlaywrightError("timeout"))

        # Should not raise — the post-login settle wait is best-effort.
        await UrlRecorder._form_login(
            page,
            {
                "login_url": "https://auth/login",
                "username": "u",
                "password": "p",
            },  # pragma: allowlist secret
        )


_SSO = {
    "type": "basic",
    "username": "u@example.com",
    "password": "p",  # pragma: allowlist secret
    "login_url": "https://auth.example/realms/app/protocol/openid-connect/auth?client_id=app",
}


_COOKIES = {"type": "cookies", "cookies": [{"name": "session", "value": "abc"}]}


def _content_snapshot(**overrides: object) -> dict[str, object]:
    """The snapshot of an ordinary, signed-in content page.

    The header links "Sign in" ON PURPOSE: that is the false positive the
    sign-in-screen detector must never fire on.
    """
    snapshot: dict[str, object] = {
        "password": False,
        "heading": "Quarterly dashboard",
        "title": "Acme — Dashboard",
        "controls": ["Sign in", "Pricing", "Docs"],
        "auth_fields": 0,
    }
    snapshot.update(overrides)
    return snapshot


def _signed_in_page(url: str, *, snapshot: dict[str, object] | None = None) -> MagicMock:
    page = MagicMock()
    page.url = url
    page.evaluate = AsyncMock(return_value=snapshot or _content_snapshot())
    return page


class TestSignInIsVerified:
    """Prod run 01a17e6f: a recording that "signed in" and recorded the sign-in page.

    A failed sign-in must fail the recording with the ``site_login_failed`` code the
    orchestrator turns into "couldn't sign in - check the login", never footage."""

    async def test_landing_on_a_content_page_passes_even_with_a_header_sign_in_link(self):
        """An ordinary page that merely LINKS "Sign in" in its header is not a
        sign-in screen — the check must stay quiet on it."""
        page = _signed_in_page("https://app.example/dashboard")
        assert await UrlRecorder._assert_signed_in(page, _SSO, "https://app.example/") is None
        # It really inspected the page before passing, with the default selector.
        page.evaluate.assert_awaited_once()
        assert page.evaluate.await_args.args[1] == "input[type='password']"

    async def test_bounced_back_to_the_identity_provider_fails(self):
        page = _signed_in_page("https://auth.example/realms/app/login-actions/authenticate")
        with pytest.raises(SiteLoginFailedError, match=r"^site_login_failed: still on the sign-in"):
            await UrlRecorder._assert_signed_in(page, _SSO, "https://app.example/")

    async def test_a_same_origin_landing_on_the_login_path_fails(self):
        """The app's OWN sign-in page: same origin as the recorded site, so the
        origin-difference rule says nothing. The path does."""
        login = {**_SSO, "login_url": "https://app.example/login"}
        page = _signed_in_page("https://app.example/login?error=1")
        with pytest.raises(SiteLoginFailedError, match=r"^site_login_failed: still on the sign-in"):
            await UrlRecorder._assert_signed_in(page, login, "https://app.example/")

    async def test_a_visible_password_field_fails(self):
        login = {**_SSO, "login_url": "https://app.example/login"}
        page = _signed_in_page(
            "https://app.example/dashboard", snapshot=_content_snapshot(password=True)
        )
        with pytest.raises(SiteLoginFailedError, match=r"^site_login_failed: the page still asks"):
            await UrlRecorder._assert_signed_in(page, login, "https://app.example/")

    async def test_an_sso_only_sign_in_wall_fails(self):
        """The detector hole: an app-origin landing that renders only "Sign in
        with SSO" — no password FIELD — used to pass and got recorded."""
        page = _signed_in_page(
            "https://app.example/",
            snapshot=_content_snapshot(
                heading="Sign in", title="Acme", controls=["Sign in with SSO", "Help"]
            ),
        )
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the page is a sign-in"
        ):
            await UrlRecorder._assert_signed_in(page, _SSO, "https://app.example/")

    async def test_a_page_that_navigates_under_the_probe_fails(self):
        """The SPA bounce this guard exists for: the probe raises "execution
        context destroyed" because the page just navigated to its sign-in screen.
        Swallowing that reads as "signed in" and records the login page — the very
        defect this check was added to stop."""
        login = {**_SSO, "login_url": "https://app.example/login"}
        page = _signed_in_page("https://app.example/dashboard")
        page.evaluate = AsyncMock(side_effect=PlaywrightError("Execution context was destroyed"))
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the page could not be checked"
        ):
            await UrlRecorder._assert_signed_in(page, login, "https://app.example/")

    async def test_a_probe_that_answers_nothing_fails(self):
        """A page that cannot be OBSERVED cannot prove the sign-in worked."""
        page = _signed_in_page("https://app.example/dashboard")
        page.evaluate = AsyncMock(return_value=None)
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the page could not be checked"
        ):
            await UrlRecorder._assert_signed_in(page, _SSO, "https://app.example/")

    async def test_a_recording_with_no_credentials_is_not_checked(self):
        """No credential replayed ⇒ nothing to verify; the tenant may be
        recording a public sign-in page on purpose."""
        page = _signed_in_page(
            "https://app.example/login", snapshot=_content_snapshot(password=True)
        )
        await UrlRecorder._assert_signed_in(page, {"type": "none"}, "https://app.example/login")
        page.evaluate.assert_not_called()

    async def test_an_empty_cookie_login_is_not_checked(self):
        page = _signed_in_page("https://app.example/login")
        await UrlRecorder._assert_signed_in(
            page, {"type": "cookies", "cookies": []}, "https://app.example/login"
        )
        page.evaluate.assert_not_called()

    async def test_every_replayed_credential_is_verified(self):
        assert UrlRecorder._verifies_sign_in(_SSO) is True
        assert UrlRecorder._verifies_sign_in(_COOKIES) is True
        assert UrlRecorder._verifies_sign_in({"type": "cookies", "cookie_header": "s=1"}) is True
        assert UrlRecorder._verifies_sign_in({"type": "bearer", "token": "t"}) is True
        assert UrlRecorder._verifies_sign_in({"type": "basic", "username": "u"}) is True
        assert UrlRecorder._verifies_sign_in({"type": "none"}) is False
        assert UrlRecorder._verifies_sign_in({}) is False

    async def test_an_unfillable_sign_in_form_fails_without_retry(self):
        page = MagicMock()
        page.goto = AsyncMock()
        page.fill = AsyncMock(side_effect=PlaywrightError("Timeout 30000ms exceeded"))
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the sign-in form on auth\.example"
        ) as err:
            await UrlRecorder._sign_in(page, _SSO)
        assert err.value.retryable is False

    async def test_an_unreachable_sign_in_page_reports_the_real_cause(self):
        """A goto timeout / DNS failure is a REACHABILITY failure. Reporting it as
        "no username / password field found" sends the tenant hunting for a
        selector that was never the problem."""
        page = MagicMock()
        page.goto = AsyncMock(side_effect=PlaywrightError("net::ERR_NAME_NOT_RESOLVED"))
        with pytest.raises(
            SiteLoginFailedError,
            match=r"^site_login_failed: the sign-in page on auth\.example could not be opened",
        ) as err:
            await UrlRecorder._sign_in(page, _SSO)
        # The wrong diagnosis is gone, and the non-retryable verdict survives.
        assert "no username / password field" not in str(err.value)
        assert err.value.retryable is False


class TestCookieSessionsAreVerified:
    """Finding: ``type: "cookies"`` was dispatched verbatim and never verified.

    Expired or invalid cookies make the app redirect to its sign-in page; the
    probe used to return immediately unless the auth was a form login, so the
    job reported ``completed`` with a video of the login page."""

    async def test_expired_cookies_landing_on_the_sign_in_page_fail(self):
        page = _signed_in_page(
            "https://app.example/dashboard", snapshot=_content_snapshot(password=True)
        )
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the page still asks"
        ) as err:
            await UrlRecorder._assert_signed_in(page, _COOKIES, "https://app.example/dashboard")
        assert "saved cookie session" in str(err.value)

    async def test_a_cookie_session_that_works_passes(self):
        page = _signed_in_page("https://app.example/dashboard")
        assert (
            await UrlRecorder._assert_signed_in(page, _COOKIES, "https://app.example/dashboard")
            is None
        )

    async def test_a_probe_that_cannot_see_the_page_fails_a_cookie_session_closed(self):
        page = _signed_in_page("https://app.example/dashboard")
        page.evaluate = AsyncMock(side_effect=PlaywrightError("Target closed"))
        with pytest.raises(
            SiteLoginFailedError, match=r"^site_login_failed: the page could not be checked"
        ):
            await UrlRecorder._assert_signed_in(page, _COOKIES, "https://app.example/dashboard")


_APP_PAGE = "https://app.example/account"
#: Real sign-in screens: (heading, title). No password field and no SSO action, so
#: only the heading/title wording can make them a sign-in screen.
_SIGN_IN_HEADINGS = [
    ("Sign in", "Acme"),
    ("Sign in to Acme", "Acme"),
    ("Acme", "Log in | Acme"),
    ("Acme \u2013 Sign in", "Acme"),
    ("SIGN IN", "Acme"),
    ("Sign into Acme", "Acme"),
    ("Sign in or create account", "Amazon Sign-In"),
    ("Sign in / Register", "Acme Sign-In"),
    ("Acme", "Acme: Sign in"),
    ("Login/Register", "Acme"),
    ("Sign in required", "Acme"),
    ("Acme\u2014Sign in", "Acme"),
    ("Acme-Sign in", "Acme"),
    ("Sign\u00a0in", "Acme"),
]
#: Signed-in account pages whose heading names sign-in as a SECTION.
_ACCOUNT_PAGE_HEADINGS = [
    ("Sign-in activity", "Sign-in activity | Acme"),
    ("Login history", "Security \u2014 Acme account"),
    ("Recent sign-in sessions", "Acme"),
]


def _heading_snapshot(heading: str, title: str) -> dict[str, object]:
    """A page with a change-email field (an auth field) but no password field."""
    return _content_snapshot(
        heading=heading,
        title=title,
        controls=["Change email", "Continue", "Help"],
        auth_fields=1,
    )


class TestSignInScreenDetection:
    """The ONE page-based predicate, exercised without a browser."""

    def test_a_password_field_is_decisive(self):
        assert UrlRecorder._sign_in_evidence(_content_snapshot(password=True)) == "password"

    def test_an_sso_wall_with_sign_in_wording(self):
        snapshot = _content_snapshot(
            heading="Sign in to Acme", title="Acme", controls=["Sign in with SSO"]
        )
        assert UrlRecorder._sign_in_evidence(snapshot) == "wall"

    def test_an_sso_wall_with_only_the_product_name_as_heading(self):
        snapshot = _content_snapshot(heading="Acme", title="Acme", controls=["Sign in with SSO"])
        assert UrlRecorder._sign_in_evidence(snapshot) == "wall"

    def test_an_email_first_sign_in_form(self):
        """No password field yet (email → next), but the heading says sign in."""
        snapshot = _content_snapshot(heading="Sign in to Acme", controls=["Next"], auth_fields=1)
        assert UrlRecorder._sign_in_evidence(snapshot) == "wall"

    def test_a_content_page_with_a_header_sign_in_link_is_not_a_sign_in_screen(self):
        assert UrlRecorder._sign_in_evidence(_content_snapshot()) is None

    def test_a_help_page_about_signing_in_is_not_a_sign_in_screen(self):
        snapshot = _content_snapshot(
            heading="How to sign in to Acme", controls=["Docs", "Pricing", "Blog"]
        )
        assert UrlRecorder._sign_in_evidence(snapshot) is None

    def test_a_sparse_signed_in_page_whose_only_auth_link_is_a_header_sign_in(self):
        """A plain "Sign in" LINK is not an SSO action. With only two controls on
        the page, treating it as one would trip the bare-SSO-wall rule."""
        snapshot = _content_snapshot(heading="Acme", title="Acme", controls=["Sign in", "Docs"])
        assert UrlRecorder._sign_in_evidence(snapshot) is None

    def test_a_bare_sso_wall_has_at_most_two_controls(self):
        snapshot = _content_snapshot(
            heading="Acme", title="Acme", controls=["Sign in with SSO", "Help"]
        )
        assert UrlRecorder._sign_in_evidence(snapshot) == "wall"

    def test_three_controls_beside_an_sso_action_is_not_a_bare_wall(self):
        snapshot = _content_snapshot(
            heading="Acme", title="Acme", controls=["Sign in with SSO", "Help", "Pricing"]
        )
        assert UrlRecorder._sign_in_evidence(snapshot) is None

    @pytest.mark.parametrize(("heading", "title"), _SIGN_IN_HEADINGS)
    def test_sign_in_wording_anywhere_in_the_heading_or_title(self, heading, title):
        assert UrlRecorder._sign_in_evidence(_heading_snapshot(heading, title)) == "wall"

    @pytest.mark.parametrize(("heading", "title"), _ACCOUNT_PAGE_HEADINGS)
    def test_an_account_page_naming_sign_in_is_not_a_sign_in_screen(self, heading, title):
        assert UrlRecorder._sign_in_evidence(_heading_snapshot(heading, title)) is None

    @pytest.mark.parametrize(("heading", "title"), _SIGN_IN_HEADINGS)
    async def test_a_cookie_session_on_a_sign_in_heading_fails(self, heading, title):
        page = _signed_in_page(_APP_PAGE, snapshot=_heading_snapshot(heading, title))
        with pytest.raises(SiteLoginFailedError, match=r"^site_login_failed: "):
            await UrlRecorder._assert_signed_in(page, _COOKIES, _APP_PAGE)

    @pytest.mark.parametrize(("heading", "title"), _ACCOUNT_PAGE_HEADINGS)
    async def test_a_cookie_session_on_an_account_page_passes(self, heading, title):
        page = _signed_in_page(_APP_PAGE, snapshot=_heading_snapshot(heading, title))
        assert await UrlRecorder._assert_signed_in(page, _COOKIES, _APP_PAGE) is None


class TestStagingIsFreedWhenCaptureRaises:
    """A raised sign-in failure never returns a result dict, so nobody else can
    remove the per-job staging dir — the recorder must free it itself."""

    @pytest.mark.parametrize("method", ["capture_screens", "capture_screen_clips"])
    async def test_a_raised_sign_in_failure_frees_the_staging_dir(
        self, method: str, tmp_path: Path
    ) -> None:
        recorder = _make_recorder()
        staging = tmp_path / "staging"
        staging.mkdir()
        boom = SiteLoginFailedError("site_login_failed: still on the sign-in page")
        with (
            patch(
                "src.services.recording.url_recorder.tempfile.mkdtemp",
                return_value=str(staging),
            ),
            patch.object(UrlRecorder, "_capture_with_playwright", AsyncMock(side_effect=boom)),
            patch.object(
                UrlRecorder, "_capture_clips_with_playwright", AsyncMock(side_effect=boom)
            ),
            pytest.raises(SiteLoginFailedError),
        ):
            await getattr(recorder, method)(job_id="j", url="https://x.example/", steps=[])

        assert not staging.exists(), "a raised sign-in failure must not leave its staging dir"

    async def test_record_url_frees_the_staging_dir_on_a_raised_sign_in_failure(
        self, tmp_path: Path
    ) -> None:
        recorder = _make_recorder()
        staging = tmp_path / "staging"
        staging.mkdir()
        boom = SiteLoginFailedError("site_login_failed: still on the sign-in page")
        with (
            patch(
                "src.services.recording.url_recorder.tempfile.mkdtemp",
                return_value=str(staging),
            ),
            patch.object(UrlRecorder, "_record_with_playwright", AsyncMock(side_effect=boom)),
            pytest.raises(SiteLoginFailedError),
        ):
            await recorder.record_url(job_id="j", url="https://x.example/")

        assert not staging.exists(), "a raised sign-in failure must not leave its staging dir"


# ---------------------------------------------------------------------------
# record_url (orchestration)
# ---------------------------------------------------------------------------


class TestRecordUrl:
    async def test_success_flow(self):
        recorder = _make_recorder()

        fake_video = MagicMock()
        fake_video.suffix = ".mp4"
        fake_video.stat.return_value.st_size = 99999

        mock_probe = MagicMock()
        mock_probe.duration = 8.0
        mock_probe.width = 1920
        mock_probe.height = 1080
        mock_probe.has_audio = False

        with (
            patch.object(
                recorder, "_record_with_playwright", new_callable=AsyncMock, return_value=fake_video
            ) as mock_record,
            patch.object(
                recorder, "_post_process", new_callable=AsyncMock, return_value=fake_video
            ),
            patch(
                "src.services.recording.url_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_probe,
            ),
        ):
            result = await recorder.record_url(
                job_id="j-url",
                url="https://w2v.nonagentic.ai",
            )

        assert result["success"] is True
        # A real video recording declares nothing, i.e. it stays on the strict
        # ffprobe branch of the success path.
        assert "artifact_kind" not in result
        assert result["duration_seconds"] == 8.0
        assert result["file_size_bytes"] == 99999
        # URL is forwarded verbatim as the second positional arg
        assert mock_record.await_args is not None
        assert mock_record.await_args.args[1] == "https://w2v.nonagentic.ai"

    async def test_failure_returns_error(self):
        recorder = _make_recorder()
        with patch.object(
            recorder,
            "_record_with_playwright",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            result = await recorder.record_url(job_id="j-url", url="https://x.com")
        assert result["success"] is False
        assert "boom" in result["error"]

    async def test_playwright_error_caught(self):
        recorder = _make_recorder()
        with patch.object(
            recorder,
            "_record_with_playwright",
            new_callable=AsyncMock,
            side_effect=PlaywrightError("navigation failed"),
        ):
            result = await recorder.record_url(job_id="j-url", url="https://x.com")
        assert result["success"] is False
        assert "navigation failed" in result["error"]


# ---------------------------------------------------------------------------
# settle-before-capture + blank-lead trim (white-screen fix)
# ---------------------------------------------------------------------------


async def _passthrough_to_thread(func, *args, **kwargs):
    """Run the to_thread target inline so we can assert on its call args."""
    return func(*args, **kwargs)


class TestSettleAndTrim:
    def test_resolve_config_passes_settle_knobs(self):
        recorder = _make_recorder()
        cfg = recorder._resolve_config({"post_load_delay_ms": 3500, "wait_for_selector": "#main"})
        assert cfg["post_load_delay_ms"] == 3500
        assert cfg["wait_for_selector"] == "#main"

    async def test_settle_waits_for_selector_then_delays(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.wait_for_selector = AsyncMock()
        page.screenshot = AsyncMock(return_value=_png(blank=False))
        with patch(
            "src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await UrlRecorder._settle_after_load(
                page, {"wait_for_selector": "#main", "post_load_delay_ms": 2000}
            )
        page.wait_for_selector.assert_awaited_once()
        # Non-blank frame ⇒ only the settle delay sleeps (no blank re-check).
        sleep.assert_awaited_once_with(2.0)

    async def test_settle_selector_timeout_is_swallowed(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.wait_for_selector = AsyncMock(side_effect=PlaywrightError("timeout"))
        page.screenshot = AsyncMock(return_value=_png(blank=False))
        with patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock):
            # Must not raise — the content wait is best-effort.
            await UrlRecorder._settle_after_load(page, {"wait_for_selector": "#main"})

    async def test_settle_is_noop_without_knobs(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.wait_for_selector = AsyncMock()
        page.screenshot = AsyncMock(return_value=_png(blank=False))
        with patch(
            "src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await UrlRecorder._settle_after_load(page, {})
        page.wait_for_selector.assert_not_awaited()
        # No knobs + non-blank frame ⇒ no sleeps.
        sleep.assert_not_awaited()

    async def test_settle_clamps_runaway_post_load_delay(self):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.wait_for_selector = AsyncMock()
        page.screenshot = AsyncMock(return_value=_png(blank=False))
        with patch(
            "src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await UrlRecorder._settle_after_load(page, {"post_load_delay_ms": 999_999})
        # Clamped to _MAX_POST_LOAD_DELAY_MS (30000ms = 30.0s), not 999.999s.
        sleep.assert_awaited_once_with(30.0)

    async def test_settle_warns_and_rechecks_on_blank_frame(self, caplog):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.wait_for_selector = AsyncMock()
        page.screenshot = AsyncMock(return_value=_png(blank=True))  # uniform ⇒ blank
        with (
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
            caplog.at_level("WARNING"),
        ):
            await UrlRecorder._settle_after_load(page, {})
        # Blank ⇒ a re-check screenshot is taken and the emptiness is surfaced.
        assert page.screenshot.await_count == 2
        assert "blank" in caplog.text.lower()

    async def test_post_process_trims_lead_when_settle_set(self):
        recorder = _make_recorder()
        config = {
            "output_format": "mp4",
            "post_load_delay_ms": 3500,
            "_lead_seconds": 5.0,
            "quality_preset": "standard",
        }
        with (
            patch.object(FFmpegOps, "convert_to_mp4", return_value=Path("/tmp/x.mp4")) as conv,
            patch(
                "src.services.recording.url_recorder.asyncio.to_thread",
                new=AsyncMock(side_effect=_passthrough_to_thread),
            ),
        ):
            await recorder._post_process(Path("/tmp/x.webm"), config)
        # 5.0s lead minus the 0.5s safety margin.
        assert conv.call_args.kwargs["start_offset_s"] == pytest.approx(4.5)

    async def test_post_process_no_trim_without_settle_knob(self):
        recorder = _make_recorder()
        config = {
            "output_format": "mp4",
            "_lead_seconds": 5.0,  # present, but trim is gated on post_load_delay_ms
            "quality_preset": "standard",
        }
        with (
            patch.object(FFmpegOps, "convert_to_mp4", return_value=Path("/tmp/x.mp4")) as conv,
            patch(
                "src.services.recording.url_recorder.asyncio.to_thread",
                new=AsyncMock(side_effect=_passthrough_to_thread),
            ),
        ):
            await recorder._post_process(Path("/tmp/x.webm"), config)
        assert conv.call_args.kwargs["start_offset_s"] == 0.0


# ---------------------------------------------------------------------------
# min-hold (single-viewport pages must still yield a usable-length clip)
# ---------------------------------------------------------------------------


class TestMinHold:
    def test_resolve_scroll_config_includes_min_clip(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config(None)
        assert scroll["min_clip_seconds"] == _DEFAULT_MIN_CLIP_SECONDS

    async def test_hold_pads_short_phase_to_min_clip(self):
        # ~1s elapsed, end_delay 2s, min 8s ⇒ hold pads to reach 8s (7s).
        import time

        phase_start = time.monotonic() - 1.0
        with patch(
            "src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await UrlRecorder._hold_until_min(phase_start, end_delay=2.0, min_clip=8.0)
        assert sleep.await_args is not None
        assert sleep.await_args.args[0] == pytest.approx(7.0, abs=0.2)

    async def test_hold_falls_back_to_end_delay_when_already_long(self):
        import time

        phase_start = time.monotonic() - 12.0  # already past the minimum
        with patch(
            "src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            await UrlRecorder._hold_until_min(phase_start, end_delay=2.0, min_clip=8.0)
        sleep.assert_awaited_once_with(2.0)

    async def test_auto_scroll_single_viewport_holds_to_min_clip(self):
        recorder = _make_recorder()
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        # scrollHeight == innerHeight ⇒ total_scroll == 0 (single viewport).
        page.evaluate = AsyncMock(side_effect=[1000, 1000])
        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        with patch("src.services.recording.url_recorder.asyncio.sleep", new=fake_sleep):
            await recorder._auto_scroll(page, recorder._resolve_scroll_config(None))

        # No scroll steps; the padded end-hold floors the clip near min_clip (8s).
        assert max(sleeps) == pytest.approx(_DEFAULT_MIN_CLIP_SECONDS, abs=0.5)


class TestWalkAndCapture:
    """Per-screen capture: screenshot after each step that has a capture_id."""

    @staticmethod
    def _page_with_shots():
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        page.locator = MagicMock(return_value=loc)
        page.screenshot = AsyncMock(side_effect=lambda: _png(blank=False))
        return page, loc

    async def test_one_shot_per_capture_id(self):
        page, _loc = self._page_with_shots()
        steps = [
            {
                "action": "click",
                "selector": '.nav-item >> text="Home"',
                "capture_id": "home",
                "settle_ms": 0,
            },
            {
                "action": "click",
                "selector": '.nav-item >> text="Build"',
                "capture_id": "build",
                "settle_ms": 0,
            },
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps)
        assert [cid for cid, _ in shots] == ["home", "build"]
        assert page.screenshot.await_count == 2

    async def test_step_without_capture_id_advances_but_does_not_shoot(self):
        # The persona-switch case: click reveals nav, captures nothing.
        page, loc = self._page_with_shots()
        steps = [
            {"action": "click", "selector": ".persona-switch", "settle_ms": 0},  # no capture_id
            {
                "action": "click",
                "selector": '.nav-item >> text="Governance"',
                "capture_id": "gov",
                "settle_ms": 0,
            },
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps)
        assert [cid for cid, _ in shots] == ["gov"]
        assert loc.first.click.await_count == 2  # both steps clicked
        assert page.screenshot.await_count == 1  # only the captured step

    async def test_failed_click_still_captures(self):
        # A stale selector soft-fails; the screenshot is still taken so one bad
        # selector never empties the whole capture set.
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock(side_effect=PlaywrightError("no element"))
        page.locator = MagicMock(return_value=loc)
        page.screenshot = AsyncMock(return_value=_png(blank=False))
        shots = await UrlRecorder._walk_and_capture(
            page, [{"action": "click", "selector": ".missing", "capture_id": "x", "settle_ms": 0}]
        )
        assert [cid for cid, _ in shots] == ["x"]

    async def test_duplicate_capture_id_skipped(self):
        page, _loc = self._page_with_shots()
        steps = [
            {"action": "click", "selector": ".a", "capture_id": "dup", "settle_ms": 0},
            {"action": "click", "selector": ".b", "capture_id": "dup", "settle_ms": 0},
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps)
        assert [cid for cid, _ in shots] == ["dup"]


class TestWalkAndCaptureModifiers:
    """Per-step goto / viewport / scroll_to / wait_for modifiers."""

    @staticmethod
    def _page():
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        loc.first.scroll_into_view_if_needed = AsyncMock()
        loc.first.bounding_box = AsyncMock(
            return_value={"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
        )
        page.locator = MagicMock(return_value=loc)
        page.screenshot = AsyncMock(side_effect=lambda: _png(blank=False))
        page.goto = AsyncMock()
        page.set_viewport_size = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.evaluate = AsyncMock()
        return page, loc

    async def test_goto_navigates_for_multi_url_shot(self):
        page, _loc = self._page()
        steps = [
            {"goto": "https://a.example/", "capture_id": "a", "settle_ms": 0},
            {"goto": "https://b.example/pricing", "capture_id": "b", "settle_ms": 0},
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps, base_url="https://base.example/")
        assert [cid for cid, _ in shots] == ["a", "b"]
        # Each shot is captured at the URL its step named, in order — a count of
        # 2 also holds if both navigations went to the same page.
        assert [c.args[0] for c in page.goto.await_args_list] == [
            "https://a.example/",
            "https://b.example/pricing",
        ]

    async def test_goto_skipped_when_equal_to_current_url(self):
        page, _loc = self._page()
        steps = [{"goto": "https://base.example/", "capture_id": "a", "settle_ms": 0}]
        await UrlRecorder._walk_and_capture(page, steps, base_url="https://base.example/")
        assert page.goto.await_count == 0  # already at base — no re-navigation

    async def test_goto_retries_then_recovers(self):
        # networkidle times out (flaky for SPAs) but a more lenient strategy lands.
        page, _loc = self._page()
        page.goto = AsyncMock(side_effect=[PlaywrightError("networkidle timeout"), None])
        steps = [{"goto": "https://b.example/governance", "capture_id": "b", "settle_ms": 0}]
        shots = await UrlRecorder._walk_and_capture(page, steps, base_url="https://base.example/")
        assert [cid for cid, _ in shots] == ["b"]  # recovered → screenshot taken
        assert page.goto.await_count == 2  # retried after the first failure

    async def test_failed_goto_skips_screenshot_no_stale_duplicate(self):
        # Every navigation attempt fails → skip the shot rather than capture a
        # stale duplicate of the previous screen (the bug this fix closes).
        page, _loc = self._page()
        page.goto = AsyncMock(side_effect=PlaywrightError("nav failed"))
        steps = [
            {"goto": "https://a.example/", "capture_id": "a", "settle_ms": 0},
            {"goto": "https://b.example/governance", "capture_id": "b", "settle_ms": 0},
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps, base_url="https://base.example/")
        assert shots == []  # no screenshots — never a stale duplicate
        assert page.screenshot.await_count == 0

    async def test_viewport_resizes_then_dedups(self):
        page, _loc = self._page()
        steps = [
            {"viewport": "tablet", "capture_id": "a", "settle_ms": 0},
            {"viewport": "tablet", "capture_id": "b", "settle_ms": 0},  # same → no resize
            {"viewport": "mobile", "capture_id": "c", "settle_ms": 0},
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps, base_viewport="desktop")
        # Every step still produces a shot; only the redundant RESIZE is dropped.
        assert [cid for cid, _ in shots] == ["a", "b", "c"]
        # tablet, then mobile — step "b" repeats tablet and must not re-resize,
        # because a resize costs a relayout and can reflow the page mid-capture.
        assert [c.args[0] for c in page.set_viewport_size.await_args_list] == [
            {"width": 1024, "height": 768},
            {"width": 390, "height": 844},
        ]

    async def test_viewport_desktop_seed_is_noop(self):
        page, _loc = self._page()
        steps = [{"viewport": "desktop", "capture_id": "a", "settle_ms": 0}]
        await UrlRecorder._walk_and_capture(page, steps, base_viewport="desktop")
        assert page.set_viewport_size.await_count == 0

    async def test_scroll_to_selector_and_bottom(self):
        page, loc = self._page()
        steps = [
            {"scroll_to": ".features", "capture_id": "a", "settle_ms": 0},
            {"scroll_to": "bottom", "capture_id": "b", "settle_ms": 0},
        ]
        await UrlRecorder._walk_and_capture(page, steps)
        assert loc.first.scroll_into_view_if_needed.await_count == 1
        assert page.evaluate.await_count == 3
        assert any("scrollHeight" in call.args[0] for call in page.evaluate.await_args_list)

    async def test_wait_for_selector_awaited(self):
        page, _loc = self._page()
        steps = [{"wait_for": "#main", "capture_id": "a", "settle_ms": 0}]
        await UrlRecorder._walk_and_capture(page, steps)
        assert page.wait_for_selector.await_count == 1
        assert page.wait_for_selector.await_args.args[0] == "#main"

    async def test_wait_for_timeout_still_captures(self):
        page, _loc = self._page()
        page.wait_for_selector = AsyncMock(side_effect=PlaywrightError("timeout"))
        steps = [{"wait_for": "#never", "capture_id": "a", "settle_ms": 0}]
        shots = await UrlRecorder._walk_and_capture(page, steps)
        assert [cid for cid, _ in shots] == ["a"]  # soft-fail keeps the shot

    async def test_legacy_steps_unchanged(self):
        # No new fields → identical behaviour to before (regression guard).
        page, loc = self._page()
        steps = [
            {"action": "click", "selector": ".nav", "capture_id": "a", "settle_ms": 0},
            {"action": "wait", "capture_id": "b", "settle_ms": 0},
        ]
        shots = await UrlRecorder._walk_and_capture(page, steps)
        assert [cid for cid, _ in shots] == ["a", "b"]
        assert page.goto.await_count == 0
        assert page.set_viewport_size.await_count == 0
        assert page.wait_for_selector.await_count == 0
        assert loc.first.click.await_count == 1  # only the click step

    async def test_clip_walker_applies_setup_modifiers(self):
        # Parity: the clip recorder honours goto + viewport too (setup runs
        # before the clip start mark, so the navigation is not in-frame).
        page, _loc = self._page()
        steps = [
            {"goto": "https://x.example/", "viewport": "mobile", "capture_id": "a", "settle_ms": 0}
        ]
        boundaries, stills, _tracks = await UrlRecorder._walk_and_record_clips(
            page,
            steps,
            rec_start=0.0,
            dwell_ms=0.0,
            base_url="https://base/",
            base_viewport="desktop",
        )
        assert [cid for cid, _ in stills] == ["a"]
        assert [cid for cid, *_ in boundaries] == ["a"]
        assert page.goto.await_args.args[0] == "https://x.example/"
        assert page.set_viewport_size.await_args.args[0] == {"width": 390, "height": 844}


class TestScreenCaptureStepSchema:
    """The new per-step directive fields round-trip and validate."""

    def test_accepts_new_fields(self):
        from src.api.schemas.recording import ScreenCaptureStep

        step = ScreenCaptureStep(
            capture_id="s1",
            goto="https://example.com/pricing",
            scroll_to=".features",
            wait_for="#ready",
            viewport="tablet",
        )
        dumped = step.model_dump()
        assert dumped["goto"] == "https://example.com/pricing"
        assert dumped["scroll_to"] == ".features"
        assert dumped["wait_for"] == "#ready"
        assert dumped["viewport"] == "tablet"

    def test_defaults_are_none(self):
        from src.api.schemas.recording import ScreenCaptureStep

        step = ScreenCaptureStep(capture_id="s1")
        assert step.goto is None
        assert step.scroll_to is None
        assert step.wait_for is None
        assert step.viewport is None

    def test_rejects_bad_viewport(self):
        from pydantic import ValidationError

        from src.api.schemas.recording import ScreenCaptureStep

        with pytest.raises(ValidationError):
            ScreenCaptureStep(capture_id="s1", viewport="watch")  # ty: ignore[invalid-argument-type]

    def test_rejects_non_http_goto(self):
        from pydantic import ValidationError

        from src.api.schemas.recording import ScreenCaptureStep

        with pytest.raises(ValidationError):
            ScreenCaptureStep(capture_id="s1", goto="ftp://example.com")


class TestZipScreens:
    def test_zip_is_self_describing(self, tmp_path):
        import zipfile

        shots = [("home", _png(blank=False)), ("build", _png(blank=True))]
        zip_path = UrlRecorder._zip_screens(shots, tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert sorted(zf.namelist()) == ["build.png", "home.png"]
            assert zf.read("home.png") == shots[0][1]


class TestCaptureScreens:
    async def test_success_returns_zip_result(self):
        recorder = _make_recorder()
        shots = [("home", _png(blank=False)), ("build", _png(blank=False))]
        fake_zip = MagicMock()
        fake_zip.stat.return_value.st_size = 4242

        with (
            patch.object(
                recorder, "_capture_with_playwright", new_callable=AsyncMock, return_value=shots
            ),
            patch(
                "src.services.recording.url_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=fake_zip,
            ),
        ):
            result = await recorder.capture_screens(
                job_id="j-cap",
                url="https://commonpractice.nonagentic.ai/demo",
                steps=[{"selector": ".nav", "capture_id": "home"}],
            )

        assert result["success"] is True
        assert result["screen_ids"] == ["home", "build"]
        # The producer must DECLARE that its ``video_path`` is an archive. The
        # success path dispatches its integrity check on this field; if the
        # producer stops emitting it the zip is fed to ffprobe again and every
        # screens job fails, with the consumer-side tests still green.
        assert result["artifact_kind"] == ARTIFACT_KIND_SCREENS_ZIP
        assert result["file_size_bytes"] == 4242
        assert result["has_audio"] is False

    async def test_no_screens_is_failure(self):
        recorder = _make_recorder()
        with patch.object(
            recorder, "_capture_with_playwright", new_callable=AsyncMock, return_value=[]
        ):
            result = await recorder.capture_screens(
                job_id="j-cap", url="https://x.com/demo", steps=[{"action": "wait"}]
            )
        assert result["success"] is False
        assert "no screens" in result["error"]

    async def test_playwright_error_caught(self):
        recorder = _make_recorder()
        with patch.object(
            recorder,
            "_capture_with_playwright",
            new_callable=AsyncMock,
            side_effect=PlaywrightError("nav failed"),
        ):
            result = await recorder.capture_screens(
                job_id="j-cap", url="https://x.com/demo", steps=[{"capture_id": "a"}]
            )
        assert result["success"] is False
        assert "nav failed" in result["error"]


class TestWalkAndRecordClips:
    """Clip mode: per-screen clip window (start before the click) + still."""

    @staticmethod
    def _page():
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        loc.first.bounding_box = AsyncMock(
            return_value={"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
        )
        page.locator = MagicMock(return_value=loc)
        page.screenshot = AsyncMock(side_effect=lambda: _png(blank=False))
        return page, loc

    async def test_boundaries_and_stills_per_capture_id(self):
        page, _loc = self._page()
        steps = [
            {"action": "click", "selector": ".home", "capture_id": "home"},
            {"action": "click", "selector": ".build", "capture_id": "build"},
        ]
        # monotonic() is read t_start (pre-click) then t_end (post-dwell) per
        # capture step: home → (1, 7), build → (8, 14); rec_start=0.
        clock = iter([1.0, 7.0, 8.0, 14.0])
        with (
            patch(
                "src.services.recording.url_recorder.time.monotonic",
                side_effect=lambda: next(clock),
            ),
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
        ):
            boundaries, stills, _tracks = await UrlRecorder._walk_and_record_clips(
                page, steps, rec_start=0.0, dwell_ms=5000.0
            )
        # t_start is pre-roll (0.3s) BEFORE the click instant.
        assert boundaries == [("home", 0.7, 7.0), ("build", 7.7, 14.0)]
        assert [cid for cid, _ in stills] == ["home", "build"]

    async def test_step_without_capture_id_advances_without_boundary(self):
        page, loc = self._page()
        steps = [
            {"action": "click", "selector": ".persona", "settle_ms": 0},  # no capture_id
            {"action": "click", "selector": ".gov", "capture_id": "gov"},
        ]
        clock = iter([5.0, 11.0])  # only the captured step reads the clock
        with (
            patch(
                "src.services.recording.url_recorder.time.monotonic",
                side_effect=lambda: next(clock),
            ),
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
        ):
            boundaries, _stills, _tracks = await UrlRecorder._walk_and_record_clips(
                page, steps, rec_start=0.0, dwell_ms=0.0
            )
        assert [cid for cid, _, _ in boundaries] == ["gov"]
        assert loc.first.click.await_count == 2  # both steps clicked
        assert page.screenshot.await_count == 1  # only the captured step

    async def test_duplicate_capture_id_skipped(self):
        page, _loc = self._page()
        steps = [
            {"action": "click", "selector": ".a", "capture_id": "dup"},
            {"action": "click", "selector": ".b", "capture_id": "dup"},
        ]
        clock = iter([1.0, 3.0, 4.0, 6.0])
        with (
            patch(
                "src.services.recording.url_recorder.time.monotonic",
                side_effect=lambda: next(clock),
            ),
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
        ):
            boundaries, _stills, _tracks = await UrlRecorder._walk_and_record_clips(
                page, steps, rec_start=0.0, dwell_ms=0.0
            )
        assert [cid for cid, _, _ in boundaries] == ["dup"]


class TestTrimClips:
    def test_trims_one_clip_per_boundary_and_skips_nonpositive(self, tmp_path):
        boundaries = [("home", 1.0, 7.0), ("bad", 5.0, 5.0), ("build", 8.0, 12.0)]
        with patch.object(FFmpegOps, "trim_reencode") as mock_trim:
            mock_trim.side_effect = lambda webm, out, **kw: out
            clips = UrlRecorder._trim_clips(Path("/walk.webm"), boundaries, 23, tmp_path)
        assert [cid for cid, _ in clips] == ["home", "build"]  # zero-length dropped
        assert mock_trim.call_count == 2
        # Frame-accurate window passed through.
        first = mock_trim.call_args_list[0]
        assert first.kwargs["start_s"] == 1.0
        assert first.kwargs["duration_s"] == 6.0


class TestZipClips:
    def test_zip_holds_both_clips_and_stills(self, tmp_path):
        import zipfile

        clip = tmp_path / "home.mp4"
        clip.write_bytes(b"fake-mp4")
        stills = [("home", _png(blank=False)), ("build", _png(blank=True))]
        zip_path = UrlRecorder._zip_clips([("home", clip)], stills, None, tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert sorted(zf.namelist()) == ["build.png", "home.mp4", "home.png"]
            assert zf.read("home.mp4") == b"fake-mp4"


class TestCaptureScreenClips:
    async def test_success_returns_zip_result_with_both_assets(self):
        recorder = _make_recorder()
        boundaries = [("home", 1.0, 7.0)]
        stills = [("home", _png(blank=False))]
        fake_zip = MagicMock()
        fake_zip.stat.return_value.st_size = 8484

        with (
            patch.object(
                recorder,
                "_capture_clips_with_playwright",
                new_callable=AsyncMock,
                return_value=(Path("/walk.webm"), boundaries, stills, {}),
            ),
            patch(
                "src.services.recording.url_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                side_effect=[[("home", Path("/home.mp4"))], fake_zip],
            ),
        ):
            result = await recorder.capture_screen_clips(
                job_id="j-clip",
                url="https://commonpractice.nonagentic.ai/demo",
                steps=[{"selector": ".nav", "capture_id": "home"}],
                recording_config={"clip_dwell_ms": 4000},
            )

        assert result["success"] is True
        assert result["screen_ids"] == ["home"]
        assert result["artifact_kind"] == ARTIFACT_KIND_SCREENS_ZIP
        assert result["file_size_bytes"] == 8484
        assert result["has_audio"] is False

    async def test_nothing_captured_is_failure(self):
        recorder = _make_recorder()
        with patch.object(
            recorder,
            "_capture_clips_with_playwright",
            new_callable=AsyncMock,
            return_value=(Path("/walk.webm"), [], [], {}),
        ):
            result = await recorder.capture_screen_clips(
                job_id="j-clip", url="https://x.com/demo", steps=[{"action": "wait"}]
            )
        assert result["success"] is False
        assert "no screens" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestClipRecordDims:
    """``_clip_record_dims`` sizes the clip canvas to a uniform per-step viewport
    so a mobile/portrait shotlist doesn't render into a corner of a desktop clip
    (the gray-surround bug)."""

    def test_uniform_mobile_records_at_mobile(self) -> None:
        steps = [
            {"capture_id": "a", "viewport": "mobile"},
            {"capture_id": "b", "viewport": "mobile"},
        ]
        assert UrlRecorder._clip_record_dims(steps, 1920, 1080) == ("mobile", 390, 844)

    def test_mixed_viewports_keep_default(self) -> None:
        steps = [
            {"capture_id": "a", "viewport": "mobile"},
            {"capture_id": "b", "viewport": "desktop"},
        ]
        assert UrlRecorder._clip_record_dims(steps, 1920, 1080) == (None, 1920, 1080)

    def test_absent_viewport_keeps_default(self) -> None:
        steps = [{"capture_id": "a"}, {"capture_id": "b"}]
        assert UrlRecorder._clip_record_dims(steps, 1280, 720) == (None, 1280, 720)

    def test_non_capturing_steps_are_ignored(self) -> None:
        # A persona-switch step (no capture_id) at desktop must not break a uniform
        # mobile capture set.
        steps = [
            {"viewport": "desktop"},  # advances only, no screenshot
            {"capture_id": "a", "viewport": "mobile"},
        ]
        assert UrlRecorder._clip_record_dims(steps, 1920, 1080) == ("mobile", 390, 844)


class TestApplyStepInteractionFocus:
    """`_apply_step_interaction` measures the clicked element's post-reveal
    geometry (the C15 focus/cursor keyframe) only in clip mode."""

    @staticmethod
    def _page(box):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        loc.first.bounding_box = AsyncMock(return_value=box)
        page.locator = MagicMock(return_value=loc)
        return page, loc

    async def test_click_returns_scaled_focus_record(self):
        page, _loc = self._page({"x": 100.0, "y": 50.0, "width": 40.0, "height": 20.0})
        rec = await UrlRecorder._apply_step_interaction(
            page, {"action": "click", "selector": ".cta"}, 0, capture_scale=2.0
        )
        assert rec == {
            "bbox": {"x": 200.0, "y": 100.0, "w": 80.0, "h": 40.0},
            "action": "click",
            "cx": 240.0,
            "cy": 120.0,
        }

    async def test_scale_one_is_identity(self):
        page, _loc = self._page({"x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0})
        rec = await UrlRecorder._apply_step_interaction(
            page, {"action": "click", "selector": ".x"}, 0, capture_scale=1.0
        )
        assert rec is not None
        assert rec["bbox"] == {"x": 5.0, "y": 6.0, "w": 7.0, "h": 8.0}

    async def test_screenshot_mode_captures_no_focus(self):
        # capture_scale=None (the screenshot walker) must never call bounding_box
        # and always returns None — its byte output stays unchanged.
        page, loc = self._page({"x": 1.0, "y": 1.0, "width": 1.0, "height": 1.0})
        rec = await UrlRecorder._apply_step_interaction(
            page, {"action": "click", "selector": ".x"}, 0
        )
        assert rec is None
        loc.first.bounding_box.assert_not_awaited()

    async def test_wait_step_has_no_focus(self):
        page, _loc = self._page({"x": 1.0, "y": 1.0, "width": 1.0, "height": 1.0})
        rec = await UrlRecorder._apply_step_interaction(
            page, {"action": "wait"}, 0, capture_scale=2.0
        )
        assert rec is None

    async def test_hidden_element_yields_no_focus(self):
        # bounding_box returns None for a display:none element ⇒ no focus entry.
        page, _loc = self._page(None)
        rec = await UrlRecorder._apply_step_interaction(
            page, {"action": "click", "selector": ".hidden"}, 0, capture_scale=2.0
        )
        assert rec is None


class TestTrackSidecar:
    """`_walk_and_record_clips` builds the C15 track; `_zip_clips` packs a
    sidecar only for clips with a real focus keyframe."""

    @staticmethod
    def _page(box):
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        loc = MagicMock()
        loc.first.click = AsyncMock()
        loc.first.bounding_box = AsyncMock(return_value=box)
        page.locator = MagicMock(return_value=loc)
        page.screenshot = AsyncMock(side_effect=lambda: _png(blank=False))
        return page, loc

    async def test_track_scaled_into_one_frame_space(self):
        page, _loc = self._page({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
        steps = [{"action": "click", "selector": ".cta", "capture_id": "home", "settle_ms": 0}]
        clock = iter([1.0, 7.0])
        with (
            patch(
                "src.services.recording.url_recorder.time.monotonic",
                side_effect=lambda: next(clock),
            ),
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
        ):
            _b, _s, tracks = await UrlRecorder._walk_and_record_clips(
                page,
                steps,
                rec_start=0.0,
                dwell_ms=0.0,
                capture_scale=2.0,
                viewport_w=1920,
                viewport_h=1080,
            )
        track = tracks["home"]
        assert track["viewport"] == {"w": 3840, "h": 2160}
        assert track["focus"] == [
            {"t": 0.3, "bbox": {"x": 20.0, "y": 40.0, "w": 60.0, "h": 80.0}, "action": "click"}
        ]
        assert track["cursor"] == [{"t": 0.3, "x": 50.0, "y": 80.0, "click": True}]

    async def test_wait_clip_has_empty_focus(self):
        page, _loc = self._page(None)
        steps = [{"action": "wait", "capture_id": "held", "settle_ms": 0}]
        clock = iter([1.0, 3.0])
        with (
            patch(
                "src.services.recording.url_recorder.time.monotonic",
                side_effect=lambda: next(clock),
            ),
            patch("src.services.recording.url_recorder.asyncio.sleep", new_callable=AsyncMock),
        ):
            _b, _s, tracks = await UrlRecorder._walk_and_record_clips(
                page, steps, rec_start=0.0, dwell_ms=0.0, viewport_w=1920, viewport_h=1080
            )
        assert tracks["held"]["focus"] == []

    def test_zip_writes_sidecar_only_for_non_empty_focus(self, tmp_path):
        import json
        import zipfile

        clip = tmp_path / "home.mp4"
        clip.write_bytes(b"fake-mp4")
        stills = [("home", _png(blank=False)), ("held", _png(blank=False))]
        tracks = {
            "home": {
                "focus": [
                    {"t": 0.3, "bbox": {"x": 1.0, "y": 2.0, "w": 3.0, "h": 4.0}, "action": "click"}
                ],
                "cursor": [{"t": 0.3, "x": 2.5, "y": 4.0, "click": True}],
                "viewport": {"w": 3840, "h": 2160},
            },
            "held": {"focus": [], "cursor": [], "viewport": {"w": 3840, "h": 2160}},
        }
        zip_path = UrlRecorder._zip_clips([("home", clip)], stills, tracks, tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert "home.track.json" in names
            assert "held.track.json" not in names  # empty focus ⇒ no sidecar
            data = json.loads(zf.read("home.track.json"))
        assert data["viewport"] == {"w": 3840, "h": 2160}
        assert data["focus"][0]["action"] == "click"
        assert data["cursor"][0]["click"] is True

    def test_zip_without_tracks_is_backward_compatible(self, tmp_path):
        import zipfile

        clip = tmp_path / "home.mp4"
        clip.write_bytes(b"fake-mp4")
        stills = [("home", _png(blank=False))]
        zip_path = UrlRecorder._zip_clips([("home", clip)], stills, None, tmp_path)
        with zipfile.ZipFile(zip_path) as zf:
            assert sorted(zf.namelist()) == ["home.mp4", "home.png"]


class TestCaptureClipsScaling:
    """`_capture_clips_with_playwright` threads capture_scale into the Playwright
    context (device_scale_factor + record_video_size) and the sidecar space."""

    @staticmethod
    def _fake_playwright(captured):
        loc = MagicMock()
        loc.first.click = AsyncMock()
        loc.first.bounding_box = AsyncMock(
            return_value={"x": 100.0, "y": 50.0, "width": 40.0, "height": 20.0}
        )
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)
        page.goto = AsyncMock()
        page.screenshot = AsyncMock(side_effect=lambda: _png(blank=False))
        page.locator = MagicMock(return_value=loc)

        context = MagicMock()
        context.add_cookies = AsyncMock()
        context.new_page = AsyncMock(return_value=page)
        context.close = AsyncMock()
        # The recorder installs the SSRF route guard on every context it makes,
        # so ``route`` -- and ``route_web_socket``, which the guard registers so
        # ws:// targets are judged too, and ``add_init_script``, which it uses to
        # remove the unroutable ``SharedWorker`` channel -- are part of the
        # contract this double has to satisfy. A MagicMock attribute is NOT
        # awaitable, so leaving any of them off makes the guard raise
        # `TypeError` here rather than in production, which reads as "the guard
        # is broken" instead of "the double is narrower than BrowserContext".
        #
        # None of the three touch ``browser.new_context`` kwargs, which is all
        # ``captured`` records, so the scale assertions below are unaffected.
        context.route = AsyncMock()
        context.route_web_socket = AsyncMock()
        context.add_init_script = AsyncMock()

        async def _new_context(**kwargs):
            captured.update(kwargs)
            # Playwright drops the webm into record_video_dir on context close;
            # emulate it so the post-context glob finds a file.
            Path(kwargs["record_video_dir"], "v.webm").write_bytes(b"webm")
            return context

        browser = MagicMock()
        browser.new_context = AsyncMock(side_effect=_new_context)
        browser.close = AsyncMock()
        chromium = MagicMock()
        chromium.launch = AsyncMock(return_value=browser)
        pw = MagicMock()
        pw.chromium = chromium
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=pw)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    async def _run(self, scale_cfg):
        recorder = _make_recorder()
        captured: dict = {}
        staging = Path(tempfile.mkdtemp())
        steps = [{"action": "click", "selector": ".cta", "capture_id": "home", "settle_ms": 0}]
        with patch(
            "playwright.async_api.async_playwright",
            return_value=self._fake_playwright(captured),
        ):
            result = await recorder._capture_clips_with_playwright(
                "job",
                "https://x/",
                None,
                recorder._resolve_config(scale_cfg),
                steps,
                0.0,
                None,
                staging,
            )
        return captured, result

    async def test_scale_two_sets_device_scale_and_video_size(self):
        captured, (_webm, _b, _s, tracks) = await self._run({"capture_scale": 2.0})
        assert captured["device_scale_factor"] == 2.0
        assert captured["record_video_size"] == {"width": 3840, "height": 2160}
        assert captured["viewport"] == {"width": 1920, "height": 1080}  # CSS viewport unscaled
        track = tracks["home"]
        assert track["viewport"] == {"w": 3840, "h": 2160}
        assert track["focus"][0]["bbox"] == {"x": 200.0, "y": 100.0, "w": 80.0, "h": 40.0}

    async def test_scale_one_is_byte_identical_context(self):
        captured, _ = await self._run(None)
        # No device_scale_factor at 1.0 ⇒ recorded bytes match the pre-feature path.
        assert "device_scale_factor" not in captured
        assert captured["record_video_size"] == {"width": 1920, "height": 1080}


class TestCredentialScopeHopHeaders:
    """The per-hop decision the SSRF guard makes (real-browser proof lives in
    tests/integration/test_credential_scope.py)."""

    def _scope(self):
        from src.shared.security.playwright_guard import _CredentialScope

        return _CredentialScope("https://app.example", {"Authorization": "Bearer t"})

    def test_bound_origin_hop_gets_the_header(self):
        h = self._scope().hop_headers(
            {"accept": "*/*"}, "https://app.example/a", "https://app.example/b"
        )
        assert h["authorization"] == "Bearer t"

    def test_other_origin_hop_gets_none(self):
        h = self._scope().hop_headers(
            {"accept": "*/*"}, "https://app.example/a", "https://evil.example/x"
        )
        assert "authorization" not in {k.lower() for k in h}

    def test_cross_origin_redirect_drops_page_set_credentials(self):
        from src.shared.security.playwright_guard import _CredentialScope

        h = _CredentialScope(None, None).hop_headers(
            {"Authorization": "Bearer page-set", "accept": "*/*"},
            "https://app.example/a",
            "https://evil.example/landing",
        )
        assert "Authorization" not in h and h["accept"] == "*/*"
