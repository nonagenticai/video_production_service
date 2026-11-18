"""Request/response Pydantic models for video_recorder_service."""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator
from w2v_common.net import SSRFError, validate_public_url

from src.services.recording.quality_presets import QUALITY_PRESETS

_PART_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")

# --- Request Schemas ---


class InteractionStep(BaseModel):
    """One scripted step performed *while recording* a live URL.

    Lets the recorder click through an interactive demo (e.g. a platform's
    nav sections) so the clip shows the product being used, not just scrolled.
    ``selector`` is a Playwright selector — CSS or the ``text=`` engine
    (e.g. ``.nav-item >> text="Build"`` for an exact text-node match). Each
    step soft-fails: a missed target is logged and skipped, never aborts the
    recording. ``hold_ms`` keeps the resulting screen on-camera after the
    action.
    """

    action: Literal["click", "wait", "scroll"] = "click"
    selector: str | None = Field(
        None, max_length=512, description="Playwright selector to click (action=click)"
    )
    pixels: int | None = Field(
        None, ge=-20000, le=20000, description="Vertical px to scroll (action=scroll)"
    )
    hold_ms: int = Field(
        2500, ge=0, le=30000, description="Pause after the action so the screen is on-camera"
    )
    timeout_ms: int = Field(
        8000, ge=200, le=60000, description="Max wait for a click target to be actionable"
    )


class ScreenCaptureStep(BaseModel):
    """One step of a single-pass per-screen capture of a live SPA.

    The capturer navigates the base URL once, then walks this ordered step
    list. Per step it applies, in a fixed order: ``viewport`` (resize) →
    ``goto`` (full navigation to a different absolute URL) → ``click`` (advance
    the SPA in-place) → ``scroll_to`` (bring a section on-screen) → ``wait_for``
    (await a content selector) → ``settle_ms`` pause → ONE screenshot iff
    ``capture_id`` is set. Every modifier soft-fails independently (a missed
    target is logged and skipped, like :class:`InteractionStep`), so a stale
    selector never costs the screenshot. A step with no ``capture_id`` (e.g. a
    persona switch that only reveals nav) advances without capturing — needed
    because some nav is ``display:none`` until a prior click.

    ``selector`` / ``scroll_to`` / ``wait_for`` are Playwright selectors (CSS or
    the ``text=`` engine, e.g. ``.nav-item >> text="Build"``); ``scroll_to`` also
    accepts the literal ``"bottom"`` to scroll to the page foot. ``goto`` is for
    multi-page shotlists — a full navigation, distinct from an in-SPA ``click``;
    it is a no-op when it already equals the current URL. ``viewport`` overrides
    the context-wide viewport for this step (and stays in effect until the next
    ``viewport`` step).
    """

    capture_id: str | None = Field(
        None,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Screen id; screenshot taken (and keyed by this) when set",
    )
    action: Literal["click", "wait"] = "click"
    selector: str | None = Field(
        None, max_length=512, description="Playwright selector to click (action=click)"
    )
    settle_ms: int = Field(
        800, ge=0, le=30000, description="Pause after the action before the screenshot"
    )
    goto: str | None = Field(
        None,
        max_length=2048,
        description="Full navigation to this absolute http(s) URL before the shot (multi-page)",
    )
    scroll_to: str | None = Field(
        None,
        max_length=512,
        description="Scroll a selector into view before the shot, or 'bottom' for the page foot",
    )
    wait_for: str | None = Field(
        None, max_length=512, description="Await this selector before taking the screenshot"
    )
    viewport: Literal["desktop", "tablet", "mobile"] | None = Field(
        None, description="Per-step viewport override (else the context-wide viewport)"
    )

    @field_validator("goto", mode="before")
    @classmethod
    def _validate_goto(cls, v: str | None) -> str | None:
        return _validate_url_value(v) if v else v


class RecordingConfigInput(BaseModel):
    """Optional recording configuration overrides."""

    viewport_width: int | None = Field(None, ge=320, le=7680, description="Viewport width")
    viewport_height: int | None = Field(None, ge=240, le=4320, description="Viewport height")
    timeout_ms: int | None = Field(None, ge=1000, le=600000, description="Test timeout in ms")
    slow_mo_ms: int | None = Field(None, ge=0, le=10000, description="Slow motion delay in ms")
    color_scheme: str | None = Field(None, description="Color scheme: light or dark")
    headless: bool | None = Field(None, description="Run browser in headless mode")
    max_retries: int | None = Field(None, ge=0, le=10, description="Max retry attempts")
    output_format: str | None = Field(
        None, description="Output format: 'mp4' or 'webm' (overrides config default)"
    )
    quality_preset: str | None = Field(
        None,
        description="Quality preset: 'preview' (720p low), 'standard' (1080p), "
        "'high_quality' (1080p high CRF), '2k' (1440p), '4k' (2160p). "
        "Overrides viewport dimensions.",
    )
    post_load_delay_ms: int | None = Field(
        None,
        ge=0,
        le=30000,
        description=(
            "Extra settle delay (ms) after the page reaches 'networkidle' and "
            "before the scroll is recorded. Data-driven SPAs often render their "
            "main content via XHR *after* the network goes idle, so without a "
            "settle the recording captures a blank shell. Setting this also "
            "enables trimming the blank page-load lead-in from the final clip."
        ),
    )
    wait_for_selector: str | None = Field(
        None,
        max_length=512,
        description=(
            "Optional CSS selector to wait for (best-effort) after navigation, "
            "before recording the scroll — ensures real content is on screen, "
            "not a blank shell. Combined with post_load_delay_ms."
        ),
    )
    interaction_steps: list[InteractionStep] | None = Field(
        None,
        max_length=40,
        description=(
            "Optional scripted clicks/waits/scrolls performed while recording, "
            "to walk through an interactive demo. When present they REPLACE the "
            "auto-scroll (the steps define the clip). Each step soft-fails."
        ),
    )
    capture_scale: float | None = Field(
        None,
        ge=1.0,
        le=3.0,
        description=(
            "Record clips at this device-scale-factor (1.0-3.0) for zoom "
            "headroom. Scales the recorded video pixel dimensions and the "
            "focus/cursor track sidecar's bbox/cursor coordinates and viewport "
            "into one common frame-pixel space. Unset (or 1.0) keeps the "
            "recording byte-identical to the un-scaled capture."
        ),
    )

    @field_validator("color_scheme", mode="before")
    @classmethod
    def _validate_color_scheme(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"light", "dark", "no-preference"}
        if v not in allowed:
            msg = f"color_scheme must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("output_format", mode="before")
    @classmethod
    def _validate_output_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"mp4", "webm"}
        if v not in allowed:
            msg = f"output_format must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("quality_preset", mode="before")
    @classmethod
    def _validate_quality_preset(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"preview", "standard", "high_quality", "2k", "4k"}
        if v not in allowed:
            msg = f"quality_preset must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v


# Maximum bytes per env var key/value (defence against memory exhaustion).
_ENV_KEY_MAX_LEN = 64
_ENV_VALUE_MAX_LEN = 4096
# Allowed key shape: uppercase ASCII identifier. Mirrors POSIX env-var convention.
_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# Denylist of dangerous env-var key patterns. Anything matching one of these
# (case-insensitive) is rejected outright — including future variants we
# haven't enumerated. This is the security-critical primitive: PATH /
# LD_PRELOAD / DYLD_* / NODE_OPTIONS / PYTHON* / NPM_CONFIG_* / BROWSER /
# CHROME_* / PLAYWRIGHT_* are the verified RCE / data-exfil vectors against
# the `npx playwright test` subprocess sink (see scan.md §S1).
_ENV_KEY_DENYLIST_PREFIXES: tuple[str, ...] = (
    "LD_",
    "DYLD_",
    "PYTHON",
    "NODE_",
    "NPM_CONFIG_",
    "CHROME_",
    "PLAYWRIGHT_",
)
_ENV_KEY_DENYLIST_EXACT: frozenset[str] = frozenset(
    {
        "PATH",
        "NODE_OPTIONS",
        "PYTHONPATH",
        "BROWSER",
    }
)


def _validate_env_var_key(key: object) -> None:
    """Reject env-var keys that could be used to hijack the subprocess.

    Both the Pydantic boundary and the subprocess-env merge sink call this
    (defence-in-depth — `auth_config` is persisted on the job row at
    create-time and replayed later by the recorder; if the schema is ever
    relaxed the sink must still hold the line).
    """
    if not isinstance(key, str):
        msg = f"env_vars key must be a string, got {type(key).__name__}"
        raise ValueError(msg)
    if not key:
        msg = "env_vars key must be non-empty"
        raise ValueError(msg)
    if len(key) > _ENV_KEY_MAX_LEN:
        msg = f"env_vars key exceeds {_ENV_KEY_MAX_LEN} characters"
        raise ValueError(msg)
    if not _ENV_KEY_RE.match(key):
        msg = (
            f"env_vars key '{key}' must match {_ENV_KEY_RE.pattern} "
            "(uppercase letters, digits, underscores; must start with a letter)"
        )
        raise ValueError(msg)
    upper = key.upper()
    if upper in _ENV_KEY_DENYLIST_EXACT:
        msg = f"env_vars key '{key}' is in the denylist (subprocess hijack risk)"
        raise ValueError(msg)
    for prefix in _ENV_KEY_DENYLIST_PREFIXES:
        if upper.startswith(prefix):
            msg = (
                f"env_vars key '{key}' starts with denylisted prefix "
                f"'{prefix}' (subprocess hijack risk)"
            )
            raise ValueError(msg)


def _validate_env_var_value(key: str, value: object) -> str:
    """Reject non-string values and values longer than the per-value cap."""
    if not isinstance(value, str):
        msg = f"env_vars[{key!r}] must be a string, got {type(value).__name__}"
        raise ValueError(msg)
    if len(value) > _ENV_VALUE_MAX_LEN:
        msg = f"env_vars[{key!r}] exceeds {_ENV_VALUE_MAX_LEN} characters (got {len(value)})"
        raise ValueError(msg)
    return value


def sanitize_auth_env_vars(raw: object) -> dict[str, str]:
    """Sink-side filter: drop any key/value that fails validation.

    Used at the subprocess-env merge point. Unlike the Pydantic validator
    (which raises 422 on bad input), this function silently drops invalid
    entries — its purpose is to defend against a poisoned/loosened DB row
    being replayed long after the request was accepted.
    """
    if not isinstance(raw, dict):
        return {}
    safe: dict[str, str] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        try:
            _validate_env_var_key(k)
            safe_v = _validate_env_var_value(k, v)
        except ValueError:
            continue
        safe[k] = safe_v
    return safe


class AuthConfigInput(BaseModel):
    """Authentication configuration for recording.

    `env_vars` is a tightly validated map of env vars the Playwright
    subprocess receives. Dangerous keys (PATH, LD_*, DYLD_*, NODE_OPTIONS,
    PYTHON*, NODE_*, NPM_CONFIG_*, BROWSER, CHROME_*, PLAYWRIGHT_*) are
    rejected at the Pydantic boundary AND re-filtered at the subprocess
    sink — see ``sanitize_auth_env_vars``.
    """

    auth_type: str | None = Field(default=None, description="Auth type: bearer, basic, form")
    env_vars: dict[str, str] | None = Field(
        default=None,
        description=(
            "Env vars to inject into subprocess. Keys must match "
            "^[A-Z][A-Z0-9_]*$ and not be in the denylist (PATH, LD_*, "
            "DYLD_*, NODE_OPTIONS, PYTHON*, NODE_*, NPM_CONFIG_*, BROWSER, "
            "CHROME_*, PLAYWRIGHT_*)."
        ),
    )

    @field_validator("env_vars", mode="before")
    @classmethod
    def _validate_env_vars(cls, v: object) -> dict[str, str] | None:
        if v is None:
            return None
        if not isinstance(v, dict):
            msg = "env_vars must be a mapping of string to string"
            raise ValueError(msg)
        result: dict[str, str] = {}
        for k, val in v.items():
            _validate_env_var_key(k)
            if not isinstance(k, str):
                msg = "env_vars key must be a string"
                raise ValueError(msg)
            result[k] = _validate_env_var_value(k, val)
        return result


class UrlAuthInfo(BaseModel):
    """How to authenticate the headless browser before recording a live URL.

    Field shape mirrors ``analysis_service``'s ``AuthInfo`` (the one
    ``website_review`` already collects in the frontend and forwards as an
    opaque dict) so the same brief block drives both the HTML scanner and
    this recorder. Unlike the httpx-based scanner, a Playwright browser CAN
    perform an interactive form login, so ``login_url`` is honoured here:
    when set (with username/password) the recorder submits the login form
    before navigating to ``url``.
    """

    type: str = Field("none", description="none | basic | bearer | cookies")
    # type="basic"
    username: str | None = None
    password: str | None = None
    # type="bearer"
    token: str | None = None
    # type="cookies" — list of {name, value, domain?} entries, or a raw header
    cookies: list[dict[str, str]] | None = None
    cookie_header: str | None = None
    # Optional interactive form login (Playwright-only; not in analysis_service)
    login_url: str | None = None
    username_selector: str | None = Field(
        None, description="CSS selector for the login form's username/email field"
    )
    password_selector: str | None = Field(
        None, description="CSS selector for the login form's password field"
    )
    submit_selector: str | None = Field(
        None, description="CSS selector for the login form's submit button"
    )

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, v: str | None) -> str:
        if v is None:
            return "none"
        allowed = {"none", "basic", "bearer", "cookies"}
        if v not in allowed:
            msg = f"auth_info type must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("login_url", mode="before")
    @classmethod
    def _validate_login_url(cls, v: str | None) -> str | None:
        """Guard ``login_url`` exactly like its sibling ``url`` / ``webhook_url`` fields.

        This is the **highest-severity** of the URL fields on this model, not the
        lowest: ``url_recorder`` hands it straight to ``page.goto(...)`` and then
        ``page.fill(...)``s the tenant's ``username`` / ``password`` into the
        page. An unguarded value therefore does not merely fetch an internal
        address — it **posts the tenant's credentials** to whatever host the
        request names (any tailnet or cluster-internal service, or an attacker's
        collector), and the resulting session cookies persist into the recorded
        video.

        ``None`` is allowed: interactive form login is opt-in, and the recorder
        skips the login step entirely when this is unset.
        """
        return _validate_login_url_value(v)


def _reject_internal_target(v: str, *, field: str) -> None:
    """Run the shared fleet SSRF guard, re-raised as a pydantic-friendly ``ValueError``.

    ``w2v_common.net.validate_public_url`` resolves the hostname and rejects any
    address in a private / reserved / loopback / link-local / multicast range —
    critically including RFC 6598 CGNAT ``100.64.0.0/10``, which is this
    estate's Tailscale tailnet (postgres, the docker registry and the k3s master
    all live there) and which ``ipaddress.ip_address(...).is_private`` reports as
    **False**. It also unwraps IPv4-mapped / 6to4 / Teredo IPv6 and fails closed
    on DNS failure.

    ``SSRFError`` subclasses ``ValueError``, so it is caught first. Both branches
    re-raise a plain ``ValueError`` because that is what pydantic converts into a
    field error — i.e. the API answers **422**, not a 500.
    """
    try:
        validate_public_url(v)
    except SSRFError as exc:
        msg = f"{field} must not target an internal or reserved network address ({exc})"
        raise ValueError(msg) from exc
    except ValueError as exc:
        msg = f"{field} is not a valid public HTTP(S) URL ({exc})"
        raise ValueError(msg) from exc


def _validate_url_value(v: str) -> str:
    """Reject non-HTTP(S), over-long, and internal-network URLs for the URL recorder.

    The cheap scheme and length checks run first so an obviously-bad value never
    costs a name lookup.

    NOTE: because this is a pydantic field validator, the SSRF guard performs a
    synchronous ``socket.getaddrinfo`` at **request-validation time**. That is
    intended here: refusing an internal target before a recording job row is
    created is worth one DNS lookup per request, and it means an unresolvable
    host is rejected rather than handed to Playwright.
    """
    if not v or not v.startswith(("http://", "https://")):
        msg = "url must start with 'http://' or 'https://'"
        raise ValueError(msg)
    if len(v) > 2048:
        msg = "url must not exceed 2048 characters"
        raise ValueError(msg)
    _reject_internal_target(v, field="url")
    return v


def _validate_scripts_dir_value(v: str) -> str:
    """Shared validation logic for scripts_dir fields."""
    if ".." in v:
        msg = "scripts_dir must not contain '..'"
        raise ValueError(msg)
    if not v.startswith("/"):
        msg = "scripts_dir must be an absolute path (starts with '/')"
        raise ValueError(msg)
    return v


def _validate_webhook_url_value(v: str | None) -> str | None:
    """Shared validation logic for webhook_url fields.

    ``webhook_url`` is an **outbound POST target**, so it is both an SSRF sink
    and a data-exfiltration sink: an unguarded value lets a tenant make this
    service POST job payloads at any tailnet or cluster-internal address. The
    same guard as :func:`_validate_url_value` therefore applies, after the cheap
    length and scheme checks.

    NOTE: as with the ``url`` field, this runs a synchronous
    ``socket.getaddrinfo`` at request-validation time. Accepted and intended.
    """
    if v is None:
        return v
    if len(v) > 2048:
        msg = "webhook_url must not exceed 2048 characters"
        raise ValueError(msg)
    if not v.startswith(("http://", "https://")):
        msg = "webhook_url must start with 'http://' or 'https://'"
        raise ValueError(msg)
    _reject_internal_target(v, field="webhook_url")
    return v


def _validate_login_url_value(v: str | None) -> str | None:
    """Shared validation logic for ``UrlAuthInfo.login_url``.

    ``login_url`` is a **credential** sink, not just a fetch sink: the recorder
    navigates a real browser there and types the tenant's username/password into
    the page. It gets the same scheme + length + internal-target checks as
    ``url``; the only difference is that ``None`` is permitted, because the
    interactive-login step is optional.

    NOTE: as with ``url``, the SSRF guard runs a synchronous
    ``socket.getaddrinfo`` at request-validation time. Accepted and intended —
    refusing an internal login target before a job row exists is worth one DNS
    lookup, and it is the only place the credentials can still be withheld.
    """
    if v is None:
        return v
    if len(v) > 2048:
        msg = "login_url must not exceed 2048 characters"
        raise ValueError(msg)
    if not v.startswith(("http://", "https://")):
        msg = "login_url must start with 'http://' or 'https://'"
        raise ValueError(msg)
    _reject_internal_target(v, field="login_url")
    return v


class ScrollConfig(BaseModel):
    """Scroll configuration for content recording."""

    enabled: bool = Field(True, description="Whether to auto-scroll the content")
    speed_px_per_second: int = Field(
        300, ge=50, le=5000, description="Scroll speed in pixels per second"
    )
    start_delay_seconds: float = Field(
        1.0, ge=0, le=30, description="Pause before scrolling starts"
    )
    end_delay_seconds: float = Field(
        2.0, ge=0, le=30, description="Pause after scrolling completes"
    )


class RecordContentRequest(BaseModel):
    """Record scrolling HTML content as a video."""

    html_content: str = Field(
        ..., min_length=1, max_length=5_000_000, description="Raw HTML content to render and record"
    )
    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Part identifier for the recording",
    )
    recording_config: RecordingConfigInput | None = None
    scroll_config: ScrollConfig | None = None
    webhook_url: str | None = Field(None, description="URL to POST on completion/failure")

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)


class RecordAnimationRequest(BaseModel):
    """Render a branded motion-graphics template to MP4 / WebM / GIF.

    Supply EITHER ``template_id`` (a registered template in the motion-template
    library) OR raw ``html`` (a self-contained page exposing
    ``window.RENDER_DURATION_MS`` and optionally ``window.render(t)``).
    ``params`` is injected as ``window.PARAMS`` so one template serves every
    brand. Capture is deterministic (frame-stepped), so the same request always
    yields the same bytes.
    """

    template_id: str | None = Field(
        None, max_length=128, description="Registered motion template id (e.g. 'social_card')"
    )
    html: str | None = Field(
        None, min_length=1, max_length=5_000_000, description="Raw self-contained template HTML"
    )
    params: dict[str, object] | None = Field(
        None, description="Brand/content overrides injected as window.PARAMS"
    )
    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Part identifier for the render",
    )
    fps: int = Field(30, ge=1, le=60, description="Frames per second")
    width: int | None = Field(None, ge=64, le=7680, description="Output width (default 1080)")
    height: int | None = Field(None, ge=64, le=4320, description="Output height (default 1080)")
    duration_ms: int | None = Field(
        None,
        ge=100,
        le=120000,
        description="Timeline length; falls back to the template's RENDER_DURATION_MS",
    )
    output_format: Literal["mp4", "webm", "gif"] = Field("mp4", description="Container/codec")
    transparent: bool = Field(
        False,
        description=(
            "Render with a transparent background so the clip can be "
            "composited OVER other footage (lower-third, logo bug) instead of "
            "cutting full-frame. Implies a VP9/WebM container carrying a "
            "yuva420p alpha plane: h264 has no alpha channel and gif's is "
            "1-bit, so an explicitly requested output_format of 'mp4' or "
            "'gif' is REJECTED rather than silently flattened to an opaque "
            "clip. Only overlay-designed templates accept it (full-frame card "
            "templates paint their own opaque stage and are refused); inline "
            "'html' owns its own backdrop and is always allowed. Left at the "
            "default False the render is byte-identical to one issued before "
            "this field existed."
        ),
    )
    quality_preset: str | None = Field(
        None,
        description=(
            "Encode quality preset (same names as recording_config.quality_preset "
            "on the other recording routes: 'preview', 'standard', 'high_quality', "
            "'2k', '4k'). Only the preset's CRF is applied — output dimensions stay "
            "width/height. Omitted ⇒ the service default ('standard', CRF 23), "
            "byte-identical to a request issued before this field existed. An "
            "unknown name is rejected (422), never silently downgraded."
        ),
    )
    webhook_url: str | None = Field(None, description="URL to POST on completion/failure")

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)

    @field_validator("quality_preset", mode="before")
    @classmethod
    def _validate_quality_preset(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in QUALITY_PRESETS:
            msg = f"quality_preset must be one of {sorted(QUALITY_PRESETS)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("template_id")
    @classmethod
    def _validate_template_id(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("template_id must match ^[a-zA-Z0-9_\\-]+$")
        return v

    @model_validator(mode="after")
    def _require_source(self) -> RecordAnimationRequest:
        if not self.template_id and not self.html:
            raise ValueError("either template_id or html must be provided")
        return self

    @model_validator(mode="after")
    def _resolve_transparent_container(self) -> RecordAnimationRequest:
        """Pin the container for a transparent render — never degrade silently.

        Alpha survives exactly one of our encode paths: VP9/WebM. The split is
        on whether the caller actually chose a format, which
        ``model_fields_set`` answers precisely:

        * format left at its "mp4" DEFAULT → upgrade to webm. "Give me a
          transparent clip" is unambiguous about intent, and failing a request
          over a value the caller never typed is hostile.
        * format EXPLICITLY 'mp4' or 'gif' → 422. The two instructions
          contradict each other, and the only way to honour the container is to
          throw the alpha away — which the caller would discover as an opaque
          box over their footage, one compile later.

        Assignment here does not re-trigger validation (``validate_assignment``
        is off), so this cannot recurse.
        """
        if not self.transparent:
            return self
        if "output_format" not in self.model_fields_set:
            self.output_format = "webm"
        elif self.output_format != "webm":
            msg = (
                f"transparent=true requires output_format='webm' (h264 has no "
                f"alpha plane, gif alpha is 1-bit); got '{self.output_format}'"
            )
            raise ValueError(msg)
        return self


class RecordUrlRequest(BaseModel):
    """Record a live URL (e.g. an authenticated SPA) as a scrolling video.

    Navigates a headless Chromium to ``url`` (optionally authenticating
    first via ``auth_info``), auto-scrolls, and records the viewport. One
    request yields one clip; callers wanting several pages issue one request
    per path and gather the results (mirrors how ``ai_image_gen`` fans out).
    """

    url: str = Field(
        ..., min_length=1, max_length=2048, description="Live URL to navigate and record"
    )
    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Part identifier for the recording",
    )
    auth_info: UrlAuthInfo | None = None
    recording_config: RecordingConfigInput | None = None
    scroll_config: ScrollConfig | None = None
    webhook_url: str | None = Field(None, description="URL to POST on completion/failure")

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        return _validate_url_value(v)

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)


class CaptureScreensRequest(BaseModel):
    """Capture one screenshot per demo screen in a single navigation pass.

    Unlike :class:`RecordUrlRequest` (one continuous clip), this navigates the
    SPA once and walks ``steps`` (see :class:`ScreenCaptureStep`), producing a
    discrete PNG per ``capture_id``. The result is a single ``.zip`` of
    ``<capture_id>.png`` files, downloaded via the shared
    ``GET /api/record/jobs/{job_id}/video`` endpoint — the zip is
    self-describing (entry names ARE the screen ids), so frame-synced promo
    render keys each screenshot to the narration line it illustrates.
    """

    url: str = Field(
        ..., min_length=1, max_length=2048, description="Live URL to navigate and capture"
    )
    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Part identifier for the capture",
    )
    auth_info: UrlAuthInfo | None = None
    recording_config: RecordingConfigInput | None = None
    steps: list[ScreenCaptureStep] = Field(
        ...,
        min_length=1,
        max_length=60,
        description="Ordered navigate/capture steps; at least one must set capture_id",
    )
    record_clips: bool = Field(
        False,
        description=(
            "Record a short interaction clip per screen (showing the click that "
            "reveals it) alongside the still. The zip then holds BOTH <id>.mp4 "
            "and <id>.png; promo render keys the reveal line to the clip."
        ),
    )
    clip_dwell_ms: int = Field(
        6500,
        ge=0,
        le=30000,
        description=(
            "Dwell after each revealing click before ending its clip; keep ≥ the "
            "longest answer line so render's -t trims cleanly. Only used when "
            "record_clips is set."
        ),
    )

    @field_validator("url", mode="before")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        return _validate_url_value(v)


class RecordPartRequest(BaseModel):
    """Record a single part."""

    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Part ID to record",
    )
    scripts_dir: str = Field(
        ..., min_length=1, max_length=1024, description="Path to the scripts directory"
    )
    part_type: str | None = Field("browser_recording", description="Part type")
    recording_config: RecordingConfigInput | None = None
    auth_config: AuthConfigInput | None = None
    webhook_url: str | None = Field(None, description="URL to POST on completion/failure")

    @field_validator("scripts_dir", mode="before")
    @classmethod
    def _validate_scripts_dir(cls, v: str) -> str:
        return _validate_scripts_dir_value(v)

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)


class RecordBatchRequest(BaseModel):
    """Record multiple parts."""

    part_ids: list[str] = Field(
        ..., min_length=1, max_length=100, description="Part IDs to record in order"
    )
    scripts_dir: str = Field(
        ..., min_length=1, max_length=1024, description="Path to the scripts directory"
    )
    recording_config: RecordingConfigInput | None = None
    auth_config: AuthConfigInput | None = None
    webhook_url: str | None = Field(None, description="URL to POST on completion/failure")
    execution_mode: str = Field("parallel", description="Execution mode: parallel or sequential")
    failure_policy: str = Field("continue", description="Failure policy: continue or fail_fast")
    priority: int = Field(
        0, ge=-100, le=100, description="Batch priority (higher = more important)"
    )

    @field_validator("part_ids", mode="before")
    @classmethod
    def _validate_part_ids(cls, v: list[str]) -> list[str]:
        for idx, item in enumerate(v):
            if not isinstance(item, str):
                msg = f"part_ids[{idx}] must be a string"
                raise ValueError(msg)
            if len(item) > 128 or len(item) < 1:
                msg = f"part_ids[{idx}] must be between 1 and 128 characters, got {len(item)}"
                raise ValueError(msg)
            if not _PART_ID_RE.match(item):
                msg = (
                    f"part_ids[{idx}] contains invalid characters: '{item}' "
                    f"(must match ^[a-zA-Z0-9_\\-\\.]+$)"
                )
                raise ValueError(msg)
        return v

    @field_validator("scripts_dir", mode="before")
    @classmethod
    def _validate_scripts_dir(cls, v: str) -> str:
        return _validate_scripts_dir_value(v)

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _validate_execution_mode(cls, v: str) -> str:
        allowed = {"parallel", "sequential"}
        if v not in allowed:
            msg = f"execution_mode must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("failure_policy", mode="before")
    @classmethod
    def _validate_failure_policy(cls, v: str) -> str:
        allowed = {"continue", "fail_fast"}
        if v not in allowed:
            msg = f"failure_policy must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v


# --- Response Schemas ---


class AsyncJobResponse(BaseModel):
    """Response for async job creation."""

    job_id: str
    status: str = "pending"
    poll_url: str


class BatchJobResponse(BaseModel):
    """Response for batch job creation."""

    batch_id: str
    job_ids: dict[str, str]  # part_id -> job_id
    status: str = "pending"
    scripts_dir: str | None = Field(
        default=None,
        description="Resolved scripts_dir on the recording pod's filesystem",
    )
    poll_url: str | None = Field(
        default=None,
        description="GET URL for polling the batch's status (typically /api/record/batch/<batch_id>).",
    )


class BatchStatusResponse(BaseModel):
    """Batch status aggregation."""

    batch_id: str
    total: int
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    pending: int = 0
    processing: int = 0
    queued: int = 0
    status: str = "processing"
    jobs: list[JobStatusResponse] = []


class JobStatusResponse(BaseModel):
    """Job status polling response."""

    job_id: str
    status: str
    part_id: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None
    video_path: str | None = None
    progress: float | None = None
    progress_stage: str | None = None
    retry_count: int | None = None
    max_retries: int | None = None
    # Video metadata
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    capture_width: int | None = Field(
        None,
        description=(
            "Width in pixels of the raster the recorder actually captured, "
            "before any downscale to `width`. Larger than `width` when the "
            "recorder supersampled; EQUAL to `width` when the supersampling was "
            "clamped down to fit the capture raster budget (a 4k order). None "
            "means the recorder did not report a capture raster — fall back to "
            "`width`."
        ),
    )
    capture_height: int | None = Field(
        None,
        description=(
            "Height in pixels of the raster the recorder actually captured, "
            "before any downscale to `height`. See `capture_width`."
        ),
    )
    file_size_bytes: int | None = None
    has_audio: bool | None = None


class RecordingResultResponse(BaseModel):
    """Recording result."""

    job_id: str
    part_id: str | None = None
    status: str
    video_path: str | None = None
    duration_seconds: float | None = None
    file_size_bytes: int | None = None
    width: int | None = None
    height: int | None = None
    capture_width: int | None = Field(
        None,
        description=(
            "Width in pixels of the raster the recorder actually captured, "
            "before any downscale to `width`. A caller that wants its source "
            "assets to survive the capture at full detail (upscaling stills "
            "before a Ken Burns record, say) should target THIS, not `width`, "
            "and must not assume a fixed supersampling factor: the recorder "
            "steps it down to stay inside its raster budget, which shows up "
            "here as `capture_width == width`. None means the recorder did not "
            "report a capture raster — fall back to `width`."
        ),
    )
    capture_height: int | None = Field(
        None,
        description=(
            "Height in pixels of the raster the recorder actually captured, "
            "before any downscale to `height`. See `capture_width`."
        ),
    )
    has_audio: bool | None = None
    recording_time_seconds: float | None = None
    queue_wait_seconds: float | None = None
    max_frame_seconds: float | None = Field(
        None,
        description=(
            "Slowest single frame's own capture time (JS timeline seek + "
            "screenshot) during AnimationRecorder's frame-step loop -- the "
            "per-frame telemetry `animation_render_stall_window_seconds` was "
            "originally justified without. None from any recorder that does "
            "not frame-step, or from a job written before this field existed."
        ),
    )
    error: str | None = None


class PaginatedJobsResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobStatusResponse]
    total: int
    offset: int
    limit: int


# --- Artifact-sourced batch schemas ---


class RecordPartArtifactRef(BaseModel):
    """One Playwright part to record, sourced from artifact_service."""

    part_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-\.]+$",
        description="Per-part stable id; mirrors the existing /batch contract.",
    )
    artifact_id: UUID = Field(
        ...,
        description=(
            "artifact_service ID returned by content_service when the script was uploaded."
        ),
    )

    @field_validator("part_id")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v or v.strip(".") == "":
            raise ValueError("part_id cannot contain '..' or be all dots")
        return v


class RecordBatchFromArtifactsRequest(BaseModel):
    """Record a batch where each part's TypeScript lives in artifact_service."""

    project_id: UUID = Field(
        ...,
        description=(
            "artifact_service project scope. MUST equal the value content_service "
            "used at upload time (the generate-stage batch_id, exposed as "
            "GenerateOutput.job_id) — NOT the pipeline job_id. Mismatch causes "
            "404s when vps fetches scripts."
        ),
    )
    parts: list[RecordPartArtifactRef] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Ordered list of parts to record.",
    )
    recording_config: RecordingConfigInput | None = None
    auth_config: AuthConfigInput | None = None
    webhook_url: str | None = Field(None, description="POST on completion/failure")
    execution_mode: str = Field("parallel", description="parallel | sequential")
    failure_policy: str = Field("continue", description="continue | fail_fast")
    priority: int = Field(0, ge=-100, le=100)

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: str | None) -> str | None:
        return _validate_webhook_url_value(v)

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _validate_execution_mode(cls, v: str) -> str:
        allowed = {"parallel", "sequential"}
        if v not in allowed:
            msg = f"execution_mode must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v

    @field_validator("failure_policy", mode="before")
    @classmethod
    def _validate_failure_policy(cls, v: str) -> str:
        allowed = {"continue", "fail_fast"}
        if v not in allowed:
            msg = f"failure_policy must be one of {sorted(allowed)}, got '{v}'"
            raise ValueError(msg)
        return v
