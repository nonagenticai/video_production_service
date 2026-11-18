"""Tests for the shared motion library (VP-3) + the new layout families (VP-4).

`templates/__init__.py::get_template` now injects, after ``<head>`` and in a
fixed order, the embedded-font ``<style>``, then ``_motion_lib.js`` (defining
``window.MotionLib``), then ``_icons.js`` (defining ``window.ICONS``) — before
each template's own body script runs. VP-4 adds five layout families (quote,
stat, section_divider, timeline, two_up) that consume that shared library
instead of hand-rolling their own easing/stagger.

These are pure-Python / string-level guards — no Chromium. The frame-stepped
browser render is covered by the mocked ``AnimationRecorder`` tests
(``test_animation_recorder.py``); here we assert the assembled-HTML + source
contract (injection order, wiring, content-param fidelity, determinism) so a
regression fails CI rather than silently at capture time.
"""

from __future__ import annotations

import pytest

from src.services.recording.templates import (
    _TEMPLATES_DIR,
    TemplateNotFoundError,
    get_template,
    list_templates,
)

# The five layout families added by VP-4 (id == "<id>.html").
NEW_TEMPLATE_IDS = ["quote", "stat", "section_divider", "timeline", "two_up"]

# id -> content param key(s) the template's SOURCE must reference, so a family
# can't silently ignore the content the orchestrator hands it.
CONTENT_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "quote": ("quote", "attribution"),
    "stat": ("stat",),
    "section_divider": ("title",),
    "timeline": ("steps",),
    "two_up": ("two_up",),
}

# Every assembled new template must expose these hooks (font embed + shared lib
# + icons + timeline + the deterministic render entrypoint).
WIRING_NEEDLES = (
    "@font-face",
    "window.MotionLib",
    "window.ICONS",
    "MotionLib.timing",
    "window.render",
)

# Shared JS data files injected into every template.
DATA_FILES = ["_motion_lib.js", "_icons.js"]
NEW_TEMPLATE_FILES = [f"{tid}.html" for tid in NEW_TEMPLATE_IDS]

# Frame-capture is deterministic only if render is a pure function of (PARAMS,t):
# no wall-clock, no RNG. This must FAIL the moment anyone adds one.
NONDETERMINISM_TOKENS = ("Date.now", "Math.random", "new Date")
DETERMINISM_FILES = DATA_FILES + NEW_TEMPLATE_FILES


def _read(name: str) -> str:
    """Read a raw template/data file from the templates dir as UTF-8."""
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


class TestLibraryInjection:
    """The shared library is injected into EXISTING templates, in order, without
    clobbering the template's own body."""

    def test_injects_font_lib_icons_in_order(self) -> None:
        html = get_template("animated_slide")
        assert "@font-face" in html
        assert "window.MotionLib" in html
        assert "window.ICONS" in html
        # Fonts resolve before the lib defines window.MotionLib, which defines
        # before window.ICONS — the exact head assembly order.
        assert (
            html.index("@font-face") < html.index("window.MotionLib") < html.index("window.ICONS")
        )

    def test_keeps_own_body_script(self) -> None:
        # Regression: injection must not replace the template's own render body.
        assert "window.render" in get_template("animated_slide")


class TestRegistration:
    def test_new_families_registered(self) -> None:
        ids = list_templates()
        for tid in NEW_TEMPLATE_IDS:
            assert tid in ids

    def test_unknown_id_still_raises(self) -> None:
        with pytest.raises(TemplateNotFoundError):
            get_template("no_such_template")


class TestNewFamiliesWired:
    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_loads_and_is_wired(self, template_id: str) -> None:
        html = get_template(template_id)
        assert html.strip(), f"{template_id} rendered empty"
        for needle in WIRING_NEEDLES:
            assert needle in html, f"{template_id} missing {needle}"

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_references_content_params(self, template_id: str) -> None:
        # The key guard: the source must name its content param(s), so it can't
        # accept-but-ignore what the orchestrator sends.
        src = _read(f"{template_id}.html")
        for key in CONTENT_PARAM_KEYS[template_id]:
            assert key in src, f"{template_id}.html never references content param '{key}'"

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_has_defaults_so_paramless_render_is_non_empty(self, template_id: str) -> None:
        # A non-trivial DEFAULTS + Object.assign means a params-less standalone
        # render still paints (fallback content), not a blank frame.
        src = _read(f"{template_id}.html")
        assert "const DEFAULTS" in src
        assert "Object.assign" in src

    @pytest.mark.parametrize("template_id", NEW_TEMPLATE_IDS)
    def test_defers_motion_to_lib(self, template_id: str) -> None:
        # No hand-rolled easing/stagger — they must go through MotionLib.
        src = _read(f"{template_id}.html")
        assert "const ease" not in src
        assert "const STAGGER" not in src


class TestDataFiles:
    def test_motion_lib_present_and_non_empty(self) -> None:
        src = _read("_motion_lib.js")
        assert src.strip()
        assert "window.MotionLib" in src

    def test_icons_present_and_non_empty(self) -> None:
        src = _read("_icons.js")
        assert src.strip()
        assert "window.ICONS" in src

    def test_timing_caps_stagger_to_entrance_window(self) -> None:
        # Guard: the default-mark stagger MUST be capped against the entrance
        # window (mirrors orchestrator plan_card_timing's min(stagger, entrance//n))
        # so a trailing staggered element can't start after a short capture ends
        # and render invisible. A revert to an uncapped `num(P.stagger_ms, 420)`
        # must fail here.
        src = _read("_motion_lib.js")
        assert "Math.min(num(P.stagger_ms" in src, "stagger is not capped in timing()"
        assert "Math.floor(entrance" in src, "stagger cap is not bounded by the entrance window"


class TestDeterminism:
    @pytest.mark.parametrize("filename", DETERMINISM_FILES)
    def test_no_wall_clock_or_rng(self, filename: str) -> None:
        src = _read(filename)
        for token in NONDETERMINISM_TOKENS:
            assert token not in src, f"{filename} contains non-deterministic '{token}'"
