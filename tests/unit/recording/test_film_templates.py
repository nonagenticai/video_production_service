"""String-level guards for the explainer-film templates (no Chromium).

The rendered behaviour — pixels, marks, motion, fonts actually loading — is
proved in ``tests/integration/test_film_templates_render.py``. These are the
cheap invariants that must hold for that behaviour to be possible at all, and
that would otherwise only fail at capture time:

* the four film families are registered and wired to the shared film kit;
* nothing in them reads a wall clock or an RNG (frame capture is deterministic
  only if ``render`` is a pure function of ``(PARAMS, t)``);
* none of them carries built-in copy — a caller that omits content gets an empty
  card, never invented text;
* the Inter faces ship DORMANT, so a non-film page is not re-fonted;
* the legacy-guard baseline snapshot is the one taken from origin/main.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from src.services.recording.templates import (
    _FILM_FONT_FACES,
    _FONT_FACES,
    _FONTS_DIR,
    _TEMPLATES_DIR,
    FILM_FONTS_STYLE_ID,
    OVERLAY_SAFE_TEMPLATES,
    get_template,
    list_templates,
)

FILM_TEMPLATE_IDS = ["two_beat", "levels", "checklist", "handoff"]
# Pre-existing templates that gained an opt-in film layout: (id, the opt-in test).
FILM_OPT_INS = [
    ("two_up", "P.layout === 'panels'"),
    ("timeline", "P.layout === 'track'"),
    ("animated_title_card", "P.style === 'film'"),
]
NONDETERMINISM_TOKENS = ("Date.now", "Math.random", "new Date", "performance.now")
BASELINE_DIR = Path(__file__).parents[2] / "integration" / "fixtures" / "templates_baseline"
# sha256 of origin/main @ 5c779f5's copies. The pixel guard compares against
# these; editing one to make that guard pass must fail here first.
BASELINE_SHA256 = {
    "_icons.js": "219cf13390f3d1c5edefb82a07c1efb0cbc15af49d88bbf3927297f8e943619a",  # pragma: allowlist secret
    "_motion_lib.js": "233001e2f34dcdded4e84ab28f00be16a05a3993692ae4e98b5a0029e9137a2e",  # pragma: allowlist secret
    "animated_title_card.html": "b0b3265103b90511ea7e47e5d3d9faf16ac6701ed9bf6ba57cc24b2088052de1",  # pragma: allowlist secret
    "timeline.html": "cb8b6a5d88491a00709b24cdc501d683f4d69df79cc26eb1737862a6971a410f",  # pragma: allowlist secret
    "two_up.html": "07406398a0eb38e8b05488cd4a1da6d6753fc9fe5b2117d030a9efda249b93ac",  # pragma: allowlist secret
}


def _src(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def _defaults_block(html: str) -> str:
    m = re.search(r"const DEFAULTS = \{(.*?)\n\};", html, re.S)
    assert m, "template has no DEFAULTS block"
    return m.group(1)


class TestRegistrationAndWiring:
    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_registered_and_assembled(self, template_id: str) -> None:
        assert template_id in list_templates()
        html = get_template(template_id)
        for needle in ("window.MotionLib", "window.ICONS", "MotionLib.film(", "window.render"):
            assert needle in html, f"{template_id} is missing {needle}"
        assert "window.RENDER_DURATION_MS = F.dur" in html
        assert "F.ambient(t)" in html, f"{template_id} does not paint the shared ambient layer"

    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_not_overlay_safe(self, template_id: str) -> None:
        # Full-frame cards: they paint an opaque stage, so alpha must stay a 400.
        assert template_id not in OVERLAY_SAFE_TEMPLATES

    @pytest.mark.parametrize(("template_id", "gate"), FILM_OPT_INS)
    def test_film_layout_is_opt_in(self, template_id: str, gate: str) -> None:
        html = _src(f"{template_id}.html")
        assert gate in html
        # The film kit is only ever entered behind that gate.
        assert html.count("MotionLib.film(") == 1
        film_call = html.index("MotionLib.film(")
        assert html.index(gate) < film_call

    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_param_contract_is_documented_in_the_file(self, template_id: str) -> None:
        head = _src(f"{template_id}.html").split("<style>", 1)[0]
        for key in ("PARAMS", "duration_ms", "marks", "film_t_ms"):
            assert key in head, f"{template_id} header comment does not document {key}"


class TestDeterminism:
    @pytest.mark.parametrize(
        "name", [f"{t}.html" for t in FILM_TEMPLATE_IDS] + ["_motion_lib.js", "_icons.js"]
    )
    def test_no_wall_clock_or_rng(self, name: str) -> None:
        src = _src(name)
        for token in NONDETERMINISM_TOKENS:
            assert token not in src, f"{name} contains non-deterministic '{token}'"

    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_no_css_animation(self, template_id: str) -> None:
        # Film cards are driven only by render(t): a CSS animation would run on
        # its own clock and break the cross-card ambient continuity.
        src = _src(f"{template_id}.html")
        assert "@keyframes" not in src
        assert "transition:" not in src


class TestNoBuiltInCopy:
    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_content_defaults_are_empty(self, template_id: str) -> None:
        block = _defaults_block(_src(f"{template_id}.html"))
        # Every quoted default is empty, a colour, or the motif keyword.
        for value in re.findall(r"'([^']*)'", block):
            assert value == "" or value.startswith("#") or value == "dial", (
                f"{template_id} ships built-in copy {value!r}"
            )

    @pytest.mark.parametrize("template_id", FILM_TEMPLATE_IDS)
    def test_caller_strings_never_reach_innerhtml(self, template_id: str) -> None:
        src = _src(f"{template_id}.html")
        for line in src.splitlines():
            if "innerHTML" in line:
                assert "P." not in line.split("innerHTML", 1)[1], line


class TestFilmFontsShipDormant:
    def test_always_on_faces_are_exactly_mains(self) -> None:
        assert [(f, s, w) for f, s, w, _ in _FONT_FACES] == [
            ("Fraunces", "normal", 400),
            ("Fraunces", "italic", 400),
            ("IBM Plex Sans", "normal", 400),
        ]

    def test_film_family_is_one_family_with_real_weights(self) -> None:
        assert {f for f, *_ in _FILM_FONT_FACES} == {"Inter"}
        assert sorted(w for _, _, w, _ in _FILM_FONT_FACES) == [400, 600, 700]
        assert {s for _, s, _, _ in _FILM_FONT_FACES} == {"normal"}
        blobs = [(_FONTS_DIR / fn).read_bytes() for *_, fn in _FILM_FONT_FACES]
        for blob in blobs:
            assert blob[:4] == b"wOF2"
        # Three DIFFERENT files — not one face registered under three weights.
        assert len({hashlib.sha256(b).hexdigest() for b in blobs}) == 3
        # Latin subset: the whole family stays under 60 KB raw.
        assert sum(len(b) for b in blobs) < 60_000
        assert (_FONTS_DIR / "OFL-Inter.txt").exists()

    @pytest.mark.parametrize("template_id", ["animated_slide", "stat", "two_up", "two_beat"])
    def test_inter_lives_only_in_the_disabled_stylesheet(self, template_id: str) -> None:
        html = get_template(template_id)
        marker = f'<style id="{FILM_FONTS_STYLE_ID}" media="not all">'
        assert html.count(marker) == 1
        always_on, rest = html.split(marker, 1)
        dormant = rest.split("</style>", 1)[0]
        assert "font-family:'Inter'" not in always_on
        assert dormant.count("font-family:'Inter'") == 3
        assert "Fraunces" not in dormant

    def test_film_kit_is_what_enables_them(self) -> None:
        lib = _src("_motion_lib.js")
        assert f'getElementById("{FILM_FONTS_STYLE_ID}")' in lib
        enable = lib.index('ff.media = "all"')
        # ...inside filmCss(), which only MotionLib.film() calls.
        assert lib.index("function filmCss()") < enable < lib.index("function parseMarks")
        assert lib.count("filmCss()") == 2  # the definition and the one call in film()
        assert "font-synthesis:none" in lib


class TestLegacyBaselineSnapshot:
    def test_snapshot_is_origin_main(self) -> None:
        found = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in BASELINE_DIR.iterdir()}
        assert found == BASELINE_SHA256

    def test_title_card_ignores_marks_outside_film_style(self) -> None:
        # Main's card never read `marks`; the orchestrator sends them on every
        # card, so honouring them outside film style would re-time live output.
        src = _src("animated_title_card.html")
        body = src.split("let RULE_START = 900, BRAND_START = 2000;", 1)[1]
        assert body.lstrip().startswith("if (FILM){")
        assert "T.marks" not in src


class TestTwoBeatIconContract:
    """``two_beat`` takes an ``icon`` (a glyph name, looked up — never markup)."""

    def test_icon_is_documented_and_defaults_empty(self) -> None:
        html = _src("two_beat.html")
        head = html.split("<style>", 1)[0]
        assert re.search(r"^\s+icon\s+\"\"", head, re.M), "icon is not in the param contract"
        assert "icon:''" in _defaults_block(html)

    def test_icon_is_a_lookup_in_the_icon_set(self) -> None:
        html = _src("two_beat.html")
        assert "window.ICONS[ICON]" in html
        assert "iconSvg(ICON," in html
