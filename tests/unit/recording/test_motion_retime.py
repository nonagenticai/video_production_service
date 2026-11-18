"""VP-5 — the 8 EXISTING templates now consume the shared motion library.

VP-3 shipped ``window.MotionLib`` (easings / ``timing()`` / entrance+exit
``layer()`` / ``renderDurationMs``) and VP-4 added five families that use it. But
the eight pre-existing templates still hand-rolled their own ``const ease``,
``STAGGER`` and ``RENDER_DURATION_MS`` and had no exit — so the motion signature
the orchestrator already emits (``easing/stagger_ms/entrance/exit/parallax/
entrance_ms/hold_ms/exit_ms/marks``) was inert on the live deck path.

VP-5 retrofits all eight to read that signature through ``MotionLib`` while
staying **byte-identical when the signature is absent** (the default deck render —
a slide with no ``motion`` author-block gets no signature keys). These are the
pure-Python / string-level CI guards. The actual byte-identical frame proof (a
Playwright frame-diff of every element's opacity/transform/inline-style at 18
sampled ``t`` against a pre-edit golden baseline, plus a no-freeze + exit check
under a synthetic signature) runs in the local render harness described in the
VP-5 handoff — it cannot run in CI (no live Chromium, mirrors
``test_animation_recorder.py``'s mocked browser).
"""

from __future__ import annotations

import pytest

from src.services.recording.templates import get_template

# The eight templates that predate the shared library and were retrofitted.
EXISTING_TEMPLATE_IDS = [
    "animated_slide",
    "animated_title_card",
    "ken_burns",
    "kinetic_typography",
    "lower_third",
    "social_card",
    "pipeline_story",
    "data_viz",
]

# The two the live deck classifier (motion_for_deck_slide) actually dispatches to.
LIVE_DECK_TEMPLATE_IDS = ["animated_slide", "animated_title_card"]

# The five VP-4 layout families: no required-field validation, so a payload
# missing a content field used to fall through `Object.assign(DEFAULTS,
# window.PARAMS)` and silently render each template's built-in DEMO COPY —
# fabricated marketing claims a customer never wrote. Fixed by
# fix/strip-fabricated-template-defaults: missing content now renders empty.
CONTENT_FAMILY_TEMPLATE_IDS = ["stat", "timeline", "quote", "two_up", "section_divider"]

# Frame-capture is deterministic only if render is a pure function of (PARAMS,t).
NONDETERMINISM_TOKENS = ("Date.now", "Math.random", "new Date")

# Fabricated demo copy that must NEVER reappear as a DEFAULTS/fallback value in
# any template source. These are the exact strings verified (2026-07-19) to
# render when a caller omits the corresponding content field — i.e. invented
# marketing claims that could silently reach a real customer's video because
# there is no required-field validation on window.PARAMS. If a future author
# reintroduces one of these as a fallback, this is a content-safety regression,
# not a stylistic nit — do not delete this test to make it pass.
BANNED_FABRICATED_STRINGS = (
    "'94'",  # stat.html: `value || '94'` -> "94% faster onboarding"
    # (quoted to avoid false-positiving on the unrelated --muted:#94A3B8 color token)
    "faster onboarding",  # stat.html: `label || 'faster onboarding'`
    "Draft the script",  # timeline.html: invented step 1
    "Storyboard the shots",  # timeline.html: invented step 2
    "Render deterministically",  # timeline.html: invented step 3
    "Compile & ship",  # timeline.html: invented step 4
    "From brief to broadcast",  # timeline.html: invented title
    "Deterministic motion means the same frame renders every single time",  # quote.html
    "The Motion Toolkit",  # quote.html: invented attribution
    "Hand-edit every clip, hope the brand holds, pay per render",  # two_up.html: left body
    "Deterministic templates, on-brand by construction, zero per-clip cost",  # two_up.html: right body
    "Two ways to tell it",  # two_up.html: invented title
    "Chapter 02",  # section_divider.html: invented kicker
    "Building the Pipeline",  # section_divider.html: invented title
)


def _src(template_id: str) -> str:
    """The assembled template HTML (fonts + _motion_lib + _icons + body)."""
    return get_template(template_id)


class TestConsumesSharedLibrary:
    @pytest.mark.parametrize("template_id", EXISTING_TEMPLATE_IDS)
    def test_references_motionlib(self, template_id: str) -> None:
        # The retrofit's point: each existing template now goes through the shared
        # library instead of a private easing/timeline. A revert to hand-rolled
        # motion (dropping the MotionLib reference) must fail here.
        assert "MotionLib." in _src(template_id), f"{template_id} no longer references MotionLib"

    @pytest.mark.parametrize("template_id", EXISTING_TEMPLATE_IDS)
    def test_reads_params_with_fallbacks(self, template_id: str) -> None:
        # Every template consumes window.PARAMS but must still paint fallback
        # content when it is absent (the byte-identical params-less render). The
        # actual non-empty/byte-identical proof is the render harness; here we
        # only guard that the params surface is still wired in.
        assert "window.PARAMS" in _src(template_id)


class TestLibraryInjectedInOrder:
    @pytest.mark.parametrize("template_id", EXISTING_TEMPLATE_IDS)
    def test_font_then_motionlib_then_icons(self, template_id: str) -> None:
        # window.PARAMS + the lib must resolve before the template body runs:
        # head assembly order is fonts -> _motion_lib -> _icons -> body.
        html = _src(template_id)
        assert "@font-face" in html
        assert "window.MotionLib" in html
        assert "window.ICONS" in html
        assert (
            html.index("@font-face") < html.index("window.MotionLib") < html.index("window.ICONS")
        )


class TestDeckTemplatesGetTimelineAndExit:
    @pytest.mark.parametrize("template_id", LIVE_DECK_TEMPLATE_IDS)
    def test_duration_is_timeline_driven(self, template_id: str) -> None:
        # renderDurationMs(P, LEGACY_TOTAL) returns the signature timeline total
        # when a motion block drove it (no ~4s freeze), else exactly LEGACY_TOTAL.
        assert "MotionLib.renderDurationMs" in _src(template_id)

    @pytest.mark.parametrize("template_id", LIVE_DECK_TEMPLATE_IDS)
    def test_has_exit_grammar(self, template_id: str) -> None:
        # The entrance+exit primitive: layer() ramps a card back out after
        # T.holdEnd when exit_ms>0 (the fix for "motion freezes at ~4s"). Absent
        # a signature exit_ms defaults to 0 => no exit => byte-identical hold.
        assert "MotionLib.layer" in _src(template_id)


class TestDeterminism:
    @pytest.mark.parametrize("template_id", EXISTING_TEMPLATE_IDS)
    def test_no_wall_clock_or_rng(self, template_id: str) -> None:
        html = _src(template_id)
        for token in NONDETERMINISM_TOKENS:
            assert token not in html, f"{template_id} contains non-deterministic '{token}'"


class TestNoFabricatedDemoCopy:
    """fix/strip-fabricated-template-defaults: a payload missing a content
    field must render EMPTY, never the template's built-in demo copy. Scan
    every registered template (not just the 5 content families) so a banned
    string copy-pasted into a different template is still caught.
    """

    @pytest.mark.parametrize("template_id", CONTENT_FAMILY_TEMPLATE_IDS)
    def test_content_family_has_no_fallback_string_default(self, template_id: str) -> None:
        html = _src(template_id)
        for banned in BANNED_FABRICATED_STRINGS:
            assert banned not in html, (
                f"{template_id} still contains fabricated demo copy {banned!r} — "
                "a caller that omits this content field would silently render "
                "invented marketing text instead of an empty field."
            )

    def test_no_registered_template_contains_fabricated_copy(self) -> None:
        # Belt-and-suspenders: the banned strings must not leak into any
        # template, not only the five families where they originated.
        from src.services.recording.templates import list_templates

        for template_id in list_templates():
            html = _src(template_id)
            for banned in BANNED_FABRICATED_STRINGS:
                assert banned not in html, f"{template_id} contains fabricated demo copy {banned!r}"
