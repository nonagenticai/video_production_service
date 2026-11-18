"""Branded motion-graphics template library.

Each template is a self-contained HTML file that:

* exposes ``window.RENDER_DURATION_MS`` — the timeline length the animation
  recorder steps through;
* OPTIONALLY exposes ``window.render(t)`` — a deterministic, "freeze-capture
  safe" function that paints the exact frame for timestamp ``t`` (ms). Templates
  without it rely purely on CSS ``@keyframes``; the recorder still captures them
  deterministically by seeking every ``Animation.currentTime`` per frame;
* reads ``window.PARAMS`` (a JSON object injected by the recorder) to override
  brand tokens and content, falling back to its built-in defaults so it also
  renders standalone in a browser.

Brand fonts are **embedded as data URIs** at load time from the vendored
``fonts/`` woff2 files — capture must never depend on a live Google Fonts fetch
(flaky/offline → non-deterministic frames). This mirrors slide_service's
bundled-font pattern. Fraunces and IBM Plex Sans are active in every template;
Inter (real 400/600/700) is embedded in every page but stays INERT until a
template turns on the film kit (``MotionLib.film``), so a non-film render that
names 'Inter' in its font stack resolves exactly as it did before Inter was
vendored.
"""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

_TEMPLATES_DIR = Path(__file__).parent
_FONTS_DIR = _TEMPLATES_DIR / "fonts"

# Logical template id → HTML filename. Add a row here to register a template.
TEMPLATES: dict[str, str] = {
    "social_card": "social_card.html",
    "pipeline_story": "pipeline_story.html",
    # In-video templates (consumed by orchestrator motion_graphics_gen + the
    # animated-slide path). All full-bleed/adaptive — they fill whatever
    # width/height the caller passes (16:9, square, vertical).
    "animated_title_card": "animated_title_card.html",
    "animated_slide": "animated_slide.html",
    "lower_third": "lower_third.html",
    "data_viz": "data_viz.html",
    "kinetic_typography": "kinetic_typography.html",
    # Ken Burns: a gentle deterministic pan/zoom over a still image passed as a
    # data-URI (window.PARAMS.image). Lets the orchestrator animate any still
    # (AI/curated image, slide PNG) into a clip via the standard record_animation
    # path. Presets: zoom_in (default), zoom_out, pan_left/right/up/down.
    "ken_burns": "ken_burns.html",
    # A4 / C23 motion layout families (VP-4). They consume the same flat brand
    # PARAMS + the shared _motion_lib timeline, giving the orchestrator's
    # design/motion signature real layout targets beyond the bullets-vs-title
    # binary. Each renders standalone from its built-in defaults.
    #
    # Reachability today: these are selectable via the config-driven generic
    # `motion_graphics_gen` stage (`template_id: stat`, etc.). The automatic
    # deck projection (`motion_for_deck_slide`) still dispatches only to
    # `animated_slide`/`animated_title_card`; wiring the classifier onto these
    # families — and retrofitting those two LIVE templates to consume the motion
    # signature (`easing`/`stagger_ms`/`marks`/…), which they currently ignore —
    # is the follow-up VP-5 PR. Until then the motion signature is inert on the
    # live deck path (byte-identical), and this PR ships the primitive + families.
    "quote": "quote.html",
    "stat": "stat.html",
    "section_divider": "section_divider.html",
    "timeline": "timeline.html",
    "two_up": "two_up.html",
    # Explainer-film families: graphic compositions (drawn strokes, icons,
    # tracks) on the shared film kit in _motion_lib.js (MotionLib.film) — one
    # viewport unit, one ambient layer on a film clock (`film_t_ms`), beats on
    # `marks`, and a continuous foreground loop after the last entrance. Each
    # file's header comment is its param contract. `two_up` (layout "panels"),
    # `timeline` (layout "track") and `animated_title_card` (style "film") opt
    # into the same kit; without those params they render as before.
    "two_beat": "two_beat.html",
    "levels": "levels.html",
    "checklist": "checklist.html",
    "handoff": "handoff.html",
}

# (font-family, font-style, font-weight, woff2 filename) → one @font-face each.
# Fraunces / IBM Plex Sans ship the latin 400 faces only; their 500/600 weights
# are synthesised by the browser (the legacy templates' look — unchanged).
_FONT_FACES: tuple[tuple[str, str, int, str], ...] = (
    ("Fraunces", "normal", 400, "fraunces-400-normal-latin.woff2"),
    ("Fraunces", "italic", 400, "fraunces-400-italic-latin.woff2"),
    ("IBM Plex Sans", "normal", 400, "ibm-plex-sans-400-normal-latin.woff2"),
)

# The film kit's family: Inter with REAL 400/600/700 latin faces
# (fonts/INTER-PROVENANCE.md, ~53 KB raw / ~71 KB inlined per page). The film
# templates set ``font-synthesis:none``, so nothing is ever faux-bolded or
# slanted. These faces ship in a ``media="not all"`` stylesheet (id
# ``film-fonts``) that ``MotionLib.film`` switches on — until then the browser
# does not register them, which keeps every non-film render pixel-identical to
# the pre-Inter service even when a caller's font stack names 'Inter'.
_FILM_FONT_FACES: tuple[tuple[str, str, int, str], ...] = (
    ("Inter", "normal", 400, "inter-400-normal-latin.woff2"),
    ("Inter", "normal", 600, "inter-600-normal-latin.woff2"),
    ("Inter", "normal", 700, "inter-700-normal-latin.woff2"),
)
FILM_FONTS_STYLE_ID = "film-fonts"


# Registered templates that render correctly against a TRANSPARENT backdrop,
# i.e. the ones designed to be composited OVER other footage rather than cut
# to full-frame.
#
# The bar is deliberately narrow: a template qualifies only when its full-frame
# backdrop comes from ``body``'s background and nothing else, because that is
# precisely what the recorder's transparency override neutralises
# (``_inject_transparent_style``). ``lower_third`` meets it — its ``.stage`` is
# an unpainted ``position:fixed`` box and only ``body`` carries ``--bg``.
#
# Every other registered template is EXCLUDED, and would come back opaque
# anyway: animated_slide / animated_title_card / data_viz / kinetic_typography /
# quote / section_divider / stat / timeline / two_up each paint an opaque
# radial-gradient onto a full-bleed stage element, social_card and
# pipeline_story paint a solid white card, and ken_burns is a full-frame image
# pan over ``#000``. Those are cut-between cards by design; asking for alpha on
# one is a caller mistake, and it is answered with a 400 rather than a clip
# that is transparent everywhere except the 100% of the frame that matters.
#
# Inline ``html`` requests are NOT checked against this set — a caller shipping
# their own document owns its backdrop.
OVERLAY_SAFE_TEMPLATES: frozenset[str] = frozenset({"lower_third"})


class TemplateNotFoundError(ValueError):
    """Raised when an unknown template id is requested."""


def list_templates() -> list[str]:
    """Return the registered template ids."""
    return sorted(TEMPLATES)


def _font_face_rules(faces: tuple[tuple[str, str, int, str], ...]) -> str:
    """Render one ``@font-face`` rule (woff2 inlined as a data URI) per face."""
    rules: list[str] = []
    for family, style, weight, filename in faces:
        data = (_FONTS_DIR / filename).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        rules.append(
            f"@font-face{{font-family:'{family}';font-style:{style};"
            f"font-weight:{weight};font-display:block;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}"
        )
    return "".join(rules)


@lru_cache(maxsize=1)
def _font_face_style() -> str:
    """Build the embedded-font ``<style>`` blocks (cached).

    The always-on block, then the film kit's Inter faces in a stylesheet the
    browser ignores (``media="not all"``) until ``MotionLib.film`` enables it.
    """
    return (
        "<style>"
        + _font_face_rules(_FONT_FACES)
        + "</style>"
        + f'<style id="{FILM_FONTS_STYLE_ID}" media="not all">'
        + _font_face_rules(_FILM_FONT_FACES)
        + "</style>"
    )


@lru_cache(maxsize=1)
def _motion_lib_script() -> str:
    """Build the shared ``_motion_lib.js`` ``<script>`` block (cached).

    Defines ``window.MotionLib`` (named easings, the entrance+exit ``layer``
    primitive, the PARAMS timeline reader, ``applyBrandVars``) so every template
    shares one motion grammar instead of hardcoding its own easing/stagger.
    """
    js = (_TEMPLATES_DIR / "_motion_lib.js").read_text(encoding="utf-8")
    return "<script>" + js + "</script>"


@lru_cache(maxsize=1)
def _icons_script() -> str:
    """Build the shared ``_icons.js`` ``<script>`` block (cached).

    Defines ``window.ICONS`` + ``window.iconSvg`` — a small inline line-icon set
    templates can stroke in the accent when handed a glyph name via ``P.icon``.
    """
    js = (_TEMPLATES_DIR / "_icons.js").read_text(encoding="utf-8")
    return "<script>" + js + "</script>"


@lru_cache(maxsize=len(TEMPLATES))
def get_template(template_id: str) -> str:
    """Return the template HTML with embedded fonts + shared motion lib, ready to render.

    Params are NOT substituted here — the recorder injects ``window.PARAMS`` at
    capture time so one cached template string serves every brand.
    """
    filename = TEMPLATES.get(template_id)
    if filename is None:
        valid = ", ".join(list_templates())
        raise TemplateNotFoundError(f"Unknown motion template '{template_id}'. Registered: {valid}")
    html = (_TEMPLATES_DIR / filename).read_text(encoding="utf-8")
    # Inject, in order, right after <head>: the embedded-font <style> (so
    # font-family references resolve without a network fetch), then the shared
    # _motion_lib + _icons <script>s (so window.MotionLib / window.ICONS are
    # defined before the template's own body script runs). window.PARAMS is
    # injected separately by the recorder before the body script, so the lib —
    # which reads PARAMS only at render time — is available regardless of order.
    head = "<head>" + _font_face_style() + _motion_lib_script() + _icons_script()
    return html.replace("<head>", head, 1)
