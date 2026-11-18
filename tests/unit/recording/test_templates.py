"""Tests for the motion-graphics template library (src/services/recording/templates)."""

from __future__ import annotations

import pytest

from src.services.recording.templates import (
    TEMPLATES,
    TemplateNotFoundError,
    get_template,
    list_templates,
)

# Every registered template id — the contract tests run over all of them so a
# new HTML file that breaks the window.PARAMS / RENDER_DURATION_MS / embedded-font
# contract fails CI rather than at capture time.
ALL_TEMPLATE_IDS = sorted(TEMPLATES)
# Templates added for the in-video motion-graphics path (Phase 3 + library).
IN_VIDEO_TEMPLATE_IDS = [
    "animated_title_card",
    "animated_slide",
    "lower_third",
    "data_viz",
    "kinetic_typography",
    "ken_burns",
]


class TestListTemplates:
    def test_seed_templates_registered(self):
        ids = list_templates()
        assert "social_card" in ids
        assert "pipeline_story" in ids

    def test_in_video_templates_registered(self):
        ids = list_templates()
        for tid in IN_VIDEO_TEMPLATE_IDS:
            assert tid in ids

    def test_sorted(self):
        assert list_templates() == sorted(TEMPLATES)


class TestGetTemplate:
    @pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
    def test_returns_html(self, template_id):
        html = get_template(template_id)
        assert "<html" in html.lower()
        assert "render_duration_ms" in html.lower()

    @pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
    def test_fonts_embedded_as_data_uri(self, template_id):
        """Brand fonts must be embedded — capture must not fetch Google Fonts."""
        html = get_template(template_id)
        assert "@font-face" in html
        assert "data:font/woff2;base64," in html
        # No live font-network dependency at capture time.
        assert "fonts.googleapis.com" not in html

    @pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
    def test_reads_window_params(self, template_id):
        """Params are injected at capture time via window.PARAMS, not baked in."""
        assert "window.PARAMS" in get_template(template_id)

    @pytest.mark.parametrize("template_id", IN_VIDEO_TEMPLATE_IDS)
    def test_in_video_templates_define_render(self, template_id):
        """The new in-video templates drive motion through a deterministic
        window.render(t) (freeze-capture safe) rather than wall-clock CSS."""
        assert "window.render" in get_template(template_id)

    def test_unknown_raises(self):
        with pytest.raises(TemplateNotFoundError):
            get_template("does_not_exist")

    def test_caching_returns_same_object(self):
        assert get_template("social_card") is get_template("social_card")


class TestKenBurns:
    """The ken_burns template animates a still image (passed as a data-URI) via a
    deterministic pan/zoom — the standard record_animation path, no FFmpeg."""

    def test_registered(self):
        assert "ken_burns" in list_templates()

    def test_binds_image_param(self):
        html = get_template("ken_burns")
        # The injected image becomes the <img> source via window.PARAMS.image.
        assert "P.image" in html
        assert 'id="kb"' in html

    def test_defines_presets(self):
        html = get_template("ken_burns")
        for preset in ("zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"):
            assert preset in html

    def test_duration_drives_timeline(self):
        """RENDER_DURATION_MS follows the caller's duration so the pan spans the
        whole clip regardless of length."""
        html = get_template("ken_burns")
        assert "RENDER_DURATION_MS" in html
        assert "duration_ms" in html

    def test_fit_contain_mode_present(self):
        """fit=contain keeps a frame-sized designed still fully in view: a blurred
        backdrop (#kbbg) + a contain sharp layer whose scale never exceeds 1.0
        (peak = frame-fill), so slide edge content is never cropped by the motion."""
        html = get_template("ken_burns")
        # The param is honoured and defaults to the back-compat cover behaviour.
        assert "fit:'cover'" in html
        assert "P.fit === 'contain'" in html
        # Blurred backdrop layer + contain-mode class hooks.
        assert 'id="kbbg"' in html
        assert "fit-contain" in html
        # The sharp layer is bounded to <= 1.0 in contain mode (never grows past frame).
        assert "SAFE_MIN" in html
        assert "frameContain" in html

    def test_fit_contain_pan_travel_matches_slack(self):
        """contain-mode pans must drift EXACTLY the sharp layer's per-side slack
        so a frame-sized slide is never cropped: travel is derived from panScale
        as (1 - panScale) * 50, not a value that can exceed the slack. Guards the
        regression where pans held scale 0.96 but drifted 4% (2x the 2% slack)."""
        html = get_template("ken_burns")
        assert "const panScale = (1 + SAFE_MIN) / 2" in html
        assert "const panTravel = (1 - panScale) * 50" in html
        # Pans hold panScale and drift by panTravel (never a raw over-slack value).
        assert "x: (1-e)*panTravel" in html or "x: (1 - e) * panTravel" in html
