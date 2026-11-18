"""Unit tests for theme system."""

from __future__ import annotations

import pytest


class TestValidateColor:
    def test_valid_hex_colors(self):
        from src.services.compilation.themes import validate_color

        assert validate_color("#fff") == "#fff"
        assert validate_color("#FFFFFF") == "#FFFFFF"
        assert validate_color("#FF000080") == "#FF000080"

    def test_valid_rgb_colors(self):
        from src.services.compilation.themes import validate_color

        assert validate_color("rgb(255,0,0)") == "rgb(255,0,0)"
        assert validate_color("rgba(255,0,0,0.5)") == "rgba(255,0,0,0.5)"

    def test_invalid_colors(self):
        from src.services.compilation.themes import validate_color

        with pytest.raises(ValueError):
            validate_color("not-a-color")
        with pytest.raises(ValueError):
            validate_color("red; } body { display:none")

    def test_hex_three_digit(self):
        from src.services.compilation.themes import validate_color

        assert validate_color("#abc") == "#abc"
        assert validate_color("#ABC") == "#ABC"

    def test_hex_eight_digit_with_alpha(self):
        from src.services.compilation.themes import validate_color

        assert validate_color("#00FF00FF") == "#00FF00FF"

    def test_rgb_with_spaces(self):
        from src.services.compilation.themes import validate_color

        assert validate_color("rgb(255, 0, 0)") == "rgb(255, 0, 0)"

    def test_empty_string_invalid(self):
        from src.services.compilation.themes import validate_color

        with pytest.raises(ValueError):
            validate_color("")


class TestThemePresets:
    def test_all_presets_exist(self):
        from src.services.compilation.themes import THEME_PRESETS

        expected = {"default", "dark", "light", "corporate", "creative", "minimal"}
        assert set(THEME_PRESETS.keys()) == expected

    def test_get_theme_existing(self):
        from src.services.compilation.themes import get_theme

        theme = get_theme("dark")
        assert theme.name == "dark"

    def test_get_theme_fallback(self):
        from src.services.compilation.themes import get_theme

        theme = get_theme("nonexistent")
        assert theme.name == "default"

    def test_list_themes(self):
        from src.services.compilation.themes import list_themes

        themes = list_themes()
        assert len(themes) == 6

    def test_all_presets_have_valid_colors(self):
        from src.services.compilation.themes import THEME_PRESETS

        for name, preset in THEME_PRESETS.items():
            assert preset.primary_color.startswith("#"), f"{name} primary_color invalid"
            assert preset.background_color.startswith("#"), f"{name} background_color invalid"


class TestResolveThemeConfig:
    def test_with_preset_only(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config("dark", None)
        assert config["primary_color"] == "#10b981"
        assert config["background_color"] == "#111827"

    def test_with_overrides(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config("dark", {"primary_color": "#FF0000"})
        assert config["primary_color"] == "#FF0000"
        assert config["background_color"] == "#111827"

    def test_no_preset_only_overrides(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config(None, {"primary_color": "#FF0000"})
        assert config["primary_color"] == "#FF0000"
        assert "background_color" not in config

    def test_invalid_override_color(self):
        from src.services.compilation.themes import resolve_theme_config

        with pytest.raises(ValueError):
            resolve_theme_config("dark", {"primary_color": "invalid"})

    def test_no_preset_no_overrides(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config(None, None)
        assert config == {}

    def test_override_with_none_value_ignored(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config("dark", {"primary_color": None})
        # None override should be ignored, keeping the preset value
        assert config["primary_color"] == "#10b981"

    def test_theme_name_key_in_overrides_skipped(self):
        from src.services.compilation.themes import resolve_theme_config

        # The theme_name key should be skipped in overrides
        config = resolve_theme_config("dark", {"theme_name": "ignored"})
        assert "theme_name" not in config

    def test_nonexistent_preset_falls_back_to_default(self):
        from src.services.compilation.themes import resolve_theme_config

        config = resolve_theme_config("nonexistent", None)
        # get_theme falls back to "default"
        assert config["primary_color"] == "#6366f1"
