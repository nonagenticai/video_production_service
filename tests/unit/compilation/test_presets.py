"""Unit tests for compilation presets."""

from __future__ import annotations

import pytest


class TestPresets:
    def test_get_preset_existing(self):
        from src.services.compilation.presets import get_preset

        preset = get_preset("standard")
        assert preset is not None
        assert preset.target_width == 1920
        assert preset.crf == 20

    def test_get_preset_nonexistent(self):
        from src.services.compilation.presets import get_preset

        assert get_preset("nonexistent") is None

    def test_list_presets(self):
        from src.services.compilation.presets import list_presets

        presets = list_presets()
        assert len(presets) == 5
        names = {p.name for p in presets}
        assert names == {"default", "quick-preview", "standard", "high-quality", "4k"}

    def test_4k_preset_values(self):
        from src.services.compilation.presets import get_preset

        preset = get_preset("4k")
        assert preset is not None
        assert preset.target_width == 3840
        assert preset.target_height == 2160

    def test_quick_preview_preset(self):
        from src.services.compilation.presets import get_preset

        preset = get_preset("quick-preview")
        assert preset is not None
        assert preset.target_width == 854
        assert preset.target_height == 480
        assert preset.crf == 28
        assert preset.transition_type == "cut"

    def test_high_quality_preset(self):
        from src.services.compilation.presets import get_preset

        preset = get_preset("high-quality")
        assert preset is not None
        assert preset.crf == 16
        assert preset.transition_type == "fade"
        assert preset.crossfade_duration == 0.7


class TestResolvePresetConfig:
    def test_preset_only(self):
        from src.services.compilation.presets import resolve_preset_config

        config = resolve_preset_config("quick-preview")
        assert config["target_width"] == 854
        assert config["crf"] == 28
        assert config["transition_type"] == "cut"

    def test_override_takes_precedence(self):
        from src.services.compilation.presets import resolve_preset_config

        config = resolve_preset_config("standard", crf=16)
        assert config["crf"] == 16
        assert config["target_width"] == 1920  # from preset

    def test_no_preset(self):
        from src.services.compilation.presets import resolve_preset_config

        config = resolve_preset_config(None, target_width=1280)
        assert config["target_width"] == 1280
        assert "crf" not in config

    def test_invalid_preset_raises(self):
        from src.services.compilation.presets import resolve_preset_config

        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_preset_config("nonexistent")

    def test_multiple_overrides(self):
        from src.services.compilation.presets import resolve_preset_config

        config = resolve_preset_config(
            "standard",
            target_width=1280,
            target_height=720,
            crf=25,
            transition_type="cut",
            crossfade_duration=0.0,
        )
        assert config["target_width"] == 1280
        assert config["target_height"] == 720
        assert config["crf"] == 25
        assert config["transition_type"] == "cut"
        assert config["crossfade_duration"] == 0.0

    def test_no_preset_no_overrides(self):
        from src.services.compilation.presets import resolve_preset_config

        config = resolve_preset_config(None)
        assert config == {}
