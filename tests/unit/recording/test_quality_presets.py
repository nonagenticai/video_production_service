"""Tests for src/services/quality_presets.py -- Quality preset definitions."""

from __future__ import annotations

import pytest

from src.services.recording.quality_presets import (
    QUALITY_PRESETS,
    QualityPreset,
    get_quality_preset,
)


class TestQualityPreset:
    def test_dataclass_fields(self):
        preset = QualityPreset(
            viewport_width=1920,
            viewport_height=1080,
            crf=23,
            video_bitrate="5M",
            label="Standard",
        )
        assert preset.viewport_width == 1920
        assert preset.viewport_height == 1080
        assert preset.crf == 23
        assert preset.video_bitrate == "5M"
        assert preset.label == "Standard"

    def test_frozen(self):
        preset = QualityPreset(
            viewport_width=1920,
            viewport_height=1080,
            crf=23,
            video_bitrate="5M",
            label="Test",
        )
        with pytest.raises(AttributeError):
            setattr(preset, "crf", 18)  # noqa: B010


class TestQualityPresetsDict:
    def test_all_presets_exist(self):
        assert "preview" in QUALITY_PRESETS
        assert "standard" in QUALITY_PRESETS
        assert "high_quality" in QUALITY_PRESETS
        assert "2k" in QUALITY_PRESETS
        assert "4k" in QUALITY_PRESETS

    def test_preview_is_720p(self):
        p = QUALITY_PRESETS["preview"]
        assert p.viewport_width == 1280
        assert p.viewport_height == 720

    def test_standard_is_1080p(self):
        p = QUALITY_PRESETS["standard"]
        assert p.viewport_width == 1920
        assert p.viewport_height == 1080

    def test_high_quality_is_1080p_low_crf(self):
        p = QUALITY_PRESETS["high_quality"]
        assert p.viewport_width == 1920
        assert p.viewport_height == 1080
        assert p.crf < QUALITY_PRESETS["standard"].crf

    def test_2k_is_1440p(self):
        p = QUALITY_PRESETS["2k"]
        assert p.viewport_width == 2560
        assert p.viewport_height == 1440

    def test_4k_is_2160p(self):
        p = QUALITY_PRESETS["4k"]
        assert p.viewport_width == 3840
        assert p.viewport_height == 2160


class TestGetQualityPreset:
    def test_valid_preset(self):
        preset = get_quality_preset("standard")
        assert preset.viewport_width == 1920
        assert preset.viewport_height == 1080

    def test_case_insensitive(self):
        preset = get_quality_preset("PREVIEW")
        assert preset.viewport_width == 1280

    def test_strips_whitespace(self):
        preset = get_quality_preset("  4k  ")
        assert preset.viewport_width == 3840

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown quality preset"):
            get_quality_preset("ultra_hd")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Unknown quality preset"):
            get_quality_preset("")
