"""Unit tests for the opt-in animated (motion) intro/outro title card path.

Covers the param mapping (part + resolved theme -> animated_title_card PARAMS) and
the dispatch: success renders via the AnimationRecorder; any failure soft-falls
back to the static Pillow card so a bad clip never fails the compile.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.compilation._compiler_context import _JobContext
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from tests.conftest import TENANT_U1


class _Builder(_SegmentBuildersMixin):
    """Concrete handle on the mixin under test."""


def _ctx(tmp_path: Path) -> _JobContext:
    ctx = _JobContext(job_id="job1", tenant_id=TENANT_U1, job=None, settings=None)
    ctx.title_cards_dir = tmp_path
    ctx.target_w = 1920
    ctx.target_h = 1080
    ctx.theme_config = {
        "gradient_start": "#111111",
        "gradient_end": "#222222",
        "accent_color": "#14B8A6",
        "text_color": "#FFFFFF",
        "subtitle_color": "#999999",
    }
    return ctx


class _Job:
    def __init__(self, title_card_config=None):
        self.title_card_config = title_card_config


def test_title_card_motion_on_per_part_flag(tmp_path):
    ctx = _ctx(tmp_path)
    assert _Builder._title_card_motion_on(ctx, {"motion": True}) is True
    assert _Builder._title_card_motion_on(ctx, {}) is False


def test_title_card_motion_on_job_level_flag(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.job = _Job({"motion": True})
    assert _Builder._title_card_motion_on(ctx, {}) is True
    ctx.job = _Job({"duration": 5.0})
    assert _Builder._title_card_motion_on(ctx, {}) is False


def test_motion_title_params_maps_part_and_theme(tmp_path):
    params = _Builder()._motion_title_params(
        _ctx(tmp_path),
        {"title": "Hello", "subtitle": "World", "chapter": 3, "eyebrow": "Intro"},
    )
    assert params["title"] == "Hello"
    assert params["subtitle"] == "World"
    assert params["chapter"] == "Chapter 3"
    assert params["eyebrow"] == "Intro"
    assert params["bg"] == "#111111"
    assert params["bg2"] == "#222222"
    assert params["accent"] == "#14B8A6"
    assert params["text"] == "#FFFFFF"
    assert params["muted"] == "#999999"


def test_motion_title_params_omits_absent_keys(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.theme_config = {"primary_color": "#abcdef"}  # only the accent fallback present
    params = _Builder()._motion_title_params(ctx, {"title": "T"})
    assert params["accent"] == "#abcdef"  # primary_color fallback
    assert "bg" not in params
    assert "text" not in params
    assert "chapter" not in params  # absent chapter is not stamped


@pytest.mark.asyncio
async def test_motion_title_card_success(tmp_path, monkeypatch):
    builder = _Builder()
    ctx = _ctx(tmp_path)
    produced = tmp_path / "raw.mp4"
    produced.write_bytes(b"x")  # the recorder's output, moved into title_cards_dir

    rec = MagicMock()
    rec.record_animation = AsyncMock(return_value={"success": True, "video_path": str(produced)})
    monkeypatch.setattr("src.services.recording.animation_recorder.AnimationRecorder", lambda: rec)
    monkeypatch.setattr(
        builder,
        "_segment_with_optional_narration",
        MagicMock(side_effect=lambda _c, _p, path: path),
    )

    await builder._process_title_card_motion(
        ctx, "intro", {"title": "Hi", "estimated_duration_seconds": 4}
    )

    assert ctx.segments == [tmp_path / "intro.mp4"]
    assert (tmp_path / "intro.mp4").exists()  # moved into place
    rec.record_animation.assert_awaited_once()


@pytest.mark.asyncio
async def test_motion_title_card_falls_back_to_static_on_failure(tmp_path, monkeypatch):
    builder = _Builder()
    ctx = _ctx(tmp_path)

    rec = MagicMock()
    rec.record_animation = AsyncMock(return_value={"success": False, "error": "boom"})
    monkeypatch.setattr("src.services.recording.animation_recorder.AnimationRecorder", lambda: rec)
    static = MagicMock()
    monkeypatch.setattr(builder, "_process_title_card", static)

    await builder._process_title_card_motion(ctx, "intro", {"title": "Hi"})

    static.assert_called_once_with(ctx, "intro", {"title": "Hi"})
    assert ctx.segments == []  # motion path appended nothing; static (mocked) handles it
