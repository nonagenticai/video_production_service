"""Tests for src/services/recording/content_recorder.py -- ContentRecorder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from playwright.async_api import Error as PlaywrightError

from src.services.recording.content_recorder import (
    _DEFAULT_END_DELAY,
    _DEFAULT_SCROLL_SPEED,
    _DEFAULT_START_DELAY,
    ContentRecorder,
)


def _mock_settings():
    """Return mock settings suitable for ContentRecorder.__init__."""
    s = MagicMock()
    s.recording.viewport_width = 1920
    s.recording.viewport_height = 1080
    s.recording.headless = True
    s.recording.output_format = "mp4"
    s.recording.default_quality_preset = "standard"
    s.ffmpeg.preset = "medium"
    s.ffmpeg.timeout = 300
    s.ffmpeg.default_crf = 23
    return s


def _make_recorder():
    """Create a ContentRecorder with mocked settings."""
    with patch(
        "src.services.recording.content_recorder.get_settings",
        return_value=_mock_settings(),
    ):
        return ContentRecorder()


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_defaults(self):
        recorder = _make_recorder()
        config = recorder._resolve_config(None)
        assert config["viewport_width"] == 1920
        assert config["viewport_height"] == 1080
        assert config["headless"] is True
        assert config["output_format"] == "mp4"
        assert config["quality_preset"] == "standard"
        assert config["color_scheme"] == "light"

    def test_overrides(self):
        recorder = _make_recorder()
        config = recorder._resolve_config(
            {
                "viewport_width": 1280,
                "viewport_height": 720,
                "headless": False,
                "output_format": "webm",
                "color_scheme": "dark",
            }
        )
        assert config["viewport_width"] == 1280
        assert config["viewport_height"] == 720
        assert config["headless"] is False
        assert config["output_format"] == "webm"
        assert config["color_scheme"] == "dark"

    def test_partial_overrides_keep_defaults(self):
        recorder = _make_recorder()
        config = recorder._resolve_config({"viewport_width": 800})
        assert config["viewport_width"] == 800
        # Other values remain at preset defaults
        assert config["headless"] is True
        assert config["color_scheme"] == "light"

    def test_quality_preset_override(self):
        recorder = _make_recorder()
        config = recorder._resolve_config({"quality_preset": "preview"})
        assert config["quality_preset"] == "preview"
        # viewport_width/height come from the preset, not the settings defaults
        assert config["viewport_width"] == 1280
        assert config["viewport_height"] == 720

    def test_none_values_ignored(self):
        recorder = _make_recorder()
        config = recorder._resolve_config({"viewport_width": None, "headless": None})
        # Should keep defaults when override values are None
        assert config["viewport_width"] == 1920
        assert config["headless"] is True


# ---------------------------------------------------------------------------
# _resolve_scroll_config
# ---------------------------------------------------------------------------


class TestResolveScrollConfig:
    def test_defaults(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config(None)
        assert scroll["enabled"] is True
        assert scroll["speed_px_per_second"] == _DEFAULT_SCROLL_SPEED
        assert scroll["start_delay_seconds"] == _DEFAULT_START_DELAY
        assert scroll["end_delay_seconds"] == _DEFAULT_END_DELAY

    def test_overrides(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config(
            {
                "enabled": False,
                "speed_px_per_second": 500,
                "start_delay_seconds": 2.0,
                "end_delay_seconds": 3.0,
            }
        )
        assert scroll["enabled"] is False
        assert scroll["speed_px_per_second"] == 500
        assert scroll["start_delay_seconds"] == 2.0
        assert scroll["end_delay_seconds"] == 3.0

    def test_partial_overrides(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config({"speed_px_per_second": 100})
        assert scroll["speed_px_per_second"] == 100
        assert scroll["enabled"] is True
        assert scroll["start_delay_seconds"] == _DEFAULT_START_DELAY

    def test_none_values_ignored(self):
        recorder = _make_recorder()
        scroll = recorder._resolve_scroll_config({"speed_px_per_second": None})
        assert scroll["speed_px_per_second"] == _DEFAULT_SCROLL_SPEED


# ---------------------------------------------------------------------------
# record_content (orchestration)
# ---------------------------------------------------------------------------


class TestRecordContent:
    async def test_success_flow(self):
        recorder = _make_recorder()

        fake_video = MagicMock()
        fake_video.suffix = ".mp4"
        fake_video.stat.return_value.st_size = 12345

        mock_probe = MagicMock()
        mock_probe.duration = 10.5
        mock_probe.width = 1920
        mock_probe.height = 1080
        mock_probe.has_audio = False

        with (
            patch.object(
                recorder,
                "_record_with_playwright",
                new_callable=AsyncMock,
                return_value=fake_video,
            ),
            patch.object(
                recorder,
                "_post_process",
                new_callable=AsyncMock,
                return_value=fake_video,
            ),
            patch(
                "src.services.recording.content_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_probe,
            ),
        ):
            result = await recorder.record_content(
                job_id="j1",
                html_content="<h1>Test</h1>",
            )

        assert result["success"] is True
        assert result["duration_seconds"] == 10.5
        assert result["file_size_bytes"] == 12345

    async def test_html_content_forwarded_verbatim(self):
        """Regression: _record_with_playwright receives html_content, not a URL."""
        recorder = _make_recorder()

        fake_video = MagicMock()
        fake_video.suffix = ".mp4"
        fake_video.stat.return_value.st_size = 1

        mock_probe = MagicMock()
        mock_probe.duration = 1.0
        mock_probe.width = 1920
        mock_probe.height = 1080
        mock_probe.has_audio = False

        html = "<html><body><h1>Inline CSS only</h1></body></html>"

        with (
            patch.object(
                recorder,
                "_record_with_playwright",
                new_callable=AsyncMock,
                return_value=fake_video,
            ) as mock_record,
            patch.object(
                recorder,
                "_post_process",
                new_callable=AsyncMock,
                return_value=fake_video,
            ),
            patch(
                "src.services.recording.content_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_probe,
            ),
        ):
            await recorder.record_content(job_id="j-html", html_content=html)

        mock_record.assert_awaited_once()
        await_args = mock_record.await_args
        assert await_args is not None
        forwarded_second_arg = await_args.args[1]
        assert forwarded_second_arg == html
        assert not forwarded_second_arg.startswith("http://")

    async def test_failure_returns_error(self):
        recorder = _make_recorder()

        with patch.object(
            recorder,
            "_record_with_playwright",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Recording failed"),
        ):
            result = await recorder.record_content(
                job_id="j-fail",
                html_content="<h1>Fail</h1>",
            )

        assert result["success"] is False
        assert "Recording failed" in result["error"]

    async def test_playwright_error_caught(self):
        """Regression: `page.set_content` failures (PlaywrightError) don't escape."""
        recorder = _make_recorder()

        with patch.object(
            recorder,
            "_record_with_playwright",
            new_callable=AsyncMock,
            side_effect=PlaywrightError("Page crashed"),
        ):
            result = await recorder.record_content(
                job_id="j-pw-err",
                html_content="<h1>Boom</h1>",
            )

        assert result["success"] is False
        assert "Page crashed" in result["error"]

    async def test_progress_callbacks(self):
        recorder = _make_recorder()
        callback = AsyncMock()

        fake_video = MagicMock()
        fake_video.suffix = ".mp4"
        fake_video.stat.return_value.st_size = 500

        mock_probe = MagicMock()
        mock_probe.duration = 5.0
        mock_probe.width = 1920
        mock_probe.height = 1080
        mock_probe.has_audio = False

        with (
            patch.object(
                recorder,
                "_record_with_playwright",
                new_callable=AsyncMock,
                return_value=fake_video,
            ),
            patch.object(
                recorder,
                "_post_process",
                new_callable=AsyncMock,
                return_value=fake_video,
            ),
            patch(
                "src.services.recording.content_recorder.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=mock_probe,
            ),
        ):
            result = await recorder.record_content(
                job_id="j-cb",
                html_content="<p>test</p>",
                progress_callback=callback,
            )

        assert result["success"] is True
        # Outer callbacks are: launching_browser (0.3), post_processing (0.8), completed (1.0).
        # `_record_with_playwright` is mocked, so its internal callbacks don't fire.
        assert callback.await_count == 3
        stages = [call.args[1] for call in callback.call_args_list]
        assert "launching_browser" in stages
        assert "post_processing" in stages
        assert "completed" in stages
        assert "starting_html_server" not in stages


# ---------------------------------------------------------------------------
# _post_process
# ---------------------------------------------------------------------------


class TestPostProcess:
    async def test_webm_to_mp4_conversion(self):
        recorder = _make_recorder()
        video_path = MagicMock()
        video_path.suffix = ".webm"
        mp4_output = MagicMock()
        video_path.with_suffix.return_value = mp4_output

        with patch(
            "src.services.recording.content_recorder.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=mp4_output,
        ) as mock_thread:
            result = await recorder._post_process(
                video_path, {"output_format": "mp4", "quality_preset": "standard"}
            )

        assert result is mp4_output
        mock_thread.assert_awaited_once()

    async def test_webm_kept_when_format_is_webm(self):
        recorder = _make_recorder()
        video_path = MagicMock()
        video_path.suffix = ".webm"

        result = await recorder._post_process(video_path, {"output_format": "webm"})
        assert result is video_path

    async def test_mp4_not_converted(self):
        recorder = _make_recorder()
        video_path = MagicMock()
        video_path.suffix = ".mp4"

        result = await recorder._post_process(video_path, {"output_format": "mp4"})
        assert result is video_path
