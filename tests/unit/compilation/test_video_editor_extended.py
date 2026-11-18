"""Extended unit tests for video editor covering remove_dead_time with freeze detection."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services.compilation.video_editor import VideoEditor
from src.services.ffmpeg import FFmpegOps


class TestRemoveDeadTimeWithFreeze:
    """Tests for remove_dead_time when freeze segments are detected."""

    @patch.object(FFmpegOps, "concat")
    @patch.object(FFmpegOps, "speed_adjust")
    @patch.object(FFmpegOps, "trim")
    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_single_freeze_range(
        self, mock_dur, mock_subprocess, mock_trim, mock_speed, mock_concat
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            # Simulate freezedetect output with one freeze range
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=("[freezedetect] freeze_start: 5.0\n[freezedetect] freeze_end: 10.0\n"),
                stdout="",
            )

            # Mock the trim/speed operations to create output files
            def trim_side_effect(input_path, output_path, start, end):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"trimmed")
                return output_path

            def speed_side_effect(input_path, output_path, factor):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sped")
                return output_path

            def concat_side_effect(concat_file, output_path, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"concatted")
                return output_path

            mock_trim.side_effect = trim_side_effect
            mock_speed.side_effect = speed_side_effect
            mock_concat.side_effect = concat_side_effect

            result = editor.remove_dead_time(video_path)
            assert result.name == "input-nodeadtime.mp4"
            # Should have: live0 (0-5), frozen1 -> sped1 (5-10), live2 (10-30)
            assert mock_trim.call_count >= 2
            mock_speed.assert_called_once()
            mock_concat.assert_called_once()

    @patch.object(FFmpegOps, "concat")
    @patch.object(FFmpegOps, "speed_adjust")
    @patch.object(FFmpegOps, "trim")
    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_multiple_freeze_ranges(
        self, mock_dur, mock_subprocess, mock_trim, mock_speed, mock_concat
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=(
                    "[freezedetect] freeze_start: 3.0\n"
                    "[freezedetect] freeze_end: 6.0\n"
                    "[freezedetect] freeze_start: 15.0\n"
                    "[freezedetect] freeze_end: 20.0\n"
                ),
                stdout="",
            )

            def trim_side_effect(input_path, output_path, start, end):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"trimmed")
                return output_path

            def speed_side_effect(input_path, output_path, factor):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sped")
                return output_path

            def concat_side_effect(concat_file, output_path, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"concatted")
                return output_path

            mock_trim.side_effect = trim_side_effect
            mock_speed.side_effect = speed_side_effect
            mock_concat.side_effect = concat_side_effect

            result = editor.remove_dead_time(video_path)
            assert "nodeadtime" in result.name
            assert mock_speed.call_count == 2

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_freeze_start_parse_error_skipped(self, mock_dur, mock_subprocess):
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            # freeze_start value is unparseable
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=(
                    "[freezedetect] freeze_start: NOT_A_NUMBER\n[freezedetect] freeze_end: 10.0\n"
                ),
                stdout="",
            )

            result = editor.remove_dead_time(video_path)
            # With a parse error on freeze_start, no valid range is captured
            assert result == video_path

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_freeze_end_parse_error_skipped(self, mock_dur, mock_subprocess):
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=(
                    "[freezedetect] freeze_start: 5.0\n[freezedetect] freeze_end: NOT_A_NUMBER\n"
                ),
                stdout="",
            )

            result = editor.remove_dead_time(video_path)
            # freeze_end parse error, no valid range
            assert result == video_path

    @patch.object(FFmpegOps, "concat")
    @patch.object(FFmpegOps, "speed_adjust")
    @patch.object(FFmpegOps, "trim")
    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_freeze_at_start_no_live_prefix(
        self, mock_dur, mock_subprocess, mock_trim, mock_speed, mock_concat
    ):
        """When the freeze starts at 0.0, there's no live segment before it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=("[freezedetect] freeze_start: 0.0\n[freezedetect] freeze_end: 5.0\n"),
                stdout="",
            )

            def trim_side_effect(input_path, output_path, start, end):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"trimmed")
                return output_path

            def speed_side_effect(input_path, output_path, factor):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sped")
                return output_path

            def concat_side_effect(concat_file, output_path, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"concatted")
                return output_path

            mock_trim.side_effect = trim_side_effect
            mock_speed.side_effect = speed_side_effect
            mock_concat.side_effect = concat_side_effect

            result = editor.remove_dead_time(video_path)
            assert "nodeadtime" in result.name

    @patch.object(FFmpegOps, "concat")
    @patch.object(FFmpegOps, "speed_adjust")
    @patch.object(FFmpegOps, "trim")
    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_freeze_covers_entire_video(
        self, mock_dur, mock_subprocess, mock_trim, mock_speed, mock_concat
    ):
        """When the freeze covers the entire video, still processes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")

            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr=("[freezedetect] freeze_start: 0.0\n[freezedetect] freeze_end: 10.0\n"),
                stdout="",
            )

            def trim_side_effect(input_path, output_path, start, end):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"trimmed")
                return output_path

            def speed_side_effect(input_path, output_path, factor):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"sped")
                return output_path

            def concat_side_effect(concat_file, output_path, **kwargs):
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"concatted")
                return output_path

            mock_trim.side_effect = trim_side_effect
            mock_speed.side_effect = speed_side_effect
            mock_concat.side_effect = concat_side_effect

            result = editor.remove_dead_time(video_path)
            assert "nodeadtime" in result.name
