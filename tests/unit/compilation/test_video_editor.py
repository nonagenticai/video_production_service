"""Unit tests for video editor service.

Rewritten to verify actual arguments passed to FFmpeg operations and
correct threshold behavior, rather than just asserting mocks were called.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services.compilation.video_editor import VideoEditor
from src.services.ffmpeg import FFmpegOps


class TestVideoEditorInit:
    def test_default_output_dir_resolves(self):
        editor = VideoEditor()
        expected = (Path(tempfile.gettempdir()) / "edited").resolve()
        assert editor.output_dir == expected
        assert editor.crf == 20

    def test_custom_output_dir_and_crf(self):
        editor = VideoEditor(output_dir=Path("/custom/dir"), crf=18)
        assert editor.output_dir == Path("/custom/dir").resolve()
        assert editor.crf == 18


class TestVideoEditorEditedPath:
    def test_format_includes_stem_suffix_and_extension(self):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        result = editor._edited_path(Path("/input/clip.mp4"), "trimmed")
        assert result.name == "clip-trimmed.mp4"
        assert result.parent == Path("/tmp/test").resolve()

    def test_preserves_original_file_extension(self):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        result = editor._edited_path(Path("/input/clip.webm"), "speed2")
        assert result.suffix == ".webm"
        assert result.stem == "clip-speed2"

    def test_different_suffixes_produce_different_paths(self):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        trimmed = editor._edited_path(Path("/input/clip.mp4"), "trimmed")
        sped = editor._edited_path(Path("/input/clip.mp4"), "speed2")
        assert trimmed != sped
        assert trimmed.stem == "clip-trimmed"
        assert sped.stem == "clip-speed2"


class TestVideoEditorSpeedAdjust:
    @patch.object(FFmpegOps, "speed_adjust")
    def test_passes_correct_output_path_and_factor(self, mock_speed):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        mock_speed.return_value = Path("/tmp/test/clip-speed2.0.mp4")

        with patch("pathlib.Path.mkdir"):
            result = editor.speed_adjust(Path("/input/clip.mp4"), 2.0)

        # Verify the delegated call received the correct arguments
        call_args = mock_speed.call_args[0]
        assert call_args[0] == Path("/input/clip.mp4")  # input
        assert call_args[1] == Path("/tmp/test").resolve() / "clip-speed2.0.mp4"  # output
        assert call_args[2] == 2.0  # factor
        assert result == Path("/tmp/test/clip-speed2.0.mp4")


class TestVideoEditorTrim:
    @patch.object(FFmpegOps, "trim")
    def test_passes_correct_timestamps(self, mock_trim):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        mock_trim.return_value = Path("/tmp/test/clip-trimmed.mp4")

        with patch("pathlib.Path.mkdir"):
            editor.trim(Path("/input/clip.mp4"), 5.0, 15.0)

        call_args = mock_trim.call_args[0]
        assert call_args[0] == Path("/input/clip.mp4")  # input
        assert call_args[2] == 5.0  # start_s
        assert call_args[3] == 15.0  # end_s


class TestVideoEditorAutoTrim:
    @patch.object(FFmpegOps, "probe_duration", return_value=1.5)
    def test_returns_original_when_video_too_short(self, mock_dur):
        """Video must be > trim_start + trim_end + 1.0s to be trimmed."""
        editor = VideoEditor()
        result = editor.auto_trim(Path("/input/short.mp4"))
        assert result == Path("/input/short.mp4")

    @patch.object(FFmpegOps, "probe_duration", return_value=2.0)
    def test_returns_original_at_exact_threshold(self, mock_dur):
        """2.0s == 0.5 + 0.5 + 1.0, should NOT be trimmed (<=, not <)."""
        editor = VideoEditor()
        result = editor.auto_trim(Path("/input/exact.mp4"))
        assert result == Path("/input/exact.mp4")

    @patch.object(FFmpegOps, "trim")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_computes_correct_trim_range(self, mock_dur, mock_trim):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        mock_trim.return_value = Path("/tmp/test/clip-trimmed.mp4")

        with patch("pathlib.Path.mkdir"):
            editor.auto_trim(Path("/input/long.mp4"))

        call_args = mock_trim.call_args[0]
        assert call_args[2] == 0.5  # start_s (default trim_start)
        assert call_args[3] == 29.5  # end_s (30.0 - 0.5 default trim_end)

    @patch.object(FFmpegOps, "trim")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_respects_custom_trim_values(self, mock_dur, mock_trim):
        editor = VideoEditor(output_dir=Path("/tmp/test"))
        mock_trim.return_value = Path("/tmp/test/clip-trimmed.mp4")

        with patch("pathlib.Path.mkdir"):
            editor.auto_trim(Path("/input/long.mp4"), trim_start=1.0, trim_end=2.0)

        call_args = mock_trim.call_args[0]
        assert call_args[2] == 1.0  # custom start
        assert call_args[3] == 28.0  # 30.0 - 2.0


class TestVideoEditorRemoveDeadTime:
    @patch.object(FFmpegOps, "probe_duration", return_value=1.0)
    def test_too_short_returns_original(self, mock_dur):
        """Video shorter than 2x freeze_threshold should be returned unchanged."""
        editor = VideoEditor()
        result = editor.remove_dead_time(Path("/input/short.mp4"))
        assert result == Path("/input/short.mp4")

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_no_freeze_detected_returns_original(self, mock_dur, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="", stdout="")
        editor = VideoEditor()
        result = editor.remove_dead_time(Path("/input/video.mp4"))
        assert result == Path("/input/video.mp4")

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_timeout_gracefully_returns_original(self, mock_dur, mock_subprocess):
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)
        editor = VideoEditor()
        result = editor.remove_dead_time(Path("/input/video.mp4"))
        assert result == Path("/input/video.mp4")

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_oserror_gracefully_returns_original(self, mock_dur, mock_subprocess):
        mock_subprocess.side_effect = OSError("command not found")
        editor = VideoEditor()
        result = editor.remove_dead_time(Path("/input/video.mp4"))
        assert result == Path("/input/video.mp4")

    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_freezedetect_command_includes_correct_threshold(self, mock_dur, mock_subprocess):
        """Verify the freezedetect filter uses the correct duration threshold."""
        mock_subprocess.return_value = MagicMock(returncode=0, stderr="", stdout="")
        editor = VideoEditor()
        editor.remove_dead_time(Path("/input/video.mp4"), freeze_threshold_s=3.0)

        call_args = mock_subprocess.call_args[0][0]
        assert "freezedetect" in " ".join(call_args)
        vf_arg = call_args[call_args.index("-vf") + 1]
        assert "d=3.0" in vf_arg
