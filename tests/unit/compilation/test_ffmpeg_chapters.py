"""Unit tests for FFmpegOps.embed_chapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.services.ffmpeg import FFmpegOps


class TestEmbedChapterMarkers:
    def test_empty_chapters_returns_video_path(self, tmp_path):
        """When chapters list is empty, return the original video path."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        result = FFmpegOps.embed_chapters(video_path, output_path, chapters=[])
        assert result == video_path

    @patch.object(FFmpegOps, "run")
    def test_embed_single_chapter(self, mock_run, tmp_path):
        """Embed a single chapter marker."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "out" / "output.mp4"

        chapters = [
            {
                "chapter_number": 1,
                "title": "Intro",
                "start_seconds": 0.0,
                "duration_seconds": 10.0,
            }
        ]

        mock_run.return_value = MagicMock(returncode=0)

        result = FFmpegOps.embed_chapters(video_path, output_path, chapters)

        assert result == output_path
        mock_run.assert_called_once()
        # Verify ffmpeg args contain -map_metadata
        call_args = mock_run.call_args[0][0]
        assert "-map_metadata" in call_args
        assert "1" in call_args

    @patch.object(FFmpegOps, "run")
    def test_embed_multiple_chapters(self, mock_run, tmp_path):
        """Embed multiple chapter markers."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "chaptered.mp4"

        chapters = [
            {
                "chapter_number": 1,
                "title": "Intro",
                "start_seconds": 0.0,
                "duration_seconds": 5.0,
            },
            {
                "chapter_number": 2,
                "title": "Main Content",
                "start_seconds": 5.0,
                "duration_seconds": 20.0,
            },
            {
                "chapter_number": 3,
                "title": "Outro",
                "start_seconds": 25.0,
                "duration_seconds": 5.0,
            },
        ]

        mock_run.return_value = MagicMock(returncode=0)
        result = FFmpegOps.embed_chapters(video_path, output_path, chapters)
        assert result == output_path

    @patch.object(FFmpegOps, "run")
    def test_chapter_without_title_uses_default(self, mock_run, tmp_path):
        """Chapter without title gets a default title."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        chapters = [
            {
                "chapter_number": 3,
                "start_seconds": 10.0,
                "duration_seconds": 15.0,
            }
        ]

        # The ffmetadata file is deleted once run() returns, so read it from
        # inside the call — that text IS the chapter list the player will show.
        captured: dict[str, str] = {}

        def _capture(argv):
            captured["metadata"] = Path(argv[argv.index("-i", 2) + 1]).read_text(encoding="utf-8")
            return MagicMock(returncode=0)

        mock_run.side_effect = _capture
        FFmpegOps.embed_chapters(video_path, output_path, chapters)

        mock_run.assert_called_once()
        # A chapter with no title must fall back to "Chapter <n>", NOT to an
        # empty title — an empty `title=` line ships a nameless entry in the
        # player's chapter menu.
        assert "title=Chapter 3" in captured["metadata"]
        assert "title=\n" not in captured["metadata"]
        # …and the timings still come from the chapter dict.
        assert "START=10000" in captured["metadata"]
        assert "END=25000" in captured["metadata"]

    @patch.object(FFmpegOps, "run")
    def test_metadata_file_cleanup(self, mock_run, tmp_path):
        """Metadata temp file should be cleaned up after embedding."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        chapters = [
            {
                "chapter_number": 1,
                "title": "Test",
                "start_seconds": 0.0,
                "duration_seconds": 10.0,
            }
        ]

        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.embed_chapters(video_path, output_path, chapters)

        metadata_path = video_path.parent / f"{video_path.stem}_chapters.txt"
        # After embed, the metadata file should be deleted
        assert not metadata_path.exists()

    @patch.object(FFmpegOps, "run")
    def test_metadata_file_cleanup_oserror_handled(self, mock_run, tmp_path):
        """OSError during metadata file cleanup should be silently handled."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_path = tmp_path / "output.mp4"

        chapters = [
            {
                "chapter_number": 1,
                "title": "Test",
                "start_seconds": 0.0,
                "duration_seconds": 10.0,
            }
        ]

        mock_run.return_value = MagicMock(returncode=0)

        # Make unlink raise OSError
        with patch.object(Path, "unlink", side_effect=OSError("permission denied")):
            # Should not raise
            result = FFmpegOps.embed_chapters(video_path, output_path, chapters)
            assert result == output_path
