"""An UNMEASURED ffprobe must never reach the compiler as a real measurement.

``FFmpegOps.probe_all`` swallows an ffprobe failure -- a crash, a non-zero
exit, or the 10s ``FFPROBE_TIMEOUT`` elapsing while two other ffmpeg
subprocesses saturate a 3-core pod -- into
``ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=False)``.
Its own docstring calls ``probed`` "the signal a caller MUST check". Before
this module, exactly ONE caller in the service checked it, and all three
convenience accessors dropped it, so a single transient probe timeout
degraded a compile SILENTLY -- no ERROR line, job ``completed``, tenant
billed -- via a fabricated 0.0 / (0, 0) / False.

These tests pin the accessors to "an unmeasured probe RAISES", and each
regression test below asserts the concrete downstream corruption that the
silent zero used to produce.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.compilation._assembly import _AssemblyMixin
from src.services.compilation.quality_validator import QualityValidator
from src.services.ffmpeg import FFmpegOps
from src.services.ffmpeg.base import ProbeResult

_UNPROBED = ProbeResult(duration=0.0, width=0, height=0, has_audio=False, probed=False)
_EMPTY_BUT_PROBED = ProbeResult(
    duration=0.0, width=0, height=0, has_audio=False, has_video=False, probed=True
)


class TestAccessorsRefuseAnUnmeasuredProbe:
    """The gap is made loud once, at the only point that can tell the difference."""

    @patch.object(FFmpegOps, "probe_all", return_value=_UNPROBED)
    def test_probe_duration_raises_instead_of_returning_zero(self, _mock) -> None:
        with pytest.raises(RuntimeError, match="UNMEASURED"):
            FFmpegOps.probe_duration(Path("/f.mp4"))

    @patch.object(FFmpegOps, "probe_all", return_value=_UNPROBED)
    def test_probe_dimensions_raises_instead_of_returning_zero_zero(self, _mock) -> None:
        with pytest.raises(RuntimeError, match="UNMEASURED"):
            FFmpegOps.probe_dimensions(Path("/f.mp4"))

    @patch.object(FFmpegOps, "probe_all", return_value=_UNPROBED)
    def test_probe_has_audio_raises_instead_of_returning_false(self, _mock) -> None:
        # The most expensive of the three: a silent False builds a video-only
        # filtergraph in compose/filters/transcode, so the master ships MUTE
        # and every gate downstream is green.
        with pytest.raises(RuntimeError, match="UNMEASURED"):
            FFmpegOps.probe_has_audio(Path("/f.mp4"))

    @patch.object(FFmpegOps, "probe_all", return_value=_EMPTY_BUT_PROBED)
    def test_a_genuinely_empty_file_that_probed_ok_still_returns_zero(self, _mock) -> None:
        """Control: the discriminator is ``probed``, NOT ``duration == 0``.

        Without this case the fix above would be indistinguishable from
        "raise whenever the numbers are zero", which would break every real
        0-second/audio-less file.
        """
        assert FFmpegOps.probe_duration(Path("/f.mp4")) == 0.0
        assert FFmpegOps.probe_dimensions(Path("/f.mp4")) == (0, 0)
        assert FFmpegOps.probe_has_audio(Path("/f.mp4")) is False


class TestChapterTimestampsCannotBeBuiltFromAFabricatedZero:
    """The corruption a silent 0.0 produced in the delivered artifact."""

    def test_unprobed_segment_no_longer_yields_shifted_negative_chapters(
        self, tmp_path: Path
    ) -> None:
        segments = []
        for name in ("a", "b"):
            seg = tmp_path / f"{name}.mp4"
            seg.write_bytes(b"x")
            segments.append(seg)
        parts_by_id = {
            "a": {"chapter": 1, "title": "One", "part_type": "title_card"},
            "b": {"chapter": 2, "title": "Two"},
        }
        mixin = _AssemblyMixin()

        # Segment "a" probes fine (60s); segment "b"'s probe times out.
        def _probe(path: Path) -> ProbeResult:
            return (
                ProbeResult(duration=60.0, width=1920, height=1080, has_audio=True)
                if (path.stem == "a")
                else _UNPROBED
            )

        with (
            patch.object(FFmpegOps, "probe_all", side_effect=_probe),
            pytest.raises(RuntimeError, match="UNMEASURED"),
        ):
            mixin._compute_chapter_timestamps(
                segments, parts_by_id, transition_type="crossfade", crossfade_dur=1.0
            )

    def test_all_segments_probed_still_produces_chapters(self, tmp_path: Path) -> None:
        """Control: the guard must not fire on a healthy compile."""
        seg = tmp_path / "a.mp4"
        seg.write_bytes(b"x")
        good = ProbeResult(duration=60.0, width=1920, height=1080, has_audio=True)
        with patch.object(FFmpegOps, "probe_all", return_value=good):
            chapters = _AssemblyMixin()._compute_chapter_timestamps(
                [seg], {"a": {"chapter": 1, "title": "One"}}
            )
        assert chapters == [
            {
                "chapter_number": 1,
                "title": "One",
                "subtitle": "",
                "start_seconds": 0.0,
                "duration_seconds": 60.0,
            }
        ]


class TestQualityValidatorNamesTheGap:
    def test_unprobed_output_is_reported_as_probe_readable_not_missing_video(
        self, tmp_path: Path
    ) -> None:
        video = tmp_path / "out.mp4"
        video.write_bytes(b"not really a video")
        with patch.object(FFmpegOps, "probe_all", return_value=_UNPROBED):
            report = QualityValidator().validate(video)
        assert report.valid is False
        checks = [i["check"] for i in report.issues]
        # The gap must be named as itself. It used to surface as
        # "video_stream_present", which points the operator at the encoder.
        assert "probe_readable" in checks
        assert "video_stream_present" not in checks
