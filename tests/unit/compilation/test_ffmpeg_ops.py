"""Unit tests for FFmpeg operations.

Rewritten to validate actual command arguments and behavior rather than
just checking return values that come straight from mocks.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.services.ffmpeg import FFmpegError, FFmpegOps, ProbeResult

# ---------------------------------------------------------------------------
# FFmpegError tests
# ---------------------------------------------------------------------------


class TestFFmpegError:
    def test_basic_error_has_sensible_defaults(self):
        err = FFmpegError("something failed")
        assert str(err) == "something failed"
        assert err.command == ""
        assert err.return_code == -1
        assert err.stderr == ""

    def test_error_preserves_all_details(self):
        err = FFmpegError(
            "ffmpeg failed",
            command="ffmpeg -y -i in.mp4 out.mp4",
            return_code=1,
            stderr="Error opening file",
        )
        assert err.command == "ffmpeg -y -i in.mp4 out.mp4"
        assert err.return_code == 1
        assert err.stderr == "Error opening file"

    def test_is_runtime_error_subclass(self):
        err = FFmpegError("test")
        assert isinstance(err, RuntimeError)


# ---------------------------------------------------------------------------
# FFmpegOps.run tests -- verify command construction and error handling
# ---------------------------------------------------------------------------


class TestFFmpegOpsRun:
    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_prepends_ffmpeg_y_to_args(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        FFmpegOps.run(["-i", "input.mp4", "output.mp4"])
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert args[1] == "-y"
        assert args[2:] == ["-i", "input.mp4", "output.mp4"]

    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_nonzero_exit_raises_with_return_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="File not found", stdout="")
        with pytest.raises(FFmpegError) as exc_info:
            FFmpegOps.run(["-i", "missing.mp4", "out.mp4"])
        assert exc_info.value.return_code == 1
        assert "File not found" in exc_info.value.stderr

    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_missing_ffmpeg_raises_descriptive_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")
        with pytest.raises(FFmpegError, match="ffmpeg not found on PATH"):
            FFmpegOps.run(["-i", "input.mp4", "output.mp4"])

    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_timeout_raises_ffmpeg_error_not_timeout_expired(self, mock_run):
        """A timed-out ffmpeg must surface the repo's own error, not TimeoutExpired.

        ``run`` is a paid-stage entry point: callers in ``filters.py`` /
        ``compose.py`` only catch ``FFmpegError``, so a raw
        ``subprocess.TimeoutExpired`` escapes un-normalized (the sibling
        ``probe.py`` already normalizes its timeout).
        """
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="ffmpeg",
            timeout=42,
            stderr="frame= 120 fps=2.0\n[libx264 @ 0x1] stuck on filtergraph\n",
        )
        # Anchored: substring checks ("timed out" in msg, "42" in msg) also pass
        # for a garbled message like "timed out 42 42", which is not a contract.
        with pytest.raises(FFmpegError, match=r"^ffmpeg timed out after 42s$") as exc_info:
            FFmpegOps.run(["-i", "input.mp4", "output.mp4"], timeout=42)
        assert exc_info.value.command == "ffmpeg -y -i input.mp4 output.mp4"
        # ffmpeg's partial output survives the normalization: without it the
        # operator gets "timed out" and no indication of WHERE it hung, while
        # the non-zero-exit path hands them an 800-char tail.
        assert "stuck on filtergraph" in exc_info.value.stderr
        # ...but the message itself stays clean: it is persisted to the job's
        # error column and shipped to the tenant webhook.
        assert "filtergraph" not in str(exc_info.value)

    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_custom_timeout_is_forwarded(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        FFmpegOps.run(["-i", "input.mp4", "output.mp4"], timeout=60)
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 60

    @patch("src.services.ffmpeg.base.subprocess.run")
    def test_default_timeout_used_when_not_specified(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        FFmpegOps.run(["-i", "input.mp4", "output.mp4"])
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == FFmpegOps._default_timeout


# ---------------------------------------------------------------------------
# FFmpegOps._probe tests
# ---------------------------------------------------------------------------


class TestFFmpegOpsProbe:
    @patch("src.services.ffmpeg.probe.subprocess.run")
    def test_probe_returns_stripped_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout='  {"format": {}}  ', stderr="")
        result = FFmpegOps._probe(["-show_format", "input.mp4"])
        assert result == '{"format": {}}'

    @patch("src.services.ffmpeg.probe.subprocess.run")
    def test_probe_ffprobe_not_found_raises(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(RuntimeError, match="ffprobe not found"):
            FFmpegOps._probe(["-show_format", "input.mp4"])

    @patch("src.services.ffmpeg.probe.subprocess.run")
    def test_probe_timeout_raises(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffprobe", timeout=10)
        with pytest.raises(RuntimeError, match="ffprobe timed out"):
            FFmpegOps._probe(["-show_format", "input.mp4"])

    @patch("src.services.ffmpeg.probe.subprocess.run")
    def test_probe_nonzero_exit_raises_with_stderr_snippet(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: could not find stream info"
        )
        with pytest.raises(RuntimeError, match="ffprobe failed"):
            FFmpegOps._probe(["-show_format", "input.mp4"])


# ---------------------------------------------------------------------------
# FFmpegOps.probe_all tests
# ---------------------------------------------------------------------------


class TestProbeAll:
    def setup_method(self):
        FFmpegOps._probe_cache.clear()

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_parses_video_stream_dimensions(self, mock_probe):
        data = {
            "format": {"duration": "30.5"},
            "streams": [
                {"codec_type": "video", "width": 1920, "height": 1080},
                {"codec_type": "audio"},
            ],
        }
        mock_probe.return_value = json.dumps(data)

        with patch("pathlib.Path.resolve", return_value=Path("/fake/video.mp4")):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_mtime=12345.0)
                result = FFmpegOps.probe_all(Path("/fake/video.mp4"))

        assert result.duration == 30.5
        assert result.width == 1920
        assert result.height == 1080
        assert result.has_audio is True

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_no_audio_stream_detected(self, mock_probe):
        data = {
            "format": {"duration": "15.0"},
            "streams": [{"codec_type": "video", "width": 1280, "height": 720}],
        }
        mock_probe.return_value = json.dumps(data)

        with patch("pathlib.Path.resolve", return_value=Path("/fake/no_audio.mp4")):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_mtime=12345.0)
                result = FFmpegOps.probe_all(Path("/fake/no_audio.mp4"))

        assert result.has_audio is False
        assert result.width == 1280
        assert result.height == 720

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_probe_failure_returns_zeroed_result(self, mock_probe):
        mock_probe.side_effect = RuntimeError("ffprobe failed")

        with patch("pathlib.Path.resolve", return_value=Path("/fake/bad.mp4")):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_mtime=12345.0)
                result = FFmpegOps.probe_all(Path("/fake/bad.mp4"))

        assert result.duration == 0.0
        assert result.width == 0
        assert result.height == 0
        assert result.has_audio is False

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_caching_avoids_repeated_probes(self, mock_probe):
        data = {
            "format": {"duration": "10.0"},
            "streams": [{"codec_type": "video", "width": 640, "height": 480}],
        }
        mock_probe.return_value = json.dumps(data)

        with patch("pathlib.Path.resolve", return_value=Path("/fake/cached.mp4")):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_mtime=99999.0)
                result1 = FFmpegOps.probe_all(Path("/fake/cached.mp4"))
                result2 = FFmpegOps.probe_all(Path("/fake/cached.mp4"))

        assert result1 == result2
        assert mock_probe.call_count == 1

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_invalid_duration_string_defaults_to_zero(self, mock_probe):
        data = {"format": {"duration": "not_a_number"}, "streams": []}
        mock_probe.return_value = json.dumps(data)

        with patch("pathlib.Path.resolve", return_value=Path("/fake/inv_dur.mp4")):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_mtime=11111.0)
                result = FFmpegOps.probe_all(Path("/fake/inv_dur.mp4"))

        assert result.duration == 0.0

    @patch("src.services.ffmpeg.ops.FFmpegOps._probe")
    def test_stat_oserror_still_produces_result(self, mock_probe):
        data = {"format": {"duration": "5.0"}, "streams": []}
        mock_probe.return_value = json.dumps(data)

        with patch("pathlib.Path.resolve", return_value=Path("/fake/nostat.mp4")):
            with patch("pathlib.Path.stat", side_effect=OSError("no such file")):
                result = FFmpegOps.probe_all(Path("/fake/nostat.mp4"))

        assert result.duration == 5.0


# ---------------------------------------------------------------------------
# FFmpegOps convenience probe methods
# ---------------------------------------------------------------------------


class TestProbeConvenienceMethods:
    @patch.object(FFmpegOps, "probe_all")
    def test_probe_duration_extracts_duration_field(self, mock_probe_all):
        mock_probe_all.return_value = ProbeResult(
            duration=42.5, width=1920, height=1080, has_audio=True
        )
        assert FFmpegOps.probe_duration(Path("/f.mp4")) == 42.5

    @patch.object(FFmpegOps, "probe_all")
    def test_probe_dimensions_returns_width_height_tuple(self, mock_probe_all):
        mock_probe_all.return_value = ProbeResult(
            duration=10.0, width=3840, height=2160, has_audio=False
        )
        w, h = FFmpegOps.probe_dimensions(Path("/f.mp4"))
        assert w == 3840
        assert h == 2160

    @patch.object(FFmpegOps, "probe_all")
    def test_probe_has_audio_returns_boolean(self, mock_probe_all):
        mock_probe_all.return_value = ProbeResult(
            duration=10.0, width=1920, height=1080, has_audio=True
        )
        assert FFmpegOps.probe_has_audio(Path("/f.mp4")) is True


# ---------------------------------------------------------------------------
# FFmpegOps.configure_from_settings
# ---------------------------------------------------------------------------


class TestConfigureFromSettings:
    @patch("src.services.ffmpeg.base.get_settings")
    def test_applies_settings_to_class_attrs(self, mock_settings):
        mock_settings.return_value = MagicMock(
            ffmpeg=MagicMock(timeout=600, preset="fast", ffprobe_timeout=20),
            compilation=MagicMock(max_concurrent_ffmpeg_subprocesses=4),
        )
        original_timeout = FFmpegOps._default_timeout
        original_preset = FFmpegOps._default_preset
        original_probe_timeout = FFmpegOps.FFPROBE_TIMEOUT
        original_max = FFmpegOps._max_concurrent_subprocesses
        original_sem = FFmpegOps._run_semaphore

        try:
            FFmpegOps.configure_from_settings()
            assert FFmpegOps._default_timeout == 600
            assert FFmpegOps._default_preset == "fast"
            assert FFmpegOps.FFPROBE_TIMEOUT == 20
            assert FFmpegOps._max_concurrent_subprocesses == 4
            assert FFmpegOps._run_semaphore is not None
        finally:
            FFmpegOps._default_timeout = original_timeout
            FFmpegOps._default_preset = original_preset
            FFmpegOps.FFPROBE_TIMEOUT = original_probe_timeout
            FFmpegOps._max_concurrent_subprocesses = original_max
            FFmpegOps._run_semaphore = original_sem


# ---------------------------------------------------------------------------
# FFmpegOps video operations -- verify correct command arguments
# ---------------------------------------------------------------------------


class TestFFmpegOpsVideoOperations:
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_convert_to_mp4_with_audio_uses_aac_codec(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            result = FFmpegOps.convert_to_mp4(Path("/in.webm"), Path("/out.mp4"))
        assert result == Path("/out.mp4")
        args = mock_run.call_args[0][0]
        # Verify codec arguments are present and correct
        assert "-c:v" in args
        cv_idx = args.index("-c:v")
        assert args[cv_idx + 1] == "libx264"
        assert "-c:a" in args
        ca_idx = args.index("-c:a")
        assert args[ca_idx + 1] == "aac"
        # Should NOT use anullsrc when audio is present
        assert "anullsrc" not in " ".join(args)

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_convert_to_mp4_without_audio_adds_silent_track(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.convert_to_mp4(Path("/in.webm"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        assert "anullsrc" in " ".join(args)
        assert "-shortest" in args

    @patch.object(FFmpegOps, "run")
    def test_trim_passes_correct_ss_and_duration(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.trim(Path("/in.mp4"), Path("/out.mp4"), 5.0, 15.0)
        args = mock_run.call_args[0][0]
        ss_idx = args.index("-ss")
        assert args[ss_idx + 1] == "5.0"
        t_idx = args.index("-t")
        assert args[t_idx + 1] == "10.0"  # duration = end - start

    def test_trim_invalid_range_raises(self):
        with pytest.raises(ValueError, match="end_s"):
            FFmpegOps.trim(Path("/in.mp4"), Path("/out.mp4"), 10.0, 5.0)

    def test_trim_equal_range_raises(self):
        with pytest.raises(ValueError, match="end_s"):
            FFmpegOps.trim(Path("/in.mp4"), Path("/out.mp4"), 5.0, 5.0)

    @patch.object(FFmpegOps, "run")
    def test_trim_reencode_seeks_caps_and_reencodes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.trim_reencode(Path("/in.webm"), Path("/out.mp4"), start_s=2.0, duration_s=6.5)
        args = mock_run.call_args[0][0]
        # input-seek before -i (frame-accurate with re-encode), duration cap,
        # video-only H.264 output.
        ss_idx = args.index("-ss")
        i_idx = args.index("-i")
        assert ss_idx < i_idx
        assert args[ss_idx + 1] == "2.0"
        assert args[args.index("-t") + 1] == "6.5"
        assert "-an" in args
        assert "libx264" in args

    @patch.object(FFmpegOps, "run")
    def test_trim_reencode_no_seek_when_start_zero(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.trim_reencode(Path("/in.webm"), Path("/out.mp4"), start_s=0.0, duration_s=4.0)
        args = mock_run.call_args[0][0]
        assert "-ss" not in args

    def test_trim_reencode_nonpositive_duration_raises(self):
        with pytest.raises(ValueError, match="duration_s"):
            FFmpegOps.trim_reencode(Path("/in.webm"), Path("/out.mp4"), start_s=1.0, duration_s=0.0)

    @patch.object(FFmpegOps, "run")
    def test_image_to_video_uses_loop_and_duration(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_video(Path("/img.png"), Path("/out.mp4"), 5.0)
        args = mock_run.call_args[0][0]
        assert "-loop" in args
        loop_idx = args.index("-loop")
        assert args[loop_idx + 1] == "1"
        assert "-t" in args
        t_idx = args.index("-t")
        assert args[t_idx + 1] == "5.0"
        assert "anullsrc" in " ".join(args)

    @patch.object(FFmpegOps, "run")
    def test_image_to_motion_video_pan_left_has_crop_filter(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_motion_video(
            Path("/img.png"), Path("/out.mp4"), 5.0, motion_preset="pan_left"
        )
        args = mock_run.call_args[0][0]
        vf_idx = args.index("-vf")
        vf_value = args[vf_idx + 1]
        assert "crop=" in vf_value
        assert "in_w-out_w" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_image_to_video_requests_crf_18(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_video(Path("/img.png"), Path("/out.mp4"), 5.0)
        args = mock_run.call_args[0][0]
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "18"

    @patch.object(FFmpegOps, "run")
    def test_image_to_video_scale_requests_lanczos(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_video(Path("/img.png"), Path("/out.mp4"), 5.0)
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert "flags=lanczos+accurate_rnd+full_chroma_int" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_image_to_motion_video_requests_crf_18(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_motion_video(
            Path("/img.png"), Path("/out.mp4"), 5.0, motion_preset="zoom_in"
        )
        args = mock_run.call_args[0][0]
        assert "-crf" in args
        assert args[args.index("-crf") + 1] == "18"

    @pytest.mark.parametrize(
        "preset", ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]
    )
    @patch.object(FFmpegOps, "run")
    def test_image_to_motion_video_scale_requests_lanczos(self, mock_run, preset):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.image_to_motion_video(
            Path("/img.png"), Path("/out.mp4"), 5.0, motion_preset=preset
        )
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert "flags=lanczos+accurate_rnd+full_chroma_int" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_concat_uses_concat_demuxer_with_crf(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.concat(Path("/list.txt"), Path("/out.mp4"), crf=22)
        args = mock_run.call_args[0][0]
        f_idx = args.index("-f")
        assert args[f_idx + 1] == "concat"
        crf_idx = args.index("-crf")
        assert args[crf_idx + 1] == "22"

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_crossfade_both_audio_uses_amix_and_adelay(
        self, mock_dur, mock_dims, mock_audio, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.crossfade(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"), fade_duration=0.5)
        args = mock_run.call_args[0][0]
        fc_idx = args.index("-filter_complex")
        fc_value = args[fc_idx + 1]
        assert "xfade" in fc_value
        assert "amix" in fc_value
        assert "adelay" in fc_value
        assert "apad" in fc_value
        # Verify acrossfade is NOT used (it's replaced by amix approach)
        assert "acrossfade" not in fc_value

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_crossfade_no_audio_omits_audio_mapping(
        self, mock_dur, mock_dims, mock_audio, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.crossfade(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        # No [a] audio mapping should appear
        assert (
            "[a]"
            not in " ".join(args)
            .replace("[v0]", "")
            .replace("[v1]", "")
            .replace("[v]", "")
            .split("filter_complex")[0]
        )

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_hstack_splits_target_width_and_uses_hstack_filter(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/left.mp4"), Path("/right.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        fc_idx = args.index("-filter_complex")
        fc_value = args[fc_idx + 1]
        assert "hstack=inputs=2" in fc_value
        assert "scale=960:1080" in fc_value
        assert "pad=960:1080" in fc_value
        map_indices = [i for i, a in enumerate(args) if a == "-map"]
        mapped = [args[i + 1] for i in map_indices]
        assert "[v]" in mapped
        assert "0:a" in mapped

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_hstack_audio_from_right_maps_right_audio(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/left.mp4"), Path("/right.mp4"), Path("/out.mp4"), audio_from=1)
        args = mock_run.call_args[0][0]
        map_indices = [i for i, a in enumerate(args) if a == "-map"]
        mapped = [args[i + 1] for i in map_indices]
        assert "1:a" in mapped
        assert "0:a" not in mapped

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_hstack_no_audio_omits_audio_mapping(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/left.mp4"), Path("/right.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        map_indices = [i for i, a in enumerate(args) if a == "-map"]
        mapped = [args[i + 1] for i in map_indices]
        assert mapped == ["[v]"]

    def test_hstack_rejects_odd_target_width(self):
        with pytest.raises(ValueError, match="target_width"):
            FFmpegOps.hstack(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"), target_width=1921)

    def test_hstack_rejects_invalid_audio_from(self):
        with pytest.raises(ValueError, match="audio_from"):
            FFmpegOps.hstack(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"), audio_from=2)

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_fade_to_black_uses_concat_with_fade_filters(
        self, mock_dur, mock_dims, mock_audio, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.fade_to_black(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        fc_idx = args.index("-filter_complex")
        fc_value = args[fc_idx + 1]
        assert "fade=t=out" in fc_value
        assert "fade=t=in" in fc_value
        assert "concat=n=2" in fc_value

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_normalize_stream_with_audio_includes_aac_encoding(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"), width=1280, height=720)
        args = mock_run.call_args[0][0]
        vf_idx = args.index("-vf")
        vf_value = args[vf_idx + 1]
        assert "scale=1280:720" in vf_value
        assert "pad=1280:720" in vf_value
        ca_idx = args.index("-c:a")
        assert args[ca_idx + 1] == "aac"
        assert "anullsrc" not in " ".join(args)

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_normalize_stream_without_audio_adds_silent_track(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        assert "anullsrc" in " ".join(args)

    @patch.object(FFmpegOps, "run")
    def test_extract_thumbnail_uses_first_frame(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.extract_thumbnail(Path("/in.mp4"), Path("/thumb.jpg"))
        args = mock_run.call_args[0][0]
        # Should extract a single frame
        assert "-vframes" in args or "-frames:v" in args

    @patch.object(FFmpegOps, "run")
    def test_strip_audio_uses_an_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.strip_audio(Path("/in.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        assert "-an" in args

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_speed_adjust_with_audio_uses_atempo_and_setpts(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.speed_adjust(Path("/in.mp4"), Path("/out.mp4"), 2.0)
        args = mock_run.call_args[0][0]
        fc_idx = args.index("-filter_complex")
        fc_value = args[fc_idx + 1]
        assert "setpts=" in fc_value
        assert "atempo=" in fc_value
        assert "0.5*PTS" in fc_value  # 1/2.0 = 0.5

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_speed_adjust_without_audio_strips_audio(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.speed_adjust(Path("/in.mp4"), Path("/out.mp4"), 2.0)
        args = mock_run.call_args[0][0]
        assert "-an" in args


# ---------------------------------------------------------------------------
# Downscale quality: real resizes must request lanczos, matching the
# slide/animation path (animation_recorder.py:423 uses flags=lanczos).
# Default swscale (bilinear) blurs captured screen pixels on downscale.
# ---------------------------------------------------------------------------


class TestScaleFiltersUseLanczos:
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_crossfade_scale_requests_lanczos(self, mock_dur, mock_dims, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.crossfade(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"))
        fc_value = mock_run.call_args[0][0][mock_run.call_args[0][0].index("-filter_complex") + 1]
        assert fc_value.count("flags=lanczos") == 2

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_hstack_scale_requests_lanczos(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/left.mp4"), Path("/right.mp4"), Path("/out.mp4"))
        fc_value = mock_run.call_args[0][0][mock_run.call_args[0][0].index("-filter_complex") + 1]
        assert fc_value.count("flags=lanczos") == 2

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_fade_to_black_scale_requests_lanczos(self, mock_dur, mock_dims, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.fade_to_black(Path("/a.mp4"), Path("/b.mp4"), Path("/out.mp4"))
        fc_value = mock_run.call_args[0][0][mock_run.call_args[0][0].index("-filter_complex") + 1]
        assert fc_value.count("flags=lanczos") == 2

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_normalize_stream_scale_requests_lanczos(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert "flags=lanczos" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_extract_thumbnail_scale_requests_lanczos(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.extract_thumbnail(Path("/in.mp4"), Path("/thumb.jpg"))
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert "flags=lanczos" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_generate_preview_scale_requests_lanczos(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.generate_preview(Path("/in.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert "flags=lanczos" in vf_value

    @patch.object(FFmpegOps, "run")
    def test_frames_to_mp4_even_dims_scale_asks_for_no_resampler(self, mock_run):
        """The even-dims filter rounds each dimension down to the nearest even
        number -- at most a 1px change, required by yuv420p. It is NOT a resize,
        so a resampler choice cannot improve it, and on the alpha twin
        full_chroma_int would touch chroma on a path carrying transparency.
        Lanczos belongs on the real resizes, not here."""
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.frames_to_mp4(Path("/frames"), Path("/out.mp4"), fps=30)
        args = mock_run.call_args[0][0]
        vf_value = args[args.index("-vf") + 1]
        assert vf_value == "scale=trunc(iw/2)*2:trunc(ih/2)*2"


# ---------------------------------------------------------------------------
# Atempo chain edge cases
# ---------------------------------------------------------------------------


class TestAtempoChain:
    def test_very_slow_produces_chain_of_half_steps(self):
        chain = FFmpegOps._build_atempo_chain(0.25)
        parts = chain.split(",")
        # 0.25 = 0.5 * 0.5, so needs 2 steps
        assert len(parts) == 2
        for p in parts:
            assert p.startswith("atempo=")

    def test_very_fast_produces_reasonable_chain(self):
        chain = FFmpegOps._build_atempo_chain(4.0)
        parts = chain.split(",")
        product = 1.0
        for p in parts:
            product *= float(p.split("=")[1])
        assert abs(product - 4.0) < 0.01

    def test_factor_below_0_25_clamped(self):
        result_clamped = FFmpegOps._build_atempo_chain(0.1)
        result_min = FFmpegOps._build_atempo_chain(0.25)
        assert result_clamped == result_min


# ---------------------------------------------------------------------------
# Constant-frame-rate (CFR) gating -- aspect A13 / contract C10
# ---------------------------------------------------------------------------


def _assert_cfr_argv(args: list[str], fps: int) -> None:
    """Assert the argv contains the -fps_mode cfr -r {fps} tuple in order."""
    assert "-fps_mode" in args
    fm_idx = args.index("-fps_mode")
    assert args[fm_idx + 1] == "cfr"
    r_idx = args.index("-r")
    assert args[r_idx + 1] == str(fps)


class TestCfrGating:
    """cfr=True emits -fps_mode cfr -r {fps} for all 5 re-encoding ops;
    default (cfr=False, fps=30) is byte-identical (no -fps_mode, filter fps=30)."""

    # --- normalize_stream ---------------------------------------------------
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_normalize_stream_cfr_emits_fps_mode(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"), fps=48, cfr=True)
        args = mock_run.call_args[0][0]
        _assert_cfr_argv(args, 48)
        vf = args[args.index("-vf") + 1]
        assert "fps=48" in vf

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=False)
    def test_normalize_stream_cfr_no_audio_branch(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"), fps=48, cfr=True)
        args = mock_run.call_args[0][0]
        _assert_cfr_argv(args, 48)
        assert "fps=48" in args[args.index("-vf") + 1]

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_normalize_stream_default_is_byte_identical(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.normalize_stream(Path("/in.mp4"), Path("/out.mp4"))
        args = mock_run.call_args[0][0]
        assert "-fps_mode" not in args
        assert "fps=30" in args[args.index("-vf") + 1]

    # --- concat -------------------------------------------------------------
    @patch.object(FFmpegOps, "run")
    def test_concat_cfr_emits_fps_mode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.concat(Path("/list.txt"), Path("/out.mp4"), fps=48, cfr=True)
        _assert_cfr_argv(mock_run.call_args[0][0], 48)

    @patch.object(FFmpegOps, "run")
    def test_concat_default_is_byte_identical(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.concat(Path("/list.txt"), Path("/out.mp4"))
        assert "-fps_mode" not in mock_run.call_args[0][0]

    # --- crossfade ----------------------------------------------------------
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_crossfade_cfr_emits_fps_mode(self, mock_dur, mock_dims, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.crossfade(Path("/a.mp4"), Path("/b.mp4"), Path("/o.mp4"), fps=48, cfr=True)
        args = mock_run.call_args[0][0]
        _assert_cfr_argv(args, 48)
        assert "fps=48" in args[args.index("-filter_complex") + 1]

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_crossfade_default_is_byte_identical(self, mock_dur, mock_dims, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.crossfade(Path("/a.mp4"), Path("/b.mp4"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        assert "-fps_mode" not in args
        assert "fps=30" in args[args.index("-filter_complex") + 1]

    # --- hstack -------------------------------------------------------------
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_hstack_cfr_emits_fps_mode(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/l.mp4"), Path("/r.mp4"), Path("/o.mp4"), fps=48, cfr=True)
        args = mock_run.call_args[0][0]
        _assert_cfr_argv(args, 48)
        assert "fps=48" in args[args.index("-filter_complex") + 1]

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_hstack_default_is_byte_identical(self, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.hstack(Path("/l.mp4"), Path("/r.mp4"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        assert "-fps_mode" not in args
        assert "fps=30" in args[args.index("-filter_complex") + 1]

    # --- fade_to_black ------------------------------------------------------
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_fade_to_black_cfr_emits_fps_mode(self, mock_dur, mock_dims, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.fade_to_black(Path("/a.mp4"), Path("/b.mp4"), Path("/o.mp4"), fps=48, cfr=True)
        args = mock_run.call_args[0][0]
        _assert_cfr_argv(args, 48)
        assert "fps=48" in args[args.index("-filter_complex") + 1]

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_dimensions", return_value=(1920, 1080))
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    def test_fade_to_black_default_is_byte_identical(
        self, mock_dur, mock_dims, mock_audio, mock_run
    ):
        mock_run.return_value = MagicMock(returncode=0)
        FFmpegOps.fade_to_black(Path("/a.mp4"), Path("/b.mp4"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        assert "-fps_mode" not in args
        assert "fps=30" in args[args.index("-filter_complex") + 1]

    # --- copy paths stay untouched -----------------------------------------
    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_duration", return_value=10.0)
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    def test_mix_voiceover_copy_path_has_no_fps_mode(self, mock_audio, mock_dur, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.mix_voiceover(Path("/v.mp4"), Path("/vo.mp3"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        assert "-fps_mode" not in args
        assert args[args.index("-c:v") + 1] == "copy"


class TestMixVoiceoverFreeze:
    """An over-long voiceover must freeze the last video frame in EVERY branch.

    Regression: only the no-audio branch used tpad; the music + has-audio
    branches padded audio to the (shorter) video and `amix=duration=longest`
    over `-c:v copy`, so the narration tail played over a dead/absent video
    tail. The fix mirrors the no-audio freeze into all branches.
    """

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_duration", side_effect=[10.0, 25.0])
    def test_music_branch_freezes_video_when_vo_longer(self, mock_dur, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=True):
            FFmpegOps.mix_voiceover(
                Path("/v.mp4"), Path("/vo.mp3"), Path("/o.mp4"), music_path=Path("/m.mp3")
            )
        args = mock_run.call_args[0][0]
        fc = args[args.index("-filter_complex") + 1]
        # extra = 25.0 - 10.0 + 0.5 = 15.5 → video held to cover the mixed audio
        assert "[0:v]tpad=stop_mode=clone:stop_duration=15.500[v];" in fc
        assert args[args.index("-map") + 1] == "[v]"
        assert "-c:v" not in args  # tpad forces a re-encode; copy fast-path dropped

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_duration", side_effect=[10.0, 25.0])
    def test_has_audio_branch_freezes_video_when_vo_longer(self, mock_dur, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.mix_voiceover(Path("/v.mp4"), Path("/vo.mp3"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        fc = args[args.index("-filter_complex") + 1]
        assert "[0:v]tpad=stop_mode=clone:stop_duration=15.500[v];" in fc
        assert args[args.index("-map") + 1] == "[v]"
        assert "-c:v" not in args

    @patch.object(FFmpegOps, "run")
    @patch.object(FFmpegOps, "probe_has_audio", return_value=True)
    @patch.object(FFmpegOps, "probe_duration", side_effect=[10.0, 8.0])
    def test_has_audio_branch_keeps_copy_when_vo_not_longer(self, mock_dur, mock_audio, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        with patch("pathlib.Path.mkdir"):
            FFmpegOps.mix_voiceover(Path("/v.mp4"), Path("/vo.mp3"), Path("/o.mp4"))
        args = mock_run.call_args[0][0]
        fc = args[args.index("-filter_complex") + 1]
        assert "tpad" not in fc
        assert args[args.index("-c:v") + 1] == "copy"


class TestRemoveDeadTimeThreadsCfr:
    """remove_dead_time forwards cfr/fps into its final concat (video_editor:152)."""

    @patch.object(FFmpegOps, "concat")
    @patch.object(FFmpegOps, "speed_adjust")
    @patch.object(FFmpegOps, "trim")
    @patch("src.services.compilation.video_editor.subprocess.run")
    @patch.object(FFmpegOps, "probe_duration", return_value=30.0)
    def test_concat_receives_cfr_and_fps(
        self, mock_dur, mock_subprocess, mock_trim, mock_speed, mock_concat
    ):
        import tempfile

        from src.services.compilation.video_editor import VideoEditor

        with tempfile.TemporaryDirectory() as tmpdir:
            editor = VideoEditor(output_dir=Path(tmpdir))
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"fake video")
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stderr="[freezedetect] freeze_start: 5.0\n[freezedetect] freeze_end: 10.0\n",
                stdout="",
            )

            def _mk(*a, **k):
                out = a[1]
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"x")
                return out

            mock_trim.side_effect = _mk
            mock_speed.side_effect = _mk
            mock_concat.side_effect = _mk

            editor.remove_dead_time(video_path, fps=48, cfr=True)
            mock_concat.assert_called_once()
            assert mock_concat.call_args.kwargs.get("cfr") is True
            assert mock_concat.call_args.kwargs.get("fps") == 48
