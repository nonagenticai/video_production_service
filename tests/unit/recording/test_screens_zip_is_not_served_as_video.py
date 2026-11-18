"""A screens archive must not be served with a video ``Content-Type``.

The #157/#160 shape, left unfixed in the SERVING path. Two of the five producers
(``create_screens_job`` -> ``UrlRecorder.capture_screens`` /
``capture_screen_clips``) store a ZIP of per-screen stills/clips under the
``video_path`` key, and ``store_video`` keeps the ``.zip`` suffix.
``GET /jobs/{id}/video`` -- which ``schemas/recording.py`` documents as the
screens download path -- mapped that suffix through a video-only table
(``{"mp4", "webm", "gif"}``) with a ``"video/webm"`` default, so every screens
download went out as ``Content-Type: video/webm``.

#160 fixed the COMPLETION path by having the producer declare ``artifact_kind``
and the consumer dispatch on it. The knowledge then died there, because the kind
was never PERSISTED. It is now, and the route dispatches on the same declared
fact rather than sniffing a suffix.

The default stays the strict/video case in both directions: a producer that
declares nothing, and every row written before the column existed, behaves
exactly as before.
"""

from __future__ import annotations

from src.services.recording.artifacts import (
    ARTIFACT_KIND_SCREENS_ZIP,
    ARTIFACT_KIND_VIDEO,
    artifact_media_type,
)


class TestArtifactMediaType:
    def test_a_screens_zip_is_served_as_a_zip(self):
        """The defect, directly. The suffix is `.zip` and the kind says so."""
        assert artifact_media_type(ARTIFACT_KIND_SCREENS_ZIP, "zip") == "application/zip"

    def test_the_kind_wins_over_the_suffix(self):
        """Dispatch is on the DECLARED kind, never on what the file looks like.

        Even a screens archive stored under a misleading suffix must not be
        announced as video -- the producer's declaration is the fact.
        """
        assert artifact_media_type(ARTIFACT_KIND_SCREENS_ZIP, "webm") == "application/zip"
        assert artifact_media_type(ARTIFACT_KIND_SCREENS_ZIP, None) == "application/zip"

    def test_video_kinds_are_unchanged(self):
        assert artifact_media_type(ARTIFACT_KIND_VIDEO, "mp4") == "video/mp4"
        assert artifact_media_type(ARTIFACT_KIND_VIDEO, "webm") == "video/webm"
        assert artifact_media_type(ARTIFACT_KIND_VIDEO, "gif") == "image/gif"

    def test_an_undeclared_row_keeps_the_strict_video_default(self):
        """#160's principle: declaring nothing must change nothing.

        NULL is what every row written before `artifact_kind` existed carries.
        """
        assert artifact_media_type(None, "mp4") == "video/mp4"
        assert artifact_media_type(None, "webm") == "video/webm"
        assert artifact_media_type(None, "weird") == "video/webm"
        assert artifact_media_type(None, None) == "video/webm"


class TestArtifactKindIsPersisted:
    """The root cause. Without the write, the route has nothing to dispatch on."""

    def test_the_column_exists_on_the_model(self):
        from src.database.models import RecordingJob

        assert hasattr(RecordingJob, "artifact_kind")

    def test_mark_completed_accepts_and_writes_it(self):
        import inspect

        from src.services.recording.job_manager import RecordingJobManager

        sig = inspect.signature(RecordingJobManager.mark_completed)
        assert "artifact_kind" in sig.parameters, (
            "mark_completed cannot persist the kind, so the serving path is still blind"
        )

    def test_the_completion_path_passes_the_declared_kind(self):
        """A column nobody writes is the same defect one step later."""
        from pathlib import Path

        src = Path("src/services/recording/recorder_service.py").read_text()
        assert "artifact_kind=artifact_kind_of(result)" in src, (
            "the recorder never persists the kind it just verified against"
        )

    def test_the_route_dispatches_on_the_kind_not_the_suffix_table(self):
        from pathlib import Path

        src = Path("src/api/routes/recording.py").read_text()
        assert "artifact_media_type(" in src
        assert '"webm": "video/webm"' not in src, (
            "the video-only suffix table is still in the serving path"
        )
