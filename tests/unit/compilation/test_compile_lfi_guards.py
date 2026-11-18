"""LFI proof for the tenant-supplied local-path asset references on the compile path.

The SSRF work (PRs #106/#109) closed the http(s) half of these overloaded fields
and explicitly left the local half alone. That half is its own primitive: the
compiler reads whatever path the tenant names off the pod and composites it into
a video the tenant can then **download**, so ``slide_image_url='file:///etc/
passwd'`` is an arbitrary file read with an exfiltration channel attached.

Why the guard is CONFINEMENT and not denial
-------------------------------------------
A bare local path is a legitimate production value on
``recordings[].video_path``. ``GET /api/record/jobs/{id}/result`` publishes
``{STORAGE_LOCAL_PATH}/{job_id}/{part_id}.mp4`` (this service runs the ``local``
storage backend in prod with ``/app/data`` on a PVC shared by both replicas), and
re-submitting that path on ``POST /api/compile`` is the documented record →
compile round trip. ``test_the_record_to_compile_round_trip_still_works`` is the
regression control for exactly that: a guard that breaks it is the wrong guard.

So the boundary is the storage root. Inside it — files this service wrote
itself — is allowed; everything else is refused. ``compilation.
allow_local_file_inputs`` (default False **in code**) opens the gate for dev and
for fixtures that live in a tmpdir.

No test here reads a real sensitive file: the "secret" is a fixture written into
the test's own tmpdir, and the assertion is that the guard refuses it *because it
sits outside the configured root*.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.api.schemas.compilation import CompileRequest, FindingInput, RecordingInput, ShotListInput
from src.services.compilation._compiler_context import _JobContext
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from src.utils.url_guard import resolve_local_asset_path
from tests.conftest import TENANT_U1

_SETTINGS = "src.config.get_settings"

# The seven tenant-controlled asset references on a finding. Every one is copied
# into a plain shot-list part dict by ``FindingsVideoService.build_shot_list``
# and reaches ``_materialize_image_ref`` from there.
FINDING_ASSET_FIELDS = (
    "evidence_path",
    "image_url",
    "generated_image_url",
    "comparison_frame_url",
    "comparison_image_url",
    "slide_image_url",
    "slide_preview_url",
)


def _settings(*, root: Path, allow_local: bool = False) -> MagicMock:
    s = MagicMock()
    s.compilation.allow_local_file_inputs = allow_local
    s.storage.local_path = str(root)
    return s


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """A stand-in for ``/app/data`` — the only tree local refs may name."""
    root = tmp_path / "app-data"
    root.mkdir()
    return root


@pytest.fixture
def outside_file(tmp_path: Path) -> Path:
    """A file OUTSIDE the storage root, standing in for ``/etc/passwd``."""
    secret = tmp_path / "outside" / "passwd"
    secret.parent.mkdir()
    secret.write_text("root:x:0:0:sensitive")
    return secret


class TestResolveLocalAssetPath:
    """The guard itself."""

    def test_a_path_inside_the_storage_root_is_accepted(self, storage_root: Path) -> None:
        asset = storage_root / "job-1" / "part-1.mp4"
        asset.parent.mkdir()
        asset.write_bytes(b"video")
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path(str(asset)) == asset.resolve()

    def test_a_path_outside_the_storage_root_is_refused(
        self, storage_root: Path, outside_file: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path(str(outside_file)) is None

    def test_a_file_uri_outside_the_root_is_refused(
        self, storage_root: Path, outside_file: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path(f"file://{outside_file}") is None

    def test_dot_dot_traversal_out_of_the_root_is_refused(self, storage_root: Path) -> None:
        """``/app/data/../etc/passwd`` starts with the root as a STRING."""
        escaped = f"{storage_root}/../outside/passwd"
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path(escaped) is None

    def test_a_symlink_planted_inside_the_root_cannot_escape_it(
        self, storage_root: Path, outside_file: Path
    ) -> None:
        """The compile work dir is writable, so a symlink is a realistic bypass.

        A prefix comparison done before ``resolve()`` would accept this.
        """
        link = storage_root / "innocent.mp4"
        link.symlink_to(outside_file)
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path(str(link)) is None

    def test_a_relative_ref_is_resolved_against_the_root_not_the_cwd(
        self, storage_root: Path
    ) -> None:
        """The minio backend's reference shape is a bare ``{job_id}/{name}`` key."""
        asset = storage_root / "job-1" / "part-1.mp4"
        asset.parent.mkdir()
        asset.write_bytes(b"video")
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path("job-1/part-1.mp4") == asset.resolve()

    def test_the_gate_opens_for_dev_and_test_fixtures(
        self, storage_root: Path, outside_file: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)):
            assert resolve_local_asset_path(str(outside_file)) == outside_file

    def test_an_unreadable_storage_config_fails_closed(self, outside_file: Path) -> None:
        """A broken storage config must refuse local refs, never wave them through."""
        broken = MagicMock()
        broken.compilation.allow_local_file_inputs = False
        type(broken.storage).local_path = property(
            lambda _self: (_ for _ in ()).throw(RuntimeError("no storage config"))
        )
        with patch(_SETTINGS, return_value=broken):
            assert resolve_local_asset_path(str(outside_file)) is None

    def test_an_empty_ref_is_refused(self, storage_root: Path) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert resolve_local_asset_path("file://") is None


# --- Sink layer --------------------------------------------------------------


def _ctx(tmp_path: Path) -> _JobContext:
    ctx = _JobContext(job_id="job-1", tenant_id=TENANT_U1, job=MagicMock(), settings=MagicMock())
    ctx.title_cards_dir = tmp_path / "title_cards"
    ctx.edited_dir = tmp_path / "edited"
    ctx.title_cards_dir.mkdir(parents=True, exist_ok=True)
    ctx.edited_dir.mkdir(parents=True, exist_ok=True)
    return ctx


class _Builder(_SegmentBuildersMixin):
    def __init__(self) -> None:
        self._storage = MagicMock()
        self._storage.retrieve_uri_sync = MagicMock(return_value=None)
        self._storage.retrieve_uri = AsyncMock(return_value=None)


class TestImageRefLfiSink:
    """``_materialize_image_ref`` — the ``file://`` / bare-path branch."""

    @pytest.mark.parametrize("prefix", ["", "file://"])
    def test_a_file_outside_the_root_is_never_read(
        self, prefix: str, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            result = _Builder()._materialize_image_ref(
                _ctx(tmp_path), "p1", f"{prefix}{outside_file}"
            )
        assert result is None

    def test_the_refusal_is_the_guard_and_not_a_missing_file(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """Non-vacuity control: the same ref IS returned once the gate opens.

        Without this, ``result is None`` above would also hold if the fixture
        simply did not exist, and the assertion would prove nothing.
        """
        with patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", str(outside_file))
        assert result == outside_file

    def test_an_asset_inside_the_root_is_still_served(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        asset = storage_root / "job-9" / "slide.png"
        asset.parent.mkdir()
        asset.write_bytes(b"\x89PNG")
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            result = _Builder()._materialize_image_ref(_ctx(tmp_path), "p1", str(asset))
        assert result == asset.resolve()


class TestRecordingVideoPathLfiSink:
    """``_resolve_recording_video_path`` — the bare-path branch."""

    async def test_the_record_to_compile_round_trip_still_works(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """LOAD-BEARING regression control.

        ``{STORAGE_LOCAL_PATH}/{job_id}/{part_id}.mp4`` is what
        ``GET /api/record/jobs/{id}/result`` hands back, and re-submitting it on
        ``recordings[].video_path`` is the documented round trip. A guard that
        breaks this is the wrong guard, so this must pass with the gate CLOSED.
        """
        recorded = storage_root / "rec-job-7" / "part-3.mp4"
        recorded.parent.mkdir()
        recorded.write_bytes(b"video")
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", str(recorded)
            )
        assert result == recorded.resolve()

    @pytest.mark.parametrize("prefix", ["", "file://"])
    async def test_a_path_outside_the_root_is_refused(
        self, prefix: str, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", f"{prefix}{outside_file}"
            )
        assert result is None

    async def test_the_refusal_is_the_guard_and_not_a_missing_file(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """Non-vacuity control for the case above."""
        with patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)):
            result = await _Builder()._resolve_recording_video_path(
                _ctx(tmp_path), "p1", str(outside_file)
            )
        assert result == outside_file

    async def test_a_blocked_path_skips_the_part_rather_than_failing_the_job(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """Fail-soft contract: one hostile part must not abort a compilation."""
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            # Must return, not raise.
            assert (
                await _Builder()._resolve_recording_video_path(
                    _ctx(tmp_path), "p1", str(outside_file)
                )
                is None
            )


class TestTitleCardBackgroundImageLfiSink:
    """``background_image`` reaches Pillow through a different branch."""

    def test_a_background_outside_the_root_is_not_handed_to_pillow(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        ctx = _ctx(tmp_path)
        ctx.title_gen = MagicMock()
        ctx.target_w, ctx.target_h = 1920, 1080
        ctx.narration_audio_dir = None
        builder = _Builder()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch.object(
                _SegmentBuildersMixin, "_segment_with_optional_narration", lambda *a: a[3]
            ),
        ):
            builder._process_title_card(
                ctx, "p1", {"title": "t", "background_image": str(outside_file)}
            )

        assert ctx.title_gen.generate_video.call_args.kwargs["background_image"] is None


# --- Admission layer ---------------------------------------------------------


class TestFileUriIsRefusedAtAdmission:
    """``file://`` never has a legitimate producer here, so it is a 422.

    Bare paths deliberately still pass the schema — only the sink knows the
    storage root that decides them — which is why
    ``test_tenant_url_ssrf_schemas.py::test_local_and_s3_paths_pass_through_untouched``
    keeps passing unchanged.
    """

    def _shot_list(self) -> ShotListInput:
        return ShotListInput(parts=[])

    def test_recording_video_path_rejects_a_file_uri(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            RecordingInput(part_id="r1", video_path="file:///etc/passwd")

    def test_narration_audio_dir_rejects_a_file_uri(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            CompileRequest(
                shot_list=self._shot_list(),
                recordings=[],
                narration_audio_dir="file:///var/run/secrets",
            )

    def test_a_bare_path_still_passes_admission(self) -> None:
        """Regression control for the round trip — see the class docstring."""
        assert (
            RecordingInput(part_id="r1", video_path="/app/data/job-1/p1.mp4").video_path
            == "/app/data/job-1/p1.mp4"
        )


class TestFindingInputAssetRefsAreGuarded:
    """GAP 3: ``FindingsVideoService`` copies these into a plain dict.

    That hand-off means ``ShotListPartInput``'s validators never see them, so
    ``POST /api/compile/findings-video`` had no admission layer at all — while
    ``POST /api/compile`` did.
    """

    @pytest.mark.parametrize("field", FINDING_ASSET_FIELDS)
    def test_file_uri_is_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            FindingInput.model_validate({"title": "t", field: "file:///etc/passwd"})

    @pytest.mark.parametrize("field", FINDING_ASSET_FIELDS)
    def test_internal_http_target_is_rejected(self, field: str) -> None:
        with pytest.raises(
            ValidationError, match="must not target an internal or reserved network address"
        ):
            FindingInput.model_validate({"title": "t", field: "http://100.64.0.5:5432/x"})

    @pytest.mark.parametrize("field", FINDING_ASSET_FIELDS)
    def test_legitimate_refs_still_pass(self, field: str) -> None:
        """Controls: the shapes real findings actually carry must keep working.

        The http case uses a public IP literal so ``getaddrinfo`` stays local —
        a hostname would make the assertion depend on the runner's resolver, and
        the guard fails CLOSED on a DNS failure.
        """
        for value in ("https://93.184.216.34/evidence.png", "s3://bucket/f.png", "/app/data/f.png"):
            model = FindingInput.model_validate({"title": "t", field: value})
            assert getattr(model, field) == value
