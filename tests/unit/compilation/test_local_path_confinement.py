"""Confinement proof for the tenant-supplied on-disk fields PR #111 left open.

#111 closed ``image_url`` / ``slide_image_url`` / ``background_image`` /
``video_path`` and the findings-video asset refs by confining them to the
storage root. It deliberately stopped there. This module covers the rest, and
the two primitives that remained:

**F1 — arbitrary file READ.** ``subtitles.file_path``, ``watermark.image_path``,
``clips_dir``, ``parts[].logo_path``, ``theme.logo_path`` /
``theme.background_image`` and the local branch of ``narration_audio_dir`` all
reached ``Path()`` with no validator at any layer. The sharpest of them is
``subtitles`` with ``mode='burn'``: ffmpeg renders the file's **text** into the
frames, so the file's contents come back to the caller as pixels in a video they
download. ``TestSubtitleFilePathIsConfined.test_the_exploit_shape_is_what_the_
guard_stops`` reproduces that shape and then shows it blocked.

**F2 — arbitrary file WRITE (higher severity).** ``part_id`` / ``id`` carried no
constraint, and they are f-string-joined onto directories the compiler owns:
``{title_cards_dir}/{part_id}_asset{ext}`` is a ``write_bytes`` of
attacker-supplied ``data:`` payload bytes, and ``{normalized_dir}/{part_id}
.mp4`` / ``{edited_dir}/{part_id}_trim.mp4`` are ffmpeg outputs. An id of
``../../../../app/x`` therefore both escapes the confinement applied to those
directories and writes attacker-chosen bytes to an attacker-chosen path.
``TestPartIdIsAFilenameNotAPath`` reproduces the write and then shows admission
refusing the id.

Why confinement and not denial
------------------------------
Unchanged from #111 and load-bearing: ``GET /api/record/jobs/{id}/result``
publishes ``{STORAGE_LOCAL_PATH}/{job_id}/{part_id}.mp4``, prod runs
``STORAGE_BACKEND=local`` with ``/app/data`` on a RWX PVC across both replicas,
and re-submitting a published path is the documented record → compile round
trip. Every class below therefore carries an **accept** case; a guard that fails
those is the wrong guard and would break prod rendering.

Per-field confinement root, and why
-----------------------------------
* ``watermark.image_path``, ``subtitles.file_path``, ``clips_dir``,
  ``narration_audio_dir``, ``parts[].logo_path``, ``theme.logo_path`` /
  ``theme.background_image`` → **storage root**. Every one of them names a file
  this service itself produced or the tenant uploaded into its own storage; none
  has a bundled-asset producer. ``WatermarkSettings.image_path`` in
  ``src/config.py`` looks like an operator default that would argue for a second
  root, but it has no reader anywhere in the service (asserted below), so the
  request body is the only producer.
* ``theme.font_path`` / ``theme.font_bold_path`` → **no tenant root at all**.
  The font loader reads ``settings.theme.font_*`` (operator ``THEME_FONT_*``
  env) and never the request's theme dict; the legitimate targets are files
  baked into the image under ``/usr/share/fonts/truetype/brand``, i.e. *outside*
  the storage root, so confining them there would break rendering. Asserted in
  ``TestFontPathsAreOperatorConfigNotTenantInput``.
* ``narration_audio_dir``'s **http(s)** branch → **not guarded**. That path is
  produced internally (``{work_dir}/narration_audio``), not by the tenant.
  Asserted in ``TestInternallyProducedPathsAreNotGuarded``.

No test here reads a real sensitive file: the "secret" is a fixture inside the
test's own tmpdir, and every assertion is that the guard refused it *because it
sits outside the configured root*. Nothing touches the network.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from src.api.schemas.compilation import (
    CompileRequest,
    EditDecisionInput,
    RecordingInput,
    ShotListInput,
    ShotListPartInput,
    SubtitleInput,
    ThemeInput,
    TitleCardRequest,
    WatermarkInput,
)
from src.services.compilation._compiler_context import _JobContext
from src.services.compilation._effects import _EffectsMixin
from src.services.compilation._segment_builders import _SegmentBuildersMixin
from src.services.compilation.compiler_service import CompilerService
from tests.conftest import TENANT_U1, TENANT_U3

_SETTINGS = "src.config.get_settings"


def _settings(*, root: Path, allow_local: bool = False) -> MagicMock:
    s = MagicMock()
    s.compilation.allow_local_file_inputs = allow_local
    s.storage.local_path = str(root)
    return s


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """Stand-in for ``/app/data`` — the only tree local refs may name."""
    root = tmp_path / "app-data"
    root.mkdir()
    return root


@pytest.fixture
def outside_file(tmp_path: Path) -> Path:
    """A file OUTSIDE the storage root, standing in for ``/etc/passwd``."""
    secret = tmp_path / "outside" / "passwd"
    secret.parent.mkdir()
    secret.write_text("root:x:0:0:sensitive-value-that-must-never-be-rendered")
    return secret


@pytest.fixture
def sibling_prefix_file(tmp_path: Path) -> Path:
    """``/app/data-evil/x`` when the root is ``/app/data``.

    Prefix confusion: a ``str.startswith`` confinement accepts this, an ancestor
    check refuses it. The fixture name mirrors that distinction on purpose.
    """
    sibling = tmp_path / "app-data-evil"
    sibling.mkdir()
    target = sibling / "x"
    target.write_text("sibling-prefix-payload")
    return target


@pytest.fixture
def escaping_symlink(storage_root: Path, outside_file: Path) -> Path:
    """A symlink INSIDE the root pointing out of it.

    The compile work dir is writable by the service, so this is a realistic
    bypass; a prefix comparison done before ``resolve()`` would accept it.
    """
    link = storage_root / "innocent.srt"
    link.symlink_to(outside_file)
    return link


@pytest.fixture
def hostile_refs(
    storage_root: Path,
    outside_file: Path,
    sibling_prefix_file: Path,
    escaping_symlink: Path,
) -> dict[str, str]:
    """The four rejection shapes every confined field must refuse.

    Returned as a mapping so a failure names the shape that got through rather
    than an index.
    """
    return {
        "dot_dot_traversal": f"{storage_root}/../outside/passwd",
        "absolute_outside_every_root": str(outside_file),
        "symlink_escape": str(escaping_symlink),
        "sibling_prefix_confusion": str(sibling_prefix_file),
    }


HOSTILE_SHAPES = (
    "dot_dot_traversal",
    "absolute_outside_every_root",
    "symlink_escape",
    "sibling_prefix_confusion",
)


def _ctx(tmp_path: Path) -> _JobContext:
    ctx = _JobContext(job_id="job-1", tenant_id=TENANT_U1, job=MagicMock(), settings=MagicMock())
    ctx.work_dir = tmp_path / "work"
    ctx.title_cards_dir = tmp_path / "work" / "title_cards"
    ctx.edited_dir = tmp_path / "work" / "edited"
    ctx.normalized_dir = tmp_path / "work" / "normalized"
    ctx.output_path = tmp_path / "work" / "demo.mp4"
    for d in (ctx.work_dir, ctx.title_cards_dir, ctx.edited_dir, ctx.normalized_dir):
        d.mkdir(parents=True, exist_ok=True)
    ctx.output_path.write_bytes(b"video")
    return ctx


class _Builder(_SegmentBuildersMixin):
    def __init__(self) -> None:
        self._storage = MagicMock()


# --- F1: watermark.image_path -----------------------------------------------


class TestWatermarkImagePathIsConfined:
    """``_apply_watermark`` → ffmpeg ``overlay_image`` → downloadable video."""

    @pytest.mark.parametrize("shape", HOSTILE_SHAPES)
    def test_a_hostile_reference_never_reaches_ffmpeg(
        self, shape: str, hostile_refs: dict[str, str], storage_root: Path, tmp_path: Path
    ) -> None:
        overlay = MagicMock()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation.video_filters.overlay_image", overlay),
        ):
            _EffectsMixin._apply_watermark(_ctx(tmp_path), {"image_path": hostile_refs[shape]})
        overlay.assert_not_called()

    def test_an_in_root_watermark_is_still_overlaid(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Accept case — the control that stops this shipping a prod breakage."""
        logo = storage_root / "brand" / "wm.png"
        logo.parent.mkdir()
        logo.write_bytes(b"\x89PNG")
        overlay = MagicMock()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation.video_filters.overlay_image", overlay),
        ):
            _EffectsMixin._apply_watermark(_ctx(tmp_path), {"image_path": str(logo)})
        overlay.assert_called_once()
        assert overlay.call_args.args[1] == logo.resolve()

    def test_the_refusal_is_the_guard_and_not_a_missing_file(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """Non-vacuity: the SAME ref is accepted once the gate opens.

        Without this control, ``assert_not_called`` above would also hold if the
        fixture merely did not exist, or if the unguarded code crashed — and the
        test would prove nothing about the guard.
        """
        overlay = MagicMock()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)),
            patch("src.services.compilation.video_filters.overlay_image", overlay),
        ):
            _EffectsMixin._apply_watermark(_ctx(tmp_path), {"image_path": str(outside_file)})
        overlay.assert_called_once()
        assert overlay.call_args.args[1] == outside_file

    def test_the_operator_default_has_no_reader_so_storage_root_is_the_whole_boundary(
        self,
    ) -> None:
        """``WatermarkSettings.image_path`` would justify a second allowed root.

        It does not, because nothing reads it: the request body is the only
        producer of a watermark path. This asserts that premise rather than
        trusting the comment that records it.
        """
        import subprocess

        repo_src = Path(__file__).resolve().parents[3] / "src"
        hits = subprocess.run(  # noqa: S603
            ["/usr/bin/grep", "-rn", "settings.watermark", str(repo_src)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert hits.stdout == "", f"WatermarkSettings gained a reader: {hits.stdout}"


# --- F1: subtitles.file_path -------------------------------------------------


class TestSubtitleFilePathIsConfined:
    """``_apply_subtitles`` — the sharpest sink: burn renders text into frames."""

    @pytest.mark.parametrize("shape", HOSTILE_SHAPES)
    def test_a_hostile_reference_is_never_burned_in(
        self, shape: str, hostile_refs: dict[str, str], storage_root: Path, tmp_path: Path
    ) -> None:
        ffmpeg = MagicMock()
        ctx = _ctx(tmp_path)
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation._effects._ffmpeg", return_value=ffmpeg),
        ):
            _EffectsMixin._apply_subtitles(ctx, {"file_path": hostile_refs[shape], "mode": "burn"})
        ffmpeg.burn_subtitles.assert_not_called()
        ffmpeg.embed_subtitles.assert_not_called()
        assert ctx.subtitle_error == "Subtitle file refused by the local-path guard"

    def test_the_exploit_shape_is_what_the_guard_stops(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """F1 reproduced, then blocked.

        The exploit is ``subtitles={'file_path': '/etc/passwd', 'mode': 'burn'}``:
        ffmpeg is handed the file and renders its text into the MP4. With the
        gate OPEN the file reaches ``burn_subtitles`` — that is the primitive,
        demonstrated. With the gate CLOSED (the shipped default) it does not.
        """
        ffmpeg_open, ffmpeg_closed = MagicMock(), MagicMock()

        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)),
            patch("src.services.compilation._effects._ffmpeg", return_value=ffmpeg_open),
        ):
            _EffectsMixin._apply_subtitles(
                _ctx(tmp_path), {"file_path": str(outside_file), "mode": "burn"}
            )
        # The primitive: the secret file was handed to the subtitle burner.
        ffmpeg_open.burn_subtitles.assert_called_once()
        assert ffmpeg_open.burn_subtitles.call_args.args[1] == outside_file

        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation._effects._ffmpeg", return_value=ffmpeg_closed),
        ):
            _EffectsMixin._apply_subtitles(
                _ctx(tmp_path), {"file_path": str(outside_file), "mode": "burn"}
            )
        ffmpeg_closed.burn_subtitles.assert_not_called()

    def test_an_in_root_subtitle_file_is_still_burned_in(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Accept case."""
        srt = storage_root / "job-3" / "captions.srt"
        srt.parent.mkdir()
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
        ffmpeg = MagicMock()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation._effects._ffmpeg", return_value=ffmpeg),
        ):
            _EffectsMixin._apply_subtitles(_ctx(tmp_path), {"file_path": str(srt), "mode": "burn"})
        ffmpeg.burn_subtitles.assert_called_once()
        assert ffmpeg.burn_subtitles.call_args.args[1] == srt.resolve()

    def test_a_config_without_mode_embeds_and_never_burns(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Owner rule: subtitles are never burned in unless ``mode="burn"`` is explicit."""
        srt = storage_root / "job-4" / "captions.srt"
        srt.parent.mkdir()
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
        ffmpeg = MagicMock()
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation._effects._ffmpeg", return_value=ffmpeg),
        ):
            _EffectsMixin._apply_subtitles(_ctx(tmp_path), {"file_path": str(srt)})
        ffmpeg.embed_subtitles.assert_called_once()
        ffmpeg.burn_subtitles.assert_not_called()

    def test_the_refusal_does_not_echo_the_rejected_path(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """The "not found" branch reports the path; the refusal must not.

        ``subtitle_error`` reaches the caller through ``degradations``, so
        echoing a refused path would downgrade the closed LFI into a
        filesystem existence oracle.
        """
        ctx = _ctx(tmp_path)
        with (
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
            patch("src.services.compilation._effects._ffmpeg", return_value=MagicMock()),
        ):
            _EffectsMixin._apply_subtitles(ctx, {"file_path": str(outside_file), "mode": "burn"})
        assert ctx.subtitle_error is not None
        assert str(outside_file) not in ctx.subtitle_error
        assert outside_file.name not in ctx.subtitle_error


# --- F1: clips_dir + narration_audio_dir -------------------------------------


class TestTenantSuppliedDirectoriesAreConfined:
    """``clips_dir`` and the local branch of ``narration_audio_dir``."""

    @pytest.mark.parametrize("field", ["clips_dir", "narration_audio_dir"])
    @pytest.mark.parametrize("shape", HOSTILE_SHAPES)
    def test_a_hostile_directory_is_refused(
        self, field: str, shape: str, hostile_refs: dict[str, str], storage_root: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert (
                CompilerService._confined_local_dir("job-1", hostile_refs[shape], field=field)
                is None
            )

    @pytest.mark.parametrize("field", ["clips_dir", "narration_audio_dir"])
    def test_an_in_root_directory_is_accepted(self, field: str, storage_root: Path) -> None:
        """Accept case — the AI-clip and narration round trips must keep working."""
        d = storage_root / "job-5" / "clips"
        d.mkdir(parents=True)
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert CompilerService._confined_local_dir("job-1", str(d), field=field) == d.resolve()

    def test_the_refusal_is_the_guard_and_not_a_missing_directory(
        self, storage_root: Path, tmp_path: Path
    ) -> None:
        """Non-vacuity: the same directory IS returned once the gate opens."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir(exist_ok=True)
        with patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)):
            assert (
                CompilerService._confined_local_dir("job-1", str(outside_dir), field="clips_dir")
                == outside_dir
            )

    def test_absent_is_not_refused(self, storage_root: Path) -> None:
        """``None``/empty means "not supplied", which is not a guard rejection."""
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            assert CompilerService._confined_local_dir("j", None, field="clips_dir") is None
            assert CompilerService._confined_local_dir("j", "", field="clips_dir") is None


class TestInternallyProducedPathsAreNotGuarded:
    """An internally produced path must NOT be pushed through the tenant guard.

    When ``narration_audio_dir`` is an http(s) URL the compiler downloads into
    ``{work_dir}/narration_audio`` — a directory it just created, outside the
    storage root in every test and in prod alike. Forcing that through
    confinement would silently disable narration for every URL caller.
    """

    async def test_the_download_dir_is_used_even_when_outside_the_storage_root(
        self, tmp_path: Path
    ) -> None:
        """The contract, asserted behaviourally rather than by reading source.

        The download branch moved out of ``_initialize`` into the module-level
        ``_download_narration_audio`` so the sink could be driven directly (see
        ``test_artifact_fetch_caps``). This used to inspect ``_initialize``'s
        source for the ``narr_dir = ...`` line, which the move broke. Driving the
        function is a stronger check of the same contract: ``work_dir`` here is a
        tmp dir OUTSIDE the storage root, so if the download dir were pushed
        through ``_confined_local_dir`` it would come back ``None`` and narration
        would be silently disabled for every URL caller.
        """
        from src.services.compilation import compiler_service

        ctx = _JobContext(job_id="j", tenant_id=TENANT_U3, job=MagicMock(), settings=MagicMock())
        ctx.work_dir = tmp_path / "work"
        ctx.work_dir.mkdir(parents=True, exist_ok=True)

        def _handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"ID3\x03\x00\x00\x00")

        # Bind the real class BEFORE patching: the patch target is the httpx
        # module attribute itself, so resolving it inside the factory would
        # recurse into the patch.
        real_client = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(_handle)
            return real_client(*args, **kwargs)

        storage_root = tmp_path / "storage"
        storage_root.mkdir(parents=True, exist_ok=True)
        with (
            patch("src.services.compilation.compiler_service.httpx.AsyncClient", _factory),
            patch(_SETTINGS, return_value=_settings(root=storage_root)),
        ):
            await compiler_service._download_narration_audio(
                ctx, "http://93.184.216.34/narration.mp3"
            )

        assert ctx.narration_audio_dir == ctx.work_dir / "narration_audio", (
            "the internally-produced narration download dir must not be tenant-confined"
        )

    def test_the_local_branch_is_still_confined(self) -> None:
        """The other half: a raw tenant string DOES go through the guard."""
        import inspect

        from src.services.compilation import compiler_service

        src = inspect.getsource(compiler_service.CompilerService._initialize)
        _, _, local_branch = src.partition("else:")
        assert "_confined_local_dir" in local_branch


# --- F1: logo_path -----------------------------------------------------------


class TestLogoPathIsConfined:
    """``parts[].logo_path`` → ``TitleCardGenerator`` → ``Image.open``."""

    def _run(self, ctx: _JobContext, logo: str) -> MagicMock:
        ctx.title_gen = MagicMock()
        builder = _Builder()
        with patch.object(
            _SegmentBuildersMixin, "_segment_with_optional_narration", lambda *a: a[3]
        ):
            builder._process_title_card(ctx, "p1", {"title": "t", "logo_path": logo})
        return ctx.title_gen

    @pytest.mark.parametrize("shape", HOSTILE_SHAPES)
    def test_a_hostile_logo_is_not_handed_to_pillow(
        self, shape: str, hostile_refs: dict[str, str], storage_root: Path, tmp_path: Path
    ) -> None:
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            gen = self._run(_ctx(tmp_path), hostile_refs[shape])
        assert gen.generate_video.call_args.kwargs["logo_path"] is None

    def test_an_in_root_logo_is_still_rendered(self, storage_root: Path, tmp_path: Path) -> None:
        """Accept case — branded title cards must keep their logo."""
        logo = storage_root / "brand" / "logo.png"
        logo.parent.mkdir()
        logo.write_bytes(b"\x89PNG")
        with patch(_SETTINGS, return_value=_settings(root=storage_root)):
            gen = self._run(_ctx(tmp_path), str(logo))
        assert gen.generate_video.call_args.kwargs["logo_path"] == str(logo.resolve())

    def test_the_refusal_is_the_guard_and_not_a_missing_file(
        self, storage_root: Path, outside_file: Path, tmp_path: Path
    ) -> None:
        """Non-vacuity control."""
        with patch(_SETTINGS, return_value=_settings(root=storage_root, allow_local=True)):
            gen = self._run(_ctx(tmp_path), str(outside_file))
        assert gen.generate_video.call_args.kwargs["logo_path"] == str(outside_file)


class TestFontPathsAreOperatorConfigNotTenantInput:
    """``theme.font_path`` / ``font_bold_path``: storage-root confinement is WRONG.

    Fonts legitimately live in the image at ``/usr/share/fonts/truetype/brand``
    (Dockerfile ``COPY assets/fonts/*.ttf``) — outside the storage root. That is
    only safe because the loader never reads the tenant's value: it reads
    ``settings.theme.font_*``. This asserts that separation, which is the whole
    justification for not applying the storage root to those two fields.
    """

    def test_the_font_loader_reads_operator_config_not_the_request_theme(
        self, tmp_path: Path
    ) -> None:
        from src.services.compilation.title_card_generator import _get_font

        operator_font = tmp_path / "operator.ttf"
        settings = MagicMock()
        settings.theme.font_path = str(operator_font)
        settings.theme.font_bold_path = ""

        loaded: list[str] = []
        sentinel = MagicMock(name="loaded-font")

        def _truetype(name, *args, **kwargs):
            loaded.append(str(name))
            return sentinel

        with (
            patch(_SETTINGS, return_value=settings),
            patch("PIL.ImageFont.truetype", _truetype),
        ):
            font = _get_font(20)

        assert loaded == [str(operator_font)], "operator THEME_FONT_PATH must be tried first"
        assert font is sentinel

    def test_a_tenant_theme_font_path_reaches_no_sink(self) -> None:
        """The request-side spelling is never consulted by the generator."""
        import inspect

        from src.services.compilation import title_card_generator

        src = inspect.getsource(title_card_generator)
        assert 'tc.get("font_path")' not in src
        assert 'tc.get("font_bold_path")' not in src
        assert "settings.theme.font_path" in src


# --- F2: part_id is a filename, not a path -----------------------------------


class TestPartIdIsAFilenameNotAPath:
    """The arbitrary-WRITE primitive: a traversing id escapes every job dir."""

    def test_the_write_primitive_is_real_when_the_id_is_unconstrained(self, tmp_path: Path) -> None:
        """F2 reproduced at the sink, bypassing the schema.

        ``_materialize_image_ref`` writes the decoded ``data:`` payload to
        ``{title_cards_dir}/{part_id}_asset{ext}``. Handed a traversing id
        directly — which is exactly what an unvalidated request body used to
        produce — attacker bytes land at an attacker-chosen path OUTSIDE the job
        directory. This documents the mechanism the admission guard below closes;
        it is not asserting a defect in the shipped code, which can no longer
        receive such an id.
        """
        ctx = _ctx(tmp_path)
        target_dir = tmp_path / "victim"
        target_dir.mkdir()
        traversing_id = "../../victim/pwned"
        _Builder()._materialize_image_ref(ctx, traversing_id, "data:image/png;base64,cHduZWQ=")
        assert (target_dir / "pwned_asset.png").read_bytes() == b"pwned"

    @pytest.mark.parametrize(
        "hostile_id",
        [
            "../../../../app/x",
            "../../etc/passwd",
            "/app/data/../../etc/shadow",
            "a/b",
            "a\\b",
            "..",
            ".",
            "",
            "with space",
            "semi;colon",
        ],
    )
    def test_admission_refuses_an_id_that_is_not_a_single_filename(self, hostile_id: str) -> None:
        with pytest.raises(ValidationError):
            ShotListPartInput(id=hostile_id)
        with pytest.raises(ValidationError):
            RecordingInput(part_id=hostile_id, video_path="/app/data/j/p.mp4")
        with pytest.raises(ValidationError):
            EditDecisionInput(part_id=hostile_id)

    def test_the_exact_exploit_request_body_no_longer_validates(self) -> None:
        """F2 blocked, at the shape the exploit actually posts.

        ``parts[0].id='../../../../app/x'`` carried alongside a ``data:`` image
        is the full primitive; the whole ``CompileRequest`` must now 422.
        """
        with pytest.raises(ValidationError, match=re.escape("must not contain '..'")):
            CompileRequest(
                shot_list=ShotListInput.model_validate(
                    {
                        "parts": [
                            {
                                "id": "../../../../app/x",
                                "part_type": "slide_preview",
                                "image_url": "data:image/png;base64,cHduZWQ=",
                            }
                        ],
                        "recording_order": ["../../../../app/x"],
                    }
                ),
                recordings=[],
            )

    @pytest.mark.parametrize(
        "legit_id",
        ["part-1", "part_1", "intro", "ai1", "01HZX.CHAPTER-2", "0198f0c2-6d1a-7c3e-8b2f"],
    )
    def test_the_ids_real_callers_use_still_validate(self, legit_id: str) -> None:
        """Accept case.

        These are the shapes ``POST /api/record/*`` emits — the recording schemas
        have enforced the same ``^[a-zA-Z0-9_\\-\\.]+$`` since they were written,
        so the record → compile hand-off is unaffected.
        """
        assert ShotListPartInput(id=legit_id).id == legit_id
        assert RecordingInput(part_id=legit_id, video_path="/app/data/j/p.mp4").part_id == legit_id
        assert EditDecisionInput(part_id=legit_id).part_id == legit_id


# --- Admission layer for the new path fields ---------------------------------


class TestFileUriIsRefusedAtAdmissionOnTheNewFields:
    """``file://`` has no producer in this service, so it is a 422, as in #111."""

    def test_watermark_image_path(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            WatermarkInput(image_path="file:///etc/passwd")

    def test_subtitles_file_path(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            SubtitleInput(file_path="file:///etc/passwd")

    def test_clips_dir(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            CompileRequest(
                shot_list=ShotListInput(parts=[]), recordings=[], clips_dir="file:///etc"
            )

    @pytest.mark.parametrize(
        "field", ["background_image", "logo_path", "font_path", "font_bold_path"]
    )
    def test_theme_asset_paths(self, field: str) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            ThemeInput.model_validate({field: "file:///etc/passwd"})

    def test_shot_list_part_logo_path(self) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            ShotListPartInput(id="p1", logo_path="file:///etc/passwd")

    @pytest.mark.parametrize("field", ["background_image", "logo_path"])
    def test_title_card_request_asset_paths(self, field: str) -> None:
        with pytest.raises(ValidationError, match="must not be a local file:// reference"):
            TitleCardRequest.model_validate({"title": "t", field: "file:///etc/passwd"})

    def test_an_internal_http_target_is_refused_on_the_new_fields(self) -> None:
        """A public-IP literal keeps ``getaddrinfo`` local; the guard fails closed."""
        with pytest.raises(
            ValidationError, match="must not target an internal or reserved network address"
        ):
            WatermarkInput(image_path="http://100.64.0.5:5432/x")

    def test_a_bare_path_still_passes_admission(self) -> None:
        """Regression control for the round trip: only the SINK decides bare paths."""
        assert WatermarkInput(image_path="/app/data/j/wm.png").image_path == "/app/data/j/wm.png"
        assert SubtitleInput(file_path="/app/data/j/c.srt").file_path == "/app/data/j/c.srt"
        assert (
            CompileRequest(
                shot_list=ShotListInput(parts=[]), recordings=[], clips_dir="/app/data/j/clips"
            ).clips_dir
            == "/app/data/j/clips"
        )
        assert (
            ThemeInput(font_path="/usr/share/fonts/truetype/brand/Fraunces-Regular.ttf").font_path
            == "/usr/share/fonts/truetype/brand/Fraunces-Regular.ttf"
        )
