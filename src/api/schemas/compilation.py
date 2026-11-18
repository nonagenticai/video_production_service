"""Request/response Pydantic models for video_compiler_service."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from src.utils.motion_presets import MOTION_PRESETS, normalize_motion_preset
from src.utils.url_guard import (
    validate_local_asset_ref,
    validate_optional_remote_url,
    validate_path_component,
    validate_webhook_url,
)

# --- Request Schemas ---


class LLMConfigOverride(BaseModel):
    """Optional overrides for the LLM provider used during compilation."""

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None


class RecordingInput(BaseModel):
    """A single recording result.

    ``video_path`` is overloaded: a local filesystem path, an ``s3://`` URI, or
    an http(s) URL that the compiler downloads. Only the http(s) branch is an
    SSRF sink, so only that branch is guarded — see
    :func:`src.utils.url_guard.validate_optional_remote_url`.
    """

    part_id: str
    video_path: str
    duration_seconds: float = 0.0
    file_size_bytes: int = 0
    width: int = 1920
    height: int = 1080
    has_audio: bool = False
    success: bool = True

    @field_validator("video_path", mode="before")
    @classmethod
    def _validate_video_path(cls, v: object) -> object:
        """Reject an internal-network target when ``video_path`` is a URL.

        Left untouched for local paths and ``s3://`` URIs, which are the common
        case and carry no outbound-request surface.
        """
        return validate_optional_remote_url(v, field="video_path")

    @field_validator("part_id", mode="before")
    @classmethod
    def _validate_part_id(cls, v: object) -> object:
        """See :func:`src.utils.url_guard.validate_path_component`."""
        return validate_path_component(v, field="part_id")


class ShotListPartInput(BaseModel):
    """A single shot list part."""

    id: str
    title: str = ""
    subtitle: str | None = None
    chapter: int | None = None
    part_type: str = "browser_recording"
    estimated_duration_seconds: float = 10.0
    # WP-1: AI clip metadata
    strip_ai_audio: bool = True
    ai_provider: str | None = None
    ai_prompt: str | None = None
    clip_type: str | None = None
    # WP-3: Enhanced title card config
    background_image: str | None = None
    logo_path: str | None = None
    logo_position: str | None = None  # top-left, top-right, center, bottom-left, bottom-right
    logo_scale: float | None = None
    animation: str | None = None  # fade_in, slide_in
    bullet_points: list[str] | None = None
    # Deck/image-aware segment metadata
    slide_image_url: str | None = None
    slide_role: str | None = None
    image_url: str | None = None
    motion_preset: str | None = None
    asset_id: str | None = None
    asset_kind: str | None = None
    asset_source: str | None = None
    approval_status: str | None = None
    provenance: dict | None = None

    @field_validator("motion_preset", mode="before")
    @classmethod
    def _validate_motion_preset(cls, v: object) -> object:
        """Reject a still-motion preset the renderer cannot draw.

        The field used to be a bare ``str | None`` forwarded verbatim to
        ``FFmpegOps.image_to_motion_video``, where an unrecognised value fell
        through to a plain ``scale=`` filter: a completely static clip on a
        green job. Admission-time rejection is the only place this is visible.
        """
        if v is None:
            return None
        try:
            return normalize_motion_preset(str(v))
        except ValueError as exc:
            raise ValueError(f"motion_preset must be one of {', '.join(MOTION_PRESETS)}") from exc

    @field_validator("image_url", "slide_image_url", "background_image", mode="before")
    @classmethod
    def _validate_asset_urls(cls, v: object, info: ValidationInfo) -> object:
        """Reject internal-network targets on the asset refs the compiler downloads.

        ``_materialize_image_ref`` dispatches these on their prefix: ``data:``,
        ``s3://``, ``file://`` and bare paths are handled locally, while an
        http(s) value becomes an outbound GET. Only the http(s) case is
        validated so the local/data-URI callers are unaffected.

        This is necessary but NOT sufficient on its own — the sink follows
        redirects, so it re-validates every hop via ``safe_get_sync``.
        """
        return validate_optional_remote_url(v, field=info.field_name or "asset url")

    @field_validator("logo_path", mode="before")
    @classmethod
    def _validate_logo_path(cls, v: object) -> object:
        """Guard the per-part logo overlay.

        This one is NOT decorative: ``_process_title_card`` passes it straight
        to ``TitleCardGenerator.generate_video``, which hands it to
        ``Image.open`` and composites the result into a downloadable video.
        Confinement to the storage root happens at that sink; this layer refuses
        the ``file://`` spelling and internal http(s) targets.
        """
        return validate_local_asset_ref(v, field="logo_path")

    @field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls, v: object) -> object:
        """The part id becomes a filename — see ``validate_path_component``."""
        return validate_path_component(v, field="id")


class ShotListInput(BaseModel):
    """Full shot list."""

    parts: list[ShotListPartInput]
    recording_order: list[str] = Field(default_factory=list)


class ThemeInput(BaseModel):
    """Theme configuration."""

    theme_name: str | None = None
    primary_color: str | None = None
    background_color: str | None = None
    text_color: str | None = None
    accent_color: str | None = None
    subtitle_color: str | None = None
    gradient_start: str | None = None
    gradient_end: str | None = None
    # WP-3: Enhanced theme config
    background_image: str | None = None
    logo_path: str | None = None
    logo_position: str | None = None
    logo_scale: float | None = None
    animation: str | None = None
    font_path: str | None = None
    font_bold_path: str | None = None

    @field_validator("background_image", "logo_path", "font_path", "font_bold_path", mode="before")
    @classmethod
    def _validate_theme_asset_paths(cls, v: object, info: ValidationInfo) -> object:
        """Guard the four on-disk references a theme override can carry.

        Reachability differs per field and is worth stating, because the
        conclusion is *not* "these are already safe":

        * ``background_image`` / ``logo_path`` reach ``Image.open`` through
          ``TitleCardGenerator``.
        * ``font_path`` / ``font_bold_path`` reach nothing — ``_get_font`` reads
          ``settings.theme.font_*``, never this dict.

        All four are additionally intercepted today by ``resolve_theme_config``,
        which runs ``validate_color`` over every non-``theme_name`` override and
        so raises on any value that is not a hex/rgb color. That is an
        **accident** of a color validator applied to non-color keys, not a
        security boundary — it fails the job with "Invalid color" rather than
        refusing an LFI attempt, and it evaporates the moment someone fixes that
        loop to skip non-color keys as its own comment already claims to. The
        guard belongs on the field.
        """
        return validate_local_asset_ref(v, field=info.field_name or "theme asset path")


class EditDecisionInput(BaseModel):
    """AI edit decision for a part."""

    part_id: str
    speed_factor: float = 1.0
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    remove_dead_time: bool = False

    @field_validator("part_id", mode="before")
    @classmethod
    def _validate_part_id(cls, v: object) -> object:
        """The part id becomes a filename — see ``validate_path_component``."""
        return validate_path_component(v, field="part_id")


# WP-5: Color correction
class ColorCorrectionInput(BaseModel):
    """Color correction settings for the compilation."""

    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    temperature: float = 0.0  # -1.0 (cool) to 1.0 (warm)


# WP-5: Watermark
class WatermarkInput(BaseModel):
    """Watermark overlay settings.

    ``image_path`` is a pure on-disk reference — nothing downloads it — and
    ``_EffectsMixin._apply_watermark`` overlays whatever it names onto the
    output video the tenant then downloads. ``WatermarkSettings.image_path`` in
    ``src/config.py`` looks like an operator-supplied default that might argue
    for a second allowed root, but it has **no reader anywhere in the service**:
    the only producer of this value is the request body, so the storage root is
    the whole of the correct boundary.
    """

    image_path: str
    position: str = "bottom-right"  # top-left, top-right, center, bottom-left, bottom-right
    opacity: float = 0.7
    scale: float = 0.15

    @field_validator("image_path", mode="before")
    @classmethod
    def _validate_image_path(cls, v: object) -> object:
        """See :func:`src.utils.url_guard.validate_local_asset_ref`."""
        return validate_local_asset_ref(v, field="watermark.image_path")


# WP-10: Subtitle configuration
class SubtitleInput(BaseModel):
    """Subtitle/caption configuration.

    ``file_path`` is the sharpest of the local-path fields: ``mode='burn'``
    renders the file's **text** into the video frames, so an unconfined value is
    not merely "read a file" but "read a file and hand its contents back to the
    caller as pixels".
    """

    file_path: str  # Path to SRT or VTT file
    # "embed" (soft track, the default) or "burn" (hardcode). Owner rule:
    # subtitles are never burned into the picture unless explicitly asked.
    mode: Literal["embed", "burn"] = "embed"
    font_size: int = 24
    font_color: str = "white"
    position: str = "bottom"  # "top" or "bottom"

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, v: object) -> object:
        """See :func:`src.utils.url_guard.validate_local_asset_ref`."""
        return validate_local_asset_ref(v, field="subtitles.file_path")


class CompileRequest(BaseModel):
    """Request to compile recordings into a video."""

    shot_list: ShotListInput
    recordings: list[RecordingInput]
    narration_audio_dir: str | None = None
    clips_dir: str | None = None
    transition_type: str | None = None
    crossfade_duration: float | None = Field(default=None, ge=0.0, le=10.0)
    # xfade_transition: specific FFmpeg xfade effect name (e.g. "fade", "wipeleft",
    # "dissolve", "circleopen", etc.).  Applies when transition_type is "fade" or
    # "smart".  Defaults to "fade" when not set.
    xfade_transition: str | None = None
    target_width: int | None = None
    target_height: int | None = None
    crf: int | None = None
    theme: ThemeInput | None = None
    theme_name: str | None = None
    edit_decisions: list[EditDecisionInput] | None = None
    preset: str | None = None
    # WP-2: Webhook
    webhook_url: str | None = None
    webhook_secret: str | None = None
    # WP-5: Filters
    color_correction: ColorCorrectionInput | None = None
    watermark: WatermarkInput | None = None
    # WP-8: Template
    template: str | None = None
    # WP-10: Subtitles
    subtitles: SubtitleInput | None = None
    llm_config: LLMConfigOverride | None = None

    @field_validator("webhook_url", mode="before")
    @classmethod
    def _validate_webhook_url(cls, v: object) -> object:
        """Guard the outbound completion webhook.

        ``webhook_service.deliver`` POSTs the full job payload here, so an
        unguarded value is simultaneously an SSRF primitive against every
        cluster-internal and tailnet service and a data-exfiltration channel.
        Mirrors the recording schema's ``webhook_url`` validator.
        """
        return validate_webhook_url(v)

    @field_validator("narration_audio_dir", mode="before")
    @classmethod
    def _validate_narration_audio_dir(cls, v: object) -> object:
        """Guard ``narration_audio_dir`` only when it is an http(s) URL.

        The compiler treats this as a local directory unless it starts with
        ``http://`` / ``https://``, in which case it downloads it. Local
        directory values — the overwhelmingly common case — pass through
        untouched.
        """
        return validate_optional_remote_url(v, field="narration_audio_dir")

    @field_validator("clips_dir", mode="before")
    @classmethod
    def _validate_clips_dir(cls, v: object) -> object:
        """Guard the AI-clip source directory.

        Unlike ``narration_audio_dir`` this one has no URL branch at all — the
        compiler only ever does ``Path(job.clips_dir) / f'{part_id}.mp4'`` — so
        the value is purely a local reference and the sink confines it to the
        storage root.
        """
        return validate_local_asset_ref(v, field="clips_dir")


class ThumbnailRequest(BaseModel):
    """Request to generate a thumbnail at a specific timestamp."""

    timestamp: float = 1.0
    width: int = 640
    height: int = 360


class TitleCardRequest(BaseModel):
    """Request to generate a single title card video.

    ``background_image`` and ``logo_path`` are currently **inert** on this
    route: ``_compilation_extras.generate_title_card`` builds the generator from
    ``body.theme`` and then calls ``generate_video`` without forwarding either
    field, so neither reaches Pillow. They are guarded anyway — the fields are
    published on the request model, so wiring them up later must not silently
    open an arbitrary-read, and a guard that has to be *remembered* at that
    point is a guard that will be missed.
    """

    title: str
    subtitle: str = ""
    chapter: int | None = None
    duration: float = 5.0
    width: int = 1920
    height: int = 1080
    theme: ThemeInput | None = None
    # WP-3: Enhanced title card options
    background_image: str | None = None
    logo_path: str | None = None
    logo_position: str | None = None
    logo_scale: float | None = None
    animation: str | None = None
    bullet_points: list[str] | None = None

    @field_validator("background_image", "logo_path", mode="before")
    @classmethod
    def _validate_asset_paths(cls, v: object, info: ValidationInfo) -> object:
        """See the class docstring and :func:`validate_local_asset_ref`."""
        return validate_local_asset_ref(v, field=info.field_name or "asset path")


# --- WP-6: Cleanup Request ---


class CleanupRequest(BaseModel):
    """Request for on-demand job cleanup."""

    job_ids: list[str] | None = None
    min_age_hours: float | None = None
    status_filter: list[str] | None = None
    dry_run: bool = False


class CleanupResponse(BaseModel):
    """Response from on-demand cleanup."""

    deleted_count: int
    job_ids: list[str]
    dry_run: bool


# --- WP-8: Custom Preset Schemas ---


class CustomPresetCreateRequest(BaseModel):
    """Request to create a custom preset."""

    name: str
    description: str = ""
    extends: str | None = None
    target_width: int | None = None
    target_height: int | None = None
    crf: int | None = None
    fps: int | None = None
    transition_type: str | None = None
    crossfade_duration: float | None = Field(default=None, ge=0.0, le=10.0)


class CustomPresetResponse(BaseModel):
    """Response for a custom preset."""

    id: str
    name: str
    description: str = ""
    extends: str | None = None
    target_width: int | None = None
    target_height: int | None = None
    crf: int | None = None
    fps: int | None = None
    transition_type: str | None = None
    crossfade_duration: float | None = None
    created_at: str | None = None


class CustomPresetListResponse(BaseModel):
    """List of custom presets."""

    presets: list[CustomPresetResponse]


# --- WP-8: Template Schemas ---


class TemplateCreateRequest(BaseModel):
    """Request to create a compilation template."""

    name: str
    description: str = ""
    preset_name: str | None = None
    theme_name: str | None = None
    theme_config: dict | None = None
    music_config: dict | None = None
    title_card_config: dict | None = None
    transition_type: str | None = None
    crossfade_duration: float | None = Field(default=None, ge=0.0, le=10.0)


class TemplateResponse(BaseModel):
    """Response for a compilation template."""

    id: str
    name: str
    description: str = ""
    preset_name: str | None = None
    theme_name: str | None = None
    theme_config: dict | None = None
    music_config: dict | None = None
    title_card_config: dict | None = None
    transition_type: str | None = None
    crossfade_duration: float | None = None
    is_builtin: bool = False
    created_at: str | None = None


class TemplateListResponse(BaseModel):
    """List of compilation templates."""

    templates: list[TemplateResponse]


# --- WP-9: Findings Video Schemas ---


class FindingInput(BaseModel):
    """A single finding for findings video generation.

    Every ``*_url`` / ``evidence_path`` field below is an asset reference that
    ``FindingsVideoService.build_shot_list`` copies into a plain shot-list part
    **dict**, which then reaches ``_materialize_image_ref`` exactly like a
    ``ShotListPartInput`` ref would. Because that hand-off is a dict, the
    validators on ``ShotListPartInput`` never run on these values, so the same
    guard has to be declared here or ``POST /api/compile/findings-video``
    silently bypasses the admission layer that ``POST /api/compile`` has.
    """

    title: str
    description: str = ""
    severity: str = "info"  # critical, high, medium, low, info
    category: str = "general"
    evidence_path: str | None = None
    image_url: str | None = None
    generated_image_url: str | None = None
    comparison_frame_url: str | None = None
    comparison_image_url: str | None = None
    slide_image_url: str | None = None
    slide_preview_url: str | None = None
    asset_kind: str | None = None
    asset_id: str | None = None
    asset_source: str | None = None
    slide_role: str | None = None
    approval_status: str | None = None
    motion_preset: str | None = None
    provenance: dict | None = None

    @field_validator("motion_preset", mode="before")
    @classmethod
    def _validate_motion_preset(cls, v: object) -> object:
        """Reject a still-motion preset the renderer cannot draw.

        The field used to be a bare ``str | None`` forwarded verbatim to
        ``FFmpegOps.image_to_motion_video``, where an unrecognised value fell
        through to a plain ``scale=`` filter: a completely static clip on a
        green job. Admission-time rejection is the only place this is visible.
        """
        if v is None:
            return None
        try:
            return normalize_motion_preset(str(v))
        except ValueError as exc:
            raise ValueError(f"motion_preset must be one of {', '.join(MOTION_PRESETS)}") from exc

    @field_validator(
        "evidence_path",
        "image_url",
        "generated_image_url",
        "comparison_frame_url",
        "comparison_image_url",
        "slide_image_url",
        "slide_preview_url",
        mode="before",
    )
    @classmethod
    def _validate_asset_refs(cls, v: object, info: ValidationInfo) -> object:
        """Apply the same asset-ref guard ``ShotListPartInput`` carries.

        The http(s) branch is already refused at the sink by ``safe_get_sync``,
        so this is not an open SSRF; what it buys is a 422 before a job row
        exists, plus the ``file://`` LFI rejection — neither of which the sink
        can give back once the row is persisted.
        """
        return validate_optional_remote_url(v, field=info.field_name or "asset ref")


class FindingsVideoRequest(BaseModel):
    """Request to generate a findings video."""

    findings: list[FindingInput]
    title: str = "Findings Report"
    subtitle: str = ""
    preset: str | None = None
    theme_name: str | None = None
    theme: ThemeInput | None = None


# --- WP-9: Batch Compilation Schemas ---


class BatchCompileRequest(BaseModel):
    """Request for batch compilation."""

    configs: list[CompileRequest]
    max_concurrent: int = 3


class BatchStatusResponse(BaseModel):
    """Status of a batch compilation."""

    batch_id: str
    status: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    job_ids: list[str] = []
    errors: list[dict] | None = None
    created_at: str | None = None
    completed_at: str | None = None


# --- WP-10: Audio Export Schemas ---


class AudioExportRequest(BaseModel):
    """Request to export audio from a compiled video."""

    job_id: str
    format: str = "mp3"  # mp3, wav, flac
    bitrate: str = "192k"


class AudioExportResponse(BaseModel):
    """Response for audio export."""

    path: str
    format: str
    duration_seconds: float
    file_size_bytes: int


# --- WP-10: Video Preview Schemas ---


class PreviewRequest(BaseModel):
    """Request to generate a video preview."""

    job_id: str | None = None
    recordings: list[RecordingInput] | None = None
    shot_list: ShotListInput | None = None
    duration_seconds: float = 15.0
    width: int = 640
    height: int = 360


class PreviewResponse(BaseModel):
    """Response for video preview generation."""

    path: str
    duration_seconds: float
    width: int
    height: int
    file_size_bytes: int


# --- Response Schemas ---


class AsyncJobResponse(BaseModel):
    """Response returned when an async compilation job is created."""

    job_id: str
    status: str = "pending"
    poll_url: str


class JobStatusResponse(BaseModel):
    """Current status of a compilation job."""

    job_id: str
    status: str
    created_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    result_url: str | None = None


class ChapterTimestampResponse(BaseModel):
    """A chapter marker with start time and duration in the compiled video."""

    chapter_number: int
    title: str
    subtitle: str = ""
    start_seconds: float
    duration_seconds: float


class ThumbnailResponse(BaseModel):
    """Response with thumbnail info."""

    path: str
    width: int
    height: int
    timestamp: float


# WP-1: Compilation report
class StageReportResponse(BaseModel):
    """Timing report for a single compilation stage."""

    stage: str
    duration_seconds: float
    details: dict | None = None


class DegradationResponse(BaseModel):
    """One requested feature that was dropped to keep the video deliverable."""

    feature: str
    reason: str


class CompilationReportResponse(BaseModel):
    """Full compilation report with per-stage timings.

    ``status`` discriminates a clean compile from a degraded one. The job ROW
    stays ``completed`` for both -- a video was produced and is billable -- so
    this nested field is the only place a caller can learn that something the
    user asked for was dropped. ``degradations`` names each dropped feature and
    a fixed, sanitised reason.

    ``status`` is one of:

    * ``success``        -- the producer recorded no drops from the sites it
      COVERS. It is not a guarantee that nothing was dropped: coverage is
      partial and enumerated in ``_finalize._finalize`` and
      ``tests/unit/test_degradation_coverage.py``.
    * ``partial_success`` -- at least one COVERED feature was dropped
    * ``unknown``        -- the report predates status derivation, so NOTHING can
      be concluded. This is NOT a synonym for success: defaulting absence to
      "success" would make the API assert a clean compile for a report that
      never carried the claim.

    ``subtitle_error`` is deliberately NOT surfaced. The producer writes it into
    the report blob, but its "file not found" branch echoes a resolved server
    path and its exception branch echoes up to 4000 chars of ffmpeg detail --
    the path-existence oracle ``_apply_subtitles``' own docstring refuses to
    return. The subtitle drop is reported through ``degradations`` instead,
    with a fixed reason.
    """

    job_id: str
    compilation_time_seconds: float
    stages: list[StageReportResponse]
    segments_processed: int = 0
    segments_excluded: int = 0
    transitions_used: int = 0
    audio_mixing_details: dict | None = None
    input_duration_seconds: float = 0.0
    output_duration_seconds: float = 0.0
    status: str = "unknown"
    degradations: list[DegradationResponse] = []


# WP-4: Quality metrics
class QualityIssue(BaseModel):
    """A single quality issue found during validation."""

    check: str
    severity: str  # error, warning, info
    message: str


class QualityMetricsResponse(BaseModel):
    """Quality validation results."""

    valid: bool
    issues: list[QualityIssue] = []
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    resolution: str = ""
    audio_peak_db: float | None = None
    encoding_valid: bool = True


class CompilationResultResponse(BaseModel):
    """Full result of a completed compilation including output path and metrics."""

    job_id: str
    status: str
    output_path: str | None = None
    total_duration_seconds: float | None = None
    total_parts: int | None = None
    file_size_bytes: int | None = None
    chapter_timestamps: list[ChapterTimestampResponse] | None = None
    compilation_time_seconds: float | None = None
    thumbnail_url: str | None = None
    presigned_url: str | None = None
    error: str | None = None
    # WP-1: Compilation report
    compilation_report: CompilationReportResponse | None = None
    # WP-4: Quality metrics
    quality_metrics: QualityMetricsResponse | None = None
    token_usage: dict | None = None


class TitleCardResponse(BaseModel):
    """Response for a generated title card video."""

    video_path: str
    duration: float
    width: int
    height: int


class ThemePresetResponse(BaseModel):
    """Response representing a single theme preset."""

    name: str
    primary_color: str
    background_color: str
    text_color: str
    accent_color: str
    subtitle_color: str
    gradient_start: str
    gradient_end: str


class ThemeListResponse(BaseModel):
    """Response containing all available theme presets."""

    themes: list[ThemePresetResponse]


class PresetResponse(BaseModel):
    """Response representing a single compilation preset."""

    name: str
    description: str
    target_width: int
    target_height: int
    crf: int
    fps: int
    transition_type: str
    crossfade_duration: float


class PresetListResponse(BaseModel):
    """Response containing all available compilation presets."""

    presets: list[PresetResponse]


class VideoUrlResponse(BaseModel):
    """Response containing a presigned (or fallback) URL for a compiled video."""

    url: str | None = None
    expires_in: int = 3600
    fallback_download_url: str


# --- Edit Plan Schemas ---


class EditPlanRequest(BaseModel):
    """Request to generate an AI edit plan."""

    recordings: list[RecordingInput]
    shot_list: ShotListInput
    target_duration_minutes: float | None = None
    llm_config: LLMConfigOverride | None = None


class EditPlanDecisionResponse(BaseModel):
    """A single AI edit decision for a recording part."""

    part_id: str
    speed_factor: float = 1.0
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    remove_dead_time: bool = False
    notes: str = ""


class EditPlanResponse(BaseModel):
    """Response containing AI-generated edit decisions."""

    decisions: list[EditPlanDecisionResponse]
    estimated_duration_seconds: float | None = None
    notes: str = ""
    token_usage: dict | None = None


# --- Job List & Cancel Schemas ---


class JobListItem(BaseModel):
    """Summary of a single job in a list."""

    job_id: str
    status: str
    created_at: str | None = None
    completed_at: str | None = None
    preset: str | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobListItem]
    total: int
    page: int
    page_size: int


class JobCancelResponse(BaseModel):
    """Response after cancelling a job."""

    job_id: str
    status: str
    message: str


# --- Usage Analytics Schemas ---


class UsageSummaryResponse(BaseModel):
    """Aggregated usage statistics."""

    tenant_id: UUID
    compilations_started: int = 0
    compilations_completed: int = 0
    compilations_failed: int = 0
    total_output_duration_seconds: float = 0.0
    total_output_size_bytes: int = 0


class DailyUsageItem(BaseModel):
    """Usage stats for a single day."""

    date: str
    compilations_started: int = 0
    compilations_completed: int = 0
    compilations_failed: int = 0
    total_output_duration_seconds: float = 0.0
    total_output_size_bytes: int = 0


class DailyUsageResponse(BaseModel):
    """Daily usage breakdown."""

    days: list[DailyUsageItem]


# --- WebSocket Progress Schemas ---


class ProgressUpdate(BaseModel):
    """WebSocket progress update payload broadcast during compilation."""

    event: str
    job_id: str
    stage: str
    progress_pct: float = 0.0
    current_part: int | None = None
    total_parts: int | None = None
    message: str = ""


# --- WP-10: Presigned URL Token ---


class PresignedTokenResponse(BaseModel):
    """Response for local storage presigned URL generation."""

    url: str
    token: str
    expires_at: str
    expires_in: int = 3600
