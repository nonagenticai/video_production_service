# Operation ID Plan: video_production_service

Canonical registry of `operation_id` values for every route in
`video_production_service`. These IDs feed the org-wide n8n custom-node
generator and the `ultrareview` cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (nlp_service,
  worker_management, kag, analysis_service, video_compiler_service,
  video_analyzer_service),
- **globally unique** — disambiguated against the rest of the fleet's verbs.

This service hosts two production domains (recording, compilation) plus
analytics, events, metrics, and a health surface. Verbs are prefixed
with `recording_` / `compilation_` / `compiled_` / `production_` where
the generic verb (`list_jobs`, `get_job_status`, `download_video`,
`get_usage_*`) would collide across the fleet.

## Files to edit

- `src/asgi.py` — root endpoint
- `src/api/health.py` — health probes
- `src/api/metrics.py` — Prometheus
- `src/api/events.py` — SSE
- `src/api/analytics.py` — usage analytics
- `src/api/routes/recording.py` — `/api/record/*`
- `src/api/routes/compilation.py` — `/api/compile` (core)
- `src/api/routes/_compilation_jobs.py` — `/api/compile/jobs/*` (via `register()`)
- `src/api/routes/_compilation_presets.py` — `/api/compile/{themes,presets,templates}/*` (via `register()`)
- `src/api/routes/_compilation_extras.py` — `/api/compile/{title-card,edit-plan,batch,...}` (via `register()`)

## Mapping Tables

### `src/asgi.py` (tag: `health`)

| Method | Function | operation_id   | Rationale                                  |
|--------|----------|----------------|--------------------------------------------|
| GET    | `root`   | `service_root` | Service banner; matches the fleet standard. |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id   | Rationale                                                  |
|--------|----------------|----------------|------------------------------------------------------------|
| GET    | `health_check` | `health_check` | Primary health probe; matches the fleet standard verb.     |
| GET    | `health_live`  | `health_live`  | Liveness sub-probe; one-to-one with path.                  |
| GET    | `health_ready` | `health_ready` | Readiness sub-probe; one-to-one with path.                 |

### `src/api/metrics.py` (tag: `metrics`)

| Method | Function   | operation_id         | Rationale                                                  |
|--------|------------|----------------------|------------------------------------------------------------|
| GET    | `metrics`  | `prometheus_metrics` | Generic `metrics` would collide; explicit Prometheus verb. |

### `src/api/events.py` (tag: `events`)

| Method | Function         | operation_id                | Rationale                                                  |
|--------|------------------|-----------------------------|------------------------------------------------------------|
| GET    | `stream_events`  | `stream_production_events`  | Domain-prefix; `stream_events` would collide fleet-wide.   |

### `src/api/analytics.py` (tag: `analytics`)

| Method | Function              | operation_id                       | Rationale                                                                          |
|--------|-----------------------|------------------------------------|------------------------------------------------------------------------------------|
| GET    | `get_usage_summary`   | `get_production_usage_summary`     | `get_usage_summary` would collide with any other service exposing usage analytics. |
| GET    | `get_usage_daily`     | `get_production_usage_daily`       | Same — domain-prefix.                                                              |

### `src/api/routes/recording.py` (tag: `record`)

| Method | Function                       | operation_id                    | Rationale                                                                                              |
|--------|--------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------|
| POST   | `record_part`                  | `record_video_part`             | Verb is the domain action; "video_part" disambiguates from generic `record_part`.                       |
| POST   | `record_content`               | `record_html_content`           | HTML-content recording is distinct from browser-script recording.                                       |
| POST   | `record_url`                   | `record_url`                    | Live-URL recording (authenticated SPA capture); distinct from HTML-content + browser-script recording.  |
| POST   | `capture_screens`              | `capture_screens`               | Single-pass per-screen screenshot capture (one PNG per demo screen, zipped); distinct from clip recording. |
| POST   | `record_batch`                 | `record_video_batch`            | Generic `record_batch` would collide; prefix with the resource verb.                                    |
| GET    | `list_jobs`                    | `list_recording_jobs`           | Domain-prefix; generic `list_jobs` collides with compilation's `list_jobs`.                             |
| GET    | `get_job_status`               | `get_recording_job_status`      | Same — domain-prefix to disambiguate from compilation's same-function-name handler.                     |
| GET    | `get_job_result`               | `get_recording_result`          | Domain-prefix.                                                                                          |
| GET    | `get_job_video`                | `download_recording_video`      | `get_job_video` is implementation-leaking; rename to the operation verb and prefix with the domain.     |
| POST   | `cancel_job`                   | `cancel_recording_job`          | Domain-prefix to disambiguate from compilation's `cancel_job`.                                          |
| GET    | `get_batch_status`             | `get_recording_batch_status`    | Domain-prefix; compilation has a `get_batch_status` as well.                                            |
| POST   | `record_batch_from_artifacts`  | `record_batch_from_artifacts`   | Already domain-specific; the source name reads as a verb_noun phrase.                                    |

### `src/api/routes/compilation.py` (tag: `compile`)

| Method | Function        | operation_id               | Rationale                                                                                  |
|--------|-----------------|----------------------------|--------------------------------------------------------------------------------------------|
| POST   | `compile_video` | `start_video_compilation`  | `compile_video` reads as an imperative; the fleet pattern is `start_<noun>_<verb>` (matches `start_video_analysis` in video_analyzer_service). |

### `src/api/routes/_compilation_jobs.py` (tag: `compile`)

Registered via `register(router)` onto the `/api/compile` router.

| Method | Function                  | operation_id                       | Rationale                                                                                    |
|--------|---------------------------|------------------------------------|----------------------------------------------------------------------------------------------|
| GET    | `list_jobs`               | `list_compilation_jobs`            | Domain-prefix; recording's `list_jobs` is `list_recording_jobs`.                              |
| DELETE | `cancel_job`              | `cancel_compilation_job`           | Domain-prefix to disambiguate from recording.                                                 |
| GET    | `get_job_status`          | `get_compilation_job_status`       | Domain-prefix.                                                                                |
| GET    | `get_job_result`          | `get_compilation_result`           | Domain-prefix.                                                                                |
| GET    | `get_job_video`           | `download_compiled_video`          | `get_job_video` is implementation-leaking; the verb is "download", noun is "compiled video".  |
| GET    | `get_job_video_url`       | `get_compiled_video_url`           | Returns presigned/streaming URL — verb_noun shape.                                            |
| GET    | `get_job_thumbnail`       | `download_compilation_thumbnail`   | GET handler returns the thumbnail bytes; "download" matches the operation.                    |
| POST   | `generate_thumbnail`      | `generate_compilation_thumbnail`   | POST handler regenerates; domain-prefix.                                                      |
| GET    | `get_compilation_report`  | `get_compilation_report`           | Already domain-specific.                                                                      |

### `src/api/routes/_compilation_presets.py` (tags: `themes`, `presets`, `templates`)

| Method | Function                | operation_id                          | Rationale                                                                       |
|--------|-------------------------|---------------------------------------|---------------------------------------------------------------------------------|
| GET    | `list_themes`           | `list_compilation_themes`             | Domain-prefix; "themes" is too generic at fleet scope.                          |
| GET    | `get_theme`             | `get_compilation_theme`               | Same — domain-prefix.                                                           |
| GET    | `list_presets`          | `list_compilation_presets`            | Domain-prefix; compilation presets are distinct from any other "preset" concept. |
| GET    | `list_custom_presets`   | `list_custom_compilation_presets`     | Distinguish from built-in preset listing.                                       |
| GET    | `get_preset`            | `get_compilation_preset`              | Domain-prefix.                                                                  |
| POST   | `create_custom_preset`  | `create_custom_compilation_preset`    | Domain-prefix.                                                                  |
| DELETE | `delete_custom_preset`  | `delete_custom_compilation_preset`    | Domain-prefix.                                                                  |
| POST   | `create_template`       | `create_compilation_template`         | Domain-prefix; "template" is too generic fleet-wide.                            |
| GET    | `list_templates`        | `list_compilation_templates`          | Same — domain-prefix.                                                           |
| DELETE | `delete_template`       | `delete_compilation_template`         | Same — domain-prefix.                                                           |

### `src/api/routes/_compilation_extras.py` (tags: `compile`, `admin`)

| Method | Function                | operation_id                            | Rationale                                                                       |
|--------|-------------------------|-----------------------------------------|---------------------------------------------------------------------------------|
| POST   | `generate_title_card`   | `generate_compilation_title_card`       | Domain-prefix; sync helper used by compilation.                                  |
| POST   | `generate_edit_plan`    | `generate_compilation_edit_plan`        | Domain-prefix.                                                                  |
| POST   | `on_demand_cleanup`     | `run_compilation_cleanup`               | `cleanup` is too generic at fleet scope; "run" emphasizes the imperative.       |
| POST   | `create_findings_video` | `create_findings_video`                 | Already a domain-specific noun (findings video); kept as-is.                    |
| POST   | `batch_compile`         | `start_compilation_batch`               | Verb_noun shape; matches `start_video_compilation`.                             |
| GET    | `get_batch_status`      | `get_compilation_batch_status`          | Domain-prefix to disambiguate from recording's batch status.                    |
| POST   | `audio_export`          | `export_compilation_audio`              | Verb_noun shape; "export" is the operation, "compilation audio" the noun.       |
| POST   | `generate_preview`      | `generate_compilation_preview`          | Domain-prefix.                                                                  |

## Verification

```bash
cd video_production_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of every right-hand column above.
Drift (rename, missed decorator, fallback) fails the test loudly.

## Disambiguation rationale

Concrete renames where the function name was generic enough to collide
across the fleet or within this service:

- `compile_video` → `start_video_compilation` — match
  `start_video_analysis` in video_analyzer_service; the fleet standard is
  `start_<noun>_<verb>` for async job submission.
- `list_jobs` / `get_job_status` / `cancel_job` / `get_job_video` /
  `get_job_result` / `get_batch_status` exist in BOTH recording and
  compilation domains — every one is domain-prefixed (`recording_` /
  `compilation_`) to keep the IDs unique within this service.
- `get_job_video` / `get_job_thumbnail` (GET handlers that stream bytes
  or redirect to presigned URLs) → `download_*` because the operation is
  "download", not "get a property".
- `on_demand_cleanup` → `run_compilation_cleanup` — `cleanup` is too
  generic fleet-wide; the verb is "run".
- `batch_compile` → `start_compilation_batch` — verb_noun shape.
- `audio_export` → `export_compilation_audio` — verb first, noun second.
- `get_usage_summary` / `get_usage_daily` → `get_production_usage_summary`
  / `get_production_usage_daily` — service-prefix because every service
  with analytics has the same generic function names.

## Future routes

When adding a route in this service, **always** set `operation_id="…"`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
