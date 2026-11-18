# Video Production Service

Records browser and motion-graphics animations with Playwright/Chromium and edits them with FFmpeg. Port 6308.

## What Video Production Service Does

The service exposes two cooperating domains behind one FastAPI app. Its live role in the w2v pipeline is **recording**: when the orchestrator needs a clip — a captured web page, a raw-HTML render, or a branded motion-graphics animation — it calls `/api/record/*`, and the service drives a headless Chromium browser through Playwright to produce a deterministic MP4, WebM, or GIF. Recording jobs are asynchronous: the caller receives a `job_id` and a poll URL, work runs on a bounded pool of subprocesses, and the finished clip is stored (local disk or MinIO) and streamed back on demand.

The most distinctive recording surface is `record_animation` (`POST /api/record/animation`). This service **owns the branded motion-graphics HTML templates** in `src/services/recording/templates/` — animated title cards, animated deck slides, Ken Burns pan/zoom for stills, lower thirds, kinetic typography, pipeline-story, social cards, and data-viz. `AnimationRecorder` loads a template, injects `window.PARAMS` (brand and content) inline — because Playwright's `set_content` does not fire `add_init_script`, params must be inlined or every template renders its defaults — then frame-steps Chromium to capture a deterministic clip. The orchestrator calls one template per shot and splices the resulting clips into the final video.

The second domain is **compilation/editing** (`/api/compile/*`): an FFmpeg pipeline that assembles clips into a finished video with transitions, title cards, color/watermark filters, themes, presets, and audio export. Compilation jobs run the same async job-manager pattern, expose a WebSocket for live progress, and validate output quality before completing. A per-process ceiling on concurrent ffmpeg subprocesses (default 2) caps combined decode-buffer memory so the pod is not SIGKILLed under concurrent load.

This service is **not** `video_compiler_service` (namespace `video-compiler`, port 8200), which runs the orchestrator's `video_compile` shot-list → MP4 hot-path stage. The two are separate, easily-confused repos; a `video_compile`-stage symptom belongs to the other service. This service's own compilation domain is a distinct FFmpeg editing surface, and its primary production duty is the recording/`record_animation` path on port 6308 in namespace `w2v`.

State lives in PostgreSQL 18 (psycopg 3) via PgBouncer — recording jobs, compilation jobs, custom presets, templates, batches, and per-tenant usage rows — with Redis providing job-status caching, Server-Sent-Event pub/sub, and rate limiting. Every request is tenant-isolated by license token, and Phase D row-level security keeps each tenant's rows bound across the multi-commit background jobs that recording and compilation both use.

## Features

- **Playwright/Chromium recording** — captures web pages, raw HTML content, and URLs as MP4/WebM/GIF in a headless browser so the pipeline can turn any renderable surface into video footage
- **Branded motion-graphics templates** — owns the animated title-card, slide, Ken Burns, lower-third, kinetic-typography, pipeline-story, social-card, and data-viz HTML templates so brand-consistent motion shots are generated deterministically via `record_animation`
- **Deterministic animation capture** — frame-steps Chromium with inlined `window.PARAMS` so every render is reproducible rather than wall-clock-dependent
- **FFmpeg compilation & editing** — assembles clips with transitions, title cards, watermark/color filters, themes, and presets so recorded footage becomes a finished video
- **Async job model** — recording and compilation both return a `job_id` + poll URL and run on bounded worker pools so callers never block on multi-minute renders
- **Batch operations** — records or compiles many items in one request (including batch-from-artifacts) so high-fan-out runs avoid per-item HTTP overhead
- **Quality presets** — named viewport/CRF/bitrate presets (`preview`→`4k`) so browser footage is captured at the run's native resolution tier rather than upscaled
- **Live progress streaming** — SSE (`/events/{tenant_id}`) and a per-job WebSocket so the frontend can show real-time job progress
- **ffmpeg memory guard** — a process-wide ceiling on concurrent ffmpeg subprocesses caps combined decode memory so the pod is not OOM-killed under load
- **License token authentication** — introspects tokens against the licenses service with 5-minute caching so every request is tenant-isolated without per-call latency
- **Phase D row-level security** — an `after_begin` GUC re-bind listener keeps the tenant bound across multi-commit jobs so background pipelines stay tenant-isolated under enforce
- **Automatic cleanup** — background reapers expire stale/old recording and compilation jobs and their artifacts so storage and the DB do not grow unbounded

## Authentication

Auth enabled by default (`AUTH_ENABLED=true`). The `AuthMiddleware` (delegating to `w2v_common.auth.middleware`) reads a license token from `X-Tenant-Id: Bearer <license_token>` — **not** the `Authorization` header, which is reserved for the SPA's Keycloak JWT — and introspects it against the licenses service, caching results for 5 minutes.

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected via `/license/introspect` on the licenses service (5-minute cache) |
| 2 | Trusted network | Bypass assigning the default tenant for requests from private CIDRs (in-cluster callers) |

Outbound service-to-service calls forward `X-Tenant-Id`, `X-License-Id`, and `X-Subject-Id` as plain headers (no `Bearer` prefix). Exempt paths (`/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico`, `/metrics`, plus the `/docs`, `/redoc`, `/health/` prefixes) never require a token.

Phase D RLS is installed at startup (`install_rls_tenant_rebind`) but inert until the runtime connects as the restricted role with `TENANT_RLS_MODE=enforce`; `rls_resolvers.py` holds the SECURITY DEFINER cross-tenant resolvers.

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6308
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Recording `/api/record`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/part` | Record a single browser part |
| POST | `/content` | Record from raw HTML content |
| POST | `/animation` | Record a branded motion-graphics template (`record_animation`) |
| POST | `/url` | Record a URL (auto-scroll + viewport capture) |
| POST | `/capture/screens` | Per-step screenshot capture → zip |
| POST | `/batch` | Record multiple parts as a batch |
| POST | `/batch_from_artifacts` | Batch record from prior artifacts |
| GET | `/jobs` | List recording jobs (paginated) |
| GET | `/jobs/{job_id}` | Get job status |
| GET | `/jobs/{job_id}/result` | Get recording result |
| GET | `/jobs/{job_id}/video` | Download/stream recorded video |
| POST | `/jobs/{job_id}/cancel` | Cancel a recording |
| GET | `/batch/{batch_id}` | Get batch status |

### Compilation `/api/compile`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/` | Start a compilation job |
| GET | `/jobs` | List compilation jobs (paginated) |
| GET | `/jobs/{job_id}` | Get job status |
| DELETE | `/jobs/{job_id}` | Delete a compilation job |
| GET | `/jobs/{job_id}/result` | Get compilation result |
| GET | `/jobs/{job_id}/video` | Download compiled video |
| GET | `/jobs/{job_id}/video-url` | Get a presigned video URL |
| GET | `/jobs/{job_id}/thumbnail` | Get job thumbnail |
| POST | `/jobs/{job_id}/thumbnail` | Generate job thumbnail |
| GET | `/jobs/{job_id}/report` | Get compilation stage report |
| POST | `/batch` | Start a batch compilation |
| GET | `/batch/{batch_id}` | Get batch status |
| POST | `/title-card` | Generate a title card |
| POST | `/edit-plan` | Generate an edit plan |
| POST | `/findings-video` | Build a findings video |
| POST | `/audio-export` | Export the audio track |
| POST | `/preview` | Generate a compilation preview |
| POST | `/admin/cleanup` | Trigger on-demand job cleanup |
| GET | `/themes` | List available themes |
| GET | `/themes/{name}` | Get a theme |
| GET | `/presets` | List built-in presets |
| GET | `/presets/{name}` | Get a preset |
| GET | `/presets/custom` | List custom presets |
| POST | `/presets/custom` | Create a custom preset |
| DELETE | `/presets/custom/{preset_id}` | Delete a custom preset |
| GET | `/templates` | List compilation templates |
| POST | `/templates` | Create a compilation template |
| DELETE | `/templates/{template_id}` | Delete a compilation template |

### Compilation WebSocket

| Path | Description |
|------|-------------|
| `/api/compile/jobs/{job_id}/ws` | Real-time compilation progress stream. **Authenticated on the handshake, then gated.** The caller presents a licence token as `X-Tenant-Id: Bearer <token>` (preferred) or a non-JWT `Authorization: Bearer <token>`; `stamp_ws_identity` introspects it and stamps the tenant/subject the licences service asserts, and only then does `LICENSE_GATE_WS` render its verdict (close `1008 license_<state>` / `1013 license_unavailable`). A bare `X-Tenant-Id: <uuid>` is **not** an identity and there is no `?license=` query lane — see `src/api/ws_identity.py`. The HTTP `LICENSE_GATE` is unmountable here: it is typed for a `Request` and raises a `TypeError` on every handshake on a WebSocket route. Order matters — the gate without the stamping step closes `1008 license_none` for *every* caller, entitled or not. **Then authorized per job:** `{job_id}` must belong to the caller's verified tenant (RLS, via `CompilationJobManager.get_job` — the same rule the HTTP job routes use) or the socket closes `1008 job_not_found` **before** subscribing to the Redis channel. A job owned by another tenant and a job that does not exist get a byte-identical refusal, so `job_id` cannot be enumerated. |

### Analytics `/api/analytics`

Licence-gated (`LICENSE_GATE`), like `/api/record` and `/api/compile`: both routes are tenant-scoped, billable reads.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/usage` | Usage summary for the calling tenant |
| GET | `/usage/daily` | Daily usage breakdown |

### Events

| Method | Path | Description |
|--------|------|-------------|
| GET | `/events/{tenant_id}` | Server-Sent Events stream for job progress. Licence-gated (`LICENSE_GATE`), and capped: a single stream lives at most `API_SSE_MAX_STREAM_SECONDS` (default 3600s) before the server closes it with a `stream_expired` frame, so the client reconnects **through** the gate and a revocation cannot be outlived by an already-open stream. `/events` is also covered by the per-tenant rate limiter. `{tenant_id}` **must equal the caller's own tenant** — any other value is `403` (the Redis pub/sub channel has no RLS backstop, so the check is the only boundary). |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint listing |
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/healthz` | Liveness alias |
| GET | `/readyz` | Readiness alias |
| GET | `/metrics` | Prometheus metrics |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6308` | Listen port |
| `API_DEBUG` | `false` | Enable /docs and debug mode |
| `API_RELOAD` | `false` | Auto-reload on code changes |
| `AUTH_ENABLED` | `true` | License-token auth middleware |
| `AUTH_DEFAULT_TENANT_ID` | `default` | Tenant for trusted-network requests |
| `AUTH_CACHE_TTL` | `300` | License introspection cache TTL (seconds) |
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_JSON_LOGS` | `false` | Structured JSON logging |
| `DATABASE_URL` | -- | PostgreSQL connection string (psycopg 3; required) |
| `DATABASE_POOL_SIZE` | `10` | SQLAlchemy pool size |
| `DATABASE_POOL_MIN_SIZE` | `2` | Minimum pooled connections |
| `DATABASE_MAX_OVERFLOW` | `20` | Connections beyond pool size |
| `REDIS_URL` | -- | Redis connection string (required) |
| `REDIS_MAX_CONNECTIONS` | `10` | Redis pool max connections |
| `RECORDING_VIEWPORT_WIDTH` | `1920` | Chromium recording viewport width |
| `RECORDING_VIEWPORT_HEIGHT` | `1080` | Chromium recording viewport height |
| `RECORDING_TIMEOUT_MS` | `120000` | Playwright test timeout (ms) |
| `RECORDING_HEADLESS` | `true` | Run Chromium headless |
| `RECORDING_MAX_CONCURRENT_RECORDINGS` | `3` | Max concurrent recording subprocesses |
| `RECORDING_MAX_RETRIES` | `2` | Retry attempts for failed recordings |
| `RECORDING_OUTPUT_FORMAT` | `mp4` | Output format: `mp4` or `webm` |
| `RECORDING_DEFAULT_QUALITY_PRESET` | `standard` | `preview`, `standard`, `high_quality`, `2k`, `4k` |
| `COMPILATION_CRF` | `20` | H.264 CRF (lower = better) |
| `COMPILATION_TARGET_WIDTH` | `1920` | Output width |
| `COMPILATION_TARGET_HEIGHT` | `1080` | Output height |
| `COMPILATION_TARGET_FPS` | `30` | Output frame rate |
| `COMPILATION_TRANSITION_TYPE` | `smart` | `cut`, `fade`, or `smart` |
| `COMPILATION_MAX_CONCURRENT_FFMPEG_SUBPROCESSES` | `2` | Process-wide ffmpeg concurrency ceiling |
| `FFMPEG_TIMEOUT` | `300` | FFmpeg command timeout (seconds) |
| `FFMPEG_PRESET` | `medium` | H.264 encoding preset |
| `FFMPEG_DEFAULT_CRF` | `23` | Default CRF for encoding |
| `STORAGE_BACKEND` | `local` | `local` or `minio` |
| `STORAGE_LOCAL_PATH` | `/app/data` | Local storage path |
| `STORAGE_MINIO_ENDPOINT` | `minio.test:8389` | MinIO endpoint |
| `STORAGE_MINIO_BUCKET` | `w2v-video-production` | MinIO bucket name |
| `STORAGE_CROSS_BUCKET_ALLOWLIST` | *(empty)* | Comma-separated buckets — besides `STORAGE_MINIO_BUCKET` — that a tenant-supplied `s3://bucket/key` asset reference on a compile request may be read from. **Fail-closed:** empty means own bucket only. A refused read logs an error naming this variable. |
| `STORAGE_RETENTION_HOURS` | `168` | Job artifact retention (7 days) |
| `RATELIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATELIMIT_MAX_CONCURRENT_JOBS` | `3` | Max concurrent jobs per tenant |
| `RATELIMIT_DAILY_QUOTA` | `100` | Compilations per tenant per day |
| `RATELIMIT_REQUESTS_PER_MINUTE` | `30` | API requests per minute per tenant |
| `OTEL_ENABLED` | `false` | Enable OpenTelemetry instrumentation |
| `OTEL_SERVICE_NAME` | `video-production-service` | OTel service name |
| `OTEL_OTLP_ENDPOINT` | -- | OTLP gRPC exporter endpoint |

## Architecture

```
src/
├── asgi.py                          # FastAPI app, lifespan, middleware, root + OpenAPI
├── config.py                        # Pydantic Settings (extends w2v_common base classes)
├── exceptions.py                    # Exception hierarchy
├── api/
│   ├── routes/
│   │   ├── recording.py             # /api/record/* endpoints
│   │   ├── compilation.py           # /api/compile/* router owner + compile-job create
│   │   ├── _compilation_jobs.py     # compile list/cancel/status/result/video/report + ws
│   │   ├── _compilation_presets.py  # themes/presets/templates handlers
│   │   ├── _compilation_extras.py   # title-card/edit-plan/findings/batch/audio/preview
│   │   └── _compilation_helpers.py  # shared deps + background-task wrappers
│   ├── schemas/
│   │   ├── recording.py             # Recording request/response models
│   │   └── compilation.py           # Compilation request/response models
│   ├── analytics.py                 # /api/analytics usage endpoints
│   ├── events.py                    # /events/{tenant_id} SSE stream
│   ├── health.py                    # Health + readiness endpoints
│   ├── metrics.py                   # Prometheus /metrics
│   ├── dependencies.py              # FastAPI dependency injection
│   └── ws.py                        # WebSocket ConnectionManager
├── auth/
│   ├── middleware.py                # AuthMiddleware (delegates to w2v_common)
│   └── models.py                    # User model
├── database/
│   ├── models.py                    # SQLAlchemy ORM (jobs, presets, templates, batches, usage)
│   ├── rls_events.py                # Phase D after_begin GUC re-bind listener
│   ├── rls_resolvers.py             # SECURITY DEFINER cross-tenant resolvers
│   └── migrations/versions/         # 001_initial_schema, 002_rls
├── services/
│   ├── recording/
│   │   ├── animation_recorder.py    # record_animation: frame-step templates → clip
│   │   ├── templates/               # branded motion-graphics HTML (title card, slide, ken_burns, ...)
│   │   ├── playwright_recorder.py   # Playwright browser recording
│   │   ├── content_recorder.py      # raw-HTML recording
│   │   ├── url_recorder.py          # URL recording
│   │   ├── recorder_service.py      # recording orchestration
│   │   ├── job_manager.py / concurrency.py  # job CRUD + semaphore concurrency
│   │   ├── quality_presets.py       # preset viewport/CRF/bitrate definitions
│   │   ├── event_publisher.py / artifact_fetcher.py  # Redis events + artifact fetch
│   │   └── webhook_service.py / cleanup_service.py  # webhook delivery + cleanup
│   ├── compilation/
│   │   ├── compiler_service.py      # compilation orchestration
│   │   ├── _assembly.py / _effects.py / _finalize.py / _segment_builders.py  # pipeline stages
│   │   ├── batch_service.py         # batch compilation
│   │   ├── video_editor.py          # trim/edit operations
│   │   ├── video_filters.py         # color correction, watermark
│   │   ├── title_card_generator.py  # title card generation (Pillow)
│   │   ├── quality_validator.py     # output quality validation
│   │   ├── themes.py / presets.py / template_service.py  # theme/preset/template defs
│   │   ├── edit_plan_service.py / findings_video_service.py  # edit plan + findings builders
│   │   ├── rate_limiter.py / job_manager.py  # per-tenant limiting + job CRUD
│   │   └── webhook_service.py / cleanup_service.py  # webhook delivery + cleanup
│   ├── ffmpeg/                      # FFmpeg ops (base/ops/compose/filters/probe/transcode/subtitles)
│   ├── storage_service.py           # local + MinIO storage backends
│   └── protocols.py                 # service protocol/interface definitions
├── middleware/
│   ├── rate_limiter.py              # rate-limit middleware
│   ├── metrics.py                   # Prometheus metrics middleware
│   └── request_id.py                # request-id ContextVar middleware
├── observability/
│   └── setup.py                     # optional OpenTelemetry setup
└── utils/
    └── error_formatting.py          # error message formatting
```

## Testing

```bash
uv run pytest tests/ -v --cov=src              # All tests with coverage
uv run pytest tests/unit/recording/ -v         # Recording domain tests
uv run pytest tests/unit/compilation/ -v       # Compilation domain tests
uv run ruff check src/ tests/                  # Lint
uv run ruff format --check src/ tests/         # Format check
uv run ty check                                # Type check
```

Coverage threshold: 80% (enforced in pyproject.toml). `./scripts/ci-local.sh` mirrors the CI Lint & Unit Tests job step-for-step (uv sync → ruff check → ruff format --check → ty check → pytest+coverage → detect-secrets → pip-audit → bandit).

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check`, `pytest` against a Postgres 18 service container (and Chromium, installed when the repo uses Playwright), and `scripts/pip-audit.sh`.

No image is published from this repository; build it locally with the Dockerfile.

## Docker

```bash
docker build -t video-production-service .
docker run -p 6308:6308 video-production-service
docker run -p 6308:6308 -e LOG_LEVEL=DEBUG -e AUTH_ENABLED=false \
  -e DATABASE_URL=postgresql+psycopg://host/video_production_service \
  -e REDIS_URL=redis://host:6379/0 video-production-service
```

The runtime bakes in Chromium/Playwright and ffmpeg, so the image is multi-GB.

## Dependencies

- **w2v_orchestrator** — primary caller; drives `record_animation` per template and splices resulting clips
- **licenses service** — validates license tokens via `/license/introspect`
- **video_compiler_service** (port 8200, namespace `video-compiler`) — the separate service that runs the orchestrator's `video_compile` shot-list stage; do not confuse with this service
- **MinIO** (`minio.test:8389`, bucket `w2v-video-production`) — artifact object storage
- **PostgreSQL 18** — persistent job/preset/template/usage storage via PgBouncer (`pgbouncer.pgbouncer:6432`), psycopg 3 driver
- **Redis** — job-status caching, SSE pub/sub, rate limiting
- **ffmpeg / ffprobe** — video conversion, probing, assembly, transitions
- **Chromium** — headless browser runtime for Playwright recording
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — schema validation and settings
- **SQLAlchemy** (async) — ORM
- **psycopg** (v3) — PostgreSQL driver
- **Playwright** — browser automation for recording
- **Pillow** — title card image generation
- **prometheus-client** — metrics
- **minio / miniopy-async** — object storage clients
- **OpenTelemetry** (optional extra) — distributed tracing with OTLP export

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. Tenant and licence identity is forwarded from the caller. It runs with
no licence switch set; unset means off.
