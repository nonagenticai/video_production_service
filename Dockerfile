# Multi-stage Dockerfile for video_production_service
#
# Single-context build: w2v-common resolves from the public git source declared
# via the [[tool.uv.index]] entry in pyproject.toml.
# Build with: docker buildx build -f Dockerfile .

ARG BASE_REGISTRY=docker.io
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from its public git tag
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Install all runtime deps:
# - libpq5, curl, ffmpeg: core
# - libnss3, libatk-bridge2.0-0, libdrm2, libxkbcommon0, libgbm1, xvfb: Playwright browser deps
# - Node.js 26: cluster standard, required for authored Playwright test scripts
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ffmpeg \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgbm1 \
    xvfb \
    ca-certificates \
    fontconfig \
    && curl -fsSL https://deb.nodesource.com/setup_26.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright Chromium to a shared location accessible by productionservice.
# This ~800MB layer depends on the dependency lock, never application source,
# so it lives here, ABOVE `COPY src/`. Placed below the source copy (as it was) it
# was invalidated on every commit, forcing a full rebuild AND a re-push of the whole
# Chromium layer every build; on the CI runner's ~1.5 MB/s uplink that push alone
# ran ~45 min and hit the build cap. Above the source copy it is stable, so the
# registry build cache (ci.yaml: registry_build_cache) makes it a cache hit and the
# push skips the blob. `--with-deps` is kept deliberately: the explicit apt list
# above is only a partial subset of Chromium's OS deps, and dropping it risks a
# missing-lib crash that per-PR CI (Ready-check only) would not catch.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers
# Python recorders and the Node CLI must use the same browser revision. Derive
# the CLI version from the frozen Python lock so a dependency upgrade cannot
# leave the image with browsers from an older, hard-coded Playwright release.
COPY uv.lock /tmp/playwright-uv.lock
RUN PLAYWRIGHT_VERSION="$(python -c 'import tomllib; from pathlib import Path; lock = tomllib.loads(Path("/tmp/playwright-uv.lock").read_text()); print(next(p["version"] for p in lock["package"] if p["name"] == "playwright"))')" && \
    npm install -g "playwright@${PLAYWRIGHT_VERSION}" && \
    playwright install chromium --with-deps && \
    chmod -R o+rx /opt/playwright-browsers

# Brand fonts for the motion-graphics templates. The base slim image ships only
# DejaVu/Liberation, so a template asking for e.g. 'Poppins' silently fell back to
# a generic serif. Install the real font family + rebuild the fontconfig cache so
# Chromium renders it by name.
COPY assets/fonts/*.ttf /usr/share/fonts/truetype/brand/
RUN fc-cache -f >/dev/null 2>&1 || true

# Create non-root user
RUN useradd -m -u 1000 productionservice && \
    mkdir -p /app /app/logs /app/data /app/shared-scripts /app/shared-media && \
    chown -R productionservice:productionservice /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=productionservice:productionservice /app/.venv /app/.venv

# Copy application code and alembic config
# Migrations live at src/database/migrations/ and are included via the src/ copy above.
COPY --chown=productionservice:productionservice src/ ./src/
COPY --chown=productionservice:productionservice alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

# Exercise the actual Python runtime and non-root permissions before an image
# can ship. A healthy HTTP process alone cannot detect a missing browser build.
COPY --chown=productionservice:productionservice scripts/smoke-browser.py ./scripts/smoke-browser.py
RUN python scripts/smoke-browser.py

# Port 6308 (Video Production Service)
EXPOSE 6308

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

# Schema is owned by alembic -- the gitops Deployment runs
# `alembic upgrade head` in an initContainer before this container
# starts. Do NOT re-run it from CMD here; the per-pod sequencing of
# (initContainer → migrate, then main container → serve) is what
# guarantees the migrate-completes-before-traffic invariant. If you
# need to run this image locally without the initContainer, exec
# `alembic upgrade head` by hand before invoking the CMD.
CMD ["python", "-m", "src.asgi"]
