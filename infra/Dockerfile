# =============================================================================
# Stage 1 — builder
# Installs uv and resolves/installs all production dependencies into a venv.
# The resulting virtual environment is later copied to the slim runtime image.
# =============================================================================
# Pin to a specific patch version for reproducibility across builds.
# For stricter supply-chain control, append @sha256:<digest> for your target platform.

FROM python:3.14-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy uv from the official image — no curl|sh, no build-time network fetch.
# Version pinned here; COPY --from= does not support ARG variable substitution
# on all BuildKit versions used by GitHub Actions runners.
COPY --from=ghcr.io/astral-sh/uv:0.11.18 /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifests first to benefit from Docker layer caching
COPY pyproject.toml .
COPY README.md .
# If a lockfile exists, copy it too; otherwise uv will resolve from pyproject.toml
COPY uv.lock ./

# Install production dependencies into an explicit virtual environment.
# --locked  : respect the lockfile if present, fail if out of date
# --no-dev  : skip development dependencies
#
# Each ARG maps to a [project.optional-dependencies] group in pyproject.toml.
# Pass empty string "" to disable an extra.
#
#   docker build \
#     --build-arg LLM_EXTRAS=anthropic \
#     --build-arg INFRA_EXTRAS=redis \
#     --build-arg OBS_EXTRAS=observability \
#     -f infra/Dockerfile .
ARG LLM_EXTRAS="anthropic"
ARG INFRA_EXTRAS=""
ARG OBS_EXTRAS=""

RUN uv sync --locked --no-dev \
    --extra ${LLM_EXTRAS} \
    ${INFRA_EXTRAS:+--extra ${INFRA_EXTRAS}} \
    ${OBS_EXTRAS:+--extra ${OBS_EXTRAS}}

# =============================================================================
# Stage 2 — runtime
# Minimal image that only carries the application code and pre-built venv.
# Runs as a non-root user for defence-in-depth.
# =============================================================================
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python to use the venv copied from the builder stage
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Upgrade base OS packages for Trivy/CVE fixes (no curl — health check uses httpx in venv)
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /sbin/nologin --no-create-home appuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Copy application source code
# Copy each package explicitly to avoid accidentally including local secrets
COPY api/      ./api/
COPY core/     ./core/
COPY agents/   ./agents/
COPY pack_kernel/ ./pack_kernel/
COPY domain_packs/ ./domain_packs/
COPY connectors/ ./connectors/
COPY control_plane/ ./control_plane/

# Transfer ownership of the working directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Kubernetes liveness/readiness probes and docker-compose healthcheck target this port
EXPOSE 8000

# Container-level health check — fails fast if the API stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Default command — override via docker run / k8s deployment args if needed
# Use "python -m" so the venv's python is used regardless of script shebangs
# (shebangs bake the builder-stage path which differs from runtime).
# FORWARDED_ALLOW_IPS is read by the app (TRUST_PROXY_HEADERS) and passed to
# uvicorn --forwarded-allow-ips when set (K8s Ingress / ALB deployments).
ENV FORWARDED_ALLOW_IPS=""
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 ${FORWARDED_ALLOW_IPS:+--forwarded-allow-ips=$FORWARDED_ALLOW_IPS}"]
