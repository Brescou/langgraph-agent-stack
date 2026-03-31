# =============================================================================
# Stage 1 — builder
# Installs uv and resolves/installs all production dependencies into a venv.
# The resulting virtual environment is later copied to the slim runtime image.
# =============================================================================
FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_VERSION=0.5.21

# Install uv from the official installer (no pip, no poetry)
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure uv is on PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /build

# Copy dependency manifests first to benefit from Docker layer caching
COPY pyproject.toml .
# If a lockfile exists, copy it too; otherwise uv will resolve from pyproject.toml
COPY uv.lock* ./

# Install production dependencies into an explicit virtual environment
# --frozen  : respect the lockfile if present, fail if out of date
# --no-dev  : skip development dependencies
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2 — runtime
# Minimal image that only carries the application code and pre-built venv.
# Runs as a non-root user for defence-in-depth.
# =============================================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell Python to use the venv copied from the builder stage
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install curl for the HEALTHCHECK only; keep the layer tiny
RUN apt-get update && apt-get install -y --no-install-recommends curl \
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

# Transfer ownership of the working directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Kubernetes liveness/readiness probes and docker-compose healthcheck target this port
EXPOSE 8000

# Container-level health check — fails fast if the API stops responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command — override via docker run / k8s deployment args if needed
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
