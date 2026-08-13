#!/usr/bin/env bash
# tests/smoke_test_docker.sh — Docker image smoke test.
#
# Builds the Docker image, starts the container, verifies the /health
# endpoint returns 200, then tears everything down.
#
# Usage:
#   bash tests/smoke_test_docker.sh
#
# Requirements:
#   - Docker must be installed and running.
#   - An ANTHROPIC_API_KEY env var (even a dummy value works for the health check).
#
# Exit codes:
#   0 — all checks passed
#   1 — a check failed

set -euo pipefail

IMAGE_NAME="langgraph-agent-stack:smoke-test"
CONTAINER_NAME="langgraph-smoke-$$"
VOLUME_NAME="langgraph-smoke-data-$$"
PORT=18910
MAX_WAIT=30

cleanup() {
    echo "[smoke] Cleaning up..."
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || true
    docker volume rm "$VOLUME_NAME" 2>/dev/null || true
    docker rmi "$IMAGE_NAME" 2>/dev/null || true
}
trap cleanup EXIT

echo "═══════════════════════════════════════════════"
echo " Docker Smoke Test — langgraph-agent-stack"
echo "═══════════════════════════════════════════════"

# 1. Build the image
echo "[smoke] Building Docker image..."
docker build -f infra/Dockerfile -t "$IMAGE_NAME" . --quiet

# 1b. Named volume /app/data must inherit uid 1001 from the image.
# Compose mounts sqlite-data:/app/data; if the directory is missing from the
# image, Docker creates the mount as root:root and appuser cannot write SQLite.
echo "[smoke] Verifying named volume /app/data is writable by uid 1001..."
docker volume create "$VOLUME_NAME" >/dev/null
if docker run --rm \
    --user 1001:1001 \
    --entrypoint sh \
    -v "$VOLUME_NAME:/app/data" \
    "$IMAGE_NAME" \
    -c 'touch /app/data/.write-probe && rm /app/data/.write-probe'; then
    echo "[smoke] ✓ Named volume /app/data is writable by uid 1001"
else
    echo "[smoke] ✗ Named volume /app/data is not writable by uid 1001"
    echo "[smoke] Image must mkdir -p /app/data before USER appuser so an empty named volume copies those permissions."
    exit 1
fi

# 2. Start the container
echo "[smoke] Starting container on port $PORT..."
docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$PORT:8000" \
    -e LLM_PROVIDER=anthropic \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-ant-smoke-test-dummy}" \
    -e MEMORY_BACKEND=sqlite \
    -e SQLITE_PATH=/tmp/smoke.db \
    -e ENVIRONMENT=development \
    "$IMAGE_NAME" >/dev/null

# 3. Wait for the container to be healthy
echo "[smoke] Waiting for /health (max ${MAX_WAIT}s)..."
elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT" ]; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "[smoke] ✓ /health responded 200 after ${elapsed}s"
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

if [ "$elapsed" -ge "$MAX_WAIT" ]; then
    echo "[smoke] ✗ /health did not respond within ${MAX_WAIT}s"
    echo "[smoke] Container logs:"
    docker logs "$CONTAINER_NAME" 2>&1 | tail -30
    exit 1
fi

# 4. Verify health response payload
HEALTH_BODY=$(curl -sf "http://localhost:$PORT/health")
echo "[smoke] Health response: $HEALTH_BODY"

if echo "$HEALTH_BODY" | grep -q '"status"'; then
    echo "[smoke] ✓ Health payload contains status field"
else
    echo "[smoke] ✗ Health payload missing status field"
    exit 1
fi

# 5. Verify OpenAPI docs are served
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "http://localhost:$PORT/docs")
if [ "$HTTP_CODE" = "200" ]; then
    echo "[smoke] ✓ /docs returns 200"
else
    echo "[smoke] ✗ /docs returned $HTTP_CODE"
    exit 1
fi

# 6. Verify container runs as non-root
CONTAINER_USER=$(docker exec "$CONTAINER_NAME" whoami 2>/dev/null || echo "unknown")
if [ "$CONTAINER_USER" = "appuser" ]; then
    echo "[smoke] ✓ Container runs as non-root user (appuser)"
else
    echo "[smoke] ⚠ Container user: $CONTAINER_USER (expected appuser)"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo " All smoke tests passed ✓"
echo "═══════════════════════════════════════════════"
