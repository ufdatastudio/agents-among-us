#!/bin/bash
# Build the container image using Podman (or Docker)
# Usage: ./container/build-podman.sh [tag]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="agents-among-us"
TAG="${1:-latest}"

echo "Building ${IMAGE_NAME}:${TAG}..."
echo "Project root: ${PROJECT_ROOT}"

cd "$PROJECT_ROOT"

# Use podman if available, otherwise docker
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
else
    CONTAINER_CMD="docker"
fi

echo "Using: ${CONTAINER_CMD}"

$CONTAINER_CMD build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f Dockerfile \
    .

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run with GPU support:"
echo "  ${CONTAINER_CMD} run --rm -it --gpus all -p 8080:8080 ${IMAGE_NAME}:${TAG}"
