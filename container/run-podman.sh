#!/bin/bash
# Run the container using Podman (or Docker)
# Usage: ./container/run-podman.sh [options]
#
# Options:
#   --no-gpu      Run without GPU support
#   --detach      Run in background
#   --port PORT   Use custom port (default: 8080)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="agents-among-us:latest"
PORT=8080
GPU_FLAG="--gpus all"
DETACH_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-gpu)
            GPU_FLAG=""
            shift
            ;;
        --detach|-d)
            DETACH_FLAG="-d"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use podman if available, otherwise docker
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    # Podman uses --device for GPU access
    if [ -n "$GPU_FLAG" ]; then
        GPU_FLAG="--device nvidia.com/gpu=all"
    fi
else
    CONTAINER_CMD="docker"
fi

echo "Running ${IMAGE_NAME}..."
echo "Port: ${PORT}"
echo "GPU: ${GPU_FLAG:-disabled}"

cd "$PROJECT_ROOT"

# Pass .env file if it exists (API keys for external providers)
ENV_FLAG=""
if [ -f "$(pwd)/.env" ]; then
    ENV_FLAG="--env-file $(pwd)/.env"
    echo "Loading API keys from .env"
fi

$CONTAINER_CMD run --rm -it \
    $DETACH_FLAG \
    $GPU_FLAG \
    $ENV_FLAG \
    -p "${PORT}:8080" \
    -v "$(pwd)/logs:/app/logs:z" \
    -v "$(pwd)/frontend/data:/app/frontend/data:z" \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/app/.cache/huggingface:z" \
    --name agents-among-us \
    "$IMAGE_NAME"
