#!/bin/bash
# Run the Apptainer container on HiPerGator
# Usage: ./container/run-apptainer.sh [options]
#
# Options:
#   --sif PATH    Path to .sif file (default: ./agents-among-us.sif)
#   --port PORT   Port to use (default: 8080)
#   --bind PATH   Additional bind mount

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SIF_PATH="${PROJECT_ROOT}/agents-among-us.sif"
PORT=8080
EXTRA_BINDS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sif)
            SIF_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --bind)
            EXTRA_BINDS="${EXTRA_BINDS} --bind $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ! -f "$SIF_PATH" ]; then
    echo "Error: Container image not found: ${SIF_PATH}"
    echo "Run build-apptainer.sh first to build the container."
    exit 1
fi

# Create local directories for bind mounts
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/frontend/data"

# Set up HuggingFace cache directory
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo "Running Apptainer container..."
echo "Container: ${SIF_PATH}"
echo "Port: ${PORT}"
echo "HF Cache: ${HF_CACHE}"
echo ""

# Check if apptainer or singularity is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi

# Load API keys from .env if it exists
ENV_FLAGS=""
if [ -f "${PROJECT_ROOT}/.env" ]; then
    echo "Loading API keys from .env"
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        # Strip surrounding quotes from value
        value="${value%\"}"
        value="${value#\"}"
        export "$key=$value"
    done < "${PROJECT_ROOT}/.env"
fi

# Run with GPU support and bind mounts
# --nv enables NVIDIA GPU support
# Standard HiPerGator paths: /blue, /orange, /home
$CONTAINER_CMD run \
    --nv \
    --bind "${PROJECT_ROOT}/logs:/app/logs" \
    --bind "${PROJECT_ROOT}/frontend/data:/app/frontend/data" \
    --bind "${HF_CACHE}:/app/.cache/huggingface" \
    --bind /blue:/blue \
    $EXTRA_BINDS \
    "$SIF_PATH"
