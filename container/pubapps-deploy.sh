#!/bin/bash
# HiPerGator PUBAPPS Deployment Script
#
# This script is designed for deployment on UF HiPerGator PUBAPPS infrastructure.
# See: https://docs.rc.ufl.edu/services/web_hosting/deployment/
#
# Before running this script:
# 1. Open a support ticket with UF Research Computing
# 2. Discuss your deployment requirements (access level, resources, port)
# 3. Get your assigned port number and VM details
#
# Usage: ./container/pubapps-deploy.sh --port <PORT>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SIF_PATH="${PROJECT_ROOT}/agents-among-us.sif"
PORT=""
MODE="podman"  # or "apptainer"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --sif)
            SIF_PATH="$2"
            MODE="apptainer"
            shift 2
            ;;
        --podman)
            MODE="podman"
            shift
            ;;
        --apptainer)
            MODE="apptainer"
            shift
            ;;
        --help|-h)
            echo "HiPerGator PUBAPPS Deployment"
            echo ""
            echo "Usage: $0 --port <PORT> [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT      Required. The port assigned by RC support."
            echo "  --podman         Use Podman (default)"
            echo "  --apptainer      Use Apptainer/Singularity"
            echo "  --sif PATH       Path to .sif file (implies --apptainer)"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate port
if [ -z "$PORT" ]; then
    echo "Error: --port is required"
    echo "Run with --help for usage information"
    exit 1
fi

echo "=============================================="
echo "HiPerGator PUBAPPS Deployment"
echo "=============================================="
echo "Mode: ${MODE}"
echo "Port: ${PORT}"
echo "Project: ${PROJECT_ROOT}"
echo ""

# Create required directories
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/frontend/data"

# Set up HuggingFace cache
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

if [ "$MODE" = "apptainer" ]; then
    # Apptainer deployment
    if [ ! -f "$SIF_PATH" ]; then
        echo "Building Apptainer container..."
        "${SCRIPT_DIR}/build-apptainer.sh" "$SIF_PATH"
    fi

    if command -v apptainer &> /dev/null; then
        CONTAINER_CMD="apptainer"
    elif command -v singularity &> /dev/null; then
        CONTAINER_CMD="singularity"
    else
        echo "Error: Neither apptainer nor singularity found"
        exit 1
    fi

    echo "Starting with Apptainer..."
    exec $CONTAINER_CMD run \
        --nv \
        --env PORT="${PORT}" \
        --bind "${PROJECT_ROOT}/logs:/app/logs" \
        --bind "${PROJECT_ROOT}/frontend/data:/app/frontend/data" \
        --bind "${HF_CACHE}:/app/.cache/huggingface" \
        "$SIF_PATH"

else
    # Podman deployment
    IMAGE_NAME="agents-among-us:latest"

    # Check if image exists, build if not
    if ! podman image exists "$IMAGE_NAME" 2>/dev/null; then
        echo "Building Podman image..."
        "${SCRIPT_DIR}/build-podman.sh"
    fi

    echo "Starting with Podman..."
    exec podman run --rm \
        --device nvidia.com/gpu=all \
        -e PORT="${PORT}" \
        -p "${PORT}:${PORT}" \
        -v "${PROJECT_ROOT}/logs:/app/logs:z" \
        -v "${PROJECT_ROOT}/frontend/data:/app/frontend/data:z" \
        -v "${HF_CACHE}:/app/.cache/huggingface:z" \
        --name agents-among-us \
        "$IMAGE_NAME"
fi
