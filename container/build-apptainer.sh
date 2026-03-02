#!/bin/bash
# Build the Apptainer/Singularity container for HiPerGator
# Usage: ./container/build-apptainer.sh [output_path]
#
# This should be run on a system with sudo access or fakeroot enabled.
# On HiPerGator, you may need to build on a dev node or use --fakeroot.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEF_FILE="${PROJECT_ROOT}/agents-among-us.def"
OUTPUT_PATH="${1:-${PROJECT_ROOT}/agents-among-us.sif}"

echo "Building Apptainer container..."
echo "Definition: ${DEF_FILE}"
echo "Output: ${OUTPUT_PATH}"
echo ""

cd "$PROJECT_ROOT"

# Check if apptainer or singularity is available
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi

echo "Using: ${CONTAINER_CMD}"
echo ""

# Build with fakeroot if available, otherwise try sudo
if $CONTAINER_CMD build --help 2>&1 | grep -q -- '--fakeroot'; then
    echo "Building with --fakeroot..."
    $CONTAINER_CMD build --fakeroot "$OUTPUT_PATH" "$DEF_FILE"
else
    echo "Building with sudo..."
    sudo $CONTAINER_CMD build "$OUTPUT_PATH" "$DEF_FILE"
fi

echo ""
echo "Build complete: ${OUTPUT_PATH}"
echo ""
echo "To run on HiPerGator:"
echo "  apptainer run --nv --bind /blue:/blue ${OUTPUT_PATH}"
