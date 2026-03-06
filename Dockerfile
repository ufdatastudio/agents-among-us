# Dockerfile for Agents Among Us
# Compatible with Docker and Podman
# Requires NVIDIA GPU support for LLM inference
# Uses uv for Python and package management

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install minimal system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set up working directory
WORKDIR /app

# Install Python 3.10 via uv
RUN uv python install 3.10

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock /app/

# Install all dependencies via uv sync
RUN uv sync --extra gpu --extra api --no-install-project

# Download all NLTK data to a shared location
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p $NLTK_DATA && \
    uv run python -c "import nltk; nltk.download('all', download_dir='$NLTK_DATA')"

# Production image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install curl for health checks and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv in production image
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy uv cache and venv from builder
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=builder /app/.venv /app/.venv

# Copy NLTK data
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy application code
COPY . /app

# Create necessary directories (game_configs written to logs/ at runtime)
RUN mkdir -p /app/logs /app/logs/game_configs /app/frontend/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV LLM_MODE=LOCAL
ENV PORT=8080

# Create startup script that supports configurable port
RUN echo '#!/bin/bash\n\
cd /app\n\
PORT="${PORT:-8080}"\n\
echo "Starting Agents Among Us on port ${PORT}..."\n\
exec uv run python -c "\n\
import sys\n\
sys.path.insert(0, \"/app\")\n\
from frontend.app import app\n\
import os\n\
port = int(os.environ.get(\"PORT\", 8080))\n\
app.run(host=\"0.0.0.0\", port=port, debug=False)\n\
"\n\
' > /app/start.sh && chmod +x /app/start.sh

# Expose default Flask port (can be overridden via PORT env var)
EXPOSE 8080

# Health check uses PORT env var
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/health || exit 1

# Run startup script
ENTRYPOINT ["/app/start.sh"]
