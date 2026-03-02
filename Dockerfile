# Dockerfile for Agents Among Us
# Compatible with Docker and Podman
# Requires NVIDIA GPU support for LLM inference
# Uses uv for Python and package management

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

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

# Install Python 3.10 via uv and create venv
RUN uv python install 3.10 && uv venv --python 3.10

# Install PyTorch with CUDA 12.4 support
RUN uv pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install core ML dependencies
RUN uv pip install \
    transformers==4.47.0 \
    accelerate==1.2.1 \
    bitsandbytes==0.45.0 \
    safetensors==0.4.5 \
    sentencepiece==0.2.0 \
    tokenizers==0.21.0 \
    huggingface-hub==0.27.0 \
    datasets==3.2.0 \
    peft==0.14.0

# Install web and data processing dependencies
RUN uv pip install \
    flask==3.1.0 \
    pandas==2.2.3 \
    numpy==2.0.2 \
    scikit-learn==1.6.0 \
    xgboost==2.1.3 \
    joblib==1.4.2 \
    nltk==3.9.1 \
    matplotlib==3.9.3 \
    pillow==11.0.0 \
    tqdm==4.67.1 \
    pyyaml==6.0.2 \
    loguru==0.7.3

# Download NLTK data
RUN uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"

# Production image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy application code
COPY . /app

# Create necessary directories
RUN mkdir -p /app/logs /app/frontend/data /app/config/game_configs

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
