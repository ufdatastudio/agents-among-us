#!/bin/bash
#SBATCH --job-name=sr-diar
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=00:40:00
#SBATCH --output=logs/sr-diar-%j.out
# Diarization pilot (P1c): Kokoro speech, per-reader mixes, Nemotron 3 Diarization labels scored against the true segments.
# The model and Kokoro are cached on /blue. See scripts/diarization_pilot.py.
# Usage: sbatch scripts/hipergator/diarization_pilot.sbatch
# Submit from the repository root.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p logs results/diarization_pilot
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-project --python 3.12 --with kokoro --with soundfile --with transformers --with torch \
  --with scipy --with numpy --with loguru --with librosa \
  --with "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  python -m scripts.diarization_pilot \
  --out "results/diarization_pilot/summary_${SLURM_JOB_ID}.json" \
  --wav-dir "results/diarization_pilot/wav_${SLURM_JOB_ID}"
