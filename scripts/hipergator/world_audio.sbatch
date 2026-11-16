#!/bin/bash
#SBATCH --job-name=sr-voices
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000,hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=00:20:00
#SBATCH --output=logs/sr-voices-%j.out
# Voices for a recorded game: Kokoro speaks every statement of the game folder, one WAV file for each. See scripts/world_audio.py.
# The weights are cached on /blue. The folder of clips goes to scripts.make_world_replay --speech to give a video sound.
# Usage: mkdir -p logs && GAME=/path/to/Game_<id>_Run0 OUT=results/speech/<name> sbatch scripts/hipergator/world_audio.sbatch
# Submit from the repository root.
set -euo pipefail
: "${GAME:?set GAME to the game folder}"
: "${OUT:?set OUT to the folder for the clips}"
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-project --python 3.12 --with kokoro --with soundfile --with torch --with numpy --with loguru \
  --with "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  python -m scripts.world_audio "${GAME}" --out "${OUT}" --device cuda
