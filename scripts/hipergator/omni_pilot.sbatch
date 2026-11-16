#!/bin/bash
#SBATCH --job-name=sr-omni-pilot
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/sr-omni-pilot-%j.out
# Native-ear pilot: Kokoro speaks the scripted statements of the Whisper pilot, and Nemotron Omni listens to the clean
# clips and to the overlap mixes through a vLLM server that this job starts. See scripts/omni_pilot.py.
# The weights must be in the cache already, so run scripts/hipergator/fetch_model.sbatch first.
# Usage: mkdir -p logs && [REPS=10] [LEVELS=1,2,3,5,10] sbatch scripts/hipergator/omni_pilot.sbatch
# Submit from the repository root.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
MODEL=$(uv run --no-project python -m config.eligible_models nemotron-3-nano-omni)
export HF_HUB_OFFLINE=1
PORT=$((20000 + SLURM_JOB_ID % 20000))
OUT="results/omni_pilot"
mkdir -p logs "${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The clean clips, with the voices and the statements of the Whisper pilot.
uv run --no-project --python 3.12 --with kokoro --with soundfile --with transformers --with torch \
  --with scipy --with numpy --with loguru \
  --with "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
  python -m scripts.omni_pilot clips --out "${OUT}/clips_${SLURM_JOB_ID}" --device cuda

# The settings of the model card for an RTX Pro: trust-remote-code for its own code, the triton MoE backend
# because of a FlashInfer bug on that card, and the reasoning parser that the card gives.
uv run --no-project --with "vllm[audio]" --with loguru vllm serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --max-model-len 32768 --max-num-seqs 8 \
  --trust-remote-code --moe-backend triton --reasoning-parser nemotron_v3 \
  --limit-mm-per-prompt '{"audio": 1}' \
  --served-model-name "${MODEL}" > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

# Wait up to 40 minutes for the first load, which reads 62 GB.
READY=0
for ((i = 0; i < 480; i++)); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then READY=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "server exited early"; tail -80 "logs/vllm-${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 5
done
if [ "${READY}" -ne 1 ]; then echo "server not ready after $((i * 5)) s"; tail -80 "logs/vllm-${SLURM_JOB_ID}.log"; exit 1; fi
echo "server ready on ${PORT} after about $((i * 5)) s"
grep -m3 -iE "vllm|version" "logs/vllm-${SLURM_JOB_ID}.log" || true

uv run --no-project --with "vllm[audio]" --with loguru python -m scripts.omni_pilot trials \
  --clips "${OUT}/clips_${SLURM_JOB_ID}" --url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" \
  --levels "${LEVELS:-1,2,3,5,10}" --reps "${REPS:-10}" --out "${OUT}/summary_${SLURM_JOB_ID}.json"
