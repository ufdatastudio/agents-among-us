#!/bin/bash
#SBATCH --job-name=sr-p2
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/sr-p2-%j.out
# Start a vLLM server on one GPU and play the P2 grid against it: 11 discussion protocols, each
# on the same scenario seeds, with several games at once. The job owns its server, so nothing
# stays running after it ends.
# Usage: [GAMES_PER_CELL=5] [ROUNDS=10] [WORKERS=6] [PROTOCOLS=sequential,clean-180] [TAG=...] \
#        sbatch scripts/hipergator/run_p2.sbatch [model]
# Submit from the repository root. Only models on the eligible list in config/eligible_models.py start,
# and the runner checks the model again. TAG names the output file results/p2/<model>_<TAG>.jsonl, and a
# job that ran out of time can be resubmitted with the same TAG to play only the games that are missing.
# For a gpt-oss model also set AAU_MAX_TOKENS (a larger reply limit) and VLLM_REASONING_EFFORT.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
MODEL=$(uv run --no-project python -m config.eligible_models "${1:-granite-3.3-8b-instruct}")
export HF_HUB_OFFLINE=1
PORT=$((20000 + SLURM_JOB_ID % 20000))
mkdir -p logs results/p2
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --no-project --with vllm vllm serve "${MODEL}" \
  --host 127.0.0.1 --port "${PORT}" \
  --max-model-len 16384 \
  --enable-prompt-tokens-details \
  --served-model-name "${MODEL}" > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

# Wait up to 15 minutes for the server to answer.
READY=0
for ((i = 0; i < 180; i++)); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null; then READY=1; break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "server exited early"; tail -60 "logs/vllm-${SLURM_JOB_ID}.log"; exit 1; fi
  sleep 5
done
if [ "${READY}" -ne 1 ]; then echo "server not ready after 15 minutes"; tail -60 "logs/vllm-${SLURM_JOB_ID}.log"; exit 1; fi
echo "server ready on ${PORT} after about $((i * 5)) s"

# APP_MODE=api loads no .env file and imports no torch, and the games only talk to the local server.
export APP_MODE=api
export LLM_MODE=LOCAL
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
TAG=${TAG:-${SLURM_JOB_ID}}
uv run --extra api python -m scripts.run_p2 \
  --model "${MODEL}" --tag "${TAG}" \
  --games-per-cell "${GAMES_PER_CELL:-5}" --rounds "${ROUNDS:-10}" --workers "${WORKERS:-6}" \
  ${PROTOCOLS:+--protocols "${PROTOCOLS}"}

SAFE_NAME=${MODEL//\//_}
uv run --extra api python -m scripts.summarize_p2 "results/p2/${SAFE_NAME}_${TAG}.jsonl"
