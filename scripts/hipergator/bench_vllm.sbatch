#!/bin/bash
#SBATCH --job-name=sr-bench
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=00:25:00
#SBATCH --output=logs/sr-bench-%j.out
# Start a vLLM server on one GPU, run the concurrency benchmark against it, and stop the server.
# The job owns its server, so nothing stays running after it ends.
# Usage: [MAX_TOKENS=160] [REASONING_EFFORTS="default low"] sbatch scripts/hipergator/bench_vllm.sbatch [navigator model name or hf checkpoint id]
# MAX_TOKENS is the completion limit. REASONING_EFFORTS lists efforts for a gpt-oss model, one benchmark each.
# Submit from the repository root. Only models on the eligible list in config/eligible_models.py
# start. The check exits with an error for Qwen, DeepSeek and other families from countries of concern.
# A Qwen model starts only when QWEN_OVERRIDE holds the phrase the check prints in its warning,
# as in: QWEN_OVERRIDE="<phrase>" sbatch scripts/hipergator/bench_vllm.sbatch Qwen/Qwen2.5-7B-Instruct
# Confirm with Research Computing first. The override covers no other family.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
MODEL=$(uv run --no-project python -m config.eligible_models "${1:-granite-3.3-8b-instruct}" ${QWEN_OVERRIDE:+--override "${QWEN_OVERRIDE}"})
# The weights come from the cache on /blue.
export HF_HUB_OFFLINE=1
PORT=$((20000 + SLURM_JOB_ID % 20000))
SAFE_NAME=${MODEL//\//_}
mkdir -p logs results/bench
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

# One benchmark per reasoning effort. "default" sends no effort, so the model uses its own default.
for EFFORT in ${REASONING_EFFORTS:-default}; do
  EXTRA=()
  SUFFIX=""
  if [ "${EFFORT}" != "default" ]; then EXTRA=(--reasoning-effort "${EFFORT}"); SUFFIX="_${EFFORT}"; fi
  uv run --no-project --with openai --with loguru python -m scripts.bench_vllm_concurrency \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" --max-tokens "${MAX_TOKENS:-160}" \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --out "results/bench/${SAFE_NAME}_${SLURM_JOB_ID}${SUFFIX}.json"
done
