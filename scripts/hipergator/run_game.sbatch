#!/bin/bash
#SBATCH --job-name=sr-game
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=01:30:00
#SBATCH --output=logs/sr-game-%j.out
# Start a vLLM server on one GPU, play full games against it, and time each game.
# The job owns its server, so nothing stays running after it ends.
# Usage: [MODES="timed turns"] [GAMES=1] [ROUNDS=3] [CAPTURE=false] sbatch scripts/hipergator/run_game.sbatch [model]
# Submit from the repository root. Only models on the eligible list in config/eligible_models.py start,
# and the engine checks every vllm: model again when it sets up the game. There is no Qwen override here.
# MODES lists the discussion modes to time, each GAMES times. ROOM_BUDGET_S, ROOM_TICK_S and
# ROOM_OVERLAP set the room. CAPTURE=true makes every agent write private reasoning first.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
MODEL=$(uv run --no-project python -m config.eligible_models "${1:-granite-3.3-8b-instruct}")
export HF_HUB_OFFLINE=1
PORT=$((20000 + SLURM_JOB_ID % 20000))
SAFE_NAME=${MODEL//\//_}
COMP="room_${SAFE_NAME}"
mkdir -p logs logs/game_configs results/room_runs
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

# One composition with 8 honest and 2 Byzantine agents, all on the served model.
uv run --no-project python - "${COMP}" "vllm:${MODEL}" "${CAPTURE:-false}" <<'PY'
import json
import sys

name, model, capture = sys.argv[1], sys.argv[2], sys.argv[3] == "true"
composition = {
    "name": name,
    "honest_model": [model],
    "byzantine_model": [model],
    "honest_count": 8,
    "byzantine_count": 2,
    "capture_thoughts": capture,
}
with open(f"logs/game_configs/{name}.json", "w", encoding="utf-8") as handle:
    json.dump(composition, handle)
PY

# APP_MODE=api loads no .env file and imports no torch, and the game only talks to the local server.
# LLM_MODE=LOCAL keeps main.py from defaulting to CONTROLLER on Linux. A vllm: model never uses either.
export APP_MODE=api
export LLM_MODE=LOCAL
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
for MODE in ${MODES:-timed turns}; do
  for ((g = 1; g <= ${GAMES:-1}; g++)); do
    GAME_ID="${SLURM_JOB_ID}_${MODE}_${g}"
    START=${SECONDS}
    uv run --extra api python -m main \
      --composition_name "${COMP}" --game_id "${GAME_ID}" --num_rounds "${ROUNDS:-3}" \
      --discussion_mode "${MODE}" --room_budget_s "${ROOM_BUDGET_S:-180}" --room_tick_s "${ROOM_TICK_S:-10}" \
      --room_overlap "${ROOM_OVERLAP:-clean}" --room_workers "${ROOM_WORKERS:-10}" \
      > "logs/game-${GAME_ID}.out" 2>&1 || echo "game ${GAME_ID} exited with an error, see logs/game-${GAME_ID}.out"
    ELAPSED=$((SECONDS - START))
    echo "game ${GAME_ID} mode=${MODE} rounds=${ROUNDS:-3} took ${ELAPSED} s"
    echo "{\"game\": \"${GAME_ID}\", \"mode\": \"${MODE}\", \"model\": \"${MODEL}\", \"rounds\": ${ROUNDS:-3}, \"seconds\": ${ELAPSED}}" >> "results/room_runs/${SAFE_NAME}_${SLURM_JOB_ID}.jsonl"
  done
done
