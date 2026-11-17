#!/bin/bash
#SBATCH --job-name=sr-duplex
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=01:00:00
#SBATCH --output=logs/sr-duplex-%j.out
# Plays one short meeting in which agents listen and speak through streaming sessions of an in-process vLLM engine.
# See scripts/duplex_spike.py and core/duplex.py. The results go to results/duplex/<name>/.
# Usage: [OUT=results/duplex/gemma] [AGENTS=6] [BUDGET=90] [BEAT=2.0] [SEED=1] [TEMPERATURE=0.8] [STATEMENT_WORDS=30] \
#        [VLLM_WITH=vllm] [ENGINE_ARGS=json] [SPIKE_ARGS="--replay ... --game ... --listener Agent_3 --audio"] \
#        sbatch scripts/hipergator/duplex_spike.sbatch [model]
# ENGINE_ARGS is a JSON object of more engine arguments. Omni needs {"trust_remote_code": true, "moe_backend": "triton",
# "limit_mm_per_prompt": {"audio": 2000}} and VLLM_WITH="vllm[audio]". SPIKE_ARGS is split on spaces.
# Submit from the repository root or a copy of config, core, scripts and agents. Only models on the eligible list start.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
export HF_HUB_OFFLINE=1
mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODEL=${1:-gemma-4-31b-it}
uv run --no-project --with "${VLLM_WITH:-vllm}" --with loguru python -m scripts.duplex_spike \
  --model "${MODEL}" --out "${OUT:-results/duplex/${SLURM_JOB_ID}}" \
  --agents "${AGENTS:-6}" --budget "${BUDGET:-90}" --beat "${BEAT:-2.0}" --seed "${SEED:-1}" \
  --temperature "${TEMPERATURE:-0.8}" --statement-words "${STATEMENT_WORDS:-30}" \
  ${ENGINE_ARGS:+--engine-args "${ENGINE_ARGS}"} ${SPIKE_ARGS:-}
