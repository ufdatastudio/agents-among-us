#!/bin/bash
# submit_globus.sh
# Launches games using Globus Compute for remote GPU inference.
# No SLURM GPU allocation needed on the submitting node — Globus Compute
# provisions workers on the endpoint automatically.
#
# Usage:
#   COMP_NAME="Llama3_Gemma" GLOBUS_COMPUTE_ENDPOINT="<your-uuid>" bash submit_globus.sh
#
# Or with SLURM (CPU-only, for orchestration):
#   sbatch --export=ALL,COMP_NAME="Llama3_Gemma",GLOBUS_COMPUTE_ENDPOINT="<uuid>" submit_globus.sh

#SBATCH --job-name=amongus_globus
#SBATCH --output=job_logs/globus_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --partition=hpg-default

if [ -z "$COMP_NAME" ]; then
    echo "Error: COMP_NAME is not set."
    exit 1
fi

if [ -z "$GLOBUS_COMPUTE_ENDPOINT" ]; then
    echo "Error: GLOBUS_COMPUTE_ENDPOINT is not set."
    echo "Run 'globus-compute-endpoint list' to find your endpoint UUID."
    exit 1
fi

export LLM_MODE="GLOBUS"
export GLOBUS_COMPUTE_ENDPOINT

JOB_ID=${SLURM_ARRAY_JOB_ID:-$$}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SESSION_ID="Session_${JOB_ID}_${TASK_ID}_${COMP_NAME}"

# uv is bundled with the conda module on HiPerGator
module load conda

GAMES_PER_JOB=2

echo "Running Composition: $COMP_NAME via Globus Compute"
echo "Endpoint: $GLOBUS_COMPUTE_ENDPOINT"

START_INDEX=$(( TASK_ID * GAMES_PER_JOB ))

for (( j=0; j<GAMES_PER_JOB; j++ )); do
    CURRENT_GAME_IDX=$(( START_INDEX + j ))

    echo ">>> Starting Game $j (Global Index $CURRENT_GAME_IDX)"

    uv run -m main \
        --composition_name "$COMP_NAME" \
        --job_index $CURRENT_GAME_IDX \
        --game_id "$SESSION_ID"

    echo "Game $j finished."
done
