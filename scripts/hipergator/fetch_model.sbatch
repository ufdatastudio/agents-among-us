#!/bin/bash
#SBATCH --job-name=sr-fetch
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=03:00:00
#SBATCH --output=logs/sr-fetch-%j.out
# Download the weights of an eligible model into the Hugging Face cache on /blue, so that the GPU jobs, which run
# offline, find them. A CPU job does this to keep a GPU idle for no reason.
# Usage: mkdir -p logs && sbatch scripts/hipergator/fetch_model.sbatch <navigator name or checkpoint id>
# Submit from the repository root. The model must pass config/eligible_models.py.
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/scripts/hipergator/env.sh"
MODEL=$(uv run --no-project python -m config.eligible_models "${1:?name a model}")
export HF_HUB_OFFLINE=0
echo "fetching ${MODEL} into ${HF_HOME}"
uv run --no-project --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('${MODEL}'))
"
du -sh "${HF_HOME}/hub/models--${MODEL//\//--}"
