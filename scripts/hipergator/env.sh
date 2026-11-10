# Shared environment for HiPerGator jobs. Keeps model weights and wheel caches on /blue, not in the 40 GB home.
# Follows byzantine-agents/scripts/hipergator/env.sh. The Hugging Face token is read from $HF_HOME/token, and no .env file is sourced.
export SR_BLUE=/blue/ufdatastudios/christan
export HF_HOME=${SR_BLUE}/cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_CACHE_DIR=${SR_BLUE}/cache/uv
export VLLM_CACHE_ROOT=${SR_BLUE}/cache/vllm
export TORCH_HOME=${SR_BLUE}/cache/torch
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${TORCH_HOME}"
