# Agents Among Us

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-lightgrey.svg?logo=flask)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Interactive web demo and distributed inference framework for **Agents Among Us**, inspired by the popular mobile game. Observe the zero-shot social deduction and deception capabilities in Large Language Models (LLMs).

Despite their impressive capabilities, LLMs struggle to effectively utilize textual evidence in this social deduction environment. This framework lets you explore these limitations firsthand by comparing generative agent decisions against supervised machine learning "observers" trained on the exact same interaction logs.

<div align="center">
  <img width="2553" height="1157" alt="image" src="https://github.com/user-attachments/assets/2b0414de-91e2-47ff-922b-0b14d82536cf" />
  <br>
  <em>Live gameplay simulation interface</em>
</div>



## Links

- [Watch the system demonstration video here](https://youtu.be/fQIflkO-zg4)
- [Access live demo website here]()
- [UF Data Studio](https://github.com/ufdatastudio)
## Features

- **Interactive Simulation:** Flask-based UI with a live map, discussion chats, and real-time observer suspicion scores.
- **Distributed Architecture:** Decoupled asynchronous worker scripts supporting multi-GPU and multi-node execution via SLURM or Globus Compute.
- **Customizable Crews:** Configure homogeneous or heterogeneous crews of open-weight LLMs with specific roles (Honest or Byzantine).
- **Observer Pipeline:** Built-in classical ML classifier suite to calculate the "observer-player gap" on generated datasets.
- **Phase-Dependent Memory:** Automated state management that filters context windows based on movement, discussion, and voting phases.

## Installation

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
# Base install (API-only, no GPU)
uv sync --extra api

# Full install with GPU inference
uv sync --extra gpu --extra api
```

### Container Installation (Podman/Apptainer)

Both Podman and Apptainer/Singularity are supported. Two container variants are available:

**Full container** (11 GB, requires NVIDIA GPU):
```bash
./container/build-apptainer.sh     # Apptainer
./container/build-podman.sh        # Podman
```

**Navigator-only container** (~750 MB, no GPU required):
```bash
./container/build-navigator.sh
```

The container writes game configs and live state to `logs/`, which must be writable. The run scripts bind-mount `./logs` to `/app/logs` automatically. For manual Apptainer runs, include the bind:
```bash
# Full (GPU)
apptainer run --nv --bind ./logs:/app/logs agents-among-us.sif

# Navigator-only (no GPU)
apptainer run --bind ./logs:/app/logs agents-among-us-navigator.sif
```

To use API models inside the full container, place a `.env` file in the project root before running. The run scripts (`run-podman.sh`, `run-apptainer.sh`) load it automatically.

## APP_MODE

The `APP_MODE` environment variable controls which features are available at runtime. This lets the same codebase run in GPU-rich clusters, API-only servers, or lightweight PubApps VMs.

| Feature | `full` (default) | `api` | `navigator` |
|---------|-------------------|-------|-------------|
| Local GPU models | Yes | No | No |
| Navigator API | Yes | Yes | Yes |
| Anthropic/OpenAI API | Yes | Yes | No |
| Load `.env` | Yes | No | No |
| Requires CUDA/torch | Yes | No | No |

```bash
# Navigator-only (UF PubApps)
APP_MODE=navigator uv run python -m frontend.app

# All API providers, no local models
APP_MODE=api uv run python -m frontend.app

# Full mode with GPU (default)
uv run python -m frontend.app
```

## Usage

You can run the framework in four primary modes depending on your compute environment.

### 1. Interactive Web UI (Local/Hosted)

To run the Flask frontend on your local machine or a virtual machine:

```bash
uv run python -m frontend.app
```

The app binds to port `8080`. Open `http://localhost:8080` in your browser. Launching games from the UI automatically spawns the backend controller processes.

### 2. High-Performance Cluster (SLURM)

For large-scale dataset generation across compute nodes, this framework provides bash scripts configured for SLURM workload managers.

**Step 1: Configure the Controller (`main.py`)**
When spawning decoupled inference workers via SLURM, the main game controller (`main.py`) does not need GPU access. To prevent it from attempting to reserve GPU memory, open `main.py` and ensure `CUDA_VISIBLE_DEVICES` is set to `-1`:

```python
# In main.py
os.environ["LLM_MODE"] = "LOCAL"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Enable this for distributed GPU runs

```

**Step 2: Define Target Compositions**
Open `submit_all.sh` and populate the `compositions` array with the exact names of the model matchups you wish to run. These names must match the configurations generated by `config/model_composition.py`.

```bash
compositions=(
    "Llama3_Apertus"
    "Qwen2.5_Hermes4"
)

```

**Step 3: Execute the Batch Submitter**
Run the wrapper script from the repository root:

```bash
bash submit_all.sh

```

**Bash Scripts:**

* **`submit_all.sh`**: Iterates through defined `compositions`. For each entry, it submits a SLURM job via `sbatch`, passing the composition name as an environment variable (`--export=ALL,COMP_NAME="$comp"`).
* **`submit_games.sh`**: The core distributed job script. Uses `uv` for Python execution (provided by the `conda` module on HiPerGator).
* **Allocation:** Requests SLURM resources (e.g., `--nodes=1`, `--gpus-per-node=1`, `--mem=120gb`).
* **Worker Spawning:** Calls `config/generate_batch_list.py` to determine which models are required for the `COMP_NAME`. It then uses `srun` to asynchronously launch `worker.py` instances pinned to the allocated GPUs (can be specified in submit_games.sh)
* **Synchronization:** Genearates and polls the `logs/<COMP_NAME>/.../ipc` directory for `ready_*.signal` files. LLMs are fully loaded into VRAM before the game engine starts.
* **Execution:** Once workers signal readiness, the script iteratively runs `main.py` to orchestrate the games, utilizing file-based IPC to request text generation from the background workers.



### 3. Globus Compute (Remote Endpoint)

[Globus Compute](https://globus-compute.readthedocs.io/) provides a managed alternative to the SLURM file-based IPC approach. The game controller submits inference tasks to a Globus Compute endpoint, which provisions GPU workers automatically. This removes the need for `worker.py`, signal files, and manual SLURM coordination.

**Step 1: Install and configure the endpoint**

On the compute system (e.g., HiPerGator login node):

```bash
python3 -m pipx install globus-compute-endpoint
globus-compute-endpoint configure agents-among-us
```

Edit `~/.globus_compute/agents-among-us/config.yaml` to match your cluster. The repository includes a configuration targeting HiPerGator's `hpg-b200` partition with one GPU per worker.

**Step 2: Start the endpoint**

```bash
globus-compute-endpoint start agents-among-us
```

Authenticate via the browser URL on first run. Once running, `globus-compute-endpoint list` shows the endpoint UUID.

**Step 3: Run games**

```bash
# Direct execution
COMP_NAME="Llama3_Gemma" \
GLOBUS_COMPUTE_ENDPOINT="<your-endpoint-uuid>" \
bash submit_globus.sh

# Or via SLURM (CPU-only orchestration node)
sbatch --export=ALL,COMP_NAME="Llama3_Gemma",GLOBUS_COMPUTE_ENDPOINT="<uuid>" submit_globus.sh
```

The `submit_globus.sh` script sets `LLM_MODE=GLOBUS` and requests only CPU resources (2 CPUs, 8 GB RAM) since the endpoint handles GPU provisioning. Each `generate()` call in the game engine submits a task to the endpoint and blocks until the result is returned.

### 4. Manual Headless Execution

For local debugging or running specific configurations without the UI or SLURM:

```bash
# Start the inference worker in Terminal 1:
uv run python worker.py --game_id SESSION1 \
    --model_names meta-llama/Llama-3.1-8B-Instruct --comp_name MyComp

# Run the game controller in Terminal 2:
uv run python main.py --composition_name MyComp \
    --job_index 0 --game_id SESSION1 --num_rounds 10
```

### 5. HiPerGator PubApps Deployment

For hosting on UF Research Computing's [PubApps](https://docs.rc.ufl.edu/services/web_hosting/) infrastructure. PubApps VMs do not have GPUs, so use the lightweight navigator container.

**Prerequisites:**
1. Open a support ticket with UF Research Computing to request a PubApps instance
2. Specify your resource requirements (CPU, memory)
3. Obtain your assigned port number and VM access

**Deployment Steps:**

Once you have SSH access to your PubApps VM:

```bash
# Clone the repository
git clone https://github.com/ufdatastudio/agents-among-us.git
cd agents-among-us

# Build the navigator container (~750 MB, no GPU)
./container/build-navigator.sh

# Run the deployment script with your assigned port
./container/pubapps-deploy.sh --port <YOUR_PORT>
```

For GPU-enabled deployments on allocated nodes, the full container and `--gpu` flag are available:
```bash
./container/pubapps-deploy.sh --port <YOUR_PORT> --gpu
```

**Service Management:**

```bash
./container/pubapps-deploy.sh status     # Check service status
./container/pubapps-deploy.sh logs       # View container logs
./container/pubapps-deploy.sh restart    # Restart the service
./container/pubapps-deploy.sh stop       # Stop the service
./container/pubapps-deploy.sh uninstall  # Remove the deployment
```

Your application will be accessible at `https://<your-project>.rc.ufl.edu` after the reverse proxy is configured by RC support.

## API Model Support

The framework supports external API providers alongside local GPU inference. API models use a `provider:model_id` naming convention (e.g., `navigator:llama-3.3-70b-instruct`). This enables running games on machines without GPUs or comparing proprietary models against open-weight models.

### Supported Providers

| Provider | Prefix | Example Model ID |
|----------|--------|-----------------|
| UF Navigator | `navigator:` | `navigator:llama-3.3-70b-instruct` |
| OpenAI | `openai:` | `openai:gpt-4o-mini` |
| Anthropic | `anthropic:` | `anthropic:claude-sonnet-4-20250514` |

UF Navigator provides free API credits ($25/month) and routes to models from multiple vendors (Llama, Gemma, Claude, GPT-4o, Gemini).

### Setup

1. Copy the example environment file and add your keys:
```bash
cp .env.example .env
```

Edit `.env` with your API keys:
```
NAVIGATOR_TOOLKIT_API_KEY=your-navigator-key
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key
```

Only the keys for providers you plan to use are required. Navigator keys are available at [UF AI Navigator](https://ai.it.ufl.edu).

2. Start the web UI:
```bash
uv run python -m frontend.app
```

3. In the configuration page, select API models from the dropdown for any agent. The model list is divided into local models (top) and API models grouped by provider (bottom). If a key is detected from `.env`, a green "(.env set)" indicator appears next to the corresponding field. In `navigator` or `api` mode, only the relevant models and key fields are shown.

### Mixed Games

API and local models can coexist in the same game. For example, you could configure Agent 0 as a Byzantine running `navigator:gpt-4o` and Agents 1-3 as Honest running `google/gemma-2-9b-it` locally. The game engine routes each agent's generation call to the appropriate backend.

### Token Usage

API token consumption is tracked per model and displayed in the game UI next to each API agent's model name (e.g., `Navigator/gpt-4o (1532t)`). Token counts are also exported to `stats.csv` as `api_input_tokens` and `api_output_tokens` columns.

## Configuration & Adding New Models

The framework is model-agnostic and relies on Hugging Face transformers. You can test your own models or community fine-tunes by modifying the backend configurations.

**Adding Custom Models:**
To add a new model, simply add its Hugging Face repository ID to `config/model_composition.py`.

1. Define the model path:
```python
MY_CUSTOM_MODEL = "username/my-custom-llm-7B"

```


2. Add it to the appropriate weight class dictionary (`HW_MODELS_MAP` for Heavyweight, `LW_MODELS_MAP` for Lightweight):
```python
LW_MODELS_MAP = {
    "Llama3": LLAMA3_8B,
    "MyCustom": MY_CUSTOM_MODEL,
}

```

## Dataset

The preprocessed dataset containing over 10,000 parsed game logs and approximately 290,000 labeled utterances used for the offline observer pipeline is located at:

`results/observer_dataset.csv`

## Code Structure

```text
.
├── main.py                        # Orchestrates game runs (movement → discussion → voting)
├── worker.py                      # Async decoupled inference worker (SLURM IPC)
├── pyproject.toml                 # Project config with api/gpu optional dependencies
├── .env.example                   # Template for API key environment variables
├── submit_games.sh                # SLURM job script for cluster execution
├── submit_globus.sh               # Globus Compute launcher (CPU-only orchestration)
├── Dockerfile                     # Podman/Docker container definition
├── agents-among-us.def            # Apptainer container definition (full, GPU)
├── agents-among-us-navigator.def  # Apptainer container definition (navigator, no GPU)
├── agents/                        # Agent behavior definitions and prompts
├── config/                        # Game settings, model compositions, and APP_MODE config
├── container/                     # Container build/run scripts and PubApps deployment
├── core/                          # Core simulation logic, state management, API clients, and stopwords
├── frontend/                      # Flask application and UI assets
└── results/                       # Data analysis, classifiers, and parsed datasets
```

| Module | Description |
| --- | --- |
| `main.py` | Controller that manages phase transitions and evaluates win conditions. |
| `worker.py` | Loads models, handles local quantization, and processes generation requests via SLURM IPC. |
| `submit_globus.sh` | Launcher for Globus Compute mode (CPU-only orchestration, endpoint handles GPU). |
| `agents/` | Contains `honest_agent.py` and `byzantine_agent.py` with role-specific logic. |
| `config/` | Includes `app_mode.py`, `cache_models.py`, `settings.py`, etc. |
| `container/` | Podman and Apptainer build scripts, PubApps deployment automation. |
| `core/` | Includes `game_engine.py`, `state.py`, `llm.py`, `api_clients.py`, `globus_compute.py`, and `stopwords.py`. |
| `frontend/` | Flask routes (`app.py`), HTML templates, and live state visualizers. |
| `results/` | Machine learning pipeline for the offline observers and metric calculation. |

## Paper

This demo is based on the system demonstration paper:

**Agents Among Us: Identifying Deceptive Agents in a Multi-Agent Social Deduction Environment**

*Under Review: ACL 2026 System Demonstrations*

## Citation

If you use this framework or dataset in your research, please cite:

```bibtex
@inproceedings{agentsamongus2026,
    title = "Agents Among Us: Identifying Deceptive Agents in a Multi-Agent Social Deduction Environment",
    author = "Anonymous",
    booktitle = "",
    year = "2026",
    publisher = ""
}

```

*(Note: Citation details will be updated upon publication to reflect the full author list).*

## Issues and Contributions

Found a bug or have a feature request?
Please [open an issue](https://www.google.com/search?q=https://github.com/ufdatastudio/agents-among-us/issues) on GitHub.

Contributions to extend the game map, add new roles, or optimize inference workflows are always welcome!
