# Agents Among Us

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
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

- [Read the paper (IVA 2026)](https://doi.org/10.1145/3806774.3827986)
- [Watch the system demonstration video here](https://youtu.be/fQIflkO-zg4)
- [Access live demo website here]()
- [UF Data Studio](https://github.com/ufdatastudio)
## Features

- **Interactive Simulation:** Flask-based UI with a live map, discussion chats, and real-time observer suspicion scores.
- **Distributed Architecture:** Decoupled asynchronous worker scripts supporting multi-GPU and multi-node execution via SLURM or Globus Compute.
- **Customizable Crews:** Configure homogeneous or heterogeneous crews of open-weight LLMs with specific roles (Honest or Byzantine).
- **Observer Pipeline:** Built-in classical ML classifier suite to calculate the "observer-player gap" on generated datasets.
- **Phase-Dependent Memory:** Automated state management that filters context windows based on movement, discussion, and voting phases.

## Getting started

The `situation-room` and `godot-world` branches have the same code, and `godot-world` adds the plan and the tile map of the Godot world. This section goes from a test on your laptop to the experiment runs on HiPerGator. The web UI in [Usage](#usage) plays a game in a browser.

### Set up and test

```bash
git clone https://github.com/ufdatastudio/agents-among-us.git
cd agents-among-us
git checkout situation-room
uv sync --extra api
APP_MODE=navigator uv run --extra api --with pytest python -m pytest tests agents -q
```

The tests need no GPU and no key.

### Play one game with no GPU

UF Navigator serves the models of the experiments, and a key is at [ai.it.ufl.edu](https://ai.it.ufl.edu). A game reads its crew from `logs/game_configs/<name>.json`. This one has 4 honest agents and 1 Byzantine agent, all running Granite 3.3 8B.

```bash
mkdir -p logs/game_configs
cat > logs/game_configs/quickstart.json <<'EOF'
{"name": "quickstart", "honest_model": ["navigator:granite-3.3-8b-instruct"], "byzantine_model": ["navigator:granite-3.3-8b-instruct"], "honest_count": 4, "byzantine_count": 1, "capture_thoughts": false}
EOF
export NAVIGATOR_TOOLKIT_API_KEY=<your key>
APP_MODE=api LLM_MODE=LOCAL uv run --extra api python -m main --composition_name quickstart --game_id quick1 \
  --num_rounds 2 --seed 1 --room_seed 1 --discussion_mode timed --room_overlap interleaved --room_budget_s 60
```

`APP_MODE=api` reads no `.env` file, so the key comes from your shell. The game prints `Game Over. Result: ...` and writes its folder to `logs/quickstart/Game_quick1_Run0/`. That game took about a minute and used about 400,000 input tokens.

| File | Content |
|------|---------|
| `discussion.log` | The statements of each meeting in the order they were spoken |
| `room/` | For each meeting of a timed game, `room_round<N>.jsonl` with its events and `<agent>_round<N>_heard.log` with what each agent heard |
| `votes.jsonl` | The vote of each agent in each meeting |
| `roundResults.log` | Who was eliminated and who was ejected in each round |
| `stats.csv` | One row per agent, with the vote counts, the result and the token use |
| `thought.log` | Private reasoning, when `capture_thoughts` is true |

The main flags of `main.py` are in the table below. A seed fixes the scenario, so games with the same seed start every agent in the same room.

| Flag | Values | Meaning |
|------|--------|---------|
| `--discussion_mode` | `turns`, `timed` | Take turns, or speak at once in a timed room |
| `--room_overlap` | `clean`, `interleaved`, `noisy` | How overlapping speech is shown to a listener |
| `--room_budget_s` | seconds | Length of a timed meeting, 180 by default |
| `--room_max_words` | words | Words kept from one statement, 40 by default |
| `--num_rounds` | rounds | Rounds before the game ends, 10 by default |
| `--seed`, `--room_seed` | integers | Scenario seed, and the seed of the onsets and shuffles in the room |

Every agent has to vote, and a meeting with a missing vote is held again, up to 3 times (`--meeting_attempts`).

### Play against your own model server

Any OpenAI-compatible server works. `scripts/run_p2.py` plays the 11 protocols of the experiments, each on the same scenario seeds, against the server that `VLLM_BASE_URL` names. A model has to be on the eligible list in `config/eligible_models.py`, which lists the models that Navigator also serves and that do not come from a country of concern. The runner refuses the others.

```bash
uv run --no-project --with vllm vllm serve ibm-granite/granite-3.3-8b-instruct \
  --max-model-len 32768 --served-model-name ibm-granite/granite-3.3-8b-instruct
# In a second terminal:
APP_MODE=api LLM_MODE=LOCAL VLLM_BASE_URL=http://127.0.0.1:8000/v1 uv run --extra api python -m scripts.run_p2 \
  --model ibm-granite/granite-3.3-8b-instruct --tag try1 --games-per-cell 1 --protocols sequential,clean-60
```

The protocols are `sequential`, `none` (no discussion), and `clean`, `interleaved` and `noisy` at 60, 180 and 300 s. A game that is not finished in 3,600 s is stopped (`--game-timeout-s`). The runner appends each game to `results/p2/<model>_<tag>.jsonl`, and running it again with the same `--tag` plays only the games that did not finish.

### Run the experiments on HiPerGator

One job starts a vLLM server on 1 GPU and plays the grid against it. Replace the `/blue` folder, the account and the QoS with your own if you are not in the `ufdatastudios` group.

```bash
# Once. Set SR_BLUE in scripts/hipergator/env.sh to a folder of yours on /blue, then cache the weights on a login node.
source scripts/hipergator/env.sh
uv run --no-project --with huggingface_hub hf download ibm-granite/granite-3.3-8b-instruct

# For each run, start from a fresh checkout so that later edits do not change games that are still to start.
git fetch origin
git worktree add --detach .worktrees/run1 origin/situation-room
cd .worktrees/run1
uv run --extra api python -c pass
uv run --no-project python -m config.eligible_models granite-3.3-8b-instruct
TAG=run1-granite GAMES_PER_CELL=5 sbatch scripts/hipergator/run_p2.sbatch granite-3.3-8b-instruct
```

These variables set the run, and the script header lists the rest.

| Variable | Default | Meaning |
|----------|---------|---------|
| `TAG` | the job id | Name of the run and of its result file |
| `GAMES_PER_CELL` | 5 | Seeds for each protocol |
| `PROTOCOLS` | all 11 | Protocol names separated by commas, for a smaller test |
| `WORKERS` | 6 | Games played at once on the server |
| `AAU_MAX_TOKENS` | 160 | Reply limit in tokens. Gemma 4 needs 1024, and gpt-oss needs 768 with `VLLM_REASONING_EFFORT=low` and `--mem=128gb` |
| `CAPTURE` | off | Set it to 1 for agents to write private reasoning before they answer |
| `MAX_MODEL_LEN` | 32768 | Context of the server, which a 300 s meeting needs |

Follow a run with `squeue -u $USER` and `tail -f logs/sr-p2-<job id>.out`. After the first games finish, check `api_errors`, `redone_meetings` and `stopped_for_votes` in the result file. Games are written to `logs/p2_<model>_<tag>/`, and the job ends by printing a summary table, which `uv run python -m scripts.summarize_p2 results/p2/*.jsonl` prints again for any result files. If a job runs out of time or a game times out, submit the same command again with the same `TAG`. The group account and QoS are shared, so look at `squeue -A ufdatastudios` before you submit a large run.

`docs/situation-room.md` describes how the tables and figures of the results are made, and the replay below shows how to draw one meeting.

### Try the audio and the world map

The audio stack speaks each statement with Kokoro and listens with Whisper large-v3, and Nemotron 3 Diarization says who spoke when. Cache the models once with `hf download` as above (`hexgrad/Kokoro-82M`, `openai/whisper-large-v3` and `nvidia/Nemotron-3-Diarization`), then run the two pilots from the repository root.

```bash
sbatch scripts/hipergator/audio_pilot.sbatch        # speech recognition against overlap, in results/audio_pilot/
sbatch scripts/hipergator/diarization_pilot.sbatch  # diarization error rate against overlap, in results/diarization_pilot/
uv run python -m scripts.draw_world_map --out docs/media/godot-world-map.png   # the tile map of the Godot world
```

## Situation room

In the situation room, all agents of a meeting speak during the same timed window. Their speech overlaps, each agent hears only what was spoken before the current tick, and every agent has to vote before the meeting ends. The design, the P2 experiments and their results are in [docs/situation-room.md](docs/situation-room.md).

The replay below is drawn from the logs of one recorded game by `scripts/make_replay.py`. Nine gpt-oss-120b agents meet in a timed room with interleaved overlap and a 180 s budget (scenario seed 3, round 1), and the clock runs 2.5 times faster than the meeting did. The seats show the last words each agent spoke, the right panel shows what was said, and the lanes at the bottom show who spoke when. The roles are shown to you, and the agents were not told them.

<div align="center">
  <a href="docs/media/situation-room-replay.mp4"><img src="docs/media/situation-room-replay.gif" alt="Replay of one meeting in the situation room, where nine agents speak at once and then vote"></a>
  <br>
  <em>The first 24 s of the replay. The full 84 s video is <a href="docs/media/situation-room-replay.mp4">situation-room-replay.mp4</a>.</em>
</div>

<div align="center">
  <img src="docs/media/situation-room-replay-t002.png" alt="The meeting at 0:02, with all nine living agents starting to speak">
  <br>
  <em>0:02. Agent_5 was found dead and the nine living agents start to speak at once. The blue rings mark the seats that are speaking.</em>
</div>

<div align="center">
  <img src="docs/media/situation-room-replay-t062.png" alt="The meeting at 1:02, with overlapping statements that mostly accuse Agent_0">
  <br>
  <em>1:02. Most statements on screen accuse Agent_0. Agent_0 and Agent_1 are the Byzantine agents, marked with a B. The lanes show that the statements overlap.</em>
</div>

<div align="center">
  <img src="docs/media/situation-room-replay-votes.png" alt="The votes of the nine agents, drawn as arrows">
  <br>
  <em>After 3:00 each agent votes, and the arrows show who voted for whom. Agent_0 received 7 votes.</em>
</div>

<div align="center">
  <img src="docs/media/situation-room-replay-result.png" alt="The result of the vote, with Agent_0 ejected">
  <br>
  <em>Agent_0, a Byzantine agent, is ejected. The two Byzantine agents voted for others, Agent_0 for Agent_6 and Agent_1 for Agent_2.</em>
</div>

To make a replay of another meeting, point the script at a game folder from a run of `scripts/run_p2.py`:

```bash
uv run --with matplotlib python -m scripts.make_replay logs/p2_<model>/Game_<tag>_interleaved-180_3_Run0 --round 1 --out docs/media
```

The video needs `ffmpeg`. Add `--no-video` for the still frames only, and `--stills 30,90,votes,result` to choose them.

## Godot world

The plan for a 2D top-down world in Godot, where agents walk through rooms, hear speech within walls and meet in the situation room, is in [docs/godot-world.md](docs/godot-world.md). No Godot project exists yet. The first piece of code is the tile map, which `core/world_map.py` builds and `scripts/draw_world_map.py` draws.

<div align="center">
  <img src="docs/media/godot-world-map.png" alt="The tile map of the first world, with 14 rooms laid out like the ship of the simulation, joined by corridors with walls and doors">
  <br>
  <em>The first map is 80 by 45 tiles. The 14 rooms are laid out like the ship in <code>frontend/static/images/Skeld.png</code>, the amber bars mark the doors, and the walls are one layer that blocks bodies, views and voices.</em>
</div>

The corridors and doors give each room the same neighbours as `ROOMS` in `config/settings.py`, and a test checks it. To draw the map again:

```bash
uv run python -m scripts.draw_world_map --out docs/media/godot-world-map.png
```

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

For hosting on UF Research Computing's [PubApps](https://docs.rc.ufl.edu/services/web_hosting/) infrastructure. The live demo at **[agents-among-us.rc.ufl.edu](https://agents-among-us.rc.ufl.edu)** uses the navigator-only Podman flow (5a) on `pubufdatastudios1`. The original Apptainer/Globus flow (5b) remains supported for GPU-enabled allocations.

#### 5a. Navigator-only (current live deployment)

A rootless Podman container managed by a systemd Quadlet unit. Build from `Containerfile.navigator`, operate via `container/pubapps-navigator-deploy.sh`. No GPU, no baked-in `.env`; visitors enter their own UF Navigator API key in the UI.

**Quick deploy** on a freshly provisioned PubApps VM:

```bash
git clone https://github.com/ufdatastudio/agents-among-us.git ~/agents-among-us
cd ~/agents-among-us
./container/pubapps-navigator-deploy.sh setup --port 8080
```

**Daily operations** (run from `~/agents-among-us`):

```bash
./container/pubapps-navigator-deploy.sh status     # systemctl --user status
./container/pubapps-navigator-deploy.sh logs       # podman logs -f
./container/pubapps-navigator-deploy.sh rebuild    # after a git pull
./container/pubapps-navigator-deploy.sh uninstall  # remove the Quadlet (image stays)
```

The full runbook, covering prerequisites, monitoring, updating, troubleshooting, and the RC support workflow, lives in [docs/pubapps-deployment.md](docs/pubapps-deployment.md).

#### 5b. Apptainer + `pubapps-deploy.sh` (GPU and Globus paths)

The earlier deployment flow remains available. PubApps VMs do not have GPUs, so the lightweight navigator container is used by default; `--gpu` and `--globus` are available for GPU-allocated nodes.

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

**With Globus Compute (recommended for GPU inference without a local GPU):**

PubApps VMs typically lack GPUs. The `--globus` flag lets the web UI route local-model inference to a Globus Compute endpoint on HiPerGator, which provisions GPU workers via SLURM automatically. API models (Navigator, OpenAI, Anthropic) continue to work alongside Globus in the same game.

Prerequisites:
1. Configure and start a Globus Compute endpoint on HiPerGator (see [Section 3: Globus Compute](#3-globus-compute-remote-endpoint))
2. Authenticate with `globus-compute-endpoint start` at least once to populate `~/.globus_compute/`

```bash
./container/pubapps-deploy.sh --port <YOUR_PORT> --globus <endpoint-uuid>
```

The deploy script bind-mounts `~/.globus_compute/` into the container for authentication tokens and sets `LLM_MODE=GLOBUS` with the endpoint UUID.

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

UF Navigator provides free API credits ($25/month) and routes to models from multiple vendors (Llama, Gemma, Nemotron, GPT OSS, Claude, Gemini).

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

API and local models can coexist in the same game. For example, you could configure Agent 0 as a Byzantine running `navigator:claude-sonnet-5.5` and Agents 1-3 as Honest running `google/gemma-2-9b-it` locally. The game engine routes each agent's generation call to the appropriate backend.

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

## Citation

If you use this framework or dataset in your research, please cite our IVA 2026 paper ([doi:10.1145/3806774.3827986](https://doi.org/10.1145/3806774.3827986)):

```bibtex
@inproceedings{kurian2026evaluating,
    title = "Evaluating LLM Social Cognition Through Multi-Agentic Strategic Games",
    author = "Kurian, Kevin  and
      Scroggins, Kevin  and
      Dorley, Emmanuel  and
      Grant, Christan",
    booktitle = "Proceedings of the International Conference on Intelligent Virtual Agents (IVA)",
    month = sep,
    year = "2026",
    address = "Puebla, Mexico",
    publisher = "Association for Computing Machinery",
    doi = "10.1145/3806774.3827986",
    url = "https://doi.org/10.1145/3806774.3827986"
}
```

## Issues and Contributions

Found a bug or have a feature request?
Please [open an issue](https://www.google.com/search?q=https://github.com/ufdatastudio/agents-among-us/issues) on GitHub.

Contributions to extend the game map, add new roles, or optimize inference workflows are always welcome!
