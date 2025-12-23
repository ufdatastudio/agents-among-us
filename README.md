# The Byzantine Brains: Fault-Tolerant Consensus with LLMs

<img width="1173" height="631" alt="image" src="https://github.com/user-attachments/assets/1a1bbbb2-c002-459d-a8d9-ecaa221fac7c" />


## Project Overview
The **Byzantine Brains** project explores fault-tolerant consensus-building in distributed AI systems using large language models (LLMs). Inspired by the classic Byzantine Generals Problem and social deduction games, this project simulates an environment where autonomous agents must collaborate to maintain a spaceship while identifying and ejecting malicious "Byzantine" actors.

The system utilizes local LLMs (specifically Llama-3.1 via Hugging Face) to drive agent logic, allowing them to perceive their environment, form memories, debate in natural language, and cast votes to achieve consensus.

## Objectives
TBD

## Technologies
This project is built using Python and leverages state-of-the-art NLP libraries for agent cognition and Pygame for real-time visualization.

* **Core Language:** Python 3.10+
* **Machine Learning:**
    * **PyTorch:**
    * **Hugging Face Transformers:**
    * **BitsAndBytes:** Used for 4-bit quantization to run large models efficiently on consumer hardware.
* **Visualization:**
    * **Pygame:** Renders the live map, agent movements, and status logs.
* **Data Management:**
    * **JSON/CSV:** For state serialization and statistical logging.

## Project Structure

```text
ByzantineBrains/
├── agents/                 # Agent Logic
│   ├── base_agent.py       # Abstract base class for all agents
│   ├── honest_agent.py     # Logic for Crewmates (Reasoning, Tasking, Voting)
│   └── byzantine_agent.py  # Logic for Impostors (Deception, Tagging, Sabotage)
├── config/                 # Configuration
│   └── settings.py         # Global constants (Map layout, Model paths, Rules)
├── core/                   # System Utilities
│   ├── llm.py              # ModelManager: Handles loading and inference of LLMs
│   ├── logger.py           # LogManager: Handles CSV/Text logging of game events
│   └── state.py            # GameState: Manages the "World Truth" and JSON export
├── game/                   # Game Loop
│   └── game_engine.py      # Controls phases (Movement, Discussion, Voting)
├── logs/                   # Generated logs for each simulation run
├── main.py                 # Headless entry point for the simulation backend
├── live_map.py             # Graphical User Interface (GUI) and visualizer
└── README.md
```

## Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** installed. You will also need a GPU with CUDA support (NVIDIA) for efficient LLM inference

### 2. Clone the Repository
```bash
git clone [https://github.com/NealShankarGit/ByzantineBrains.git](https://github.com/NealShankarGit/ByzantineBrains.git)
cd ByzantineBrains
```
# Install PyTorch with CUDA support (adjust index-url for your specific CUDA version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Project Dependencies
pip install transformers accelerate bitsandbytes pygame

## Model Configuration
By default, the project is configured to use `meta-llama/Llama-3.1-8B-Instruct`. You must have access to this model via Hugging Face to run the simulation.

1.  **Authenticate:** Log in to Hugging Face via the CLI:
    ```bash
    huggingface-cli login
    ```
2.  **Customization:** You can change the target model by modifying the `AGENT_LLM_CONFIG` list in `config/settings.py`.
3.  **Quantization:** The system defaults to 4-bit quantization to optimize memory usage. This behavior is toggled via the `QUANTIZATION` boolean in `config/settings.py` and implemented in `core/llm.py`.

## Usage

### Option 1: Live Visualization (Recommended)
The project includes a `pygame`-based GUI that acts as both a visualizer and a launcher.

1.  Run the visualizer:
    ```bash
    python live_map.py
    ```
2.  **Start:** Click the **START SIM** button in the top right corner. This automatically spawns the `main.py` game engine process via a subprocess.
3.  **Observe:** Watch the agents move, interact, and discuss in real-time. The sidebar displays live agent status, and the bottom panel shows the event log.
4.  **Control:** Click **STOP SIM** to abort the run, or **CLEAR** to delete the map data and reset the view.

### Option 2
To run the simulation without the graphics window (useful for faster batch data collection):

```bash
python main.py
