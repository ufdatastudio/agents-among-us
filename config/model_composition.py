# --- Model Constants ---
LLAMA31_70B = "meta-llama/Llama-3.1-70B-Instruct"
LLAMA31_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA32_1B= "meta-llama/Llama-3.2-1B-Instruct"
QWEN25_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"


# Each dictionary represents one "Setup" that a game instance can run.
COMPOSITION = [
    # Composition 0: Baseline - Everyone uses Llama 3
    {
        "name": "Baseline_Llama3",
        "mode": "default",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": LLAMA31_8B,
        "byzantine_model": LLAMA31_8B
    },

    # Composition 1: David vs Goliath (a)
    {
        "name": "DvGa",
        "mode": "default",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": QWEN25_1_5B,
        "byzantine_model": LLAMA31_8B
    },

    # Composition 2: David vs Goliath (b)
    {
        "name": "DvGb",
        "mode": "default",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": LLAMA32_1B,
        "byzantine_model": LLAMA31_8B
    },
    # Composition 3 - David vs Goliath (c)
        {
        "name": "DvGc",
        "mode": "default",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": LLAMA31_8B,
        "byzantine_model": LLAMA31_70B
    },

        # Composition 4 - David vs Goliath (d) 
    {
        "name": "DvGd",
        "mode": "default",
        "honest_count": 9,
        "byzantine_count": 1,
        "honest_model": LLAMA31_8B,
        "byzantine_model": LLAMA31_70B
    }
]