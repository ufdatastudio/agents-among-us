# config/model_composition.py
# Defines model constants and constructs heterogeneous/homogeneous compositions used to spawn agents in experiments.
import itertools

# Model Weight Class (Heavyweight)
LLAMA33_70B = "meta-llama/Llama-3.3-70B-Instruct" 
DEEPSEEK_R1_70B = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 
GENETICLEMONADE_70B = "zerofata/L3.3-GeneticLemonade-Final-v2-70B" 
HERMES4_70B = "NousResearch/Hermes-4-70B" 
QWEN25_72B= "Qwen/Qwen2.5-72B-Instruct" 
QWEN3_80B = "Qwen/Qwen3-Next-80B-A3B-Instruct"  
APERTUS_70B = "unsloth/Apertus-70B-Instruct-2509-unsloth-bnb-4bit" 
ARCEE_NOVA_73B = "arcee-ai/Arcee-Nova" 
MIXTRAL_8X7B = "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled"  
ATHENE_73B = "Nexusflow/Athene-V2-Chat" 
HYPERNOVA_60B = "MultiverseComputingCAI/HyperNova-60B"

# small models (added for easier testing - not rlly good though)
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
QWEN_1_5B = "Qwen/Qwen2.5-1.5B-Instruct" 
TINY_LLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


MODELS_MAP = {
    "Llama3.3": LLAMA33_70B,
    "Hypernova": HYPERNOVA_60B,
    "Genetic": GENETICLEMONADE_70B,
    "Hermes4": HERMES4_70B,
    "Qwen2.5": QWEN25_72B,
    "Qwen3": QWEN3_80B,
    "Apertus": APERTUS_70B,
    "Arcee": ARCEE_NOVA_73B,
    "Mixtral": MIXTRAL_8X7B,
    "Athene": ATHENE_73B,
    "Llama3B": LLAMA_3B,
    "Qwen1.5B": QWEN_1_5B,
    "TinyLlama": TINY_LLAMA,
}


HW_MODELS = list(MODELS_MAP.values())
HW_NAMES = list(MODELS_MAP.keys())

COMPOSITION = []


# Heterogeneous Compositions
# ==========================================
# All unique combinations of 2 imposters from the 13 models
pairs = list(itertools.combinations(zip(HW_NAMES, HW_MODELS), 2))
for (name1, model1), (name2, model2) in pairs:
    # Imposters are the selected pair
    imposters = [model1, model2]
    
    # Crew is everyone ELSE
    crew = [m for m in HW_MODELS if m not in imposters]
    
    comp_entry = {
        "name": f"{name1}_{name2}",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": crew,      # List of 8 unique models
        "byzantine_model": imposters # List of 2 unique models
    }
    
    COMPOSITION.append(comp_entry)

# Homogenous Compositions
COMPOSITION.extend([
    {
        "name": "llama_3.3_70B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [LLAMA33_70B],
        "byzantine_model": [LLAMA33_70B]
    },

    {
        "name": "hypernova_60B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [HYPERNOVA_60B],
        "byzantine_model": [HYPERNOVA_60B]
    },

    {
        "name": "genetic_lemonade_70B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [GENETICLEMONADE_70B],
        "byzantine_model": [GENETICLEMONADE_70B]
    },

    {
        "name": "hermes4_70B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [HERMES4_70B],
        "byzantine_model": [HERMES4_70B]
    },

    {
        "name": "qwen2.5_72B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [QWEN25_72B],
        "byzantine_model": [QWEN25_72B]
    },

    {
        "name": "qwen3_80B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [QWEN3_80B],
        "byzantine_model": [QWEN3_80B]
    },

    {
        "name": "apertus_70B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [APERTUS_70B],
        "byzantine_model": [APERTUS_70B]
    },
    
    {
        "name": "arcee_nova_73B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [ARCEE_NOVA_73B],
        "byzantine_model": [ARCEE_NOVA_73B]
    },

    {
        "name": "mixtral_8x7B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [MIXTRAL_8X7B],
        "byzantine_model": [MIXTRAL_8X7B]
    },

    {
        "name": "athene_chat_73B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [ATHENE_73B],
        "byzantine_model": [ATHENE_73B]
    },

    {
        "name": "llama_3B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [LLAMA_3B],
        "byzantine_model": [LLAMA_3B]
    },

    {
        "name": "qwen_1.5B",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [QWEN_1_5B],
        "byzantine_model": [QWEN_1_5B]
    },

    {
        "name": "tiny_llama",
        "honest_count": 8,
        "byzantine_count": 2,
        "honest_model": [TINY_LLAMA],
        "byzantine_model": [TINY_LLAMA]
    }
])