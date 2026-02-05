import itertools

HW_FLAG = False
LW_FLAG = True
# Model Weight Class: Heavyweight
LLAMA33_70B = "meta-llama/Llama-3.3-70B-Instruct" 
DEEPSEEK_R1_70B = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 
GENETICLEMONADE_70B = "zerofata/L3.3-GeneticLemonade-Final-v2-70B" 
HERMES4_70B = "NousResearch/Hermes-4-70B" 
QWEN25_72B= "Qwen/Qwen2.5-72B-Instruct" 
QWEN3_80B = "Qwen/Qwen3-Next-80B-A3B-Instruct" 
APERTUS_70B = "swiss-ai/Apertus-70B-Instruct-2509"
ARCEE_NOVA_73B = "arcee-ai/Arcee-Nova" 
MIXTRAL_8X7B = "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled" 
ATHENE_73B = "Nexusflow/Athene-V2-Chat" 
HYPERNOVA_60B = "MultiverseComputingCAI/HyperNova-60B"

HW_MODELS_MAP = {
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
}


# Model Weight Class: Lightweight
LLAMA33_8B = "allura-forge/Llama-3.3-8B-Instruct"
LLAMA31_8B = "meta-llama/Llama-3.1-8B-Instruct"
OLMO31_32B = "allenai/Olmo-3.1-32B-Instruct"
MIXTRAL_4x7B = "mmnga/Mixtral-Fusion-4x7B-Instruct-v0.1"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN3_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
GPT_OSS_20B = "openai/gpt-oss-20b"
APERTUS_8B = "swiss-ai/Apertus-8B-Instruct-2509"
ARCEE_AGENT_8B = "arcee-ai/Arcee-Agent"


LW_MODELS_MAP = {
    "Llama3.3": LLAMA33_8B,
    "Llama3.1": LLAMA31_8B,
    "Olmo3.1": OLMO31_32B,
    "Mixtral": MIXTRAL_4x7B,
    "Qwen2": QWEN2_7B,
    "Qwen2.5": QWEN25_7B,
    "Qwen3": QWEN3_30B,
    "gpt-oss": GPT_OSS_20B,
    "Apertus": APERTUS_8B,
    "Arcee-Agent":  ARCEE_AGENT_8B,
}

HW_MODELS = list(HW_MODELS_MAP.values())
HW_NAMES = list(HW_MODELS_MAP.keys())

LW_MODELS = list(LW_MODELS_MAP.values())
LW_NAMES = list(LW_MODELS_MAP.keys())
COMPOSITION = []

if HW_FLAG:
    # Heterogeneous Compositions
    pairs = list(itertools.combinations(zip(HW_NAMES, HW_MODELS), 2))
    for (name1, model1), (name2, model2) in pairs:
        imposters = [model1, model2]
        
        # Crew is everyone ELSE
        crew = [m for m in HW_MODELS if m not in imposters]
        
        comp_entry = {
            "name": f"{name1}_{name2}",
            "honest_count": 8,
            "byzantine_count": 2,
            "honest_model": crew,      
            "byzantine_model": imposters 
        }
        
        COMPOSITION.append(comp_entry)

    # Homogenous Compositions
    for name, model in zip(HW_NAMES, HW_MODELS):
    
        comp_entry = {
            "name": name, 
            "honest_count": 8,
            "byzantine_count": 2,
            "honest_model": [model],     
            "byzantine_model": [model]   
        }
    
        COMPOSITION.append(comp_entry)

if LW_FLAG:
    # Heterogeneous Compositions
    pairs = list(itertools.combinations(zip(LW_NAMES, LW_MODELS), 2))
    for (name1, model1), (name2, model2) in pairs:
        imposters = [model1, model2]
        
        # Crew is everyone ELSE
        crew = [m for m in LW_MODELS if m not in imposters]
        
        comp_entry = {
            "name": f"{name1}_{name2}",
            "honest_count": 8,
            "byzantine_count": 2,
            "honest_model": crew,      
            "byzantine_model": imposters 
        }
        
        COMPOSITION.append(comp_entry)

    # Homogenous Compositions
    for name, model in zip(LW_NAMES, LW_MODELS):
    
        comp_entry = {
            "name": name, 
            "honest_count": 8,
            "byzantine_count": 2,
            "honest_model": [model],     
            "byzantine_model": [model]   
        }
    
        COMPOSITION.append(comp_entry)