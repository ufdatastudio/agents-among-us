import torch
import gc
import os
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM

# Force unbuffered output
print = functools.partial(print, flush=True)

models_to_cache = [
    # Add model names of models to cache for faster loading
    # Names should match those used in model_composition.py
]

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cache_model(model_name):
    print(f"Processing: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        print(f"Successfully loaded: {model_name}")
        
        del model
        del tokenizer
        flush_memory()

    except Exception as e:
        print(f"FAILED: {model_name}")
        print(f"Error: {e}")
        flush_memory()

if __name__ == "__main__":
    print(f"Starting cache for {len(models_to_cache)} models...")
    
    for model in models_to_cache:
        cache_model(model)
        
    print("Batch Job Complete.")