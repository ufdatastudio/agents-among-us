# Save this as model_manager.py

import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Set transformers logging to a quiet level
logging.getLogger("transformers").setLevel(logging.WARNING)

# --- Global Cache for Models and Tokenizers ---

# This dictionary will store loaded models to prevent reloading them on every call.
_loaded_models = {}

def _load_model(model_name):
    """
    Private function to load a model and tokenizer.
    It checks a local cache first, then downloads if necessary.
    """
    # If model is already loaded, return it from the cache immediately.
    if model_name in _loaded_models:
        return _loaded_models[model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(f"Warning: No GPU detected for {model_name}. Running on CPU will be very slow.")

    try:
        # Try to load from local files first to save time
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map= "auto",
            local_files_only=True
        )
    except (OSError, EnvironmentError):
        # If not found locally, download from Hugging Face Hub
        print(f"'{model_name}' not found in cache, downloading from Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    
    # Store the loaded model and tokenizer in the cache
    _loaded_models[model_name] = (model, tokenizer, device)
    print(f"--- {model_name} loaded successfully. ---")
    return model, tokenizer, device

def generate(model_name, system_prompt, user_prompt):
    """
    The main function to be called by agents.
    It loads a model (or retrieves it from cache) and generates a response.
    """
    model, tokenizer, device = _load_model(model_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    attention_mask = torch.ones_like(inputs).to(device)
    
    # Generate the output
    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=50, # Sufficient for short action responses
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and return the response
    input_length = inputs.shape[1]
    response = outputs[0][input_length:]
    decoded_action = tokenizer.decode(response, skip_special_tokens=True).strip()
    
    return decoded_action