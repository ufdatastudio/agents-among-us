import os
import json
import time
import gc
import re
import logging
import platform

CURRENT_MODE = os.environ.get("LLM_MODE", "LOCAL")

IS_MAC = platform.system() == "Darwin"

if CURRENT_MODE != "CONTROLLER":
    import torch

    if not IS_MAC:
        try: 
            import unsloth 
            import transformers
            UNSLOTH_AVAILABLE = True
        except ImportError:
            import transformers
            UNSLOTH_AVAILABLE = True
            print("Unsloth not available.")
    else:
        import transformers
        UNSLOTH_AVAILABLE = False

from config.settings import QUANTIZATION

# models that have prequantized versions available
UNSLOTH = {
    "unsloth/Apertus-70B-Instruct-2509-unsloth-bnb-4bit",
    }

MXFP4_MODELS = {
    "MultiverseComputingCAI/HyperNova-60B",
}

TOKENIZE_MODELS = {
    "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled",
    "Nexusflow/Athene-V2-Chat",
    "MultiverseComputingCAI/HyperNova-60B"

}

ATTENTION = {
    'swiss-ai/Apertus-70B-Instruct-2509'
}

# Suppress heavy logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class ModelManager:
    _instance = None
    
    # detect operating system
    if IS_MAC:
        if torch.backends .mps.is_available() and torch.backends.mps.is_built():
            self._device = "mps"
            print("Using Apple Silicon GPU (MPS).")
        else:
            self._device = "cpu"
            print("Using Apple Silicon CPU (MPS unavailable).")
    else:
        if torch.cude.is_avilable():
            self.device = "cude"
            print("Using NVIDIA (CUDA) GPU.")
        else:
            self.device = "cpu"
            print("Using CPU (CUDA unavailable).")
    
    self.mode = os.environ.get("LLM_MODE", "LOCAL") # LOCAL, CONTROLLER, WORKER
    self.game_id = None
    self.base_ipc_path = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def set_game_context(self, game_id, comp_name):
        """Called by main.py to set up the IPC path."""
        self.game_id = game_id
        self.base_ipc_path = os.path.join("logs", comp_name, f"Game_{game_id}", "ipc")
        os.makedirs(self.base_ipc_path, exist_ok=True)

    def load_model(self, model_name):
        """
        Loads a model if it's not already in memory.
        Tweaked to work on Mac as well.
        """ 

        if self.mode == "CONTROLLER":
            print(f"[Controller] Model '{model_name}' marked for remote execution.")
            return

        if model_name in self.models:
            return

        print(f"Loading Model: {model_name}...")  
        print(f"Device: {self._device}") # just to check device for peace of mind

        try: # AI generated logic (may need tweaks)
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Fix enumrate typo if present
            if tokenizer.chat_template and "enumrate" in tokenizer.chat_template:
                tokenizer.chat_template = tokenizer.chat_template.replace("enumrate", "enumerate")

            # Determine dtype based on device
            if self._device == "cuda":
                if torch.cuda.is_bf16_supported():
                    compute_dtype = torch.bfloat16
                else:
                    compute_dtype = torch.float16
            elif self._device == "mps":
                compute_dtype = torch.float16  # MPS works well with float16
            else:
                compute_dtype = torch.float32  # CPU uses float32
            
            print(f"Using dtype: {compute_dtype}")
            
            # Check if we should use quantization
            is_unsloth = model_name in UNSLOTH
            is_mxfp4 = model_name in MXFP4_MODELS
            
            use_bnb_quantization = (
                QUANTIZATION 
                and self._device == "cuda"
                and UNSLOTH_AVAILABLE
                and not is_unsloth
                and not is_mxfp4
            )

            # Load model based on configuration
            if use_bnb_quantization:
                print("--> Loading with BitsAndBytes quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto"
                )
                
            elif is_mxfp4:
                print("--> Loading MXFP4 model...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto",
                    torch_dtype=compute_dtype
                )
                
            elif is_unsloth and UNSLOTH_AVAILABLE and self._device == "cuda":
                print("--> Loading via Unsloth FastLanguageModel...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=32678,
                    dtype=compute_dtype,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(model)
                
            else:
                # Standard loading (Mac, CPU, or non-quantized)
                print("--> Loading standard model (no quantization)...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map=self._device
                )            



        

# ================================================================= #

# NEW CODE ENDS HERE - everything below is original llm.py
# To switch, comment out new code above and remove comment marks from below"

'''
# core/llm.py
# ModelManager: handles loading, unloading, and local/remote generation for LLMs used by agents (supports quantization and worker IPC).
import os
import json
import time
import gc
import re
import logging

CURRENT_MODE = os.environ.get("LLM_MODE", "LOCAL")
if CURRENT_MODE != "CONTROLLER":
    import torch
    import gc
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


from config.settings import QUANTIZATION

# models that have prequantized versions available
UNSLOTH = {
    "unsloth/Apertus-70B-Instruct-2509-unsloth-bnb-4bit",
    }

MXFP4_MODELS = {
    "MultiverseComputingCAI/HyperNova-60B",
}

TOKENIZE_MODELS = {
    "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled",
    "Nexusflow/Athene-V2-Chat",
    "MultiverseComputingCAI/HyperNova-60B"

}

ATTENTION = {
    'swiss-ai/Apertus-70B-Instruct-2509'
}

# Suppress heavy logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._device = "cuda"
        self.mode = os.environ.get("LLM_MODE", "LOCAL") # LOCAL, CONTROLLER, WORKER
        self.game_id = None
        self.base_ipc_path = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def set_game_context(self, game_id, comp_name):
        """Called by main.py to set up the IPC path."""
        self.game_id = game_id
        self.base_ipc_path = os.path.join("logs", comp_name, f"Game_{game_id}", "ipc")
        os.makedirs(self.base_ipc_path, exist_ok=True)

    def load_model(self, model_name):
        """
        Loads a model if it's not already in memory.
        """

        if self.mode == "CONTROLLER":
            print(f"[Controller] Model '{model_name}' marked for remote execution.")
            return

        if model_name in self.models:
            return

        print(f"Loading Model: {model_name}...")
        
        try:
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.chat_template and "enumrate" in tokenizer.chat_template:
                tokenizer.chat_template = tokenizer.chat_template.replace("enumrate", "enumerate")

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.float16
            
            is_unsloth = model_name in UNSLOTH
            is_mxfp4 = model_name in MXFP4_MODELS

            use_bnb_quantization = (
                QUANTIZATION 
                and self._device == "cuda" 
                and not is_unsloth
                and not is_mxfp4
            )

            if use_bnb_quantization:
                quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ) 
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto"

                )

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

            elif is_mxfp4:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto",
                    torch_dtype=compute_dtype
                )
            else:
                print(f"--> Loading via UNSLOTH FastLanguageModel...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=model_name,
                        max_seq_length=32678,
                        dtype=compute_dtype,
                        load_in_4bit=True,
                        device_map="auto",
                    )
                FastLanguageModel.for_inference(model)


            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            print(f"Model {model_name} loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise e



    def unload_all_models(self):
  
        # 1. Clear Dictionary References
        self.models.clear()
        self.tokenizers.clear()
        
        # 2. Force Python Garbage Collection
        gc.collect()
        
        # 3. Empty PyTorch CUDA Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() # Wait for all kernels to finish



    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1):
        """
        Polymorphic generate:
        - CONTROLLER: Writes to file, waits for response.
        - LOCAL: Runs torch directly.
        """
        if self.mode == "CONTROLLER":
            return self._generate_remote(model_name, system_prompt, user_prompt, temperature)
        else:
            return self._generate_local(model_name, system_prompt, user_prompt, temperature)


    def _generate_remote(self, model_name, system_prompt, user_prompt, temperature):
        """Writes request to disk and polls for response."""
        if not self.game_id:
            raise ValueError("Game ID not set in ModelManager. Call set_game_context first.")

        request_id = f"req_{time.time()}_{os.getpid()}"
        request_file = os.path.join(self.base_ipc_path, f"{request_id}.json")
        response_file = os.path.join(self.base_ipc_path, f"{request_id}_response.json")

        payload = {
            "model_name": model_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "temperature": temperature,
            "id": request_id
        }

        # 1. Write Request
        with open(request_file, "w") as f:
            json.dump(payload, f)        
        
        start_time = time.time()
        while not os.path.exists(response_file):
            if time.time() - start_time > 500:
                print(f"[Timeout] Waiting for {model_name}...")
                return "SKIP (Timeout)"
            time.sleep(0.5)

        # 3. Read Response
        try:
            with open(response_file, "r") as f:
                data = json.load(f)
            response_text = data.get("response", "")
        except Exception as e:
            print(f"Error reading response: {e}")
            response_text = "ERROR"

        # 4. Cleanup
        try:
            os.remove(request_file)
            os.remove(response_file)
        except:
            pass

        return response_text
  
    def _generate_local(self, model_name, system_prompt, user_prompt, temperature=0.1):
        """
        Generates response using the specified model.
        """
        if model_name not in self.models:
            self.load_model(model_name)

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        try:

            if "mixtral" in model_name.lower() or "mistral" in model_name.lower():
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

            if model_name in TOKENIZE_MODELS:
                if model_name in MXFP4_MODELS: 
                    inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    reasoning_effort="low",
                    tokenize=True,         
                    return_tensors="pt",   
                    return_dict=True       
                ).to(model.device)
                    
                else: 
                        
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,         
                        return_tensors="pt",   
                        return_dict=True       
                    ).to(model.device)
            
            else:
                text_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,        
                    add_generation_prompt=True,
                )
                
                
                inputs = tokenizer(
                    [text_prompt], 
                    return_tensors="pt", 
                    add_special_tokens=False  
                ).to(model.device)

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            input_len = inputs['input_ids'].shape[1]
            response = outputs[0][input_len:]
            decoded_response = tokenizer.decode(response, skip_special_tokens=True).strip()
            if "<think>" in decoded_response:
                decoded_response = re.sub(r'<think>.*?</think>', '', decoded_response, flags=re.DOTALL)
            if "assistantfinal" in decoded_response:
                decoded_response = decoded_response.split("assistantfinal")[-1].strip()

            
            if decoded_response.count('"') % 2 != 0:
                decoded_response += '"'
            quotes = re.findall(r'"([^"]*)"', decoded_response)

            if quotes:
                decoded_response = quotes[-1]
            
            return decoded_response
            
        except Exception as e:
            print(f"\n[LLM ERROR on {model_name}]: {e}")
            return "move" 
'''