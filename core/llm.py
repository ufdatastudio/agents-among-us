# core/llm.py
import os
import json
import time
import re

if os.environ.get("LLM_MODE", "LOCAL") != "CONTROLLER":
    import torch
    import gc
    from unsloth import FastLanguageModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config, AutoConfig
    #from transformers.utils import logging
    #logging.disable_progress_bar()

CONCATENATE = {
    "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled",
    "google/gemma-2-9b-it",
}
QUANTIZE = {
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 
    "zerofata/L3.3-GeneticLemonade-Final-v2-70B",
    "NousResearch/Hermes-4-70B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-Next-80B-A3B-Instruct", 
    "swiss-ai/Apertus-70B-Instruct-2509",
    "arcee-ai/Arcee-Nova",
    "Aratako/Mixtral-8x7B-Instruct-v0.1-upscaled",
    "Nexusflow/Athene-V2-Chat",
}
MXFP4_MODELS = {
    "MultiverseComputingCAI/HyperNova-60B",
    "openai/gpt-oss-20b",
}

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._device = "cuda"
        self.mode = os.environ.get("LLM_MODE", "LOCAL") # LOCAL, CONTROLLER
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
        
        gc.collect()
        torch.cuda.empty_cache()

        initial_free, total = torch.cuda.mem_get_info()
        print(f"Loading Model: {model_name}:  [Memory] Before Load: {initial_free/1024**3:.2f} GB free", flush=True)

        try:
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.chat_template and "enumrate" in tokenizer.chat_template:
                tokenizer.chat_template = tokenizer.chat_template.replace("enumrate", "enumerate")
          
            is_mxfp4 = model_name in MXFP4_MODELS
            use_quantize = model_name in QUANTIZE

            use_bnb_quantization = (
                use_quantize 
                and self._device == "cuda" 
                and not is_mxfp4
            )

            if use_bnb_quantization:
                quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ) 
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto",
                    dtype=torch.bfloat16,

                )

                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id

            elif is_mxfp4:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto",
                    dtype=torch.bfloat16,
                )
            
            else:
                    model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True,
                    device_map="auto",
                    dtype=torch.bfloat16,

                )
            

                # model, tokenizer = FastLanguageModel.from_pretrained(
                #         model_name=model_name,
                #         max_seq_length=8192,
                #         load_in_4bit=True,
                #         device_map="auto",
                #         dtype=torch.bfloat16,
                #         gpu_memory_utilization=0.25
                #     )
                # FastLanguageModel.for_inference(model)




            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            torch.cuda.empty_cache()

            final_free, _ = torch.cuda.mem_get_info()
            mem_taken = (initial_free - final_free) / 1024**3
            print(f"Loaded {model_name} | VRAM Usage: {mem_taken:.2f} GiB | Memory Remaining: {final_free / 1024**3:.2f} GiB", flush=True)

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
            torch.cuda.synchronize() 

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

        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        request_id = f"req_{safe_model_name}_{time.time()}_{os.getpid()}"
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
            f.flush()
            os.fsync(f.fileno())        
        
        start_time = time.time()
        while not os.path.exists(response_file):
            if time.time() - start_time > 180:
                print(f"[Timeout] Waiting for {model_name}...")
                return "SKIP (Timeout)"
            time.sleep(0.05)

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

            if model_name in CONCATENATE:
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

           
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
            
            # else:
            #     text_prompt = tokenizer.apply_chat_template(
            #         messages,
            #         tokenize=False,        
            #         add_generation_prompt=True,
            #     )
                
                
                # inputs = tokenizer(
                #     [text_prompt], 
                #     return_tensors="pt", 
                #     add_special_tokens=False  
                # ).to(model.device)

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