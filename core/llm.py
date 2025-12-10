# core/llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
from config.settings import QUANTIZATION

# Suppress heavy logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class ModelManager:
    _instance = None
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name):
        """
        Loads a model if it's not already in memory.
        """
        if model_name in self.models:
            return

        print(f"Loading Model: {model_name} on {self._device}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if QUANTIZATION and self._device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            )
            
            model.to(self._device)
            
            # Ensure pad token is set
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            print(f"Model {model_name} loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise e

    def generate(self, model_name, system_prompt, user_prompt, temperature=0.1):
        """
        Generates response using the specified model.
        """
        # Ensure model is loaded (lazy load safety)
        if model_name not in self.models:
            self.load_model(model_name)

        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self._device)

            attention_mask = torch.ones_like(inputs).to(self._device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )

            input_len = inputs.shape[1]
            response = outputs[0][input_len:]
            return tokenizer.decode(response, skip_special_tokens=True).strip()
            
        except Exception as e:
            print(f"\n[LLM ERROR on {model_name}]: {e}")
            return "move" # Fail-safe