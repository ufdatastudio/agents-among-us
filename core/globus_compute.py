"""Globus Compute integration for remote LLM inference.

Registers a standalone inference function with Globus Compute and
provides a GlobusInferenceExecutor that ModelManager uses to submit
generation tasks to a remote endpoint.
"""

import os

from globus_compute_sdk import Client, Executor
from loguru import logger as log


def remote_inference(model_name, system_prompt, user_prompt, temperature):
    """Standalone function executed on the Globus Compute endpoint worker.

    This function runs in an isolated process on the remote compute node.
    It loads the model on first call (cached in the worker process) and
    generates a response.

    Args:
        model_name: HuggingFace model identifier.
        system_prompt: System prompt for the model.
        user_prompt: User prompt for the model.
        temperature: Sampling temperature.

    Returns:
        The generated text response as a string.
    """
    import gc
    import re

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    # Use a module-level cache so models persist across calls within the same worker
    if not hasattr(remote_inference, "_models"):
        remote_inference._models = {}
        remote_inference._tokenizers = {}

    # Load model if not cached
    if model_name not in remote_inference._models:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.chat_template and "enumrate" in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace("enumrate", "enumerate")

        is_mxfp4 = model_name in MXFP4_MODELS
        use_quantize = model_name in QUANTIZE
        use_bnb = use_quantize and torch.cuda.is_available() and not is_mxfp4

        if use_bnb:
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
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_safetensors=True,
                device_map="auto",
                dtype=torch.bfloat16,
            )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        remote_inference._models[model_name] = model
        remote_inference._tokenizers[model_name] = tokenizer

    model = remote_inference._models[model_name]
    tokenizer = remote_inference._tokenizers[model_name]

    # Build messages
    if model_name in CONCATENATE:
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
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
            return_dict=True,
        ).to(model.device)
    else:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
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

    input_len = inputs["input_ids"].shape[1]
    response = outputs[0][input_len:]
    decoded = tokenizer.decode(response, skip_special_tokens=True).strip()

    # Post-processing (same as ModelManager._postprocess_response)
    if "<think>" in decoded:
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL)
    if "assistantfinal" in decoded:
        decoded = decoded.split("assistantfinal")[-1].strip()
    if decoded.count('"') % 2 != 0:
        decoded += '"'
    quotes = re.findall(r'"([^"]*)"', decoded)
    if quotes:
        decoded = quotes[-1]

    return decoded.strip()


class GlobusInferenceExecutor:
    """Wraps a Globus Compute Executor for submitting inference tasks.

    Args:
        endpoint_id: UUID of the Globus Compute endpoint.
    """

    def __init__(self, endpoint_id):
        self.endpoint_id = endpoint_id
        self.gcc = Client()
        self.fn_uuid = self.gcc.register_function(remote_inference)
        log.info("Registered remote_inference function: {}", self.fn_uuid)
        self.gce = Executor(endpoint_id=endpoint_id, client=self.gcc)
        log.info("Globus Compute executor connected to endpoint: {}", endpoint_id)

    def submit(self, model_name, system_prompt, user_prompt, temperature):
        """Submit an inference task and return the future."""
        future = self.gce.submit(
            remote_inference,
            model_name,
            system_prompt,
            user_prompt,
            temperature,
        )
        return future

    def shutdown(self):
        """Clean up the executor."""
        self.gce.shutdown()


def create_executor():
    """Create a GlobusInferenceExecutor from the GLOBUS_COMPUTE_ENDPOINT env var.

    Returns:
        A GlobusInferenceExecutor instance.

    Raises:
        ValueError: If GLOBUS_COMPUTE_ENDPOINT is not set.
    """
    endpoint_id = os.environ.get("GLOBUS_COMPUTE_ENDPOINT")
    if not endpoint_id:
        raise ValueError(
            "GLOBUS_COMPUTE_ENDPOINT environment variable must be set. "
            "Run 'globus-compute-endpoint list' to find your endpoint UUID."
        )
    return GlobusInferenceExecutor(endpoint_id)
