"""Models the situation room experiments may run on HiPerGator.

A model is eligible when UF Navigator also serves it and neither the model nor its
post-training comes from a country of concern. That rules out the Qwen family,
DeepSeek and the other families in BLOCKED, and any model whose card says it was
improved using or built on them. The audit of the public model cards, dated
2026-11-10, is described in docs/situation-room.md.

Check a model before starting a server for it:
    uv run --no-project python -m config.eligible_models <navigator name or checkpoint id>
"""

import re
import sys
from collections.abc import Sequence

# Navigator model name -> Hugging Face checkpoint.
ELIGIBLE = {
    "ministral-8b": "mistralai/Ministral-8B-Instruct-2410",
    "granite-3.3-8b-instruct": "ibm-granite/granite-3.3-8b-instruct",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "mistral-small-3.1": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    # The revision that matches Navigator's deployment is still to confirm.
    "magistral-small": "mistralai/Magistral-Small-2509",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "meta-muse-glimmer-30b": "meta-models/Muse-Glimmer-30B",
    "llama-3.3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

# Navigator model name -> (Hugging Face checkpoint, reason for exclusion).
EXCLUDED = {
    "llama-3.1-nemotron-nano-8B": (
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "the model card says 'Improved using Qwen'",
    ),
    "nemotron-3-nano-30b-a3b": (
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "the model card says 'Improved using Qwen' and cites a Qwen3-based reward model",
    ),
    "nemotron-3-super-120b-a12b": (
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        "the model card does not name its teacher models, so Qwen or DeepSeek outputs cannot be ruled out",
    ),
    "medgemma-27b-it": ("google/medgemma-27b-it", "tuned for medicine and not used in the game"),
}

# Names that mark a model family from a country of concern. The first group is distinctive
# enough to match anywhere, and the second needs word boundaries to avoid false matches.
BLOCKED = re.compile(
    r"qwen|deepseek|alibaba|tongyi|baidu|tencent|hunyuan|zhipu|moonshot|minimax|internlm|bytedance"
    r"|doubao|baichuan|huawei|xiaomi|sensetime|\b(?:glm|ernie|kimi|yi[-_]\d|01[-.]ai)",
    re.IGNORECASE,
)


def check_eligible(model: str) -> str:
    """Returns the checkpoint of an eligible model, or explains why it is not.

    Args:
        model: A Navigator model name or a Hugging Face checkpoint id.

    Returns:
        The Hugging Face checkpoint to serve.

    Raises:
        ValueError: If the name marks a family from a country of concern, the model
            is on the exclusion list, or it is not on the eligible list.
    """
    if BLOCKED.search(model):
        raise ValueError(f"{model} belongs to a model family from a country of concern")
    for name, checkpoint in ELIGIBLE.items():
        if model in (name, checkpoint):
            return checkpoint
    for name, (checkpoint, reason) in EXCLUDED.items():
        if model in (name, checkpoint):
            raise ValueError(f"{model} is excluded: {reason}")
    raise ValueError(f"{model} is not on the eligible list in config/eligible_models.py")


def main(argv: Sequence[str]) -> int:
    """Prints the checkpoint for a model, or the reason it is not eligible.

    Args:
        argv: The model name or checkpoint id.

    Returns:
        0 when the model is eligible and 1 otherwise.
    """
    if len(argv) != 1:
        print("usage: python -m config.eligible_models <navigator name or checkpoint id>", file=sys.stderr)
        return 1
    try:
        print(check_eligible(argv[0]))
    except ValueError as error:
        print(f"not eligible: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
