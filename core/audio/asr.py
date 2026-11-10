"""Speech recognition for the situation room, using Whisper large-v3."""

import math
from collections.abc import Sequence

from loguru import logger
import numpy as np
from scipy.signal import resample_poly
import torch
from transformers import pipeline

from core.audio.mix import SAMPLE_RATE

WHISPER_RATE = 16000


def to_whisper_rate(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resamples audio to the 16 kHz Whisper expects.

    Args:
        samples: Mono float samples.
        sample_rate: Rate of the input in hertz.

    Returns:
        Float32 samples at 16 kHz.
    """
    divisor = math.gcd(sample_rate, WHISPER_RATE)
    return resample_poly(samples, WHISPER_RATE // divisor, sample_rate // divisor).astype(np.float32)


class Transcriber:
    """Turns speech into text, optionally with word timestamps."""

    def __init__(self, model_id: str = "openai/whisper-large-v3", device: str | None = None) -> None:
        """Loads the Whisper pipeline.

        Args:
            model_id: Hugging Face id of the Whisper model.
            device: Torch device, or None to use CUDA when it is available.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=dtype, device=device)
        logger.info("Whisper {} ready on {}", model_id, device)

    def transcribe_batch(
        self,
        clips: Sequence[np.ndarray],
        sample_rate: int = SAMPLE_RATE,
        words: bool = False,
    ) -> list[dict]:
        """Transcribes several clips in one batch.

        Args:
            clips: Mono float arrays.
            sample_rate: Rate of the clips in hertz.
            words: Whether to return a start and end time for every word.

        Returns:
            One dict per clip with `text` and, when requested, `words`, a list of
            dicts with `word`, `start` and `end` in seconds.
        """
        inputs = [{"raw": to_whisper_rate(c, sample_rate), "sampling_rate": WHISPER_RATE} for c in clips]
        outputs = self._pipe(
            inputs,
            batch_size=len(inputs),
            chunk_length_s=30,
            return_timestamps="word" if words else False,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )
        results = []
        for output in outputs:
            entry = {"text": output["text"].strip()}
            if words:
                entry["words"] = [
                    {"word": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                    for c in output.get("chunks", [])
                ]
            results.append(entry)
        return results
