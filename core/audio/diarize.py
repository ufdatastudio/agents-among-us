"""Speaker diarization for the situation room, using Nemotron 3 Diarization.

The model is a streaming Sortformer with about 100 million parameters. It labels up to 8 speakers in the order in
which they arrive and runs from the Hugging Face weights with the `transformers` library, so it needs no NeMo install.
"""

import math

from loguru import logger
import numpy as np
from scipy.signal import resample_poly
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from core.audio.der import Segment
from core.audio.mix import SAMPLE_RATE

MODEL_ID = "nvidia/Nemotron-3-Diarization"
MAX_SPEAKERS = 8
STREAMING_MODES = ("low_latency", "very_low_latency", "ultra_low_latency")


class Diarizer:
    """Says who spoke when in a recording."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        """Loads the model and its processor.

        Args:
            model_id: Hugging Face id of the diarization model.
            device: Torch device, or None to use CUDA when it is available.
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to(device).eval()
        self.rate = self._processor.feature_extractor.sampling_rate
        logger.info("Diarizer {} ready on {} at {} Hz", model_id, device, self.rate)

    def diarize(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, streaming_mode: str | None = None) -> list[Segment]:
        """Labels the speakers of a recording.

        Args:
            samples: Mono float samples.
            sample_rate: Rate of the samples in hertz.
            streaming_mode: None to see the whole recording at once, or one of `STREAMING_MODES` to feed it in chunks
                the way a live listener would, with the latency that the mode names.

        Returns:
            Segments in time order, labeled "0" to "7" by the order in which each speaker first spoke.

        Raises:
            ValueError: If `streaming_mode` is not one of `STREAMING_MODES`.
        """
        if streaming_mode is not None and streaming_mode not in STREAMING_MODES:
            raise ValueError(f"streaming_mode must be None or one of {STREAMING_MODES}, not {streaming_mode!r}")
        audio = self._resample(samples, sample_rate)
        with torch.inference_mode():
            if streaming_mode is None:
                inputs = self._processor(audio, sampling_rate=self.rate).to(self._model.device, dtype=self._model.dtype)
                logits = self._model(**inputs).logits
                found = self._processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
            else:
                found = self._processor.extract_speaker_dict(self._stream(audio, streaming_mode))[0]
        return [Segment(str(item["Speaker"]), float(item["Start"]), float(item["End"])) for item in found]

    def _resample(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resamples to the rate the model expects."""
        divisor = math.gcd(sample_rate, self.rate)
        return resample_poly(samples, self.rate // divisor, sample_rate // divisor).astype(np.float32)

    def _stream(self, audio: np.ndarray, mode: str) -> torch.Tensor:
        """Feeds a recording through the model chunk by chunk and joins the frame scores.

        Args:
            audio: Mono float samples at the model rate.
            mode: One of `STREAMING_MODES`.

        Returns:
            Logits of shape (1, frames, speakers).
        """
        processor, model = self._processor, self._model
        processor.set_streaming_mode(mode)

        def chunks():
            yield processor(
                audio[: processor.num_samples_first_audio_chunk],
                sampling_rate=self.rate,
                is_streaming=True,
                is_first_audio_chunk=True,
            )
            frame = processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(frame)
            while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
                yield processor(audio[start:end], sampling_rate=self.rate, is_streaming=True, is_first_audio_chunk=False)
                frame += processor.num_mel_frames_per_step
                start = processor.audio_chunk_start(frame)
            yield processor(
                audio[start:],
                sampling_rate=self.rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                is_last_audio_chunk=True,
            )

        cache, logits = None, []
        for inputs in chunks():
            inputs = inputs.to(model.device, dtype=model.dtype)
            outputs = model(**inputs, speaker_cache=cache)
            logits.append(outputs.logits)
            cache = outputs.speaker_cache
        return torch.cat(logits, dim=1)
