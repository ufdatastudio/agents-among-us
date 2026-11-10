"""Speech synthesis for the situation room, using Kokoro."""

import random
from collections.abc import Sequence

from kokoro import KPipeline
from loguru import logger
import numpy as np

# American English voices only, because the pipeline uses the American text front end.
VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova",
    "af_jessica", "am_adam", "am_michael", "am_eric", "am_onyx", "am_puck",
)


def assign_voices(agents: Sequence[str], seed: int) -> dict[str, str]:
    """Gives each agent a distinct voice for the whole game.

    Args:
        agents: Agent names.
        seed: Seed that makes the assignment repeatable.

    Returns:
        A voice name for each agent.

    Raises:
        ValueError: If there are more agents than voices.
    """
    if len(agents) > len(VOICES):
        raise ValueError(f"{len(agents)} agents but only {len(VOICES)} voices")
    return dict(zip(agents, random.Random(seed).sample(VOICES, len(agents))))


class Synthesizer:
    """Turns text into speech at 24 kHz with a chosen voice."""

    def __init__(self, repo_id: str = "hexgrad/Kokoro-82M", device: str | None = None) -> None:
        """Loads the Kokoro pipeline.

        Args:
            repo_id: Hugging Face repository with the weights and voices.
            device: Torch device, or None to let Kokoro choose.
        """
        self._pipeline = KPipeline(lang_code="a", repo_id=repo_id, device=device)
        logger.info("Kokoro pipeline ready on {}", device or "default device")

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """Speaks a text.

        Args:
            text: What to say.
            voice: One of VOICES.
            speed: Speaking speed, where 1.0 is the voice's natural rate.

        Returns:
            Mono float32 samples at 24 kHz, empty if nothing was produced.
        """
        chunks = []
        for result in self._pipeline(text, voice=voice, speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is not None:
                chunks.append(audio.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
