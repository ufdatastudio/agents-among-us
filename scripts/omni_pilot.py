"""Native-ear pilot for the situation room: can Nemotron Omni tell what several voices said at once?

This repeats the Whisper pilot of `scripts/audio_pilot.py` with the Omni model as the listener. The statements, the
voices, the speaker subsets and the onsets are the ones of that pilot for the same seed, so the two tables compare
directly. Omni gets the mix as audio, writes what it heard with a line for each voice it can tell apart, and is asked
how many people talk. Scoring uses the text of the speakers and nothing about the audio, as in the Whisper pilot.

Two steps, because Kokoro and vLLM do not share an environment.

Usage:
    uv run --no-project --python 3.12 --with kokoro --with soundfile --with transformers --with torch \
        --with scipy --with numpy --with loguru --with <spacy model> python -m scripts.omni_pilot clips --out clips
    uv run --no-project --with "vllm[audio]" python -m scripts.omni_pilot trials --clips clips \
        --url http://127.0.0.1:8000/v1 --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 --out result.json
"""

import argparse
import base64
import io
import json
import random
import re
import statistics
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from core.audio.metrics import (
    word_error_rate,
    word_precision,
    word_recall,
)
from core.audio.mix import (
    SAMPLE_RATE,
    Clip,
    mix_for_reader,
    overlap_fraction,
    read_wav,
    write_wav,
)
from core.audio.speech import Synthesizer
from core.audio.tts import assign_voices
from scripts.audio_pilot import MIN_CONTENT_LENGTH, NAMES, unique_recall, utterance

TRANSCRIBE = (
    "Transcribe this recording word for word. Several people may talk at the same time. "
    "If you can tell the voices apart, write the words of each person on a line of their own that starts with "
    "Speaker 1:, Speaker 2: and so on. Write nothing else."
)
COUNT = "How many different people are talking in this recording? Answer with one number."
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
LABEL = re.compile(r"\bspeaker\s*(\d+)\s*:", re.IGNORECASE)
TEXTS = "texts.json"

Ask = Callable[[np.ndarray, str], str]


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encodes mono float samples as the bytes of a 16-bit WAV file.

    Args:
        samples: Float samples in [-1, 1].
        sample_rate: Sampling rate in hertz.

    Returns:
        The file content.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def chat_body(model: str, samples: np.ndarray, prompt: str, max_tokens: int = 1024) -> dict:
    """Builds the request that gives the model a recording and a question.

    The sampling and the switch that turns reasoning off are the instruct settings of the model card.

    Args:
        model: Name the server serves the model under.
        samples: The recording.
        prompt: The question.
        max_tokens: Longest answer.

    Returns:
        The JSON body of a chat completion request.
    """
    url = "data:audio/wav;base64," + base64.b64encode(wav_bytes(samples)).decode("ascii")
    return {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": url}}, {"type": "text", "text": prompt}]}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_k": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def ask_server(url: str, model: str, samples: np.ndarray, prompt: str, timeout: float = 600.0) -> str:
    """Asks a vLLM server about a recording.

    Args:
        url: Base URL of the server, ending in `/v1`.
        model: Name the server serves the model under.
        samples: The recording.
        prompt: The question.
        timeout: Seconds to wait for the answer.

    Returns:
        The text of the answer.

    Raises:
        RuntimeError: If the server answers with an error.
    """
    request = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions",
        data=json.dumps(chat_body(model, samples, prompt)).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"server answered {error.code}: {error.read().decode('utf-8', 'replace')[:500]}") from error


def strip_labels(text: str) -> str:
    """Removes the `Speaker 1:` labels from a transcript, so they do not count as words."""
    return " ".join(LABEL.sub(" ", text).split())


def labeled_speakers(text: str) -> int:
    """Counts the different `Speaker N:` labels in a transcript."""
    return len({int(n) for n in LABEL.findall(text)})


def parse_count(text: str) -> int | None:
    """Reads the number of people from an answer.

    Args:
        text: The answer, such as `3` or `There are three people`.

    Returns:
        The first number in it, as digits or as a word up to twelve, or None.
    """
    for token in re.findall(r"\d+|[a-z]+", text.lower()):
        if token.isdigit():
            return int(token)
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def build_clips(out: Path, seed: int, synth: Synthesizer, voices: dict[str, str] | None = None) -> dict[str, str]:
    """Writes the clean clip of each agent and the statements, the same as the Whisper pilot for the seed.

    Args:
        out: Folder for `clean_<name>.wav` and `texts.json`. It is created.
        seed: Seed of the pilot.
        synth: The synthesizer.
        voices: Voice of each agent, or None to draw them with the seed.

    Returns:
        What each agent says.
    """
    rng = random.Random(seed)
    voices = voices or assign_voices(NAMES, seed)
    texts = {name: utterance(name, rng) for name in NAMES}
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        write_wav(out / f"clean_{name}.wav", synth.synthesize(texts[name], voices[name]))
    (out / TEXTS).write_text(json.dumps({"seed": seed, "texts": texts, "voices": voices}, indent=1), encoding="utf-8")
    return texts


def read_clips(folder: Path) -> tuple[int, dict[str, str], dict[str, np.ndarray]]:
    """Reads what `build_clips` wrote.

    Args:
        folder: The folder.

    Returns:
        The seed, the statements and the clean clip of each agent.
    """
    info = json.loads((folder / TEXTS).read_text(encoding="utf-8"))
    audio = {name: read_wav(folder / f"clean_{name}.wav")[0] for name in info["texts"]}
    return info["seed"], info["texts"], audio


def score(chosen: Sequence[Clip], texts: dict[str, str], transcript: str) -> dict:
    """Scores a transcript of a mix against what the speakers said.

    Args:
        chosen: The clips in the mix.
        texts: What each agent said.
        transcript: What the model wrote, without labels.

    Returns:
        Word recall, recall of content words, precision and the number of speakers whose own words were found.
    """
    reference = " ".join(texts[c.speaker] for c in chosen)
    per_speaker = {c.speaker: unique_recall(c.speaker, list(chosen), texts, transcript) for c in chosen}
    scored = {name: round(value, 3) for name, value in per_speaker.items() if value is not None}
    return {
        "recall": round(word_recall(reference, transcript), 3),
        "content_recall": round(word_recall(reference, transcript, MIN_CONTENT_LENGTH), 3),
        "precision": round(word_precision(reference, transcript), 3),
        "speakers_recovered": sum(value >= 0.5 for value in scored.values()),
        "speakers_scored": len(scored),
        "unique_word_recall_by_speaker": scored,
    }


def listen(ask: Ask, samples: np.ndarray) -> tuple[str, int | None, str, float, int]:
    """Asks for the transcript and the number of people in a recording.

    Args:
        ask: Function that sends a recording and a question and returns the answer.
        samples: The recording.

    Returns:
        The raw transcript, the number of people the model gave or None, the transcript without labels, the seconds
        both questions took, and the number of questions that failed.
    """
    start, failed, answers = time.perf_counter(), 0, []
    for prompt in (TRANSCRIBE, COUNT):
        try:
            answers.append(ask(samples, prompt))
        except (RuntimeError, urllib.error.URLError, TimeoutError, OSError) as error:
            logger.warning("question failed: {}", error)
            failed += 1
            answers.append("")
    raw, count = answers
    return raw, parse_count(count), strip_labels(raw), time.perf_counter() - start, failed


def run_trials(
    audio: dict[str, np.ndarray], texts: dict[str, str], ask: Ask, levels: Sequence[int], reps: int, seed: int
) -> dict:
    """Plays the clean clips and the overlap trials of the Whisper pilot to a listener.

    Args:
        audio: One clean clip per agent.
        texts: What each agent said.
        ask: Function that sends a recording and a question and returns the answer.
        levels: Numbers of simultaneous speakers.
        reps: Trials for each level.
        seed: Seed of the pilot. Trials draw with `seed + 1`, as the Whisper pilot does.

    Returns:
        The clean measurements and one row for each level.
    """
    clean = []
    for name in NAMES:
        raw, count, text, seconds, failed = listen(ask, audio[name])
        clean.append({"agent": name, "wer": round(word_error_rate(texts[name], text), 4), "count": count, "seconds": round(seconds, 2), "failed": failed})
    rng = random.Random(seed + 1)
    rows = []
    for level in levels:
        trials = []
        for _ in range(reps):
            chosen = [Clip(name, rng.uniform(0.0, 2.0), audio[name]) for name in rng.sample(NAMES, level)]
            until_s = max(c.end_s for c in chosen)
            mix = mix_for_reader(chosen, reader="Listener", until_s=until_s)
            raw, count, text, seconds, failed = listen(ask, mix)
            trials.append({
                "speakers": level, "names": [c.speaker for c in chosen], "mix_seconds": round(until_s, 1),
                "overlap_fraction": round(overlap_fraction(chosen), 3), **score(chosen, texts, text),
                "labeled_speakers": labeled_speakers(raw), "counted_speakers": count, "seconds": round(seconds, 2), "failed": failed,
                "references": {c.speaker: texts[c.speaker] for c in chosen}, "transcript": raw,
            })
        rows.append(summarize(level, trials))
        logger.info(
            "k={} recall={} precision={} speakers_recovered={} counted_right={} labeled={}",
            level, rows[-1]["recall"], rows[-1]["precision"], rows[-1]["speakers_recovered_share"],
            rows[-1]["count_right_share"], rows[-1]["labeled_speakers_mean"],
        )
    return {
        "clean": {
            "wer_mean": round(statistics.mean(c["wer"] for c in clean), 4), "wer_max": max(c["wer"] for c in clean),
            "count_right_share": round(sum(c["count"] == 1 for c in clean) / len(clean), 3),
            "seconds_mean": round(statistics.mean(c["seconds"] for c in clean), 2), "rows": clean,
        },
        "overlap": rows,
    }


def summarize(level: int, trials: list[dict]) -> dict:
    """Averages the trials of one level.

    Args:
        level: Number of simultaneous speakers.
        trials: The trials of the level.

    Returns:
        Means, the share of trials in which the model gave the right number of people, and the first trial as an example.
    """
    def mean(key: str) -> float:
        return round(statistics.mean(t[key] for t in trials), 3)

    scored = sum(t["speakers_scored"] for t in trials)
    counted = [t["counted_speakers"] for t in trials if t["counted_speakers"] is not None]
    return {
        "speakers": level,
        "trials": len(trials),
        "overlap_fraction": mean("overlap_fraction"),
        "recall": mean("recall"),
        "recall_sd": round(statistics.pstdev(t["recall"] for t in trials), 3),
        "content_recall": mean("content_recall"),
        "precision": mean("precision"),
        "speakers_recovered_share": round(sum(t["speakers_recovered"] for t in trials) / scored, 3) if scored else None,
        "labeled_speakers_mean": mean("labeled_speakers"),
        "count_right_share": round(sum(t["counted_speakers"] == level for t in trials) / len(trials), 3),
        "count_mean": round(statistics.mean(counted), 2) if counted else None,
        "seconds_mean": mean("seconds"),
        "failed_questions": sum(t["failed"] for t in trials),
        "example": {"speakers": trials[0]["names"], "references": trials[0]["references"], "transcript": trials[0]["transcript"]},
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Writes the clean clips, or plays them and the overlap trials to a running server.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    clips = steps.add_parser("clips", help="synthesize the clean clips with Kokoro")
    clips.add_argument("--out", type=Path, required=True)
    clips.add_argument("--seed", type=int, default=7)
    clips.add_argument("--device", default=None)
    trials = steps.add_parser("trials", help="play the clips and the mixes to a vLLM server")
    trials.add_argument("--clips", type=Path, required=True)
    trials.add_argument("--url", required=True)
    trials.add_argument("--model", required=True)
    trials.add_argument("--out", type=Path, required=True)
    trials.add_argument("--levels", default="1,2,3,5,10")
    trials.add_argument("--reps", type=int, default=10)
    args = parser.parse_args(argv)
    if args.step == "clips":
        from core.audio.tts import Synthesizer as Kokoro

        texts = build_clips(args.out, args.seed, Kokoro(device=args.device))
        logger.info("wrote {} clips to {}", len(texts), args.out)
        return
    seed, texts, audio = read_clips(args.clips)
    def ask(samples: np.ndarray, prompt: str) -> str:
        return ask_server(args.url, args.model, samples, prompt)

    ask(audio[NAMES[0]], COUNT)
    start = time.perf_counter()
    result = run_trials(audio, texts, ask, [int(x) for x in args.levels.split(",")], args.reps, seed)
    report = {"seed": seed, "model": args.model, "prompt": TRANSCRIBE, "count_prompt": COUNT, "texts": texts, **result, "wall_s": round(time.perf_counter() - start, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote {} in {:.0f} s, clean WER {}", args.out, report["wall_s"], result["clean"]["wer_mean"])


if __name__ == "__main__":
    main()
