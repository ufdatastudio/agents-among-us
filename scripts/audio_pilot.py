"""Audio feasibility pilot for the situation room (phase P1b).

Synthesizes a scripted meeting with one voice per agent, mixes the clips the way
the room would for a listener, and transcribes the mix with Whisper large-v3.

Measurements:
    clean: each clip transcribed alone, the ceiling for speech recognition.
    overlap: k speakers start within 2 s of each other, for several k. Reports how
        many reference words and speakers the transcript recovers.
    tick: time to synthesize one tick's utterances and to transcribe the 10
        readers' mixes of a 10 s window in one batch.

Run with the models cached and HF_HUB_OFFLINE=1:
    uv run --no-project --python 3.12 --with kokoro --with soundfile --with transformers \
        --with torch --with scipy --with numpy --with loguru --with <spacy model> \
        python -m scripts.audio_pilot --out result.json --wav-dir wavs
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from loguru import logger
import numpy as np

from core.audio.asr import Transcriber
from core.audio.metrics import normalize_words, word_error_rate, word_precision, word_recall
from core.audio.mix import SAMPLE_RATE, Clip, mix_for_reader, overlap_fraction, speakers_at, write_wav
from core.audio.tts import Synthesizer, assign_voices

# Spoken names. The game's agent colors are easy to pronounce and to recognize.
NAMES = ["Red", "Orange", "Yellow", "Lime", "Green", "Cyan", "Blue", "Purple", "Brown", "Pink"]
# Single-word room names, so a recognizer that writes "med bay" does not count as an error.
ROOMS = ["Cafeteria", "Medical", "Electrical", "Storage", "Admin", "Weapons", "Navigation", "Shields"]
TEMPLATES = [
    "I was in {room} with {other} and we both finished our tasks, so I think {accused} is lying about where they were.",
    "I found the body in {room} right after {other} walked out, and I saw {accused} heading toward {room2}.",
    "Nobody was with {accused} in {room}, which is suspicious, so I will vote for {accused} unless someone speaks up.",
    "{other} says they were in {room}, but I saw them in {room2} earlier, and that does not add up.",
]
MIN_CONTENT_LENGTH = 4


def utterance(agent: str, rng: random.Random) -> str:
    """Builds one scripted statement of about 25 words.

    Args:
        agent: Name of the speaker, who is never accused.
        rng: Random source.

    Returns:
        The statement.
    """
    others = [n for n in NAMES if n != agent]
    rooms = rng.sample(ROOMS, 2)
    return rng.choice(TEMPLATES).format(
        room=rooms[0], room2=rooms[1], other=rng.choice(others), accused=rng.choice(others)
    )


def unique_recall(speaker: str, chosen: list[Clip], texts: dict[str, str], transcript: str) -> float | None:
    """Scores how much of one speaker was recovered, using only that speaker's own words.

    The scripted statements share clauses, so a word that another speaker also
    says would credit the wrong person. Only content words that no other chosen
    speaker says are counted.

    Args:
        speaker: The speaker to score.
        chosen: The clips in the mix.
        texts: What each agent said.
        transcript: The recognized text of the mix.

    Returns:
        The share of the speaker's unique content words found, or None if the
        speaker has none.
    """
    others = {w for c in chosen if c.speaker != speaker for w in normalize_words(texts[c.speaker])}
    unique = [w for w in normalize_words(texts[speaker]) if len(w) >= MIN_CONTENT_LENGTH and w not in others]
    heard = set(normalize_words(transcript))
    return sum(w in heard for w in unique) / len(unique) if unique else None


def overlap_trial(
    audio: dict[str, np.ndarray],
    texts: dict[str, str],
    transcriber: Transcriber,
    level: int,
    rng: random.Random,
    wav_path: Path | None,
) -> dict:
    """Draws `level` speakers with random onsets, mixes them for a silent listener, and transcribes the mix.

    Args:
        audio: One clean clip per agent.
        texts: What each agent said.
        transcriber: The speech recognizer.
        level: Number of simultaneous speakers.
        rng: Random source for the speakers and their onsets.
        wav_path: Where to save the mix, or None.

    Returns:
        Measurements for this trial.
    """
    chosen = [Clip(name, rng.uniform(0.0, 2.0), audio[name]) for name in rng.sample(NAMES, level)]
    until_s = max(c.end_s for c in chosen)
    mix = mix_for_reader(chosen, reader="Listener", until_s=until_s)
    start = time.perf_counter()
    result = transcriber.transcribe_batch([mix], words=True)[0]
    asr_s = time.perf_counter() - start
    if wav_path is not None:
        write_wav(wav_path, mix)

    reference = " ".join(texts[c.speaker] for c in chosen)
    per_speaker = {c.speaker: unique_recall(c.speaker, chosen, texts, result["text"]) for c in chosen}
    scored = {name: round(value, 3) for name, value in per_speaker.items() if value is not None}
    words = result.get("words", [])
    attributable = [
        len(speakers_at(chosen, (w["start"] + (w["end"] or w["start"])) / 2)) == 1
        for w in words
        if w["start"] is not None
    ]
    return {
        "speakers": level,
        "names": [c.speaker for c in chosen],
        "references": {c.speaker: texts[c.speaker] for c in chosen},
        "mix_seconds": round(until_s, 1),
        "overlap_fraction": round(overlap_fraction(chosen), 3),
        "recall": round(word_recall(reference, result["text"]), 3),
        "content_recall": round(word_recall(reference, result["text"], MIN_CONTENT_LENGTH), 3),
        "precision": round(word_precision(reference, result["text"]), 3),
        "speakers_recovered": sum(value >= 0.5 for value in scored.values()),
        "speakers_scored": len(scored),
        "unique_word_recall_by_speaker": scored,
        "words_with_one_speaker": round(sum(attributable) / len(attributable), 3) if attributable else None,
        "asr_seconds": round(asr_s, 2),
        "transcript": result["text"],
    }


def summarize_trials(level: int, trials: list[dict]) -> dict:
    """Averages the trials of one overlap level.

    Args:
        level: Number of simultaneous speakers.
        trials: Output of overlap_trial() for this level.

    Returns:
        Means over the trials, the spread of recall, and the first trial's transcript as an example.
    """

    def mean(key: str) -> float:
        values = [t[key] for t in trials if t[key] is not None]
        return round(statistics.mean(values), 3) if values else None

    recovered = sum(t["speakers_recovered"] for t in trials)
    scored = sum(t["speakers_scored"] for t in trials)
    return {
        "speakers": level,
        "trials": len(trials),
        "overlap_fraction": mean("overlap_fraction"),
        "recall": mean("recall"),
        "recall_sd": round(statistics.pstdev(t["recall"] for t in trials), 3),
        "content_recall": mean("content_recall"),
        "precision": mean("precision"),
        "speakers_recovered_share": round(recovered / scored, 3) if scored else None,
        "words_with_one_speaker": mean("words_with_one_speaker"),
        "asr_seconds_mean": mean("asr_seconds"),
        "example": {
            "speakers": trials[0]["names"],
            "references": trials[0]["references"],
            "transcript": trials[0]["transcript"],
        },
    }


def main() -> None:
    """Runs the clean, overlap and tick measurements and writes a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--levels", default="1,2,3,5,10")
    parser.add_argument("--reps", type=int, default=10, help="trials per overlap level")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    voices = assign_voices(NAMES, args.seed)
    texts = {name: utterance(name, rng) for name in NAMES}

    synthesizer = Synthesizer()
    synthesizer.synthesize("Warming up the voice.", voices[NAMES[0]])
    audio, tts_rows = {}, []
    for name in NAMES:
        start = time.perf_counter()
        audio[name] = synthesizer.synthesize(texts[name], voices[name])
        elapsed = time.perf_counter() - start
        seconds = len(audio[name]) / SAMPLE_RATE
        tts_rows.append({"agent": name, "voice": voices[name], "audio_s": round(seconds, 2), "synth_s": round(elapsed, 2)})
        logger.info("TTS {} {}: {:.1f} s of audio in {:.2f} s", name, voices[name], seconds, elapsed)
        if args.wav_dir is not None:
            write_wav(args.wav_dir / f"clean_{name}.wav", audio[name])

    transcriber = Transcriber()
    transcriber.transcribe_batch([audio[NAMES[0]]])
    start = time.perf_counter()
    clean = transcriber.transcribe_batch([audio[n] for n in NAMES])
    clean_s = time.perf_counter() - start
    wers = [word_error_rate(texts[n], c["text"]) for n, c in zip(NAMES, clean)]
    logger.info("Clean speech: mean WER {:.3f}, max {:.3f}, batch of 10 in {:.1f} s", statistics.mean(wers), max(wers), clean_s)

    trial_rng = random.Random(args.seed + 1)
    levels = []
    for level in [int(x) for x in args.levels.split(",")]:
        trials = []
        for rep in range(args.reps):
            wav_path = args.wav_dir / f"mix_k{level}.wav" if args.wav_dir is not None and rep == 0 else None
            trials.append(overlap_trial(audio, texts, transcriber, level, trial_rng, wav_path))
        row = summarize_trials(level, trials)
        levels.append(row)
        logger.info(
            "k={} overlap={} recall={}±{} content_recall={} precision={} speakers_recovered={} one_speaker_words={}",
            level, row["overlap_fraction"], row["recall"], row["recall_sd"], row["content_recall"],
            row["precision"], row["speakers_recovered_share"], row["words_with_one_speaker"],
        )

    # One tick of the cascade ear: every reader transcribes the mix of the other 9 voices.
    until_s = 10.0
    onset_rng = random.Random(args.seed + 2)
    clips = [Clip(n, onset_rng.uniform(0.0, 2.0), audio[n]) for n in NAMES]
    mixes = [mix_for_reader(clips, reader=n, until_s=until_s) for n in NAMES]
    start = time.perf_counter()
    transcriber.transcribe_batch(mixes)
    tick_asr_s = time.perf_counter() - start
    logger.info("Tick: 10 reader mixes of {:.0f} s transcribed in {:.1f} s", until_s, tick_asr_s)

    report = {
        "seed": args.seed,
        "texts": texts,
        "words_per_text_mean": round(statistics.mean(len(normalize_words(t)) for t in texts.values()), 1),
        "tts": tts_rows,
        "clean": {
            "wer_mean": round(statistics.mean(wers), 4),
            "wer_max": round(max(wers), 4),
            "batch_of_10_seconds": round(clean_s, 2),
        },
        "overlap": levels,
        "tick": {"window_s": until_s, "readers": len(NAMES), "asr_seconds": round(tick_asr_s, 2)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
