"""Diarization pilot for the situation room (phase P1c).

Synthesizes scripted statements with one Kokoro voice per agent, mixes them the way the room would for a silent
listener, and asks Nemotron 3 Diarization who spoke when. The room knows the true segments of every clip, so each
trial gets a diarization error rate against them.

Measurements:
    overlap: k speakers start within a window of w seconds, for several k and w. Shows how the error grows with the
        number of speakers and with the share of speech that overlaps. The model labels at most 8 speakers, and a
        listener in a 10 agent room hears 9 voices.
    meeting: 5, 7 or 9 speakers with 3 statements each, spread over a meeting of 60 s or 180 s. Shows whether labels
        stay attached to the same voice over a whole discussion, and whether the limit of 8 speakers or the
        similarity of the voices is the cause when speakers are lost.

Each mix is scored with the model seeing the whole recording and with streaming input at one latency.

Run with the models cached and HF_HUB_OFFLINE=1:
    uv run --no-project --python 3.12 --with kokoro --with soundfile --with transformers --with torch \
        --with scipy --with numpy --with loguru --with librosa --with <spacy model> \
        python -m scripts.diarization_pilot --out result.json --wav-dir wavs
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from loguru import logger
import numpy as np

from core.audio.der import diarization_error_rate, oracle_segments
from core.audio.diarize import Diarizer
from core.audio.mix import SAMPLE_RATE, Clip, mix_for_reader, overlap_fraction, write_wav
from core.audio.tts import Synthesizer, assign_voices
from scripts.audio_pilot import NAMES, utterance

TTS_SPEED = 0.85
MIN_GAP_S = 0.5


def score(mix: np.ndarray, clips: list[Clip], diarizer: Diarizer, modes: list[str | None]) -> dict:
    """Diarizes a mix in each mode and scores it against the clips.

    Args:
        mix: What the listener hears.
        clips: The clips that make up the mix.
        diarizer: The model.
        modes: Streaming modes to run, where None is the whole recording at once.

    Returns:
        For each mode, the parts of the error rate, the number of speakers found and the run time.
    """
    reference = oracle_segments(clips)
    scores = {}
    for mode in modes:
        start = time.perf_counter()
        segments = diarizer.diarize(mix, SAMPLE_RATE, streaming_mode=mode)
        elapsed = time.perf_counter() - start
        result = diarization_error_rate(reference, segments)
        scores[mode or "offline"] = {
            "der": round(result.der, 4),
            "miss": round(result.miss_s / result.reference_s, 4),
            "false_alarm": round(result.false_alarm_s / result.reference_s, 4),
            "confusion": round(result.confusion_s / result.reference_s, 4),
            "speakers_found": len({s.speaker for s in segments}),
            "seconds": round(elapsed, 3),
        }
    return scores


def overlap_trial(
    audio: dict[str, np.ndarray], level: int, window_s: float, rng: random.Random, diarizer: Diarizer, modes: list[str | None]
) -> tuple[dict, np.ndarray]:
    """Draws `level` speakers with onsets inside a window and scores the mix.

    Args:
        audio: One clean clip per agent.
        level: Number of speakers.
        window_s: Onsets are drawn uniformly from 0 to this many seconds.
        rng: Random source.
        diarizer: The model.
        modes: Streaming modes to run.

    Returns:
        The trial's row and the mix.
    """
    clips = [Clip(name, rng.uniform(0.0, window_s), audio[name]) for name in rng.sample(NAMES, level)]
    until_s = max(c.end_s for c in clips)
    mix = mix_for_reader(clips, reader="Listener", until_s=until_s)
    row = {
        "kind": "overlap", "speakers": level, "window_s": window_s, "mix_s": round(until_s, 1),
        "overlap_fraction": round(overlap_fraction(clips), 3), "scores": score(mix, clips, diarizer, modes),
    }
    return row, mix


def meeting_trial(
    synthesizer: Synthesizer,
    voices: dict[str, str],
    speakers: int,
    length_s: float,
    statements: int,
    rng: random.Random,
    diarizer: Diarizer,
    modes: list[str | None],
) -> tuple[dict, np.ndarray]:
    """Builds a meeting with several statements per speaker and scores the mix a listener hears.

    A speaker never talks over itself, as in the room.

    Args:
        synthesizer: Kokoro.
        voices: A voice for each agent name.
        speakers: Number of speakers the listener hears, at most 9 because the listener is one of the 10 agents.
        length_s: Statements start inside this many seconds.
        statements: Statements per speaker.
        rng: Random source.
        diarizer: The model.
        modes: Streaming modes to run.

    Returns:
        The trial's row and the mix.
    """
    listener = rng.choice(NAMES)
    clips = []
    for name in rng.sample([n for n in NAMES if n != listener], speakers):
        previous_end = 0.0
        for start in sorted(rng.uniform(0.0, length_s - 8.0) for _ in range(statements)):
            samples = synthesizer.synthesize(utterance(name, rng), voices[name], speed=TTS_SPEED)
            clip = Clip(name, max(start, previous_end + MIN_GAP_S), samples)
            clips.append(clip)
            previous_end = clip.end_s
    until_s = max(c.end_s for c in clips)
    mix = mix_for_reader(clips, reader=listener, until_s=until_s)
    row = {
        "kind": "meeting", "speakers": speakers, "window_s": length_s, "mix_s": round(until_s, 1),
        "overlap_fraction": round(overlap_fraction(clips), 3), "scores": score(mix, clips, diarizer, modes),
    }
    return row, mix


def summarize(rows: list[dict]) -> list[dict]:
    """Averages the trials that share a kind, a number of speakers and a window.

    Args:
        rows: Trial rows.

    Returns:
        One row per group with the mean overlap and, for each mode, the mean parts of the error rate.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["kind"], row["speakers"], row["window_s"]), []).append(row)
    summary = []
    for (kind, speakers, window_s), group in groups.items():
        entry = {
            "kind": kind, "speakers": speakers, "window_s": window_s, "trials": len(group),
            "mix_s": round(statistics.mean(r["mix_s"] for r in group), 1),
            "overlap_fraction": round(statistics.mean(r["overlap_fraction"] for r in group), 3),
        }
        for mode in group[0]["scores"]:
            entry[mode] = {
                key: round(statistics.mean(r["scores"][mode][key] for r in group), 4)
                for key in ("der", "miss", "false_alarm", "confusion", "speakers_found", "seconds")
            }
            entry[mode]["der_sd"] = round(statistics.pstdev(r["scores"][mode]["der"] for r in group), 4)
        summary.append(entry)
    return summary


def main() -> None:
    """Runs the overlap and meeting measurements and writes a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--levels", default="1,2,3,5,8,9,10")
    parser.add_argument("--windows", default="2,6,12", help="onset windows in seconds for the overlap trials")
    parser.add_argument("--reps", type=int, default=10, help="trials per overlap cell")
    parser.add_argument("--meeting-lengths", default="60,180")
    parser.add_argument("--meeting-speakers", default="5,7,9", help="speakers heard in a meeting")
    parser.add_argument("--meeting-reps", type=int, default=10)
    parser.add_argument("--statements", type=int, default=3, help="statements per speaker in a meeting")
    parser.add_argument("--streaming-mode", default="low_latency", help="streaming mode beside the offline run, or none")
    args = parser.parse_args()

    modes: list[str | None] = [None] + ([] if args.streaming_mode == "none" else [args.streaming_mode])
    rng = random.Random(args.seed)
    voices = assign_voices(NAMES, args.seed)
    synthesizer = Synthesizer()
    synthesizer.synthesize("Warming up the voice.", voices[NAMES[0]])
    audio = {name: synthesizer.synthesize(utterance(name, rng), voices[name], speed=TTS_SPEED) for name in NAMES}
    diarizer = Diarizer()
    diarizer.diarize(audio[NAMES[0]], SAMPLE_RATE)

    rows = []
    for window_s in [float(x) for x in args.windows.split(",")]:
        for level in [int(x) for x in args.levels.split(",")]:
            for rep in range(args.reps):
                row, mix = overlap_trial(audio, level, window_s, rng, diarizer, modes)
                rows.append(row)
                if args.wav_dir is not None and rep == 0:
                    write_wav(args.wav_dir / f"overlap_k{level}_w{window_s:g}.wav", mix)
            cell = [r for r in rows if r["kind"] == "overlap" and r["speakers"] == level and r["window_s"] == window_s]
            logger.info(
                "overlap k={} window={:g}s overlap={:.2f} {}", level, window_s,
                statistics.mean(r["overlap_fraction"] for r in cell),
                {m: round(statistics.mean(r["scores"][m]["der"] for r in cell), 3) for m in cell[0]["scores"]},
            )
    for length_s in [float(x) for x in args.meeting_lengths.split(",")]:
        for speakers in [int(x) for x in args.meeting_speakers.split(",")]:
            for rep in range(args.meeting_reps):
                row, mix = meeting_trial(synthesizer, voices, speakers, length_s, args.statements, rng, diarizer, modes)
                rows.append(row)
                if args.wav_dir is not None and rep == 0:
                    write_wav(args.wav_dir / f"meeting_{length_s:g}s_k{speakers}.wav", mix)
            cell = [r for r in rows if r["kind"] == "meeting" and r["window_s"] == length_s and r["speakers"] == speakers]
            logger.info(
                "meeting {:g}s k={} overlap={:.2f} {}", length_s, speakers, statistics.mean(r["overlap_fraction"] for r in cell),
                {m: round(statistics.mean(r["scores"][m]["der"] for r in cell), 3) for m in cell[0]["scores"]},
            )

    report = {
        "seed": args.seed, "tts_speed": TTS_SPEED, "modes": [m or "offline" for m in modes],
        "summary": summarize(rows), "trials": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
