"""Tests for the native-ear pilot, with a fake listener and a fake server."""

import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from core.audio.mix import SAMPLE_RATE
from scripts.audio_pilot import NAMES
from scripts.omni_pilot import (
    COUNT,
    TRANSCRIBE,
    ask_server,
    build_clips,
    chat_body,
    labeled_speakers,
    parse_count,
    read_clips,
    run_trials,
    strip_labels,
    wav_bytes,
)
from tests.conftest import FakeVoice


@pytest.fixture()
def clips(tmp_path):
    texts = build_clips(tmp_path / "clips", seed=7, synth=FakeVoice(0.05), voices={name: "v0" for name in NAMES})
    return read_clips(tmp_path / "clips"), texts


def test_clips_and_statements_are_written_and_read_back(clips):
    (seed, texts, audio), written = clips
    assert seed == 7 and texts == written and sorted(audio) == sorted(NAMES)
    assert all(len(samples) > 0 for samples in audio.values())


def test_the_statements_are_those_of_the_whisper_pilot_for_the_same_seed(clips):
    import random

    from scripts.audio_pilot import utterance

    rng = random.Random(7)
    assert clips[1] == {name: utterance(name, rng) for name in NAMES}


def test_wav_bytes_make_a_wav_file_that_reads_back():
    samples = np.sin(np.linspace(0, 100, SAMPLE_RATE)).astype(np.float32) * 0.5
    with wave.open(io.BytesIO(wav_bytes(samples)), "rb") as handle:
        assert (handle.getnchannels(), handle.getframerate(), handle.getnframes()) == (1, SAMPLE_RATE, SAMPLE_RATE)


def test_the_request_has_the_recording_the_question_and_the_instruct_settings():
    body = chat_body("m", np.zeros(SAMPLE_RATE, dtype=np.float32), TRANSCRIBE)
    audio, text = body["messages"][0]["content"]
    assert audio["type"] == "audio_url" and audio["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert text == {"type": "text", "text": TRANSCRIBE}
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["top_k"] == 1 and body["temperature"] == 0.2


def test_labels_are_removed_and_counted():
    raw = "Speaker 1: I was in Storage.\nSpeaker 2: Red lied.\nSpeaker 1: Really."
    assert strip_labels(raw) == "I was in Storage. Red lied. Really."
    assert labeled_speakers(raw) == 2 and labeled_speakers("no labels") == 0


@pytest.mark.parametrize(
    "text, count",
    [("3", 3), ("There are three people talking.", 3), ("I hear about 12 voices", 12), ("Several", None), ("", None)],
)
def test_the_number_of_people_is_read_from_the_answer(text, count):
    assert parse_count(text) == count


def test_a_listener_that_writes_every_word_gets_full_recall_and_the_right_counts(clips):
    (seed, texts, audio), _ = clips
    everything = " ".join(texts.values())

    def ask(samples, prompt):
        return "3" if prompt == COUNT else f"Speaker 1: {everything}"

    result = run_trials(audio, texts, ask, [1, 3], reps=2, seed=seed)
    assert result["clean"]["wer_mean"] > 0 and result["clean"]["count_right_share"] == 0.0
    one, three = result["overlap"]
    assert one["recall"] == 1.0 and three["recall"] == 1.0
    assert one["precision"] < 1.0
    assert one["speakers_recovered_share"] == 1.0 and three["speakers_recovered_share"] == 1.0
    assert one["count_right_share"] == 0.0 and three["count_right_share"] == 1.0
    assert one["labeled_speakers_mean"] == 1.0


def test_a_listener_that_hears_nothing_scores_zero_and_failures_are_counted(clips):
    (seed, texts, audio), _ = clips

    def ask(samples, prompt):
        raise RuntimeError("server answered 500")

    row = run_trials(audio, texts, ask, [2], reps=2, seed=seed)["overlap"][0]
    assert row["recall"] == 0.0 and row["speakers_recovered_share"] == 0.0
    assert row["failed_questions"] == 4 and row["count_mean"] is None


def test_the_trials_use_the_speaker_subsets_of_the_whisper_pilot(clips):
    import random

    (seed, texts, audio), _ = clips
    seen = []

    def ask(samples, prompt):
        seen.append(len(samples))
        return ""

    run_trials(audio, texts, ask, [3], reps=1, seed=seed)
    rng = random.Random(seed + 1)
    names = rng.sample(NAMES, 3)
    onsets = [rng.uniform(0.0, 2.0) for _ in names]
    expected = round(max(o + len(audio[n]) / SAMPLE_RATE for n, o in zip(names, onsets)) * SAMPLE_RATE)
    assert seen[-2] == expected


class Answer(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        Answer.seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        reply = json.dumps({"choices": [{"message": {"content": "Speaker 1: hello"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


def test_ask_server_posts_the_request_and_returns_the_answer():
    server = HTTPServer(("127.0.0.1", 0), Answer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        text = ask_server(f"http://127.0.0.1:{server.server_port}/v1", "m", np.zeros(100, dtype=np.float32), COUNT)
    finally:
        server.shutdown()
    assert text == "Speaker 1: hello"
    path, body = Answer.seen[-1]
    assert path == "/v1/chat/completions" and body["model"] == "m"
