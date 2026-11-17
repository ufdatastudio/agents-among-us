# Duplex sessions for the room

Status is built and tested with a scripted engine. The first live meeting ran on Gemma 4 31B on 2026-11-17, and the Omni audio run is queued. The numbers below come from that one text meeting, so they describe the channel and say nothing yet about how well agents play.

## Why

The tick loop of the room rebuilds a whole prompt for each agent at every tick, and a statement enters the ledger when it ends. Speech that overlaps is then rendered as windows of words in a text percept. That design cannot serve a model that hears audio while it speaks, and it re-reads the whole meeting at each tick.

A duplex session changes three things.

- A statement is written to the ledger as chunks while it is spoken. A chunk is a few words or a slice of audio, with the time at which it was fully spoken.
- An agent keeps one session with its model for the whole meeting. At each beat it receives the chunks that others finished since the last beat, and the model writes a short burst of tokens that become the agent's own chunks.
- Each reader follows the ledger through a subscription that remembers its position, so the same log serves the agents, a spectator and a replay.

This is time-sliced duplex. The model listens and speaks inside one context at the resolution of a beat, and it does not emit tokens while it receives them. No model on the eligible list does that. Open speech-to-speech models exist, such as NVIDIA PersonaPlex-7B, which is built on Moshi. Reports say its synthetic training dialogues were generated with Qwen3-32B and gpt-oss-120b, and vLLM does not serve it as far as I found. I have not read its model card, so its eligibility is undecided.

## Parts

| File | What it does |
|------|--------------|
| `core/ledger.py` | Adds the `chunk` record type and `Subscription`. A subscription returns each record once, only to a reader who may read it, and only after the record's `time_s`. A record that is not audible yet delays the later ones, so a reader never sees a record without the earlier ones. `wait()` lets a live reader sleep until a record arrives |
| `core/stream.py` | `StreamWriter` appends the chunks of one statement for text or audio. `render_heard` turns a batch of new chunks into lines such as `[00:12] Agent_5: words` and gives a model that hears the audio arrays in the order of their placeholders. The lines of one batch do not depend on earlier batches, so text a model has read is never rewritten. `write_replay` puts recorded clips into the ledger in the order they were spoken |
| `core/duplex.py` | `TurnFormat` finds how a chat template joins turns by rendering sentinel messages. `DuplexAgent` is one agent with a session, and `run_duplex_meeting` plays the beats. At a beat every agent reads one snapshot, all agents answer at once, and the bursts are written after all have arrived |
| `core/vllm_stream.py` | `VllmHandle` is a session on vLLM's streaming input. It leaves vLLM unchanged |
| `scripts/duplex_spike.py`, `scripts/hipergator/duplex_spike.sbatch` | Plays a meeting on one GPU. A small scenario with a body in Storage, or a replay of a recorded game with live listeners |
| `scripts/duplex_viewer.py`, `scripts/duplex_viewer.html` | The dashboard described at the end |

## What vLLM provides

vLLM takes a request whose prompt arrives in pieces ([post on streaming input](https://vllm.ai/blog/streaming-realtime)). I read the code of 0.34.0, which the cluster jobs use.

- The first piece starts a session, and later pieces are appended to the same context and its cached keys and values. The model writes after each piece up to the token limit of that piece. The limit can differ for each piece.
- When a piece is appended, the engine keeps the output tokens but the last sampled one, because that token has no cached keys and values yet. `core/vllm_stream.py` decodes the text of a burst from its token ids without that token, so the ledger matches the context.
- Stop strings, `n > 1` and final-only output are refused. Idle sessions are not preempted, so many sessions use memory while they wait.
- The route is the engine API, `AsyncLLM.generate` with an async generator of `StreamingInput`. I found no route on `/v1/chat/completions`. The Realtime WebSocket serves transcription models only.
- Each burst is one assistant message. The chat template writes the end of that message and the start of the next user message between bursts, and `TurnFormat` copies them from the template. A burst that continues a sentence starts without the space that a token inside a sentence has, so `DuplexAgent.commit` adds one.

## Audio

An audio chunk is a slice of a WAV file, stored as a path, a sample range and the words of the slice. A model that hears gets the slice as an array and a placeholder in the text. A model that cannot hear gets the words. The Omni placeholder is `<so_embedding>`. Whether Omni accepts audio pieces in a streaming session is what the replay run tests. Its listener writes text, because the model has no voice, so the other agents of a replay are recorded Kokoro clips.

## First meeting

Gemma 4 31B, 5 agents, 90 s of room time in beats of 2 s, bursts of at most 8 tokens, thinking off, one RTX PRO 6000 (run `gemma-a`).

| Measure | Value |
|---------|-------|
| Wall time | 23 s for 45 beats, 0.49 s per beat from the first beat to the last |
| Tokens sent per agent in a beat | about 90, with a context that grew to about 4.4k tokens |
| Beats with words | 99.6% |
| Voices at once | 4.98 on average, and 2 or more in every beat |
| Bursts cut at the token limit | 44.9% |
| Statements | 125, with 6.3 words each |
| Statements that repeat earlier words | 17.6% |

The flat time per beat while the context grows fits the session keeping its cache. I did not read the cache counters.

The agents answered each other within 1 to 2 beats. They also spoke in almost every beat, although the rules asked them to listen, and many fell into loops such as "Vote 0! Do it!". Words were glued at the seams of bursts ("Agent_0was"), which the join of bursts now fixes. Runs with a stronger silence rule and with beats of 4 s are queued.

## Dashboard

`scripts/duplex_viewer.py` reads the result folders of `scripts/duplex_spike.py` and writes one HTML file with the data inside, so it opens without a server.

```bash
uv run python -m scripts.duplex_viewer results/duplex/gemma-a results/duplex/gemma-b --out results/duplex/viewer.html
```

- A table compares the runs. Its measures are the share of beats with words, the share of bursts cut at the limit, statements and words in each, the share of statements that repeat earlier words, the voices at once, the share of beats with two or more voices, the engine time per beat and the tokens sent. They describe the channel and say nothing about the quality of the talk.
- A slider, play and speed controls move through the beats. Lanes show who spoke when, and a strip counts the voices at once.
- The transcript up to the beat is listed by statement or by beat. The by-beat list shows the interleaving as the listeners received it.
- An inspector shows, for one agent and beat, what it heard, what the model wrote, whether the burst ended by the limit or by the model, the tokens and the latency, and the exact text sent to the model.

Add `--fragment` to leave out the document tags, which the artifact tool wants.

## Running

```bash
rsync -a --delete --exclude __pycache__ agents config core scripts hpg:<work>/duplex/
OUT=results/duplex/gemma-b sbatch scripts/hipergator/duplex_spike.sbatch gemma-4-31b-it
```

For Omni, set `VLLM_WITH="vllm[audio]"`, `ENGINE_ARGS` to the JSON of the engine arguments and `SPIKE_ARGS` to the replay options. The header of the sbatch file has the full line.

## Open questions

- Whether a model can be told to stay silent in most beats, and what that does to the talk.
- Whether Omni takes audio pieces in a session, and whether hearing the audio changes what it says compared with reading the words.
- How votes work after a duplex meeting. The meeting here ends without one.
- Whether the ledger board of condition B can read chunks, so that claims are found in speech that is still being spoken.
