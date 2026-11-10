# Situation Room

Design for timed, simultaneous discussion in Agents Among Us, plus the experiments that use it.
Status is design only. Nothing here is implemented yet. Work happens on the `situation-room` branch.

## Goal

Replace turn-taking with a situation room. All agents speak during the same timed window, their speech overlaps, and each agent works out who said what, and why, from the combined record.
Text is one modality. Agents also hear each other as audio and learn who is talking through diarization. They argue their points by answering the justifications they heard.
The experiments compare how models handle this across scenarios.
They run on HiPerGator with local models and make no API calls.

Each agent is an adapter that hears the room and replies to it. An adapter can run in the controller process, on its own GPU, on another node, or behind an API, and the room does not change.

## How discussion works today

- `GameEngine.run_discussion_phase` (`core/game_engine.py:694`) loops over agents in a fixed order, caller first, for `num_discussion_messages` passes.
- Each agent reads its role's discussion log file (`participate_in_discussion`, `agents/honest_agent.py:159`), so every message sees all earlier ones. No two agents speak at once.
- `vote()` reads the same file (`agents/honest_agent.py:221`), and votes are cast one agent at a time.
- In CONTROLLER mode each generation writes a JSON request file and polls for the reply (`core/llm.py:373`). `worker.py` serves one request at a time with a single-prompt HF `generate`, so parallel calls would queue on the GPU.
- Request ids use `time.time()` and the pid, which can collide across threads.
- Nothing is seeded. `core/state.py` picks start rooms with `random.choice`.

## Design

### Virtual clock

Discussion runs on a virtual clock. Each word costs `1 / words_per_s` seconds of room time for every model.
Inference speed has no effect on who gets to speak, so a 70B model and an 8B model are charged the same.
A wall-clock timer can be added later as its own scenario if inference latency becomes a variable of interest.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `discussion_mode` | `sequential` | `sequential` keeps today's behavior, `timed` uses the situation room |
| `budget_s` | 60 | Length of the discussion in room time |
| `tick_s` | 10 | Time between decision points |
| `words_per_s` | 2.5 | Speaking rate in the `text` modality, about 150 words per minute. The `audio` modality uses the length of the synthesized speech |
| `onset_jitter_s` | 2 | Seeded delay before a speaker starts within a tick |
| `quiet_close_ticks` | 2 | Close early after this many ticks with nobody speaking, 0 disables |
| `overlap` | `clean` | `clean`, `interleaved` or `noisy` |
| `window_s` | 2 | Width of one interleaving window |
| `dropout` | 0.0 | Fraction of words replaced by `[inaudible]` in `noisy` mode |
| `tag_policy` | `every_chunk` | `every_chunk` or `first_chunk` speaker tags in `noisy` mode |
| `modality` | `text` | `text` or `audio`, what agents hear and say. Set per agent or for the whole room |
| `diarization` | `oracle` | `oracle`, `estimated` or `none`, how speaker identity reaches an agent in the `audio` modality |
| `barge_in` | `off` | With `on`, an agent that is mid-utterance can reply at a tick, which cuts off its current utterance at that moment |
| `seed` | none | Controls start rooms, seating, onset jitter, dropout and sampling |

These sit in the composition JSON next to `skip_discussion` and `num_discussion_messages`.

### Tick loop

```
t = 0
while t < budget_s:
    ready = adapters of active agents whose last utterance has finished by t
    for a in ready:
        a.hear(room.percept(a.name, t))
    replies = collect(a.reply(Tick(t)) for a in ready)
    for r in replies:
        if not silent(r):
            room.append(r.agent, r.text, start=t + jitter(r.agent, t))
    t += tick_s
room.close(budget_s)
```

- All ready agents hear the same room state at `t` and reply in parallel. The controller commits the replies after every reply has arrived or the timeout passes. A missing reply counts as silence.
- An utterance occupies its speaker from `start` to `start + words / words_per_s`. The speaker cannot speak again until it ends, unless `barge_in` is `on`. In the `audio` modality the duration is the length of the synthesized speech.
- Listeners at later ticks hear only the words spoken so far. A 40-word message that starts at 0 s takes 16 s, so a listener at the 10 s tick has heard its first 25 words.
- Words not yet spoken when the budget closes are dropped.
- At tick 1 everyone has heard only the meeting report, so most agents speak at once. Later ticks are replies to what was heard.
- Replying `<silent/>` keeps an agent quiet for that tick.
- A meeting costs at most `budget_s / tick_s` calls per active agent. With 10 agents and the defaults that is 60 calls against 20 today. Busy agents make no call and silent replies are short, and prefix caching reduces the prompt cost.

Voting becomes one more simultaneous tick. Every agent sees the final snapshot and votes in parallel. The tally and ejection logic stay as they are.

### The room

The room is a grow-only map of per-author logs.

```
Utterance  author, tick, start, words, text
Lane       append-only list of one author's utterances, written by that author only
Room       lanes by author, free_at(author), append(), render(reader, t), close(t), ground_truth()
```

- Each lane has a single writer, so merging lanes is a union and nothing conflicts. A read takes a frozen snapshot and merges the lanes by timestamp with `heapq.merge`.
- The controller is the only committer, so the room needs no locks. Concurrency sits in the agent adapters, which generate in parallel.
- Everything in a run follows from the seed and the logged replies, so transcripts can be audited and analyses reproduced.

Alternatives considered.

- A lock-protected shared list with arrival-order posting. Message order would follow GPU speed, and model speed would become a game variable.
- Redis or Kafka. A new service on HiPerGator with no benefit at 10 writers.
- Asynchronous posting on the wall clock. It has the same arrival-order problem as the shared list.

### Agent adapters

The room knows nothing about models, GPUs or APIs, and it stores only public speech. Each agent is an adapter that hears the room and replies to it.
Private state (role, action log, results log) stays with the adapter. The controller hands it over once, at the start of a meeting.

```
Percept   reader, tick, t, cursor, modality, text, audio_ref, speakers
Tick      tick, t, remaining_s
Reply     agent, tick, text, addressed_to, or silence
Vote      agent, target

AgentAdapter
    begin_meeting(meeting, private)     role, action log, results log, meeting reason
    hear(percept)                       speech since the adapter's cursor
    reply(tick) -> Reply                an utterance, or silence
    vote(percept, candidates) -> Vote
    end_meeting()
```

| Adapter | Where the agent runs |
|---------|----------------------|
| `InProcessAdapter` | Wraps today's agent classes and `ModelManager` in the controller process. Tests, small runs and the PubApps demo |
| `VllmAdapter` | Calls a vLLM server through the existing `OpenAICompatibleClient`. One GPU per agent, or several agents per GPU |
| `ApiAdapter` | Navigator, OpenAI or Anthropic through the existing clients. Not used for the HiPerGator experiments |
| `RemoteAdapter` | A proxy for an adapter running in another process or node, reached over a transport |
| `HumanAdapter` | The existing human input path, added later |

The composition JSON picks a backend per agent. When it does not, the model name decides as it does today, with a provider prefix meaning API and a Hugging Face id meaning local.

Hearing is incremental. `hear` delivers only what is new since the adapter's cursor, the way a consumer offset works in a log. An adapter keeps its own running context, and a vLLM server can reuse the cached prefix.

Each adapter receives only its own render. Other readers' renders, dropout masks and the ground truth stay with the controller.

`Percept.modality` is `text` or `audio`. For `audio`, `audio_ref` points to the reader's mix on the shared filesystem and `speakers` lists the diarization labels the reader is allowed to see. The next section describes how an adapter hears and speaks in each modality.

The v1 barrier needs every adapter to reply once per tick. The same interface supports adapters that hear continuously and post when ready, which is the wall-clock scenario listed under open questions.

### Transports

Messages are plain JSON, so an adapter can run in another process, on another GPU or node, or behind an API without code changes.

| Transport | Use |
|-----------|-----|
| `InProcess` | Direct calls on a thread pool. Tests and single-GPU runs |
| `File` | JSON files in the IPC directory on the shared filesystem, one inbox and one outbox per agent, written with an atomic rename. Follows the convention in `worker.py` and `core/llm.py`, and works across SLURM nodes without opening ports |
| `Network` | HTTP or ZeroMQ carrying the same JSON. Later |

The controller owns the virtual clock, so adapters with different latencies stay in step. A reply tagged with an earlier tick is discarded and logged.

### Hearing and speaking

An adapter is made of an ear, a brain and a mouth. The brain is the language model that decides what to say. It runs on a vLLM server or an API and does not change with the modality.

| Part | Job | `text` modality | `audio` modality |
|------|-----|-----------------|------------------|
| Ear | Turns a percept into model input | `TextEar` passes the rendered transcript | `CascadeEar` runs speech recognition and diarization and builds a labeled transcript. `NativeEar` passes the audio to an audio-capable model |
| Mouth | Turns a reply into an utterance | `TextMouth` stores the words with timestamps | `TtsMouth` synthesizes speech in the agent's voice, and timestamps come from the audio length |

Ears and mouths are chosen per agent, so one game can mix modalities.

The audio pipeline.

1. Each agent gets a voice at game start, drawn with the seed. The voice is the agent's speaker identity for the whole game.
2. The mouth writes a WAV file to the shared filesystem. The room stores the text, the audio path and the duration. Onsets stay on the virtual clock, so synthesis speed does not affect the game.
3. The room mixes audio per reader. The mix at time `t` sums every other speaker's audio up to `t`, each placed at its start offset. The reader's own voice is left out, which takes the place of `(you)`. The overlap is acoustic, and the controller does not cut or reorder anything. Per-speaker gain and background noise take the place of `dropout`.
4. The ear hands the brain what it can use. A cascade ear transcribes the mix, labels speakers according to `diarization`, and writes the same interleaved transcript the `text` modality uses. A native ear sends the waveform since the reader's cursor together with the labels.
5. The mix and the transcripts are logged, so any reader's view of any tick can be replayed.

Candidate components, to confirm in P1b for license, availability on HiPerGator, and the country rule under Model eligibility.

- Speech recognition. Whisper large-v3.
- Speech synthesis. Kokoro.
- Diarization. pyannote or NVIDIA NeMo.
- Audio-capable language models for the native ear. Voxtral (Mistral), Gemma 3n (Google) and Granite Speech (IBM).

Qwen audio models are excluded by the country rule.

### Diarization

Speaker identity is its own channel, with 3 levels.

| Level | What the agent receives |
|-------|-------------------------|
| `oracle` | The exact speaker labels and segment times from the room. In the cascade ear every transcribed span is tagged with the true speaker |
| `estimated` | Labels from a diarization model run on the mix. It can merge or split speakers and miss overlapped speech, and agents see those errors |
| `none` | No labels. The agent separates speakers by voice and content, which needs a native ear. A cascade ear delivers unlabeled text |

A roll call before the discussion, outside the budget, lets each agent say its name once. Under `none`, that is how agents bind voices to names.

The `text` modality has matching conditions. `every_chunk` tags correspond to `oracle`, `first_chunk` tags to a degraded channel, and removing tags to `none`. That allows comparing modalities at matched diarization.

The room records the true segments, so the diarization error rate for each reader and meeting is logged under `estimated`. Game outcomes can then be related to how well speakers were identified.

### Arguing

Agents argue by answering what others said. They need the justification and the speaker, which is why audio quality and diarization are experiment factors.

- The prompts ask an agent to address speakers by name, answer the justification it heard, and give its own.
- A reply may begin with `<to>Agent_5</to>`. The tag is stripped before synthesis and display, and the room stores it as `addressed_to`.
- Every utterance records the cursor its speaker had at the start, so analyses can link a rebuttal to what the speaker had heard.
- With `barge_in` on, an agent mid-utterance can cut itself off and answer an accusation at the next tick, so a rebuttal can follow an accusation within 1 tick.

Measures of arguing.

- Rebuttal rate. The accused addresses the accuser within 2 ticks.
- Answered justification, scored with an LLM judge. `config/judge_prompts.py` already has a judge that flags a changed suspect given without justification, and it extends to rebuttals.
- Persuasion. Votes move toward a target after an argument against it.
- Consistency. Claimed locations match the ground-truth action log, reported separately for crew and impostors.

### Rendering text

A reader at time `t` sees the words in each lane with timestamp below `t`.

`clean` shows one block per utterance with its time span. Utterances still in progress show the words spoken so far followed by `...`.

```
[00:00-00:14] Agent_3: I found the body in Electrical and saw Agent_5 leaving.
[00:01-00:11] Agent_5: I was in MedBay with Agent_4 the whole time, it was not me.
[00:12-...  ] Agent_7: Agent_3 reported fast, that seems ...
```

`interleaved` cuts words into `window_s` windows. Each line is one speaker's words in one window, and lines with the same timestamp were spoken at the same time.

```
[00:00] Agent_3: I found the body in
[00:00] Agent_5: I was in MedBay with
[00:02] Agent_3: Electrical and saw Agent_5 leaving
[00:02] Agent_5: Agent_4 the whole time it
```

`noisy` is `interleaved` plus dropout and, with `tag_policy=first_chunk`, speaker tags only on a speaker's first line.

```
[00:00] Agent_3: I found the body in
[00:00] Agent_5: [inaudible] was in MedBay [inaudible]
[00:02] Electrical and saw Agent_5 leaving
[00:02] Agent_4 the whole time [inaudible]
```

Rules for every mode.

- A reader's own words are marked `(you)`.
- Simultaneous lines are shuffled per `(seed, reader, tick)`, so position bias averages out across games.
- Dropout masks come from `(seed, reader, utterance, word index)`. Words a reader missed at tick k stay missed at tick k+1.
- Agents are told that everyone speaks at once, that words may be interleaved, and how much room time remains.

### Ground truth and the Observer

The clean utterances with start times go to `room.jsonl`, together with every reader's render at every tick.
The discussion log used by the UI and the Observer input (`Agent`, `Text`, `Reported`, `S_Num`) come from the clean utterances, and `S_Num` becomes the utterance index per author.
The SGD, SVM and LR classifiers were trained on sequential transcripts, so their scores will shift. Treat them as a measurement per protocol, since they are not calibrated probabilities here.

## Code changes

| File | Change |
|------|--------|
| `core/situation_room.py` | New. `Utterance`, `Lane`, `Room`, the message types (`Percept`, `Tick`, `Reply`, `Vote`), and the three renderers |
| `core/adapters/` | New package. `base.py` (the `AgentAdapter` protocol), `in_process.py`, `vllm.py`, `api.py`, `remote.py` |
| `core/adapters/ears.py`, `core/adapters/mouths.py` | New. `TextEar`, `CascadeEar`, `NativeEar`, `TextMouth`, `TtsMouth` |
| `core/audio/` | New package. `tts.py` (voices, WAV output), `mix.py` (per-reader mix), `asr.py`, `diarize.py` |
| `core/transport.py` | New. `InProcess` and `File` transports, with `Network` left for later |
| `core/timed_discussion.py` | New. Tick loop over adapters, simultaneous vote tick, and the sequential runner kept as the control |
| `core/game_engine.py` | Pick the runner from `discussion_mode`, build one adapter per agent from the composition, accept `seed` |
| `core/llm.py` | Unique request ids and timeouts for the in-process adapter, plus `LLM_MODE=VLLM` |
| `core/state.py` | Seeded start rooms |
| `agents/honest_agent.py`, `agents/byzantine_agent.py` | Take `percept.text` in discussion and voting in place of the discussion log file, add the room instructions and `<silent/>` |
| `agents/prompts.py` | Room instructions, with the `<think>` appendix unchanged |
| `core/logger.py` | Write `room.jsonl` and per-utterance start times |
| `submit_games.sh` | Longer time limit, vLLM startup, seed from the array index |
| `tests/test_situation_room.py` | pytest with a fake generator and no GPU |

Tests cover merge order across lanes, truncation at the deadline, busy agents skipping ticks, dropout stability across ticks, and that one seed reproduces one transcript.
Audio tests check that the mix places each speaker at its offset, leaves out the reader's own voice, and that the recorded duration equals the length of the WAV file.
A transport test plays the same game with fake deterministic adapters in the controller process and again in separate processes over the `File` transport, and expects identical `room.jsonl` files.
Run them with `uv run --with pytest python -m pytest tests`.

The human seat is not supported in timed mode in v1.

## Running on HiPerGator

vLLM runs on HiPerGator. The byzantine-agents project (`scripts/hipergator/` under `/home/christan/toblue/christan/projects/byzantine-agents`) supplies the recipe.

- `env.sh` puts `HF_HOME`, `UV_CACHE_DIR`, `VLLM_CACHE_ROOT` and `TORCH_HOME` under `/blue/ufdatastudios/christan/cache`, because the home directory is 40 GB.
- `vllm_serve.sbatch` runs `uv run --no-project --with vllm vllm serve <model>` under account and QoS `ufdatastudios` and writes `endpoints/<job id>.url`.
- Partitions are `hpg-turin` (L4, 24 GB), `hpg-rtx6000` (RTX PRO 6000, 96 GB) and `hpg-b200` (B200, 180 GB). The QoS allows 16 GPUs at a time, shared with the group.
- Per-model flags that were needed there. gpt-oss needs `--attention-backend TRITON_ATTN`. Llama weights are gated and need the Hugging Face token. Multi-GPU servers need `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` and a list of excluded nodes.
- A server must not sit idle. byzantine-agents queues a release job that cancels it after the last grid, and the situation room jobs do the same.

The plan for this project.

- One vLLM OpenAI-compatible server per model per GPU, started by the SLURM script. Each writes its base URL to an endpoint file, as in byzantine-agents, and adapters read it in place of the `ready_*.signal` files.
- Each `VllmAdapter` calls its server through the existing `OpenAICompatibleClient`. Adapters reply in parallel, so vLLM batches the simultaneous requests for one server.
- Placement is a setting. By default agents of the same model share a server. Setting `placement` to `per_agent` starts one adapter and one server per agent with `srun --gpus-per-task=1`, up to one GPU per agent. The room and the tick loop are the same either way.
- Adapters on different nodes use the `File` transport over the shared filesystem.
- Automatic prefix caching is on, because each tick resends a longer version of the same prompt.
- Each request includes a sampling seed derived from `(seed, agent, tick)`.
- The lightweight tier fits several models per GPU as in `submit_games.sh` today, with `gpu_memory_utilization` per server summing below 1. The heavy tier needs tensor parallelism across GPUs.
- Experiment jobs set no Navigator, OpenAI or Anthropic keys, and those providers are disabled when `LLM_MODE=VLLM`.
- Batch composition changes numerics, so regenerating a game gives different text. The logged transcripts are the record.

### Model eligibility

The byzantine-agents README records that only models from organizations outside the foreign countries of concern may run on HiPerGator (Florida Statute 288.860, UF Research Computing training module 4).
The HW and LW lists in `config/model_composition.py` include Qwen models (Qwen3-Next 80B, Qwen2.5 72B, Qwen2.5 7B, Qwen2 7B and Qwen2.5 1.5B) and models that appear to build on Qwen (Athene V2, Arcee Nova and OpenPipe Qwen3 14B).
Treat those as excluded until Research Computing confirms otherwise. The model matrix comes from Llama, Gemma, Mistral, gpt-oss, Nemotron, Granite, Olmo and Apertus. The same check applies to every speech and diarization model.

## Experiments

| Phase | Question | Setup | Games |
|-------|----------|-------|-------|
| P0 | Does the room behave correctly | Build and test locally with a fake generator | 0 |
| P1 | What does a game cost and does it read well | 1 GPU, 1 small model, 5 games per protocol. Measure minutes per game and parse failures, check that `interleaved` is legible, tune defaults. Run one game with each agent on its own GPU to confirm the barrier works across processes | about 25 |
| P1b | Does the audio stack work | TTS with 10 voices, the per-reader mix, speech recognition and diarization on 1 GPU, using a scripted 60 s meeting. Measure the time per tick and the word error rate against the known text | 0 |
| P2 | How much does the protocol matter | 3 models (small, medium, large), homogeneous. Sequential, timed `clean`, `interleaved` and `noisy` at budgets 30, 60 and 120 s, and no discussion. 11 cells per model, 50 games per cell | 1,650 |
| P3 | How does each model perform | Each eligible model as crew against a fixed reference model as impostors, then swapped, under sequential, timed `clean` and timed `interleaved`. 50 games per cell | 3,000 per tier |
| P4 | Which knobs matter | `tick_s` 5, 10, 20. Impostors 1, 2, 3. `dropout` 0, 0.1, 0.3. `barge_in` off and on. 11 cells per model, 50 games per cell, on 1 or 2 models | 550 per model |
| P5 | Does hearing change the game | Text `interleaved` as the baseline, then audio with a cascade ear at `oracle` and `estimated` diarization, and a native ear at `oracle` and `none`. 5 cells per model, 50 games per cell, on 1 or 2 audio-capable models | 250 per model |

Run the lightweight tier first, since several of its models share a GPU.
The reference model for P3 is chosen after P2, from the models whose win rate is stable across protocols.

Every protocol replays the same seeds, so scenarios differ only in how agents talk.

Metrics.

- Win rate by role, vote accuracy (`correct_votes` over `correct_votes + incorrect_votes`), skipped votes, rounds survived, ejection of an impostor in the first meeting.
- Words spoken before the cutoff and fraction of ticks spent silent, by model and role.
- Parse failures, tokens, calls and wall-clock minutes per game.
- Observer classifier scores by protocol.
- The measures of arguing listed above, and the diarization error rate for each reader under `estimated`.

Comprehension probe. In 20% of meetings, a discarded extra call after the last tick asks each agent which agents each speaker mentioned. The answers are scored against the name mentions in the clean utterances, as precision and recall per reader. No LLM judge is needed, and the call leaves the game state unchanged.

Sizing. At 50 games per cell the 95% interval on a win rate is about ±14 points, and about ±10 at 100 games. Start with 20 games per cell to screen, then add games to the cells that matter. Fit `win ~ model * role * protocol + (1 | seed)` for the final comparison, with Wilson intervals for single cells.

## Open questions

- Default for `barge_in`, to be set from P4.
- Wall-clock scenario, to measure what inference latency costs a model. Adapters would hear continuously and post when ready.
- Which audio-capable models, speech models and diarization models pass the license and eligibility checks. pyannote models need a Hugging Face agreement, and the P1b pilot has to confirm access from a compute node.
- Human seat in timed mode. The current human input path uses a 30 s timeout per message.
- Default for `quiet_close_ticks`, to be set from the P1 pilot.
- The movement phase is still sequential. Making it simultaneous is a separate change.
