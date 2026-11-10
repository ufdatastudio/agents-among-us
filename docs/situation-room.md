# Situation Room

Design for timed, simultaneous discussion in Agents Among Us, plus the experiments that use it.
Status is design plus two HiPerGator pilots, one for vLLM concurrency and one for the audio stack. Phase 0 is done. The room, the three text renderers, the tick loop, the vote tick, an in-process transport and a scripted adapter exist and are tested with no GPU, as do `core/audio/` and the benchmark scripts. The engine and agent integration and the `File` transport are still to do. Work happens on the `situation-room` branch.

## Goal

Replace turn-taking with a situation room. All agents speak during the same timed window, their speech overlaps, and each agent works out who said what, and why, from the combined record.
Text is one modality. Agents also hear each other as audio and learn who is talking through diarization. They argue their points by answering the justifications they heard.
The experiments compare how models handle this across scenarios.
They run on HiPerGator with local models and make no API calls.
They run in English. The game prompts, the scenarios and the agents' speech are English, Kokoro uses American English voices, and Whisper is set to English.

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
| `budget_s` | 180 | Length of the discussion in room time. The default is generous so agents can argue before the vote |
| `tick_s` | 10 | Time between decision points |
| `words_per_s` | 2.5 | Speaking rate in the `text` modality, about 150 words per minute. The `audio` modality uses the length of the synthesized speech |
| `max_words` | 40 | Words per utterance. The prompt states the limit and the room drops any words beyond it |
| `max_tokens` | 96 | Generation limit per reply, a guard against runaway output |
| `onset_jitter_s` | 2 | Seeded delay before a speaker starts within a tick. In the `audio` modality it sets how much speakers overlap |
| `tts_speed` | 0.85 | Kokoro speaking speed in the `audio` modality. At 1.0 Kokoro spoke about 3 words per second, and 0.85 brings it near `words_per_s` |
| `quiet_close_ticks` | 3 | Close early after this many ticks with nobody speaking, 0 disables. The discussion never closes while someone is still talking |
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
- A tick lasts as long as its slowest reply, so `max_words` and `max_tokens` bound the tail.
- A meeting costs at most `budget_s / tick_s` calls per active agent. With 10 agents and the defaults that is 180 calls against 20 today. Busy agents make no call and silent replies are short, and prefix caching reduces the prompt cost.

Voting becomes one more simultaneous tick. Every agent sees the final snapshot and votes in parallel. The tally and ejection logic stay as they are.

### The room

The room is a grow-only map of per-author logs.

```
Utterance  author, tick, start_s, pace_s, words, intended, cursor, addressed_to
Lane       append-only list of one author's utterances, written by that author only
Room       lanes by author, free_at(author), append(), percept(reader, t, tick), close(t),
           statements(), ground_truth(), write_jsonl()
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

A percept is a snapshot of everything the reader has heard, and its `cursor` is the room time it covers. The `interleaved` and `noisy` renderings only grow from tick to tick, so each prompt extends the last and a vLLM server can reuse the cached prefix. The `clean` rendering rewrites an utterance as it grows and does not have that property. Delivering only what is new since the cursor is a later optimization.

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
| Ear | Turns a percept into model input | `TextEar` passes the rendered transcript | `SeparatedEar` recognizes each speaker's audio on its own and labels it according to `diarization`. `MixEar` recognizes the reader's mix as one stream. `NativeEar` passes the audio to an audio-capable model |
| Mouth | Turns a reply into an utterance | `TextMouth` stores the words with timestamps | `TtsMouth` synthesizes speech in the agent's voice, and timestamps come from the audio length |

Ears and mouths are chosen per agent, so one game can mix modalities.

The audio pipeline.

1. Each agent gets a voice at game start, drawn with the seed. The voice is the agent's speaker identity for the whole game.
2. The mouth writes a WAV file to the shared filesystem. The room stores the text, the audio path and the duration. Onsets stay on the virtual clock, so synthesis speed does not affect the game.
3. The room mixes audio per reader. The mix at time `t` sums every other speaker's audio up to `t`, each placed at its start offset. The reader's own voice is left out, which takes the place of `(you)`. The overlap is acoustic, and the controller does not cut or reorder anything. Per-speaker gain and background noise take the place of `dropout`.
4. The ear hands the brain what it can use. A separated ear transcribes each speaker's audio on its own and writes the same interleaved transcript the `text` modality uses, with the labels `diarization` allows. This is the clean-audio condition. A mix ear transcribes the reader's mix as one stream, which is the cocktail-party condition. A native ear sends the waveform since the reader's cursor together with the labels.
5. The mix and the transcripts are logged, so any reader's view of any tick can be replayed.

Whisper large-v3 (speech recognition) and Kokoro (speech synthesis) are both in Navigator's catalog, so the audio stack follows the same rule as the language models. Both run from the `/blue` cache on HiPerGator, as the P1b measurements below show.

Three components are not on Navigator and wait for approval.

- Native ear. None of Navigator's language models takes audio, and the Gemma 4 31B config has `audio_config: null`. The one candidate found, `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, is not in Navigator's catalog, builds on a Nemotron 3 Nano backbone that is itself excluded, and its card says it was improved using Qwen models. It is excluded, and a question about it is with Research Computing. The native ear has no model until an eligible one is found.
- Model-based diarization. `nvidia/Nemotron-3-Diarization` is a streaming Sortformer with the OpenMDW license and no gating. Until it is approved, `estimated` labels come from the oracle labels with errors injected at a chosen diarization error rate, which also makes that rate a controlled factor.
- Speech separation. The mix ear needs it to do better than Whisper alone.

### Diarization

Speaker identity is its own channel, with 3 levels.

| Level | What the agent receives |
|-------|-------------------------|
| `oracle` | The exact speaker labels and segment times from the room. In the cascade ear every transcribed span is tagged with the true speaker |
| `estimated` | Labels with errors that agents can see. They come from a diarization model run on the mix, or from the oracle labels with injected errors until a model is approved. Errors merge or split speakers and miss overlapped speech |
| `none` | No labels. The agent separates speakers by voice and content, which needs a native ear. A cascade ear delivers unlabeled text |

A roll call before the discussion, outside the budget, lets each agent say its name once. Under `none`, that is how agents bind voices to names. In the audio modality agents are called by color name, such as Red and Orange, because `Agent_3` is hard to say and to recognize.

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
- Simultaneous lines are shuffled per `(seed, reader, window)`, so position bias averages out across games and the order of earlier lines does not change from tick to tick.
- Dropout masks come from `(seed, reader, utterance, word index)`. Words a reader missed at tick k stay missed at tick k+1.
- Agents are told that everyone speaks at once, that words may be interleaved, and how much room time remains.

### Ground truth and the Observer

The clean utterances with start times go to `room.jsonl`, together with every reader's render at every tick.
The discussion log used by the UI and the Observer input (`Agent`, `Text`, `Reported`, `S_Num`) come from the clean utterances, and `S_Num` becomes the utterance index per author.
The SGD, SVM and LR classifiers were trained on sequential transcripts, so their scores will shift. Treat them as a measurement per protocol, since they are not calibrated probabilities here.

## Code changes

| File | Change |
|------|--------|
| `core/situation_room.py` | Done in Phase 0. `RoomConfig`, `Utterance`, `Lane`, `Room`, the message types (`Meeting`, `Tick`, `Percept`, `Reply`, `Vote`), `parse_reply`, and the `clean`, `interleaved` and `noisy` renderers |
| `core/adapters/` | `base.py` (the `AgentAdapter` protocol) and `scripted.py` (a deterministic fake for tests and dry runs) are done. `in_process.py`, `vllm.py`, `api.py` and `remote.py` are still to write |
| `core/adapters/ears.py`, `core/adapters/mouths.py` | New. `TextEar`, `SeparatedEar`, `MixEar`, `NativeEar`, `TextMouth`, `TtsMouth` |
| `core/audio/` | Started in P1b. `mix.py` (per-reader mix, overlap, WAV), `metrics.py` (word error rate, recall), `tts.py` (Kokoro voices) and `asr.py` (Whisper) exist and are tested. `diarize.py` is still to write |
| `core/transport.py` | `InProcessTransport` is done. The `File` transport is still to write, with `Network` left for later |
| `core/timed_discussion.py` | The tick loop over adapters and the simultaneous vote tick are done. The sequential runner stays in `core/game_engine.py` as the control |
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
- Prompt layout is fixed as system prompt, private memory, the append-only board, and the tick instruction last. Each tick then extends the previous prompt, which lets the prefix cache work.

### Measured concurrency on one L4

`scripts/bench_vllm_concurrency.py` sends the requests of a tick all at once to a vLLM 0.33.0 server on an NVIDIA L4 (24 GB) with an 8B model in bf16. The prompts are synthetic and about 1.3k tokens, with temperature 1.0 and `max_tokens` 160. Each concurrency row is the median of 3 rounds whose prompts share almost no prefix with an earlier round (3% to 8% of tokens came from the cache), and the sequential row is a single run. Neither server logged a preemption.

The first run used Llama 3.1 8B, which Navigator retired on 2026-08-10 and which is not on the eligible list. The second run uses Granite 3.3 8B, an eligible model, and is the reference. The result files are in `results/bench/`.

| Requests at once | Granite 3.3 8B round time | Llama 3.1 8B round time |
|------------------|---------------------------|-------------------------|
| 1 | 0.7 s | 0.6 s |
| 2 | 1.2 s | 3.0 s |
| 5 | 3.8 s | 11.3 s (one reply ran to the 160-token limit) |
| 10 | 6.9 s | 6.6 s |
| 20 | 13.6 s | 14.0 s |
| 10, one after another | 14.2 s | 24.7 s |

- Batching helps. 10 simultaneous requests finish in 6.9 s on Granite against 14.2 s one after another, which is 2.1 times faster. Granite's replies are short, about 14 tokens on average, so there is less to overlap. Llama's replies averaged about 40 tokens and the gain was 3.7 times.
- An L4 is at capacity near 10 agents. The round time doubles from 10 to 20 requests on both models, so the limit is mostly prompt processing of about 13k tokens per tick. A larger GPU would shorten the tick, and the prefix cache reduces the prompt cost, as the next point shows.
- The tick lasts as long as its slowest reply. On Granite one reply ran to the 160-token limit at tick 4 and the round took 11.5 s against 2.9 s to 6.3 s for the other ticks. This is the reason for `max_words` and `max_tokens`.
- The growing-board test uses 10 agents over 6 ticks. On Granite the prompt grows from 616 to 2,004 tokens, and the share of prompt tokens served from the prefix cache was 16%, 66%, 74%, 79%, 82% and 85% for ticks 1 to 6. Llama gave 18% to 85% on shorter prompts.
- A 6-tick meeting of 10 agents takes about 34 s of wall time on Granite, plus a voting tick. The default 18-tick meeting would take roughly 1.7 minutes. The movement phase is still sequential and has not been measured. It is likely to dominate game time, and P1 will measure it.
- The result covers two 8B models, one GPU type and synthetic prompts. Larger models, other GPUs and reasoning models such as gpt-oss need their own runs.

### Measured audio stack (P1b)

`scripts/audio_pilot.py` synthesizes scripted 20-word statements with 10 Kokoro voices, mixes them the way the room would for a silent listener, and transcribes the result with Whisper large-v3 in fp16 on an NVIDIA RTX PRO 6000. The result file is `results/audio_pilot/summary_50646536.json`. The WAV files stay on HiPerGator in `results/audio_pilot/wav_50646536/`.

- Synthesis takes about 0.07 s per statement for 6.8 s of audio. At speed 1.0 Kokoro speaks about 3 words per second, which is why `tts_speed` defaults to 0.85.
- Clean speech is recognized almost perfectly. The mean word error rate over the 10 statements is 0.5% and the worst is 4.5%. A batch of 10 takes 1.2 s. Any loss in the audio conditions therefore comes from overlap.
- Overlap is the problem. Speakers start at random times within 2 s of each other, and each row averages 10 trials with a random subset of speakers.

| Speakers at once | Overlapped share of speaking time | Word recall | Precision | Speakers recovered |
|------------------|-----------------------------------|-------------|-----------|--------------------|
| 1 | 0% | 99.6% | 99.6% | 100% |
| 2 | 77% | 52% | 99.6% | 55% |
| 3 | 79% | 44% | 97% | 50% |
| 5 | 83% | 24% | 90% | 26% |
| 10 | 88% | 12% | 62% | 6% |

- Whisper keeps one stream and drops the others. It can also fuse two. With Green and Brown speaking, the transcript read "I was in storage with Purple, and we both finished our tasks, so I think Purple is lying about where they were". The first half is Green's, and Green accused Red. The accusation of Purple is Brown's. An agent reading that line would hear evidence that nobody gave.
- At 10 speakers precision falls to 62%, so 38% of the recognized words appear in no speaker's text.
- Time labels cannot attach words to speakers in a mix. When 2 or more people spoke, only 11% to 13% of the recognized words fell at a moment with exactly one speaker active, against 97% for a single speaker.
- Recognizing one mix took 1.0 s to 2.9 s on average. One batch of 10 reader mixes of 10 s took 1.2 s in the first run and 18 s in the second. The slow batch probably contained a repetition loop on babble, so Whisper's generation length needs a cap per window.

What this means for the design.

- The default audio ear is the separated ear, which recognizes each speaker's stream on its own. The mix ear is the stress condition, and its difficulty follows how many agents speak at once.
- At tick 1 every agent speaks at once, so a mix ear would recover about 12% of the words. `onset_jitter_s` is the lever for overlap, and the next P1b step maps word recall against overlap share with wider onsets.
- A mix ear that does better than Whisper alone needs a speech separation model.

### Model eligibility

The byzantine-agents README records that only models from organizations outside the foreign countries of concern may run on HiPerGator (Florida Statute 288.860, UF Research Computing training module 4).
The experiments follow two rules. The model must also be served by Navigator, so results apply to the Navigator demo. Neither the model nor its post-training may come from a country of concern. That excludes the Qwen family and DeepSeek, models built on them (Athene V2, Arcee Nova and OpenPipe Qwen3 14B), and any model whose card says it was improved using them. It removes most of the HW and LW lists in `config/model_composition.py`.
`config/eligible_models.py` enforces the rules. It lists the eligible models, matches Qwen, DeepSeek and other family names, and gives the reason for each exclusion. The HiPerGator job scripts run `python -m config.eligible_models <model>` before they start a server, and the job fails for any model that is not on the list.
A Qwen checkpoint gets a warning and does not start unless someone types the override phrase, as `--override "<phrase>"` on the command or `QWEN_OVERRIDE="<phrase>"` on the `sbatch` line. The warning prints the phrase and advises confirming with Research Computing first. The override opens Qwen names only. DeepSeek and the other families, and the models excluded for a card that cites Qwen, stay refused.
The table lists the eligible models with the Hugging Face checkpoint to use. Navigator may serve a different revision or quantization, so each run records the checkpoint revision.

| Navigator model | Hugging Face checkpoint | Class | On /blue |
|-----------------|-------------------------|-------|----------|
| `ministral-8b` | `mistralai/Ministral-8B-Instruct-2410` (research license) | 8B | no |
| `granite-3.3-8b-instruct` | `ibm-granite/granite-3.3-8b-instruct` | 8B | yes |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 21B MoE | yes |
| `mistral-small-3.1` | `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | 24B | no |
| `magistral-small` | `mistralai/Magistral-Small-2506`, `-2507` or `-2509`, to match Navigator | 24B reasoning | no |
| `gemma-4-31b-it` | `google/gemma-4-31B-it` | 31B | yes |
| `meta-muse-glimmer-30b` | `meta-models/Muse-Glimmer-30B`, which Navigator's lifecycle page may list as legacy | 30B | no |
| `llama-3.3-70b-instruct` | `meta-llama/Llama-3.3-70B-Instruct` | 70B | yes |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 117B MoE | yes |

The audit on 2026-11-10 read the public model card of every Navigator open-weight model and of the audio models, looking for a Qwen, DeepSeek or other Chinese-origin base, teacher or reward model. The 9 models above are clean. Their cards mention Qwen or DeepSeek only in benchmark tables and Chinese only as a supported language. The Llama 3.3 70B card is gated, so its check rests on the base model in its metadata, Llama 3.1 70B, and Muse Glimmer 30B lists no base model.

Excluded after the audit.

- `llama-3.1-nemotron-nano-8B`. The card says "Improved using Qwen".
- `nemotron-3-nano-30b-a3b`. The card says "Improved using Qwen", and its reinforcement learning used a reward model built on Qwen3.
- `nemotron-3-super-120b-a12b`. The card names GPT-OSS-120B as one teacher and leaves the others unnamed, so Qwen or DeepSeek outputs cannot be ruled out. It returns to the list if its dataset card shows otherwise.
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`, the native-ear candidate. The card says it was improved using Qwen3-VL-30B-A3B-Instruct, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B, Qwen2.5-VL-72B-Instruct and gpt-oss-120b. Its architecture section names the excluded Nemotron 3 Nano 30B-A3B as the language backbone. The training data table lists Qwen models as annotators for most of its public datasets, and also DeepSeek OCR, GLM-4.1V-9B-Thinking and Kimi-K2.5 for a few. Navigator does not list it. The license is the NVIDIA Open Model Agreement.
- `medgemma-27b-it`, which is tuned for medicine and not used in the game.

The audio stack passed. Whisper large-v3 (OpenAI) and Kokoro (built on StyleTTS 2) have no Chinese-origin base, and the audio environment has no packages from the Qwen or DeepSeek ecosystems. `nvidia/Nemotron-3-Diarization` is an NVIDIA model, but it trains on Mandarin meeting corpora (AISHELL-4 and AliMeeting), which its pending approval should consider.

The 8B models fit one L4 or part of an RTX PRO 6000. The 70B and 120B models need 2 or more GPUs, as the byzantine-agents notes describe.
The reasoning models (gpt-oss and Magistral) spend part of `max_tokens` on thinking, so their limit needs its own check in P1.

## Experiments

| Phase | Question | Setup | Games |
|-------|----------|-------|-------|
| P0 | Does the room behave correctly | Done. 46 tests with a scripted adapter and no GPU. Three deliberately injected bugs (always-ready agents, a shuffle that depends on the tick, a barge-in that does not cut) each made a test fail | 0 |
| P1 | What does a game cost and does it read well | 1 GPU, 1 small model, 5 games per protocol. Measure minutes per game and parse failures, check that `interleaved` is legible, tune defaults. Run one game with each agent on its own GPU to confirm the barrier works across processes | about 25 |
| P1b | Does the audio stack work | Done once on 1 RTX PRO 6000 with Kokoro and Whisper large-v3, see Measured audio stack. Still to do. Map word recall against overlap share with wider onsets, cap Whisper's generation length, and test a speech separation model once it is approved | 0 |
| P2 | How much does the protocol matter | 3 models (small, medium, large), homogeneous. Sequential, timed `clean`, `interleaved` and `noisy` at budgets 60, 180 and 300 s, and no discussion. 11 cells per model, 50 games per cell | 1,650 |
| P3 | How does each model perform | Each of the 9 eligible models as crew against a fixed reference model as impostors, then swapped, under sequential, timed `clean` and timed `interleaved`. 50 games per cell | 2,700 |
| P4 | Which knobs matter | `tick_s` 5, 10, 20. Impostors 1, 2, 3. `dropout` 0, 0.1, 0.3. `barge_in` off and on. 11 cells per model, 50 games per cell, on 1 or 2 models | 550 per model |
| P5 | Does hearing change the game | Text `interleaved` as the baseline. Audio with a separated ear at `oracle` and `estimated` diarization. Audio with a mix ear as the cocktail-party stress test. A native ear at `oracle` and `none` if a model is approved. 4 cells per model without the native ear, 50 games per cell, on 1 or 2 models | 200 per model, plus 100 for the native ear |

Run the models that fit on one GPU first.
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
- Approval for models that are not on Navigator. `nvidia/Nemotron-3-Diarization` for model-based diarization, a speech separation model for the mix ear, and an eligible audio-capable model for the native ear, for which none is known yet.
- Human seat in timed mode. The current human input path uses a 30 s timeout per message.
- Default for `quiet_close_ticks`, to be set from the P1 pilot.
- The movement phase is still sequential. Making it simultaneous is a separate change.
