# A Godot world for Agents Among Us

This is a plan and not an implementation. It lives on the branch `godot-world`, which starts from `situation-room` because the world reuses the situation room, the agent adapters, the audio stack and the transcript analysis of that branch.

## Goal

A ship that people can watch and play in, and that agents can live in. Rooms have sprites and sounds. Humans can join from a browser in different places, and agents join through streams of what they see and hear. The same world runs as a visible simulation for people and as a faster headless simulation for experiments. Meetings happen in the situation room of `docs/situation-room.md`.

Three things are new compared with the Python engine.

- The world is continuous. An agent runs through corridors and rooms, and sound reaches it through distance and walls, where the Python engine moves agents between rooms by name in 3 steps per round.
- An agent perceives. It receives video and audio streams, and it writes its own memory of what is happening. The Python engine writes a text observation log for the agent, so the agent never has to notice anything.
- Humans and agents share one game with the same rules.

## What stays in Python

The experiments stay in the Python stack. The world is a body for the agents, and the room controller, the adapters, the vote reading, the logging and the analysis do not move.

| Part | Where it lives |
|------|----------------|
| Rooms, movement, physics, doors, bodies, tasks, kills, tick clock | Godot |
| Sprites, animation, sound playback, browser client | Godot |
| Per-agent video frames and audio mixes | Godot produces the raw material, see Perception |
| Agent prompts, memory update, vote reading | Python, the existing agents and adapters |
| The meeting (timed room, overlap rendering, votes) | Python, `core/situation_room.py` and `core/timed_discussion.py` unchanged |
| Experiment runner, logs, analysis, figures | Python |

A bridge connects them. The Godot server speaks JSON messages over a WebSocket, with binary messages for frames and audio. Each agent is a Python client that receives its percepts and sends actions. A browser player is a Godot client of the same server.

## Principles

- One rule book. The authority is the Godot server. A human and an agent send the same action messages, and the server checks them.
- Virtual time. The world advances in fixed steps that a controller grants, as the room does with ticks. A slow model does not change the game, and an experiment can run faster than the wall clock. A human game runs the controller at the wall clock rate.
- Reproducible runs. A seed fixes the map layout, the starts and the dice. A run records every action and can be replayed.
- The model rule applies to every model in the loop, including vision and audio models. Only eligible models run, and a model that reads video or audio needs the same check as a text model.
- Original assets. Among Us is a trademark of its makers. Sprites, sounds and names are original or licensed for this use, and CC0 packs are the default.

## Architecture

```
 browser (Godot web export)            Python agents (one process per agent or one for all)
        |  WebSocket                          |  WebSocket
        v                                     v
 +-----------------------------  Godot world server  ------------------------------+
 | rules  physics  doors  bodies  tasks  kills  fixed-step clock  recorder          |
 | sight: per-agent frames     sound: sources, distance, walls    meeting hand-off   |
 +-----------------------------------------|----------------------------------------+
                                           | meeting call
                                           v
                          situation room controller (Python, existing)
```

- The server runs as a native Godot process, headless when no one watches. The same project exports to the web for the client.
- A meeting call pauses the world, moves the agents to the meeting room and hands control to the situation room controller. The votes come back as a world event and the world applies the ejection.
- A human joins a meeting as a text player first. Voice comes later and reuses the audio stack.

## Perception

Every experiment compares what an agent perceives, so the world gives three levels of perception through one interface.

1. Oracle text. The server writes what an agent could see in text, as the Python engine does today. This is the control, and it makes the Godot world comparable to the current results.
2. Audio. The server emits sound events, and each agent receives the sounds that reach it, as text from an ASR step or as audio for a model that hears.
3. Video. The server renders a frame from each agent's position at a low rate, for example 256 by 256 pixels at 2 frames a second, and the agent writes a memory note from the last frames.

The memory update is a step of the agent. At an interval of game time the agent reads its newest percepts and its last memory and writes a new memory in text. At a meeting it reads that memory and no raw percept. This gives a measure that the Python engine cannot give, namely how accurate the memory is. The claim check of `scripts/analyze_p2_text.py` applies to the memory as it applies to statements, because the server knows where everyone was.

### Audio

Sound has sources (footsteps, doors, a kill, an alarm, speech), a position and a loudness. A listener hears a source with a gain from the distance and from the walls and doors between them. Two ways to build this.

- Godot mixes. Each agent has a viewport with its own listener, and the mix of its audio bus is captured. This is the natural use of the engine, and it needs a test that one listener can be read per agent.
- Python mixes. The server sends each source's track and the gain for each listener, and `core/audio/mix.py` builds the mix. This gives control and exact repeatability, and the stack already measures recognition on such mixes.

The plan starts with Python mixing for agents, because the audio pilot gives a baseline there, and Godot's own playback serves human players. The first test decides whether Godot can give per-agent mixes, which would remove the duplicate.

The room physics are simple in the first version. The gain falls with distance by an attenuation exponent, up to a maximum distance (`AudioStreamPlayer2D` has both settings). Each wall between source and listener takes away a fixed number of decibels, and an open door takes away none. Reverb and room acoustics come after that.

## The world

- 2D and top-down for the first version. It is cheaper to render, easier on agents that read frames, and the existing maps are room graphs that fit a tile map. A 3D ship can follow if it is worth the cost.
- Maps come from the room graphs of `config/settings.py`, laid out as tile maps with corridors and doors. Rooms keep their names, so a claim such as "I was in Storage" is still checkable.
- Movement uses `CharacterBody2D` at a fixed physics rate. The rules that count in a round (3 moves, kills, bodies, reports) are rebuilt in terms of time and distance, and a test ties them to the Python engine's rules.
- The language is GDScript. Godot 4 has no web export for C# projects, which would end the browser client, and the first test confirms it.
- The Godot version is 4.3 or later, because the single-threaded web export came with 4.3.

## Browser and multiplayer

The web client runs the same project exported to HTML, WebAssembly and a pack file, which any static server can deliver.

- Headers. A web export without threads needs no special headers, which is the 4.3 default. An export with thread support needs the page to be cross-origin isolated, which means the response headers `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`, or the service worker of the PWA export mode that supplies them. The plan starts without threads and moves to threads only if the browser audio is poor, since the Godot documentation notes that audio can suffer in single-threaded mode on some systems.
- Network. Browsers cannot send raw UDP, so the client talks to the server over WebSocket. Whether Godot's high-level multiplayer works over a WebSocket peer from a web export is to be tested in the first step. Voice over WebRTC is a later option.
- Hosting. Players in different places need a server that is reachable from the browser. The PubApps VM runs containers behind a proxy, and the plan needs to know whether that proxy passes WebSocket upgrades and custom headers. This is a question for Research Computing. HiPerGator compute nodes are for the headless simulations, not for public players.
- Players. Players join with a code and a chosen name, and the server stores no personal data.

## Phases

Each phase ends with a test that can fail.

| Phase | Work | Test that ends it |
|-------|------|-------------------|
| G0 | Spike on the open technical questions below | Each question has a written answer and a decision |
| G1 | World core. Map, movement, doors, bodies, tasks, kills, meeting call, fixed-step clock, recorder, the oracle text perception and the Python bridge | A scripted game runs headless, and the same actions in the Python engine and in Godot give the same rounds, bodies and ejections |
| G2 | Agents in the world with oracle text and the memory update | The existing agents finish games, and the memory accuracy and win rates have a baseline next to the P2 results |
| G3 | Sound events, distance and wall gain, per-listener mixes, speech in the world | A listener behind a wall hears a source at the gain the physics predicts, and an ASR check recovers the words at the distances the mixes give |
| G4 | Video frames per agent and a vision model | The same games run with oracle text, audio, video and audio with video, and the memory accuracy of each is measured |
| G5 | Browser client, join codes, human text chat, humans and agents in one game | Two humans in two places finish a game with agents on the hosted server |
| G6 | The situation room in the world. Humans in timed meetings, with text and then voice | A mixed meeting runs through the timed room, and the human seat works in `core/timed_discussion.py` |

## First test (G0)

- Web export. A single-threaded export of a small scene runs in Chrome and Firefox, with audio, and its audio quality is judged. Check whether the 4.3 or later behavior matches the documentation.
- Headless. `godot --headless` runs the simulation faster than the wall clock. Rendering is a dummy under `--headless`, so video frames need a display driver. Check what runs on a compute node, for example a virtual display or an offscreen renderer, and the cost of a frame.
- Per-listener audio. Whether one Godot process can give each agent its own audio mix, for example one viewport and listener per agent with the bus captured through `AudioEffectCapture`, or whether the mixes have to be built in Python.
- Browser multiplayer. A browser client and a native server exchange state over a WebSocket peer, and the round trip time is measured.
- Language. A GDScript project exports to the web, and a C# project does not.
- Determinism. The same seed and the same actions give the same states on two machines.
- Tests of GDScript. A test framework such as GUT or gdUnit4 runs in a headless job.

## Experiments the world allows

- Perception. Oracle text, audio, video and both, with the memory accuracy, the win rate and the cost in tokens and frames of each.
- Distance and information. Whether louder or quieter worlds (the attenuation and the wall loss) change who knows what at a meeting, and the false claims that follow.
- Humans among agents. Whether human players detect Byzantine agents better than agents do, and whether agents detect human impostors, which needs ethics approval.
- The situation room with embodiment. The P2 comparison of turns against the room, repeated with agents that walked and listened to get their memories.

## Risks and questions

- Human participants. A study with human players needs review by the university's human subjects board before any data is collected. Demo play without recorded data is a separate matter, and the decision is with the research team.
- Video cost. A vision model reads every frame it is shown. At 10 agents and 2 frames a second a game makes about 20 frames a second of model input, so the rate and the resolution are the main levers, and a cheaper path is to describe the frames in text with one small model.
- Eligible models. The Nemotron Omni model, which reads video and audio, waits for an answer from Research Computing. Until a vision model is approved, G4 cannot start, and G3 can use the existing ASR.
- Wall clock against virtual time. A human game runs at the wall clock rate, and an agent needs seconds to answer. The controller grants steps and waits for the agents, so a human game with slow agents pauses between steps. The plan needs a rule for how long it waits and what a slow agent does.
- Rendering on a cluster. Frames need a renderer on a node without a display. If this is hard, video runs on a node with a GPU and a virtual display.
- Hosting. The proxy question above, and whether the game server runs on the VM or on a separate host.
- Scope. G1 is a rebuild of the rules in a new engine, and the equivalence test is what keeps it honest.

## Decisions for the research team

- 2D top-down for the first version, as proposed above.
- GDScript, and the Godot version, which should be a stable 4.x release from 4.3 on.
- Whether the first public milestone is the watchable simulation (G1 and G2 with a browser viewer) or the playable game (G5).
- Where the server is hosted.
