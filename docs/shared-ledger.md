# Shared ledger for the room and the world

Status is a design with condition B built. The records, the board and the room wiring exist and are tested, and the comparison runs have not been played. The ledger idea comes from the group's work on crews of agents that operate a shared database. That work is in a private repository and a manuscript under review, so this page describes the ledger in general terms and cites only public work. The questions are whether the room and the Godot world should keep their state in one ledger, what that changes for agents inside and outside a meeting, and how to measure it.

## How state is kept today

- Before a meeting each agent reads two text files, its own action log and the results log of the game (`GameAgentAdapter.begin_meeting`). Nothing records who could observe an event, so the engine decides what each agent sees when it writes the file.
- During a meeting the room keeps one lane of utterances for each speaker, and at each tick it renders a text percept for each reader from the lanes (`Room.percept` and the `clean`, `interleaved` and `noisy` renderers). A tick is a barrier. All ready agents hear the same room state at time t, reply in parallel, and the controller commits the replies together.
- The vote tick, the rewind of a meeting that lacks a vote and the replay video each rebuild this state with their own code.

The Godot plan needs more. Agents outside the meeting (dead agents, agents in other rooms), spectators and human players have to read and write the same state. Speech is heard within walls, so each utterance needs a list of listeners that the map computes, and a spectator needs a read of everything.

## The ledger in general terms

A ledger is an append-only log with one sequencer that gives every record a time. A record has an author, a type (fact, claim, belief, commitment, vote or action), content, citations of earlier records, a status, and the agents who may read it. The prompt of an agent is the result of a query over the records it may read, in place of a pasted transcript. A write can be checked before it is admitted, for example that it cites records the writer can read and that its content follows from them, and a rejected write returns to the writer with the reason. Every record keeps its author and citations, so the history can be queried and replayed.

## Mapping

| In the room and the world | In the ledger |
|---------------------------|---------------|
| Where each agent was at a tick, a body found, a task finished | A fact written by the engine, readable by the agents who could see it |
| A spoken statement | A claim with its speaker, start time and words, readable by the agents who hear it |
| An accusation, or "I saw X in R at tick k" | A claim with a subject that cites the speaker's own facts |
| The private reasoning of an agent (thought capture) | A belief, readable by its author only |
| A promise to vote for someone or to meet in a room | A commitment |
| A vote | A vote |
| A move, a kill, a report, a door | An action |
| One tick of the room | A batch committed at the tick barrier, where every read sees the records from before the tick |
| A meeting without all votes, rewound and held again | The records of that attempt are discarded |
| The replay video | A read of the ledger by a spectator |
| A human player | A writer whose input becomes the same action records |

Walls become the lists of who may read a record. The ledger stores whether an agent can read a record, and how well the agent heard it is decided when it reads (muffled through a wall, clear in the room), so one claim can render differently for two listeners.

## Where checks apply in a game of lies

Impostors win by lying, so a check that rejects false claims when they are written would remove the game, and a rejection that others could see would reveal who is lying. That leaves two designs.

- Reader side. Nothing is rejected. Each reader sees every claim with its status against the reader's own facts, which is supported by a fact it saw, contradicted by a fact it saw, or unsupported.
- Writer side for honest agents only. A claim of an honest agent has to cite facts in its own view, and a rejection goes back to that agent alone, so it can try again and nobody else sees it. This targets the false claims of honest agents, which are 17% of the honest claims of Gemma 4 31B in P2 against 30% of its Byzantine claims, and it leaves the impostors free.

The reader-side design leaves how agents speak unchanged. The writer-side design changes how honest agents speak, so the two are separate conditions.

## Other designs for shared agent state

The descriptions come from the full text of S-Bus, PatchBoard, CoAgent and MAP-Graph and from the abstracts of the others, all read on 2026-11-12. Only S-Bus states a code release. PatchBoard, MAP-Graph and CoAgent ran their main results on Qwen or DeepSeek models, which the model rule of this project excludes, so a comparison has to reimplement their ideas on eligible models.

| Work | Shared structure and mechanism | Fit for the room |
|------|--------------------------------|------------------|
| AgileLog (Bhat et al., SOSP 2026, arXiv 2604.14590) | A forkable shared log. An agent works on a fork that is isolated from the live log in logic and in performance | Forks give the rewind of a meeting and a way to run several protocols from one meeting start |
| PatchBoard (Zhang et al., arXiv 2605.29313) | A shared JSON state tree that workers change through JSON Patch operations, which a sequential kernel accepts only if they pass syntax, role-specific write grants, schema validity and invariants. Each worker reads a bounded view, and a rejection returns to that worker. It reports 84.6% success on 630 ALFWorld episodes against 30.8% for LangGraph. Its fault tests observed no contamination from unauthorized writes or bad paths, and false claims that satisfy the schema remain possible, with 0.43 contamination | Close on audit, role views and checks. It replaces dialogue with state writes, claims have one status field that only a verifier writes, and beliefs and votes are not described, so it fits as a condition with no talk |
| CoAgent (Lyu et al., arXiv 2606.15376) | Concurrency control for peer agents that write files, rows or cluster objects through tools with declared footprints. A serialization order is fixed at launch, each read returns the value as of the reader's rank, writes apply in place, a misordered write is undone and reapplied, and a lower-ranked write notifies the readers it affects. It reports 95% correct against 13% with no control on 10 cells of 2 agents | A meeting has no write conflicts, because claims append and votes are independent, and a reader would not see claims of higher-ranked agents. It assumes cooperative agents. The server settles world actions such as doors and tasks |
| S-Bus (Khan, arXiv 2605.17076) | Keyed shards of opaque text. The server logs each read over HTTP, and a commit that rests on a stale read of another shard is rejected | Ruled out as the shared structure of the room, see the next section |
| CodeCRDT (Pugachev, arXiv 2510.18893) | CRDTs for concurrent code generation with eventual consistency | A set of appended claims already converges, and the game needs a fixed order of events, which eventual consistency does not give |
| STORM (Liu et al., arXiv 2605.20563) | A consistent view of a shared codebase, with conflicting edits detected when they are written | The same fit as CoAgent |
| MAP-Graph (Wang et al., arXiv 2608.10509) | A provenance graph of agents, tools, resources, messages, memories, claims and actions, kept for one task beside a vector store. A permission filter decides what each role may read, retrieval is ranked by similarity times a trust that follows the derivation path, and an action gate can block, redact or reverify an action. It reports 94.96% task success against 74.67% for the best baseline on 2,700 templated tasks | The closest on typed records, per-reader filtering and provenance. Trust comes from source labels and writer reliability constants, claims are extracted by text rules, writers are sequential, and it has no contradiction handling or replay |

Eugene Wu's group (DAPLab, Columbia) is the other place to look. I read its publication page, his homepage and the arXiv abstracts on 2026-11-12.

- I found no paper of the group that proposes a shared data structure for several agents with consistency levels. Its published pieces are about the environment of one agent. Agentic Data Environments (arXiv 2607.07397) names branching for state safety and data flow control for data safety, BranchBench (arXiv 2604.17180) benchmarks branchable databases, and StateFork (arXiv 2609.38648, title only) builds branchable infrastructure. Data flow control (arXiv 2606.05679) enforces policies over how data flows through queries, and Environment Steering (arXiv 2609.35807) models the execution state of an agent as tables and steers the agent when a policy is violated.
- Multi-agent coordination appears as a benchmark, Passes Alone, Fails Together (arXiv 2609.25396, title and venue only), and as a topic of a course announced for spring 2027, Data Systems with and for Agents.
- Two parts transfer to the room. Branching gives paired runs from one meeting start, and data flow policies give a way to state what an agent may read.

## Where this work sits

The table compares what each design states, from the papers above. A dash means the paper does not state it.

| Design | Shared state | Typed claims | View of each reader | Why a write happened | Checks on writes | Concurrency | Liars |
|--------|--------------|--------------|---------------------|----------------------|------------------|-------------|-------|
| S-Bus | Keyed text shards | - | - (a write token) | - (read versions) | A stale read of another shard | Optimistic, per shard | Out of scope |
| CoAgent | Objects behind tools | - | By serialization rank | - | None, undo and notify | Fixed order with undo | Out of scope |
| PatchBoard | JSON state tree | Claim list, one verifier status | By role and budget | Worker, trigger, view hash | Syntax, grants, schema, invariants | One sequential kernel | Faults injected, schema-valid falsehoods pass |
| MAP-Graph | Provenance graph | Claims extracted by rules | By permission, then ranked by trust | Derivation edges | Action gate on trust | - | Poisoned sources in the benchmark |
| AgileLog | Forkable log | - | A fork | - | - | Linearizable log | - |
| This work | Typed append-only records | Claims, beliefs, votes with citations | Per reader, from the map | Citations to records the writer could read | Reader-side status, later honest-only | Tick barrier | Impostors that lie by design |

Four points set this work apart.

- The adversary is part of the game. The neighbours assume cooperative agents or script the faults in a benchmark, and the room has agents whose job is to lie, with the engine as a source of verified facts. A claim is judged against what the reader itself observed, in place of a trust constant attached to its source.
- The view of a reader comes from the world. Who may read a record is computed from the map, and what a reader heard is rendered from overlapping speech, so two readers get two views of one statement. Permission filters in the neighbours rest on roles and scopes.
- The read set is exact. The percept is the result of a query, and the room logs it, so which records an agent used is known. S-Bus shows what is lost without that, since 74% of the reads it measures go through conversation history.
- The tick is a barrier, so every reader sees one snapshot and no concurrency control is needed. CoAgent and S-Bus address agents that overwrite shared state, which the room does not do.

Four parts are not new, and the plan uses them in place of claiming them. An append-only log with a sequencer is the shared log of AgileLog and of consensus systems. Validating the premises of a write is S-Bus and CoAgent. A kernel that checks writes against a schema is PatchBoard. A provenance graph with permission filtering and trust is MAP-Graph. The claim is limited to the combination for conversations among agents that include liars, and it is a claim to test. Nothing here shows yet that a board helps agents vote, and P6 is the test.

The 3 closest designs also make comparison conditions. The ideas can be reimplemented in the room on eligible models. A state board with schema-checked writes and no speech follows PatchBoard, a board that ranks claims by the trust of their source follows MAP-Graph, and optimistic validation of world actions follows S-Bus and CoAgent if the Godot server ever needs it.

## S-Bus

S-Bus is the closest of these designs, so the full text was read. It is an independent preprint of May 2026 with code at github.com/sajjadanwar0/sbus. Its shared state is a set of keyed shards with opaque string contents. A server logs each read that an agent makes over HTTP, and a commit to a shard is rejected with a conflict (HTTP 409) when another shard that the agent read has changed since. The agent then reads again and retries. The paper calls the guarantee observable-read isolation, and it targets structural races, which are commits based on a stale read. The experiments use mostly GPT-4o-mini with 4 to 64 agents on tasks in the style of SWE-bench, and they report no corruption in more than 800,000 commit attempts, with PostgreSQL SERIALIZABLE and Redis WATCH/MULTI as baselines.

By the paper's own account it has no typed records, no per-agent visibility beyond a write token on each shard, no citations or provenance for a write, no query or replay of the log, and no context built by a query, since agents fetch whole shards. It assumes trusted agents, and it puts Byzantine agents and forged read sets out of scope.

It is ruled out as the shared structure of the room, for 3 reasons.

- The room has no stale-read race inside a tick. A tick is a barrier, so every reader sees one snapshot, and the writes append claims and votes that no other agent overwrites. In the world the authoritative server settles races between actions.
- A conversation is a single shard that every agent writes. The paper calls its check the wrong primitive for collaborative writing to one shard, and in that experiment the outputs were contradictory with the check on (90 of 90) and with it off (77 of 90).
- It assumes honest agents, and the room has impostors.

The room borrows 2 findings from it.

- The paper measures that 74% of the reads of an agent (0.739 over 128,622 reads) go through its conversation history, where no HTTP call shows them, so a read set rebuilt from observed reads misses most of what the agent used. A percept built by a query has an exact read set, and the room already logs every percept (`Room.percept_log`).
- Agents over-claim the shards they used, by 32% by the judgment of an LLM and by 49% by the judgment of a human annotator. A citation that an agent writes has to be checked against the records the agent could read.

If the world server later lets agents write mutable state that other agents change while the game runs, S-Bus is the baseline to compare. The paper has internal inconsistencies, such as 8 and 88 domains for one workload and two totals of commit attempts (809,710 and 884,110), so its numbers need care.

## The comparison to run (P6)

All conditions use the same 20 scenario seeds, interleaved overlap and a budget of 180 s, which the budget table in `docs/situation-room.md` supports, and P2 already has the baseline row of each model.

- T. The transcript rendering of today, already measured.
- B. Board, reader side. Utterances and engine facts are ledger records, and the percept of a reader is the interleaved transcript followed by a query that groups the claims by subject, each next to what the reader saw. The transcript stays, so the overlap of speech is the same as in T and the board is the only difference.
- W. The board with writer-side checks for honest agents.
- F. A fork of the ledger at the first meeting, so that T, B and W start from one state and differ from the meeting onward.

The measures are honest vote accuracy against the 22% to 30% of a random pick, honest wins, false claims by role, accusation accuracy, tokens per tick and rejections per honest claim. Vote accuracy rests on 240 to 250 honest votes in each cell of P2, so its intervals are narrower than the 27 to 40 points of the win rates. The fork matters for power. Games with one seed diverge during the movement phase because sampling is not deterministic, and that adds variance to every comparison, while a fork at the first meeting pairs the meetings.

A condition of 20 games took about 80 minutes of one RTX PRO 6000 for Gemma 4 31B in P2, 12 minutes for Granite 3.3 8B and 17 minutes for gpt-oss-120b, from the minutes per game at 6 games at a time. B and W on the 3 models then cost about 4 GPU-hours plus the server starts. Prompts will change size, so the first games need a look. The runs use vLLM on HiPerGator and the code is tested with the scripted adapter, with no calls to Navigator.

## Condition B as built

The code is `core/ledger.py` (records, readers, append, view, JSON lines), `core/board.py` (facts, claims, the board query) and `core/claims.py` (the rules that read claims, accusations and observation logs, moved out of the analysis script so that the room and the analysis use one copy).

- The engine writes one fact for each observation an agent logs, with the room, the agents beside it and the agents in the adjacent rooms, and only that agent may read it.
- The room writes the claims and accusations of a statement when the statement is committed, readable by everyone in the meeting. A statement is on the board from the time its last word is spoken, so a listener never sees the board ahead of the transcript. A statement that is cut off when the room closes is corrected by a note that cites its claims, and the ledger keeps both.
- The board lists at most 10 agents with 6 lines each. Each claim line says who made it, and then what the reader saw of the agent in that room: saw them there (with the round and tick, and whether it was next door), saw them only elsewhere, or did not see them. A claim about the reader is checked against where the reader was. Accusations and vouching are counted for each agent. The board says that not having seen something proves nothing, since a claim is true if it was true at any tick and a reader sees only part of the ship.
- With nothing claimed yet the percept stays empty, so the first tick reads as in T.
- `--room_board` turns the board on, the protocol `board-180` is `interleaved-180` with it, and a timed game writes `room/ledger.jsonl` in its folder. A game without the flag writes the ledger and leaves the prompts unchanged.
- Claims come from the rules of the analysis script, so they are as noisy as the claim measures of P2. A sentence that lists rooms gives a claim for each.

To play it on HiPerGator, with the model settings that the baseline row used:

```bash
TAG=p6-granite PROTOCOLS=board-180 GAMES_PER_CELL=20 sbatch scripts/hipergator/run_p2.sbatch granite-3.3-8b-instruct
```

## Build order

1. Done. `core/ledger.py`, a minimal ledger written for the room. Its field names differ from the group's ledger, so loading its traces there needs a small converter.
2. Done. The room writes claims, and the board follows the transcript in the percept.
3. Done for facts of positions. Events such as bodies and kills, and readers computed from the tile map, come later.
4. Next. Play `board-180` on the 3 models against the P2 rows of `interleaved-180`, with paired seeds.
5. Writer-side checks, then forks for paired runs, then the Godot server as the sequencer with spectators as readers.

## Open decisions

- Whether B comes before W, or whether the two run together.
- Whether the minimal ledger stays in this public repository. It is written for the room and copies no code or text from the private repository.
- Whether impostors may ever be constrained by checks.
