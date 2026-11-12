# Shared ledger for the room and the world

Status is design only, and no code exists. The ledger idea comes from the group's work on crews of agents that operate a shared database. That work is in a private repository and a manuscript under review, so this page describes the ledger in general terms and cites only public work. The questions are whether the room and the Godot world should keep their state in one ledger, what that changes for agents inside and outside a meeting, and how to measure it.

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

The descriptions come from the abstracts read on 2026-11-12, except S-Bus, whose full text was read.

| Work | Shared structure and mechanism | Fit for the room |
|------|--------------------------------|------------------|
| AgileLog (Bhat et al., SOSP 2026, arXiv 2604.14590) | A forkable shared log. An agent works on a fork that is isolated from the live log in logic and in performance | Forks give the rewind of a meeting and a way to run several protocols from one meeting start |
| PatchBoard (Zhang et al., arXiv 2605.29313) | A shared JSON state that agents change through schema-checked patches in place of dialogue, with a kernel that checks role-specific write contracts. It reports 84.6% success on ALFWorld against 30.8% for LangGraph | It replaces speech with state writes, so it fits as a condition with no talk and leaves out what the room studies |
| CoAgent (Lyu et al., arXiv 2606.15376) | Concurrency control for agents that write one git tree, cluster or document. It fixes a serialization order, applies writes speculatively and undoes misplaced ones | A meeting has no write conflicts, because claims append and votes are independent. The server settles world actions such as doors and tasks |
| S-Bus (Khan, arXiv 2605.17076) | Keyed shards of opaque text. The server logs each read over HTTP, and a commit that rests on a stale read of another shard is rejected | Ruled out as the shared structure of the room, see the next section |
| CodeCRDT (Pugachev, arXiv 2510.18893) | CRDTs for concurrent code generation with eventual consistency | A set of appended claims already converges, and the game needs a fixed order of events, which eventual consistency does not give |
| STORM (Liu et al., arXiv 2605.20563) | A consistent view of a shared codebase, with conflicting edits detected when they are written | The same fit as CoAgent |
| MAP-Graph (Wang et al., arXiv 2608.10509) | A typed graph of agents, sources, memories, claims and actions, with provenance, permission filtering and retrieval weighted by the trust of the path. It has no concurrency control | Closest to typed records with views, and a reference for ranking claims by the trust of their source |

Eugene Wu's group (DAPLab, Columbia) is the other place to look. I read its publication page, his homepage and the arXiv abstracts on 2026-11-12.

- I found no paper of the group that proposes a shared data structure for several agents with consistency levels. Its published pieces are about the environment of one agent. Agentic Data Environments (arXiv 2607.07397) names branching for state safety and data flow control for data safety, BranchBench (arXiv 2604.17180) benchmarks branchable databases, and StateFork (arXiv 2609.38648, title only) builds branchable infrastructure. Data flow control (arXiv 2606.05679) enforces policies over how data flows through queries, and Environment Steering (arXiv 2609.35807) models the execution state of an agent as tables and steers the agent when a policy is violated.
- Multi-agent coordination appears as a benchmark, Passes Alone, Fails Together (arXiv 2609.25396, title and venue only), and as a topic of a course announced for spring 2027, Data Systems with and for Agents.
- Two parts transfer to the room. Branching gives paired runs from one meeting start, and data flow policies give a way to state what an agent may read.

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
- B. Board, reader side. Utterances and engine facts are ledger records, and the percept of a reader is a query that groups the statements by subject, each with its status against the reader's facts.
- W. The board with writer-side checks for honest agents.
- F. A fork of the ledger at the first meeting, so that T, B and W start from one state and differ from the meeting onward.

The measures are honest vote accuracy against the 22% to 30% of a random pick, honest wins, false claims by role, accusation accuracy, tokens per tick and rejections per honest claim. Vote accuracy rests on 240 to 250 honest votes in each cell of P2, so its intervals are narrower than the 27 to 40 points of the win rates. The fork matters for power. Games with one seed diverge during the movement phase because sampling is not deterministic, and that adds variance to every comparison, while a fork at the first meeting pairs the meetings.

A condition of 20 games took about 80 minutes of one RTX PRO 6000 for Gemma 4 31B in P2, 12 minutes for Granite 3.3 8B and 17 minutes for gpt-oss-120b, from the minutes per game at 6 games at a time. B and W on the 3 models then cost about 4 GPU-hours plus the server starts. Prompts will change size, so the first games need a look. The runs use vLLM on HiPerGator and the code is tested with the scripted adapter, with no calls to Navigator.

## Build order

1. `core/ledger.py` with records, readers, append, view and a board query, tested with the scripted adapter. A minimal copy with field names that match the group's ledger keeps this repository self-contained and lets the traces load in the group's tools.
2. The room writes utterances as claims, and a board renderer sits beside the three text renderers.
3. The engine writes facts for positions and events, with the readers computed from the room each agent is in, and later from the tile map.
4. Condition B on a small model with the scripted adapter, then P6 on HiPerGator.
5. Writer-side checks, then forks for paired runs, then the Godot server as the sequencer with spectators as readers.

## Open decisions

- Whether B comes before W, or whether the two run together.
- Whether the minimal ledger goes into this public repository now or stays in the private one until the paper is public.
- Whether impostors may ever be constrained by checks.
