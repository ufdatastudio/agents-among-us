"""Spike: agents that listen and speak through streaming sessions of one vLLM engine, with the ledger as the channel.

A short meeting about a body in Storage is played beat by beat (see `core/duplex.py`). Each agent has one session whose
context grows by the chunks it hears, and each answer is a burst of a few tokens that goes to the ledger as a chunk. The
run writes what the sessions saw and wrote so that the behavior of the model can be read, not only counted.

With --replay the other agents are a recorded game. Its speech is written to the ledger as chunks in the order it
was spoken, and only the listeners run in sessions. With --audio a listener gets the chunks as audio and with no
flag it gets the words, so the same meeting can be heard both ways.

Usage:
    uv run --no-project --with vllm python -m scripts.duplex_spike --model gemma-4-31b-it --out results/duplex/gemma
    uv run --no-project --with vllm python -m scripts.duplex_spike --model gemma-4-31b-it --replay results/speech/game \
        --game logs/Game_x_Run0 --listener Agent_3 --budget 180 --out results/duplex/replay
    uv run --no-project --with transformers python -m scripts.duplex_spike --model gemma-4-31b-it --probe
"""

import argparse
import asyncio
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from agents.prompts import BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX, HONEST_DISCUSSION_INSTRUCTIONS_PREFIX
from config.eligible_models import check_eligible
from core.audio.speech import INDEX
from core.duplex import Burst, DuplexAgent, DuplexMeeting, StreamHandle, StreamPrompt, TurnFormat, run_duplex_meeting
from core.ledger import CHUNK, Ledger
from core.situation_room import format_time
from core.stream import chunk_of, write_replay

ROOMS = ("Cafeteria", "Storage", "Shields", "Engine", "Navigation", "Reactor")
WORDS_PER_S = 2.5
TOKENS_PER_WORD = 1.6
DEFAULT_TEMPLATE_KWARGS = {"enable_thinking": False}
AUDIO_PLACEHOLDERS = {"nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16": "<so_embedding>"}

DUPLEX_RULES = """
6. This meeting is not in turns. Time runs in beats of {beat_s:.0f} seconds. Each message you get has what others said during the last beat, as lines like "[00:12] Agent_3: words", or "..." when nobody spoke. People talk over each other, so a line may stop in the middle of a sentence.
7. Most of the time you listen. Reply with <silent/> in every beat in which you have nothing new to add. After you finish a statement, reply with <silent/> for at least two beats so that others can answer.
8. When you speak, reply with only the next few words, at most {max_words} words. A sentence may take several beats. When you speak again in the middle of a sentence, continue where your last reply stopped and do not repeat words.
9. Keep each statement to {statement_words} words or fewer, then stop. Do not repeat what you or anyone else already said, and do not shout the same vote again.
10. Speak in first person. Do NOT write your name, a time stamp, quotes or anyone else's lines. Output only your spoken words, or <silent/>.
"""

SYSTEM = {
    "honest": "You are {name}, an honest crew member of a ship. A crew member was killed, and you are in a meeting to find the Byzantine agent who did it.",
    "byzantine": "You are {name}, a Byzantine agent hidden among the crew of a ship. You killed a crew member, and you are in a meeting where you must avoid blame.",
}


@dataclass(frozen=True)
class AgentSetup:
    """What one agent knows at the start of the meeting.

    Attributes:
        name: Agent name.
        role: `honest` or `byzantine`.
        memory: Lines of what the agent saw in the previous round.
    """

    name: str
    role: str
    memory: str


def scenario(agents: int, seed: int) -> list[AgentSetup]:
    """Builds a small round that ends with a body in Storage.

    Agent_0 is Byzantine and kills the last agent in Storage at tick 3. Everyone is in a seeded room at each tick and
    sees the agents in the same room. Agent_1 finds the body and calls the meeting.

    Args:
        agents: Number of agents, at least 4.
        seed: Seed for the rooms.

    Returns:
        One setup for each agent, except the victim.

    Raises:
        ValueError: If there are fewer than 4 agents.
    """
    if agents < 4:
        raise ValueError("a meeting needs at least 4 agents")
    rng = random.Random(seed)
    names = [f"Agent_{i}" for i in range(agents)]
    victim = names[-1]
    places = {n: [rng.choice(ROOMS) for _ in range(3)] for n in names}
    places["Agent_0"][2] = places[victim][2] = "Storage"
    places["Agent_1"][2] = "Storage"
    setups = []
    for name in names[:-1]:
        lines = ["Round 1."]
        for tick in range(3):
            room = places[name][tick]
            others = [n for n in names if n != name and places[n][tick] == room and not (tick == 2 and n == victim and name != "Agent_0")]
            lines.append(f"Tick {tick + 1}: you were in {room}. You saw: {', '.join(others) or 'nobody'}.")
        lines.append(f"Then Agent_1 found {victim} dead in Storage and called a meeting.")
        setups.append(AgentSetup(name, "byzantine" if name == "Agent_0" else "honest", "\n".join(lines)))
    return setups


def beat_tokens(beat_s: float) -> int:
    """Gives the token limit of one burst, for the words that fit in a beat at the speaking pace of the room.

    Args:
        beat_s: Room time of one beat.

    Returns:
        The number of tokens.
    """
    return max(2, math.ceil(beat_s * WORDS_PER_S * TOKENS_PER_WORD))


def intro(setup: AgentSetup, beat_s: float, budget_s: float, statement_words: int) -> str:
    """Writes the first message of a session, without the first beat.

    Args:
        setup: What the agent knows.
        beat_s: Room time of one beat.
        budget_s: Room time of the meeting.
        statement_words: Words after which a statement should stop.

    Returns:
        The message.
    """
    prefix = BYZANTINE_DISCUSSION_INSTRUCTIONS_PREFIX if setup.role == "byzantine" else HONEST_DISCUSSION_INSTRUCTIONS_PREFIX
    rules = DUPLEX_RULES.format(
        beat_s=beat_s, max_words=max(2, round(beat_s * WORDS_PER_S)), statement_words=statement_words
    )
    return (
        f"=== Your personal memory log of what you saw ===\n{setup.memory}\n==================\n"
        f"The meeting lasts {format_time(budget_s)}.\n{prefix}{rules}"
    )


def round_log(text: str, round_num: int) -> str:
    """Cuts the part of an action log that belongs to one round.

    Args:
        text: The action log of an agent.
        round_num: The round to keep.

    Returns:
        The lines from the header of the round up to the header of the next one.
    """
    start = re.search(rf"^Round {round_num}/\d+\s*$", text, re.MULTILINE)
    if start is None:
        return text.strip()
    end = re.search(rf"^Round {round_num + 1}/\d+\s*$", text[start.end() :], re.MULTILINE)
    return text[start.start() : start.end() + end.start() if end else len(text)].strip()


def replay_setup(
    game: Path, speech: Path, listeners: list[str], round_num: int
) -> tuple[list[AgentSetup], list[tuple[str, Path, float, str]]]:
    """Reads a recorded meeting for live listeners.

    Args:
        game: Folder of the recorded game, with the action log of each honest agent.
        speech: Folder of the clips that `scripts/world_audio.py` wrote for that game.
        listeners: Honest agents who take part live.
        round_num: Round whose meeting is replayed.

    Returns:
        The setup of each listener, and for each recorded clip of the other agents the author, the file, the start and the words.

    Raises:
        FileNotFoundError: If a listener has no action log, so it was not an honest agent of the game.
    """
    setups = []
    for name in listeners:
        log = game / "Honest" / name / "action.log"
        if not log.exists():
            raise FileNotFoundError(f"{log} does not exist, so {name} is not an honest agent of {game}")
        setups.append(AgentSetup(name, "honest", round_log(log.read_text(encoding="utf-8"), round_num)))
    rows = json.loads((speech / INDEX).read_text(encoding="utf-8"))["clips"]
    clips = [
        (r["author"], speech / r["file"], r["start_s"], r["text"])
        for r in rows
        if r["round_num"] == round_num and r["author"] not in listeners
    ]
    return setups, clips


class RecordingHandle:
    """Passes a session through and keeps what it was sent and what it wrote."""

    def __init__(self, inner: StreamHandle, name: str) -> None:
        """Wraps a session.

        Args:
            inner: The session.
            name: Agent name.
        """
        self.inner = inner
        self.name = name
        self.rows: list[dict] = []

    async def send(self, prompt: StreamPrompt, max_tokens: int) -> Burst:
        """Sends a chunk and records it with the burst."""
        burst = await self.inner.send(prompt, max_tokens)
        self.rows.append({"sent": prompt.text, "burst": burst.text, "finish": burst.finish_reason, **{
            "prompt_tokens": burst.prompt_tokens, "output_tokens": burst.output_tokens, "wall_s": round(burst.wall_s, 3)}})
        return burst

    async def close(self) -> None:
        """Closes the session."""
        await self.inner.close()


def transcript(ledger: Ledger) -> str:
    """Lists every stream of the ledger with its first and last time, ordered by start.

    Args:
        ledger: The ledger of the meeting.

    Returns:
        One line for each statement.
    """
    streams: dict[str, dict] = {}
    for record in ledger.records:
        if record.type != CHUNK:
            continue
        chunk = chunk_of(record)
        entry = streams.setdefault(chunk.stream, {"author": chunk.author, "start": chunk.start_s, "end": chunk.end_s, "text": ""})
        entry["end"] = max(entry["end"], chunk.end_s)
        entry["text"] += chunk.words
    rows = sorted(streams.values(), key=lambda e: (e["start"], e["author"]))
    return "\n".join(f"[{format_time(e['start'])}-{format_time(e['end'])}] {e['author']}: {' '.join(e['text'].split())}" for e in rows)


def summarize(meeting: DuplexMeeting) -> dict:
    """Totals the counts of a meeting.

    Args:
        meeting: The result of the run.

    Returns:
        Totals over the agents, the counts of each agent, and the time of the meeting.
    """
    stats = {name: asdict(s) for name, s in meeting.stats.items()}
    total = {key: sum(s[key] for s in stats.values()) for key in next(iter(stats.values()))}
    return {"beats": meeting.beats, "wall_s": round(meeting.wall_s, 1), "total": total, "agents": stats}


def write_results(
    out: Path, ledger: Ledger, meeting: DuplexMeeting, handles: dict[str, RecordingHandle], checkpoint: str, args: dict, roles: dict
) -> None:
    """Writes the ledger, the transcript, the counts and the log of each session.

    Args:
        out: Output folder. It is created.
        ledger: The ledger of the meeting.
        meeting: The result of the run.
        handles: The recording session of each live agent.
        checkpoint: Hugging Face checkpoint.
        args: The command line as a dict.
        roles: Role of each live agent, when the scenario sets them.
    """
    out.mkdir(parents=True, exist_ok=True)
    ledger.write_jsonl(out / "ledger.jsonl")
    (out / "transcript.txt").write_text(transcript(ledger) + "\n", encoding="utf-8")
    stats = {**summarize(meeting), "model": checkpoint, "args": args, "roles": roles}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    for name, handle in handles.items():
        (out / f"session_{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in handle.rows), encoding="utf-8")


def build_engine(checkpoint: str, max_model_len: int, memory_fraction: float, extra: dict) -> tuple[Any, Any]:
    """Starts a vLLM engine and loads the tokenizer.

    Args:
        checkpoint: Hugging Face checkpoint.
        max_model_len: Context length.
        memory_fraction: Share of the GPU memory for the engine.
        extra: More arguments for the engine, such as `trust_remote_code`.

    Returns:
        The engine and the tokenizer.
    """
    from transformers import AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    args = AsyncEngineArgs(
        model=checkpoint, max_model_len=max_model_len, gpu_memory_utilization=memory_fraction, enable_prefix_caching=True,
        **extra,
    )
    return AsyncLLM.from_engine_args(args), AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=extra.get("trust_remote_code", False))


def turn_format(tokenizer: Any, template_kwargs: dict) -> TurnFormat:
    """Finds the turn format of the chat template of a tokenizer.

    Args:
        tokenizer: A Hugging Face tokenizer with a chat template.
        template_kwargs: Arguments for the template, such as `enable_thinking`.

    Returns:
        The format.
    """

    def render(messages: list[dict], add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
        )

    return TurnFormat.probe(render)


async def play(args: argparse.Namespace, checkpoint: str) -> DuplexMeeting:
    """Plays the meeting and writes the results.

    Args:
        args: The command line.
        checkpoint: Hugging Face checkpoint.

    Returns:
        The meeting.
    """
    from core.vllm_stream import VllmStreamEngine

    engine, tokenizer = build_engine(checkpoint, args.max_model_len, args.memory_fraction, json.loads(args.engine_args))
    fmt = turn_format(tokenizer, json.loads(args.template_kwargs))
    ledger = Ledger()
    opener = VllmStreamEngine(engine, tokenizer, args.temperature)
    if args.replay:
        setups, clips = replay_setup(Path(args.game), Path(args.replay), args.listener.split(","), args.round)
        write_replay(ledger, clips, args.beat, round_num=args.round)
    else:
        setups = scenario(args.agents, args.seed)
    placeholder = AUDIO_PLACEHOLDERS.get(checkpoint) if args.audio else None
    handles = {}
    agents = []
    for setup in setups:
        handles[setup.name] = RecordingHandle(opener.open(setup.name), setup.name)
        agents.append(
            DuplexAgent(
                setup.name, handles[setup.name], fmt, SYSTEM[setup.role].format(name=setup.name),
                intro(setup, args.beat, args.budget, args.statement_words), ledger.subscribe(setup.name),
                args.beat, beat_tokens(args.beat), audio_placeholder=placeholder, round_num=args.round,
            )
        )
    try:
        meeting = await run_duplex_meeting(agents, ledger, args.budget, args.beat)
    finally:
        engine.shutdown()
    roles = {} if args.replay else {setup.name: setup.role for setup in setups}
    write_results(Path(args.out), ledger, meeting, handles, checkpoint, vars(args), roles)
    logger.info("meeting of {} beats took {:.0f} s, results in {}", meeting.beats, meeting.wall_s, args.out)
    return meeting


def probe(checkpoint: str, template_kwargs: dict) -> None:
    """Prints how the chat template joins turns, without loading the model.

    Args:
        checkpoint: Hugging Face checkpoint.
        template_kwargs: Arguments for the template.
    """
    from transformers import AutoTokenizer

    fmt = turn_format(AutoTokenizer.from_pretrained(checkpoint), template_kwargs)
    first = fmt.opening("SYSTEM PROMPT", "FIRST MESSAGE")
    logger.info("gap: {!r}", fmt.gap)
    logger.info("cue: {!r}", fmt.cue)
    logger.info("opening: {!r}", first)
    logger.info("next: {!r}", fmt.next_turn("[00:02] ..."))


def main(argv: list[str] | None = None) -> int:
    """Runs the spike.

    Args:
        argv: Command line arguments, or None for `sys.argv`.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="eligible model name or checkpoint")
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--budget", type=float, default=90.0, help="room time of the meeting in seconds")
    parser.add_argument("--beat", type=float, default=2.0, help="room time of one beat in seconds")
    parser.add_argument("--statement-words", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    parser.add_argument("--template-kwargs", default=json.dumps(DEFAULT_TEMPLATE_KWARGS))
    parser.add_argument("--engine-args", default="{}", help="JSON with more arguments for the vLLM engine")
    parser.add_argument("--replay", help="folder of clips from scripts.world_audio to replay as the other agents")
    parser.add_argument("--game", help="folder of the recorded game, for the log of the listener")
    parser.add_argument("--listener", default="Agent_3", help="honest agents who run live in a replay, separated by commas")
    parser.add_argument("--round", type=int, default=1, help="round whose meeting is replayed")
    parser.add_argument("--audio", action="store_true", help="give the listener audio chunks, for a model that hears")
    parser.add_argument("--out", default="results/duplex/spike")
    parser.add_argument("--probe", action="store_true", help="print the turn format of the chat template and stop")
    args = parser.parse_args(argv)
    checkpoint = check_eligible(args.model)
    if args.replay and not args.game:
        parser.error("--replay needs --game")
    if args.audio and not (args.replay and checkpoint in AUDIO_PLACEHOLDERS):
        parser.error("--audio needs --replay and a model that hears")
    if args.probe:
        probe(checkpoint, json.loads(args.template_kwargs))
        return 0
    asyncio.run(play(args, checkpoint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
