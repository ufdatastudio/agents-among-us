"""Measure how a vLLM server handles simultaneous situation-room requests.

The situation room sends one request per ready agent at every tick, all at once.
This script checks that a vLLM server batches such requests and how much of a
growing prompt it serves from its prefix cache.

Measurements:
    sweep: n simultaneous requests with distinct prompts, for several n.
    sequential: the same requests one after another, as worker.py does today.
    growing: several agents over several ticks, each prompt extending the last.

Run it against a server started with --enable-prompt-tokens-details so that
cached token counts are reported:
    uv run --no-project --with openai --with loguru python -m scripts.bench_vllm_concurrency \
        --base-url http://host:port/v1 --model <served model name> --out result.json
"""

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are Agent_{agent}, an agent in a spaceship social deduction game. "
    "Everyone speaks at once during a timed discussion, and words from different "
    "speakers may interleave. Work out who said what, answer what others said, and "
    "argue your case. Reply with at most 40 words, or <silent/> if you have nothing to add."
)
ROOMS = [
    "Cafeteria", "MedBay", "Electrical", "Storage", "Admin",
    "Weapons", "Navigation", "Shields", "O2", "Reactor",
]
CLAIMS = [
    "I was in {room} with {other} the whole round and saw nothing odd.",
    "I found the body in {room} and saw {other} leaving just before.",
    "{other} was alone in {room}, so I think they are lying about their route.",
    "I did my task in {room}, and {other} can confirm that.",
    "{other} reported too quickly, and that makes me suspicious.",
    "I will vote {other} unless someone explains what happened in {room}.",
]
TICK_SECONDS = 10


def private_memory(agent: int, salt: int, lines: int = 30) -> str:
    """Builds an agent's private action log, which stays fixed during a meeting.

    Args:
        agent: Agent index.
        salt: Value that changes the content, so repeated runs do not hit the cache.
        lines: Number of log lines.

    Returns:
        The log as text.
    """
    rng = random.Random(f"memory-{agent}-{salt}")
    rows = [
        f"Step {i}: you moved to {rng.choice(ROOMS)} and saw Agent_{rng.randrange(10)}."
        for i in range(lines)
    ]
    return "=== Your private memory ===\n" + "\n".join(rows)


def board_lines(tick: int, num_agents: int, salt: int) -> list[str]:
    """Builds the lines every speaker adds to the board during one tick.

    Args:
        tick: Tick number, starting at 1.
        num_agents: Number of speakers.
        salt: Value that changes the content.

    Returns:
        One line per speaker.
    """
    rng = random.Random(f"board-{tick}-{salt}")
    start = (tick - 1) * TICK_SECONDS
    lines = []
    for speaker in range(num_agents):
        claim = rng.choice(CLAIMS).format(room=rng.choice(ROOMS), other=f"Agent_{rng.randrange(num_agents)}")
        lines.append(f"[{start + rng.randint(0, 2):03d}s] Agent_{speaker}: {claim}")
    return lines


def prompt_head(agent: int, tick: int, num_agents: int, salt: int) -> str:
    """Builds the part of a prompt that only grows from one tick to the next.

    The private memory comes first and the board follows, so the head of tick
    k is a prefix of the head of tick k + 1.

    Args:
        agent: Agent index.
        tick: Tick number, starting at 1.
        num_agents: Number of speakers.
        salt: Value that changes the content.

    Returns:
        The prompt head.
    """
    board: list[str] = []
    for earlier in range(1, tick):
        board.extend(board_lines(earlier, num_agents, salt))
    return private_memory(agent, salt) + "\n=== The room so far ===\n" + "\n".join(board)


def user_prompt(agent: int, tick: int, num_agents: int, salt: int) -> str:
    """Builds the full user prompt for one agent at one tick.

    Args:
        agent: Agent index.
        tick: Tick number, starting at 1.
        num_agents: Number of speakers.
        salt: Value that changes the content.

    Returns:
        The head followed by the instruction for this tick.
    """
    head = prompt_head(agent, tick, num_agents, salt)
    return f"{head}\n=== Now ===\nRoom time is {(tick - 1) * TICK_SECONDS} s. Reply now."


# Extra JSON fields sent with every request, set from --reasoning-effort.
EXTRA_BODY: dict | None = None


def call(client: OpenAI, model: str, system: str, user: str, max_tokens: int, seed: int) -> dict:
    """Sends one chat request and records its timing and token counts.

    Args:
        client: OpenAI client pointed at the vLLM server.
        model: Served model name.
        system: System prompt.
        user: User prompt.
        max_tokens: Completion limit.
        seed: Sampling seed.

    Returns:
        Latency in seconds, prompt, completion and cached token counts, and whether the reply
        text was empty, which happens when a reasoning model uses the whole budget on reasoning.
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=1.0,
        max_tokens=max_tokens,
        seed=seed,
        **({"extra_body": EXTRA_BODY} if EXTRA_BODY else {}),
    )
    latency = time.perf_counter() - start
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "empty": not (response.choices[0].message.content or "").strip(),
        "latency_s": latency,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cached_tokens": getattr(details, "cached_tokens", 0) or 0,
    }


def summarize(wall_s: float, results: list[dict]) -> dict:
    """Condenses the results of one round of requests.

    Args:
        wall_s: Time from the first submission to the last reply.
        results: Output of call() for each request.

    Returns:
        Wall time, latency statistics, throughput and the cached share of the prompt.
    """
    latencies = [r["latency_s"] for r in results]
    prompt = sum(r["prompt_tokens"] for r in results)
    completion = sum(r["completion_tokens"] for r in results)
    return {
        "requests": len(results),
        "wall_s": round(wall_s, 3),
        "mean_latency_s": round(statistics.mean(latencies), 3),
        "max_latency_s": round(max(latencies), 3),
        "prompt_tokens_mean": round(prompt / len(results)),
        "completion_tokens_mean": round(completion / len(results), 1),
        "completion_tokens_max": max(r["completion_tokens"] for r in results),
        "empty_fraction": round(sum(r["empty"] for r in results) / len(results), 3),
        "completion_tokens_per_s": round(completion / wall_s, 1),
        "cached_fraction": round(sum(r["cached_tokens"] for r in results) / prompt, 3),
    }


def run_round(client: OpenAI, model: str, jobs: list[tuple[str, str, int]], max_tokens: int) -> dict:
    """Sends all requests of a round at once and waits for every reply.

    Args:
        client: OpenAI client pointed at the vLLM server.
        model: Served model name.
        jobs: A (system, user, seed) tuple for each request.
        max_tokens: Completion limit.

    Returns:
        The summary of the round.
    """
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        start = time.perf_counter()
        futures = [pool.submit(call, client, model, s, u, max_tokens, seed) for s, u, seed in jobs]
        results = [future.result() for future in futures]
        wall = time.perf_counter() - start
    return summarize(wall, results)


def run_sequential(client: OpenAI, model: str, jobs: list[tuple[str, str, int]], max_tokens: int) -> dict:
    """Sends the requests one after another, the way worker.py serves them today.

    Args:
        client: OpenAI client pointed at the vLLM server.
        model: Served model name.
        jobs: A (system, user, seed) tuple for each request.
        max_tokens: Completion limit.

    Returns:
        The summary of the sequence.
    """
    results = []
    start = time.perf_counter()
    for system, user, seed in jobs:
        results.append(call(client, model, system, user, max_tokens, seed))
    return summarize(time.perf_counter() - start, results)


def make_jobs(count: int, tick: int, num_agents: int, salt: int) -> list[tuple[str, str, int]]:
    """Builds the requests for one tick.

    Args:
        count: Number of requests.
        tick: Tick number, starting at 1.
        num_agents: Number of speakers on the board.
        salt: Value that changes the content.

    Returns:
        A (system, user, seed) tuple for each request.
    """
    return [
        (
            SYSTEM_PROMPT.format(agent=agent),
            user_prompt(agent, tick, num_agents, salt),
            1000 * salt + 10 * tick + agent,
        )
        for agent in range(count)
    ]


def main() -> None:
    """Runs the sweep, the sequential baseline and the growing-board test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--levels", default="1,2,5,10,20")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--ticks", type=int, default=6)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default=None, help="For gpt-oss models")
    args = parser.parse_args()

    global EXTRA_BODY
    EXTRA_BODY = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=600)
    report: dict = {"model": args.model, "max_tokens": args.max_tokens, "reasoning_effort": args.reasoning_effort}

    logger.info("Warming up")
    run_round(client, args.model, make_jobs(2, 2, args.agents, salt=900), args.max_tokens)

    # Every round gets its own salt, so no round can reuse prefix-cache entries left by another.
    sweep = {}
    for level in [int(x) for x in args.levels.split(",")]:
        rounds = [
            run_round(
                client, args.model, make_jobs(level, 4, args.agents, salt=10_000 + 100 * level + rep), args.max_tokens
            )
            for rep in range(args.reps)
        ]
        sweep[level] = sorted(rounds, key=lambda r: r["wall_s"])[len(rounds) // 2]
        logger.info("sweep n={} median round: {}", level, sweep[level])
    report["sweep"] = sweep

    report["sequential"] = run_sequential(
        client, args.model, make_jobs(args.agents, 4, args.agents, salt=20_000), args.max_tokens
    )
    logger.info("sequential n={}: {}", args.agents, report["sequential"])

    growing = []
    for tick in range(1, args.ticks + 1):
        summary = run_round(client, args.model, make_jobs(args.agents, tick, args.agents, salt=30_000), args.max_tokens)
        summary["tick"] = tick
        growing.append(summary)
        logger.info("growing tick {}: {}", tick, summary)
    report["growing"] = growing

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote {}", args.out)


if __name__ == "__main__":
    main()
