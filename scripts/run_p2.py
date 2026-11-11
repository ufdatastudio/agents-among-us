"""Plays the P2 grid of docs/situation-room.md against a running OpenAI-compatible server.

One game is one `main.py` process, and several run at once against the same server. A seed
visits every protocol before the next seed starts, so an interrupted run still has paired
cells. Finished games are appended to a JSON lines file, and a rerun skips them.

Usage, inside a job that has started vLLM and set VLLM_BASE_URL:
    python -m scripts.run_p2 --model ibm-granite/granite-3.3-8b-instruct --tag job123 --games-per-cell 5
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from config.eligible_models import check_eligible
from config.protocols import P2_PROTOCOLS, Protocol, protocol_by_name

RESULT_LINE = re.compile(r"Game Over\. Result: (.*)")
ROUND_LINE = re.compile(r"--- Round (\d+) Movement Phase")
STATEMENT_LINE = re.compile(r"^Agent_\d+: ", re.MULTILINE)
MEETING_LINE = re.compile(r"^=== Round \d+ ===", re.MULTILINE)


def composition(name: str, model: str, honest: int = 8, byzantine: int = 2, capture: bool = False) -> dict:
    """Builds the composition of a game in which every agent runs the same served model.

    Args:
        name: Scenario name, which is also the folder under `logs/`.
        model: Hugging Face checkpoint the server serves.
        honest: Number of honest agents.
        byzantine: Number of Byzantine agents.
        capture: Whether agents write private reasoning first.

    Returns:
        A composition dict for `main.py`.
    """
    return {
        "name": name,
        "honest_model": [f"vllm:{model}"],
        "byzantine_model": [f"vllm:{model}"],
        "honest_count": honest,
        "byzantine_count": byzantine,
        "capture_thoughts": capture,
    }


def plan(protocols: Sequence[Protocol], seeds: Iterable[int], done: set[tuple[str, int]]) -> list[tuple[Protocol, int]]:
    """Lists the games still to play, a seed at a time.

    Args:
        protocols: Protocols to compare.
        seeds: Scenario seeds.
        done: Pairs of protocol name and seed that have finished.

    Returns:
        Pairs of protocol and seed. Every protocol of one seed comes before the next seed.
    """
    return [(p, seed) for seed in seeds for p in protocols if (p.name, seed) not in done]


def game_command(protocol: Protocol, seed: int, game_id: str, scenario: str, rounds: int, room_workers: int) -> list[str]:
    """Builds the `main.py` command of one game.

    Args:
        protocol: The discussion protocol.
        seed: Scenario seed, also the seed of the room's onsets and shuffles.
        game_id: Unique id of the game.
        scenario: Composition name.
        rounds: Maximum number of rounds.
        room_workers: Agents that may generate at once in a timed meeting.

    Returns:
        The command line.
    """
    return [
        sys.executable, "-m", "main",
        "--composition_name", scenario, "--game_id", game_id, "--num_rounds", str(rounds),
        "--seed", str(seed), "--room_seed", str(seed), "--room_workers", str(room_workers),
        *protocol.args,
    ]


def parse_output(text: str) -> dict:
    """Reads the outcome of a game from what `main.py` printed.

    Args:
        text: Standard output and error of the game.

    Returns:
        The result line, the winning team, and the rounds whose movement phase started. The result
        and winner are None when the game never reached its end.
    """
    match = RESULT_LINE.search(text)
    result = match.group(1).strip() if match else None
    rounds = [int(r) for r in ROUND_LINE.findall(text)]
    return {
        "result": result,
        "winner": None if result is None else ("honest" if "Honest" in result else "byzantine"),
        "rounds_played": max(rounds, default=0),
    }


def read_stats(path: Path) -> dict:
    """Sums the vote and ejection counts of the game's stats file by team.

    Args:
        path: The `stats.csv` of one game.

    Returns:
        Correct, incorrect and skipped votes and ejections for each team, or an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    totals: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            team = "honest" if row["alignment"] == "H" else "byzantine"
            for key in ("correct_votes", "incorrect_votes", "skipped_votes", "ejections"):
                totals[f"{team}_{key}"] = totals.get(f"{team}_{key}", 0) + int(float(row[key] or 0))
    return totals


TOKEN_COLUMNS = (
    "api_input_tokens", "api_output_tokens", "api_calls", "api_reasoning_tokens", "api_reasoning_estimated_calls",
    "api_truncated_calls", "api_empty_replies", "api_max_output_tokens",
)


def read_tokens(path: Path) -> dict:
    """Reads the token totals of a game from its stats file.

    The totals are kept per model and not per agent, so every agent of a model repeats them.
    A game in which all agents run one model therefore has one set of totals, and the maximum
    over the rows gives it.

    Args:
        path: The `stats.csv` of one game.

    Returns:
        Input, output and reasoning tokens, calls, replies cut at the limit, empty replies and the
        longest reply, or an empty dict if the file is missing.
    """
    if not path.exists():
        return {}
    totals: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in TOKEN_COLUMNS:
                totals[column] = max(totals.get(column, 0), int(float(row.get(column) or 0)))
    return totals


def read_discussion(path: Path) -> dict:
    """Counts the meetings and statements in a game's discussion log.

    Args:
        path: The `discussion.log` of one game.

    Returns:
        The number of meetings and of statements.
    """
    text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    return {"meetings": len(MEETING_LINE.findall(text)), "statements": len(STATEMENT_LINE.findall(text))}


def read_redone(path: Path) -> int:
    """Counts the meetings of a game that were rewound because an agent had no valid vote.

    Args:
        path: The `voided_meetings.jsonl` of one game.

    Returns:
        The number of rewound meetings, 0 if the file is missing.
    """
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_done(path: Path) -> set[tuple[str, int]]:
    """Reads the games that finished from an earlier run.

    Args:
        path: The JSON lines file.

    Returns:
        Pairs of protocol name and seed of games that exited cleanly and reached a result.
    """
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("returncode") == 0 and record.get("result"):
            done.add((record["protocol"], record["seed"]))
    return done


def run_game(protocol: Protocol, seed: int, args: argparse.Namespace, scenario: str) -> dict:
    """Plays one game in its own process and collects its record.

    Args:
        protocol: The discussion protocol.
        seed: Scenario seed.
        args: Parsed command line.
        scenario: Composition name.

    Returns:
        The record of the game.
    """
    game_id = f"{args.tag}_{protocol.name}_{seed}"
    out_dir = Path("logs") / "p2_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    live_dir = Path("logs") / "p2_live"
    live_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "APP_MODE": "api", "LLM_MODE": "LOCAL", "AAU_LIVE_STATE": str(live_dir / f"{game_id}.json")}
    start = time.time()
    with (out_dir / f"{game_id}.out").open("w", encoding="utf-8") as handle:
        try:
            code = subprocess.run(
                game_command(protocol, seed, game_id, scenario, args.rounds, args.room_workers),
                env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=args.game_timeout_s,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -1
    wall = time.time() - start
    game_dir = Path("logs") / scenario / f"Game_{game_id}_Run0"
    text = (out_dir / f"{game_id}.out").read_text(encoding="utf-8", errors="ignore")
    record = {
        "tag": args.tag, "model": args.model, "protocol": protocol.name, "seed": seed, "game_id": game_id,
        "max_rounds": args.rounds, "returncode": code, "wall_s": round(wall, 1),
        # The reasoning effort and the reply limit the games ran with, so results of one model at
        # different settings are not mixed.
        "reasoning_effort": os.environ.get("VLLM_REASONING_EFFORT") or "default",
        "max_tokens": int(os.environ.get("AAU_MAX_TOKENS", "160")),
        **parse_output(text), **read_discussion(game_dir / "discussion.log"),
        # Meetings held again for lack of a vote, and whether the game stopped because the last attempt lacked one.
        "redone_meetings": read_redone(game_dir / "voided_meetings.jsonl"),
        "stopped_for_votes": "MeetingFailed" in text,
        "stats": read_stats(game_dir / "stats.csv"), "tokens": read_tokens(game_dir / "stats.csv"),
    }
    logger.info("{} {} seed {}: {} in {:.0f} s", game_id, protocol.name, seed, record["result"], wall)
    return record


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the command line.

    Args:
        argv: Arguments, or None for `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Hugging Face checkpoint on the eligible list")
    parser.add_argument("--tag", required=True, help="Label for this run, used in game ids")
    parser.add_argument("--games-per-cell", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6, help="Games played at once")
    parser.add_argument("--room-workers", type=int, default=10, help="Agents generating at once in a timed meeting")
    parser.add_argument("--protocols", default=",".join(p.name for p in P2_PROTOCOLS))
    parser.add_argument("--capture", action="store_true", help="Agents write private reasoning first")
    parser.add_argument("--game-timeout-s", type=int, default=3600)
    parser.add_argument("--out", type=Path, default=None, help="JSON lines file, default results/p2/<model>_<tag>.jsonl")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plays every game of the plan that has not finished.

    Args:
        argv: Arguments, or None for `sys.argv`.

    Returns:
        0 when every planned game finished, and 1 otherwise.
    """
    args = parse_args(argv)
    model = check_eligible(args.model)
    args.model = model
    protocols = [protocol_by_name(name) for name in args.protocols.split(",")]
    out = args.out or Path("results/p2") / f"{model.replace('/', '_')}_{args.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    scenario = f"p2_{model.replace('/', '_')}"
    config_dir = Path("logs") / "game_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{scenario}.json").write_text(json.dumps(composition(scenario, model, capture=args.capture)))

    seeds = range(args.seed_start, args.seed_start + args.games_per_cell)
    games = plan(protocols, seeds, load_done(out))
    logger.info("{} games to play with {} at a time, writing {}", len(games), args.workers, out)
    lock = threading.Lock()
    failures = 0

    def work(item: tuple[Protocol, int]) -> None:
        nonlocal failures
        record = run_game(item[0], item[1], args, scenario)
        with lock:
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            failures += record["returncode"] != 0 or record["result"] is None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, games))
    logger.info("Done with {} failed games", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
