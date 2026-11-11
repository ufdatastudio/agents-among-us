"""Summarizes the games written by scripts/run_p2.py, one row per model and protocol.

Usage:
    python -m scripts.summarize_p2 results/p2/*.jsonl
"""

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path

from config.protocols import P2_PROTOCOLS

PROTOCOL_ORDER = {p.name: i for i, p in enumerate(P2_PROTOCOLS)}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Gives the Wilson score interval of a proportion.

    Args:
        successes: Number of successes.
        n: Number of trials.
        z: Normal quantile, 1.96 for a 95% interval.

    Returns:
        The lower and upper bound, or (0, 1) when there are no trials.
    """
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def load(paths: Iterable[Path]) -> list[dict]:
    """Reads the game records of several JSON lines files.

    Args:
        paths: Files written by run_p2.

    Returns:
        Every record.
    """
    return [json.loads(line) for path in paths for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def summarize(records: Iterable[dict]) -> list[dict]:
    """Groups finished games by model and protocol.

    Args:
        records: Game records. A game with no result counts as a failure and is left out of the rates.

    Returns:
        One row per model and protocol, in the order of the P2 protocols.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["model"], _settings(record), record["protocol"]), []).append(record)
    rows = []
    for (model, effort, protocol), games in groups.items():
        finished = [g for g in games if g.get("result")]
        wins = sum(g["winner"] == "honest" for g in finished)
        low, high = wilson(wins, len(finished))
        correct = sum(g.get("stats", {}).get("honest_correct_votes", 0) for g in finished)
        incorrect = sum(g.get("stats", {}).get("honest_incorrect_votes", 0) for g in finished)
        skipped = sum(g.get("stats", {}).get("honest_skipped_votes", 0) for g in finished)
        redone = [g["redone_meetings"] for g in games if "redone_meetings" in g]
        tokens = {k: sum(g.get("tokens", {}).get(k, 0) for g in finished) for k in ("api_output_tokens", "api_reasoning_tokens", "api_calls", "api_truncated_calls", "api_empty_replies", "api_errors")}
        rows.append(
            {
                "model": model,
                "effort": effort,
                "token_limit": max((g.get("max_tokens", 0) for g in games), default=0),
                "protocol": protocol,
                "games": len(finished),
                "failed": len(games) - len(finished),
                "honest_win_rate": wins / len(finished) if finished else None,
                "ci_low": low,
                "ci_high": high,
                "honest_vote_accuracy": correct / (correct + incorrect) if correct + incorrect else None,
                "honest_skipped_share": _ratio(skipped, correct + incorrect + skipped),
                "redone_per_game": statistics.mean(redone) if redone else None,
                "mean_rounds": statistics.mean(g["rounds_played"] for g in finished) if finished else None,
                "mean_meetings": statistics.mean(g["meetings"] for g in finished) if finished else None,
                "mean_statements": statistics.mean(g["statements"] for g in finished) if finished else None,
                "mean_minutes": statistics.mean(g["wall_s"] for g in finished) / 60 if finished else None,
                "reasoning_share": _ratio(tokens["api_reasoning_tokens"], tokens["api_output_tokens"]),
                "truncated_share": _ratio(tokens["api_truncated_calls"], tokens["api_calls"]),
                "empty_share": _ratio(tokens["api_empty_replies"], tokens["api_calls"]),
                "tokens_per_call": _ratio(tokens["api_output_tokens"], tokens["api_calls"]),
                "errors_per_game": _ratio(tokens["api_errors"], len(finished)),
            }
        )
    return sorted(rows, key=lambda r: (r["model"], r["effort"], PROTOCOL_ORDER.get(r["protocol"], len(PROTOCOL_ORDER))))


def _settings(record: dict) -> str:
    """Names the settings of a game that change its results, so that runs with different ones stay apart.

    Args:
        record: A game record.

    Returns:
        The reasoning effort, followed by `+think` when agents wrote private reasoning and by `words N`
        when the statement limit is not the default of 40.
    """
    label = record.get("reasoning_effort") or "default"
    if record.get("capture"):
        label += " +think"
    if record.get("room_max_words", 40) != 40:
        label += f" words {record['room_max_words']}"
    return label


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def format_table(rows: Sequence[dict]) -> str:
    """Formats the rows as a Markdown table.

    Args:
        rows: Output of summarize().

    Returns:
        The table.
    """
    lines = [
        "| Model | Effort | Token limit | Protocol | Games | Failed | Honest win rate | 95% interval | Honest vote accuracy | Rounds | Meetings | Statements | Minutes | Tokens per call | Reasoning share | Cut at limit | Empty | Honest votes skipped | Meetings redone per game | API errors per game |",
        "|-------|--------|-------------|----------|-------|--------|-----------------|--------------|----------------------|--------|----------|------------|---------|-----------------|-----------------|--------------|-------|---------------------|--------------------------|---------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['effort']} | {r['token_limit']} | {r['protocol']} | {r['games']} | {r['failed']} "
            f"| {_fmt(r['honest_win_rate'], '.2f')} | {r['ci_low']:.2f} to {r['ci_high']:.2f} "
            f"| {_fmt(r['honest_vote_accuracy'], '.2f')} | {_fmt(r['mean_rounds'], '.1f')} | {_fmt(r['mean_meetings'], '.1f')} "
            f"| {_fmt(r['mean_statements'], '.0f')} | {_fmt(r['mean_minutes'], '.1f')} | {_fmt(r['tokens_per_call'], '.0f')} "
            f"| {_fmt(r['reasoning_share'], '.0%')} | {_fmt(r['truncated_share'], '.1%')} | {_fmt(r['empty_share'], '.1%')} "
            f"| {_fmt(r['honest_skipped_share'], '.0%')} | {_fmt(r['redone_per_game'], '.2f')} | {_fmt(r['errors_per_game'], '.1f')} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """Prints the summary table of the given files.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(format_table(summarize(load(args.files))))


if __name__ == "__main__":
    main()
