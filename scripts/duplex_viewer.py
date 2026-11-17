"""Builds one HTML page that replays duplex meetings with a time slider.

The page reads what `scripts/duplex_spike.py` wrote for a run: the ledger of chunks, the log of each session and the
counts. A slider moves through the beats of the meeting. At each beat the page shows who had spoken so far, what each
agent heard in that beat, what the model wrote in answer and why the burst ended. Everything is embedded in the file,
so it opens without a server.

Usage:
    uv run python -m scripts.duplex_viewer results/duplex/gemma-a results/duplex/gemma-b --out results/duplex/viewer.html
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger

from core.duplex import SILENT_TAG
from core.ledger import CHUNK, Ledger
from core.stream import chunk_of
from scripts.duplex_spike import scenario

TEMPLATE = Path(__file__).with_name("duplex_viewer.html")
PLACEHOLDER = "/*__RUNS__*/[]"
_NOT_WORD = re.compile(r"[^a-z0-9 ]")
SKELETON = (
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n{page}\n</body>\n</html>\n'
)


def _silent(burst: str) -> bool:
    """Tells whether a burst has no words, which is an empty reply or only the silent tag."""
    return not SILENT_TAG.sub("", burst).strip()


def _plain(text: str) -> str:
    """Lowers a statement and drops punctuation, so that repeats of the same words compare equal."""
    return " ".join(_NOT_WORD.sub("", text.lower()).split())


def run_metrics(run: dict) -> dict:
    """Measures how a meeting went, for the table that compares runs.

    The measures describe the channel and the behavior of the models, and none of them says whether the talk was good.

    Args:
        run: The result of `load_run`.

    Returns:
        The number of beats, the share of session beats with words, the share of bursts cut at the token limit, the
        statements and their mean words, the share of statements whose words repeat an earlier statement, the mean
        number of voices at once, the share of beats with two or more voices, the engine seconds per beat and the
        tokens sent.
    """
    rows = [r for rows in run["sessions"].values() for r in rows]
    beats = max([len(rows) for rows in run["sessions"].values()] + [round(run["budget"] / run["beat"])])
    spoken = [r for r in rows if not _silent(r["burst"])]
    texts: dict[str, str] = {}
    for chunk in run["chunks"]:
        texts[chunk["stream"]] = texts.get(chunk["stream"], "") + chunk["text"]
    statements = [" ".join(t.split()) for t in texts.values() if t.strip()]
    seen: set[str] = set()
    repeats = 0
    for statement in statements:
        key = _plain(statement)
        repeats += key in seen
        seen.add(key)
    voices = []
    for b in range(beats):
        low, high = b * run["beat"], (b + 1) * run["beat"]
        voices.append(len({c["agent"] for c in run["chunks"] if c["text"].strip() and c["start"] < high - 1e-6 and c["end"] > low + 1e-6}))
    words = sum(len(s.split()) for s in statements)

    def share(count: int, whole: int) -> float | None:
        return round(count / whole, 4) if whole else None

    return {
        "beats": beats,
        "speech_beats": share(len(spoken), len(rows)),
        "cut_bursts": share(sum(r["finish"] == "length" for r in rows), len(rows)),
        "statements": len(statements),
        "words_per_statement": round(words / len(statements), 1) if statements else None,
        "repeated_statements": share(repeats, len(statements)),
        "voices_at_once": round(sum(voices) / len(voices), 2) if voices else None,
        "overlap_beats": share(sum(v >= 2 for v in voices), len(voices)),
        "engine_s_per_beat": round(sum(r.get("wall_s", 0.0) for r in rows) / len(rows), 3) if rows else None,
        "tokens_sent": sum(r.get("prompt_tokens", 0) for r in rows),
    }


def load_run(folder: Path) -> dict:
    """Reads one run folder into the data that the page shows.

    Args:
        folder: Folder with `ledger.jsonl`, `stats.json` and a `session_<agent>.jsonl` for each live agent.

    Returns:
        A dict with the settings of the run, every chunk, and the rows of each session.

    Raises:
        FileNotFoundError: If the folder has no ledger or no stats.
    """
    ledger_path, stats_path = folder / "ledger.jsonl", folder / "stats.json"
    for path in (ledger_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist, so {folder} is not a run of scripts.duplex_spike")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    args = stats.get("args", {})
    ledger = Ledger.read_jsonl(ledger_path)
    chunks = []
    for record in ledger.records:
        if record.type != CHUNK:
            continue
        chunk = chunk_of(record)
        chunks.append(
            {
                "agent": chunk.author, "stream": chunk.stream, "seq": chunk.seq, "modality": chunk.modality,
                "start": chunk.start_s, "end": chunk.end_s, "text": chunk.words, "final": chunk.final,
            }
        )
    sessions = {}
    for path in sorted(folder.glob("session_*.jsonl")):
        name = path.stem.removeprefix("session_")
        sessions[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    roles = stats.get("roles") or {}
    if not roles and args.get("agents") and not args.get("replay"):
        roles = {s.name: s.role for s in scenario(args["agents"], args.get("seed", 1))}
    authors = list(dict.fromkeys([c["agent"] for c in chunks] + list(sessions)))
    run = {
        "name": folder.name,
        "model": stats.get("model", ""),
        "beat": args.get("beat", 2.0),
        "budget": args.get("budget", max((c["end"] for c in chunks), default=0.0)),
        "mode": "replay" if args.get("replay") else "live",
        "audio": bool(args.get("audio")),
        "agents": sorted(authors, key=lambda a: (len(a), a)),
        "live": sorted(sessions),
        "roles": roles,
        "chunks": chunks,
        "sessions": sessions,
        "total": stats.get("total", {}),
        "agent_stats": stats.get("agents", {}),
        "wall_s": stats.get("wall_s", 0.0),
    }
    run["metrics"] = run_metrics(run)
    return run


def build_page(runs: list[dict], template: Path = TEMPLATE, standalone: bool = True) -> str:
    """Puts the runs into the page.

    Args:
        runs: The result of `load_run` for each run.
        template: The page without its data.
        standalone: Whether to add the head and body tags that a browser needs to open the file. The artifact tool adds them itself.

    Returns:
        The HTML.

    Raises:
        ValueError: If the template has no place for the data.
    """
    page = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in page:
        raise ValueError(f"{template} has no {PLACEHOLDER} to fill")
    data = json.dumps(runs, separators=(",", ":")).replace("</", "<\\/")
    page = page.replace(PLACEHOLDER, data)
    return SKELETON.format(page=page) if standalone else page


def main(argv: list[str] | None = None) -> int:
    """Writes the page.

    Args:
        argv: Command line arguments, or None for `sys.argv`.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run folders")
    parser.add_argument("--out", type=Path, default=Path("results/duplex/viewer.html"))
    parser.add_argument("--fragment", action="store_true", help="leave out the head and body tags, for the artifact tool")
    args = parser.parse_args(argv)
    runs = [load_run(folder) for folder in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_page(runs, standalone=not args.fragment), encoding="utf-8")
    logger.info("wrote {} with {} runs, {} bytes", args.out, len(runs), args.out.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
