"""Draws the figures of docs/situation-room.md from the numbers of the experiments.

Two steps. `data` collects the numbers into `docs/figures/data.json` from the P2 result files, the P2 game
folders (through scripts/analyze_p2_text.py) and the measurement tables of the doc. `plot` draws every figure
from that file, so the figures can be redrawn without the game folders, which are too large for the repository.

Usage:
    uv run python -m scripts.make_figures data --p2 results/p2/*.jsonl \
        --logs granite=logs/p2_ibm-granite_granite-3.3-8b-instruct gptoss=... gemma=... --out docs/figures/data.json
    uv run --with matplotlib python -m scripts.make_figures plot --data docs/figures/data.json --out docs/figures

Colors follow the reference palette of the dataviz guidance. A model keeps one color in every figure, the role
of an agent is shown by texture (Byzantine agents are hatched), and every figure has a table in the doc.
"""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from scripts.analyze_p2_text import (
    accusation_profile,
    claim_profile,
    influence_profile,
    listening_profile,
    load_games,
    speech_profile,
)
from scripts.summarize_p2 import wilson

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
NEUTRAL = "#8a8985"
MODEL_NAMES = {
    "granite-3.3-8b-instruct": "Granite 3.3 8B",
    "gpt-oss-120b": "gpt-oss-120b",
    "gemma-4-31B-it": "Gemma 4 31B",
}
MODEL_COLORS = {"Granite 3.3 8B": "#2a78d6", "gpt-oss-120b": "#eb6834", "Gemma 4 31B": "#1baf7a"}
MODELS = list(MODEL_COLORS)
SETTINGS = ("no discussion", "turns", "timed room")
ACTS = (("accuse", "Accuse"), ("vote_call", "Call a vote"), ("defend", "Defend self"), ("report", "Report"), ("question", "Question"), ("other", "Other"))
WHO = ("nobody", "honest only", "byzantine only", "both")


def short_model(name: str) -> str:
    """Gives a model's display name from the name in a game record."""
    return MODEL_NAMES[name.split("/")[-1].removeprefix("vllm:")]


def setting_of(protocol: str) -> str:
    """Names the discussion setting of a protocol."""
    return "no discussion" if protocol == "none" else "turns" if protocol == "sequential" else "timed room"


def collect_p2(files: Sequence[Path]) -> dict:
    """Summarizes the P2 result files by model and setting, and by protocol for each model.

    Args:
        files: JSON lines files written by scripts/run_p2.py.

    Returns:
        For each model, the games, honest wins, interval, cast honest votes, correct votes and skipped votes of
        each setting under `settings`, and of each protocol under `protocols`.
    """
    cells: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for path in files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not record.get("result"):
                continue
            stats = record.get("stats", {})
            for key in (("settings", setting_of(record["protocol"])), ("protocols", record["protocol"])):
                cell = cells[(short_model(record["model"]), *key)]
                cell["games"] += 1
                cell["wins"] += record["winner"] == "honest"
                cell["correct"] += stats.get("honest_correct_votes", 0)
                cell["incorrect"] += stats.get("honest_incorrect_votes", 0)
                cell["skipped"] += stats.get("honest_skipped_votes", 0)
    out: dict = {}
    for (model, kind, name), cell in cells.items():
        low, high = wilson(cell["wins"], cell["games"])
        out.setdefault(model, {"settings": {}, "protocols": {}})[kind][name] = {**cell, "ci_low": low, "ci_high": high}
    return out


def collect_text(folders: Sequence[Path]) -> dict:
    """Measures the transcripts of P2 games, for the clean room and for turn-taking.

    Args:
        folders: Folders of game folders, one for each model.

    Returns:
        For each model and group (`clean` or `sequential`), the counters of the speech, claim, accusation, influence
        and listening profiles. The keys of the profiles are joined with a vertical bar.
    """
    games = load_games(folders)
    keep = [g for g in games if g.statements and (g.protocol.startswith("clean") or g.protocol == "sequential")]
    out: dict = defaultdict(lambda: defaultdict(dict))

    def group(g) -> str:
        return "sequential" if g.protocol == "sequential" else "clean"

    for profile, name in ((speech_profile, "speech"), (claim_profile, "claims"), (accusation_profile, "accusations")):
        merged = defaultdict(lambda: defaultdict(float))
        for g in keep:
            for (model, grp, role), cell in profile([g]).items():
                target = merged[(short_model(model), group(g), role)]
                for key, value in cell.items():
                    if key == "labels":
                        for label, count in value.items():
                            target[f"label_{label}"] += count
                    elif key == "words":
                        target["words_total"] += sum(value)
                    else:
                        target[key] += value
        for (model, grp, role), cell in merged.items():
            out[model][grp].setdefault(name, {})[role] = dict(cell)
    for profile, name in ((influence_profile, "influence"), (listening_profile, "listening")):
        merged = defaultdict(lambda: defaultdict(float))
        for g in keep:
            for (model, grp), cell in profile([g]).items():
                for key, value in cell.items():
                    merged[(short_model(model), group(g))][key] += value
        for (model, grp), cell in merged.items():
            out[model][grp][name] = dict(cell)
    return {model: dict(groups) for model, groups in out.items()}


def parse_table(text: str, header: str) -> list[list[str]]:
    """Reads the rows of the Markdown table whose header line starts with a given text.

    Args:
        text: The Markdown.
        header: Start of the header line, such as `| Requests at once | Granite`.

    Returns:
        The rows below the header, each a list of cells, without the separator line.
    """
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def numbers(cell: str) -> list[float]:
    """Lists the numbers in a table cell, reading `1,409` as 1409."""
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", cell.replace(",", ""))]


def collect_doc(doc: Path) -> dict:
    """Reads the measurement tables of the doc.

    Args:
        doc: The Markdown doc.

    Returns:
        The concurrency tables of the two GPUs, the reasoning benchmark and the audio overlap table.
    """
    text = doc.read_text(encoding="utf-8")

    def concurrency(header: str) -> dict:
        rows = parse_table(text, header)
        names = [c.removesuffix(" round time") for c in header.strip("| ").split(" | ")[1:]]
        series = {n: {"at once": {}, "one after another": None} for n in names}
        for row in rows:
            for name, cell in zip(names, row[1:]):
                if "one after another" in row[0]:
                    series[name]["one after another"] = numbers(cell)[0]
                else:
                    series[name]["at once"][int(numbers(row[0])[0])] = numbers(cell)[0]
        return series

    reasoning = []
    for row in parse_table(text, "| Model | Effort | Mean tokens per reply"):
        mean, share, meeting = numbers(row[2]), numbers(row[5]), numbers(row[8])
        reasoning.append(
            {"model": row[0], "effort": row[1], "mean": [mean[0], mean[-1]], "share": [share[0], share[-1]], "meeting_s": meeting[0]}
        )
    audio = [
        {"speakers": int(numbers(r[0])[0]), "overlap": numbers(r[1])[0], "recall": numbers(r[2])[0], "precision": numbers(r[3])[0], "recovered": numbers(r[4])[0]}
        for r in parse_table(text, "| Speakers at once")
    ]
    return {
        "concurrency_l4": concurrency("| Requests at once | Granite 3.3 8B round time | Llama 3.1 8B round time |"),
        "concurrency_rtx": concurrency("| Requests at once | gpt-oss-20b round time | Gemma 4 31B round time |"),
        "reasoning": reasoning,
        "audio": audio,
    }


def build_data(p2_files: Sequence[Path], log_folders: Sequence[Path], doc: Path) -> dict:
    """Collects every number the figures need.

    Args:
        p2_files: P2 result files.
        log_folders: Folders of P2 game folders.
        doc: The doc with the measurement tables.

    Returns:
        The data of the figures, which `json.dump` can write.
    """
    return {"p2": collect_p2(p2_files), "text": collect_text(log_folders), **collect_doc(doc)}


def rate(count: float, total: float) -> float | None:
    """Gives count over total, or None when the total is 0."""
    return count / total if total else None


def pct(value: float | None, digits: int = 0) -> str:
    """Formats a share as a percentage."""
    return "-" if value is None else f"{value * 100:.{digits}f}%"


def setup_style(plt) -> None:
    """Sets the look every figure shares: a quiet surface, hairline grid and ink-colored text."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
            "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
            "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
            "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
            "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9, "hatch.linewidth": 1.0,
            "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        }
    )


def tint(color: str, amount: float = 0.35) -> str:
    """Mixes a color with the surface, which gives the light fill under a hatch."""
    c = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(s[i] + (c[i] - s[i]) * amount):02x}" for i in range(3))


def bar(ax, x, height, width, color, hatched=False, horizontal=False, **kwargs):
    """Draws a bar in a model's color, solid or hatched at 45 degrees with the same hue."""
    style = {"facecolor": tint(color), "edgecolor": color, "hatch": "////", "linewidth": 0} if hatched else {"facecolor": color, "edgecolor": SURFACE, "linewidth": 0.6}
    if horizontal:
        return ax.barh(x, height, width, **style, **kwargs)
    return ax.bar(x, height, width, **style, **kwargs)


def kind_legend(ax, solid: str, hatched: str, loc="upper right", **kwargs):
    """Adds a legend for the two textures, in neutral gray because the hue belongs to the model."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=NEUTRAL, edgecolor=SURFACE, label=solid),
        Patch(facecolor=tint(NEUTRAL), edgecolor=NEUTRAL, hatch="////", linewidth=0, label=hatched),
    ]
    return ax.legend(handles=handles, loc=loc, **kwargs)


def model_legend(ax, loc="upper right", **kwargs):
    """Adds a legend of the three models, each with its own color."""
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor=SURFACE, label=m) for m in MODELS]
    return ax.legend(handles=handles, loc=loc, **kwargs)


def label_tips(ax, rects, texts, vertical=True, pad=3):
    """Prints a value at the tip of each bar."""
    for rect, text in zip(rects, texts):
        if text is None:
            continue
        if vertical:
            ax.annotate(text, (rect.get_x() + rect.get_width() / 2, rect.get_height()), xytext=(0, pad), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
        else:
            ax.annotate(text, (rect.get_width(), rect.get_y() + rect.get_height() / 2), xytext=(pad, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK2)


def percent_axis(ax, axis="y", top=1.0):
    """Formats an axis in percent from 0."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    formatter = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
    if axis == "y":
        ax.set_ylim(0, top)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(MultipleLocator(0.2 if top > 0.5 else 0.1))
    else:
        ax.set_xlim(0, top)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MultipleLocator(0.2 if top > 0.5 else 0.1))


def fig_win_rate(plt, data, out):
    """Honest win rate by setting and model, with 95% Wilson intervals."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    width = 0.24
    for j, model in enumerate(MODELS):
        for i, setting in enumerate(SETTINGS):
            cell = data["p2"][model]["settings"][setting]
            value = cell["wins"] / cell["games"]
            x = i + (j - 1) * (width + 0.03)
            rect = bar(ax, x, value, width, MODEL_COLORS[model])
            ax.errorbar(x, value, yerr=[[value - cell["ci_low"]], [cell["ci_high"] - value]], color=INK2, linewidth=1, capsize=2.5, capthick=1)
            ax.annotate(pct(value), (x, cell["ci_high"]), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), SETTINGS)
    percent_axis(ax, "y", 1.05)
    ax.set_ylabel("Honest win rate")
    model_legend(ax, loc="upper left", ncol=3)
    games = "; ".join(f"{m}: " + ", ".join(str(data["p2"][m]["settings"][s]["games"]) for s in SETTINGS) for m in MODELS)
    fig.text(0.01, 0.01, f"Games per bar (no discussion, turns, timed room). {games}.\nWhiskers are 95% Wilson intervals.", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "p2_win_rate.png", dpi=200)


def fig_granite_cells(plt, data, out):
    """Granite honest win rate in each of the 11 protocols."""
    cells = data["p2"]["Granite 3.3 8B"]["protocols"]
    order = ["sequential"] + [f"{o}-{b}" for o in ("clean", "interleaved", "noisy") for b in (60, 180, 300)] + ["none"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, name in enumerate(order):
        cell = cells[name]
        value = cell["wins"] / cell["games"]
        y = len(order) - 1 - i
        ax.plot([cell["ci_low"], cell["ci_high"]], [y, y], color=INK2, linewidth=1.2, solid_capstyle="round")
        ax.plot(value, y, "o", color=MODEL_COLORS["Granite 3.3 8B"], markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(pct(value), (cell["ci_high"], y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(order)), order[::-1])
    percent_axis(ax, "x", 1.0)
    ax.set_xlabel("Honest win rate with a 95% Wilson interval (20 games per protocol)")
    ax.grid(axis="x", visible=True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "p2_granite_protocols.png", dpi=200)


def fig_votes(plt, data, out):
    """Accuracy of cast honest votes, and share of honest ballots skipped, by setting and model."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    width = 0.24
    for j, model in enumerate(MODELS):
        for i, setting in enumerate(SETTINGS):
            cell = data["p2"][model]["settings"][setting]
            cast = cell["correct"] + cell["incorrect"]
            x = i + (j - 1) * (width + 0.03)
            for ax, value in ((axes[0], rate(cell["correct"], cast)), (axes[1], rate(cell["skipped"], cast + cell["skipped"]))):
                rect = bar(ax, x, value or 0, width, MODEL_COLORS[model])
                label = None if value is None else pct(value)
                if ax is axes[0] and cast < 10:
                    label = f"{label} (n={cast})"
                label_tips(ax, rect, [label])
    for ax, title in zip(axes, ("Honest votes that went to a Byzantine", "Honest ballots that were skips")):
        ax.set_xticks(range(3), SETTINGS)
        ax.set_title(title)
        percent_axis(ax, "y", 1.12)
    handles = model_legend(axes[0], loc="upper left").legend_handles
    axes[0].get_legend().remove()
    fig.legend(handles=handles, labels=MODELS, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "p2_votes.png", dpi=200)


def fig_speech(plt, data, out):
    """Speech acts of honest and Byzantine agents in the clean room, one panel for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), sharex=True)
    height = 0.34
    for ax, model in zip(axes, MODELS):
        speech = data["text"][model]["clean"]["speech"]
        for k, (role, hatched) in enumerate((("honest", False), ("byzantine", True))):
            cell = speech[role]
            for i, (key, _) in enumerate(ACTS):
                value = cell.get(f"label_{key}", 0) / cell["n"]
                y = len(ACTS) - 1 - i + (0.19 if k == 0 else -0.19)
                rect = bar(ax, y, value, height, MODEL_COLORS[model], hatched=hatched, horizontal=True)
                label_tips(ax, rect, [pct(value)], vertical=False)
        ax.set_yticks(range(len(ACTS)), [label for _, label in ACTS][::-1])
        percent_axis(ax, "x", 1.0)
        ax.set_title(model)
        ax.grid(axis="x", visible=True)
        ax.grid(axis="y", visible=False)
    handles = kind_legend(axes[2], "Honest agents", "Byzantine agents", loc="lower right").legend_handles
    axes[2].get_legend().remove()
    fig.legend(handles=handles, labels=["Honest agents", "Byzantine agents"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Share of the role's statements with each speech act (a statement can have several)", fontsize=8.5, color=INK2, y=0.07)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "text_speech_acts.png", dpi=200)


def fig_length(plt, data, out):
    """Words per statement in the room and in turns, and the share cut at the 40-word limit."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    width = 0.34
    for j, model in enumerate(MODELS):
        for k, (group, hatched) in enumerate((("clean", False), ("sequential", True))):
            speech = data["text"][model][group]["speech"]
            words = (speech["honest"]["words_total"] + speech["byzantine"]["words_total"]) / (speech["honest"]["n"] + speech["byzantine"]["n"])
            x = j + (-0.19 if k == 0 else 0.19)
            rect = bar(axes[0], x, words, width, MODEL_COLORS[model], hatched=hatched)
            label_tips(axes[0], rect, [f"{words:.0f}"])
        speech = data["text"][model]["clean"]["speech"]
        for k, (role, hatched) in enumerate((("honest", False), ("byzantine", True))):
            cell = speech[role]
            value = cell["cut"] / cell["timed"]
            rect = bar(axes[1], j + (-0.19 if k == 0 else 0.19), value, width, MODEL_COLORS[model], hatched=hatched)
            label_tips(axes[1], rect, [pct(value)])
    for ax in axes:
        ax.set_xticks(range(3), MODELS)
    axes[0].set_title("Words per statement")
    axes[0].set_ylim(0, 110)
    kind_legend(axes[0], "Clean room", "Turns", loc="upper right")
    axes[1].set_title("Room statements cut at the 40-word limit")
    percent_axis(axes[1], "y", 0.2)
    kind_legend(axes[1], "Honest agents", "Byzantine agents", loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "text_length.png", dpi=200)


def fig_claims(plt, data, out):
    """Claims about places that never held, by role and model, in the room and in turns."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    width = 0.34
    for ax, group, title in zip(axes, ("clean", "sequential"), ("Clean room", "Turns")):
        for j, model in enumerate(MODELS):
            claims = data["text"][model][group]["claims"]
            for k, (role, hatched) in enumerate((("honest", False), ("byzantine", True))):
                cell = claims.get(role, {})
                total = cell.get("claims", 0)
                value = cell.get("false", 0) / total if total else 0
                rect = bar(ax, j + (-0.19 if k == 0 else 0.19), value, width, MODEL_COLORS[model], hatched=hatched)
                label_tips(ax, rect, [pct(value) + (f"\nn={int(total)}" if total < 100 else "")])
        ax.set_xticks(range(3), MODELS)
        ax.set_title(title)
        percent_axis(ax, "y", 0.6)
    kind_legend(axes[0], "Honest agents", "Byzantine agents", loc="upper left")
    axes[0].set_ylabel("Claims that never held")
    fig.tight_layout()
    fig.savefig(out / "text_false_claims.png", dpi=200)


def fig_accusations(plt, data, out):
    """Share of honest accusations aimed at a Byzantine, against what a random pick would give."""
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    width = 0.34
    for j, model in enumerate(MODELS):
        for k, (group, hatched) in enumerate((("clean", False), ("sequential", True))):
            cell = data["text"][model][group]["accusations"]["honest"]
            value, chance = cell["at_byz"] / cell["accuse"], cell["chance_byz"] / cell["accuse"]
            x = j + (-0.19 if k == 0 else 0.19)
            bar(ax, x, value, width, MODEL_COLORS[model], hatched=hatched)
            ax.plot([x - width / 2, x + width / 2], [chance, chance], color=INK, linewidth=2, solid_capstyle="butt")
            ax.annotate(pct(value), (x, max(value, chance)), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(3), MODELS)
    percent_axis(ax, "y", 0.6)
    ax.set_ylabel("Honest accusations aimed at a Byzantine")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=NEUTRAL, edgecolor=SURFACE, label="Clean room"),
        Patch(facecolor=tint(NEUTRAL), edgecolor=NEUTRAL, hatch="////", linewidth=0, label="Turns"),
        Line2D([0], [0], color=INK, linewidth=2, label="A random pick"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "text_accusation_accuracy.png", dpi=200)


def fig_influence(plt, data, out):
    """Honest votes for an honest candidate by who accused them, and votes for the most accused agent."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={"width_ratios": [1.7, 1]})
    width = 0.24
    for j, model in enumerate(MODELS):
        influence = data["text"][model]["clean"]["influence"]
        for i, who in enumerate(WHO):
            pairs, votes = influence.get(f"honest|{who}|pairs", 0), influence.get(f"honest|{who}|votes", 0)
            value = votes / pairs if pairs else 0
            x = i + (j - 1) * (width + 0.03)
            rect = bar(axes[0], x, value, width, MODEL_COLORS[model])
            label_tips(axes[0], rect, [f"{pct(value)}" if pairs >= 30 else f"{pct(value)}*"])
        for k, group in enumerate(("clean", "sequential")):
            cell = data["text"][model][group]["influence"]
            value = cell["plurality_votes"] / cell["plurality_meetings_votes"]
            rect = bar(axes[1], j + (-0.19 if k == 0 else 0.19), value, 0.34, MODEL_COLORS[model], hatched=bool(k))
            label_tips(axes[1], rect, [pct(value)])
    axes[0].set_xticks(range(4), ["nobody", "honest\nagents only", "Byzantine\nagents only", "both"])
    axes[0].set_xlabel("Who accused the honest candidate in the meeting")
    axes[0].set_ylabel("Voter and candidate pairs\nthat ended in a vote")
    percent_axis(axes[0], "y", 0.8)
    model_legend(axes[0], loc="upper left", fontsize=7.5)
    axes[1].set_xticks(range(3), MODELS, fontsize=7.5)
    axes[1].set_title("Votes for the most accused agent")
    percent_axis(axes[1], "y", 1.0)
    handles = kind_legend(axes[1], "Clean room", "Turns", loc="lower right").legend_handles
    axes[1].get_legend().remove()
    fig.legend(handles=handles, labels=["Clean room", "Turns"], loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.0))
    fig.text(0.01, 0.01, "* fewer than 30 pairs", fontsize=7, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "text_influence.png", dpi=200)


def fig_listening(plt, data, out):
    """How often the accused agent speaks again, and how often its answer names the accuser."""
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    width = 0.34
    for j, model in enumerate(MODELS):
        for k, (group, hatched) in enumerate((("clean", False), ("sequential", True))):
            c = data["text"][model][group]["listening"]
            values = (c["answerable"] / c["accused"], c["answered"] / c["answerable"])
            for ax, value in zip(axes, values):
                rect = bar(ax, j + (-0.19 if k == 0 else 0.19), value, width, MODEL_COLORS[model], hatched=hatched)
                label_tips(ax, rect, [pct(value)])
    for ax, title in zip(axes, ("The accused agent speaks again in the meeting", "That statement names the accuser")):
        ax.set_xticks(range(3), MODELS)
        ax.set_title(title)
        percent_axis(ax, "y", 1.1)
    handles = kind_legend(axes[0], "Clean room", "Turns", loc="lower left").legend_handles
    axes[0].get_legend().remove()
    fig.legend(handles=handles, labels=["Clean room", "Turns"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    axes[0].set_ylabel("Share of accusations")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "text_listening.png", dpi=200)


def fig_concurrency(plt, data, out):
    """Round time against requests at once, on the two GPUs."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    colors = {"Granite 3.3 8B": MODEL_COLORS["Granite 3.3 8B"], "Llama 3.1 8B": NEUTRAL, "gpt-oss-20b": MODEL_COLORS["gpt-oss-120b"], "Gemma 4 31B": MODEL_COLORS["Gemma 4 31B"]}
    for ax, key, title in zip(axes, ("concurrency_l4", "concurrency_rtx"), ("NVIDIA L4, 24 GB", "NVIDIA RTX PRO 6000, 96 GB")):
        for name, series in data[key].items():
            xs = sorted(int(k) for k in series["at once"])
            ys = [series["at once"][str(x)] for x in xs]
            ax.plot(xs, ys, "-o", color=colors[name], linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ax.plot([10], [series["one after another"]], "D", color=colors[name], markerfacecolor=SURFACE, markersize=7, markeredgewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Requests at once")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_ylabel("Round time (s)")
        ax.set_ylim(bottom=0)
    from matplotlib.lines import Line2D

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    handles.append(Line2D([0], [0], marker="D", color=INK2, markerfacecolor=SURFACE, linestyle="", markersize=7, markeredgewidth=1.5))
    labels.append("10 requests one after another (hollow diamond)")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "bench_concurrency.png", dpi=200)


def fig_reasoning(plt, data, out):
    """Tokens per reply, reasoning share and meeting time of gpt-oss at each reasoning effort."""
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    efforts = ["low", "medium", "high"]
    base = MODEL_COLORS["gpt-oss-120b"]
    shade = {"gpt-oss-20b": tint(base, 0.55), "gpt-oss-120b": base}
    width = 0.3
    for k, model in enumerate(("gpt-oss-20b", "gpt-oss-120b")):
        rows = {r["effort"]: r for r in data["reasoning"] if r["model"] == model}
        for i, effort in enumerate(efforts):
            r = rows[effort]
            x = i + (-0.17 if k == 0 else 0.17)
            lo, hi = r["mean"]
            axes[0].bar(x, hi - lo if hi > lo else 30, width, bottom=lo, color=shade[model], edgecolor=SURFACE, linewidth=0.6)
            axes[0].annotate(f"{lo:,.0f}-{hi:,.0f}", (x, hi), xytext=(0, 3), textcoords="offset points", ha="right" if k == 0 else "left", fontsize=7, color=INK2)
            lo, hi = r["share"]
            axes[1].bar(x, (hi - lo) / 100 if hi > lo else 0.01, width, bottom=lo / 100, color=shade[model], edgecolor=SURFACE, linewidth=0.6)
            axes[1].annotate(f"{lo:.0f}-{hi:.0f}%" if hi > lo else f"{lo:.0f}%", (x, hi / 100), xytext=(0, 3), textcoords="offset points", ha="right" if k == 0 else "left", fontsize=7, color=INK2)
            rect = axes[2].bar(x, r["meeting_s"], width, color=shade[model], edgecolor=SURFACE, linewidth=0.6)
            label_tips(axes[2], rect, [f"{r['meeting_s']:.0f} s"], pad=2)
    axes[0].axhline(4096, color=INK2, linewidth=1)
    axes[0].annotate("limit of 4,096 tokens", (-0.45, 4096), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    axes[0].set_title("Mean tokens per reply (range over load)")
    axes[0].set_ylim(0, 4800)
    axes[1].set_title("Share of tokens spent reasoning")
    percent_axis(axes[1], "y", 1.15)
    axes[2].set_title("Time of a 6-tick meeting")
    axes[2].set_ylim(0, 400)
    for ax in axes:
        ax.set_xticks(range(3), efforts)
        ax.set_xlabel("Reasoning effort")
    from matplotlib.patches import Patch

    axes[2].legend(handles=[Patch(facecolor=shade[m], label=m) for m in ("gpt-oss-20b", "gpt-oss-120b")], loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "bench_reasoning.png", dpi=200)


def fig_audio(plt, data, out):
    """Word recall, precision and speakers recovered against the number of speakers at once."""
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    xs = [r["speakers"] for r in data["audio"]]
    # Violet, red and green of the palette, which no model uses, each with its own marker.
    series = (("Word recall", "recall", "#6250d6", "o"), ("Precision", "precision", "#e34948", "s"), ("Speakers recovered", "recovered", "#008300", "^"))
    for label, key, color, marker in series:
        ys = [r[key] / 100 for r in data["audio"]]
        ax.plot(xs, ys, "-", marker=marker, color=color, linewidth=2, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{label} {pct(ys[-1])}", (xs[-1], ys[-1]), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xlabel("Speakers at once")
    percent_axis(ax, "y", 1.05)
    ax.set_xlim(0.6, 15.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "audio_overlap.png", dpi=200)


FIGURES = (fig_concurrency, fig_reasoning, fig_audio, fig_win_rate, fig_granite_cells, fig_votes, fig_speech, fig_length, fig_claims, fig_accusations, fig_influence, fig_listening)


def plot_all(data: dict, out: Path) -> list[Path]:
    """Draws every figure.

    Args:
        data: The data written by `build_data`.
        out: Folder for the PNG files.

    Returns:
        The files written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The data may come straight from build_data, whose dictionaries have numbers as keys, and a file has text.
    data = json.loads(json.dumps(data))
    setup_style(plt)
    out.mkdir(parents=True, exist_ok=True)
    for draw in FIGURES:
        draw(plt, data, out)
        plt.close("all")
    return sorted(out.glob("*.png"))


def main(argv: Sequence[str] | None = None) -> None:
    """Collects the data or draws the figures.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("data", help="collect the numbers into a JSON file")
    collect.add_argument("--p2", nargs="+", type=Path, required=True)
    collect.add_argument("--logs", nargs="+", type=Path, required=True, help="folders of game folders")
    collect.add_argument("--doc", type=Path, default=Path("docs/situation-room.md"))
    collect.add_argument("--out", type=Path, default=Path("docs/figures/data.json"))
    draw = sub.add_parser("plot", help="draw the figures from the JSON file")
    draw.add_argument("--data", type=Path, default=Path("docs/figures/data.json"))
    draw.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)
    if args.command == "data":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(build_data(args.p2, args.logs, args.doc), indent=1, sort_keys=True), encoding="utf-8")
    else:
        for path in plot_all(json.loads(args.data.read_text(encoding="utf-8")), args.out):
            print(path)


if __name__ == "__main__":
    main()
