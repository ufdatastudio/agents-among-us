"""Renders a recorded game as a video on the tile map of the Godot world.

Agents walk the corridors to their rooms at each tick, tags leave bodies, a report calls everyone to the table of the
Cafeteria, the statements of the meeting appear as they were spoken, and the votes and the ejection follow. The viewer
sees the roles, which the agents never saw. The walks come from `scripts/world_replay.py`, and nothing here changes a game.

Usage:
    uv run --with matplotlib python -m scripts.make_world_replay logs/p2_x/Game_tag_interleaved-180_3_Run0 --out docs/media
    uv run --with matplotlib python -m scripts.make_world_replay GAME --still 12.5 --still 60 --out /path/for/pictures
    uv run --with matplotlib python -m scripts.make_world_replay GAME --gif-start 0 --gif-seconds 14 --out docs/media

The video needs ffmpeg on the path.
"""

import argparse
import math
import multiprocessing
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import matplotlib
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from core.world_map import HEIGHT, WIDTH, WorldMap, build_map
from scripts.analyze_p2_text import short_model
from scripts.draw_world_map import draw_doors, draw_tiles
from scripts.make_replay import render_gif, wrap
from scripts.world_replay import MEETING_ROOM, Frame, Pacing, Replay, Timeline, build_timeline, load_replay

FINAL_W, FINAL_H = 1600, 900
S = 2
TILE = 15
MAP_W, MAP_H = WIDTH * TILE, HEIGHT * TILE
PANEL_X = MAP_W

BG = (15, 18, 24)
PANEL = (22, 26, 34)
STRIP = (19, 23, 30)
RULE = (46, 53, 66)
INK = (236, 239, 244)
MUTED = (150, 158, 172)
DIM = (96, 104, 118)
BYZANTINE = (255, 92, 77)
ACCENT = (96, 165, 255)
GOOD = (104, 206, 146)
TABLE_FILL = (30, 36, 47)
TABLE_EDGE = (110, 124, 150)
AGENT_COLORS = (
    (91, 155, 213), (243, 156, 52), (95, 195, 107), (176, 122, 161), (79, 195, 195),
    (230, 211, 74), (245, 141, 182), (184, 134, 90), (166, 216, 84), (141, 160, 203),
)
TOKEN_R = 7.5


def agent_number(name: str) -> int:
    """Returns the number of an agent name such as `Agent_7`."""
    return int(name.rsplit("_", 1)[1])


def agent_color(name: str) -> tuple[int, int, int]:
    """Returns the color of an agent, which is fixed by its number."""
    return AGENT_COLORS[agent_number(name) % len(AGENT_COLORS)]


def dim(color: tuple[int, int, int], keep: float = 0.45) -> tuple[int, int, int]:
    """Mixes a color toward the background."""
    return tuple(int(c * keep + b * (1 - keep)) for c, b in zip(color, BG))


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Loads the DejaVu fonts that ship with matplotlib, at the supersampled sizes of the frames."""
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    regular, bold = str(base / "DejaVuSans.ttf"), str(base / "DejaVuSans-Bold.ttf")
    sizes = {"room": 10, "token": 10, "heading": 22, "title": 20, "name": 15, "body": 14, "small": 12, "clock": 16, "big": 28}
    heavy = {"room", "token", "heading", "title", "name", "clock", "big"}
    return {name: ImageFont.truetype(bold if name in heavy else regular, size * S) for name, size in sizes.items()}


def circle(draw: ImageDraw.ImageDraw, c: tuple[float, float], r: float, fill=None, outline=None, width: int = 0) -> None:
    """Draws a circle by its center."""
    draw.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=fill, outline=outline, width=width)


def clock(seconds: float) -> str:
    """Writes seconds as `m:ss`."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class Renderer:
    """Draws the frames of a game."""

    def __init__(self, replay: Replay, timeline: Timeline, world: WorldMap) -> None:
        """Draws the map once, to copy for every frame.

        Args:
            replay: The game.
            timeline: Its plan.
            world: The map.
        """
        self.replay, self.timeline, self.world = replay, timeline, world
        self.fonts = load_fonts()
        self.base = self._base()
        self.title = f"{short_model(replay.model)}"
        self.subtitle = f"{replay.protocol}, seed {replay.seed}"

    def xy(self, point: tuple[float, float]) -> tuple[float, float]:
        """Converts a position in tiles to pixels of the supersampled frame."""
        return point[0] * TILE * S, point[1] * TILE * S

    def _base(self) -> Image.Image:
        """Draws the map, the panel and the strip that never change."""
        image = Image.new("RGB", (FINAL_W * S, FINAL_H * S), BG)
        draw = ImageDraw.Draw(image)
        scale = TILE * S
        draw_tiles(draw, self.world, scale)
        draw_doors(draw, self.world, scale)
        for room in self.world.rooms.values():
            cx, cy = room.center()
            name = re.sub(r"(?<=[a-z])(?=Engine)", " ", room.name)
            draw.text((cx * scale + 1, cy * scale + 2), name, font=self.fonts["room"], fill=(14, 17, 23), anchor="mm")
            draw.text((cx * scale, cy * scale), name, font=self.fonts["room"], fill=(214, 220, 232), anchor="mm")
        draw.rectangle((0, MAP_H * S, MAP_W * S, FINAL_H * S), fill=STRIP)
        draw.rectangle((PANEL_X * S, 0, FINAL_W * S, FINAL_H * S), fill=PANEL)
        draw.line((0, MAP_H * S, MAP_W * S, MAP_H * S), fill=RULE, width=2 * S)
        draw.line((PANEL_X * S, 0, PANEL_X * S, FINAL_H * S), fill=RULE, width=2 * S)
        return image

    def frame(self, t: float) -> Image.Image:
        """Draws the frame at a video time.

        Args:
            t: Video time in seconds.

        Returns:
            The frame at the final size.
        """
        state = self.timeline.at(t)
        image = self.base.copy()
        draw = ImageDraw.Draw(image)
        centers = {tok.name: self.xy((tok.x, tok.y)) for tok in state.tokens}
        if state.table is not None:
            self._table(draw, state)
        self._bodies(draw, state)
        if state.banner is not None and state.banner.focus is not None:
            self._focus(draw, state)
        self._arrows(draw, state, centers)
        for tok in state.tokens:
            self._token(draw, tok, t)
        self._strikes(draw, state)
        self._panel(draw, state)
        self._strip(draw, state)
        return image.reduce(S)

    def _token(self, draw: ImageDraw.ImageDraw, tok, t: float) -> None:
        """Draws an agent as a numbered disc, with a red ring for a Byzantine agent."""
        c = self.xy((tok.x, tok.y))
        r = TOKEN_R * S * tok.scale
        if r < 1:
            return
        if tok.speaking:
            circle(draw, c, r + (6 + 2 * math.sin(t * 12)) * S, outline=INK, width=2 * S)
        if self.replay.roles[tok.name] == "byzantine":
            circle(draw, c, r + 2.5 * S, fill=BYZANTINE)
        circle(draw, c, r, fill=agent_color(tok.name), outline=(12, 14, 20), width=S)
        if tok.scale > 0.5:
            draw.text(c, str(agent_number(tok.name)), font=self.fonts["token"], fill=(12, 14, 20), anchor="mm")

    def _bodies(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws each body as a flat disc with a cross."""
        for name, x, y in state.bodies:
            cx, cy = self.xy((x, y))
            rx, ry = TOKEN_R * S * 1.25, TOKEN_R * S * 0.7
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=dim(agent_color(name)), outline=(200, 205, 215), width=S)
            d = 3.2 * S
            draw.line((cx - d, cy - d, cx + d, cy + d), fill=INK, width=2 * S)
            draw.line((cx - d, cy + d, cx + d, cy - d), fill=INK, width=2 * S)

    def _focus(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws a pulsing ring at the place a banner is about."""
        c = self.xy(state.banner.focus)
        phase = (state.t * 1.4) % 1.0
        circle(draw, c, (14 + 26 * phase) * S, outline=ACCENT, width=3 * S)
        circle(draw, c, 12 * S, outline=ACCENT, width=2 * S)

    def _strikes(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws a tag as a red line from the killer to the victim that ends in a flash."""
        for killer, victim, progress in state.strikes:
            a, b = self.xy(killer), self.xy(victim)
            head = min(1.0, progress * 2.0)
            end = (a[0] + (b[0] - a[0]) * head, a[1] + (b[1] - a[1]) * head)
            draw.line((a, end), fill=BYZANTINE, width=4 * S)
            if progress > 0.5:
                circle(draw, b, (6 + 22 * (progress - 0.5) * 2) * S, outline=BYZANTINE, width=3 * S)

    def _table(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws the table of the meeting with the clock of the room."""
        cx, cy = self.xy(self.world.rooms[MEETING_ROOM].center())
        circle(draw, (cx, cy), 3.6 * TILE * S, fill=TABLE_FILL, outline=TABLE_EDGE, width=3 * S)
        if state.phase is None or state.phase.kind not in ("gather", "meeting"):
            return
        elapsed, length = state.table
        number = state.phase.round_num
        draw.text((cx, cy - 24 * S), f"Round {number}", font=self.fonts["small"], fill=MUTED, anchor="mm")
        draw.text((cx, cy + 1 * S), clock(elapsed), font=self.fonts["big"], fill=INK, anchor="mm")
        draw.text((cx, cy + 26 * S), f"of {clock(length)}", font=self.fonts["small"], fill=MUTED, anchor="mm")

    def _arrows(self, draw: ImageDraw.ImageDraw, state: Frame, centers: dict[str, tuple[float, float]]) -> None:
        """Draws each vote that is shown as an arrow from the voter to its choice, and a mark for a skip."""
        for voter, target in state.arrows:
            a = centers.get(voter)
            if a is None:
                continue
            color = agent_color(voter)
            b = centers.get(target)
            if b is None:
                if target.upper() == "SKIP":
                    draw.text((a[0], a[1] - 17 * S), "skip", font=self.fonts["small"], fill=MUTED, anchor="mm")
                continue
            length = math.dist(a, b)
            if length < 1:
                continue
            ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
            pad = (TOKEN_R + 4) * S
            start, end = (a[0] + ux * pad, a[1] + uy * pad), (b[0] - ux * pad, b[1] - uy * pad)
            draw.line((start, end), fill=color, width=2 * S)
            wing = 6 * S
            left = (end[0] - ux * wing - uy * wing * 0.6, end[1] - uy * wing + ux * wing * 0.6)
            right = (end[0] - ux * wing + uy * wing * 0.6, end[1] - uy * wing - ux * wing * 0.6)
            draw.polygon((end, left, right), fill=color)

    def _panel(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws the roster and the counts on the right."""
        x = (PANEL_X + 24) * S
        draw.text((x, 22 * S), self.title, font=self.fonts["title"], fill=INK, anchor="la")
        draw.text((x, 50 * S), self.subtitle, font=self.fonts["body"], fill=MUTED, anchor="la")
        heading = state.phase.label if state.phase else ""
        draw.text((x, 92 * S), heading, font=self.fonts["heading"], fill=INK, anchor="la")
        top = 140
        for k, name in enumerate(sorted(self.replay.roles, key=agent_number)):
            y = (top + k * 56) * S
            kind, round_num = state.status[name]
            role = self.replay.roles[name]
            alive = kind == "alive"
            c = (x + 16 * S, y + 22 * S)
            if alive:
                circle(draw, c, 14 * S, fill=agent_color(name), outline=BYZANTINE if role == "byzantine" else None, width=3 * S)
                draw.text(c, str(agent_number(name)), font=self.fonts["token"], fill=(12, 14, 20), anchor="mm")
            else:
                circle(draw, c, 14 * S, outline=DIM, width=2 * S)
                draw.text(c, str(agent_number(name)), font=self.fonts["token"], fill=DIM, anchor="mm")
            ink = INK if alive else DIM
            draw.text((x + 44 * S, y + 6 * S), name, font=self.fonts["name"], fill=ink, anchor="la")
            note = "in play" if alive else f"{'tagged' if kind == 'killed' else 'ejected'} in round {round_num}"
            draw.text((x + 44 * S, y + 28 * S), note, font=self.fonts["small"], fill=MUTED if alive else DIM, anchor="la")
            label, color = ("Byzantine", BYZANTINE) if role == "byzantine" else ("Honest", MUTED)
            draw.text(((FINAL_W - 24) * S, y + 22 * S), label, font=self.fonts["small"], fill=color if alive else DIM, anchor="rm")
        honest = sum(1 for n, (k, _) in state.status.items() if k == "alive" and self.replay.roles[n] == "honest")
        byz = sum(1 for n, (k, _) in state.status.items() if k == "alive" and self.replay.roles[n] == "byzantine")
        draw.line((x, 712 * S, (FINAL_W - 24) * S, 712 * S), fill=RULE, width=S)
        draw.text((x, 724 * S), f"In play: {honest} honest, {byz} Byzantine", font=self.fonts["name"], fill=INK, anchor="la")
        if state.tally:
            parts = [f"{('skip' if k.upper() == 'SKIP' else k)} {v}" for k, v in sorted(state.tally.items(), key=lambda kv: (-kv[1], kv[0]))]
            lines = wrap("Votes: " + ", ".join(parts), self.fonts["small"], (FINAL_W - 24 - PANEL_X - 24) * S)
            for i, line in enumerate(lines[:3]):
                draw.text((x, (752 + i * 18) * S), line, font=self.fonts["small"], fill=MUTED, anchor="la")
        note = "Roles are shown to the viewer. The agents never saw them."
        draw.text((x, (FINAL_H - 22) * S), note, font=self.fonts["small"], fill=DIM, anchor="ls")

    def _strip(self, draw: ImageDraw.ImageDraw, state: Frame) -> None:
        """Draws the banner or the statements under the map."""
        x = 24 * S
        top = MAP_H + 12
        if state.banner is not None and not state.captions:
            draw.text((x, (top + 28) * S), state.banner.title, font=self.fonts["heading"], fill=INK, anchor="la")
            if state.banner.detail:
                draw.text((x, (top + 66) * S), state.banner.detail, font=self.fonts["body"], fill=MUTED, anchor="la")
            return
        if state.captions:
            y = top
            shown = state.captions[-3:]
            for speaker, text, finished in shown:
                self._caption(draw, speaker, text, finished, y)
                y += 68
            if state.banner is not None:
                draw.text(((MAP_W - 24) * S, (top + 4) * S), state.banner.title, font=self.fonts["name"], fill=INK, anchor="ra")
            return
        draw.text((x, (top + 4) * S), "What has happened", font=self.fonts["small"], fill=DIM, anchor="la")
        for i, (round_num, text) in enumerate(state.events[-6:]):
            draw.text((x, (top + 30 + i * 26) * S), f"Round {round_num}", font=self.fonts["small"], fill=MUTED, anchor="la")
            draw.text((x + 80 * S, (top + 28 + i * 26) * S), text, font=self.fonts["body"], fill=INK, anchor="la")

    def _caption(self, draw: ImageDraw.ImageDraw, speaker: str, text: str, finished: bool, y: float) -> None:
        """Draws one statement with its speaker."""
        c = (40 * S, (y + 22) * S)
        role = self.replay.roles[speaker]
        circle(draw, c, 14 * S, fill=agent_color(speaker), outline=BYZANTINE if role == "byzantine" else None, width=3 * S)
        draw.text(c, str(agent_number(speaker)), font=self.fonts["token"], fill=(12, 14, 20), anchor="mm")
        left = 72 * S
        draw.text((left, y * S), speaker, font=self.fonts["name"], fill=INK if not finished else MUTED, anchor="la")
        if role == "byzantine":
            draw.text((left + 100 * S, (y + 2) * S), "Byzantine", font=self.fonts["small"], fill=BYZANTINE, anchor="la")
        lines = wrap(text, self.fonts["body"], (MAP_W - 24 - 72) * S)[-2:]
        for i, line in enumerate(lines):
            draw.text((left, (y + 22 + i * 18) * S), line, font=self.fonts["body"], fill=INK if not finished else MUTED, anchor="la")


_WORKER: Renderer | None = None


def _init_worker(game: str, speed: float) -> None:
    """Builds a renderer in a worker process."""
    global _WORKER
    world = build_map()
    replay = load_replay(Path(game))
    _WORKER = Renderer(replay, build_timeline(replay, world, Pacing(meeting_speed=speed)), world)


def _render(t: float) -> bytes:
    """Renders one frame in a worker process."""
    return _WORKER.frame(t).tobytes()


def render_video(game: Path, speed: float, path: Path, fps: int, jobs: int) -> float:
    """Writes the replay as an H.264 video, by piping frames to ffmpeg.

    Args:
        game: The game folder.
        speed: Factor by which meeting time is played faster.
        path: Output file.
        fps: Frames per second.
        jobs: Number of worker processes.

    Returns:
        The duration of the video in seconds.
    """
    world = build_map()
    replay = load_replay(game)
    duration = build_timeline(replay, world, Pacing(meeting_speed=speed)).duration
    frames = int(duration * fps)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FINAL_W}x{FINAL_H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(jobs, initializer=_init_worker, initargs=(str(game), speed)) as pool:
        for i, raw in enumerate(pool.imap(_render, [k / fps for k in range(frames)], chunksize=10)):
            process.stdin.write(raw)
            if i % 400 == 0:
                logger.info("{}: frame {} of {}", path.name, i, frames)
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    logger.info("wrote {} ({:.0f} s)", path, duration)
    return duration


def slug(text: str) -> str:
    """Makes a file name part from a label."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main(argv: Sequence[str] | None = None) -> None:
    """Writes the videos or still pictures of games.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("games", type=Path, nargs="+", help="game folders, `Game_<id>_Run0`")
    parser.add_argument("--out", type=Path, default=Path("docs/media"))
    parser.add_argument("--speed", type=float, default=4.0, help="meeting time is played this many times faster")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (multiprocessing.cpu_count() or 2) - 1)))
    parser.add_argument("--still", type=float, action="append", default=[], help="write a picture at this video time instead of a video")
    parser.add_argument("--name", help="file name stem, for one game")
    parser.add_argument("--gif-seconds", type=float, default=0.0, help="also cut a GIF of this many seconds from the video")
    parser.add_argument("--gif-start", type=float, default=0.0, help="video time at which the GIF starts")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        world = build_map()
        replay = load_replay(game)
        stem = args.name or f"world-{slug(short_model(replay.model))}-{replay.protocol}-seed{replay.seed}"
        if args.still:
            timeline = build_timeline(replay, world, Pacing(meeting_speed=args.speed))
            renderer = Renderer(replay, timeline, world)
            logger.info("{}: {:.1f} s long", stem, timeline.duration)
            for t in args.still:
                renderer.frame(t).save(args.out / f"{stem}-t{t:g}.png")
            continue
        video = args.out / f"{stem}.mp4"
        render_video(game, args.speed, video, args.fps, args.jobs)
        if args.gif_seconds > 0:
            render_gif(video, args.out / f"{stem}.gif", args.gif_start, args.gif_seconds, width=800)


if __name__ == "__main__":
    main()
