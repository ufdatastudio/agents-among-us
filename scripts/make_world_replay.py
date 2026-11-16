"""Renders a recorded game as a video on the tile map of the Godot world.

Agents walk the corridors to their rooms at each tick, tags leave bodies, a report calls everyone to the table of the
Cafeteria, the statements of the meeting appear as they were spoken, and the votes and the ejection follow. The viewer
sees every role. Honest agents never knew who was Byzantine, and Byzantine agents knew their teammates. The walks come
from `scripts/world_replay.py`, and nothing here changes a game. A video starts with a screen that lists the models, the
locations and the options of the game, and it has sound when a folder of clips from `scripts/world_audio.py` is given.

Usage:
    uv run --with matplotlib python -m scripts.make_world_replay logs/p2_x/Game_tag_interleaved-180_3_Run0 --out docs/media
    uv run --with matplotlib python -m scripts.make_world_replay GAME --still 12.5 --still 60 --out /path/for/pictures
    uv run --with matplotlib python -m scripts.make_world_replay GAME --gif-start 0 --gif-seconds 14 --out docs/media
    uv run --with matplotlib python -m scripts.make_world_replay GAME --card --out /path/for/pictures
    uv run --with matplotlib python -m scripts.make_world_replay GAME --speech speech/GAME --run results/p2/x.jsonl --out docs/media

The video needs ffmpeg on the path.
"""

import argparse
import json
import math
import multiprocessing
import re
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from core.audio.mix import SAMPLE_RATE, write_wav
from core.audio.speech import read_voices
from core.world_map import HEIGHT, WIDTH, WorldMap, build_map
from scripts.analyze_p2_text import short_model
from scripts.draw_world_map import draw_doors, draw_tiles
from scripts.make_replay import render_gif, wrap
from scripts.world_audio import build_track
from scripts.world_replay import (
    MEETING_ROOM,
    Frame,
    Pacing,
    Replay,
    Timeline,
    build_timeline,
    load_replay,
)
from scripts.world_setup import Setup, describe_setup, locations_text

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
    sizes = {"room": 10, "token": 10, "heading": 22, "title": 20, "name": 15, "body": 14, "small": 12, "clock": 16, "big": 28, "cbody": 16, "clabel": 13, "chead": 17}
    heavy = {"room", "token", "heading", "title", "name", "clock", "big", "chead"}
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

    def intro(self, setup: Setup) -> Image.Image:
        """Draws the setup screen: the models, the map with its locations, the game and the options.

        Args:
            setup: What the screen says.

        Returns:
            The picture at the final size.
        """
        image = Image.new("RGB", (FINAL_W * S, FINAL_H * S), BG)
        draw = ImageDraw.Draw(image)
        f = self.fonts
        line = 21
        draw.text((48 * S, 28 * S), setup.title, font=f["big"], fill=INK, anchor="la")
        draw.text((48 * S, 68 * S), f"{setup.subtitle}: {short_model(self.replay.model)}, {self.replay.protocol}, seed {self.replay.seed}", font=f["cbody"], fill=MUTED, anchor="la")
        draw.line((48 * S, 104 * S, (FINAL_W - 48) * S, 104 * S), fill=RULE, width=S)
        left, width = 48, 690
        draw.text((left * S, 122 * S), "LOCATIONS", font=f["chead"], fill=ACCENT, anchor="la")
        height = round(width * MAP_H / MAP_W)
        thumb = self.base.crop((0, 0, MAP_W * S, MAP_H * S)).resize((width * S, height * S), Image.LANCZOS)
        image.paste(thumb, (left * S, 154 * S))
        draw.rectangle((left * S, 154 * S, (left + width) * S, (154 + height) * S), outline=RULE, width=S)
        y = 154 + height + 16
        for text in wrap(locations_text(self.world), f["cbody"], width * S):
            draw.text((left * S, y * S), text, font=f["cbody"], fill=INK, anchor="la")
            y += line
        y += 8
        for text in wrap("Rooms: " + ", ".join(setup.rooms) + ".", f["cbody"], width * S):
            draw.text((left * S, y * S), text, font=f["cbody"], fill=MUTED, anchor="la")
            y += line
        x, label = 800, 140
        column = FINAL_W - 48 - x - label
        y = 122
        for section in setup.sections:
            draw.text((x * S, y * S), section.heading.upper(), font=f["chead"], fill=ACCENT, anchor="la")
            y += 30
            for name, text in section.rows:
                draw.text((x * S, (y + 2) * S), name, font=f["clabel"], fill=MUTED, anchor="la")
                lines = wrap(text, f["cbody"], column * S)
                for k, text_line in enumerate(lines):
                    draw.text(((x + label) * S, (y + k * line) * S), text_line, font=f["cbody"], fill=INK, anchor="la")
                y += max(1, len(lines)) * line + 6
            y += 12
        y = FINAL_H - 22 - 19 * len(setup.notes)
        draw.line((48 * S, (y - 10) * S, (FINAL_W - 48) * S, (y - 10) * S), fill=RULE, width=S)
        for note in setup.notes:
            draw.text((48 * S, y * S), note, font=f["clabel"], fill=MUTED, anchor="la")
            y += 19
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
        note = wrap("Roles are shown to the viewer. Honest agents never knew them.", self.fonts["small"], (FINAL_W - 24 - PANEL_X - 24) * S)
        for i, line in enumerate(note):
            draw.text((x, (FINAL_H - 22 - 18 * (len(note) - 1 - i)) * S), line, font=self.fonts["small"], fill=DIM, anchor="ls")

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
_CARD: Image.Image | None = None
_INTRO = 0.0
FADE_S = 0.6


def _init_worker(game: str, speed: float, intro_s: float, setup: Setup | None) -> None:
    """Builds a renderer, and the setup screen when there is one, in a worker process."""
    global _WORKER, _CARD, _INTRO
    world = build_map()
    replay = load_replay(Path(game))
    _WORKER = Renderer(replay, build_timeline(replay, world, Pacing(meeting_speed=speed)), world)
    _INTRO = intro_s if setup is not None else 0.0
    _CARD = _WORKER.intro(setup) if setup is not None else None


def _render(t: float) -> bytes:
    """Renders one frame of the video in a worker process, where the setup screen comes first and fades into the game."""
    if _CARD is not None and t < _INTRO:
        if t < _INTRO - FADE_S:
            return _CARD.tobytes()
        return Image.blend(_CARD, _WORKER.frame(0.0), (t - (_INTRO - FADE_S)) / FADE_S).tobytes()
    return _WORKER.frame(t - _INTRO).tobytes()


def render_video(
    game: Path,
    speed: float,
    path: Path,
    fps: int,
    jobs: int,
    setup: Setup | None = None,
    intro_s: float = 0.0,
    speech: Path | None = None,
) -> float:
    """Writes the replay as an H.264 video, by piping frames to ffmpeg.

    Args:
        game: The game folder.
        speed: Factor by which meeting time is played faster. It must be 1 when there is speech.
        path: Output file.
        fps: Frames per second.
        jobs: Number of worker processes.
        setup: What the setup screen says, or None for a video that starts with the game.
        intro_s: Seconds the setup screen stays. It is ignored without a setup.
        speech: Folder of clips from `scripts/world_audio.py`, or None for a silent video.

    Returns:
        The duration of the video in seconds.
    """
    world = build_map()
    replay = load_replay(game)
    timeline = build_timeline(replay, world, Pacing(meeting_speed=speed))
    intro_s = intro_s if setup is not None else 0.0
    duration = timeline.duration + intro_s
    frames = int(duration * fps)
    with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FINAL_W}x{FINAL_H}", "-r", str(fps), "-i", "-"]
        if speech is not None:
            track = Path(scratch) / "speech.wav"
            write_wav(track, build_track(timeline, speech, intro_s, frames / fps), SAMPLE_RATE)
            command += ["-i", str(track), "-c:a", "aac", "-b:a", "96k", "-ar", "48000"]
        command += ["-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(jobs, initializer=_init_worker, initargs=(str(game), speed, intro_s, setup)) as pool:
            for i, raw in enumerate(pool.imap(_render, [k / fps for k in range(frames)], chunksize=10)):
                process.stdin.write(raw)
                if i % 400 == 0:
                    logger.info("{}: frame {} of {}", path.name, i, frames)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    logger.info("wrote {} ({:.0f} s{})", path, duration, ", with sound" if speech is not None else "")
    return duration


def find_run(path: Path, replay: Replay) -> dict | None:
    """Finds the row that the P2 runner wrote for a game.

    Args:
        path: A JSON lines file of rows.
        replay: The game.

    Returns:
        The row whose `game_id` is the game's, or None.
    """
    wanted = replay.game_id.removesuffix("_Run0")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and json.loads(line).get("game_id", "").removesuffix("_Run0") == wanted:
            return json.loads(line)
    return None


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
    parser.add_argument("--speed", type=float, default=None, help="meeting time is played this many times faster, 4 for a silent video and 1 with speech")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (multiprocessing.cpu_count() or 2) - 1)))
    parser.add_argument("--intro", type=float, default=14.0, help="seconds of the setup screen at the start of a video, 0 for none")
    parser.add_argument("--card", action="store_true", help="write the setup screen as a picture instead of a video")
    parser.add_argument("--speech", type=Path, help="folder of clips from scripts.world_audio, for one game, which gives the video sound")
    parser.add_argument("--run", type=Path, help="JSON lines file with the row of the game, for the model settings")
    parser.add_argument("--served", help="how the model was served, such as 'vLLM on UF HiPerGator'")
    parser.add_argument("--still", type=float, action="append", default=[], help="write a picture at this video time instead of a video")
    parser.add_argument("--name", help="file name stem, for one game")
    parser.add_argument("--gif-seconds", type=float, default=0.0, help="also cut a GIF of this many seconds from the video")
    parser.add_argument("--gif-start", type=float, default=0.0, help="video time at which the GIF starts")
    args = parser.parse_args(argv)
    if args.speech is not None and len(args.games) != 1:
        parser.error("--speech belongs to one game, so give one game folder")
    speed = args.speed if args.speed is not None else (1.0 if args.speech is not None else 4.0)
    if args.speech is not None and speed != 1.0:
        parser.error("--speech needs --speed 1, because voices cannot be sped up with the meeting")
    args.out.mkdir(parents=True, exist_ok=True)
    for game in args.games:
        world = build_map()
        replay = load_replay(game)
        row = find_run(args.run, replay) if args.run is not None else None
        if args.run is not None and row is None:
            logger.warning("{} has no row for {}, so the model settings are left out", args.run, replay.game_id)
        voices = read_voices(args.speech) if args.speech is not None else None
        setup = describe_setup(replay, world, run=row, served=args.served, voices=voices, speed=speed, walk_s=Pacing().walk)
        stem = args.name or f"world-{slug(short_model(replay.model))}-{replay.protocol}-seed{replay.seed}" + ("-with-sound" if voices else "")
        if args.card or args.still:
            timeline = build_timeline(replay, world, Pacing(meeting_speed=speed))
            renderer = Renderer(replay, timeline, world)
            logger.info("{}: {:.1f} s long", stem, timeline.duration)
            if args.card:
                renderer.intro(setup).save(args.out / f"{stem}-setup.png")
            for t in args.still:
                renderer.frame(t).save(args.out / f"{stem}-t{t:g}.png")
            continue
        video = args.out / f"{stem}.mp4"
        render_video(game, speed, video, args.fps, args.jobs, setup if args.intro > 0 else None, args.intro, args.speech)
        if args.gif_seconds > 0:
            render_gif(video, args.out / f"{stem}.gif", args.gif_start, args.gif_seconds, width=800)


if __name__ == "__main__":
    main()
