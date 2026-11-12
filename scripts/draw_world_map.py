"""Draws the tile map of the Godot world as a picture.

The map comes from `core/world_map.py`. It has the 14 rooms and the neighbours of the simulation, laid out like
`frontend/static/images/Skeld.png`, with one wall layer and doors.

Usage:
    uv run python -m scripts.draw_world_map --out docs/media/godot-world-map.png
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from core.world_map import CORRIDOR, DOOR, FLOOR, HEIGHT, PASSABLE, VOID, WALL, WIDTH, WorldMap, build_map, display_name

TILE = 20
SUPER = 2

BACKGROUND = (15, 18, 24)
FLOOR_COLORS = ((47, 56, 73), (44, 52, 68))
CORRIDOR_COLORS = ((34, 41, 54), (32, 38, 50))
WALL_COLOR = (92, 104, 128)
WALL_EDGE = (138, 152, 180)
DOOR_COLOR = (240, 176, 64)
INK = (236, 239, 244)
SHADOW = (14, 17, 23)
MUTED = (150, 158, 172)


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    """Loads the DejaVu fonts that ship with matplotlib, at the supersampled sizes of the picture."""
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return {
        "room": ImageFont.truetype(str(base / "DejaVuSans-Bold.ttf"), 13 * SUPER),
        "legend": ImageFont.truetype(str(base / "DejaVuSans.ttf"), 12 * SUPER),
        "title": ImageFont.truetype(str(base / "DejaVuSans-Bold.ttf"), 14 * SUPER),
    }


def _box(x: int, y: int, scale: int) -> tuple[int, int, int, int]:
    return x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1


def draw_tiles(draw: ImageDraw.ImageDraw, world: WorldMap, scale: int) -> None:
    """Paints the floor, corridor, wall and door tiles.

    Args:
        draw: The canvas.
        world: The map.
        scale: Pixels per tile.
    """
    for y in range(HEIGHT):
        for x in range(WIDTH):
            kind = world.kind[y][x]
            if kind == VOID:
                continue
            checker = (x + y) % 2
            if kind == FLOOR:
                draw.rectangle(_box(x, y, scale), fill=FLOOR_COLORS[checker])
            elif kind in (CORRIDOR, DOOR):
                draw.rectangle(_box(x, y, scale), fill=CORRIDOR_COLORS[checker])
            elif kind == WALL:
                draw.rectangle(_box(x, y, scale), fill=WALL_COLOR)
    edge = max(2, scale // 10)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if world.kind[y][x] != WALL:
                continue
            x0, y0, x1, y1 = _box(x, y, scale)
            if world.tile(x, y - 1) in PASSABLE:
                draw.rectangle((x0, y1 - edge, x1, y1), fill=WALL_EDGE)
            if world.tile(x, y + 1) in PASSABLE:
                draw.rectangle((x0, y0, x1, y0 + edge), fill=WALL_EDGE)
            if world.tile(x - 1, y) in PASSABLE:
                draw.rectangle((x1 - edge, y0, x1, y1), fill=WALL_EDGE)
            if world.tile(x + 1, y) in PASSABLE:
                draw.rectangle((x0, y0, x0 + edge, y1), fill=WALL_EDGE)


def draw_doors(draw: ImageDraw.ImageDraw, world: WorldMap, scale: int) -> None:
    """Marks each door tile with a bar across the opening.

    Args:
        draw: The canvas.
        world: The map.
        scale: Pixels per tile.
    """
    bar = scale * 3 // 10
    for x, y in world.doors():
        x0, y0, x1, y1 = _box(x, y, scale)
        if {world.tile(x - 1, y), world.tile(x + 1, y)} == {FLOOR, CORRIDOR}:
            draw.rectangle((x0 + (scale - bar) // 2, y0, x0 + (scale + bar) // 2, y1), fill=DOOR_COLOR)
        else:
            draw.rectangle((x0, y0 + (scale - bar) // 2, x1, y0 + (scale + bar) // 2), fill=DOOR_COLOR)


def draw_labels(draw: ImageDraw.ImageDraw, world: WorldMap, scale: int, font: ImageFont.FreeTypeFont) -> None:
    """Writes each room name in the middle of its floor.

    Args:
        draw: The canvas.
        world: The map.
        scale: Pixels per tile.
        font: The font of the labels.
    """
    for room in world.rooms.values():
        cx, cy = room.center()
        xy = (cx * scale, cy * scale)
        draw.text((xy[0] + 1, xy[1] + 2), display_name(room.name), font=font, fill=SHADOW, anchor="mm")
        draw.text(xy, display_name(room.name), font=font, fill=INK, anchor="mm")


def draw_legend(draw: ImageDraw.ImageDraw, scale: int, fonts: dict[str, ImageFont.FreeTypeFont], origin: tuple[int, int]) -> None:
    """Draws a key of the tile kinds in a free corner of the map.

    Args:
        draw: The canvas.
        scale: Pixels per tile.
        fonts: The fonts.
        origin: Tile column and row of the top left corner of the key.
    """
    x, y = origin
    px, py = x * scale, y * scale
    draw.text((px, py), "Tile map, 80 by 45 tiles", font=fonts["title"], fill=INK, anchor="la")
    rows = [("Room floor", FLOOR_COLORS[0], None), ("Corridor", CORRIDOR_COLORS[0], None), ("Wall", WALL_COLOR, WALL_EDGE), ("Door", CORRIDOR_COLORS[0], DOOR_COLOR)]
    for i, (label, fill, mark) in enumerate(rows):
        top = py + int(scale * (1.7 + i * 1.3))
        draw.rectangle((px, top, px + scale, top + scale), fill=fill)
        if mark == WALL_EDGE:
            draw.rectangle((px, top, px + scale // 6, top + scale), fill=mark)
        elif mark == DOOR_COLOR:
            draw.rectangle((px, top + scale * 35 // 100, px + scale, top + scale * 65 // 100), fill=mark)
        draw.text((px + int(scale * 1.5), top + scale // 2), label, font=fonts["legend"], fill=MUTED, anchor="lm")


def render(world: WorldMap, tile: int = TILE) -> Image.Image:
    """Draws the map with supersampling and returns it at `tile` pixels per tile.

    Args:
        world: The map.
        tile: Pixels per tile in the result.

    Returns:
        An image of `WIDTH * tile` by `HEIGHT * tile` pixels.
    """
    scale = tile * SUPER
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BACKGROUND)
    draw = ImageDraw.Draw(image)
    fonts = load_fonts()
    draw_tiles(draw, world, scale)
    draw_doors(draw, world, scale)
    draw_labels(draw, world, scale, fonts["room"])
    draw_legend(draw, scale, fonts, (65, 36))
    return image.resize((WIDTH * tile, HEIGHT * tile), Image.LANCZOS)


def main(argv: Sequence[str] | None = None) -> None:
    """Writes the picture of the map.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/media/godot-world-map.png"))
    parser.add_argument("--tile", type=int, default=TILE, help="pixels per tile")
    args = parser.parse_args(argv)
    world = build_map()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(world, args.tile).save(args.out, optimize=True)
    logger.info(f"Wrote {args.out} with {len(world.rooms)} rooms and {len(world.doors())} door tiles")


if __name__ == "__main__":
    main()
