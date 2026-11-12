"""Tests of the tile map of the Godot world, which must keep the rooms and the neighbours of the simulation."""

import pytest

from config.settings import ROOMS
from core.world_map import (
    CORRIDOR,
    CORRIDOR_LAYOUT,
    DOOR,
    FLOOR,
    HEIGHT,
    PASSABLE,
    ROOM_LAYOUT,
    VOID,
    WALL,
    WIDTH,
    Corridor,
    Room,
    build_map,
    display_name,
)
from scripts.draw_world_map import main, render


@pytest.fixture(scope="module")
def world():
    return build_map()


def test_rooms_are_those_of_the_simulation(world):
    assert set(world.rooms) == set(ROOMS)
    assert len(ROOM_LAYOUT) == len(ROOMS)


def test_neighbours_match_the_simulation(world):
    """Walking through doors and one corridor network reaches the rooms that `ROOMS` lists, and no others."""
    assert world.adjacency() == {name: set(neighbours) for name, neighbours in ROOMS.items()}


def test_every_room_floor_is_enclosed(world):
    """No floor, corridor or door tile touches the void, so nothing can leave the map."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if world.kind[y][x] in PASSABLE:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        assert world.tile(x + dx, y + dy) != VOID, f"open edge at {(x, y)}"


def test_a_door_joins_one_room_and_one_corridor(world):
    """The opposite sides of a door are a room and a corridor, so rooms never open into each other."""
    assert world.doors()
    for x, y in world.doors():
        horizontal = {world.tile(x - 1, y), world.tile(x + 1, y)} == {FLOOR, CORRIDOR}
        vertical = {world.tile(x, y - 1), world.tile(x, y + 1)} == {FLOOR, CORRIDOR}
        assert horizontal or vertical, f"door at {(x, y)}"


def test_walls_separate_rooms_and_corridors_except_at_doors(world):
    """A floor tile never touches a corridor tile directly."""
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if world.kind[y][x] == FLOOR:
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    assert world.tile(nx, ny) != CORRIDOR, f"room and corridor touch at {(x, y)}"


def test_every_room_is_reachable_from_every_other(world):
    graph = world.adjacency()
    seen, queue = set(), [next(iter(graph))]
    while queue:
        room = queue.pop()
        if room not in seen:
            seen.add(room)
            queue.extend(graph[room])
    assert seen == set(graph)


def test_each_room_has_floor_and_a_door(world):
    for name in world.rooms:
        tiles = [tile for tile, room in world.room_at.items() if room == name]
        assert tiles
        assert any(world.tile(x + dx, y + dy) == DOOR for x, y in tiles for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))), name


def test_cut_corners_remove_tiles():
    assert len(Room("a", 0, 0, 4, 4).floor()) == 16
    assert len(Room("a", 0, 0, 4, 4, (2, 0, 0, 0)).floor()) == 13
    assert len(Room("a", 0, 0, 4, 4, (2, 2, 2, 2)).floor()) == 4


def test_overlapping_areas_are_refused():
    with pytest.raises(ValueError):
        build_map((Room("a", 5, 5, 4, 4), Room("b", 8, 5, 4, 4)), ())
    with pytest.raises(ValueError):
        build_map((Room("a", 5, 5, 4, 4),), (Corridor(6, 6, 2, 2),))
    with pytest.raises(ValueError):
        build_map((Room("a", 78, 5, 4, 4),), ())


def test_a_small_layout_gets_a_door_where_a_corridor_meets_a_room():
    small = build_map((Room("a", 5, 5, 4, 4), Room("b", 20, 5, 4, 4)), (Corridor(10, 6, 9, 2),))
    assert small.adjacency() == {"a": {"b"}, "b": {"a"}}
    assert {small.tile(9, 6), small.tile(19, 6)} == {DOOR}


def test_the_layout_data_stays_inside_the_grid():
    for room in ROOM_LAYOUT:
        assert 1 <= room.x and room.x + room.w < WIDTH - 1
        assert 1 <= room.y and room.y + room.h < HEIGHT - 1
    for corridor in CORRIDOR_LAYOUT:
        assert corridor.x + corridor.w <= WIDTH and corridor.y + corridor.h <= HEIGHT


def test_labels_split_the_engines():
    assert display_name("UpperEngine") == "Upper Engine"
    assert display_name("LowerEngine") == "Lower Engine"
    assert display_name("MedBay") == "MedBay"


def test_text_rows_draw_every_tile_kind(world):
    text = "".join(world.rows())
    assert {".", ",", "#", "+", " "} <= set(text)
    assert len(world.rows()) == HEIGHT and all(len(row) == WIDTH for row in world.rows())
    assert world.kind[0][0] == VOID and WALL in {k for row in world.kind for k in row}


def test_render_size_and_door_colour(world):
    image = render(world, tile=8)
    assert image.size == (WIDTH * 8, HEIGHT * 8)
    x, y = world.doors()[0]
    red, green, blue = image.getpixel((x * 8 + 4, y * 8 + 4))
    assert red > 200 and green > 130 and blue < 130, "the middle of a door tile is amber"


def test_main_writes_the_picture(tmp_path):
    out = tmp_path / "map.png"
    main(["--out", str(out), "--tile", "6"])
    assert out.exists() and out.stat().st_size > 1000
