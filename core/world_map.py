"""The tile map of the Godot world, laid out like the ship of the simulation.

The simulation moves agents between 14 rooms that are listed with their neighbours in `config.settings.ROOMS`. The
world is a grid of tiles that are floor, corridor, wall or door, and the rooms keep their names and their neighbours, so a
claim such as "I was in Storage" stays checkable. The layout follows `frontend/static/images/Skeld.png`.

A room is a rectangle of floor tiles with optional cut corners. A corridor is a rectangle of floor tiles. A wall is any
empty tile next to a floor tile. A door is a wall tile with a room on one side and a corridor on the opposite side, so a
door never has to be placed by hand. This one layer says what blocks a body, a view and a voice.

Usage:
    from core.world_map import build_map
    world = build_map()
    world.adjacency()  # the neighbours of each room, found by walking the corridors
"""

import re
from collections import deque
from dataclasses import dataclass, field

WIDTH, HEIGHT = 80, 45

VOID, FLOOR, CORRIDOR, WALL, DOOR = range(5)
KIND_NAMES = {VOID: "void", FLOOR: "floor", CORRIDOR: "corridor", WALL: "wall", DOOR: "door"}
PASSABLE = (FLOOR, CORRIDOR, DOOR)


@dataclass(frozen=True)
class Room:
    """A room, as a rectangle of floor tiles with cut corners.

    Attributes:
        name: The name in `config.settings.ROOMS`.
        x: Column of the left floor tile.
        y: Row of the top floor tile.
        w: Width in tiles.
        h: Height in tiles.
        cut: Size of the cut at the top left, top right, bottom left and bottom right corners. A cut of 2 removes 3 tiles.
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    cut: tuple[int, int, int, int] = (0, 0, 0, 0)

    def floor(self) -> list[tuple[int, int]]:
        """Lists the floor tiles of the room."""
        top_left, top_right, bottom_left, bottom_right = self.cut
        tiles = []
        for j in range(self.h):
            for i in range(self.w):
                from_right, from_bottom = self.w - 1 - i, self.h - 1 - j
                if i + j < top_left or from_right + j < top_right or i + from_bottom < bottom_left or from_right + from_bottom < bottom_right:
                    continue
                tiles.append((self.x + i, self.y + j))
        return tiles

    def center(self) -> tuple[float, float]:
        """Returns the center of the room in tile units, where a tile spans one unit from its column and row."""
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class Corridor:
    """A rectangle of corridor floor."""

    x: int
    y: int
    w: int
    h: int

    def floor(self) -> list[tuple[int, int]]:
        """Lists the floor tiles of the corridor."""
        return [(self.x + i, self.y + j) for j in range(self.h) for i in range(self.w)]


ROOM_LAYOUT = (
    Room("Reactor", 4, 16, 6, 12, (2, 0, 2, 0)),
    Room("UpperEngine", 11, 8, 7, 8, (1, 0, 0, 0)),
    Room("LowerEngine", 11, 29, 7, 9, (0, 0, 1, 0)),
    Room("Security", 19, 17, 4, 8, (1, 1, 0, 0)),
    Room("MedBay", 25, 15, 8, 7, (0, 0, 0, 2)),
    Room("Electrical", 27, 26, 7, 7, (0, 2, 0, 0)),
    Room("Cafeteria", 34, 3, 18, 17, (5, 5, 5, 5)),
    Room("Admin", 48, 23, 7, 6, (0, 0, 0, 1)),
    Room("Storage", 36, 29, 9, 13, (0, 0, 2, 0)),
    Room("Communications", 47, 38, 7, 4, (0, 0, 1, 1)),
    Room("Weapons", 57, 7, 6, 7, (0, 1, 1, 0)),
    Room("O2", 53, 17, 5, 4, (1, 1, 0, 1)),
    Room("Navigation", 71, 17, 5, 7, (0, 2, 0, 2)),
    Room("Shields", 57, 30, 6, 8, (1, 0, 0, 2)),
)

# Each corridor network opens into the rooms that become neighbours through it.
CORRIDOR_LAYOUT = (
    # Reactor, Security, UpperEngine, LowerEngine
    Corridor(13, 17, 3, 11),
    Corridor(11, 21, 7, 3),
    # UpperEngine, MedBay, Cafeteria
    Corridor(19, 9, 14, 3),
    Corridor(27, 12, 3, 2),
    # LowerEngine, Electrical, Storage
    Corridor(19, 35, 16, 3),
    Corridor(28, 34, 3, 1),
    # Cafeteria, Admin, Storage
    Corridor(41, 21, 3, 7),
    Corridor(44, 24, 3, 3),
    # Cafeteria, Weapons
    Corridor(53, 9, 3, 3),
    # Weapons, O2, Navigation, Shields
    Corridor(59, 15, 3, 14),
    Corridor(62, 18, 8, 3),
    # Storage, Shields, Communications
    Corridor(46, 33, 10, 3),
    Corridor(49, 36, 3, 1),
)


def display_name(name: str) -> str:
    """Writes a room name for a label, so that `UpperEngine` reads as `Upper Engine`."""
    return re.sub(r"(?<=[a-z])(?=Engine)", " ", name)


@dataclass
class WorldMap:
    """A tile map with its rooms.

    Attributes:
        kind: Rows of tile kinds, indexed `kind[y][x]`.
        room_at: The room of each room floor tile.
        rooms: The rooms by name.
    """

    kind: list[list[int]]
    room_at: dict[tuple[int, int], str]
    rooms: dict[str, Room] = field(default_factory=dict)

    def tile(self, x: int, y: int) -> int:
        """Returns the kind of a tile, and `VOID` outside the grid."""
        return self.kind[y][x] if 0 <= x < WIDTH and 0 <= y < HEIGHT else VOID

    def doors(self) -> list[tuple[int, int]]:
        """Lists the door tiles."""
        return [(x, y) for y in range(HEIGHT) for x in range(WIDTH) if self.kind[y][x] == DOOR]

    def corridor_networks(self) -> list[set[tuple[int, int]]]:
        """Groups the corridor tiles into networks of tiles that touch each other."""
        seen: set[tuple[int, int]] = set()
        networks = []
        for y in range(HEIGHT):
            for x in range(WIDTH):
                if self.kind[y][x] != CORRIDOR or (x, y) in seen:
                    continue
                network, queue = {(x, y)}, deque([(x, y)])
                while queue:
                    cx, cy = queue.popleft()
                    for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        if self.tile(nx, ny) == CORRIDOR and (nx, ny) not in network:
                            network.add((nx, ny))
                            queue.append((nx, ny))
                seen |= network
                networks.append(network)
        return networks

    def adjacency(self) -> dict[str, set[str]]:
        """Finds the neighbours of each room as an agent could walk: through doors and one corridor network.

        Returns:
            For each room name, the rooms that a walk through one corridor network reaches without crossing another room.
        """
        door_links: list[tuple[str, tuple[int, int]]] = []
        for x, y in self.doors():
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if (nx, ny) in self.room_at:
                    door_links.append((self.room_at[(nx, ny)], self._corridor_beside(x, y)))
        neighbours: dict[str, set[str]] = {name: set() for name in self.rooms}
        for network in self.corridor_networks():
            opened = {room for room, tile in door_links if tile in network}
            for room in opened:
                neighbours[room] |= opened - {room}
        return neighbours

    def _corridor_beside(self, x: int, y: int) -> tuple[int, int]:
        """Returns the corridor tile that touches a door tile."""
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if self.tile(nx, ny) == CORRIDOR:
                return nx, ny
        raise ValueError(f"The door at {(x, y)} has no corridor beside it")

    def rows(self) -> list[str]:
        """Draws the map as text, with `.` for floor, `,` for corridor, `#` for wall, `+` for a door and a space for void."""
        glyph = {VOID: " ", FLOOR: ".", CORRIDOR: ",", WALL: "#", DOOR: "+"}
        return ["".join(glyph[k] for k in row) for row in self.kind]


def build_map(rooms: tuple[Room, ...] = ROOM_LAYOUT, corridors: tuple[Corridor, ...] = CORRIDOR_LAYOUT) -> WorldMap:
    """Builds the tile map of a layout.

    Args:
        rooms: The rooms.
        corridors: The corridors.

    Returns:
        The map, with walls around every floor tile and a door wherever a room and a corridor face each other across a wall.

    Raises:
        ValueError: If two floor areas share a tile or a tile lies outside the grid.
    """
    kind = [[VOID] * WIDTH for _ in range(HEIGHT)]
    room_at: dict[tuple[int, int], str] = {}
    for room in rooms:
        for x, y in room.floor():
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT) or kind[y][x] != VOID:
                raise ValueError(f"{room.name} has a tile at {(x, y)} that is outside the grid or already used")
            kind[y][x] = FLOOR
            room_at[(x, y)] = room.name
    for corridor in corridors:
        for x, y in corridor.floor():
            if not (0 <= x < WIDTH and 0 <= y < HEIGHT) or kind[y][x] == FLOOR:
                raise ValueError(f"The corridor at {(corridor.x, corridor.y)} has a tile at {(x, y)} that is outside the grid or in a room")
            kind[y][x] = CORRIDOR
    world = WorldMap(kind=kind, room_at=room_at, rooms={room.name: room for room in rooms})
    walls = [
        (x, y)
        for y in range(HEIGHT)
        for x in range(WIDTH)
        if kind[y][x] == VOID and any(world.tile(x + dx, y + dy) in (FLOOR, CORRIDOR) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    ]
    for x, y in walls:
        kind[y][x] = WALL
    doors = []
    for x, y in walls:
        for dx, dy in ((1, 0), (0, 1)):
            sides = {world.tile(x - dx, y - dy), world.tile(x + dx, y + dy)}
            if sides == {FLOOR, CORRIDOR}:
                doors.append((x, y))
    for x, y in doors:
        kind[y][x] = DOOR
    return world
