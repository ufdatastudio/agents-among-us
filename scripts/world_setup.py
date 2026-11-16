"""The facts of a recorded game that a viewer needs before the first frame: the models, the locations and the options.

The text comes from the game folder, from the row that the P2 runner wrote for the game when one is given, and from
the voices when the video has any. Nothing here is guessed, and a line whose source is missing is left out.

Usage:
    setup = describe_setup(replay, build_map(), run=row, served="vLLM on UF HiPerGator", voices=voices)
    for section in setup.sections:
        print(section.heading, section.rows)
"""

from dataclasses import dataclass

from config.settings import MAX_MOVEMENT_PHASES
from core.world_map import WorldMap, display_name
from scripts.world_replay import MEETING_ROOM, Replay

VOICE_MODEL = "Kokoro 82M (hexgrad/Kokoro-82M)"
OVERLAP = {
    "interleaved": "interleaved, cut into windows of {window_s:g} s, and lines with one timestamp were spoken together",
}


@dataclass(frozen=True)
class Section:
    """A block of the setup screen.

    Attributes:
        heading: Name of the block.
        rows: Pairs of a label and its text.
    """

    heading: str
    rows: list[tuple[str, str]]


@dataclass(frozen=True)
class Setup:
    """What the setup screen says.

    Attributes:
        title: First line.
        subtitle: Second line.
        sections: The blocks of facts.
        rooms: Names of the rooms of the map, as the map labels them.
        notes: Sentences that tell the viewer how to read the video.
    """

    title: str
    subtitle: str
    sections: list[Section]
    rooms: list[str]
    notes: list[str]


def _voice(name: str) -> str:
    """Writes a Kokoro voice name such as `af_heart` as `Heart`."""
    return name.split("_", 1)[-1].capitalize()


def _models(replay: Replay, run: dict | None, served: str | None, voices: dict[str, str] | None) -> Section:
    """Lists the language model of the agents and, when there are voices, the speech model."""
    model = (run or {}).get("model") or replay.model
    rows = [("Agents", f"{model} plays all {len(replay.roles)} agents")]
    if served:
        rows.append(("Served by", served))
    if run:
        effort = run.get("reasoning_effort")
        if effort:
            rows.append(("Reasoning", f"{effort} effort, at most {run['max_tokens']} tokens per reply" if run.get("max_tokens") else f"{effort} effort"))
    if voices:
        names = ", ".join(f"{n.rsplit('_', 1)[1]} {_voice(v)}" for n, v in sorted(voices.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])))
        rows.append(("Voices", f"{VOICE_MODEL}, American English, one voice for each agent: {names}"))
    return Section("Models", rows)


def _game(replay: Replay, run: dict | None) -> Section:
    """Lists the roster, the rules and the scenario."""
    byz = sum(1 for r in replay.roles.values() if r == "byzantine")
    ticks = max([MAX_MOVEMENT_PHASES] + [entry.ticks() for entry in replay.rounds])
    rows = [
        ("Agents", f"{len(replay.roles)}: {len(replay.roles) - byz} honest and {byz} Byzantine, who have a red ring. Byzantine agents know each other"),
        ("A round", f"up to {ticks} moves. A Byzantine agent can tag an honest agent in the same room, and anyone who finds a body can report it"),
        ("A meeting", "a report calls everyone still in play to the Cafeteria. Each agent talks and then votes, and the most voted agent is ejected unless the vote ties"),
        ("Winning", "honest agents win when no Byzantine agent is left, and Byzantine agents win when they are as many as the honest agents"),
        ("Scenario", f"seed {replay.seed}, which fixes the starting rooms" + (f". The game stops after {run['max_rounds']} rounds" if run and run.get("max_rounds") else "")),
    ]
    return Section("Game", rows)


def _options(replay: Replay, speed: float, walk_s: float) -> Section:
    """Lists the discussion protocol and the settings of the timed room, leaving out what the record does not have."""
    room = replay.room
    have = room.keys()
    rows = [("Protocol", replay.protocol)]
    if {"budget_s", "tick_s"} <= have:
        rows.append(("Meeting", f"{room['budget_s']:g} s of meeting time, in ticks of {room['tick_s']:g} s. Everyone speaks at once"))
    if {"max_words", "words_per_s", "onset_jitter_s"} <= have:
        rows.append((
            "Statements",
            f"at most {room['max_words']} words, spoken at {room['words_per_s']:g} words per second, "
            f"and a speaker starts up to {room['onset_jitter_s']:g} s after the tick",
        ))
    if "overlap" in have:
        overlap = OVERLAP.get(room["overlap"], room["overlap"]).format(window_s=room.get("window_s", 0))
        rows.append(("The transcript", f"{overlap}. Agents read it as text"))
    extras = []
    if "dropout" in have:
        extras.append(f"{room['dropout']:.0%} of words dropped" if room["dropout"] else "no words dropped")
    if "barge_in" in have:
        extras.append("speakers may interrupt" if room["barge_in"] else "no interruptions")
    if room.get("quiet_close_ticks"):
        extras.append(f"the room closes after {room['quiet_close_ticks']} quiet ticks")
    if extras:
        rows.append(("Also", ", ".join(extras)))
    if "board" in replay.protocol:
        rows.append(("Board", "each agent's prompt lists the claims made so far next to what that agent saw itself"))
    pace = "meetings play at the speed they happened" if speed == 1 else f"meetings play {speed:g} times faster than they happened"
    rows.append(("This video", f"{pace}, and every walk takes {walk_s:g} s"))
    return Section("Options", rows)


def describe_setup(
    replay: Replay,
    world: WorldMap,
    run: dict | None = None,
    served: str | None = None,
    voices: dict[str, str] | None = None,
    speed: float = 1.0,
    walk_s: float = 1.6,
) -> Setup:
    """Collects what the setup screen says.

    Args:
        replay: The game.
        world: The map.
        run: The row that the P2 runner wrote for the game, which has the model settings, or None.
        served: How the model was served, such as `vLLM on UF HiPerGator`, or None to leave it out.
        voices: The voice of each agent, or None when the video is silent.
        speed: Factor by which meetings are played faster than they happened.
        walk_s: Seconds a walk takes in the video.

    Returns:
        The setup.
    """
    heard = "The voices were added after the game, from the logged statements. The agents read text and never heard them." if voices else "This video has no sound."
    rooms = [display_name(name) for name in world.rooms]
    notes = [
        "Each disc is an agent with its number. A red ring marks a Byzantine agent, and a white pulsing ring marks whoever is speaking.",
        "Honest agents never knew who was Byzantine. The roles are shown to the viewer.",
        "Routes, spots in a room and seats are drawn by the replay. The game logged only the room of each agent at each move.",
        heard,
    ]
    return Setup(
        title="Agents Among Us",
        subtitle="A recorded game of language-model agents on the tile map",
        sections=[_models(replay, run, served, voices), _game(replay, run), _options(replay, speed, walk_s)],
        rooms=rooms,
        notes=notes,
    )


def locations_text(world: WorldMap) -> str:
    """Describes the map for the setup screen.

    Args:
        world: The map.

    Returns:
        One or two sentences.
    """
    return (
        f"{len(world.rooms)} rooms joined by corridors and doors. Meetings are held in the {display_name(MEETING_ROOM)}. "
        "Agents walk the corridors, and tags, reports and votes all happen in the rooms and at the table."
    )
