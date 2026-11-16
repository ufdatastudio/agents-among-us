"""Voices for a recorded game: speaks each statement of a game folder and lays the clips on the video's clock.

The synthesis step needs Kokoro and runs on HiPerGator, where the weights are cached. The track step needs only numpy,
so the video is rendered on any machine that has the folder of clips. The agents of the game read text and never heard
these voices.

Usage:
    uv run --no-project --with kokoro --with numpy --with loguru python -m scripts.world_audio GAME --out speech/GAME
    track = build_track(timeline, Path("speech/GAME"), offset_s=15.0, length_s=420.0)
"""

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from loguru import logger

from core.audio.mix import Clip, overlap_fraction
from core.audio.speech import (
    SpeechClip,
    Utterance,
    place_clips,
    read_speech,
    synthesize_utterances,
)
from core.audio.tts import assign_voices
from scripts.world_replay import Replay, Timeline, load_replay

PAN_WIDTH = 0.7


def agent_number(name: str) -> int:
    """Returns the number of an agent name such as `Agent_7`."""
    return int(name.rsplit("_", 1)[1])


def utterances_of(replay: Replay) -> list[Utterance]:
    """Lists the statements of every meeting in the order the video shows them.

    Args:
        replay: The game.

    Returns:
        The statements, numbered within their meeting.
    """
    found = []
    for entry in replay.rounds:
        if entry.meeting is None:
            continue
        for index, s in enumerate(entry.meeting.spoken):
            found.append(Utterance(entry.number, index, s.author, s.start_s, s.end_s, s.text))
    return found


def voices_of(replay: Replay) -> dict[str, str]:
    """Gives each agent of a game its voice, which stays the same in every run with the same seed."""
    return assign_voices(sorted(replay.roles, key=agent_number), replay.seed)


def seat_pans(timeline: Timeline) -> dict[str, float]:
    """Places each voice left or right according to where its agent sits at the table.

    Args:
        timeline: The plan of the video, which has the seats.

    Returns:
        A pan from -`PAN_WIDTH` to `PAN_WIDTH` for each agent.
    """
    xs = {name: point[0] for name, point in timeline.seats.items()}
    if not xs:
        return {}
    middle = (max(xs.values()) + min(xs.values())) / 2
    half = max((max(xs.values()) - min(xs.values())) / 2, 1e-9)
    return {name: PAN_WIDTH * (x - middle) / half for name, x in xs.items()}


def match_clips(timeline: Timeline, clips: Sequence[SpeechClip]) -> list[tuple[float, SpeechClip]]:
    """Pairs each caption of the video with the clip of the same statement.

    Args:
        timeline: The plan of the video.
        clips: Clips made from the same game.

    Returns:
        The video time at which each clip starts and the clip.

    Raises:
        ValueError: If the clips and the captions disagree on who spoke what, which means the clips are of another game.
    """
    by_round: dict[int, list] = defaultdict(list)
    for caption in timeline.captions:
        by_round[caption.round_num].append(caption)
    pairs = []
    for clip in clips:
        u = clip.utterance
        captions = by_round.get(u.round_num, [])
        if u.index >= len(captions):
            raise ValueError(f"round {u.round_num} has no statement {u.index} in this game")
        caption = captions[u.index]
        if caption.speaker != u.author or caption.text != u.text:
            raise ValueError(f"statement {u.index} of round {u.round_num} differs from the clip, so the speech is of another game")
        pairs.append((caption.start, clip))
    return pairs


def build_track(timeline: Timeline, folder: Path, offset_s: float, length_s: float) -> np.ndarray:
    """Builds the stereo track of a video.

    Args:
        timeline: The plan of the video, with meetings played at the speed they happened.
        folder: Folder that the synthesis step wrote.
        offset_s: Seconds of video before the game starts, such as a title card.
        length_s: Length of the whole video in seconds.

    Returns:
        The track, shape (samples, 2) at `SAMPLE_RATE`.

    Raises:
        ValueError: If the clips do not belong to the game, or the meetings are not played at the speed they happened.
    """
    if timeline.pacing.meeting_speed != 1.0:
        raise ValueError("voices need meetings played at the speed they happened, so the meeting speed must be 1")
    _, clips = read_speech(folder)
    pairs = match_clips(timeline, clips)
    pans = seat_pans(timeline)
    placements = [(offset_s + start, clip.samples, pans.get(clip.utterance.author, 0.0)) for start, clip in pairs]
    return place_clips(placements, length_s)


def concurrency(clips: Sequence[SpeechClip]) -> tuple[float, float]:
    """Measures how much the voices overlap.

    Args:
        clips: The clips of one game.

    Returns:
        The share of speaking time with 2 or more voices, and the mean number of voices while anyone speaks.
    """
    # Each meeting gets its own stretch of the clock, so statements of different meetings never count as overlapping.
    flat = [Clip(c.utterance.author, c.utterance.start_s + 10_000.0 * c.utterance.round_num, c.samples) for c in clips]
    total = sum(c.duration_s for c in flat)
    events = sorted([(c.start_s, 1) for c in flat] + [(c.end_s, -1) for c in flat])
    active, spoken, last = 0, 0.0, 0.0
    for time_s, change in events:
        if active:
            spoken += time_s - last
        active, last = active + change, time_s
    return overlap_fraction(flat), (total / spoken if spoken else 0.0)


def main(argv: Sequence[str] | None = None) -> None:
    """Speaks the statements of a game folder.

    Args:
        argv: Arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("game", type=Path, help="game folder, `Game_<id>_Run0`")
    parser.add_argument("--out", type=Path, required=True, help="folder for the clips and `speech.json`")
    parser.add_argument("--device", default=None, help="torch device for Kokoro, such as cuda")
    args = parser.parse_args(argv)
    from core.audio.tts import Synthesizer

    replay = load_replay(args.game)
    utterances = utterances_of(replay)
    voices = voices_of(replay)
    logger.info("{} statements by {} agents, voices {}", len(utterances), len(voices), voices)
    clips = synthesize_utterances(utterances, voices, Synthesizer(device=args.device), args.out)
    shared, mean = concurrency(clips)
    cut = [c for c in clips if c.natural_s - c.duration_s > 1e-3]
    lost = max((c.natural_s - c.duration_s for c in cut), default=0.0)
    logger.info(
        "wrote {} clips, {:.0f} s of speech, {:.0%} of speaking time has 2 or more voices (mean {:.1f} at a time), {} clips cut, at most {:.1f} s of speech lost",
        len(clips), sum(c.duration_s for c in clips), shared, mean, len(cut), lost,
    )


if __name__ == "__main__":
    main()
