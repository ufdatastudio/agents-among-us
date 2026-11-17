"""Pytest configuration shared by the tests in this directory."""

import os
from pathlib import Path

import numpy as np
import pytest

from core.audio.mix import SAMPLE_RATE

# The navigator image has no torch or transformers. Match its mode so the tests
# import the game engine without the GPU dependencies.
os.environ.setdefault("APP_MODE", "navigator")


RESULTS = """=== Round Results Log ===

=== Round 1 ===
ELIMINATION: Agent_3.
Round 1 Votes Received: {'Agent_1': 1, 'Agent_0': 2}
EJECTION: Agent_0 was ejected.
Player Statuses: {'Agent_0': 'ejected', 'Agent_1': 'active', 'Agent_2': 'active', 'Agent_3': 'eliminated'}
"""
CAUSE = "Agent_1: Body Reported: Agent_3"


def action_log(agent: str, ticks: list[tuple[str, str, str]], cause: str | None = None) -> str:
    """Writes an `action.log` of one round in the format of the engine."""
    lines = [f"=== Action Log for {agent} ===", "", "Round 1/10"]
    for number, (room, verb, target) in enumerate(ticks, 1):
        lines += [f"Current Location: [{room}]", "  -> Occupants: []", f"Action {number} Selected: {verb} -> {target}", "======================", ""]
    if cause:
        lines += [f"Discussion Started because {cause}. Additional victims confirmed eliminations: None", "Round End.", ""]
    return "\n".join(lines)


def write_game(root: Path) -> Path:
    """Writes a game of 4 agents. Agent_0 is Byzantine and tags Agent_3 in Storage, Agent_1 reports the body, and Agent_0 is ejected."""
    game = root / "Game_tag_interleaved-60_3_Run0"
    logs = {
        "Agent_0": [("Storage", "tag", "Agent_3"), ("Storage", "move", "Cafeteria")],
        "Agent_1": [("Cafeteria", "move", "Storage"), ("Storage", "report", "Agent_3")],
        "Agent_2": [("Admin", "move", "Storage"), ("Storage", "move", "Admin")],
        "Agent_3": [("Storage", "move", "Electrical")],
    }
    votes = {"Agent_0": "Agent_1", "Agent_1": "Agent_0", "Agent_2": "Agent_0"}
    for role, names in (("Byz", ["Agent_0"]), ("Honest", ["Agent_1", "Agent_2", "Agent_3"])):
        for name in names:
            folder = game / role / name
            folder.mkdir(parents=True)
            (folder / "action.log").write_text(action_log(name, logs[name], CAUSE), encoding="utf-8")
            if name in votes:
                (folder / "vote.log").write_text(f"=== Voting History ===\nRound 1: Voted for {votes[name]}\n", encoding="utf-8")
    (game / "discussion_chat.csv").write_text(
        "discussion_num,reason,agent_num,model,role,message\n"
        '1,"** MEETING CALLED by Agent_1. Body reported: Agent_3 located in Storage. **",1,vllm:org/model,Honest,Agent_0 was alone.\n',
        encoding="utf-8",
    )
    (game / "roundResults.log").write_text(RESULTS, encoding="utf-8")
    (game / "stats.csv").write_text("agent_name,alignment,won_game\nAgent_0,B,0\nAgent_1,H,1\nAgent_2,H,1\nAgent_3,H,1\n", encoding="utf-8")
    (game / "room").mkdir()
    rows = [
        '{"type": "room", "closed_at": 60.0}',
        '{"type": "utterance", "author": "Agent_1", "start_s": 12.0, "end_s": 22.0, "text": "Agent_0 was alone near the body."}',
        '{"type": "utterance", "author": "Agent_0", "start_s": 1.0, "end_s": 9.0, "text": "I was in Storage the whole time."}',
    ]
    (game / "room" / "room_round1.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return game


@pytest.fixture()
def mini_game(tmp_path: Path) -> Path:
    """A finished game of 4 agents with one tag, one report and one meeting, written in the format of the engine."""
    return write_game(tmp_path)


class FakeVoice:
    """A synthesizer that speaks every word for a fixed time, at a level that depends on the voice."""

    def __init__(self, seconds_per_word: float = 0.5) -> None:
        self.seconds_per_word = seconds_per_word
        self.calls: list[tuple[str, str, float]] = []

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self.calls.append((text, voice, speed))
        length = round(len(text.split()) * self.seconds_per_word / speed * SAMPLE_RATE)
        return np.full(length, 0.5 if voice == "v0" else 0.25, dtype=np.float32)


@pytest.fixture()
def fake_voice() -> FakeVoice:
    """A synthesizer with no speech model."""
    return FakeVoice()


class ScriptedHandle:
    """A session that answers each chunk with the next burst of a script, and stays silent when the script ends."""

    def __init__(self, script: list) -> None:
        from core.duplex import Burst

        self._burst = Burst
        self.script = list(script)
        self.prompts = []
        self.limits = []
        self.closed = False

    async def send(self, prompt, max_tokens):
        self.prompts.append(prompt)
        self.limits.append(max_tokens)
        item = self.script.pop(0) if self.script else "<silent/>"
        return item if isinstance(item, self._burst) else self._burst(item, "stop")

    async def close(self):
        self.closed = True


@pytest.fixture
def scripted():
    """Makes a scripted session from a list of bursts or strings."""
    return ScriptedHandle
