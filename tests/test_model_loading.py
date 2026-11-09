"""Regression tests for the pickled classifiers used at game time.

The expected probabilities were recorded with Python 3.13, scikit-learn 1.8.0,
and numpy 2.4.2. A dependency upgrade that changes how the pickles load or what
they predict fails these tests.

Run from the repository root:
    uv run --with pytest python -m pytest tests
"""

from pathlib import Path

import pandas as pd
import pytest

from core.game_engine import Observer
from results.context_pruner import ContextPruner

MODEL_DIR = Path(__file__).resolve().parent.parent / "results" / "classifiers" / "models"

SAMPLES = pd.DataFrame(
    {
        "Text": [
            "i was in place and saw agent_x vent near place",
            "agent_x was with me the whole time in place",
            "i think agent_x is lying about where they were",
            "completed my task in place nothing suspicious",
            "vote agent_x they were alone in place",
            "no idea what happened i was in place",
        ],
        "Reported": [0, 0, 1, 0, 0, 1],
        "Statement_Num": [1, 2, 1, 2, 1, 2],
    }
)

EXPECTED_PROBS = {
    "LogisticRegression": [0.500541, 0.799269, 0.642523, 0.421925, 0.192464, 0.91901],
    "SGD": [0.29226, 0.767714, 0.632815, 0.409205, 0.141198, 0.813286],
    "SVM": [0.380814, 0.669313, 0.460694, 0.131981, 0.094514, 0.952538],
}

PRUNER_THRESHOLDS = {
    10: 0.2210,
    9: 0.2516,
    8: 0.2600,
    7: 0.2625,
    6: 0.2720,
    5: 0.2828,
    4: 0.2718,
    "fallback": 0.2661,
}

PRUNER_LOG = """=== Round 1 ===
** MEETING CALLED by Agent_1
Agent_1: I found a body in Electrical and saw Agent_3 nearby.
Agent_2: I was in MedBay the whole time with Agent_4.
Agent_3: I was in Cafeteria, Agent_1 is lying.
Agent_4: Agent_3 was alone in Storage, I think they are suspicious.
Agent_1: Vote Agent_3.
Agent_2: I agree, Agent_3 was not with anyone.
Agent_3: I was doing my tasks and saw nothing.
Agent_4: Voting Agent_3.
"""

EXPECTED_PRUNER_STATE = {
    "Agent_1": 0.091643,
    "Agent_2": 0.718318,
    "Agent_3": 0.397356,
    "Agent_4": 0.25,
}


@pytest.fixture(scope="module")
def observer():
    return Observer(model_dir=str(MODEL_DIR))


def test_observer_loads_every_runtime_model(observer):
    assert set(observer.models) == set(EXPECTED_PROBS)


@pytest.mark.parametrize("name", sorted(EXPECTED_PROBS))
def test_observer_predictions_match_recorded_values(observer, name):
    probs = observer.models[name].predict_proba(SAMPLES)[:, 1]
    assert probs.tolist() == pytest.approx(EXPECTED_PROBS[name], abs=1e-4)


def test_observer_analyze_round_scores_every_agent(observer):
    statements = [
        {"Agent": "Agent_1", "Text": "I was in Electrical with Agent_2", "Reported": 1, "S_Num": 1},
        {"Agent": "Agent_2", "Text": "I saw Agent_3 in Storage", "Reported": 0, "S_Num": 1},
    ]
    scores = observer.analyze_round(statements)
    assert set(scores) == {"Agent_1", "Agent_2"}
    for per_model in scores.values():
        assert set(per_model) == set(EXPECTED_PROBS)
        assert all(0.0 <= p <= 1.0 for p in per_model.values())


def test_context_pruner_matches_recorded_values():
    pruner = ContextPruner()
    pruner.load_live_model(str(MODEL_DIR / "mlp_net.joblib"), PRUNER_THRESHOLDS)
    pruned, state, survivors = pruner.prune_live_log(PRUNER_LOG)
    assert survivors == {"Agent_1", "Agent_2", "Agent_3"}
    assert len(pruned.splitlines()) == 7
    assert state == pytest.approx(EXPECTED_PRUNER_STATE, abs=1e-4)
