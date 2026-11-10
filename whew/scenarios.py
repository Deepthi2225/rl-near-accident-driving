"""Scenario registry: environment class, expert driver and default file locations."""
from .envs import CutInEnv, PedestrianCrossingEnv
from .experts import expert_action, expert_action_cut_in

SCENARIOS = {
    # Scenario 1 keeps its original file names (results from v0.1-v0.4).
    "pedestrian": {
        "env": PedestrianCrossingEnv,
        "expert": expert_action,
        "data": "data/pedestrian_crossing_expert.npz",
        "coil": "checkpoints/coil.pt",
        "checkpoints": "checkpoints",
        "results": "results",
        "eval": "results/eval_pedestrian_crossing.json",
        "title": "pedestrian dart-out",
    },
    "cut_in": {
        "env": CutInEnv,
        "expert": expert_action_cut_in,
        "data": "data/cut_in_expert.npz",
        "coil": "checkpoints/cut_in/coil.pt",
        "checkpoints": "checkpoints/cut_in",
        "results": "results/cut_in",
        "eval": "results/cut_in/eval.json",
        "title": "sudden cut-in",
    },
}
