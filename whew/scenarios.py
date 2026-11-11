"""Scenario registry: environment class, expert driver and default file locations."""
from .envs import CutInEnv, PedestrianCrossingEnv, RedLightRunnerEnv, UnprotectedLeftEnv
from .experts import expert_action, expert_action_cut_in, expert_action_left_turn, expert_action_red_light

SCENARIOS = {
    # Scenario 1 keeps its original file names (results from v0.1-v0.4).
    "pedestrian": {
        "env": PedestrianCrossingEnv,
        "town": "Town10HD_Opt",
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
        "town": "Town10HD_Opt",
        "expert": expert_action_cut_in,
        "data": "data/cut_in_expert.npz",
        "coil": "checkpoints/cut_in/coil.pt",
        "checkpoints": "checkpoints/cut_in",
        "results": "results/cut_in",
        "eval": "results/cut_in/eval.json",
        "title": "sudden cut-in",
    },
    "red_light": {
        "env": RedLightRunnerEnv,
        "town": "Town05",
        "expert": expert_action_red_light,
        "data": "data/red_light_expert.npz",
        "coil": "checkpoints/red_light/coil.pt",
        "checkpoints": "checkpoints/red_light",
        "results": "results/red_light",
        "eval": "results/red_light/eval.json",
        "title": "red-light runner",
    },
    "left_turn": {
        "env": UnprotectedLeftEnv,
        "town": "Town05",
        "expert": expert_action_left_turn,
        "data": "data/left_turn_expert.npz",
        "coil": "checkpoints/left_turn/coil.pt",
        "checkpoints": "checkpoints/left_turn",
        "results": "results/left_turn",
        "eval": "results/left_turn/eval.json",
        "title": "unprotected left turn",
    },
}
