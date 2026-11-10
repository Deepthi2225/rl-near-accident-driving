"""
Evaluate driving policies on the same set of unseen scenarios.

    python scripts/evaluate.py --policies expert:aggressive expert:timid coil:aggressive coil:timid
    python scripts/evaluate.py --policies coil:timid --episodes 5 --render

Policy names:  expert:<aggressive|timid>,  coil:<aggressive|timid>,  ppo:<checkpoint path>
Each policy sees the same scenario seeds (disjoint from the data-collection seeds),
so the numbers are directly comparable. Results are also written to results/*.json.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch  # noqa: E402

from whew.coil import CoILNet, CoILPolicy  # noqa: E402
from whew.envs import PedestrianCrossingEnv  # noqa: E402
from whew.experts import COMMAND_NAMES, expert_action  # noqa: E402

COMMANDS = {v: k for k, v in COMMAND_NAMES.items()}
EVAL_SEED_OFFSET = 100_000  # data collection uses seeds 0..N
MAX_RETRIES = 5


def make_policy(spec, coil_path):
    kind, arg = spec.split(":", 1)
    if kind == "ppo":  # deterministic (mean) action of a trained PPO actor
        ckpt = torch.load(arg, map_location="cpu", weights_only=False)
        net = CoILNet(obs_dim=len(ckpt["obs_mean"]), n_commands=ckpt["n_commands"])
        net.load_state_dict(ckpt["actor"])
        pol = CoILPolicy(net, ckpt["obs_mean"], ckpt["obs_std"])
        return lambda obs, env: pol.act(obs, ckpt["command"])
    command = COMMANDS[arg]
    if kind == "expert":
        return lambda obs, env: expert_action(obs, command, env.lane_width)
    if kind == "coil":
        coil = CoILPolicy.load(coil_path)
        return lambda obs, env: coil.act(obs, command)
    raise ValueError(f"Unknown policy '{spec}'")


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    done, ret = False, 0.0
    while not done:
        obs, r, terminated, truncated, info = env.step([policy(obs, env)])
        ret += r
        done = terminated or truncated
    return info, ret


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--policies", nargs="+", default=["expert:aggressive", "expert:timid",
                                                     "coil:aggressive", "coil:timid"])
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--coil", default="checkpoints/coil.pt")
    p.add_argument("--town", default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out", default="results/eval_pedestrian_crossing.json")
    args = p.parse_args()

    make_env = lambda: PedestrianCrossingEnv(  # noqa: E731
        town=args.town, render_mode="human" if args.render else None, no_rendering=not args.render)
    env = make_env()
    results = {}
    try:
        for spec in args.policies:
            policy = make_policy(spec, args.coil)
            outcomes, times, dists, returns = Counter(), [], [], []
            for i in range(args.episodes):
                for attempt in range(MAX_RETRIES):
                    try:
                        info, ret = run_episode(env, policy, EVAL_SEED_OFFSET + i)
                        break
                    except RuntimeError as e:
                        print(f"  CARLA error ({e}); reconnecting", flush=True)
                        try:
                            env.close()
                        except RuntimeError:
                            pass
                        time.sleep(5)
                        env = make_env()
                else:
                    raise RuntimeError("CARLA keeps failing; is the simulator running?")
                outcomes[info["outcome"]] += 1
                dists.append(info["min_dist_to_ped"])
                returns.append(ret)
                if info["outcome"] == "success":
                    times.append(info["time"])
            n = args.episodes
            results[spec] = {
                "episodes": n,
                "success": outcomes["success"] / n,
                "collision": outcomes["collision"] / n,
                "timeout": outcomes["timeout"] / n,
                "off_road": outcomes["off_road"] / n,
                "mean_time_to_goal": float(np.mean(times)) if times else None,
                "mean_min_dist_to_ped": float(np.mean(dists)),
                "mean_return": float(np.mean(returns)),
            }
            r = results[spec]
            print(f"{spec:18s} success {r['success']:6.1%}  collision {r['collision']:6.1%}  "
                  f"timeout {r['timeout']:5.1%}  time {r['mean_time_to_goal'] or float('nan'):5.1f}s  "
                  f"min dist {r['mean_min_dist_to_ped']:4.1f}m  return {r['mean_return']:6.1f}", flush=True)
    finally:
        try:
            env.close()
        except RuntimeError:
            pass

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
