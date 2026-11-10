"""
Collect expert demonstrations (and report each expert's performance).

    python scripts/collect_expert.py --episodes 100              # 100 per expert, saves data/
    python scripts/collect_expert.py --episodes 3 --render --no-save   # quick visual check
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whew.envs import PedestrianCrossingEnv  # noqa: E402
from whew.experts import COMMAND_NAMES, expert_action  # noqa: E402


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=100, help="episodes per expert")
    p.add_argument("--town", type=str, default=None, help="e.g. Town04 (default: currently loaded map)")
    p.add_argument("--out", type=str, default="data/pedestrian_crossing_expert.npz")
    p.add_argument("--render", action="store_true", help="spectator camera follows the ego car")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = PedestrianCrossingEnv(town=args.town, render_mode="human" if args.render else None,
                                no_rendering=not args.render)
    print(f"Map {env.map.name}: {len(env.routes)} candidate straight routes")
    obs_rows, act_rows, cmd_rows, ep_rows = [], [], [], []
    stats = {c: Counter() for c in COMMAND_NAMES}
    times = {c: [] for c in COMMAND_NAMES}
    min_dists = {c: [] for c in COMMAND_NAMES}
    try:
        ep = 0
        for i in range(args.episodes * len(COMMAND_NAMES)):
            command = i % len(COMMAND_NAMES)
            obs, _ = env.reset(seed=args.seed + i)
            done = False
            while not done:
                a = expert_action(obs, command, env.lane_width)
                obs_rows.append(obs)
                act_rows.append(a)
                cmd_rows.append(command)
                ep_rows.append(ep)
                obs, _, terminated, truncated, info = env.step([a])
                done = terminated or truncated
            stats[command][info["outcome"]] += 1
            min_dists[command].append(info["min_dist_to_ped"])
            if info["outcome"] == "success":
                times[command].append(info["time"])
            print(f"ep {ep:3d} {COMMAND_NAMES[command]:10s} ped_crosses={info['ped_crossed']!s:5s} "
                  f"-> {info['outcome']:9s} t={info['time']:.1f}s min_dist={info['min_dist_to_ped']:.1f}m",
                  flush=True)
            ep += 1
    finally:
        env.close()

    print("\nExpert performance")
    for c, name in COMMAND_NAMES.items():
        n = sum(stats[c].values())
        if not n:
            continue
        print(f"  {name:10s} success {stats[c]['success'] / n:5.1%}  collision {stats[c]['collision'] / n:5.1%}  "
              f"timeout {stats[c]['timeout'] / n:5.1%}  off_road {stats[c]['off_road'] / n:5.1%}  "
              f"mean time {np.mean(times[c]) if times[c] else float('nan'):.1f}s  "
              f"mean min dist {np.mean(min_dists[c]):.1f}m")

    if not args.no_save:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.savez_compressed(args.out, obs=np.array(obs_rows, dtype=np.float32),
                            action=np.array(act_rows, dtype=np.float32),
                            command=np.array(cmd_rows, dtype=np.int64),
                            episode=np.array(ep_rows, dtype=np.int64))
        print(f"Saved {len(obs_rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
