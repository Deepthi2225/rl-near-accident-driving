"""
Collect expert demonstrations (and report each expert's performance).

    python scripts/collect_expert.py --episodes 100              # 100 per expert, saves data/
    python scripts/collect_expert.py --episodes 3 --render --no-save   # quick visual check

Data is saved after every episode. If CARLA stalls, the unfinished episode is
dropped, the script reconnects and continues.
"""
import argparse
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whew.experts import COMMAND_NAMES  # noqa: E402
from whew.scenarios import SCENARIOS  # noqa: E402

MAX_RETRIES = 5


def make_env(args):
    env = SCENARIOS[args.scenario]["env"](town=args.town or SCENARIOS[args.scenario]["town"], render_mode="human" if args.render else None,
                                 no_rendering=not args.render)
    env.expert_fn = SCENARIOS[args.scenario]["expert"]
    return env


def run_episode(env, command, seed):
    rows = []
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        a = env.expert_fn(obs, command, env.lane_width)
        rows.append((obs, a))
        obs, _, terminated, truncated, info = env.step([a])
        done = terminated or truncated
    return rows, info


def save(path, obs, act, cmd, ep):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, obs=np.array(obs, dtype=np.float32), action=np.array(act, dtype=np.float32),
                        command=np.array(cmd, dtype=np.int64), episode=np.array(ep, dtype=np.int64))
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--episodes", type=int, default=100, help="episodes per expert")
    p.add_argument("--town", type=str, default=None, help="e.g. Town04 (default: currently loaded map)")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="pedestrian")
    p.add_argument("--out", type=str, default=None, help="default: the scenario's data file")
    p.add_argument("--render", action="store_true", help="spectator camera follows the ego car")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out = args.out or SCENARIOS[args.scenario]["data"]

    env = make_env(args)
    print(f"Map {env.map.name}: {len(env.routes)} candidate straight routes", flush=True)
    obs_rows, act_rows, cmd_rows, ep_rows = [], [], [], []
    stats = {c: Counter() for c in COMMAND_NAMES}
    times = {c: [] for c in COMMAND_NAMES}
    min_dists = {c: [] for c in COMMAND_NAMES}
    try:
        for ep in range(args.episodes * len(COMMAND_NAMES)):
            command = ep % len(COMMAND_NAMES)
            for attempt in range(MAX_RETRIES):
                try:
                    rows, info = run_episode(env, command, args.seed + ep + 10000 * attempt)
                    break
                except RuntimeError as e:  # CARLA time-out: reconnect and redo this episode
                    print(f"ep {ep:3d} CARLA error ({e}); reconnecting (attempt {attempt + 1}/{MAX_RETRIES})",
                          flush=True)
                    try:
                        env.close()
                    except RuntimeError:
                        pass
                    time.sleep(5)
                    env = make_env(args)
            else:
                raise RuntimeError(f"Episode {ep} failed {MAX_RETRIES} times; is CARLA still running?")

            for o, a in rows:
                obs_rows.append(o)
                act_rows.append(a)
                cmd_rows.append(command)
                ep_rows.append(ep)
            stats[command][info["outcome"]] += 1
            min_dists[command].append(info["min_dist"])
            if info["outcome"] == "success":
                times[command].append(info["time"])
            print(f"ep {ep:3d} {COMMAND_NAMES[command]:10s} "
                  f"-> {info['outcome']:9s} t={info['time']:.1f}s min_dist={info['min_dist']:.1f}m",
                  flush=True)
            if not args.no_save:
                save(args.out, obs_rows, act_rows, cmd_rows, ep_rows)
    finally:
        try:
            env.close()
        except RuntimeError:
            pass

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
        print(f"Saved {len(obs_rows)} samples ({ep_rows[-1] + 1 if ep_rows else 0} episodes) to {args.out}")


if __name__ == "__main__":
    main()
