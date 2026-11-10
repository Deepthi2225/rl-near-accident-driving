"""
Train the high-level mode switcher (as in H-ReIL, Cao et al. 2020).

Every --interval simulation steps (default 5 = 0.25 s) a PPO policy chooses which
frozen CoIL branch drives next: 0 = aggressive, 1 = timid. The CoIL network itself
is not changed; the switcher only learns *when* to be bold and when to be careful.

    python scripts/train_switcher.py --seed 0 --name ppo_switcher_s0

Writes checkpoints/<name>.pt (best by rolling mean return) and results/<name>.csv.
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whew.coil import CoILPolicy  # noqa: E402
from whew.envs import PedestrianCrossingEnv  # noqa: E402
from whew.experts import AGGRESSIVE  # noqa: E402
from whew.ppo import PPO, RolloutBuffer, SwitchActorCritic  # noqa: E402


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--coil", default="checkpoints/coil.pt")
    p.add_argument("--steps", type=int, default=100_000, help="simulation steps")
    p.add_argument("--interval", type=int, default=5, help="simulation steps per mode decision")
    p.add_argument("--rollout", type=int, default=512, help="mode decisions per PPO update")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--town", default=None)
    args = p.parse_args()
    name = args.name or f"ppo_switcher_s{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    coil = CoILPolicy.load(args.coil)
    model = SwitchActorCritic(obs_dim=len(coil.obs_mean), n_modes=coil.net.n_commands)
    ppo = PPO(model, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99 ** args.interval, entropy_coef=0.01)

    make_env = lambda: PedestrianCrossingEnv(town=args.town, no_rendering=True)  # noqa: E731
    holder = {"env": make_env()}

    def reconnect(err):
        print(f"  CARLA error ({err}); reconnecting", flush=True)
        try:
            holder["env"].close()
        except RuntimeError:
            pass
        time.sleep(5)
        holder["env"] = make_env()

    def safe_reset(seed):
        for _ in range(5):
            try:
                return holder["env"].reset(seed=seed)[0]
            except RuntimeError as e:
                reconnect(e)
        raise RuntimeError("CARLA keeps failing; is the simulator running?")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{name}.csv"
    log = open(csv_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["iteration", "steps", "episodes", "success", "collision", "timeout", "off_road",
                     "mean_time_to_goal", "mean_return", "aggressive_share", "value_loss", "approx_kl"])

    recent = deque(maxlen=50)  # (outcome, return, time, aggressive share)
    best = -float("inf")
    ep_seed = 2_000_000 * (args.seed + 1)
    obs = safe_reset(ep_seed)
    ep_ret, ep_modes, total_steps, episodes, it = 0.0, [], 0, 0, 0
    t_start = time.time()
    try:
        while total_steps < args.steps:
            it += 1
            buf = RolloutBuffer()
            while len(buf) < args.rollout:
                obs_n = coil.normalize(obs)
                mode, logp, v = ppo.act(obs_n)
                mode = int(mode)
                ep_modes.append(mode)
                macro_r, done, terminated, failed = 0.0, False, False, False
                for _ in range(args.interval):
                    try:
                        obs, r, terminated, truncated, info = holder["env"].step([coil.act(obs, mode)])
                    except RuntimeError as e:  # CARLA stall: end this episode and reconnect
                        reconnect(e)
                        failed = True
                        break
                    macro_r += r
                    total_steps += 1
                    done = terminated or truncated
                    if done:
                        break
                if failed:
                    if len(buf):
                        buf.done[-1] = 1.0
                    ep_seed += 1
                    obs, ep_ret, ep_modes = safe_reset(ep_seed), 0.0, []
                    continue
                buf.add(obs_n, float(mode), logp, macro_r, float(terminated), v)
                if done and not terminated:  # time limit: bootstrap
                    buf.rew[-1] += ppo.gamma * ppo.value(coil.normalize(obs))
                    buf.done[-1] = 1.0
                ep_ret += macro_r
                if done:
                    episodes += 1
                    recent.append((info["outcome"], ep_ret, info["time"],
                                   float(np.mean([m == AGGRESSIVE for m in ep_modes]))))
                    ep_seed += 1
                    obs, ep_ret, ep_modes = safe_reset(ep_seed), 0.0, []

            stats = ppo.update(buf, last_val=ppo.value(coil.normalize(obs)))
            outs = [o for o, _, _, _ in recent]
            n = max(len(outs), 1)
            succ_times = [t for o, _, t, _ in recent if o == "success"]
            row = [it, total_steps, episodes, outs.count("success") / n, outs.count("collision") / n,
                   outs.count("timeout") / n, outs.count("off_road") / n,
                   float(np.mean(succ_times)) if succ_times else float("nan"),
                   float(np.mean([g for _, g, _, _ in recent])) if recent else float("nan"),
                   float(np.mean([s for _, _, _, s in recent])) if recent else float("nan"),
                   stats["value_loss"], stats["approx_kl"]]
            writer.writerow(row)
            log.flush()
            print(f"it {it:3d} steps {total_steps:7d} eps {episodes:4d} | last {len(recent)}: "
                  f"success {row[3]:5.1%} collision {row[4]:5.1%} time {row[7]:5.1f}s return {row[8]:6.1f} "
                  f"| aggressive {row[9]:5.1%} of decisions | kl {row[11]:.4f} "
                  f"| {(time.time() - t_start) / 60:.1f} min", flush=True)
            if len(recent) == recent.maxlen and row[8] > best:
                best = row[8]
                torch.save({"policy": model.policy.state_dict(), "interval": args.interval,
                            "n_modes": coil.net.n_commands, "obs_dim": len(coil.obs_mean),
                            "coil": args.coil, "iteration": it, "rolling_return": best},
                           f"checkpoints/{name}.pt")
    finally:
        log.close()
        try:
            holder["env"].close()
        except RuntimeError:
            pass
    if best == -float("inf"):
        torch.save({"policy": model.policy.state_dict(), "interval": args.interval,
                    "n_modes": coil.net.n_commands, "obs_dim": len(coil.obs_mean),
                    "coil": args.coil, "iteration": it, "rolling_return": None}, f"checkpoints/{name}.pt")
    print(f"Done. Best rolling return {best:.1f}; model checkpoints/{name}.pt; curve {csv_path}")


if __name__ == "__main__":
    main()
