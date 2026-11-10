"""
Train a driving policy with PPO, starting from the CoIL policy or from scratch.

    python scripts/train_ppo.py --init coil --command timid --name ppo_from_coil
    python scripts/train_ppo.py --init scratch --name ppo_scratch          # baseline

Writes checkpoints/<name>.pt (best by rolling mean return) and a learning curve to
results/<name>.csv (one row per PPO iteration).
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
from whew.experts import COMMAND_NAMES  # noqa: E402
from whew.ppo import PPO, ActorCritic, RolloutBuffer  # noqa: E402

COMMANDS = {v: k for k, v in COMMAND_NAMES.items()}


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--init", choices=["coil", "scratch"], default="coil")
    p.add_argument("--command", choices=list(COMMANDS), default="timid",
                   help="CoIL branch to fine-tune (also the branch used from scratch)")
    p.add_argument("--coil", default="checkpoints/coil.pt", help="also provides observation normalisation")
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--critic-warmup", type=int, default=3,
                   help="iterations that train only the critic (protects the CoIL actor early on)")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--town", default=None)
    args = p.parse_args()
    name = args.name or f"ppo_{args.init}_{args.command}_s{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    coil = CoILPolicy.load(args.coil)
    command = COMMANDS[args.command]
    model = ActorCritic(obs_dim=len(coil.obs_mean), n_commands=coil.net.n_commands, command=command,
                        init_log_std=-1.0 if args.init == "coil" else -0.5)
    if args.init == "coil":
        model.actor.load_state_dict(coil.net.state_dict())
    warmup = args.critic_warmup if args.init == "coil" else 0
    ppo = PPO(model, lr_actor=1e-4 if args.init == "coil" else 3e-4)

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
                     "mean_time_to_goal", "mean_return", "std", "value_loss", "approx_kl"])

    recent = deque(maxlen=50)  # (outcome, return, time)
    best = -float("inf")
    ep_seed = 1_000_000 * (args.seed + 1)
    obs = safe_reset(ep_seed)
    obs_n, ep_ret, total_steps, episodes, it = coil.normalize(obs), 0.0, 0, 0, 0
    t_start = time.time()
    try:
        while total_steps < args.steps:
            it += 1
            buf = RolloutBuffer()
            it_eps = []
            while len(buf) < args.rollout:
                a, logp, v = ppo.act(obs_n)
                try:
                    obs, r, terminated, truncated, info = holder["env"].step([a])
                except RuntimeError as e:  # CARLA stall: end the episode here and reconnect
                    if len(buf):
                        buf.done[-1] = 1.0
                    reconnect(e)
                    ep_seed += 1
                    obs = safe_reset(ep_seed)
                    obs_n, ep_ret = coil.normalize(obs), 0.0
                    continue
                done = terminated or truncated
                buf.add(obs_n, a, logp, r, float(terminated), v)
                if truncated and not terminated:  # bootstrap time-limit cut-offs
                    buf.rew[-1] += ppo.gamma * ppo.value(coil.normalize(obs))
                    buf.done[-1] = 1.0
                ep_ret += r
                total_steps += 1
                obs_n = coil.normalize(obs)
                if done:
                    episodes += 1
                    recent.append((info["outcome"], ep_ret, info["time"]))
                    it_eps.append(info["outcome"])
                    ep_seed += 1
                    obs = safe_reset(ep_seed)
                    obs_n, ep_ret = coil.normalize(obs), 0.0

            stats = ppo.update(buf, last_val=ppo.value(obs_n), train_actor=it > warmup)
            outs = [o for o, _, _ in recent]
            n = max(len(outs), 1)
            succ_times = [t for o, _, t in recent if o == "success"]
            row = [it, total_steps, episodes, outs.count("success") / n, outs.count("collision") / n,
                   outs.count("timeout") / n, outs.count("off_road") / n,
                   float(np.mean(succ_times)) if succ_times else float("nan"),
                   float(np.mean([g for _, g, _ in recent])) if recent else float("nan"),
                   model.log_std.exp().item(), stats["value_loss"], stats["approx_kl"]]
            writer.writerow(row)
            log.flush()
            print(f"it {it:3d} steps {total_steps:7d} eps {episodes:4d} | last {len(recent)}: "
                  f"success {row[3]:5.1%} collision {row[4]:5.1%} time {row[7]:5.1f}s return {row[8]:6.1f} "
                  f"| std {row[9]:.2f} kl {row[11]:.4f} {'(critic warm-up)' if it <= warmup else ''} "
                  f"| {(time.time() - t_start) / 60:.1f} min", flush=True)
            if len(recent) == recent.maxlen and it > warmup and row[8] > best:
                best = row[8]
                torch.save({"actor": model.actor.state_dict(), "command": command,
                            "obs_mean": coil.obs_mean, "obs_std": coil.obs_std,
                            "n_commands": coil.net.n_commands, "init": args.init,
                            "iteration": it, "rolling_return": best}, f"checkpoints/{name}.pt")
    finally:
        log.close()
        try:
            holder["env"].close()
        except RuntimeError:
            pass
    if best == -float("inf"):  # run too short for the 50-episode window: keep the final policy
        torch.save({"actor": model.actor.state_dict(), "command": command, "obs_mean": coil.obs_mean,
                    "obs_std": coil.obs_std, "n_commands": coil.net.n_commands, "init": args.init,
                    "iteration": it, "rolling_return": None}, f"checkpoints/{name}.pt")
    print(f"Done. Best rolling return {best:.1f}; model checkpoints/{name}.pt; curve {csv_path}")


if __name__ == "__main__":
    main()
