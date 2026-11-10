"""
Train the conditional imitation learning policy on expert data.

    python scripts/train_coil.py      # data/pedestrian_crossing_expert.npz -> checkpoints/coil.pt

Episodes (not individual samples) are split into train / validation sets, so
validation measures generalisation to unseen scenarios.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from whew.coil import CoILNet, CoILPolicy  # noqa: E402
from whew.experts import COMMAND_NAMES  # noqa: E402


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", default="data/pedestrian_crossing_expert.npz")
    p.add_argument("--out", default="checkpoints/coil.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = np.load(args.data)
    obs, act, cmd, ep = d["obs"], d["action"].reshape(-1, 1), d["command"], d["episode"]

    episodes = np.unique(ep)
    val_eps = rng.choice(episodes, size=max(1, int(len(episodes) * args.val_frac)), replace=False)
    val = np.isin(ep, val_eps)
    print(f"{len(obs)} samples from {len(episodes)} episodes: "
          f"{(~val).sum()} train / {val.sum()} validation ({len(val_eps)} held-out episodes)")

    obs_mean = obs[~val].mean(0)
    obs_std = obs[~val].std(0) + 1e-6
    norm = lambda o: torch.from_numpy((o - obs_mean) / obs_std).float()  # noqa: E731
    xt, yt, ct = norm(obs[~val]), torch.from_numpy(act[~val]), torch.from_numpy(cmd[~val])
    xv, yv, cv = norm(obs[val]), torch.from_numpy(act[val]), torch.from_numpy(cmd[val])

    net = CoILNet(obs_dim=obs.shape[1], n_commands=len(COMMAND_NAMES))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_val, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            loss = loss_fn(net(xt[idx], ct[idx]), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv, cv), yv).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train MSE {total / len(xt):.4f}  val MSE {val_loss:.4f}", flush=True)

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        for c, name in COMMAND_NAMES.items():
            m = cv == c
            print(f"  val MSE [{name}]: {loss_fn(net(xv[m], cv[m]), yv[m]).item():.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    CoILPolicy(net, obs_mean, obs_std).save(args.out, best_val_mse=best_val)
    print(f"Saved best model (val MSE {best_val:.4f}) to {args.out}")


if __name__ == "__main__":
    main()
