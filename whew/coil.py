"""
Conditional imitation learning (CoIL) policy.

A shared trunk encodes the observation; one branch per command (0 = aggressive,
1 = timid) outputs the action in [-1, 1]. The command selects which branch is used,
so a single network can drive in either style.
"""
import numpy as np
import torch
import torch.nn as nn


class CoILNet(nn.Module):
    def __init__(self, obs_dim=7, n_commands=2, hidden=64, branch_hidden=32):
        super().__init__()
        self.n_commands = n_commands
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, branch_hidden), nn.ReLU(),
                          nn.Linear(branch_hidden, 1), nn.Tanh())
            for _ in range(n_commands)
        ])

    def forward(self, obs, command):
        """obs: (B, obs_dim) normalised; command: (B,) long. Returns (B, 1)."""
        h = self.trunk(obs)
        out = torch.stack([b(h) for b in self.branches], dim=1)  # (B, n_commands, 1)
        return out[torch.arange(len(command)), command]


class CoILPolicy:
    """Wraps a trained CoILNet with its observation normalisation for driving."""

    def __init__(self, net, obs_mean, obs_std):
        self.net = net.eval()
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_std = np.asarray(obs_std, dtype=np.float32)

    def normalize(self, obs):
        return (np.asarray(obs, dtype=np.float32) - self.obs_mean) / self.obs_std

    @torch.no_grad()
    def act(self, obs, command):
        x = torch.from_numpy(self.normalize(obs)).reshape(1, -1)
        return float(self.net(x, torch.tensor([command]))[0, 0])

    def save(self, path, **extra):
        torch.save({"state_dict": self.net.state_dict(), "obs_mean": self.obs_mean,
                    "obs_std": self.obs_std, "n_commands": self.net.n_commands,
                    "obs_dim": len(self.obs_mean), **extra}, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = CoILNet(obs_dim=ckpt["obs_dim"], n_commands=ckpt["n_commands"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, ckpt["obs_mean"], ckpt["obs_std"])
