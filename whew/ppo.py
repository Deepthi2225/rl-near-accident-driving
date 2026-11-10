"""
PPO for WHEW!

Two policy types share one PPO implementation:
* ActorCritic: low-level continuous control. The actor is a CoILNet used with a fixed
  command, so PPO can start exactly from the imitation policy or from random weights.
  Gaussian policy: mean = network output in [-1, 1], state-independent learnable std.
* SwitchActorCritic: high-level mode switcher (as in H-ReIL). A categorical policy picks
  which frozen CoIL branch (aggressive / timid) drives next.
Advantages: GAE(lambda). Value function: separate MLP.
"""
import numpy as np
import torch
import torch.nn as nn

from .coil import CoILNet


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=7, n_commands=2, command=1, init_log_std=-1.0):
        super().__init__()
        self.actor = CoILNet(obs_dim=obs_dim, n_commands=n_commands)
        self.command = command
        self.log_std = nn.Parameter(torch.full((1,), init_log_std))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

    def mean(self, obs):
        cmd = torch.full((obs.shape[0],), self.command, dtype=torch.long)
        return self.actor(obs, cmd)

    def dist(self, obs):
        return torch.distributions.Normal(self.mean(obs), self.log_std.exp().expand(obs.shape[0], 1))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def sample(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return float(a[0, 0]), float(d.log_prob(a).sum())

    def evaluate(self, obs, act):
        d = self.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1)

    def actor_parameters(self):
        return list(self.actor.parameters()) + [self.log_std]


class SwitchActorCritic(nn.Module):
    """High-level policy: chooses a driving mode (CoIL command) from the observation."""

    def __init__(self, obs_dim=7, n_modes=2, hidden=64):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_modes))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.policy(obs))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def sample(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return float(a[0]), float(d.log_prob(a)[0])

    def evaluate(self, obs, act):
        d = self.dist(obs)
        return d.log_prob(act.reshape(-1).long()), d.entropy()

    @torch.no_grad()
    def greedy(self, obs):
        return int(self.policy(obs).argmax(-1)[0])

    def actor_parameters(self):
        return list(self.policy.parameters())


class RolloutBuffer:
    def __init__(self):
        self.obs, self.act, self.logp, self.rew, self.done, self.val = [], [], [], [], [], []

    def add(self, obs, act, logp, rew, done, val):
        for lst, x in zip((self.obs, self.act, self.logp, self.rew, self.done, self.val),
                          (obs, act, logp, rew, done, val)):
            lst.append(x)

    def __len__(self):
        return len(self.rew)

    def compute(self, last_val, gamma, lam):
        rew, done, val = map(lambda x: np.asarray(x, dtype=np.float32), (self.rew, self.done, self.val))
        adv = np.zeros_like(rew)
        gae, next_val = 0.0, last_val
        for t in reversed(range(len(rew))):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + gamma * next_val * nonterminal - val[t]
            gae = delta + gamma * lam * nonterminal * gae
            adv[t] = gae
            next_val = val[t]
        ret = adv + val
        return (torch.tensor(np.array(self.obs), dtype=torch.float32),
                torch.tensor(np.array(self.act), dtype=torch.float32).reshape(-1, 1),
                torch.tensor(np.array(self.logp), dtype=torch.float32),
                torch.from_numpy(adv), torch.from_numpy(ret))


class PPO:
    def __init__(self, model, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, lam=0.95, clip=0.2,
                 epochs=10, minibatch=256, entropy_coef=0.0):
        self.model = model
        self.opt_actor = torch.optim.Adam(model.actor_parameters(), lr=lr_actor)
        self.opt_critic = torch.optim.Adam(model.critic.parameters(), lr=lr_critic)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch, self.entropy_coef = epochs, minibatch, entropy_coef

    @torch.no_grad()
    def act(self, obs_norm):
        x = torch.as_tensor(obs_norm, dtype=torch.float32).reshape(1, -1)
        a, logp = self.model.sample(x)
        return a, logp, float(self.model.value(x)[0])

    @torch.no_grad()
    def value(self, obs_norm):
        return float(self.model.value(torch.as_tensor(obs_norm, dtype=torch.float32).reshape(1, -1))[0])

    def update(self, buf, last_val, train_actor=True):
        obs, act, logp_old, adv, ret = buf.compute(last_val, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "approx_kl": 0.0}
        n_batches = 0
        for _ in range(self.epochs):
            for idx in torch.randperm(len(obs)).split(self.minibatch):
                v_loss = ((self.model.value(obs[idx]) - ret[idx]) ** 2).mean()
                self.opt_critic.zero_grad()
                v_loss.backward()
                self.opt_critic.step()
                stats["value_loss"] += v_loss.item()
                if train_actor:
                    logp, entropy = self.model.evaluate(obs[idx], act[idx])
                    ratio = (logp - logp_old[idx]).exp()
                    p_loss = -torch.min(ratio * adv[idx],
                                        ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                    p_loss = p_loss - self.entropy_coef * entropy.mean()
                    self.opt_actor.zero_grad()
                    p_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.actor_parameters(), 0.5)
                    self.opt_actor.step()
                    stats["policy_loss"] += p_loss.item()
                    stats["approx_kl"] += (logp_old[idx] - logp).mean().item()
                n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in stats.items()}
