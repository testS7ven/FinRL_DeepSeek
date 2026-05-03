"""Actor-Critic MLP for PPO/CPPO."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(sizes: list[int], activation=nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        self.pi_net  = _mlp([obs_dim] + hidden_sizes + [act_dim])
        self.v_net   = _mlp([obs_dim] + hidden_sizes + [1])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def _dist(self, obs: torch.Tensor) -> Normal:
        mean = torch.tanh(self.pi_net(obs))             # ∈ (-1, 1) → matches action space
        std  = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        return self._dist(obs), self.v_net(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t        = torch.as_tensor(obs, dtype=torch.float32)
        dist, v      = self(obs_t)
        a            = dist.sample()
        a_clipped    = torch.clamp(a, -1.0, 1.0)
        logp         = dist.log_prob(a).sum()
        return a_clipped.numpy(), float(v), float(logp)

    def evaluate(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, v  = self(obs)
        logp     = dist.log_prob(act).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return logp, v, entropy
