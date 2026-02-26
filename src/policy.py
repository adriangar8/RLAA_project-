"""
Shared policy network used by both BC and RL fine-tuning.
Architecture: MLP with configurable depth and width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class PolicyNetwork(nn.Module):
    """
    Actor-only policy network (for BC + REINFORCE).
    Outputs a categorical distribution over discrete actions.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns raw logits."""
        return self.net(obs)

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action given a numpy observation."""
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.get_distribution(obs_t)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action.item()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Returns log_probs and entropy for a batch of (obs, action) pairs.
        Used by REINFORCE during training.
        """
        dist = self.get_distribution(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ActorCriticNetwork(nn.Module):
    """
    Shared-backbone actor-critic for A2C baseline comparison.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()

        # Shared trunk
        trunk_layers = []
        in_dim = obs_dim
        for h in hidden_dims[:-1]:
            trunk_layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        self.trunk = nn.Sequential(*trunk_layers)

        last_h = hidden_dims[-1]
        self.actor_head = nn.Sequential(nn.Linear(in_dim, last_h), nn.Tanh(), nn.Linear(last_h, act_dim))
        self.critic_head = nn.Sequential(nn.Linear(in_dim, last_h), nn.Tanh(), nn.Linear(last_h, 1))

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_distribution(self, obs: torch.Tensor) -> Categorical:
        features = self.trunk(obs)
        logits = self.actor_head(features)
        return Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            dist = self.get_distribution(obs_t)
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action.item()