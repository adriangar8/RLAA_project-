"""
RL Fine-tuning: REINFORCE with baseline + A2C.

Both algorithms operate on the policy initialized from BC weights.
The key claim we're testing: BC init → faster convergence + better
final performance than random init.

REINFORCE update:
  θ ← θ + α (G_t - b(s_t)) ∇_θ log π_θ(a_t | s_t)
  where b(s_t) = V(s_t) is learned via a separate value head.

A2C update (online, synchronous):
  Actor:  ∇_θ J = E[A(s,a) ∇_θ log π_θ(a|s)]
  Critic: L_V  = (V(s) - G_t)²
  Entropy: -β H(π_θ)  (encourages exploration)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from dataclasses import dataclass, field
from pathlib import Path

from src.policy import PolicyNetwork, ActorCriticNetwork


# ---------------------------------------------------------------------------
# Shared value network (used by REINFORCE as baseline)
# ---------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int] = [256, 256]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class Rollout:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def compute_returns(self, gamma: float, normalize: bool = True) -> torch.Tensor:
        """Discounted returns G_t for each step."""
        returns = []
        G = 0.0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            G = r + gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        if normalize and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def clear(self):
        self.__init__()


# ---------------------------------------------------------------------------
# REINFORCE with learned baseline
# ---------------------------------------------------------------------------

@dataclass
class REINFORCEConfig:
    env_id: str = "LunarLander-v3"
    n_episodes: int = 2000          # training episodes
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01      # entropy bonus to prevent premature convergence
    max_grad_norm: float = 1.0
    eval_freq: int = 50             # evaluate every N episodes
    eval_episodes: int = 20
    save_dir: str = "checkpoints/reinforce"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_tensorboard: bool = True


class REINFORCETrainer:
    """
    REINFORCE with baseline. Works with PolicyNetwork (actor-only).
    A separate ValueNetwork is trained as the baseline b(s_t).
    """

    def __init__(self, policy: PolicyNetwork, cfg: REINFORCEConfig):
        self.policy = policy.to(cfg.device)
        self.cfg = cfg

        obs_dim = gym.make(cfg.env_id).observation_space.shape[0]
        self.value_net = ValueNetwork(obs_dim).to(cfg.device)

        self.actor_opt = optim.Adam(policy.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.value_net.parameters(), lr=cfg.critic_lr)

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        self.writer = None
        if cfg.log_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, "tb"))

    def train(self) -> dict:
        cfg = self.cfg
        env = gym.make(cfg.env_id)
        history = {"episode_returns": [], "eval_mean": [], "eval_std": [], "eval_at": []}

        for episode in range(cfg.n_episodes):
            rollout = self._collect_episode(env)
            returns = rollout.compute_returns(cfg.gamma, normalize=True)

            obs_t = torch.FloatTensor(rollout.observations).to(cfg.device)
            actions_t = torch.LongTensor(rollout.actions).to(cfg.device)
            log_probs_t = torch.stack(rollout.log_probs).to(cfg.device)

            # --- Critic update ---
            values = self.value_net(obs_t)
            critic_loss = nn.functional.mse_loss(values, returns.to(cfg.device))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), cfg.max_grad_norm)
            self.critic_opt.step()

            # --- Actor update ---
            with torch.no_grad():
                baseline = self.value_net(obs_t)
            advantages = returns.to(cfg.device) - baseline

            log_probs_new, entropy = self.policy.evaluate_actions(obs_t, actions_t)

            actor_loss = -(log_probs_new * advantages).mean()
            entropy_loss = -cfg.entropy_coef * entropy.mean()
            total_actor_loss = actor_loss + entropy_loss

            self.actor_opt.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.actor_opt.step()

            ep_return = sum(rollout.rewards)
            history["episode_returns"].append(ep_return)

            if self.writer:
                self.writer.add_scalar("train/episode_return", ep_return, episode)
                self.writer.add_scalar("train/actor_loss", actor_loss.item(), episode)
                self.writer.add_scalar("train/critic_loss", critic_loss.item(), episode)
                self.writer.add_scalar("train/entropy", entropy.mean().item(), episode)

            # --- Periodic evaluation ---
            if (episode + 1) % cfg.eval_freq == 0:
                mean_r, std_r = self._evaluate()
                history["eval_mean"].append(mean_r)
                history["eval_std"].append(std_r)
                history["eval_at"].append(episode + 1)
                print(f"Episode {episode+1:5d}/{cfg.n_episodes} | "
                      f"train return: {np.mean(history['episode_returns'][-cfg.eval_freq:]):7.1f} | "
                      f"eval: {mean_r:.1f} ± {std_r:.1f}")
                if self.writer:
                    self.writer.add_scalar("eval/mean_return", mean_r, episode)

        env.close()
        self.save("reinforce_final.pt")
        return history

    def _collect_episode(self, env: gym.Env) -> Rollout:
        obs, _ = env.reset()
        rollout = Rollout()
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.cfg.device)
            dist = self.policy.get_distribution(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rollout.observations.append(obs)
            rollout.actions.append(action.item())
            rollout.rewards.append(reward)
            rollout.log_probs.append(log_prob.squeeze(0))
            rollout.dones.append(float(done))

            obs = next_obs

        return rollout

    def _evaluate(self) -> tuple[float, float]:
        eval_env = gym.make(self.cfg.env_id)
        returns = []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            ep_return = 0.0
            done = False
            while not done:
                action = self.policy.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                ep_return += reward
                done = terminated or truncated
            returns.append(ep_return)
        eval_env.close()
        return float(np.mean(returns)), float(np.std(returns))

    def save(self, name: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
        }, os.path.join(self.cfg.save_dir, name))


# ---------------------------------------------------------------------------
# A2C (online, synchronous) — used as comparison method
# ---------------------------------------------------------------------------

@dataclass
class A2CConfig:
    env_id: str = "LunarLander-v3"
    total_steps: int = 1_000_000
    n_steps: int = 5               # steps per update (online TD)
    gamma: float = 0.99
    lr: float = 7e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 10_000
    eval_episodes: int = 20
    save_dir: str = "checkpoints/a2c"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class A2CTrainer:
    """
    Online A2C with n-step returns.
    Uses ActorCriticNetwork (shared backbone).
    Can be initialized from a BC-pretrained actor.
    """

    def __init__(self, ac_net: ActorCriticNetwork, cfg: A2CConfig):
        self.ac = ac_net.to(cfg.device)
        self.cfg = cfg
        self.optimizer = optim.RMSprop(ac_net.parameters(), lr=cfg.lr, eps=1e-5)
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    def train(self) -> dict:
        cfg = self.cfg
        env = gym.make(cfg.env_id)
        history = {"step": [], "eval_mean": [], "eval_std": []}

        obs, _ = env.reset()
        total_steps = 0

        while total_steps < cfg.total_steps:
            # Collect n_steps
            obs_list, act_list, rew_list, val_list, logp_list, done_list = [], [], [], [], [], []

            for _ in range(cfg.n_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    logits, value = self.ac(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()

                next_obs, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                obs_list.append(obs)
                act_list.append(action.item())
                rew_list.append(reward)
                val_list.append(value.item())
                logp_list.append(dist.log_prob(action).item())
                done_list.append(float(done))

                obs = next_obs
                total_steps += 1
                if done:
                    obs, _ = env.reset()

            # Bootstrap
            next_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(cfg.device)
            with torch.no_grad():
                _, next_value = self.ac(next_obs_t)
            next_value = next_value.item() * (1 - done_list[-1])

            # Compute n-step returns
            returns = []
            G = next_value
            for r, d in zip(reversed(rew_list), reversed(done_list)):
                G = r + cfg.gamma * G * (1 - d)
                returns.insert(0, G)

            obs_t = torch.FloatTensor(np.array(obs_list)).to(cfg.device)
            act_t = torch.LongTensor(act_list).to(cfg.device)
            ret_t = torch.FloatTensor(returns).to(cfg.device)

            logits, values = self.ac(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            advantages = ret_t - values
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = cfg.value_coef * advantages.pow(2).mean()
            entropy_loss = -cfg.entropy_coef * entropy
            loss = actor_loss + critic_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            if total_steps % cfg.eval_freq < cfg.n_steps:
                mean_r, std_r = self._evaluate()
                history["step"].append(total_steps)
                history["eval_mean"].append(mean_r)
                history["eval_std"].append(std_r)
                print(f"Step {total_steps:7d} | eval: {mean_r:.1f} ± {std_r:.1f}")

        env.close()
        return history

    def _evaluate(self) -> tuple[float, float]:
        eval_env = gym.make(self.cfg.env_id)
        returns = []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = eval_env.reset()
            ep_return, done = 0.0, False
            while not done:
                action = self.ac.act(obs, deterministic=True)
                obs, r, te, tr, _ = eval_env.step(action)
                ep_return += r
                done = te or tr
            returns.append(ep_return)
        eval_env.close()
        return float(np.mean(returns)), float(np.std(returns))