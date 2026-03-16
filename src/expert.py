"""
Expert data generation.

Strategy: train a PPO expert via stable-baselines3, then roll it out
to collect (obs, action) demonstration trajectories.

This lets you control expert quality by using SB3 checkpoints at
different training stages — useful for the "expert quality" ablation.
"""

import os
import numpy as np
import gymnasium as gym
from pathlib import Path
from dataclasses import dataclass, field

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy


@dataclass
class ExpertConfig:
    env_id: str = "LunarLander-v3"
    total_timesteps: int = 500_000
    n_envs: int = 4
    save_dir: str = "checkpoints/expert"
    # Checkpoints to save during training (for quality ablation)
    checkpoint_freq: int = 50_000


def train_expert(cfg: ExpertConfig) -> PPO:
    """Train a PPO expert and save checkpoints at regular intervals."""
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(cfg.env_id, n_envs=cfg.n_envs)
    eval_env = make_vec_env(cfg.env_id, n_envs=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=cfg.checkpoint_freq // cfg.n_envs,
        save_path=cfg.save_dir,
        name_prefix="expert_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(cfg.save_dir, "best"),
        eval_freq=10_000 // cfg.n_envs,
        n_eval_episodes=20,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=os.path.join(cfg.save_dir, "tb_logs"),
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )
    model.save(os.path.join(cfg.save_dir, "expert_final"))

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50)
    print(f"Expert final performance: {mean_reward:.1f} ± {std_reward:.1f}")

    return model


def load_expert(checkpoint_path: str) -> PPO:
    """Load an expert from a checkpoint (used for quality ablation)."""
    return PPO.load(checkpoint_path)


@dataclass
class DemonstrationDataset:
    observations: np.ndarray    # (N, obs_dim)
    actions: np.ndarray         # (N,) integer actions
    episode_returns: list[float] = field(default_factory=list)

    def __len__(self):
        return len(self.observations)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, observations=self.observations, actions=self.actions,
                 episode_returns=self.episode_returns)
        print(f"Saved {len(self)} transitions to {path}")

    @classmethod
    def load(cls, path: str) -> "DemonstrationDataset":
        data = np.load(path, allow_pickle=True)
        return cls(
            observations=data["observations"],
            actions=data["actions"],
            episode_returns=list(data["episode_returns"]),
        )


def collect_demonstrations(
    expert: PPO,
    env_id: str,
    n_episodes: int,
    deterministic: bool = True,
    render: bool = False,
    action_noise_eps: float = 0.0,
) -> DemonstrationDataset:
    """
    Roll out the expert policy and collect (obs, action) pairs.

    Args:
        n_episodes: number of complete episodes to collect
        deterministic: if False, adds SB3's built-in stochasticity
        action_noise_eps: epsilon-greedy action noise probability.
            With this probability the recorded action is replaced by a
            random one, forcing the expert to visit slightly off-trajectory
            states and demonstrate recovery, a direct attack on covariate
            shift.  Only makes sense for discrete action spaces.
    """
    env = gym.make(env_id, render_mode="human" if render else None)

    all_obs, all_actions, episode_returns = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_obs, ep_actions = [], []
        ep_return = 0.0
        done = False

        while not done:
            optimal_action, _ = expert.predict(obs, deterministic=deterministic)
            if action_noise_eps > 0.0 and np.random.rand() < action_noise_eps:
                action = env.action_space.sample()
            else:
                action = optimal_action

            ep_obs.append(obs)
            ep_actions.append(int(optimal_action))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated

        all_obs.extend(ep_obs)
        all_actions.extend(ep_actions)
        episode_returns.append(ep_return)

        if (ep + 1) % 10 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes | "
                  f"avg return: {np.mean(episode_returns[-10:]):.1f}")

    env.close()

    dataset = DemonstrationDataset(
        observations=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        episode_returns=episode_returns,
    )
    print(f"\nDataset: {len(dataset)} transitions, "
          f"mean return: {np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}")
    return dataset