"""
scripts/sanity_check.py

Run this FIRST to verify the full pipeline works on your machine
before launching long experiments. Uses CartPole-v1 and a tiny
number of episodes for speed (< 2 minutes total).

Usage:
    python scripts/sanity_check.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy as sb3_eval

from src.policy import PolicyNetwork
from src.expert import collect_demonstrations, DemonstrationDataset
from src.bc import BCConfig, BehavioralCloning
from src.rl import REINFORCEConfig, REINFORCETrainer
from src.evaluate import evaluate_policy

ENV_ID = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")


# 1. Train a quick expert
print("=" * 50)
print("Step 1: Training expert (CartPole, 50k steps)...")
from stable_baselines3.common.env_util import make_vec_env
vec_env = make_vec_env(ENV_ID, n_envs=4)
expert = PPO("MlpPolicy", vec_env, verbose=0)
expert.learn(total_timesteps=50_000)
mean_r, std_r = sb3_eval(expert, gym.make(ENV_ID), n_eval_episodes=20)
print(f"Expert performance: {mean_r:.1f} ± {std_r:.1f}")
assert mean_r > 400, f"Expert too weak ({mean_r:.1f}), re-run."
print("✓ Expert OK\n")


# 2. Collect demonstrations
print("=" * 50)
print("Step 2: Collecting demonstrations (20 episodes)...")
demo = collect_demonstrations(expert, ENV_ID, n_episodes=20)
print(f"Dataset: {len(demo)} transitions")
print(f"✓ Demo collection OK\n")


# 3. BC training
print("=" * 50)
print("Step 3: Behavioral cloning (10 epochs)...")
obs_dim = gym.make(ENV_ID).observation_space.shape[0]
act_dim = gym.make(ENV_ID).action_space.n
policy = PolicyNetwork(obs_dim, act_dim, hidden_dims=[64, 64])
bc_cfg = BCConfig(n_epochs=10, batch_size=64, device=DEVICE, save_dir="/tmp/bc_test")
bc_trainer = BehavioralCloning(policy, bc_cfg)
bc_history = bc_trainer.train(demo)
print(f"Final val accuracy: {bc_history['val_acc'][-1]:.3f}")
assert bc_history["val_acc"][-1] > 0.7, "BC accuracy too low"
print("✓ BC OK\n")


# 4. REINFORCE fine-tuning (short)
print("=" * 50)
print("Step 4: REINFORCE fine-tuning (100 episodes)...")
rl_cfg = REINFORCEConfig(
    env_id=ENV_ID, n_episodes=100, eval_freq=50, eval_episodes=10,
    device=DEVICE, log_tensorboard=False, save_dir="/tmp/rl_test"
)
trainer = REINFORCETrainer(policy, rl_cfg)
rl_history = trainer.train()
print(f"Eval at episode 100: {rl_history['eval_mean'][-1]:.1f}")
print("✓ REINFORCE OK\n")


# 5. Evaluation
print("=" * 50)
print("Step 5: Final evaluation...")
stats = evaluate_policy(policy, ENV_ID, n_episodes=20, device=DEVICE)
print(f"Mean return: {stats['mean']:.1f} ± {stats['std']:.1f}")
print("✓ Evaluation OK\n")


print("=" * 50)
print("All checks passed! Pipeline is working correctly.")
print(f"Ready to run on LunarLander-v2 with full hyperparameters.")