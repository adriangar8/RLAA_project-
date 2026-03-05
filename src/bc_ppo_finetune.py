"""
Experiment: Behavioral Cloning + PPO Fine-tuning

Pipeline:
Expert → Demonstrations → BC → PPO fine-tuning → Evaluation
"""

import os
import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.policy import PolicyNetwork
from src.expert import DemonstrationDataset
from src.bc import BehavioralCloning, BCConfig
from src.evaluate import evaluate_policy


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

ENV_ID = "LunarLander-v3"

DEMO_PATH = "data/demonstrations_200ep.npz"

PPO_TIMESTEPS = 500_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------------
# Helper: copy BC weights into SB3 PPO policy
# -------------------------------------------------------------

def transfer_bc_weights_to_ppo(bc_policy, ppo_model):
    """
    Copy BC network weights into the PPO policy network.
    """

    bc_state = bc_policy.state_dict()

    ppo_state = ppo_model.policy.state_dict()

    matched = []

    for name, param in ppo_state.items():
        for bc_name, bc_param in bc_state.items():
            if param.shape == bc_param.shape:
                ppo_state[name] = bc_param.detach().cpu()
                matched.append((name, bc_name))
                break

    ppo_model.policy.load_state_dict(ppo_state)

    print(f"Transferred {len(matched)} layers from BC → PPO")


# -------------------------------------------------------------
# Step 1: Load demonstrations
# -------------------------------------------------------------

print("\nLoading demonstrations...")
demo = DemonstrationDataset.load(DEMO_PATH)

obs_dim = demo.observations.shape[1]
act_dim = len(np.unique(demo.actions))


# -------------------------------------------------------------
# Step 2: Train Behavioral Cloning
# -------------------------------------------------------------

print("\nTraining Behavioral Cloning policy...")

policy = PolicyNetwork(obs_dim, act_dim)

bc_cfg = BCConfig()

bc_trainer = BehavioralCloning(policy, bc_cfg)

bc_history = bc_trainer.train(demo)


# -------------------------------------------------------------
# Step 3: Evaluate BC
# -------------------------------------------------------------

print("\nEvaluating BC policy...")

bc_results = evaluate_policy(policy, ENV_ID)

print(
    f"BC performance: {bc_results['mean']:.1f} ± {bc_results['std']:.1f}"
)


# -------------------------------------------------------------
# Step 4: Initialize PPO
# -------------------------------------------------------------

print("\nInitializing PPO with BC weights...")

env = DummyVecEnv([lambda: gym.make(ENV_ID)])

ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
)


# -------------------------------------------------------------
# Step 5: Transfer BC weights
# -------------------------------------------------------------

transfer_bc_weights_to_ppo(policy, ppo_model)


# -------------------------------------------------------------
# Step 6: PPO Fine-tuning
# -------------------------------------------------------------

print("\nStarting PPO fine-tuning...")

ppo_model.learn(total_timesteps=PPO_TIMESTEPS)


# -------------------------------------------------------------
# Step 7: Convert PPO → evaluation policy wrapper
# -------------------------------------------------------------

class PPOPolicyWrapper:
    def __init__(self, model):
        self.model = model

    def act(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)


ppo_policy = PPOPolicyWrapper(ppo_model)


# -------------------------------------------------------------
# Step 8: Evaluate PPO fine-tuned policy
# -------------------------------------------------------------

print("\nEvaluating BC + PPO policy...")

ppo_results = evaluate_policy(ppo_policy, ENV_ID)

print(
    f"PPO fine-tuned performance: {ppo_results['mean']:.1f} ± {ppo_results['std']:.1f}"
)


# -------------------------------------------------------------
# Save results
# -------------------------------------------------------------

os.makedirs("results", exist_ok=True)

np.save(
    "results/bc_ppo_results.npy",
    {
        "bc": bc_results,
        "ppo": ppo_results,
    },
)

print("\nResults saved to results/bc_ppo_results.npy")