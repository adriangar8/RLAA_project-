"""
Experiment: BC + REINFORCE fine-tuning with KL Penalty

Hypothesis: Adding a KL divergence penalty forcing the RL policy to stay close
to the original BC policy prevents catastrophic forgetting and divergence, improving
stability especially in early RL training steps.

Pipeline:
  Expert demos (N=500) → BC ──────────────────────┐
                       → REINFORCE (no KL)       → Eval + compare
                       → REINFORCE (w/ KL)       → Eval + compare
"""

import os
import copy
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.policy import PolicyNetwork
from src.expert import DemonstrationDataset
from src.bc import BehavioralCloning, BCConfig
from src.rl import REINFORCETrainer, REINFORCEConfig
from src.evaluate import evaluate_policy, measure_covariate_shift

# -------------------------------------------------------
# Config
# -------------------------------------------------------

ENV_ID = "LunarLander-v3"
DEMO_PATH = "data/demonstrations_200ep.npz"
OBS_DIM = 8
ACT_DIM = 4
N_DEMOS = 500
N_SEEDS = 3
REINFORCE_EPISODES = 1000
KL_COEFS = [0.0, 0.01, 0.1, 0.5, 1.0]  # Beta values for KL penalty ablation

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------


def subsample_demo(
    demo: DemonstrationDataset, n: int, rng: np.random.Generator
) -> DemonstrationDataset:
    idx = rng.choice(len(demo.observations), size=n, replace=False)
    sub = DemonstrationDataset.__new__(DemonstrationDataset)
    sub.observations = demo.observations[idx]
    sub.actions = demo.actions[idx]
    sub.episode_returns = demo.episode_returns
    return sub


def train_bc(demo, seed: int) -> PolicyNetwork:
    torch.manual_seed(seed)
    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    cfg = BCConfig(noise_std=0.0)
    trainer = BehavioralCloning(policy, cfg)
    trainer.train(demo, save_as=f"kl_best_bc_seed_{seed}.pt")
    return policy


def run_reinforce(policy: PolicyNetwork, seed: int, kl_coef: float) -> dict:
    torch.manual_seed(seed)
    cfg = REINFORCEConfig(
        env_id=ENV_ID,
        n_episodes=REINFORCE_EPISODES,
        log_tensorboard=False,
        kl_coef=kl_coef,
    )
    trainer = REINFORCETrainer(policy, cfg)
    history = trainer.train()
    return history


# -------------------------------------------------------
# Main experiment
# -------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"KL Penalty Experiment (N={N_DEMOS} demos, {N_SEEDS} seeds)")
    print(f"{'='*60}\n")

    print("Loading demonstrations...")
    full_demo = DemonstrationDataset.load(DEMO_PATH)
    print(f"  Full pool: {len(full_demo.observations)} transitions\n")

    results = {
        kl: {"rl_curves": [], "rl_final_mean": [], "rl_final_std": []}
        for kl in KL_COEFS
    }
    results["eval_at"] = None

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")
        rng = np.random.default_rng(seed)

        demo = subsample_demo(full_demo, N_DEMOS, rng)

        print(f"  Training Initial BC...")
        base_policy = train_bc(demo, seed=seed)

        for kl in KL_COEFS:
            label = f"kl_{kl}"
            print(f"  [{label}] REINFORCE fine-tuning (KL={kl})...")
            policy_copy = copy.deepcopy(base_policy)
            history = run_reinforce(policy_copy, seed=seed, kl_coef=kl)

            eval_returns = history.get("eval_mean", [])
            eval_at = history.get("eval_at", [])
            results[kl]["rl_curves"].append(eval_returns)
            if results["eval_at"] is None and eval_at:
                results["eval_at"] = eval_at

            final_mean = np.mean(eval_returns[-3:]) if len(eval_returns) >= 3 else 0.0
            final_std = np.std(eval_returns[-3:]) if len(eval_returns) >= 3 else 0.0
            results[kl]["rl_final_mean"].append(final_mean)
            results[kl]["rl_final_std"].append(final_std)
            print(f"  [{label}] REINFORCE final: {final_mean:.1f} ± {final_std:.1f}")

    print(f"\n{'='*60}")
    print("Summary")
    for kl in KL_COEFS:
        label = f"kl_{kl}"
        print(f"\n{label.upper()}:")
        print(f"  REINFORCE final: {np.mean(results[kl]['rl_final_mean']):.1f} ± {np.std(results[kl]['rl_final_mean']):.1f}")
