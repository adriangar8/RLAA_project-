"""
experiments/run_ablations.py

Runs all ablation studies for the project report:

  1. Main comparison: BC only | RL from scratch | BC + REINFORCE | BC + A2C
  2. Demo size ablation: effect of N ∈ {100, 500, 1000, 2000} demonstrations
  3. Expert quality ablation: trained for 50k / 200k / 500k steps
  4. Covariate shift analysis: distribution divergence for BC vs BC+RL

Results are saved to results/ as .npz files and plots are saved to figures/.

Usage:
    python -m experiments.run_ablations --experiment main
    python -m experiments.run_ablations --experiment demo_size
    python -m experiments.run_ablations --experiment expert_quality
    python -m experiments.run_ablations --experiment all
"""

import os
import copy
import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path
from dataclasses import asdict

from src.policy import PolicyNetwork
from src.expert import ExpertConfig, train_expert, load_expert, collect_demonstrations, DemonstrationDataset
from src.bc import BCConfig, BehavioralCloning
from src.rl import REINFORCEConfig, REINFORCETrainer
from src.evaluate import evaluate_policy, plot_learning_curves, plot_demo_size_ablation


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_ID = "LunarLander-v2"
N_SEEDS = 5
OBS_DIM = 8    # LunarLander observation space
ACT_DIM = 4    # LunarLander action space
THRESHOLD = 200.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: build a fresh policy
# ---------------------------------------------------------------------------

def make_policy(seed: int = 0) -> PolicyNetwork:
    torch.manual_seed(seed)
    return PolicyNetwork(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dims=[256, 256])


# ---------------------------------------------------------------------------
# Experiment 1: Main comparison (4 conditions × 5 seeds)
# ---------------------------------------------------------------------------

def run_main_comparison(demo: DemonstrationDataset):
    """
    Compare:
      - BC only
      - RL from scratch (random init)
      - BC + REINFORCE
    across N_SEEDS seeds.
    """
    conditions = ["rl_scratch", "bc_only", "bc_reinforce"]
    all_results = {c: {"eval_at": None, "eval_mean_seeds": [], "eval_std_seeds": []} for c in conditions}

    rl_cfg = REINFORCEConfig(env_id=ENV_ID, n_episodes=2000, device=DEVICE,
                              eval_freq=50, eval_episodes=20, log_tensorboard=False)
    bc_cfg = BCConfig(n_epochs=50, device=DEVICE)

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}")
        print(f"SEED {seed+1}/{N_SEEDS}")
        print(f"{'='*60}")

        # ---- RL from scratch ----
        print("\n[1/3] RL from scratch...")
        policy_scratch = make_policy(seed)
        trainer = REINFORCETrainer(policy_scratch, copy.copy(rl_cfg))
        h = trainer.train()
        all_results["rl_scratch"]["eval_mean_seeds"].append(h["eval_mean"])
        all_results["rl_scratch"]["eval_std_seeds"].append(h["eval_std"])
        if all_results["rl_scratch"]["eval_at"] is None:
            all_results["rl_scratch"]["eval_at"] = h["eval_at"]

        # ---- BC only ----
        print("\n[2/3] BC only...")
        policy_bc = make_policy(seed)
        bc_trainer = BehavioralCloning(policy_bc, copy.copy(bc_cfg))
        bc_trainer.train(demo)
        # Evaluate BC policy without fine-tuning
        stats = evaluate_policy(policy_bc, ENV_ID, n_episodes=50, device=DEVICE)
        # Replicate across eval_at steps for consistent plotting
        bc_eval_mean = [stats["mean"]] * len(h["eval_at"])
        bc_eval_std = [stats["std"]] * len(h["eval_at"])
        all_results["bc_only"]["eval_mean_seeds"].append(bc_eval_mean)
        all_results["bc_only"]["eval_std_seeds"].append(bc_eval_std)
        if all_results["bc_only"]["eval_at"] is None:
            all_results["bc_only"]["eval_at"] = h["eval_at"]

        # ---- BC + REINFORCE ----
        print("\n[3/3] BC + REINFORCE...")
        policy_bcrl = make_policy(seed)
        bc_trainer2 = BehavioralCloning(policy_bcrl, copy.copy(bc_cfg))
        bc_trainer2.train(demo)

        trainer2 = REINFORCETrainer(policy_bcrl, copy.copy(rl_cfg))
        h2 = trainer2.train()
        all_results["bc_reinforce"]["eval_mean_seeds"].append(h2["eval_mean"])
        all_results["bc_reinforce"]["eval_std_seeds"].append(h2["eval_std"])
        if all_results["bc_reinforce"]["eval_at"] is None:
            all_results["bc_reinforce"]["eval_at"] = h2["eval_at"]

    # Aggregate across seeds: mean of means, mean of stds
    plot_data = {}
    for c in conditions:
        means = np.array(all_results[c]["eval_mean_seeds"])    # (seeds, eval_steps)
        stds  = np.array(all_results[c]["eval_std_seeds"])
        plot_data[c] = {
            "eval_at":   all_results[c]["eval_at"],
            "eval_mean": means.mean(axis=0).tolist(),
            "eval_std":  stds.mean(axis=0).tolist(),
        }

    np.save("results/main_comparison.npy", plot_data)
    plot_learning_curves(
        plot_data,
        threshold=THRESHOLD,
        title="Main Comparison: BC / RL Scratch / BC+REINFORCE",
        save_path="figures/main_comparison.png",
    )
    return plot_data


# ---------------------------------------------------------------------------
# Experiment 2: Demo size ablation
# ---------------------------------------------------------------------------

def run_demo_size_ablation(expert_path: str):
    """Test N ∈ {100, 500, 1000, 2000} demonstrations."""
    demo_sizes = [100, 500, 1000, 2000]
    expert = load_expert(expert_path)
    results = {}

    rl_cfg = REINFORCEConfig(env_id=ENV_ID, n_episodes=1500, device=DEVICE,
                              eval_freq=50, eval_episodes=20, log_tensorboard=False)
    bc_cfg = BCConfig(n_epochs=50, device=DEVICE)

    # Collect a large pool once, then subsample
    full_demo = collect_demonstrations(expert, ENV_ID, n_episodes=200)

    for n in demo_sizes:
        print(f"\n{'='*60}")
        print(f"DEMO SIZE ABLATION: N = {n}")
        print(f"{'='*60}")

        # Subsample
        idx = np.random.choice(len(full_demo), min(n, len(full_demo)), replace=False)
        demo = DemonstrationDataset(
            observations=full_demo.observations[idx],
            actions=full_demo.actions[idx],
        )

        seed_means, seed_stds = [], []
        eval_at = None

        for seed in range(N_SEEDS):
            policy = make_policy(seed)
            bc_trainer = BehavioralCloning(policy, copy.copy(bc_cfg))
            bc_trainer.train(demo)

            trainer = REINFORCETrainer(policy, copy.copy(rl_cfg))
            h = trainer.train()
            seed_means.append(h["eval_mean"])
            seed_stds.append(h["eval_std"])
            if eval_at is None:
                eval_at = h["eval_at"]

        results[n] = {
            "eval_at":   eval_at,
            "eval_mean": np.array(seed_means).mean(axis=0).tolist(),
            "eval_std":  np.array(seed_stds).mean(axis=0).tolist(),
        }

    np.save("results/demo_size_ablation.npy", results)
    plot_demo_size_ablation(results, save_path="figures/demo_size_ablation.png")
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Expert quality ablation
# ---------------------------------------------------------------------------

def run_expert_quality_ablation():
    """
    Uses expert checkpoints saved at 50k / 200k / 500k training steps.
    Each checkpoint represents a different 'quality' of expert.
    """
    checkpoints = {
        "50k steps (weak)":  "checkpoints/expert/expert_ppo_50000_steps",
        "200k steps (mid)":  "checkpoints/expert/expert_ppo_200000_steps",
        "500k steps (full)": "checkpoints/expert/best/best_model",
    }

    results = {}
    rl_cfg = REINFORCEConfig(env_id=ENV_ID, n_episodes=1500, device=DEVICE,
                              eval_freq=50, eval_episodes=20, log_tensorboard=False)
    bc_cfg = BCConfig(n_epochs=50, device=DEVICE)

    for label, ckpt_path in checkpoints.items():
        print(f"\nExpert quality: {label}")
        expert = load_expert(ckpt_path)
        demo = collect_demonstrations(expert, ENV_ID, n_episodes=200)

        seed_means, seed_stds, eval_at = [], [], None
        for seed in range(N_SEEDS):
            policy = make_policy(seed)
            bc_trainer = BehavioralCloning(policy, copy.copy(bc_cfg))
            bc_trainer.train(demo)
            trainer = REINFORCETrainer(policy, copy.copy(rl_cfg))
            h = trainer.train()
            seed_means.append(h["eval_mean"])
            seed_stds.append(h["eval_std"])
            if eval_at is None:
                eval_at = h["eval_at"]

        results[label] = {
            "eval_at":   eval_at,
            "eval_mean": np.array(seed_means).mean(axis=0).tolist(),
            "eval_std":  np.array(seed_stds).mean(axis=0).tolist(),
        }

    np.save("results/expert_quality_ablation.npy", results)
    plot_learning_curves(
        results,
        threshold=THRESHOLD,
        title="Expert Quality Ablation",
        save_path="figures/expert_quality_ablation.png",
    )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["main", "demo_size", "expert_quality", "all"],
                        default="main")
    parser.add_argument("--expert_path", type=str, default="checkpoints/expert/best/best_model",
                        help="Path to trained expert checkpoint")
    parser.add_argument("--n_demos", type=int, default=200,
                        help="Number of demo episodes for main experiment")
    args = parser.parse_args()

    if args.experiment in ("main", "all"):
        print("\n=== Training Expert ===")
        expert_cfg = ExpertConfig(env_id=ENV_ID)
        expert = train_expert(expert_cfg)
        demo = collect_demonstrations(expert, ENV_ID, n_episodes=args.n_demos)
        demo.save("data/demonstrations_200ep.npz")

        print("\n=== Running Main Comparison ===")
        run_main_comparison(demo)

    if args.experiment in ("demo_size", "all"):
        print("\n=== Running Demo Size Ablation ===")
        run_demo_size_ablation(args.expert_path)

    if args.experiment in ("expert_quality", "all"):
        print("\n=== Running Expert Quality Ablation ===")
        run_expert_quality_ablation()


if __name__ == "__main__":
    main()
