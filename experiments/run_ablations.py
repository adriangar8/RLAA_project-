"""
experiments/run_ablations.py

Runs all ablation studies for the project report:

  1. Main comparison: BC only | RL from scratch | BC + REINFORCE
  2. Demo size ablation: BC only vs BC+REINFORCE for N ∈ {100, 500, 1000, 2000}
  3. Expert quality ablation: trained for 50k / 200k / 500k steps

Usage:
    python -m experiments.run_ablations --experiment main
    python -m experiments.run_ablations --experiment demo_size
    python -m experiments.run_ablations --experiment expert_quality
    python -m experiments.run_ablations --experiment all
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import gymnasium as gym
from pathlib import Path

from src.policy import PolicyNetwork
from src.expert import ExpertConfig, train_expert, load_expert, collect_demonstrations, DemonstrationDataset
from src.bc import BCConfig, BehavioralCloning
from src.rl import REINFORCEConfig, REINFORCETrainer
from src.evaluate import evaluate_policy, plot_learning_curves

sns.set_theme(style="whitegrid", palette="colorblind")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ENV_ID = "LunarLander-v3"
N_SEEDS = 5
OBS_DIM = 8
ACT_DIM = 4
THRESHOLD = 200.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)


def make_policy(seed: int = 0) -> PolicyNetwork:
    torch.manual_seed(seed)
    return PolicyNetwork(obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dims=[256, 256])


# ---------------------------------------------------------------------------
# Experiment 1: Main comparison
# ---------------------------------------------------------------------------

def run_main_comparison(demo: DemonstrationDataset):
    conditions = ["rl_scratch", "bc_only", "bc_reinforce"]
    all_results = {c: {"eval_at": None, "eval_mean_seeds": [], "eval_std_seeds": []} for c in conditions}

    rl_cfg = REINFORCEConfig(env_id=ENV_ID, n_episodes=2000, device=DEVICE,
                              eval_freq=50, eval_episodes=20, log_tensorboard=False)
    bc_cfg = BCConfig(n_epochs=50, device=DEVICE)

    for seed in range(N_SEEDS):
        print(f"\n{'='*60}\nSEED {seed+1}/{N_SEEDS}\n{'='*60}")

        # RL from scratch
        print("\n[1/3] RL from scratch...")
        policy_scratch = make_policy(seed)
        trainer = REINFORCETrainer(policy_scratch, copy.copy(rl_cfg))
        h = trainer.train()
        all_results["rl_scratch"]["eval_mean_seeds"].append(h["eval_mean"])
        all_results["rl_scratch"]["eval_std_seeds"].append(h["eval_std"])
        if all_results["rl_scratch"]["eval_at"] is None:
            all_results["rl_scratch"]["eval_at"] = h["eval_at"]

        # BC only
        print("\n[2/3] BC only...")
        policy_bc = make_policy(seed)
        bc_trainer = BehavioralCloning(policy_bc, copy.copy(bc_cfg))
        bc_trainer.train(demo)
        stats = evaluate_policy(policy_bc, ENV_ID, n_episodes=50, device=DEVICE)
        bc_eval_mean = [stats["mean"]] * len(h["eval_at"])
        bc_eval_std  = [stats["std"]]  * len(h["eval_at"])
        all_results["bc_only"]["eval_mean_seeds"].append(bc_eval_mean)
        all_results["bc_only"]["eval_std_seeds"].append(bc_eval_std)
        if all_results["bc_only"]["eval_at"] is None:
            all_results["bc_only"]["eval_at"] = h["eval_at"]

        # BC + REINFORCE
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

    plot_data = {}
    for c in conditions:
        means = np.array(all_results[c]["eval_mean_seeds"])
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
# Compares BC only vs BC+REINFORCE at each N — the key question is whether
# RL fine-tuning helps more when demonstrations are scarce.
# ---------------------------------------------------------------------------

def _plot_demo_size_with_bconly(results: dict, save_path: str = None):
    """
    Two-panel figure:
      Left:  BC+REINFORCE learning curves per N, with BC-only as horizontal markers
      Right: Final performance of BC only vs BC+REINFORCE vs N (log scale)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ns = sorted(results.keys())
    colors = sns.color_palette("colorblind", len(ns))

    # Left: learning curves
    ax = axes[0]
    for color, n in zip(colors, ns):
        d = results[n]
        x = np.array(d["eval_at"])
        y = np.array(d["eval_mean"])
        std = np.array(d["eval_std"])
        ax.plot(x, y, label=f"BC+RL N={n:,}", color=color, linewidth=2)
        ax.fill_between(x, y - std, y + std, alpha=0.12, color=color)
        # BC-only as dashed horizontal line
        ax.axhline(d["bc_only_mean"], linestyle="--", color=color, linewidth=1, alpha=0.7)

    ax.axhline(THRESHOLD, color="black", linestyle=":", linewidth=1.5, label="Threshold (200)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title("BC+REINFORCE vs BC Only (dashed) by Demo Size", fontsize=11)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)

    # Right: final performance bar comparison
    ax = axes[1]
    bcrl_finals  = [results[n]["eval_mean"][-1]  for n in ns]
    bcrl_stds    = [results[n]["eval_std"][-1]   for n in ns]
    bconly_finals = [results[n]["bc_only_mean"]  for n in ns]
    bconly_stds   = [results[n]["bc_only_std"]   for n in ns]

    x_pos = np.arange(len(ns))
    width = 0.35
    ax.bar(x_pos - width/2, bconly_finals, width, yerr=bconly_stds,
           label="BC only", capsize=4, alpha=0.85)
    ax.bar(x_pos + width/2, bcrl_finals,  width, yerr=bcrl_stds,
           label="BC+REINFORCE", capsize=4, alpha=0.85)
    ax.axhline(THRESHOLD, color="black", linestyle=":", linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N={n:,}" for n in ns])
    ax.set_ylabel("Final Mean Return", fontsize=12)
    ax.set_title("Final Performance: BC Only vs BC+REINFORCE", fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def run_demo_size_ablation(expert_path: str):
    """
    For each N ∈ {100, 500, 1000, 2000} transitions, compare:
      - BC only (fixed policy after supervised training)
      - BC + REINFORCE (fine-tuned)
    Hypothesis: RL fine-tuning benefit grows as N shrinks (more covariate shift).
    """
    demo_sizes = [100, 500, 1000, 2000]
    expert = load_expert(expert_path)

    rl_cfg = REINFORCEConfig(env_id=ENV_ID, n_episodes=1500, device=DEVICE,
                              eval_freq=50, eval_episodes=20, log_tensorboard=False)
    bc_cfg = BCConfig(n_epochs=50, device=DEVICE)

    # Collect a large pool once, subsample per condition
    full_demo = collect_demonstrations(expert, ENV_ID, n_episodes=300)
    results = {}

    for n in demo_sizes:
        print(f"\n{'='*60}\nDEMO SIZE ABLATION: N = {n} transitions\n{'='*60}")

        idx = np.random.choice(len(full_demo), min(n, len(full_demo)), replace=False)
        demo = DemonstrationDataset(
            observations=full_demo.observations[idx],
            actions=full_demo.actions[idx],
        )

        bcrl_means, bcrl_stds, bconly_means, bconly_stds_per_seed = [], [], [], []
        eval_at = None

        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # BC only
            policy_bc = make_policy(seed)
            bc_t = BehavioralCloning(policy_bc, copy.copy(bc_cfg))
            bc_t.train(demo)
            stats = evaluate_policy(policy_bc, ENV_ID, n_episodes=50, device=DEVICE)
            bconly_means.append(stats["mean"])

            # BC + REINFORCE
            policy_bcrl = make_policy(seed)
            bc_t2 = BehavioralCloning(policy_bcrl, copy.copy(bc_cfg))
            bc_t2.train(demo)
            trainer = REINFORCETrainer(policy_bcrl, copy.copy(rl_cfg))
            h = trainer.train()
            bcrl_means.append(h["eval_mean"])
            bcrl_stds.append(h["eval_std"])
            if eval_at is None:
                eval_at = h["eval_at"]

        results[n] = {
            "eval_at":      eval_at,
            "eval_mean":    np.array(bcrl_means).mean(axis=0).tolist(),
            "eval_std":     np.array(bcrl_stds).mean(axis=0).tolist(),
            "bc_only_mean": float(np.mean(bconly_means)),
            "bc_only_std":  float(np.std(bconly_means)),
        }
        print(f"  BC only:        {np.mean(bconly_means):.1f} ± {np.std(bconly_means):.1f}")
        print(f"  BC+REINFORCE:   {np.array(bcrl_means).mean(axis=0)[-1]:.1f}")

    np.save("results/demo_size_ablation.npy", results)
    _plot_demo_size_with_bconly(results, save_path="figures/demo_size_ablation.png")
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Expert quality ablation
# ---------------------------------------------------------------------------

def run_expert_quality_ablation():
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
    parser.add_argument("--expert_path", type=str, default="checkpoints/expert/best/best_model")
    parser.add_argument("--n_demos", type=int, default=200)
    args = parser.parse_args()

    if args.experiment in ("main", "all"):
        print("\n=== Training Expert ===")
        expert_cfg = ExpertConfig(env_id=ENV_ID)
        expert = train_expert(expert_cfg)
        demo = collect_demonstrations(expert, ENV_ID, n_episodes=args.n_demos)
        Path("data").mkdir(exist_ok=True)
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