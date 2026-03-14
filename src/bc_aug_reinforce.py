"""
Experiment: BC with Data Augmentation + REINFORCE fine-tuning

Hypothesis: Adding Gaussian noise to observations during BC training forces the policy
to learn recovery from near-expert states, reducing covariate shift and providing a
better warm-start for REINFORCE — especially in the low-data (N=100) regime.

Pipeline:
  Expert demos (N=100) → BC (plain) ──────────────────────┐
                       → BC (aug, σ=0.05) → REINFORCE → Eval + compare
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
N_DEMOS = 500  # subsample to simulate low-data regime
N_SEEDS = 5
NOISE_STD = 0.1  # Gaussian noise std for augmented BC
REINFORCE_EPISODES = 2000
THRESHOLD = 200.0

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
    """Randomly subsample n transitions from a DemonstrationDataset."""
    idx = rng.choice(len(demo.observations), size=n, replace=False)
    sub = DemonstrationDataset.__new__(DemonstrationDataset)
    sub.observations = demo.observations[idx]
    sub.actions = demo.actions[idx]
    sub.episode_returns = (
        demo.episode_returns
    )  # keep full episode returns for reference
    return sub


def train_bc(demo, noise_std: float, seed: int) -> PolicyNetwork:
    """Train a BC policy (with or without augmentation) and return it."""
    torch.manual_seed(seed)
    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    cfg = BCConfig(noise_std=noise_std)
    trainer = BehavioralCloning(policy, cfg)
    trainer.train(demo)
    return policy


def run_reinforce(policy: PolicyNetwork, seed: int) -> dict:
    """Fine-tune policy with REINFORCE and return eval history."""
    torch.manual_seed(seed)
    cfg = REINFORCEConfig(
        env_id=ENV_ID,
        n_episodes=REINFORCE_EPISODES,
        log_tensorboard=False,
    )
    trainer = REINFORCETrainer(policy, cfg)
    history = trainer.train()
    return history


# -------------------------------------------------------
# Main experiment
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"BC Data Augmentation Experiment (N={N_DEMOS} demos, {N_SEEDS} seeds)")
print(f"{'='*60}\n")

# Load full demo pool once
print("Loading demonstrations...")
full_demo = DemonstrationDataset.load(DEMO_PATH)
print(f"  Full pool: {len(full_demo.observations)} transitions\n")

results = {
    "plain": {
        "bc_mean": [],
        "bc_std": [],
        "shift": [],
        "rl_curves": [],
        "rl_final_mean": [],
        "rl_final_std": [],
    },
    "aug": {
        "bc_mean": [],
        "bc_std": [],
        "shift": [],
        "rl_curves": [],
        "rl_final_mean": [],
        "rl_final_std": [],
    },
    "eval_at": None,
}

for seed in range(N_SEEDS):
    print(f"\n--- Seed {seed+1}/{N_SEEDS} ---")
    rng = np.random.default_rng(seed)

    # Subsample N_DEMOS transitions for this seed
    demo = subsample_demo(full_demo, N_DEMOS, rng)

    for label, noise_std in [("plain", 0.0), ("aug", NOISE_STD)]:
        print(f"\n  [{label}] Training BC (noise_std={noise_std})...")
        policy = train_bc(demo, noise_std=noise_std, seed=seed)

        # BC-only evaluation
        bc_eval = evaluate_policy(policy, ENV_ID, n_episodes=30, device=DEVICE)
        print(
            f"  [{label}] BC performance: {bc_eval['mean']:.1f} ± {bc_eval['std']:.1f}"
        )
        results[label]["bc_mean"].append(bc_eval["mean"])
        results[label]["bc_std"].append(bc_eval["std"])

        # Covariate shift measurement
        shift = measure_covariate_shift(
            policy, demo.observations, ENV_ID, n_episodes=50, device=DEVICE
        )
        print(f"  [{label}] Covariate shift (L2 mean-shift): {shift['mean_shift']:.4f}")
        results[label]["shift"].append(shift["mean_shift"])

        # REINFORCE fine-tuning
        print(f"  [{label}] REINFORCE fine-tuning ({REINFORCE_EPISODES} episodes)...")
        # Deep copy so both conditions start from their own BC weights independently
        policy_copy = copy.deepcopy(policy)
        history = run_reinforce(policy_copy, seed=seed)

        eval_returns = history.get("eval_mean", [])
        eval_at = history.get("eval_at", [])
        results[label]["rl_curves"].append(eval_returns)
        if results["eval_at"] is None and eval_at:
            results["eval_at"] = eval_at

        final_mean = (
            np.mean(eval_returns[-3:])
            if len(eval_returns) >= 3
            else (eval_returns[-1] if eval_returns else 0.0)
        )
        final_std = np.std(eval_returns[-3:]) if len(eval_returns) >= 3 else 0.0
        results[label]["rl_final_mean"].append(final_mean)
        results[label]["rl_final_std"].append(final_std)
        print(f"  [{label}] REINFORCE final: {final_mean:.1f} ± {final_std:.1f}")


# -------------------------------------------------------
# Aggregate and print summary
# -------------------------------------------------------

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for label in ["plain", "aug"]:
    tag = f"BC {'(aug σ=' + str(NOISE_STD) + ')' if label == 'aug' else '(no aug)'}"
    print(f"\n{tag}:")
    print(
        f"  BC-only score:    {np.mean(results[label]['bc_mean']):.1f} ± {np.std(results[label]['bc_mean']):.1f}"
    )
    print(
        f"  Cov. shift (L2):  {np.mean(results[label]['shift']):.4f} ± {np.std(results[label]['shift']):.4f}"
    )
    print(
        f"  REINFORCE final:  {np.mean(results[label]['rl_final_mean']):.1f} ± {np.std(results[label]['rl_final_mean']):.1f}"
    )


# -------------------------------------------------------
# Save results
# -------------------------------------------------------

os.makedirs("results", exist_ok=True)
np.save("results/bc_aug_results.npy", results)  # pyright: ignore[reportArgumentType]
print("\nSaved results → results/bc_aug_results.npy")


# -------------------------------------------------------
# Plot learning curves
# -------------------------------------------------------

os.makedirs("figures", exist_ok=True)

eval_at = results["eval_at"] or []

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: REINFORCE learning curves
ax = axes[0]
colors = {"plain": "#1f77b4", "aug": "#ff7f0e"}
labels = {"plain": "BC (no aug)", "aug": f"BC (aug σ={NOISE_STD})"}

for label in ["plain", "aug"]:
    curves = np.array(results[label]["rl_curves"])  # (seeds, eval_steps)
    if curves.ndim == 2 and len(eval_at) == curves.shape[1]:
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        ax.plot(eval_at, mean, label=labels[label], color=colors[label])
        ax.fill_between(eval_at, mean - std, mean + std, alpha=0.2, color=colors[label])

ax.axhline(
    THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})"
)
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Return")
ax.set_title(f"REINFORCE Fine-tuning (N={N_DEMOS} demos, {N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3)

# Right: BC-only score + covariate shift comparison
ax = axes[1]
x = np.arange(2)
width = 0.35
bc_means = [np.mean(results[l]["bc_mean"]) for l in ["plain", "aug"]]
bc_stds = [np.std(results[l]["bc_mean"]) for l in ["plain", "aug"]]
sft_means = [np.mean(results[l]["shift"]) for l in ["plain", "aug"]]
sft_stds = [np.std(results[l]["shift"]) for l in ["plain", "aug"]]

ax2 = ax.twinx()
bars1 = ax.bar(
    x - width / 2,
    bc_means,
    width,
    yerr=bc_stds,
    label="BC score",
    color=[colors[l] for l in ["plain", "aug"]],
    alpha=0.7,
)
bars2 = ax2.bar(
    x + width / 2,
    sft_means,
    width,
    yerr=sft_stds,
    label="Cov. shift (L2)",
    color=[colors[l] for l in ["plain", "aug"]],
    alpha=0.4,
    hatch="//",
)

ax.set_xticks(x)
ax.set_xticklabels(["No aug", f"Aug (σ={NOISE_STD})"])
ax.set_ylabel("BC Mean Return")
ax2.set_ylabel("Covariate Shift (L2 mean-shift)")
ax.set_title("BC Score vs Covariate Shift")

lines1, lbls1 = ax.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right")

plt.tight_layout()
plt.savefig("figures/bc_aug_comparison.png", dpi=150)
print("Saved figure → figures/bc_aug_comparison.png")
