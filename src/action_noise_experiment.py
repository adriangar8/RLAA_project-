"""
Experiment: Expert Action Noise (Epsilon-Greedy) Ablation

Hypothesis: replacing expert actions with random ones at rate `eps` during
data collection forces the expert to visit off-trajectory states and
demonstrate recovery behaviour — directly attacking covariate shift at the
*data* level rather than the *training* level (observation noise in BC).

Pipeline per seed per eps:
  Collect demos (eps-greedy expert, N=500) → BC (noise_std=0.0) → Eval + cov. shift
    └─→ REINFORCE fine-tuning (2000 episodes)

Data strategy:
  eps=0.0 reuses the existing demonstrations_200ep.npz pool.
  Other eps values are collected once and cached to data/action_noise/.

Usage:
  python -m src.action_noise_experiment           # full run
  python -m src.action_noise_experiment --test    # quick sanity check
  python -m src.action_noise_experiment --collect # only collect datasets
"""

import os
import sys
import copy
import numpy as np
import torch
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.policy import PolicyNetwork
from src.expert import load_expert, collect_demonstrations, DemonstrationDataset
from src.bc import BehavioralCloning, BCConfig
from src.rl import REINFORCETrainer, REINFORCEConfig
from src.evaluate import (
    evaluate_policy,
    measure_covariate_shift,
    plot_state_visitation_heatmap,
)

# -------------------------------------------------------
# Config
# -------------------------------------------------------

ENV_ID = "LunarLander-v3"
EXPERT_PATH = "checkpoints/expert/best/best_model"
BASE_DEMO_PATH = "data/demonstrations_200ep.npz"   # eps=0.0 pool (already exists)
ACTION_NOISE_DATA_DIR = "data/action_noise"

OBS_DIM = 8
ACT_DIM = 4
N_DEMOS = 500           # transitions subsampled per seed from the pool
N_COLLECT_EPISODES = 200  # episodes to collect for each eps > 0.0 pool
N_SEEDS = 10
ACTION_NOISE_EPS = [0.0, 0.05, 0.1, 0.2, 0.3]
REINFORCE_EPISODES = 2000
REINFORCE_EVAL_FREQ = 50
THRESHOLD = 200.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

TEST_MODE = "--test" in sys.argv
COLLECT_ONLY = "--collect" in sys.argv

if TEST_MODE:
    N_SEEDS = 1
    ACTION_NOISE_EPS = [0.0, 0.1]
    N_COLLECT_EPISODES = 10
    REINFORCE_EPISODES = 100
    REINFORCE_EVAL_FREQ = 20
    print("*** TEST MODE: reduced config for sanity check ***\n")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _pool_path(eps: float) -> str:
    if eps == 0.0:
        return BASE_DEMO_PATH
    return os.path.join(ACTION_NOISE_DATA_DIR, f"demos_eps_{eps:.2f}.npz")


def ensure_datasets():
    """Collect and cache demonstration pools for each eps > 0.0 if needed."""
    os.makedirs(ACTION_NOISE_DATA_DIR, exist_ok=True)
    expert = None  # lazy-load

    for eps in ACTION_NOISE_EPS:
        if eps == 0.0:
            continue
        path = _pool_path(eps)
        if os.path.exists(path):
            print(f"  Dataset for eps={eps} already exists: {path}")
            continue

        if expert is None:
            print(f"\nLoading expert from {EXPERT_PATH} ...")
            expert = load_expert(EXPERT_PATH)

        print(f"\nCollecting {N_COLLECT_EPISODES} episodes with action_noise_eps={eps} ...")
        dataset = collect_demonstrations(
            expert,
            env_id=ENV_ID,
            n_episodes=N_COLLECT_EPISODES,
            deterministic=True,
            action_noise_eps=eps,
        )
        dataset.save(path)
        print(f"  Saved → {path}")


def subsample_demo(
    demo: DemonstrationDataset, n: int, rng: np.random.Generator
) -> DemonstrationDataset:
    idx = rng.choice(len(demo.observations), size=n, replace=False)
    sub = DemonstrationDataset.__new__(DemonstrationDataset)
    sub.observations = demo.observations[idx]
    sub.actions = demo.actions[idx]
    sub.episode_returns = demo.episode_returns
    return sub


def train_bc(demo: DemonstrationDataset, seed: int, save_as: str) -> PolicyNetwork:
    torch.manual_seed(seed)
    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    cfg = BCConfig(noise_std=0.0)   # no obs noise — isolate the action noise effect
    trainer = BehavioralCloning(policy, cfg)
    trainer.train(demo, save_as=save_as)
    return policy


def run_reinforce(bc_policy: PolicyNetwork, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = REINFORCEConfig(
        env_id=ENV_ID,
        n_episodes=REINFORCE_EPISODES,
        gamma=0.99,
        actor_lr=3e-4,
        critic_lr=1e-3,
        entropy_coef=0.01,
        eval_freq=REINFORCE_EVAL_FREQ,
        eval_episodes=20,
        save_dir=f"checkpoints/reinforce_action_noise/seed_{seed}",
        device=DEVICE,
        log_tensorboard=False,
    )
    trainer = REINFORCETrainer(bc_policy, cfg)
    history = trainer.train()

    eval_results = {
        "eval_at": history.get("eval_at", []),
        "eval_mean": history.get("eval_mean", []),
        "eval_std": history.get("eval_std", []),
    }

    final = evaluate_policy(trainer.policy, ENV_ID, n_episodes=30, device=DEVICE)
    eval_results["final_mean"] = final["mean"]
    eval_results["final_std"] = final["std"]
    return eval_results


# -------------------------------------------------------
# Main
# -------------------------------------------------------

print(f"\n{'='*60}")
print("Expert Action Noise Experiment")
print(f"  N_DEMOS={N_DEMOS}, N_SEEDS={N_SEEDS}")
print(f"  ACTION_NOISE_EPS={ACTION_NOISE_EPS}")
print(f"  REINFORCE={REINFORCE_EPISODES} episodes")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")

# Phase 1 ─ data collection (skips if cached)
print("--- Phase 1: Ensuring demonstration pools ---")
ensure_datasets()

if COLLECT_ONLY:
    print("\nDataset collection complete. Exiting (--collect mode).")
    sys.exit(0)

# Load all pools into memory once
pools: dict[float, DemonstrationDataset] = {}
for eps in ACTION_NOISE_EPS:
    path = _pool_path(eps)
    pools[eps] = DemonstrationDataset.load(path)
    print(f"  eps={eps}: {len(pools[eps])} transitions loaded from {path}")

# Phase 2 ─ experiment loop
save_name = "action_noise_test_results.npy" if TEST_MODE else "action_noise_results.npy"
results_path = f"results/{save_name}"

if os.path.exists(results_path):
    print(f"\nResuming from existing results file: {results_path}")
    results = np.load(results_path, allow_pickle=True).item()
else:
    results: dict = {}
    for eps in ACTION_NOISE_EPS:
        key = f"eps_{eps}"
        results[key] = {
            "eps": eps,
            "bc_mean": [], "bc_std": [],
            "shift": [],
            "reinforce_curves": [], "reinforce_final_mean": [], "reinforce_final_std": [],
            "reinforce_eval_at": None,
            # Store one set of obs arrays for the heatmap (first seed only)
            "heatmap_expert_obs": None,
            "heatmap_agent_obs": None,
        }

for seed in range(N_SEEDS):
    print(f"\n{'='*60}")
    print(f"--- Seed {seed+1}/{N_SEEDS} ---")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    for eps in ACTION_NOISE_EPS:
        key = f"eps_{eps}"
        
        # Check if already completed for this seed
        if len(results[key]["bc_mean"]) > seed:
            print(f"  [eps={eps}] Seed {seed+1} already completed. Skipping.")
            continue
            
        print(f"\n  [eps={eps}] Training BC (noise_std=0.0) ...")
        demo = subsample_demo(pools[eps], N_DEMOS, rng)
        policy = train_bc(demo, seed=seed, save_as=f"bc_eps_{eps}_seed_{seed}.pt")

        bc_eval = evaluate_policy(policy, ENV_ID, n_episodes=30, device=DEVICE)
        print(f"  [eps={eps}] BC score: {bc_eval['mean']:.1f} ± {bc_eval['std']:.1f}")
        results[key]["bc_mean"].append(bc_eval["mean"])
        results[key]["bc_std"].append(bc_eval["std"])

        shift = measure_covariate_shift(
            policy, demo.observations, ENV_ID, n_episodes=50, device=DEVICE
        )
        print(f"  [eps={eps}] Cov. shift (L2): {shift['mean_shift']:.4f}")
        results[key]["shift"].append(shift["mean_shift"])

        # Capture obs arrays for heatmap
        if results[key]["heatmap_expert_obs"] is None:
            results[key]["heatmap_expert_obs"] = shift["expert_obs"]
            results[key]["heatmap_agent_obs"] = shift["agent_obs"]

        print(f"  [eps={eps}] REINFORCE fine-tuning ({REINFORCE_EPISODES} episodes) ...")
        reinforce_history = run_reinforce(copy.deepcopy(policy), seed=seed)

        reinforce_eval_mean = reinforce_history.get("eval_mean", [])
        reinforce_eval_at = reinforce_history.get("eval_at", [])
        results[key]["reinforce_curves"].append(reinforce_eval_mean)
        if results[key]["reinforce_eval_at"] is None and reinforce_eval_at:
            results[key]["reinforce_eval_at"] = reinforce_eval_at

        reinforce_final = reinforce_history.get("final_mean", 0.0)
        reinforce_final_std = reinforce_history.get("final_std", 0.0)
        results[key]["reinforce_final_mean"].append(reinforce_final)
        results[key]["reinforce_final_std"].append(reinforce_final_std)
        print(f"  [eps={eps}] REINFORCE final: {reinforce_final:.1f} ± {reinforce_final_std:.1f}")

    # Save results intermittently after each seed
    os.makedirs("results", exist_ok=True)
    np.save(results_path, results)  # pyright: ignore[reportArgumentType]
    print(f"  Saved intermediate results to {results_path}")

# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for eps in ACTION_NOISE_EPS:
    key = f"eps_{eps}"
    r = results[key]
    print(f"\neps={eps}:")
    print(f"  BC score:        {np.mean(r['bc_mean']):7.1f} ± {np.std(r['bc_mean']):.1f}")
    print(f"  Cov. shift (L2): {np.mean(r['shift']):7.4f} ± {np.std(r['shift']):.4f}")
    print(f"  REINFORCE final: {np.mean(r['reinforce_final_mean']):7.1f} ± {np.std(r['reinforce_final_mean']):.1f}")

# -------------------------------------------------------
# Save results
# -------------------------------------------------------

os.makedirs("results", exist_ok=True)
save_name = "action_noise_test_results.npy" if TEST_MODE else "action_noise_results.npy"
np.save(f"results/{save_name}", results)  # pyright: ignore[reportArgumentType]
print(f"\nSaved results → results/{save_name}")

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

os.makedirs("figures", exist_ok=True)
prefix = "action_noise_test" if TEST_MODE else "action_noise"
colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(ACTION_NOISE_EPS)))  # pyright: ignore[reportAttributeAccessIssue]

# Figure 1: REINFORCE learning curves by eps
fig, ax = plt.subplots(figsize=(10, 6))
for i, eps in enumerate(ACTION_NOISE_EPS):
    key = f"eps_{eps}"
    r = results[key]
    eval_at = r["reinforce_eval_at"]
    curves = r["reinforce_curves"]
    if eval_at and curves:
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for j, c in enumerate(curves):
            padded[j, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = eval_at[:max_len]
        ax.plot(x, mean, label=f"eps={eps}", color=colors[i], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[i])

ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Return")
ax.set_title(f"REINFORCE Fine-tuning by Action Noise Level (N={N_DEMOS}, {N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"figures/{prefix}_reinforce_curves.png", dpi=150)
print(f"Saved → figures/{prefix}_reinforce_curves.png")
plt.close()

# Figure 2: BC score + covariate shift vs eps
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
x = np.arange(len(ACTION_NOISE_EPS))
width = 0.35
bc_means = [np.mean(results[f"eps_{e}"]["bc_mean"]) for e in ACTION_NOISE_EPS]
bc_stds = [np.std(results[f"eps_{e}"]["bc_mean"]) for e in ACTION_NOISE_EPS]
shift_means = [np.mean(results[f"eps_{e}"]["shift"]) for e in ACTION_NOISE_EPS]
shift_stds = [np.std(results[f"eps_{e}"]["shift"]) for e in ACTION_NOISE_EPS]

ax1.bar(x - width/2, bc_means, width, yerr=bc_stds,
        label="BC score", color=[colors[i] for i in range(len(ACTION_NOISE_EPS))],
        alpha=0.7, capsize=3)
ax2.bar(x + width/2, shift_means, width, yerr=shift_stds,
        label="Cov. shift (L2)", color=[colors[i] for i in range(len(ACTION_NOISE_EPS))],
        alpha=0.3, hatch="//", capsize=3)

ax1.set_xticks(x)
ax1.set_xticklabels([f"eps={e}" for e in ACTION_NOISE_EPS])
ax1.set_ylabel("BC Mean Return")
ax2.set_ylabel("Covariate Shift (L2)")
ax1.set_title("BC Score & Covariate Shift vs Action Noise Level")
lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right")
plt.tight_layout()
plt.savefig(f"figures/{prefix}_bc_shift.png", dpi=150)
print(f"Saved → figures/{prefix}_bc_shift.png")
plt.close()

# Figure 3: State visitation heatmap — eps=0.0 vs eps=0.1 vs expert
heatmap_eps_values = [e for e in [0.0, 0.1] if e in ACTION_NOISE_EPS]
heatmap_obs: dict = {}
for eps in heatmap_eps_values:
    key = f"eps_{eps}"
    agent_obs = results[key].get("heatmap_agent_obs")
    expert_obs = results[key].get("heatmap_expert_obs")
    if expert_obs is not None and "Expert" not in heatmap_obs:
        heatmap_obs["Expert"] = expert_obs
    if agent_obs is not None:
        heatmap_obs[f"BC (eps={eps})"] = agent_obs

if len(heatmap_obs) >= 2:
    heatmap_path = f"figures/{prefix}_state_heatmap.png"
    plot_state_visitation_heatmap(
        heatmap_obs,
        save_path=heatmap_path,
        title="Covariate Shift: State Visitation Heatmap",
    )
    print(f"Saved → {heatmap_path}")

print("\nDone!")
