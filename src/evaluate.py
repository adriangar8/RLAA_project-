"""
Evaluation utilities.

- evaluate_policy: rollout a policy and collect episode returns
- covariate_shift_analysis: measure state distribution divergence (BC vs RL)
- plot_learning_curves: generate figures for the report
- compute_sample_efficiency: episodes to reach threshold
"""

import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from typing import Optional

sns.set_theme(style="whitegrid", palette="colorblind")


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy,
    env_id: str,
    n_episodes: int = 50,
    deterministic: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Evaluate a policy and return full statistics.

    Returns:
        dict with keys: mean, std, median, min, max, all_returns
    """
    env = gym.make(env_id)
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_return, done = 0.0, False
        while not done:
            action = policy.act(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
        returns.append(ep_return)

    env.close()
    returns = np.array(returns)
    return {
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "median": float(np.median(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "all_returns": returns,
    }


# ---------------------------------------------------------------------------
# Sample efficiency metric
# ---------------------------------------------------------------------------

def episodes_to_threshold(
    eval_means: list[float],
    eval_at: list[int],
    threshold: float = 200.0,
) -> Optional[int]:
    """
    Returns the first episode count at which mean eval reward >= threshold.
    Returns None if never reached.
    """
    for ep, mean in zip(eval_at, eval_means):
        if mean >= threshold:
            return ep
    return None


# ---------------------------------------------------------------------------
# Covariate shift analysis
# ---------------------------------------------------------------------------

def measure_covariate_shift(
    policy,
    expert_obs: np.ndarray,
    env_id: str,
    n_episodes: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Measures divergence between expert state distribution and agent state distribution.
    Uses maximum mean discrepancy (MMD) as a proxy.

    This is the empirical demonstration of the covariate shift problem.
    """
    # Collect agent state distribution
    env = gym.make(env_id)
    agent_obs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            agent_obs.append(obs)
            action = policy.act(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    agent_obs = np.array(agent_obs)

    # Subsample to same size for fair comparison
    n = min(len(expert_obs), len(agent_obs))
    idx_e = np.random.choice(len(expert_obs), n, replace=False)
    idx_a = np.random.choice(len(agent_obs), n, replace=False)
    E = expert_obs[idx_e]
    A = agent_obs[idx_a]

    # Simple L2 distance between means as a lightweight proxy
    mean_shift = np.linalg.norm(E.mean(axis=0) - A.mean(axis=0))
    std_ratio = (A.std(axis=0) / (E.std(axis=0) + 1e-8)).mean()

    return {
        "mean_shift": float(mean_shift),
        "std_ratio": float(std_ratio),
        "expert_obs": E,
        "agent_obs": A,
        "n_agent_transitions": len(agent_obs),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(values: list, window: int = 20) -> np.ndarray:
    return uniform_filter1d(np.array(values, dtype=float), size=window)


def plot_learning_curves(
    results: dict,       # {label: {"eval_at": [...], "eval_mean": [...], "eval_std": [...]}}
    threshold: float = 200.0,
    title: str = "Learning Curves",
    save_path: Optional[str] = None,
):
    """
    Plot mean ± std learning curves for multiple conditions.
    Also draws the performance threshold as a dashed line.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, data in results.items():
        x = np.array(data["eval_at"])
        y = np.array(data["eval_mean"])
        std = np.array(data["eval_std"])
        ax.plot(x, y, label=label, linewidth=2)
        ax.fill_between(x, y - std, y + std, alpha=0.15)

    ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold ({threshold})")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Episode Return", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()
    return fig


def plot_bc_training(history: dict, save_path: Optional[str] = None):
    """Plot BC training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("BC Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Action Accuracy")
    ax2.set_title("BC Accuracy")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_demo_size_ablation(
    results: dict,       # {n_demos: {"eval_mean": [...], "eval_std": [...], "eval_at": [...]}}
    save_path: Optional[str] = None,
):
    """Plot the effect of demonstration dataset size on learning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: learning curves per demo size
    ax = axes[0]
    for n_demos, data in sorted(results.items()):
        x = np.array(data["eval_at"])
        y = np.array(data["eval_mean"])
        ax.plot(x, y, label=f"N={n_demos:,}")
        ax.fill_between(x, y - np.array(data["eval_std"]), y + np.array(data["eval_std"]), alpha=0.1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Return")
    ax.set_title("Effect of Demo Dataset Size")
    ax.legend()

    # Right: final performance vs demo size
    ax = axes[1]
    ns = sorted(results.keys())
    final_means = [results[n]["eval_mean"][-1] for n in ns]
    final_stds = [results[n]["eval_std"][-1] for n in ns]
    ax.errorbar(ns, final_means, yerr=final_stds, marker="o", capsize=4)
    ax.set_xscale("log")
    ax.set_xlabel("Number of Demonstrations (log scale)")
    ax.set_ylabel("Final Mean Return")
    ax.set_title("Final Performance vs Demo Size")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
