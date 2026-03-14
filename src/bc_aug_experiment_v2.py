"""
Experiment v2: BC Data Augmentation Ablation + REINFORCE & PPO Fine-tuning

Improvements over v1 (bc_aug_reinforce.py):
  1. Noise sigma sweep: [0.0, 0.01, 0.05, 0.1, 0.2]
  2. More REINFORCE episodes: 5000 (was 2000)
  3. PPO fine-tuning alongside REINFORCE for lower-variance comparison
  4. 10 seeds (was 5)

Pipeline per seed per noise_std:
  Expert demos (N=500) → BC (noise_std) → Eval + cov. shift
    ├─→ REINFORCE (5000 eps)
    └─→ PPO (100k steps)
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate

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
N_SEEDS = 10
NOISE_STDS = [0.0, 0.01, 0.05, 0.1, 0.2]
REINFORCE_EPISODES = 5000
PPO_TIMESTEPS = 100_000
PPO_EVAL_FREQ = 10_000
THRESHOLD = 200.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Allow quick test mode via command line: python -m src.bc_aug_experiment_v2 --test
TEST_MODE = "--test" in sys.argv
if TEST_MODE:
    N_SEEDS = 1
    NOISE_STDS = [0.0, 0.1]
    REINFORCE_EPISODES = 200
    PPO_TIMESTEPS = 10_000
    PPO_EVAL_FREQ = 5_000
    print("*** TEST MODE: reduced config for sanity check ***\n")


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


def train_bc(demo, noise_std: float, seed: int) -> PolicyNetwork:
    torch.manual_seed(seed)
    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    cfg = BCConfig(noise_std=noise_std)
    trainer = BehavioralCloning(policy, cfg)
    trainer.train(demo)
    return policy


def run_reinforce(policy: PolicyNetwork, seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = REINFORCEConfig(
        env_id=ENV_ID,
        n_episodes=REINFORCE_EPISODES,
        log_tensorboard=False,
    )
    trainer = REINFORCETrainer(policy, cfg)
    history = trainer.train()
    return history


def transfer_bc_weights_to_ppo(bc_policy, ppo_model):
    """Copy BC network weights into the PPO policy network by shape matching."""
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
    print(f"    Transferred {len(matched)} layers from BC → PPO")


class PPOPolicyWrapper:
    """Wraps SB3 PPO model to match our evaluate_policy interface."""
    def __init__(self, model):
        self.model = model

    def act(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)


def run_ppo(bc_policy: PolicyNetwork, seed: int) -> dict:
    """Fine-tune BC policy with PPO, return eval history."""
    env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])

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
        seed=seed,
        verbose=0,
        device="cpu",
        policy_kwargs={"net_arch": [256, 256]},
    )

    transfer_bc_weights_to_ppo(bc_policy, ppo_model)

    # Collect eval metrics at regular intervals via callback
    eval_results = {"eval_at": [], "eval_mean": [], "eval_std": []}

    class EvalMetricsCallback(EvalCallback):
        def _on_step(self) -> bool:
            result = super()._on_step()
            if self.last_mean_reward is not None and len(eval_results["eval_at"]) < self.n_calls // self.eval_freq:
                eval_results["eval_at"].append(self.n_calls)
                eval_results["eval_mean"].append(self.last_mean_reward)
                eval_results["eval_std"].append(
                    float(np.std(self.evaluations_results[-1])) if self.evaluations_results else 0.0
                )
            return result

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=PPO_EVAL_FREQ,
        n_eval_episodes=20,
        verbose=0,
    )

    ppo_model.learn(total_timesteps=PPO_TIMESTEPS, callback=eval_callback)

    # Extract eval results from the callback
    if hasattr(eval_callback, "evaluations_results") and eval_callback.evaluations_results:
        eval_results["eval_at"] = list(range(
            PPO_EVAL_FREQ,
            PPO_EVAL_FREQ * (len(eval_callback.evaluations_results) + 1),
            PPO_EVAL_FREQ,
        ))[:len(eval_callback.evaluations_results)]
        eval_results["eval_mean"] = [float(np.mean(r)) for r in eval_callback.evaluations_results]
        eval_results["eval_std"] = [float(np.std(r)) for r in eval_callback.evaluations_results]

    # Final evaluation
    ppo_wrapper = PPOPolicyWrapper(ppo_model)
    final_eval = evaluate_policy(ppo_wrapper, ENV_ID, n_episodes=30, device="cpu")

    env.close()
    eval_env.close()

    eval_results["final_mean"] = final_eval["mean"]
    eval_results["final_std"] = final_eval["std"]
    return eval_results


# -------------------------------------------------------
# Main experiment
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"BC Augmentation Experiment v2")
print(f"  N_DEMOS={N_DEMOS}, N_SEEDS={N_SEEDS}")
print(f"  NOISE_STDS={NOISE_STDS}")
print(f"  REINFORCE={REINFORCE_EPISODES} eps, PPO={PPO_TIMESTEPS} steps")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")

# Load full demo pool
print("Loading demonstrations...")
full_demo = DemonstrationDataset.load(DEMO_PATH)
print(f"  Full pool: {len(full_demo.observations)} transitions\n")

# Initialize results structure
results = {}
for sigma in NOISE_STDS:
    key = f"sigma_{sigma}"
    results[key] = {
        "noise_std": sigma,
        "bc_mean": [], "bc_std": [],
        "shift": [],
        "reinforce_curves": [], "reinforce_final_mean": [], "reinforce_final_std": [],
        "ppo_curves": [], "ppo_final_mean": [], "ppo_final_std": [],
        "reinforce_eval_at": None,
        "ppo_eval_at": None,
    }

for seed in range(N_SEEDS):
    print(f"\n{'='*60}")
    print(f"--- Seed {seed+1}/{N_SEEDS} ---")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    demo = subsample_demo(full_demo, N_DEMOS, rng)

    for sigma in NOISE_STDS:
        key = f"sigma_{sigma}"
        print(f"\n  [σ={sigma}] Training BC...")
        policy = train_bc(demo, noise_std=sigma, seed=seed)

        # BC-only evaluation
        bc_eval = evaluate_policy(policy, ENV_ID, n_episodes=30, device=DEVICE)
        print(f"  [σ={sigma}] BC score: {bc_eval['mean']:.1f} ± {bc_eval['std']:.1f}")
        results[key]["bc_mean"].append(bc_eval["mean"])
        results[key]["bc_std"].append(bc_eval["std"])

        # Covariate shift
        shift = measure_covariate_shift(
            policy, demo.observations, ENV_ID, n_episodes=50, device=DEVICE
        )
        print(f"  [σ={sigma}] Cov. shift (L2): {shift['mean_shift']:.4f}")
        results[key]["shift"].append(shift["mean_shift"])

        # REINFORCE fine-tuning
        print(f"  [σ={sigma}] REINFORCE fine-tuning ({REINFORCE_EPISODES} eps)...")
        reinforce_policy = copy.deepcopy(policy)
        reinforce_history = run_reinforce(reinforce_policy, seed=seed)

        eval_returns = reinforce_history.get("eval_mean", [])
        eval_at = reinforce_history.get("eval_at", [])
        results[key]["reinforce_curves"].append(eval_returns)
        if results[key]["reinforce_eval_at"] is None and eval_at:
            results[key]["reinforce_eval_at"] = eval_at

        rf_final = np.mean(eval_returns[-3:]) if len(eval_returns) >= 3 else (eval_returns[-1] if eval_returns else 0.0)
        rf_final_std = np.std(eval_returns[-3:]) if len(eval_returns) >= 3 else 0.0
        results[key]["reinforce_final_mean"].append(rf_final)
        results[key]["reinforce_final_std"].append(rf_final_std)
        print(f"  [σ={sigma}] REINFORCE final: {rf_final:.1f} ± {rf_final_std:.1f}")

        # PPO fine-tuning
        print(f"  [σ={sigma}] PPO fine-tuning ({PPO_TIMESTEPS} steps)...")
        ppo_policy = copy.deepcopy(policy)
        ppo_history = run_ppo(ppo_policy, seed=seed)

        ppo_eval_mean = ppo_history.get("eval_mean", [])
        ppo_eval_at = ppo_history.get("eval_at", [])
        results[key]["ppo_curves"].append(ppo_eval_mean)
        if results[key]["ppo_eval_at"] is None and ppo_eval_at:
            results[key]["ppo_eval_at"] = ppo_eval_at

        ppo_final = ppo_history.get("final_mean", 0.0)
        ppo_final_std = ppo_history.get("final_std", 0.0)
        results[key]["ppo_final_mean"].append(ppo_final)
        results[key]["ppo_final_std"].append(ppo_final_std)
        print(f"  [σ={sigma}] PPO final: {ppo_final:.1f} ± {ppo_final_std:.1f}")


# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")

for sigma in NOISE_STDS:
    key = f"sigma_{sigma}"
    r = results[key]
    print(f"\nσ={sigma}:")
    print(f"  BC score:         {np.mean(r['bc_mean']):7.1f} ± {np.std(r['bc_mean']):.1f}")
    print(f"  Cov. shift (L2):  {np.mean(r['shift']):7.4f} ± {np.std(r['shift']):.4f}")
    print(f"  REINFORCE final:  {np.mean(r['reinforce_final_mean']):7.1f} ± {np.std(r['reinforce_final_mean']):.1f}")
    print(f"  PPO final:        {np.mean(r['ppo_final_mean']):7.1f} ± {np.std(r['ppo_final_mean']):.1f}")


# -------------------------------------------------------
# Save results
# -------------------------------------------------------

os.makedirs("results", exist_ok=True)
save_name = "bc_aug_v2_test_results.npy" if TEST_MODE else "bc_aug_v2_results.npy"
np.save(f"results/{save_name}", results)  # pyright: ignore[reportArgumentType]
print(f"\nSaved results → results/{save_name}")


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

os.makedirs("figures", exist_ok=True)
prefix = "bc_aug_v2_test" if TEST_MODE else "bc_aug_v2"

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(NOISE_STDS)))  # pyright: ignore[reportAttributeAccessIssue]

# --- Figure 1: REINFORCE learning curves by noise level ---
fig, ax = plt.subplots(figsize=(10, 6))
for i, sigma in enumerate(NOISE_STDS):
    key = f"sigma_{sigma}"
    r = results[key]
    eval_at = r["reinforce_eval_at"]
    curves = r["reinforce_curves"]
    if eval_at and curves:
        # Pad curves to same length
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for j, c in enumerate(curves):
            padded[j, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = eval_at[:max_len]
        ax.plot(x, mean, label=f"σ={sigma}", color=colors[i], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[i])

ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Episode")
ax.set_ylabel("Mean Return")
ax.set_title(f"REINFORCE Fine-tuning by Noise Level (N={N_DEMOS}, {N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"figures/{prefix}_reinforce_curves.png", dpi=150)
print(f"Saved figure → figures/{prefix}_reinforce_curves.png")
plt.close()

# --- Figure 2: PPO learning curves by noise level ---
fig, ax = plt.subplots(figsize=(10, 6))
for i, sigma in enumerate(NOISE_STDS):
    key = f"sigma_{sigma}"
    r = results[key]
    eval_at = r["ppo_eval_at"]
    curves = r["ppo_curves"]
    if eval_at and curves:
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for j, c in enumerate(curves):
            padded[j, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = eval_at[:max_len]
        ax.plot(x, mean, label=f"σ={sigma}", color=colors[i], linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[i])

ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Return")
ax.set_title(f"PPO Fine-tuning by Noise Level (N={N_DEMOS}, {N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"figures/{prefix}_ppo_curves.png", dpi=150)
print(f"Saved figure → figures/{prefix}_ppo_curves.png")
plt.close()

# --- Figure 3: BC score + covariate shift vs noise_std ---
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

x = np.arange(len(NOISE_STDS))
width = 0.35
bc_means = [np.mean(results[f"sigma_{s}"]["bc_mean"]) for s in NOISE_STDS]
bc_stds = [np.std(results[f"sigma_{s}"]["bc_mean"]) for s in NOISE_STDS]
shift_means = [np.mean(results[f"sigma_{s}"]["shift"]) for s in NOISE_STDS]
shift_stds = [np.std(results[f"sigma_{s}"]["shift"]) for s in NOISE_STDS]

ax1.bar(x - width/2, bc_means, width, yerr=bc_stds, label="BC score",
        color=[colors[i] for i in range(len(NOISE_STDS))], alpha=0.7, capsize=3)
ax2.bar(x + width/2, shift_means, width, yerr=shift_stds, label="Cov. shift (L2)",
        color=[colors[i] for i in range(len(NOISE_STDS))], alpha=0.3, hatch="//", capsize=3)

ax1.set_xticks(x)
ax1.set_xticklabels([f"σ={s}" for s in NOISE_STDS])
ax1.set_ylabel("BC Mean Return")
ax2.set_ylabel("Covariate Shift (L2)")
ax1.set_title("BC Score & Covariate Shift vs Noise Level")
lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right")
plt.tight_layout()
plt.savefig(f"figures/{prefix}_bc_shift.png", dpi=150)
print(f"Saved figure → figures/{prefix}_bc_shift.png")
plt.close()

# --- Figure 4: Final performance comparison (REINFORCE vs PPO by noise level) ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(NOISE_STDS))
width = 0.35

rf_means = [np.mean(results[f"sigma_{s}"]["reinforce_final_mean"]) for s in NOISE_STDS]
rf_stds = [np.std(results[f"sigma_{s}"]["reinforce_final_mean"]) for s in NOISE_STDS]
ppo_means = [np.mean(results[f"sigma_{s}"]["ppo_final_mean"]) for s in NOISE_STDS]
ppo_stds = [np.std(results[f"sigma_{s}"]["ppo_final_mean"]) for s in NOISE_STDS]

ax.bar(x - width/2, rf_means, width, yerr=rf_stds, label="REINFORCE", color="#1f77b4", capsize=3)
ax.bar(x + width/2, ppo_means, width, yerr=ppo_stds, label="PPO", color="#ff7f0e", capsize=3)
ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})")

ax.set_xticks(x)
ax.set_xticklabels([f"σ={s}" for s in NOISE_STDS])
ax.set_ylabel("Final Mean Return")
ax.set_title(f"REINFORCE vs PPO Final Performance by Noise Level ({N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"figures/{prefix}_final_comparison.png", dpi=150)
print(f"Saved figure → figures/{prefix}_final_comparison.png")
plt.close()

print("\nDone!")
