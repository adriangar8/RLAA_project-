"""
BC Method Comparison: REINFORCE vs A2C vs PPO fine-tuning.

Compares three RL fine-tuning methods on the same BC-initialized policy,
for both plain BC (sigma=0.0) and augmented BC (sigma=0.1, best from V1).

Questions answered:
  1. Does augmentation (sigma=0.1) help all RL methods, or is it method-specific?
  2. Which RL method extracts the most value from a BC initialization?

Design:
  - 2 BC conditions: sigma in [0.0, 0.1]
  - 3 RL fine-tuning methods: REINFORCE (5000 eps), A2C (500k steps), PPO (100k steps)
  - 5 seeds each
  - Output: results/bc_method_comparison.npy, figures/bc_method_comparison_*.png

Estimated wall time: 5 seeds x 2 conditions x 3 methods x ~35 min = ~17.5 hours
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

from src.policy import PolicyNetwork, ActorCriticNetwork
from src.expert import DemonstrationDataset
from src.bc import BehavioralCloning, BCConfig
from src.rl import REINFORCETrainer, REINFORCEConfig, A2CTrainer, A2CConfig
from src.evaluate import evaluate_policy

# -------------------------------------------------------
# Config
# -------------------------------------------------------

ENV_ID = "LunarLander-v3"
DEMO_PATH = "data/demonstrations_200ep.npz"
OBS_DIM = 8
ACT_DIM = 4
N_DEMOS = 500
N_SEEDS = 5
NOISE_STDS = [0.0, 0.1]       # plain BC vs best augmented BC (from V1)
REINFORCE_EPISODES = 5000
A2C_STEPS = 500_000
PPO_TIMESTEPS = 100_000
PPO_EVAL_FREQ = 10_000
THRESHOLD = 200.0

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

TEST_MODE = "--test" in sys.argv
RESUME = "--resume" in sys.argv
PARTIAL_SAVE = "results/bc_method_comparison_partial.npy"

if TEST_MODE:
    N_SEEDS = 1
    REINFORCE_EPISODES = 200
    A2C_STEPS = 20_000
    PPO_TIMESTEPS = 10_000
    PPO_EVAL_FREQ = 5_000
    print("*** TEST MODE ***\n")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def subsample_demo(demo: DemonstrationDataset, n: int, rng: np.random.Generator) -> DemonstrationDataset:
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
    trainer = REINFORCETrainer(copy.deepcopy(policy), cfg)
    return trainer.train()


def bc_weights_to_ac(bc_policy: PolicyNetwork) -> ActorCriticNetwork:
    """
    Copy BC PolicyNetwork actor weights into a new ActorCriticNetwork.

    PolicyNetwork layout (hidden_dims=[256,256]):
      net.0  Linear(8→256),  net.2  Linear(256→256),  net.4  Linear(256→4)

    ActorCriticNetwork layout (hidden_dims=[256,256]):
      trunk.0  Linear(8→256)
      actor_head.0  Linear(256→256),  actor_head.2  Linear(256→4)
      critic_head.*  (random init — BC has no value function)
    """
    ac = ActorCriticNetwork(OBS_DIM, ACT_DIM)
    bc_sd = bc_policy.state_dict()
    ac_sd = ac.state_dict()
    mapping = {
        "trunk.0.weight":      "net.0.weight",
        "trunk.0.bias":        "net.0.bias",
        "actor_head.0.weight": "net.2.weight",
        "actor_head.0.bias":   "net.2.bias",
        "actor_head.2.weight": "net.4.weight",
        "actor_head.2.bias":   "net.4.bias",
    }
    for ac_key, bc_key in mapping.items():
        ac_sd[ac_key] = bc_sd[bc_key].clone()
    ac.load_state_dict(ac_sd)
    return ac


def run_a2c(policy: PolicyNetwork, seed: int) -> dict:
    torch.manual_seed(seed)
    ac_net = bc_weights_to_ac(policy)
    cfg = A2CConfig(env_id=ENV_ID, total_steps=A2C_STEPS)
    trainer = A2CTrainer(ac_net, cfg)
    return trainer.train()


def transfer_bc_weights_to_ppo(bc_policy: PolicyNetwork, ppo_model):
    bc_state = bc_policy.state_dict()
    ppo_state = ppo_model.policy.state_dict()
    matched = []
    for name, param in ppo_state.items():
        for bc_name, bc_param in bc_state.items():
            if param.shape == bc_param.shape:
                ppo_state[name] = bc_param.detach().cpu()
                matched.append(name)
                break
    ppo_model.policy.load_state_dict(ppo_state)
    print(f"    Transferred {len(matched)} layers BC → PPO")


class PPOPolicyWrapper:
    def __init__(self, model):
        self.model = model

    def act(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)


def run_ppo(bc_policy: PolicyNetwork, seed: int) -> dict:
    env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    eval_env = DummyVecEnv([lambda: gym.make(ENV_ID)])
    ppo_model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
        seed=seed, verbose=0, device="cpu",
        policy_kwargs={"net_arch": [256, 256]},
    )
    transfer_bc_weights_to_ppo(bc_policy, ppo_model)
    eval_callback = EvalCallback(eval_env, eval_freq=PPO_EVAL_FREQ, n_eval_episodes=20, log_path="/tmp/ppo_eval_log", verbose=0)
    ppo_model.learn(total_timesteps=PPO_TIMESTEPS, callback=eval_callback)

    history = {"eval_at": [], "eval_mean": [], "eval_std": []}
    if eval_callback.evaluations_results:
        history["eval_at"] = [int(t) for t in eval_callback.evaluations_timesteps]
        history["eval_mean"] = [float(np.mean(r)) for r in eval_callback.evaluations_results]
        history["eval_std"] = [float(np.std(r)) for r in eval_callback.evaluations_results]

    wrapper = PPOPolicyWrapper(ppo_model)
    final = evaluate_policy(wrapper, ENV_ID, n_episodes=30, device="cpu")
    history["final_mean"] = final["mean"]
    history["final_std"] = final["std"]
    env.close()
    eval_env.close()
    return history


# -------------------------------------------------------
# Results structure
# -------------------------------------------------------

def _blank_results():
    r = {}
    for sigma in NOISE_STDS:
        key = f"sigma_{sigma}"
        r[key] = {
            "noise_std": sigma,
            "reinforce": {"curves": [], "final_mean": [], "final_std": [], "eval_at": None},
            "a2c":       {"curves": [], "final_mean": [], "final_std": [], "eval_at": None},
            "ppo":       {"curves": [], "final_mean": [], "final_std": [], "eval_at": None},
        }
    return r


# -------------------------------------------------------
# Main
# -------------------------------------------------------

print(f"\n{'='*60}")
print(f"BC Method Comparison")
print(f"  N_SEEDS={N_SEEDS}, NOISE_STDS={NOISE_STDS}")
print(f"  REINFORCE={REINFORCE_EPISODES} eps | A2C={A2C_STEPS} steps | PPO={PPO_TIMESTEPS} steps")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")

print("Loading demonstrations...")
full_demo = DemonstrationDataset.load(DEMO_PATH)
print(f"  Full pool: {len(full_demo.observations)} transitions\n")

if RESUME and os.path.exists(PARTIAL_SAVE):
    print(f"Resuming from {PARTIAL_SAVE}...")
    results = np.load(PARTIAL_SAVE, allow_pickle=True).item()
    completed_pairs: set = results.pop("_completed", set())
    print(f"  Already completed {len(completed_pairs)} (seed, sigma) pairs.\n")
else:
    results = _blank_results()
    completed_pairs = set()

for seed in range(N_SEEDS):
    print(f"\n{'='*60}")
    print(f"--- Seed {seed+1}/{N_SEEDS} ---")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    demo = subsample_demo(full_demo, N_DEMOS, rng)

    for sigma in NOISE_STDS:
        key = f"sigma_{sigma}"
        pair_id = f"{seed}_{sigma}"
        if pair_id in completed_pairs:
            print(f"\n  [σ={sigma}] Seed {seed+1} already done — skipping.")
            continue

        print(f"\n  [σ={sigma}] Training BC...")
        policy = train_bc(demo, noise_std=sigma, seed=seed)

        # --- REINFORCE ---
        print(f"  [σ={sigma}] REINFORCE ({REINFORCE_EPISODES} eps)...")
        h = run_reinforce(policy, seed=seed)
        curves = h.get("eval_mean", [])
        results[key]["reinforce"]["curves"].append(curves)
        if results[key]["reinforce"]["eval_at"] is None and h.get("eval_at"):
            results[key]["reinforce"]["eval_at"] = h["eval_at"]
        rf_final = float(np.mean(curves[-3:])) if len(curves) >= 3 else (curves[-1] if curves else 0.0)
        results[key]["reinforce"]["final_mean"].append(rf_final)
        results[key]["reinforce"]["final_std"].append(float(np.std(curves[-3:])) if len(curves) >= 3 else 0.0)
        print(f"  [σ={sigma}] REINFORCE final: {rf_final:.1f}")

        # --- A2C ---
        print(f"  [σ={sigma}] A2C ({A2C_STEPS} steps)...")
        h = run_a2c(policy, seed=seed)
        curves = h.get("eval_mean", [])
        results[key]["a2c"]["curves"].append(curves)
        if results[key]["a2c"]["eval_at"] is None and h.get("step"):
            results[key]["a2c"]["eval_at"] = h["step"]
        a2c_final = float(np.mean(curves[-3:])) if len(curves) >= 3 else (curves[-1] if curves else 0.0)
        results[key]["a2c"]["final_mean"].append(a2c_final)
        results[key]["a2c"]["final_std"].append(float(np.std(curves[-3:])) if len(curves) >= 3 else 0.0)
        print(f"  [σ={sigma}] A2C final: {a2c_final:.1f}")

        # --- PPO ---
        print(f"  [σ={sigma}] PPO ({PPO_TIMESTEPS} steps)...")
        h = run_ppo(policy, seed=seed)
        curves = h.get("eval_mean", [])
        results[key]["ppo"]["curves"].append(curves)
        if results[key]["ppo"]["eval_at"] is None and h.get("eval_at"):
            results[key]["ppo"]["eval_at"] = h["eval_at"]
        ppo_final = h.get("final_mean", 0.0)
        results[key]["ppo"]["final_mean"].append(ppo_final)
        results[key]["ppo"]["final_std"].append(h.get("final_std", 0.0))
        print(f"  [σ={sigma}] PPO final: {ppo_final:.1f}")

        # Incremental checkpoint
        completed_pairs.add(pair_id)
        results["_completed"] = completed_pairs
        np.save(PARTIAL_SAVE, results)  # pyright: ignore[reportArgumentType]
        results.pop("_completed")
        print(f"  [checkpoint] → {PARTIAL_SAVE}")


# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print(f"\n{'='*60}\nSummary\n{'='*60}")
for sigma in NOISE_STDS:
    key = f"sigma_{sigma}"
    r = results[key]
    print(f"\nσ={sigma}:")
    for method in ["reinforce", "a2c", "ppo"]:
        m = r[method]
        if m["final_mean"]:
            print(f"  {method.upper():10s}: {np.mean(m['final_mean']):7.1f} ± {np.std(m['final_mean']):.1f}")

# -------------------------------------------------------
# Save results
# -------------------------------------------------------

os.makedirs("results", exist_ok=True)
save_name = "bc_method_comparison_test.npy" if TEST_MODE else "bc_method_comparison.npy"
np.save(f"results/{save_name}", results)  # pyright: ignore[reportArgumentType]
print(f"\nSaved → results/{save_name}")

# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

os.makedirs("figures", exist_ok=True)
prefix = "bc_method_comparison_test" if TEST_MODE else "bc_method_comparison"
METHOD_COLORS = {"reinforce": "#1f77b4", "a2c": "#2ca02c", "ppo": "#ff7f0e"}
SIGMA_STYLES = {0.0: "-", 0.1: "--"}

# --- Figure 1: Learning curves per method, both sigma values ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, method in zip(axes, ["reinforce", "a2c", "ppo"]):
    for sigma in NOISE_STDS:
        key = f"sigma_{sigma}"
        r = results[key][method]
        eval_at = r["eval_at"]
        curves = r["curves"]
        if not eval_at or not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for j, c in enumerate(curves):
            padded[j, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        x = eval_at[:max_len]
        label = f"σ={sigma}"
        ax.plot(x, mean, label=label, color=METHOD_COLORS[method],
                linestyle=SIGMA_STYLES.get(sigma, "-"), linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=METHOD_COLORS[method])
    ax.axhline(THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax.set_title(method.upper())
    ax.set_xlabel("Steps / Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Mean Return")
fig.suptitle(f"BC Method Comparison: Learning Curves ({N_SEEDS} seeds)", fontsize=13)
plt.tight_layout()
plt.savefig(f"figures/{prefix}_curves.png", dpi=150)
print(f"Saved → figures/{prefix}_curves.png")
plt.close()

# --- Figure 2: Final performance bar chart ---
x = np.arange(len(NOISE_STDS))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 6))
for i, method in enumerate(["reinforce", "a2c", "ppo"]):
    means = [np.mean(results[f"sigma_{s}"][method]["final_mean"]) for s in NOISE_STDS]
    stds  = [np.std(results[f"sigma_{s}"][method]["final_mean"]) for s in NOISE_STDS]
    ax.bar(x + (i - 1) * width, means, width, yerr=stds,
           label=method.upper(), color=METHOD_COLORS[method], capsize=4)
ax.axhline(THRESHOLD, color="k", linestyle="--", linewidth=1, label=f"Threshold ({THRESHOLD})")
ax.set_xticks(x)
ax.set_xticklabels([f"σ={s}" for s in NOISE_STDS])
ax.set_ylabel("Final Mean Return")
ax.set_title(f"Final Performance: REINFORCE vs A2C vs PPO ({N_SEEDS} seeds)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"figures/{prefix}_final_comparison.png", dpi=150)
print(f"Saved → figures/{prefix}_final_comparison.png")
plt.close()

print("\nDone!")
