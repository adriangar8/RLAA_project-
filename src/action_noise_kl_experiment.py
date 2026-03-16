import os
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.policy import PolicyNetwork
from src.bc import BehavioralCloning, BCConfig
from src.rl import REINFORCETrainer, REINFORCEConfig
from src.evaluate import evaluate_policy

# -------------------------------------------------------
# Config
# -------------------------------------------------------
ENV_ID = "LunarLander-v3"
OBS_DIM = 8
ACT_DIM = 4
SEEDS = [0, 1, 2, 3]
ACTION_NOISE_EPS = [0.0, 0.2, 0.3]
KL_COEFS = [0.0, 0.1, 0.5]
REINFORCE_EPISODES = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_bc_policy(eps: float, seed: int) -> PolicyNetwork:
    path = f"/users/eleves-b/2025/ghassan.el-bounni/Documents/RLAA_project-/checkpoints/bc/bc_eps_{eps}_seed_{seed}.pt"
    # Or fallback to kl_best_bc if specific not found for 0.0
    if eps == 0.0 and not os.path.exists(path):
        path = f"/users/eleves-b/2025/ghassan.el-bounni/Documents/RLAA_project-/checkpoints/bc/kl_best_bc_seed_{seed}.pt"

    if not os.path.exists(path):
        print(f"Warning: Could not find BC checkpoint at {path}")
        raise FileNotFoundError(f"Missing BC model {path}")

    policy = PolicyNetwork(OBS_DIM, ACT_DIM)
    cfg = BCConfig(noise_std=0.0)
    trainer = BehavioralCloning(policy, cfg)
    trainer.load(path)
    return policy

def run_reinforce(policy: PolicyNetwork, seed: int, kl_coef: float) -> dict:
    torch.manual_seed(seed)
    cfg = REINFORCEConfig(
        env_id=ENV_ID,
        n_episodes=REINFORCE_EPISODES,
        log_tensorboard=False,
        kl_coef=kl_coef,
        save_dir=f"checkpoints/reinforce_action_noise_{kl_coef}/seed_{seed}",
        device=DEVICE,
    )
    # Move to device handled in trainer
    trainer = REINFORCETrainer(policy, cfg)
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

if __name__ == "__main__":
    
    print(f"\n============================================================")
    print(f"Combined Ablation: Action Noise {ACTION_NOISE_EPS} + KL Penalty {KL_COEFS}")
    print(f"============================================================\n")

    # Structure to store results: results[eps][kl]["curves" or "final"]
    results = {
        eps: {
            kl: {"curves": [], "final": []} for kl in KL_COEFS
        } for eps in ACTION_NOISE_EPS
    }
    eval_at = None

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        for eps in ACTION_NOISE_EPS:
            try:
                base_policy = load_bc_policy(eps, seed)
            except FileNotFoundError:
                print(f"Skipping eps={eps} for seed={seed} due to missing BC model.")
                continue

            for kl in KL_COEFS:
                print(f"  [BC eps={eps}] REINFORCE (KL={kl})...")
                res = run_reinforce(copy.deepcopy(base_policy), seed, kl)
                
                results[eps][kl]["curves"].append(res["eval_mean"])
                results[eps][kl]["final"].append(res["final_mean"])
                
                if eval_at is None and res["eval_at"]:
                    eval_at = res["eval_at"]

    # Save to numpy
    os.makedirs("results", exist_ok=True)
    np.save("results/action_noise_kl_ablation.npy", results)

    # Plotting
    os.makedirs("figures", exist_ok=True)
    
    # We will plot one figure per eps to avoid clutter, or all in one with different line styles
    # Option: Different colors for EPS, different line styles for KL
    fig, ax = plt.subplots(figsize=(12, 8))
    
    eps_colors = {0.0: "red", 0.2: "blue", 0.3: "green"}
    kl_styles = {0.0: ":", 0.1: "-", 0.5: "--"}

    for eps in ACTION_NOISE_EPS:
        for kl in KL_COEFS:
            data = results[eps][kl]
            if not data["curves"]: continue
            curves = data["curves"]
            
            max_len = max(len(c) for c in curves)
            padded = np.full((len(curves), max_len), np.nan)
            for j, c in enumerate(curves):
                padded[j, :len(c)] = c
                
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            
            if eval_at is None:
                x = np.arange(max_len) * 50
            else:
                x = eval_at[:max_len]
            
            lbl = f"Eps={eps} | KL={kl} | Final: {np.mean(data['final']):.1f}"
            ax.plot(x, mean, label=lbl, color=eps_colors[eps], linestyle=kl_styles[kl], linewidth=2)
            
            # Less alpha for standard deviation to not clutter completely
            ax.fill_between(x, mean - std, mean + std, alpha=0.1, color=eps_colors[eps])

    ax.axhline(200.0, color="k", linestyle="-.", linewidth=1, label="Threshold (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Return")
    ax.set_title("Combined Ablation: Action Noise BC + KL-REINFORCE")
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/action_noise_kl_ablation_curves.png", dpi=150, bbox_inches="tight")
    print("\nSaved plot to figures/action_noise_kl_ablation_curves.png")
    
    print("\nFinal Results Summary (Mean ± Std):")
    for eps in ACTION_NOISE_EPS:
        for kl in KL_COEFS:
            data = results[eps][kl]
            if data["final"]:
                print(f"  Eps={eps}, KL={kl}: {np.mean(data['final']):.1f} ± {np.std(data['final']):.1f}")
