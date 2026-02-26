# Imitation Learning + RL Fine-tuning
### CSC_52081_EP — Reinforcement Learning and Autonomous Agents
*École Polytechnique, Institut Polytechnique de Paris — 2026*

---

## Project Overview

We study the combination of **Behavioral Cloning (BC)** and **RL fine-tuning (REINFORCE)** on the `LunarLander-v2` environment. The central claim:

> Expert demonstrations bootstrap the policy efficiently, but suffer from **covariate shift** — errors compound at test time because the agent visits states outside the expert's distribution. RL fine-tuning corrects this by optimizing under the agent's own state distribution.

### Conditions compared
| Condition | Description |
|-----------|-------------|
| **RL from scratch** | Random init → REINFORCE |
| **BC only** | Supervised learning on demonstrations, no RL |
| **BC + REINFORCE** | BC init → REINFORCE fine-tuning *(main method)* |
| **BC + A2C** | BC init → A2C fine-tuning *(extension)* |

### Ablations
- **Demo dataset size**: N ∈ {100, 500, 1000, 2000} demonstrations
- **Expert quality**: Expert checkpoints at 50k / 200k / 500k training steps
- **Covariate shift analysis**: State distribution divergence for BC vs BC+RL

---

## Repository Structure

```
il_rl_project/
├── src/
│   ├── policy.py       # PolicyNetwork and ActorCriticNetwork
│   ├── expert.py       # Expert training + demonstration collection
│   ├── bc.py           # Behavioral Cloning trainer
│   ├── rl.py           # REINFORCE with baseline + A2C
│   └── evaluate.py     # Metrics and plotting utilities
├── experiments/
│   └── run_ablations.py  # All experiments (CLI)
├── scripts/
│   └── sanity_check.py   # Quick end-to-end test (CartPole)
├── notebooks/
│   └── (analysis notebooks go here)
├── results/              # Saved experiment results (.npy)
├── figures/              # Saved plots
├── checkpoints/          # Model checkpoints
│   ├── expert/
│   ├── bc/
│   └── reinforce/
└── data/                 # Saved demonstration datasets
```

---

## Setup

```bash
# Create environment
conda create -n il_rl python=3.11
conda activate il_rl

# Install dependencies
pip install -r requirements.txt

# Verify setup (< 2 min, uses CartPole)
python scripts/sanity_check.py
```

---

## Running Experiments

### Step 1 — Verify pipeline
```bash
python scripts/sanity_check.py
```

### Step 2 — Run main comparison (trains expert + 3 conditions × 5 seeds)
```bash
python -m experiments.run_ablations --experiment main --n_demos 200
```

### Step 3 — Demo size ablation
```bash
python -m experiments.run_ablations --experiment demo_size \
    --expert_path checkpoints/expert/best/best_model
```

### Step 4 — Expert quality ablation
```bash
python -m experiments.run_ablations --experiment expert_quality
```

### Run everything
```bash
python -m experiments.run_ablations --experiment all
```

---

## Monitoring

TensorBoard logs are saved under `checkpoints/*/tb_logs/`:
```bash
tensorboard --logdir checkpoints/
```

---

## Key Design Decisions

**Why PPO for the expert?** PPO is robust and SB3's implementation is well-tested. This lets us focus on BC+RL rather than expert training. We save intermediate checkpoints for the quality ablation.

**Why REINFORCE (not PPO) for fine-tuning?** The course covers REINFORCE explicitly. Using it demonstrates understanding of the policy gradient theorem. A2C is included as an extension.

**Why LunarLander?** Discrete actions (4), non-trivial dynamics, no vision, well-documented performance threshold (200+). Fast enough for 5-seed experiments on a single GPU.

**Why 5 seeds?** The lecturer explicitly mentioned statistical significance and confidence intervals. 5 seeds gives meaningful variance estimates without excessive compute.

---

## References

- Ross, S., Gordon, G., & Bagnell, D. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS. (DAgger)
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature. (DQN)
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv.
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning.* Machine Learning.
