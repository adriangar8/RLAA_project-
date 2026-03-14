# BC Data Augmentation Experiments

## Overview

This series of experiments investigates whether adding Gaussian noise to observations
during Behavioral Cloning (BC) training reduces covariate shift and improves downstream
RL fine-tuning on `LunarLander-v3`.

**Hypothesis:** Augmenting observations with small Gaussian noise (σ > 0) during BC
training forces the policy to learn more robust features, reducing the distribution
mismatch between training demos and online rollouts (covariate shift).

---

## V1 Results — `bc_aug_results.npy`

**Script:** `src/bc_aug_reinforce.py`
**Conditions:** Plain BC (σ=0.0) vs Augmented BC (σ=0.1)
**Seeds:** 5 per condition
**Fine-tuning:** REINFORCE, 2000 episodes

| Condition | BC-only Return | Covariate Shift | REINFORCE Final Return |
|-----------|---------------|-----------------|------------------------|
| Plain BC (σ=0.0) | -37.6 ± 95.1 | 0.745 | **56.1 ± 28.9** |
| Aug BC (σ=0.1)  | -204.1 ± 64.7 | 0.878 | **101.4 ± 54.4** |

**Key finding:** Augmentation (σ=0.1) improves REINFORCE final performance by ~80%
(56 → 101), despite lower BC-only score and slightly higher measured covariate shift.
This suggests augmentation trades immediate imitation quality for better RL adaptability.

**Figure:** `figures/bc_aug_comparison.png`

### Loading V1 Results

```python
import numpy as np

data = np.load("results/bc_aug_results.npy", allow_pickle=True).item()
# Keys per condition: bc_mean, bc_std, shift, rl_curves, rl_final_mean, rl_final_std
# data["plain"]["rl_final_mean"]  -> 56.1
# data["aug"]["rl_final_mean"]    -> 101.4
# data["eval_at"]                 -> array of episode numbers (50..2000)
```

---

## V2 Test Results — `bc_aug_v2_test_results.npy`

**Script:** `src/bc_aug_experiment_v2.py --test`
**Status:** Test run only (1 seed, reduced episodes). Full run interrupted by system reboot.
**Noise sweep:** [0.0, 0.01, 0.05, 0.1, 0.2]
**Fine-tuning:** REINFORCE (5000 eps) + PPO (100k steps) — reduced in test mode

The test run validates the pipeline end-to-end. Results are not statistically meaningful
(1 seed, short training). See figures `figures/bc_aug_v2_test_*.png` for visual output.

### Loading V2 Results

```python
import numpy as np

data = np.load("results/bc_aug_v2_test_results.npy", allow_pickle=True).item()
# Keys: one entry per noise level, e.g. "sigma_0.0", "sigma_0.1", etc.
# Each entry contains: noise_std, bc_mean, bc_std, shift,
#                      reinforce_curves, reinforce_final_mean, reinforce_final_std,
#                      ppo_curves, ppo_final_mean, ppo_final_std,
#                      reinforce_eval_at, ppo_eval_at
```

---

## How to Reproduce

### V1 Experiment
```bash
conda activate il_rl
python -m src.bc_aug_reinforce
# Output: results/bc_aug_results.npy, figures/bc_aug_comparison.png
```

### V2 Experiment — Quick Test (~2 min)
```bash
conda activate il_rl
python -m src.bc_aug_experiment_v2 --test
# Output: results/bc_aug_v2_test_results.npy, figures/bc_aug_v2_test_*.png
```

### V2 Experiment — Full Run (long, background)
```bash
nohup bash -c 'eval "$(conda shell.bash hook 2>/dev/null)" && conda activate il_rl && PYTHONUNBUFFERED=1 python -m src.bc_aug_experiment_v2' > logs/bc_aug_v2.log 2>&1 &
tail -f logs/bc_aug_v2.log   # monitor progress
# Output: results/bc_aug_v2_results.npy, figures/bc_aug_v2_*.png
```

---

## Implementation Notes

- Noise augmentation is implemented in `src/bc.py` via `BCConfig.noise_std`.
  Set `noise_std > 0` to enable; noise is applied to training observations only
  (validation and rollout evaluation use clean observations).
- PPO fine-tuning requires `net_arch=[256, 256]` in `policy_kwargs` to match BC's
  `PolicyNetwork` hidden dims (see `src/bc_aug_experiment_v2.py`).

---

## Next Steps

- Re-run full V2 experiment (10 seeds × 5 noise levels × REINFORCE + PPO).
- Expected outcome: identify the optimal noise level σ* in [0.01, 0.2] that maximizes
  final RL return, and quantify whether PPO or REINFORCE benefits more from augmentation.
