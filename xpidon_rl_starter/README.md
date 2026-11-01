# XPIDON + SAC V3: reward/action refinement and training diagnostics

This is a **standalone sibling directory** of `XPIDON-Composites-FV1`, not a change to the neural operator or its frozen checkpoints. This is **offline design optimization**, not live autoclave control. FE verification is not included and research thresholds are **not engineering certification**.

## Directory

```text
PIDON-validation-workspace/
├── XPIDON-Composites-FV1/         # Existing model and outputs/models/*.pkl
├── comparison/cases.csv
├── thesis_c_optimization/        # Existing manifest may be here
└── xpidon_rl_v3/
    ├── predictor.py             # Persistent checkpoint loader, unchanged from V2
    ├── rl_env.py                # V3 environment, also supports V2 scoring
    ├── train_sac.py             # SAC with monitor, seed and periodic evaluation
    ├── evaluate_sac.py          # Post-training SAC / random / no-change
    ├── plot_training.py
    └── test_environment.py
```

## Setup

Activate your existing XPIDON Python environment. Install the RL dependencies (JAX and weights remain unchanged):

```bash
python -m pip install gymnasium stable-baselines3 tensorboard matplotlib
```

## Run V3 smoke test (new output folder)

Run these commands from the `xpidon_rl_v3` directory; update the manifest path to the copy you have locally:

```bash
python test_environment.py
python train_sac.py \
  --manifest ../thesis_c_optimization/outputs/model_manifest.json \
  --cases ../comparison/cases.csv \
  --baseline cure_cycle_1 \
  --steps 100 --eval-freq 50 --eval-episodes 5 \
  --seed 42 --output ./rl_results_v3_smoke
```

Use the **actual** manifest path; it can live anywhere. Manifest workspace and model directory must point to your existing local files. Don't overwrite or regenerate the model weights.

### After smoke succeeds: a diagnostic-length run

```bash
python train_sac.py \
  --manifest ../thesis_c_optimization/outputs/model_manifest.json \
  --cases ../comparison/cases.csv \
  --baseline cure_cycle_1 \
  --steps 1000 --eval-freq 250 --eval-episodes 10 \
  --seed 42 --output ./rl_results_v3_seed42

python plot_training.py --input ./rl_results_v3_seed42/eval_history.csv \
  --output ./rl_results_v3_seed42/training_diagnostics.png

python evaluate_sac.py \
  --manifest ../thesis_c_optimization/outputs/model_manifest.json \
  --cases ../comparison/cases.csv --baseline cure_cycle_1 \
  --model ./rl_results_v3_seed42/sac_model.zip \
  --run-info ./rl_results_v3_seed42/run_info.json \
  --episodes 20 --output ./rl_eval_v3_seed42
```

For multiple seeds, repeat the **same** command using `--seed 43 --output ./rl_results_v3_seed43`, then `--seed 44 --output ./rl_results_v3_seed44`; compare held-out results and count all training + evaluation predictions.

## V3 changes and controls

- **Score**: `-hold_end_min/250 - 2*max(0,alpha_req-alpha_min)/0.02 - 2*max(0,peak_temp-temp_limit)/5 + 0.2*(feasible)`; a **research** objective only.
- **Reward**: change in score between consecutive parameter edits. Score is *not* an episode return. Both should be reported separately.
- **Actions**: still six continuous normalized edits, now linearly reduced from `action_fraction` to `action_fraction*(1-action_decay)`. Defaults: 0.12, 0.5, 8 steps.
- **Validity**: clamp within XPIDON training input bounds; invalid proposal returns penalty and leaves current design unchanged.
- **Metrics**: `train_monitor.csv.monitor.csv` (Gym Monitor format), `eval_history.csv` (fixed held-out starts), TensorBoard scalars in `tensorboard/`, saved `sac_model.zip`, `run_info.json`.
- **Control**: `--reward-mode v2 --action-decay 0` approximates previous reward/action setup. Create independent new models for true ablation; loading a V2 model for a V3 environment is not a valid identical-configuration comparison.

## Caveats to investigate

- `predictor.py` is the same as V2, with **no changes to JAX weights**. A passing script does not verify numerical agreement with a reference FE/surrogate baseline; use `evaluate_sac.py --reference-summary <baseline_summary.json>` when available.
- Periodic held-out policy tests consume extra XPIDON calls. Compare optimizers with *equal total* evaluation budgets, including data collection and training.
- A reward trajectory improving over 8 design-refinement steps is **not** evidence of training convergence.
- More RL network complexity does not automatically mean better optimization. Check feasibility, short cycle times, run-to-run variance, and FE agreement.
