# Thesis C V10 — Integrated XPIDON–FEA validation

This integrates with the existing sibling folders and **does not modify them**:

```
PIDON-validation-workspace/
  XPIDON-Composites-FV1/
  validation_fe_reference_v2/     # clone of Validation_FE_PIDON under existing comparison-compatible name
  comparison/                     # clone of xpidoncompare
  thesis_c_nsga2_v9/
  thesis_c_v10_integrated/
```

Keep the previously trained XPIDON checkpoint files under `XPIDON-Composites-FV1/outputs/models/`.

## Execute

Activate the environment already used for `comparison`. From `thesis_c_v10_integrated`:

```
python run_v10.py \
  --workspace .. \
  --representatives ../thesis_c_nsga2_v9/example_results/representative_solutions.json \
  --output ./results/validation_run_01
```

If source repository folder is named `Validation_FE_PIDON` instead of `validation_fe_reference_v2`, rename the folder or adjust the one location declaration in this script. This script imports original `run_xpidon_cases.py` functions to load subdomain checkpoints, and original `fe_reference` code to calculate independent FE fields. No original code is overwritten.

It exports comparison fields on the same time × normalized-thickness grid through the end of second dwell (10 s spacing by default, 81 spatial positions), and includes the exact hold-end time even if off-grid. Full-horizon post-cooling peaks **are not** mixed with hold-end-based metrics. Four cases: Baseline, Shortest feasible, Highest DoC, Compromise.

Reports include Temperature/DoC MAE and RMSE, hold-end minimum DoC for XPIDON and FEA, feasibility against the **research-only** 0.82 threshold, prediction/FE solve times, 300-dpi figures and raw matched fields.

### Critical research checks
- Verify `comparison/cases.csv` baseline HTC, tool and part thickness and model horizon match FE configuration and XPIDON training configuration.
- The FEA implementation is an independent 1D reference implementation; its physics, BCs, and kinetics must match XPIDON assumptions before physical validation claims.
- The optimizer used a finite sampled thickness grid. The FE-validation grid supports 81 positions and can be refined if the threshold margin is very small.
- Best-feasible from XPIDON sits extremely close to DoC=0.82; independent FE feasibility is a **new result**, not guaranteed.
- This code has been syntax-checked but **not end-to-end executed with local XPIDON checkpoint weights or FEA runtime**, which are not available in this environment.
