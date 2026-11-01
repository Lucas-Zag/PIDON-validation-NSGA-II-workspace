# Thesis C V11 — Local FE robustness and controlled timing

## What this does

This is **new** validation on the existing frozen V9 optimization designs and V10 reference FE solver. It **does not** retrain XPIDON, rerun NSGA-II, or assert physical manufacturing reliability beyond the 1D numerical FE model. The 0.82 degree-of-cure boundary remains an explicitly research-only requirement.

- Read actual V10 `cases.csv` and `validation_summary.json`, no hand-copy of optimized parameters.
- Perturb each of the six heating schedule variables individually by ±1% and ±2.5% of its configured training-range width. Do not leave the training domain; out-of-bounds perturbations are skipped rather than clipped or duplicated. Nominal point included. Fixed HTC, thickness and total horizon are identical for each case.
- Call real FE `solve_case` and extract **minimum laminate alpha at the exact end of the second hold**, using mesh nodes and time interpolation. Results are labeled FE reference, never unearned experimental validation.
- Optionally evaluate XPIDON on exactly the same finite grid to quantify deviations in perturbed cases (`--with-xpidon`).
- Benchmark FE and XPIDON separately using the existing V10 implementation, with the same heating/hold query grid, preloaded model weights, one discarded warm-up and 3 repeated measurements. Model load/JAX compilation must be measured separately if full end-to-end costs are claimed.
- Output perturbation CSV, JSON, `FE_perturbation_doc.png`, timing JSON and summary.

## Workspace layout

```
PIDON-validation-workspace/
  XPIDON-Composites-FV1/
  validation_fe_reference_v2/
  comparison/
  thesis_c_v10_integrated/run_v10.py
  thesis_c_v11/run_v11.py
  results/validation_run_01/   # unpack the uploaded V10 result here
```

## Usage

Install dependencies in the **already working** local Python environment with JAX and FE; nothing must be re-trained.

```bash
cd thesis_c_v11
python run_v11.py --workspace .. \
    --v10-script ../thesis_c_v10_integrated/run_v10.py \
    --v10-results ../results/validation_run_01 \
    --output ./results/v11_run_01
```

**Optional** to also test the other two representatives and XPIDON deviations:

```bash
python run_v11.py --workspace .. \
    --v10-results ../results/validation_run_01 \
    --v10-script ../thesis_c_v10_integrated/run_v10.py \
    --all-representatives --with-xpidon \
    --output ./results/v11_all_cases_01
```

The output directory must be new; existing results are never overwritten. Run 1 first to prioritize the shortest feasible solution. Prior V10 fields already establish the *nominal* solution is FE feasible but give **no robustness guarantee**.

## Methods/limitations

- One-factor-at-a-time variations provide **local sensitivity**, not robust worst-case over the full six-dimensional uncertainty box, and not a calibrated process uncertainty distribution. Do not report a failure probability from the tested counts.
- No margins are invented. Verify process input tolerances before interpreting perturbation fractions as realistic manufacturing variation.
- All calculations depend on the FE numerical model and its calibration; independently verify mesh/time convergence near hold-end before claiming precision beyond grid tolerances.
- Shortest-feasible is very near DoC 0.82; FE nominal only 0.822895. This study checks whether this is locally fragile.
- The benchmark is conditional on this 1D FE reference (not industrial 3D FE) and a single specific query grid; excludes model loading, JAX compilation, and NSGA-II overhead.
