# Thesis C V10 — Independent FEA Validation Interface

## Scope
V10 is a **validation preparation and post-processing tool**, not an FEA solver. It never claims validation without independent FEA fields. It reuses V9's frozen **Shortest feasible**, **Highest DoC**, and **Compromise** candidates, plus Cycle 1 baseline. Do not retrain XPIDON, rerun NSGA-II, or change the 0.82 experimental research threshold.

**Important:** The fixed `htc_bottom=70`, `htc_top=95`, `tool_thickness=0.04 m`, `composite_thickness=0.03 m`, and `initial=20°C` values are prior project settings. Verify against `comparison/cases.csv`, the original FE setup, and XPIDON configuration **before launching FEA**. If different, pass the appropriate arguments. Verify cure kinetics, resin/material properties, mesh, boundary conditions and heating programme match before citing results.

## Install
```
python -m pip install numpy matplotlib
```

## Step 1 — Prepare 4 reproducible cases
Place `thesis_c_fea_v10` alongside `thesis_c_nsga2_v9` and other project folders.
```
python v10.py prepare \
  --representatives ../thesis_c_nsga2_v9/example_results/representative_solutions.json \
  --output ./results/v10_cases
```
Produces `cases.csv`, `cases_index.json`, and per-case `case.json` including six heat parameters, hold-end time, fixed HTC and thickness assumptions, and a SHA256 case fingerprint.

## Step 2 — Run XPIDON and independent FEA for each case
This package **does not run either solver**. Use the existing repository's field extraction or your validated FEA solver, with the exact same input and conditions from each `case.json`.

Put field outputs here:
```
results/v10_cases/cases/Baseline/xpidon.csv
results/v10_cases/cases/Baseline/fea.csv
results/v10_cases/cases/Shortest_feasible/xpidon.csv
results/v10_cases/cases/Shortest_feasible/fea.csv
results/v10_cases/cases/Highest_DoC/xpidon.csv
results/v10_cases/cases/Highest_DoC/fea.csv
results/v10_cases/cases/Compromise/xpidon.csv
results/v10_cases/cases/Compromise/fea.csv
```
All CSVs use this schema (existing `part_field.csv` XPIDON convention):
```
t_s,x_norm,T_part_C,alpha
0.0,0.0,20.0,0.05
0.0,0.5,20.0,0.05
0.0,1.0,20.0,0.05
...,...,...,...
```
- `t_s` is **seconds after cure programme starts**; `x_norm` is **0–1 across composite thickness**, not tool thickness. Spatial orientation must be the same.
- Both files must contain a complete rectangular time × spatial grid and cover **0 to hold-end** with positions spanning the full laminate thickness. No time/space extrapolation is performed.
- Ideally export full-resolution FEA time and space grids; XPIDON samples are interpolated onto the FEA grid. The hold-end time is inserted exactly.
- Export composite part temperature, **not air or tool temperature**.
- If your FEA files are in other formats, export to this neutral CSV contract first. The code intentionally avoids guessing unit conversions, coordinate alignment, solver implementations or normalization.

## Step 3 — Numerical comparison and paper figures
```
python v10.py verify \
  --input ./results/v10_cases \
  --output ./results/v10_validation
```
Outputs:
- `validation_summary.json`, `validation_summary.csv`: per-case status and temperature/DoC MAE/RMSE, bias, maximum error, and FEA hold-end minimum DoC >= 0.82.
- `<case_id>/temperature_doc_evolution.png`: XPIDON vs FEA max part temperature and minimum through-thickness DoC vs time.
- `<case_id>/error_evolution.png`: MAE over laminate thickness vs time.
- `<case_id>/metrics.json`: detailed metrics and fixed case fingerprint.

Absent FEA is reported as `PENDING`; invalid data is `ERROR`. No result is fabricated. Both fields are compared **only during heating/hold [0, hold-end]**; full simulation peak temperature from V9 can differ because it includes subsequent cooling. Independent FEA material/mesh/provenance verification is a separate required reviewer check.

## Thesis C research interpretation
- `0.82` is an **experimental benchmark**, not certified material acceptance.
- Shortest feasible is near the XPIDON threshold; test FEA compliance separately.
- Baseline is known to be below 0.82 in XPIDON, so comparisons must disclose this quality difference.
- V10 does not establish robustness or certified process safety. Next add FE uncertainty margins only using real error data.
