# Local FE–XPIDON validation workflow

Expected sibling-folder structure:

```text
PIDON-validation-workspace/
├── comparison/
├── validation_fe_reference_v2/
└── XPIDON-Composites-FV1/
```

Activate the environment that can run both XPIDON and FE, then:

```bash
cd ~/Projects/PIDON-validation-workspace/comparison
chmod +x run_all.sh
./run_all.sh
```

## Where to change parameters

### Fixed paper-style studies

Edit `cases.csv`.

- `study=tool_thickness`: keep cure cycle and HTCs fixed; change only `Lt_m`.
- `study=htc_tool`: representative combined HTC/tool sensitivity.
- `study=cure_cycle`: keep HTCs, `Lt_m`, `Lc_m`, and total time fixed; change only the six cure-cycle inputs.

`prepare_cases.py` checks every input against the trained XPIDON ranges and stops before FE/XPIDON execution if a case is outside the range.

### Random zero-shot validation

Edit `comparison_config.json`:

```json
"random_zero_shot": {
  "enabled": true,
  "count": 20,
  "seed": 20260731,
  "interior_margin_fraction": 0.05
}
```

The generated random cases are saved in:

```text
outputs/cases_runtime.csv
```

### Through-thickness validation

Select the case and positions in `comparison_config.json`:

```json
"through_thickness_case_id": "cure_cycle_1"
```

The workflow predicts:

- tool bottom, midpoint, and interface;
- part interface, midpoint, and top.

## Generated figures

```text
figures/study_tool_thickness.png
figures/study_htc_tool.png
figures/study_cure_cycle.png
figures/through_thickness_<case_id>.png
figures/random_zero_shot_validation.png
```

## Generated metric files

```text
outputs/fe_vs_xpidon_metrics.csv
outputs/fe_vs_xpidon_metrics_by_position.csv
outputs/random_zero_shot_case_metrics.csv
outputs/random_zero_shot_summary.csv
outputs/case_range_report.csv
```
