# Thesis C optimisation — baseline preparation v2

Place this directory in `~/Projects/PIDON-validation-workspace`, beside
`comparison`, `validation_fe_reference_v2` and `XPIDON-Composites-FV1`.

The supplied comparison callers use Python imports, not HTTP ports. New isolated
workers call unchanged XPIDON `pred_T/pred_a` and FE `solve_case`. They do not run
comparison entrypoints, retrain networks or overwrite sibling files. All new
reports remain inside this new project. NSGA-II is the next phase, not implemented
in this preparation package.

## Read the existing workspace

```bash
cd ~/Projects/PIDON-validation-workspace/thesis_c_optimization
python3 prepare.py --workspace ..
```

This standard-library command records source hashes, checkpoint filenames,
reference cases, and existing comparison metrics in `outputs/preparation_report.json`.
Exit code 2 means unresolved preparation items, not a model/optimiser failure.

## Independent baseline prediction

Use the environment which already runs your comparison successfully. No JAX
upgrade is required. `--python /absolute/path/to/python` selects that interpreter.
Explicitly select paired checkpoints from ONE trained run. This example is valid
ONLY if your actual run contains these five paired tags:

```bash
python3 run_baseline.py --workspace .. --backend xpidon --case-id cure_cycle_1 --tags 02000 04000 06000 08000 10000
```

For a single-domain run, use `--tags 10000` instead. Do not choose tags merely from
these examples. `--model-directory /absolute/path/to/models` selects another run
folder. Tag-derived subdomain boundaries reproduce the comparison convention;
confirm exact training boundaries before using results for optimisation.

Optional independent FE check:

```bash
python3 run_baseline.py --workspace .. --backend fe --case-id cure_cycle_1
```

Both read existing `comparison/cases.csv`; repeat for `cure_cycle_2`. Changed
trained part thickness/horizon, out-of-range inputs and invalid cooling are rejected.
Default query grid is 20 s by 21 composite positions, with hold-end also included.
Use `--time-step-s` and `--spatial-points` for grid sensitivity. FE uses existing
verified solver settings then interpolates onto this query grid. Sampled extrema
are not guaranteed continuum bounds. Subdomain seam continuity still needs review.

Each invocation creates a unique project `outputs/` directory containing
`request.json`, `execution.log`, `part_field.csv` and `summary.json`. The summary
records sampled peak temperature, minimum final DoC, DoC spread, hold-end DoC,
physical timing, runtime measurements, dependency versions and input hashes.
XPIDON measures loading, first grid evaluation including JAX compilation, and
warm grid evaluation separately. XPIDON query times do not depend on a FE run.
A later optimiser should retain the loaded model across candidate evaluations.

## Reference conditions and remaining decisions

The two uploaded cure-cycle cases share hbot=70, htop=95 W/(m2 K), Lt=0.04 m,
Lc=0.03 m, and complete duration=18800 s.

| Case | r1 | T1 | hold1 | r2 | T2 | hold2 | Heating/holding end |
|---|---:|---:|---:|---:|---:|---:|---:|
| cure_cycle_1 | 2.0 | 108 | 65 | 2.0 | 176 | 108 | 251.000 min |
| cure_cycle_2 | 2.7 | 116 | 52 | 2.7 | 184 | 102 | 214.741 min |

Ramps are C/min, temperatures C, holds min. Default case 1 is a preparation
reference, not a claim that the supervisor selected the final optimisation baseline.

- Fix two objectives and DoC evaluation time from the final thesis plan.
- Complete duration is fixed at 313.333 min and cannot be a variable time objective.
  Heating/holding end varies; name that metric accurately if chosen as the objective.
- DoC can increase during cooling: full-cycle DoC is not automatically hold-end DoC.
- `T_max=250` is a normalization scale, not a sourced material safety limit.
  Set traceable alpha_min and composite temperature limits separately.
- Review the completed comparison's actual errors, random cases and feasibility.
  Configured input bounds alone do not certify model accuracy.
- Confirm one model run, exact subdomains, seam continuity and spatial/time sensitivity.
- Use XPIDON during NSGA-II search and FE only for selected Pareto verification.

## Verification

```bash
python3 -m unittest discover -s tests -v
```

13 tests passed: units, field extrema, fixed thickness/horizon, bounds, checkpoint
pairing and read-only inspection. See `docs/test_status.md` for the reduced-mesh
FE adapter smoke test. Real XPIDON weights were unavailable here; that backend
must still be verified in your local model environment.
