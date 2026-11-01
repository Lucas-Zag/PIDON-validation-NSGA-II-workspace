# Thesis C — NSGA-II Convergence Experiment (V8)

This **extends V7**, retaining the *same* XPIDON weights, design domain, evaluation function and research constraint. SAC is not used.

## Optimization definition

- Objectives: minimize heating-and-hold completion time (minutes); maximize **minimum** thickness DoC **at the same hold-end time**.
- Experimental feasibility: hold-end minimum DoC ≥ 0.82.
- Peak composite temperature is **reported only**; no unsubstantiated temperature limit is imposed.
- Only six heating-cycle inputs are optimized; the remaining inputs stay fixed as specified in the model/baseline.
- This is not a complete industrial autoclave cycle including cooling; FEA validation is still pending.

## Setup

Place `thesis_c_nsga2_convergence_v8` next to `XPIDON-Composites-FV1`, `comparison`, and `thesis_c_optimization`. Use the **same virtual environment that successfully ran V7** and install `numpy`, `matplotlib`, and `pymoo` as needed.

Run contract tests first:

```bash
python test_contract.py
```

Single-seed experiment, suitable as an initial 4,000-evaluation convergence test:

```bash
python run_convergence.py --seeds 42 --output ./results/convergence_seed42
```

Multi-seed stability experiment (each seed = one full 4,000-evaluation run):

```bash
python run_convergence.py --seeds 42 43 44 --output ./results/convergence_3seeds
```

By default, manifest is:

`../thesis_c_optimization/outputs/20261001T070827907494Z_preoptimization_xpidon/model_manifest.json`

and cases is `../comparison/cases.csv`, with baseline `cure_cycle_1`.

**Important:** Running `--seeds 42 43 44` performs **12,000 unique candidate evaluations total** (4,000 per seed). Checkpoints at 1,200/2,000/4,000 are **snapshots of the same run**, not three independently rerun optimizations. Every seed has a distinct output directory and seed. Run `--skip-existing` to finish an interrupted campaign if completed seed directories already exist. An interrupted individual seed directory is not resumable with this script.

## Automatic output

Within each `seed_N/` (provided by V7):

- `optimization_dashboard.png` — updated every five generations
- `pareto_time_doc.png`
- `convergence.jsonl`, `all_evaluations.jsonl`
- `pareto_front.json`, `representative_solutions.json`, `summary.json`, `configuration.json`

At the experiment root:

- `checkpoint_metrics.csv` — *exact* 1,200 / 2,000 / 4,000 checkpoint metrics for each seed
- `convergence_summary.json` — descriptive means and sample standard deviations across seeds
- `convergence_comparison.png` — hypervolume, best feasible time, highest feasible DoC, wall time
- `pareto_checkpoint_comparison.png` — fronts from actual checkpoint candidates
- `seed_N.log` — complete text training/optimization log

Regenerate comparative figures from completed runs without new XPIDON evaluations:

```bash
python analyze_convergence.py --root ./results/convergence_3seeds
```

## Judging convergence

Compare 1,200 → 2,000 → 4,000 evaluation checkpoints for fixed-reference hypervolume, best feasible time, best feasible DoC, and across-seed variation. A small change alone does not mathematically prove convergence; if indicators keep improving noticeably, consider a longer run. Do not fabricate accuracy, FEA errors, or runtime claims. Report surrogate-only status until independently validated.

## Caveat

V7's `run_nsga2.py` counts **unique evaluated parameter vectors**; cache hits are separately tracked as requested evaluations. The reference point for hypervolume is inherited from V7: `(320 min, -DoC=0)` in minimization coordinates. Exact checkpoint matching uses `population=40`; do not change it to a value that does not divide all checkpoints. User should verify model weights, units, and material constraints before engineering use.
