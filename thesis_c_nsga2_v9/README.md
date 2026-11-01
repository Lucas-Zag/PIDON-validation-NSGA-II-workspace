# Thesis C V9 — Reference Pareto Front and Core Figures

Post-processing **only**: no retraining, no new XPIDON calls, no claim of FEA validation or mathematical global optimality.

## Run with uploaded V8 results

```bash
python -m pip install numpy matplotlib
python plot_core_figures.py --input ../convergence_3seeds.zip --output ./results/v9
```

Or supply a directory `--input ../thesis_c_nsga2_convergence_v8/results/convergence_3seeds`.

Outputs: `pareto_time_doc.png`, `hypervolume_convergence.png`, `best_time_convergence.png`, `best_doc_convergence.png`, `cure_profile_comparison.png`, `representative_solutions.csv`, `representative_solutions.json`, `reference_pareto_front.json`, `summary.json`.

Methods: combine *feasible* solutions (minimum DoC at hold-end >=0.82) from all available seeds and compute empirical nondominated reference. Choose shortest time, highest DoC and a compromise nearest the normalized ideal point (per-objective observed min/max scaling). Curves display programmed **air** temperature, not simulated composite temperature. All reported DoC and peak temperature values are surrogate predictions pending FE verification. 

Convergence plots show across-seed mean ±1 standard deviation; hypervolume reference point is the one stored in the source V8 data. This post-processing step does **not** implement automatic stopping or certify convergence. `summary.json` also gives a leave-one-seed-out descriptive IGD+ diagnostic; **this is not an independent optimum and should not be used as proof of global optimality**.

## Research cautions

- Baseline Cycle 1 six inputs are [2.0,108,65,2.0,176,108], the baseline result is read from V8 summary and cross-checked against 251 min. If you change baseline, update input values.
- Comparison to Cycle 1 is across **different feasibility statuses**: baseline does not meet DoC >=0.82. State this clearly in thesis.
- Peak temperature is **monitored**, not an imposed hard upper limit.
- The merged front contains data from seeds being compared, so the reference is an empirical pooled front; leave-one-out is provided as a diagnostic to reduce self-reference.
- Figures are 300 DPI. For paper reproducibility, include the underlying CSV/JSON and experimental configuration in appendix.
