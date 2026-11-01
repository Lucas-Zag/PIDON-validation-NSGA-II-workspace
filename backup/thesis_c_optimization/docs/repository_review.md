# Review — 2026-10-01, updated with actual comparison callers

Reviewed repository snapshots:
- XPIDON: https://github.com/Lucas-Zag/XPIDON-Composites-FV1/tree/160ac3fe1839c6eae255c9816f940ff95e624ff9
- FE: https://github.com/Lucas-Zag/Validation_FE_PIDON/tree/af646d40afccc854d30d428b464373eed3265a1d

All eight supplied comparison files were read. Originals are unchanged; hashes
are in uploaded_source_hashes.json. The supplied scripts insert sibling paths into
sys.path and call Python APIs, not HTTP services. Workers use those APIs in isolated
processes with Python bytecode writes disabled and new-project-only outputs.

XPIDON has nine inputs, fixed composite thickness and fixed 18800 s horizon.
Paired checkpoints are reconstructed into subdomains using filename tags. New
code requires explicit selection from one run, validates fixed conditions and
input bounds, and preserves physical scaling and local normalized query time.
Tag-derived intervals need checking against exact original training intervals.

The current comparison uses FE time points for XPIDON inference. New baseline
queries are independent, eliminating FE from the candidate-evaluation path. New
metrics use a sampled composite field, not only the midpoint. Grid sensitivity
and seam continuity are not yet certified.

Six manual cases cover tool thickness, combined HTC/tool changes, and cure cycles.
Only the two cure-cycle cases share fixed HTC and geometry while varying schedules.
Both use hbot=70, htop=95, Lt=0.04, Lc=0.03, and complete time 18800 s. Their
heating/holding ends are 251 and 214.740741 min. These timings alone establish
neither temperature safety nor DoC feasibility.

The supplied config enables 20 random zero-shot cases, seed 20260731, interior
margin 0.05. Runtime cases/outputs were not supplied, so their execution and model
accuracy have not been independently confirmed. The main comparison writes
errors by position and random summaries; preparation reads those existing files.

Two code details are relevant to interpreting prior figures, without changing
that completed project: compare_and_plot.py uses an inner join, so row uniqueness
alone does not prove complete FE-row retention. The older standalone
plot_cure_cycle_study.py does not filter part_mid before plotting combined rows,
while compare_and_plot.py correctly filters part_mid. This plotting distinction
is not evidence that underlying numerical validation failed.

FE is an independent numerically verified reference aligned to published physics.
Its repository does not claim experimental material validation or the author's
unreleased original FE implementation. Real weights and comparison result CSVs
were absent here. No XPIDON accuracy or feasible Pareto front is claimed.
