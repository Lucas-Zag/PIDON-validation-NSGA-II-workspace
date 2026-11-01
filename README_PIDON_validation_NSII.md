# XPIDON-Assisted Cure Cycle Optimisation and Finite Element Validation

Research workspace for personal project: multi-objective optimisation of composite curing schedules using an Extended Physics-Informed Deep Operator Network (**XPIDON**), **NSGA-II**, and independent **one-dimensional finite element analysis (FEA)**.

This repository collects the predictive model, numerical reference solver, optimisation experiments, Pareto-front analysis, and validation studies in one place. It also contains intermediate versions and exploratory code. **The numbered versions are research milestones, not interchangeable production releases.**

## Research objective

The study investigates whether a physics-informed surrogate can efficiently identify improved two-hold composite curing schedules while retaining acceptable curing performance when candidate solutions are independently checked with FEA.

The current NSGA-II formulation in the V8/V9 experiments:

- **Minimises** heating-and-hold completion time.
- **Maximises** the *minimum through-thickness degree of cure* (DoC) at the end of the second hold.
- Uses **minimum hold-end DoC ≥ 0.82** as a *research-specific feasibility threshold*, not a certified industrial material requirement.
- Optimises **six cure-cycle settings**; other physical inputs remain fixed for the corresponding experiments.
- **Reports** peak composite temperature but **does not impose a hard peak-temperature constraint** in these runs.

The optimisation horizon is the heating-and-hold period, **not** the entire industrial autoclave cycle including cooling. The surrogate-predicted Pareto front is not automatically an FEA-validated front.

## Workflow

```text
Parameterised two-hold cure schedule
              |
              v
     Trained XPIDON surrogate
     (temperature and DoC fields)
              |
              v
       NSGA-II optimisation
     (time / minimum hold-end DoC)
              |
              v
    Pareto analysis and selection
   (shortest / highest DoC / compromise)
              |
              v
 Independent 1D FE recalculation
              |
              v
 Accuracy, feasibility, sensitivity,
        and timing assessment
```

## Repository structure

| Directory | Purpose |
| --- | --- |
| `XPIDON-Composites-FV1/` | XPIDON training, architecture, physics-informed losses, and prediction utilities. |
| `validation_fe_reference_v2/` | Independent 1D thermochemical finite-element reference solver and numerical verification. |
| `comparison/` | Case-based FE–XPIDON comparisons, zero-shot studies, and through-thickness plots. |
| `thesis_c_nsga2_convergence_v8/` | NSGA-II multi-seed and convergence experiments (1,200 / 2,000 / 4,000 evaluation checkpoints). |
| `thesis_c_nsga2_v9/` | Post-processing of V8 runs and empirical pooled Pareto front; does not rerun optimisation. |
| `thesis_c_fea_v10/` | Alternative V10 **case-export and verification interface**; does not run an FEA solver itself. |
| `thesis_c_v10_integrated/` | Integrated XPIDON–FEA comparison workflow using existing model checkpoints and FE solver. |
| `thesis_c_v11_fixed/` | Local FE sensitivity of frozen candidate designs and controlled solver timing measurements. |
| `backup/`, `xpidon_rl_starter/` | Archived or exploratory material; not required for the principal NSGA-II workflow. |

**Note:** Some module-specific READMEs describe historical sibling-folder names and paths from earlier workspace revisions. Use the actual folder names and scripts in this repository and check CLI help/configuration before running.

## Requirements

- Python environment capable of running the XPIDON JAX stack and the SciPy FE solver.
- Dependencies used by the relevant modules, including `numpy`, `scipy`, `matplotlib`, `jax`, and (for NSGA-II) `pymoo`.
- **Previously trained XPIDON checkpoint files** for inference and optimisation/FE comparison. Checkpoints may not be tracked in Git.

Use the individual module's `requirements.txt`, README, and configuration where provided. GPU/JAX installation is hardware-specific; no single installation command is guaranteed for macOS, CPU Linux, or CUDA.

```bash
git clone https://github.com/Lucas-Zag/PIDON-validation-NSGA-II-workspace.git
cd PIDON-validation-NSGA-II-workspace
```

## Reproduction guide

The commands below assume the Python environment is already configured and, where needed, trained XPIDON checkpoints and input results are available.

### 1. Check the independent FE reference

```bash
cd validation_fe_reference_v2
python -m pip install -r requirements.txt
python scripts/run_all.py
cd ..
```

The FE implementation uses 1D linear elements, a shared interface node, Robin convection boundaries, cure-kinetics ODEs, reaction heat, and SciPy's BDF time integrator. Its verification suite includes mesh/time convergence and energy-balance checks. **Numerical verification is not experimental validation.**

### 2. Run baseline and zero-shot FE–XPIDON comparisons

Review `comparison/cases.csv` and `comparison/comparison_config.json` before running. The case-preparation script checks that specified inputs fall inside the XPIDON training ranges.

```bash
cd comparison
chmod +x run_all.sh
./run_all.sh
cd ..
```

Expected reports include `outputs/fe_vs_xpidon_metrics.csv`, positional errors, random zero-shot summaries, and figures under `figures/`.

### 3. Check NSGA-II setup and convergence (V8)

```bash
cd thesis_c_nsga2_convergence_v8
python test_contract.py
python run_convergence.py --seeds 42 --output ./results/convergence_seed42
```

For a three-seed campaign:

```bash
python run_convergence.py --seeds 42 43 44 --output ./results/convergence_3seeds
python analyze_convergence.py --root ./results/convergence_3seeds
cd ..
```

These runs require the trained model assets and the optimisation inputs expected by V8. **The V8 default manifest refers to an earlier `thesis_c_optimization/` workspace, which is not present at this repository root**; update the relevant manifest/path configuration to the available checkpoint before execution. Three 4,000-unique-evaluation seed runs correspond to 12,000 unique candidate evaluations in total. Checkpoints within one seed are snapshots of a single run, not independent optimisations.

### 4. Create thesis figures from existing optimisation results (V9)

```bash
cd thesis_c_nsga2_v9
python -m pip install numpy matplotlib
python plot_core_figures.py \
  --input ../thesis_c_nsga2_convergence_v8/results/convergence_3seeds \
  --output ./results/v9
cd ..
```

V9 pools feasible candidates from supplied seeds and selects the shortest-time, highest-DoC, and normalized-ideal-point compromise solutions. Results are **empirical surrogate-model outcomes**, not a proof of global optimality. The programmed cure curves are **air-temperature profiles**, not simulated laminate-temperature curves.

### 5. Validate representative solutions with independent FEA (integrated V10)

```bash
cd thesis_c_v10_integrated
python run_v10.py \
  --workspace .. \
  --representatives ../thesis_c_nsga2_v9/example_results/representative_solutions.json \
  --output ./results/validation_run_01
cd ..
```

This workflow evaluates the baseline and three representative candidates on matched time–thickness grids, reporting temperature and DoC errors, hold-end minimum DoC, numerical FE feasibility, and timing. Verify the representative JSON, checkpoint paths, material constants, geometry, boundary conditions, and FE configuration. Existing output folders or missing checkpoint weights may require adapting the paths. The module's README notes that its code was initially syntax-checked rather than fully executed in its original development environment; confirm execution locally.

An **alternative** is `thesis_c_fea_v10/v10.py`, which prepares case files and verifies externally supplied XPIDON/FE CSV outputs; **it is not itself an FE solver**. Do not conflate the two V10 routes.

### 6. Local FE sensitivity and timing (V11)

After producing compatible integrated V10 outputs:

```bash
cd thesis_c_v11_fixed
python run_v11.py \
  --workspace .. \
  --v10-script ../thesis_c_v10_integrated/run_v10.py \
  --v10-results ../thesis_c_v10_integrated/results/validation_run_01 \
  --output ./results/v11_run_01
cd ..
```

V11 tests one-factor-at-a-time perturbations of the six cure-cycle inputs and, optionally, XPIDON predictions on the same grid. These tests provide **local numerical sensitivity**, not a calibrated probability of failure or full robust optimisation. The V11 module README uses historical directory names; the command above uses this repository's current `thesis_c_v11_fixed` folder.

## Interpretation and reporting

- **Baseline comparisons:** report the baseline's feasibility status. The V9 README states that the selected Cycle 1 baseline does **not** meet the research DoC ≥ 0.82 threshold, so a direct time-saving percentage is not a like-for-like comparison between feasible schedules.
- **FE verification:** compare representative candidates with matched material properties, physical inputs, time definitions, and spatial orientation. The finite-element model is an *independent numerical reference*, not laboratory evidence.
- **Constraints:** the 0.82 DoC criterion is a research threshold. Peak temperature is a reported metric in the documented V8/V9 formulation, not a constrained objective.
- **Convergence:** report cross-seed variability and checkpoint trends; these do not establish mathematical global optimality.
- **Timing:** distinguish model loading/JAX compilation from steady-state prediction time and FE solve time.
- **Reproducibility:** retain the model/checkpoint identity, input configurations, random seeds, case definitions, raw CSV/JSON outputs, and software environment for the thesis appendix.

## Research background

The XPIDON component builds upon:

Ramezankhani, M. *et al.* (2025). “An advanced physics-informed neural operator for comprehensive design optimization of highly-nonlinear systems: An aerospace composites processing case study.” *Engineering Applications of Artificial Intelligence*, **142**, 109886. https://doi.org/10.1016/j.engappai.2024.109886

Related codebases:

- [XPIDON-Composites-FV1](https://github.com/Lucas-Zag/XPIDON-Composites-FV1)
- [Validation_FE_PIDON](https://github.com/Lucas-Zag/Validation_FE_PIDON)

## Disclaimer

This is an academic research workspace. The model, 1D FE reference, and optimisation outcomes have not been established as certified industrial curing instructions. Material qualifications, manufacturing tolerances, and physical experiments are outside the evidence provided by numerical results alone.
