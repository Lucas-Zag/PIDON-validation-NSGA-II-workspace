# Validation-grade 1D finite-element reference solver

This project provides an independent Python reference solution for the composite-tool thermochemical curing problem used by the XPIDON study.

It is intentionally **not connected to the trained XPIDON model yet**. The current goal is to establish a trustworthy numerical reference before model comparison.

## Numerical method

- 1D linear finite elements in physical through-thickness coordinate `z`
- Tool and composite share one interface node, so temperature continuity is imposed strongly
- Conductive flux continuity follows from the assembled weak form
- Consistent heat-capacity mass matrix
- Robin convection at bottom and top surfaces
- 8552 cure-kinetics ODE at all composite nodes
- Consistent finite-element cure heat load
- Stiff time integration using SciPy `solve_ivp(method="BDF")`
- Complete two-hold cure cycle including final cooling to the initial temperature

## Parameter alignment

The default configuration uses the constants published in the official XPIDON repository:

- composite diffusivity: `3.7388048e-7 m2/s`
- tool diffusivity: `3.1276314e-6 m2/s`
- composite conductivity: `0.6386316 W/(m K)`
- tool conductivity: `13 W/(m K)`
- cure kinetics constants from the paper/repository
- total process time: `18800 s`
- temperature scale: `250 C`
- repository dimensionless heat coefficient: `0.7003080410794345`

The physical heat-generation coefficient is recovered as:

```text
b_physical = b_repository * T_max
           = 0.7003080410794345 * 250
           = 175.07701026985863 K
```

This avoids visually tuning the exotherm peak.

## Run

```bash
cd validation_fe_reference_v2
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\\Scripts\\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

python scripts/run_all.py
```

## Outputs

```text
outputs/inputs/cure_cycle_sensors.csv
outputs/reference/fe_full_field.csv
outputs/reference/fe_key_curves.csv
outputs/reference/fe_metrics.csv
outputs/verification/verification_summary.csv
outputs/verification/mesh_convergence.csv
outputs/verification/time_convergence.csv
outputs/verification/energy_balance.csv
outputs/figures/
```

## Acceptance criteria used by the verification suite

- complete cure cycle ends at initial temperature
- alpha stays in `[0,1]` and is monotonic
- interface temperature is exactly continuous by construction
- mesh-convergence change in midpoint peak temperature is below the configured tolerance
- time-convergence change in midpoint peak temperature is below the configured tolerance
- normalized energy-balance residual is below the configured tolerance
- positive reaction exotherm is present at the composite midpoint before cooling begins
- a paired no-reaction-heat control confirms that `b*dalpha/dt` raises the peak temperature

Passing these checks makes the solver suitable as a **numerically verified reference implementation** for XPIDON comparison. Final experimental/material validation still requires a traceable AS4/8552 material-property source or an independent ANSYS/ACCS/COMPRO benchmark.
