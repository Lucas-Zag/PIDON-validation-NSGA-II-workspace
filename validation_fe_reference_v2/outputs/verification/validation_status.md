# FE reference validation status

## Decision

The solver is ready to be used as the **independent numerical reference** for the next XPIDON comparison stage, subject to the scope statement below.

It is a one-dimensional, through-thickness, linear finite-element implementation of the paper's coupled tool/composite thermochemical model. It is no longer the earlier finite-difference starter.

## Problems corrected

1. The complete two-hold autoclave-air profile is used, including cooling from the second hold to 20 °C.
2. Reaction heat is included as the consistent FE load associated with `b * d(alpha)/dt`.
3. The heat-generation coefficient is not visually fitted. The physical value is recovered from the official XPIDON normalization: `0.7003080410794345 × 250 = 175.07701026985863 K`.
4. The paper-caption case is included and the midpoint temperature now has a clear reaction exotherm.
5. Exotherm is measured only before cooling starts. This avoids falsely treating thermal lag during cooling as reaction heat.

## Main paper-caption case

- Midpoint maximum temperature: **205.348 °C**
- Time of maximum temperature: **149.333 min**
- Reaction exotherm, `T_part - T_air` before cooling: **25.348 °C**
- No-reaction-heat control maximum temperature: **179.902 °C**
- Increase in peak caused by reaction heat: **25.446 °C**
- Final degree of cure: **0.843925**
- Complete process duration: **313.333 min**

## Numerical verification

| Check | Result |
|---|---:|
| Complete cooling cycle | True |
| Alpha monotonic and within [0,1] | True |
| Mesh convergence | True |
| Time-step convergence | True |
| Energy balance | True |
| Positive pre-cooling reaction exotherm | True |
| Reaction heat increases peak | True |
| All configured checks | **True** |

Numerical indicators:

- Change in peak temperature between the two finest meshes: **0.000485 °C**
- Change in peak temperature between the two finest time settings: **0.000004 °C**
- Maximum normalized energy residual: **1.656047e-05**

## Scope and remaining limitation

This establishes **code verification and numerical credibility**, and it is sufficient for a controlled FE-versus-XPIDON comparison using identical equations, parameters, cure cycles, and query points.

It is not evidence that the author's private in-house FE implementation has been reproduced exactly. The paper does not release its FE code, raw FE curves, or complete traceable AS4/8552 material database. Therefore the thesis/report should describe this as an **independent Python FE reference implementation aligned with the published equations and official repository parameters**, not as the author's original FE solver.

A stronger external validation can later be added by comparing one case against ANSYS ACCS, COMPRO/RAVEN, or a traceable experimental thermocouple/DoC dataset.
