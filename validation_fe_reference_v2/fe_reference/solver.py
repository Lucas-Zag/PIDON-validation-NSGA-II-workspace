from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

from .mesh import build_layered_mesh
from .assembly import assemble
from .kinetics import cure_rate
from .cure_cycle import CureCycle


@dataclass
class SolveResult:
    case: dict
    cycle: CureCycle
    mesh: object
    matrices: object
    t_s: np.ndarray
    T_C: np.ndarray
    alpha: np.ndarray
    physical_b_K: float
    full_field: pd.DataFrame


def solve_case(case: dict, config: dict, solver_overrides=None) -> SolveResult:
    settings = dict(config['solver'])
    if solver_overrides:
        settings.update(solver_overrides)

    nt = int(settings['n_tool_elements'])
    nc = int(settings['n_part_elements'])
    mesh = build_layered_mesh(float(case['Lt_m']), float(case['Lc_m']), nt, nc)
    hbot = float(case['hbot_W_m2K'])
    htop = float(case['htop_W_m2K'])
    mats = assemble(mesh, config, hbot, htop)
    lu, piv = lu_factor(mats.M)

    cycle = CureCycle.from_case(case, config)
    nT = len(mesh.z)
    nA = len(mesh.part_node_indices)
    T0 = float(config['initial_conditions']['temperature_C'])
    a0 = float(config['initial_conditions']['alpha'])
    bK = float(config['heat_generation']['physical_b_K'])
    y0 = np.concatenate([np.full(nT, T0), np.full(nA, a0)])

    def rhs(t, y):
        T = y[:nT]
        alpha = y[nT:]
        adot = cure_rate(T[mesh.part_node_indices], alpha, config['cure_kinetics'])
        F = np.zeros(nT)
        Ta = cycle.temperature_C(t)
        F[0] += hbot * Ta
        F[-1] += htop * Ta
        F_part = bK * (mats.M_part_local @ adot)
        F[mesh.part_node_indices] += F_part
        dT = lu_solve((lu, piv), F - mats.K_total @ T)
        return np.concatenate([dT, adot])

    output_dt = float(settings['output_dt_s'])
    t_eval = np.arange(0.0, cycle.total_process_time_s + 0.5*output_dt, output_dt)
    if t_eval[-1] > cycle.total_process_time_s:
        t_eval[-1] = cycle.total_process_time_s
    atol = np.concatenate([
        np.full(nT, float(settings['atol_temperature'])),
        np.full(nA, float(settings['atol_alpha'])),
    ])
    sol = solve_ivp(
        rhs,
        (0.0, cycle.total_process_time_s),
        y0,
        method=str(settings['method']),
        t_eval=t_eval,
        max_step=float(settings['max_step_s']),
        rtol=float(settings['rtol']),
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"FE time integration failed for {case['case_id']}: {sol.message}")

    T = sol.y[:nT, :]
    alpha = np.clip(sol.y[nT:, :], 0.0, 1.0)
    rows = []
    for it, t in enumerate(sol.t):
        Ta = cycle.temperature_C(float(t))
        # Tool nodes including interface
        for i in range(mesh.interface_node + 1):
            rows.append({
                'case_id': str(case['case_id']), 'region': 'tool',
                'node_id': int(i), 't_s': float(t), 't_min': float(t/60.0),
                'z_m': float(mesh.z[i]), 'x_local': float(mesh.z[i]/mesh.Lt_m),
                'T_FE_C': float(T[i,it]), 'alpha_FE': np.nan, 'Ta_C': float(Ta),
            })
        # Part nodes including same interface node
        for la, gi in enumerate(mesh.part_node_indices):
            rows.append({
                'case_id': str(case['case_id']), 'region': 'part',
                'node_id': int(gi), 't_s': float(t), 't_min': float(t/60.0),
                'z_m': float(mesh.z[gi]), 'x_local': float((mesh.z[gi]-mesh.Lt_m)/mesh.Lc_m),
                'T_FE_C': float(T[gi,it]), 'alpha_FE': float(alpha[la,it]), 'Ta_C': float(Ta),
            })
    df = pd.DataFrame(rows)
    return SolveResult(case, cycle, mesh, mats, sol.t, T, alpha, bK, df)
