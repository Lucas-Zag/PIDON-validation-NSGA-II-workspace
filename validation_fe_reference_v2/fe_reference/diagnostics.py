import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid


def _interp_node_history(z, values, target_z):
    # values shape: nnode x nt
    out = np.empty(values.shape[1])
    for j in range(values.shape[1]):
        out[j] = np.interp(target_z, z, values[:,j])
    return out


def key_curves(result) -> pd.DataFrame:
    m = result.mesh
    t = result.t_s
    targets = [
        ('tool_bottom','tool',0.0),
        ('tool_mid','tool',0.5*m.Lt_m),
        ('tool_interface','tool',m.Lt_m),
        ('part_interface','part',m.Lt_m),
        ('part_mid','part',m.Lt_m+0.5*m.Lc_m),
        ('part_top','part',m.Lt_m+m.Lc_m),
    ]
    rows=[]
    for name, region, zt in targets:
        Tline = _interp_node_history(m.z, result.T_C, zt)
        if region == 'part':
            zpart = m.z[m.part_node_indices]
            aline = _interp_node_history(zpart, result.alpha, zt)
        else:
            aline = np.full_like(Tline, np.nan)
        for j, ts in enumerate(t):
            rows.append({
                'case_id': result.case['case_id'], 'position': name, 'region': region,
                't_s': float(ts), 't_min': float(ts/60.0), 'z_m': float(zt),
                'T_FE_C': float(Tline[j]), 'alpha_FE': float(aline[j]) if region=='part' else np.nan,
                'Ta_C': float(result.cycle.temperature_C(float(ts))),
            })
    return pd.DataFrame(rows)


def case_metrics(result) -> dict:
    key = key_curves(result)
    mid = key[key.position=='part_mid'].sort_values('t_s').copy()
    imax = mid['T_FE_C'].idxmax()

    # A reaction exotherm must be assessed before the imposed cooling ramp.
    # Otherwise a thermally lagging part can remain hotter than cooling air and
    # produce a misleadingly large T_part - T_air value unrelated to cure heat.
    hold2_end_s = result.cycle.breakpoints_min[3] * 60.0
    reaction_window = mid[mid['t_s'] <= hold2_end_s + 1e-9]
    exotherm = reaction_window['T_FE_C'] - reaction_window['Ta_C']
    iex = exotherm.idxmax()

    dadt = np.gradient(mid['alpha_FE'].to_numpy(), mid['t_s'].to_numpy())
    i_rate = int(np.nanargmax(dadt))
    return {
        'case_id': result.case['case_id'],
        'part_mid_Tmax_C': float(mid.loc[imax,'T_FE_C']),
        'part_mid_Tmax_time_min': float(mid.loc[imax,'t_min']),
        'part_mid_reaction_exotherm_C': float(exotherm.max()),
        'part_mid_reaction_exotherm_time_min': float(mid.loc[iex,'t_min']),
        'part_mid_peak_above_hold2_C': float(mid.loc[imax,'T_FE_C'] - result.cycle.hold2_T_C),
        'part_mid_final_alpha': float(mid['alpha_FE'].iloc[-1]),
        'part_mid_max_cure_rate_1_s': float(dadt[i_rate]),
        'part_mid_max_cure_rate_time_min': float(mid['t_min'].iloc[i_rate]),
        'cooling_rate_C_min': float(result.cycle.cooling_rate_C_min),
        'hold2_end_min': float(result.cycle.breakpoints_min[3]),
        'process_end_min': float(result.cycle.total_process_time_s/60.0),
    }


def energy_balance(result) -> pd.DataFrame:
    m=result.mesh; mats=result.matrices; t=result.t_s; T=result.T_C; alpha=result.alpha
    ones=np.ones(len(m.z))
    sensible = np.array([ones @ mats.M @ T[:,j] for j in range(T.shape[1])])
    sensible_change = sensible - sensible[0]

    Ta=np.array([result.cycle.temperature_C(float(x)) for x in t])
    hbot=float(result.case['hbot_W_m2K']); htop=float(result.case['htop_W_m2K'])
    q_boundary = hbot*(Ta-T[0,:]) + htop*(Ta-T[-1,:])
    Q_boundary = np.concatenate([[0.0], cumulative_trapezoid(q_boundary,t)])

    bK=float(result.physical_b_K)
    onesp=np.ones(len(m.part_node_indices))
    reaction = np.array([bK*(onesp @ mats.M_part_local @ (alpha[:,j]-alpha[:,0])) for j in range(alpha.shape[1])])
    residual = sensible_change - Q_boundary - reaction
    scale=max(np.max(np.abs(sensible_change)),np.max(np.abs(Q_boundary))+np.max(np.abs(reaction)),1.0)
    return pd.DataFrame({
        'case_id':result.case['case_id'],'t_s':t,'t_min':t/60.0,
        'sensible_energy_change_J_m2':sensible_change,
        'boundary_energy_J_m2':Q_boundary,
        'reaction_energy_J_m2':reaction,
        'energy_residual_J_m2':residual,
        'normalized_energy_residual':residual/scale,
    })


def monotonic_alpha_check(result, tol=1e-8):
    d=np.diff(result.alpha,axis=1)
    return bool(np.nanmin(d)>=-tol), float(np.nanmin(d)), float(np.nanmin(result.alpha)), float(np.nanmax(result.alpha))
