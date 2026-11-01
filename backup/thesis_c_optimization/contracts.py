"""Physical-unit contract shared by baseline clients, with no model imports."""
import csv
import math
from pathlib import Path

from cycle import cycle_timing

RANGE_MAP = {
    'ramp1_C_min': ('r1_min','r1_max'), 'hold1_T_C': ('ht1_min','ht1_max'),
    'hold1_d_min': ('hd1_min','hd1_max'), 'ramp2_C_min': ('r2_min','r2_max'),
    'hold2_T_C': ('ht2_min','ht2_max'), 'hold2_d_min': ('hd2_min','hd2_max'),
    'hbot_W_m2K': ('hcb_min','hcb_max'), 'htop_W_m2K': ('hct_min','hct_max'),
    'Lt_m': ('lt_min','lt_max'),
}
CYCLE_COLUMNS = list(RANGE_MAP)[:6]


def read_case(path, case_id):
    with Path(path).open(newline='', encoding='utf-8') as f:
        matches = [r for r in csv.DictReader(f) if r['case_id'] == case_id]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one case {case_id!r}, found {len(matches)}')
    case = matches[0]
    for k in list(RANGE_MAP) + ['Lc_m','total_process_time_s']:
        case[k] = float(case[k])
    return case


def validate_case(case, exp, phy):
    for k,(lo,hi) in RANGE_MAP.items():
        v = float(case[k])
        if not math.isfinite(v) or not float(exp[lo]) <= v <= float(exp[hi]):
            raise ValueError(f'{k}={v} is outside [{exp[lo]}, {exp[hi]}]')
    for k,expected in [('Lc_m',phy['part_len']),('total_process_time_s',exp['t_max'])]:
        v = float(case[k])
        if not math.isfinite(v) or not math.isclose(v,float(expected),rel_tol=0,abs_tol=1e-9):
            raise ValueError(f'{k} must match trained value {expected}, received {v}')
    timing = cycle_timing(dict(zip(('r1','ht1','hd1','r2','ht2','hd2'),
                                  [float(case[k]) for k in CYCLE_COLUMNS])),
                          float(exp['T_ini']),float(exp['t_max']))
    if not timing['positive_cooling_interval']:
        raise ValueError('Cycle leaves no positive cooling interval')
    return timing


def scaled_input(case, exp):
    """Same nine-component input mapping as comparison/run_xpidon_cases.py."""
    ts, T, L, h = float(exp['t_max']),float(exp['T_max']),float(exp['l_max']),float(exp['h_max'])
    return [case['ramp1_C_min']/60*ts/T, case['hold1_T_C']/T,
            case['hold1_d_min']*60/ts, case['ramp2_C_min']/60*ts/T,
            case['hold2_T_C']/T, case['hold2_d_min']*60/ts,
            case['hbot_W_m2K']/h, case['htop_W_m2K']/h, case['Lt_m']/L]


def select_tags(model_dir, tags):
    """An explicit run selection avoids silently mixing checkpoints."""
    if not tags or len(set(tags)) != len(tags):
        raise ValueError('Select distinct checkpoint tags explicitly with --tags')
    if any(not t.isdigit() or len(t) != 5 for t in tags):
        raise ValueError('Checkpoint tags must be five digits, e.g. 02000')
    tags = sorted(tags, key=int)
    if int(tags[0]) <= 0 or int(tags[-1]) != 10000:
        raise ValueError('Selected tags must be positive and end at 10000')
    for tag in tags:
        for variable in ('T','a'):
            p = Path(model_dir)/f'xpidon_class_{tag}_{variable}.pkl'
            if not p.is_file():
                raise FileNotFoundError(p)
    return tags


def sampled_metrics(times, positions, temperature, alpha):
    """Report sampled extrema; never label them continuum safety guarantees."""
    import numpy as np
    times, positions = np.asarray(times), np.asarray(positions)
    temperature, alpha = np.asarray(temperature), np.asarray(alpha)
    shape = (len(times),len(positions))
    if temperature.shape != shape or alpha.shape != shape:
        raise ValueError('Expected fields of shape (time, position)')
    if not np.isfinite(temperature).all() or not np.isfinite(alpha).all():
        raise ValueError('Nonfinite predicted fields')
    peak = np.unravel_index(np.argmax(temperature),shape)
    return {'sampled_peak_part_temperature_C':float(temperature[peak]),
            'peak_time_s':float(times[peak[0]]), 'peak_x_norm':float(positions[peak[1]]),
            'sampled_final_alpha_min':float(alpha[-1].min()),
            'sampled_final_alpha_max':float(alpha[-1].max()),
            'sampled_final_alpha_spread':float(np.ptp(alpha[-1])),
            'sampled_alpha_min_all_times':float(alpha.min()),
            'sampled_alpha_max_all_times':float(alpha.max()),
            'largest_sampled_alpha_drop':float(min(0,np.diff(alpha,axis=0).min()))}
