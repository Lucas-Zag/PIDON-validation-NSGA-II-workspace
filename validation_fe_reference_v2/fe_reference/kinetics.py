import numpy as np


def cure_rate(T_C, alpha, p: dict):
    T_K = np.asarray(T_C, dtype=float) + 273.15
    a = np.asarray(alpha, dtype=float)
    a_safe = np.clip(a, 1.0e-12, 1.0 - 1.0e-10)

    A = float(p['A_1_s'])
    E = float(p['activation_energy_J_mol'])
    R = float(p['gas_constant_J_molK'])
    m = float(p['m'])
    n = float(p['n'])
    C = float(p['C'])
    C0 = float(p['C0'])
    CT = float(p['CT_1_K'])

    k = A * np.exp(-E / (R * T_K))
    diffusion = 1.0 + np.exp(C * (a_safe - (C0 + CT * T_K)))
    rate = (k / diffusion) * a_safe**m * (1.0 - a_safe)**n
    return np.where(a >= 1.0 - 1e-10, 0.0, np.maximum(rate, 0.0))
