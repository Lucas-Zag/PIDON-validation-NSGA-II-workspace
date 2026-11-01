from dataclasses import dataclass
import numpy as np
from .mesh import Mesh1D


@dataclass
class FEMatrices:
    M: np.ndarray
    K_cond: np.ndarray
    K_total: np.ndarray
    M_part_local: np.ndarray
    capacity_tool_J_m3K: float
    capacity_part_J_m3K: float


def material_values(config: dict):
    tool = config['materials']['tool_invar']
    part = config['materials']['composite_as4_8552']
    k_t = float(tool['conductivity_W_mK'])
    a_t = float(tool['diffusivity_m2_s'])
    k_c = float(part['conductivity_W_mK'])
    a_c = float(part['diffusivity_m2_s'])
    Cv_t = k_t / a_t
    Cv_c = k_c / a_c
    return k_t, Cv_t, k_c, Cv_c


def assemble(mesh: Mesh1D, config: dict, hbot: float, htop: float) -> FEMatrices:
    n = len(mesh.z)
    M = np.zeros((n, n), dtype=float)
    K = np.zeros((n, n), dtype=float)
    k_t, Cv_t, k_c, Cv_c = material_values(config)

    for e, (i, j) in enumerate(mesh.elements):
        L = mesh.z[j] - mesh.z[i]
        if mesh.element_regions[e] == 'tool':
            k, Cv = k_t, Cv_t
        else:
            k, Cv = k_c, Cv_c
        Me = Cv * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        Ke = k / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
        idx = np.ix_([i, j], [i, j])
        M[idx] += Me
        K[idx] += Ke

    K_total = K.copy()
    K_total[0, 0] += hbot
    K_total[-1, -1] += htop

    # Composite capacity matrix on composite-local nodes, used for consistent cure heat loading.
    npart = len(mesh.part_node_indices)
    Mpart = np.zeros((npart, npart), dtype=float)
    for e in range(len(mesh.element_regions)):
        if mesh.element_regions[e] != 'part':
            continue
        gi, gj = mesh.elements[e]
        li = gi - mesh.interface_node
        lj = gj - mesh.interface_node
        L = mesh.z[gj] - mesh.z[gi]
        Me = Cv_c * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        Mpart[np.ix_([li, lj], [li, lj])] += Me

    return FEMatrices(M, K, K_total, Mpart, Cv_t, Cv_c)
