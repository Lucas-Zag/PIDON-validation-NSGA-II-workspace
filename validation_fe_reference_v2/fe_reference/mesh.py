from dataclasses import dataclass
import numpy as np


@dataclass
class Mesh1D:
    z: np.ndarray
    elements: np.ndarray
    element_regions: np.ndarray
    interface_node: int
    part_node_indices: np.ndarray
    Lt_m: float
    Lc_m: float


def build_layered_mesh(Lt_m: float, Lc_m: float, n_tool_elements: int, n_part_elements: int) -> Mesh1D:
    if n_tool_elements < 1 or n_part_elements < 1:
        raise ValueError('Each layer needs at least one element.')
    z_tool = np.linspace(0.0, Lt_m, n_tool_elements + 1)
    z_part = np.linspace(Lt_m, Lt_m + Lc_m, n_part_elements + 1)
    z = np.concatenate([z_tool, z_part[1:]])
    elements = np.column_stack([np.arange(len(z)-1), np.arange(1, len(z))])
    regions = np.array(['tool'] * n_tool_elements + ['part'] * n_part_elements, dtype=object)
    interface_node = n_tool_elements
    part_nodes = np.arange(interface_node, len(z))
    return Mesh1D(z, elements, regions, interface_node, part_nodes, Lt_m, Lc_m)
