"""Absolute balance floors plus a floating-point forward-error allowance.

Pressure-volume work is included in the energy scale. The allowance remains
O(machine epsilon), not an adjustable relative convergence tolerance.
"""

import math
import sys


def balance_limits(before, after, nc, *, include_volume):
    phases = tuple(before) + tuple(after)
    scales = [math.fsum(abs(p.component_moles[j]) for p in phases) for j in range(nc)]
    floors = [1e-8] * nc
    if include_volume:
        scales.append(math.fsum(p.volume_m3 for p in phases))
        floors.append(1e-12)
    scales.extend(
        (
            math.fsum(abs(p.total_energy_j) + p.pressure_pa * p.volume_m3 for p in phases),
            math.fsum(abs(p.momentum_kg_m_s) for p in phases),
        )
    )
    floors.extend((1e-6, 1e-9))
    return [max(floor, 64 * sys.float_info.epsilon * scale) for floor, scale in zip(floors, scales)]


def entropy_allowance(*values):
    return max(1e-7, 64 * sys.float_info.epsilon * math.fsum(abs(v) for v in values))
