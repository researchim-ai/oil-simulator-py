"""Oil-free gas/water cells must satisfy both capillary pressure relations."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.eclipse_deck import EclipseCase
from simulator.ecl_io import EclSummary


class LinearPc:
    tables = [(None, object())]

    def endpoint(self, key, idx):
        return np.full(len(idx), {"SWL": 0.1, "SWU": 1.0, "SGU": 0.9}[key])

    def pc_value(self, idx, phase, s):
        return 100.0 * (1.0 - s) if phase == "w" else 50.0 * s

    def invert_pc(self, idx, phase, p):
        return 1.0 - p / 100.0 if phase == "w" else p / 50.0


def test_gas_water_equilibrium_uses_sum_of_capillary_pressures():
    ep = LinearPc()
    p, sw, sg = EclipseCase._equil_saturations(
        dict(woc=10.0, goc=1.0), ep, np.arange(2), np.zeros(2),
        np.array([60.0, 90.0]), np.zeros(2), np.array([120.0, 95.0]))
    np.testing.assert_allclose(sw, [0.2, 0.1], atol=1e-11)
    np.testing.assert_allclose(sg, [0.8, 0.1], atol=1e-11)
    np.testing.assert_allclose(p, [80.0, 90.0], atol=1e-9)
    # Oil pressure, reconstructed from gas, must also reproduce water pressure.
    assert abs(p[0] - ep.pc_value(np.array([0]), "w", sw[:1])[0]) < 1e-9


def test_network_initial_gas_zone_water_matches_opm():
    import torch

    data = Path(__file__).parent / "data/network"
    case = EclipseCase(str(data / "NETWORK-01.DATA"))
    reservoir, fluid, _, _ = case.build(device=torch.device("cpu"))
    i, j, k = 1, 10, 1
    nx, ny, nz = case.dims
    cell = int(reservoir.cell_map[i * ny * nz + j * nz + k])
    ref = EclSummary(data / "opm/NETWORK-01")
    sw_ref = ref.get("BWSAT", num=ref.cell_num(i + 1, j + 1, k + 1))[0]
    assert abs(float(fluid.s_w.reshape(-1)[cell]) - sw_ref) < 1e-6
