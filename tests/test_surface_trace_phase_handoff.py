"""A separator's saturated takeoff must survive a pressure change without losing stock."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from surface.equilibrium_properties import EquilibriumFluid
from surface.peng_robinson_mixture import PengRobinsonMixture


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_water_takeoff_retains_positive_incipient_hydrocarbon_phase(device):
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_pr_pump_unit.json").read_text()
    )
    fluid = EquilibriumFluid.from_dict(config["model"]["fluid"], device=device)
    # Actual gravitational water takeoff, transported into a slightly higher
    # pressure receiver. The emerging oil is far below a bulk phase cutoff.
    z = np.array([6.393208760264833e-7, 4.960950244084791e-17, 0.999999360679124])
    pressure, temperature = 1002844.9981835745, 350.0
    result = fluid.flash_pt(pressure, temperature, z)
    assert result["phase_count"] == 2
    phases = result["phases"]
    amounts = np.array([float(q["molar_fraction"].cpu()) for q in phases])
    compositions = np.array([q["mole_fractions"].cpu().numpy() for q in phases])
    assert 0 < amounts.min() < 1e-12
    # Relative checks protect trace components too: no clipping of the hexane.
    np.testing.assert_allclose(amounts @ compositions, z, rtol=5e-13, atol=0)
    oracle = PengRobinsonMixture(fluid.components, fluid.binary_interactions)
    chemical = np.array(
        [
            np.log(x) + oracle.state(pressure, temperature, x)["log_fugacity_coefficients"]
            for x in compositions
        ]
    )
    np.testing.assert_allclose(chemical[0], chemical[1], rtol=0, atol=1e-8)
    assert float(result["minimum_tpd"].cpu()) >= -1e-8
    assert float(result["log_fugacity_error"].cpu()) < 1e-8
