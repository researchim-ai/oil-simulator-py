"""Exact Euler spectrum avoids a dense constrained derivative for one phase."""

from dataclasses import replace

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.liquid import PropertyModelError
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "kind,composition,velocity",
    [
        ("vessel", [0.5, 0.5], 0.0),
        ("pipe_cell", [1.0, 0.0], -4.0),
        ("pipe_cell", [0.5, 0.5], 2.0),
        ("pipe_cell", [0.0, 1.0], 0.1),
    ],
)
def test_exact_single_phase_spectrum_matches_constrained_eos_tangent(
    device, kind, composition, velocity
):
    node = PREquilibriumNode(
        "cell",
        0.001,
        kind,
        **(
            {"length_m": 0.5, "diameter_m": (0.004 / (0.5 * torch.pi)) ** 0.5}
            if kind == "pipe_cell"
            else {}
        ),
    )
    model = PREquilibriumTransportNetwork(binary(device), [node])
    old = model.initialize(
        {
            "cell": {
                "pressure_pa": 1e6,
                "temperature_k": 350.0,
                "mole_fractions": composition,
                "velocity_m_s": velocity,
            }
        }
    )
    data = model._state_data(0, old.nodes[0][1])
    active = torch.nonzero(data["n"] > 0).flatten()
    assert len(data["phase_n"]) == 1
    actual = model._characteristics(0, data, active)
    model._analytic_single_phase_characteristics = False
    expected = model._characteristics(0, data, active)
    torch.testing.assert_close(actual.sort().values, expected.sort().values, rtol=1e-10, atol=1e-7)
    model._analytic_single_phase_characteristics = True
    model.wave_speed_bound_m_s = float(actual.abs().max().cpu()) * 0.9
    with pytest.raises(PropertyModelError, match="below"):
        model._characteristics(0, data, active)


def test_multiphase_and_slip_keep_constrained_tangent(monkeypatch):
    from test_surface_pr_equilibrium_transport import pipe

    model, state = pipe(196.2)
    data = model._state_data(0, state.nodes[0][1])
    assert not model._single_phase_acoustics(0, data)
    model.nodes = (replace(model.nodes[0], distribution_strength=0.1), *model.nodes[1:])
    assert not model._single_phase_acoustics(0, {**data, "phase_n": data["phase_n"][:1]})
