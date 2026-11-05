"""Lateral finite-volume taps conserve energy and advect axial momentum."""

import json
import math
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)


def setup(reverse=False):
    diameter = 0.01
    nodes = [
        PREquilibriumNode(
            "body", math.pi * diameter**2 / 4 * 0.1, "pipe_cell", length_m=0.1, diameter_m=diameter
        ),
        PREquilibriumNode("receiver", 0.001),
    ]
    link = PREquilibriumLink("tap", "body", "receiver", 0.001, source_port="side:draw")
    model = PREquilibriumTransportNetwork(
        binary(), nodes, [link], dense_limit=1, initial_step_s=1e-7, maximum_step_s=1e-7
    )
    initial = {
        "body": {
            "pressure_pa": 1e6,
            "temperature_k": 350.0,
            "mole_fractions": [1.0, 0.0],
            "velocity_m_s": 2.0,
        },
        "receiver": {
            "pressure_pa": 1.01e6 if reverse else 0.99e6,
            "temperature_k": 350.0,
            "mole_fractions": [1.0, 0.0],
        },
    }
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("reverse", [False, True])
def test_lateral_flux_energy_keeps_tangential_kinetic_and_outflow_momentum(reverse):
    model, state, _ = setup(reverse)
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    endpoint, _, normal, _ = model._endpoint(0, data[0], "source", model.links[0])
    assert normal == 0 and float(endpoint[3]) == 0
    assert float(endpoint[0][-1] * model.nodes[0].volume_m3) == pytest.approx(
        state.nodes[0][1].total_energy_j, rel=1e-12
    )
    off, _, _ = model._fluxes(data, {"tap": False})
    net, records, _ = model._fluxes(data, {"tap": True})
    mass_rate = (records[0]["flux"][:-2] * model.fluid.molar_masses).sum()
    assert bool(mass_rate < 0) is reverse
    expected = -mass_rate.clamp_min(0) * data[0]["u"]
    torch.testing.assert_close(net[0][-2] - off[0][-2], expected)
    assert float(net[0][-1] + net[1][-1]) == pytest.approx(0.0, abs=1e-10)
    new, report = model._step(state, 1e-7, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    assert new.time_s == 1e-7


def test_lateral_joint_jacobian_and_independent_audit(monkeypatch):
    model, state, initial = setup()
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-7, model.default_openings, [0.0, 0.0])
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    finite = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, finite, rtol=1e-5, atol=1e-8)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]


def test_lateral_names_and_geometry_are_not_axial_pipe_ends():
    model, _, _ = setup()
    with pytest.raises(ValueError, match="explicit junction"):
        PREquilibriumLink("bad", "a", "b", 0.001, "side:a", "side:b")
    with pytest.raises(ValueError, match="unique matching-area"):
        PREquilibriumTransportNetwork(
            model.fluid,
            model.nodes,
            [
                model.links[0],
                PREquilibriumLink("duplicate", "body", "receiver", 0.001, source_port="side:draw"),
            ],
        )


def test_phase_selective_lateral_outflow_removes_selected_phase_axial_momentum():
    from surface.pr_resolved_separator import PRLateralNozzle, join_pr_transport_graphs

    fluid = binary()
    phases = sorted(
        fluid.flash_pt(1e6, 196.2, [0.5, 0.5])["phases"],
        key=lambda p: float(p["density_kg_m3"]),
        reverse=True,
    )
    amounts = [0.5 / p["molar_volume_m3_mol"] for p in phases]
    composition = sum(n * p["mole_fractions"] for n, p in zip(amounts, phases)) / sum(amounts)
    cell = PREquilibriumNode(
        "cell",
        math.pi * 0.01**2 / 4 * 0.01,
        "pipe_cell",
        length_m=0.01,
        diameter_m=0.01,
        elevation_change_m=0.01,
        buoyancy_relaxation_s=1e-6,
    )
    sink = PREquilibriumNode("sink", 1e-5, elevation_m=-0.003)
    base = PREquilibriumTransportNetwork(fluid, [cell, sink], gravity_m_s2=9.80665)
    nozzle = PRLateralNozzle("draw_liquid", "cell", "sink", 0.001, 0.002)
    model = join_pr_transport_graphs([base], nozzles=[nozzle])
    state = model.initialize(
        {
            "cell": {
                "pressure_pa": 1e6,
                "temperature_k": 196.2,
                "mole_fractions": composition.tolist(),
            },
            "sink": {
                "pressure_pa": 999000.0,
                "temperature_k": 196.2,
                "mole_fractions": composition.tolist(),
            },
        }
    )
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    i = model.names.index("cell")
    _endpoint, weights, _, extra = model._endpoint(i, data[i], "source", nozzle.link())
    liquid = int(data[i]["q"]["density_kg_m3"].argmax())
    assert float(weights[liquid]) == pytest.approx(1.0)
    assert float(data[i]["u"]) == 0.0
    assert float(extra["tangential_velocity"]) == pytest.approx(
        float(data[i]["phase_velocity"][liquid])
    )
    assert float(extra["tangential_velocity"]) < 0.0
    off, _, _ = model._fluxes(data, {"draw_liquid": False})
    net, records, _ = model._fluxes(data, {"draw_liquid": True})
    mass_rate = (records[0]["flux"][:-2] * fluid.molar_masses).sum()
    assert float(mass_rate) > 0.0
    expected = -mass_rate * data[i]["phase_velocity"][liquid]
    torch.testing.assert_close(net[i][-2] - off[i][-2], expected)


def test_side_tap_cannot_use_axial_body_loss_as_its_control_valve():
    from surface.actuator import ConnectionActuator
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlledSurface, PRControlValve

    transport, _, _ = setup()
    plant = PRGravityTransportNetwork(transport, {}, [])
    valve = PRControlValve("v", "tap", 1.0, ConnectionActuator(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="HEM resistance_node"):
        PRControlledSurface(plant, [valve])


def test_mixed_lateral_sampling_carries_distributed_liquid_without_a_fixed_entrainment_fraction(
    monkeypatch,
):
    from surface.pr_resolved_separator import PRResolvedSeparator, join_pr_transport_graphs
    from surface.pr_transport_builder import initialize_grouped

    spec = PRResolvedSeparator("separator", 0.01, 0.02, 2, 1e-6)
    body = spec.build(binary(), initial_step_s=1e-7, maximum_step_s=1e-7)
    sink = PREquilibriumNode("export", 1e-5, elevation_m=spec.cell_elevation_m(1))
    external = PREquilibriumTransportNetwork(
        body.fluid, [sink], gravity_m_s2=body.gravity_m_s2, initial_step_s=1e-7, maximum_step_s=1e-7
    )
    # No subcell layer assumption: a mixed outlet samples local distributed
    # phase volumes. Settling upstream remains finite, not an imposed efficiency.
    link = PREquilibriumLink(
        "mixed_draw", spec.cell_name(1), "export", 0.001, source_port="side:mixed_draw"
    )
    model = join_pr_transport_graphs([body, external], links=[link])
    initial = {
        "separator": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
        "export": {"pressure_pa": 999000.0, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
    }
    state = initialize_grouped(model, initial)
    i = model.names.index(spec.cell_name(1))
    d = model._state_data(i, dict(state.nodes)[spec.cell_name(1)])
    _endpoint, weights, _, _extra = model._endpoint(i, d, "source", link)
    torch.testing.assert_close(weights, d["alpha"])
    assert len(weights) == 2 and bool((weights > 0).all())
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"groups": initial},
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    for step in report["intervals"][0]["steps"]:
        pickup = step["links"]["mixed_draw"]
        assert all(v > 0 for v in pickup["source_area_fractions"])
        assert pickup["flux"][0] > 0
