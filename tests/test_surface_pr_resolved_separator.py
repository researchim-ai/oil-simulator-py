"""Finite-rate separation, retained layers and graph composition."""

import json
import math
from pathlib import Path

import pytest
import torch
from test_surface_compositional_separator import binary

from surface.liquid import PropertyModelError
from surface.pr_equilibrium_transport import (
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.pr_resolved_separator import (
    PRLateralNozzle,
    PRResolvedSeparator,
    join_pr_transport_graphs,
)
from surface.pr_transport_builder import initialize_grouped


def separator(response=1e-6, cells=4):
    fluid = binary()
    spec = PRResolvedSeparator("separator", 0.01, 0.04, cells, response)
    model = spec.build(
        fluid, initial_step_s=1e-6, maximum_step_s=1e-4, minimum_step_s=1e-10, dense_limit=1
    )
    initial = {
        spec.name: {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}
    }
    return spec, model, initialize_grouped(model, initial)


def test_separation_has_finite_conservative_countercurrent_component_flux():
    spec, model, state = separator()
    before = state.to_dict()
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    _net, records, _ = model._fluxes(data, model.default_openings)
    masses = model.fluid.molar_masses
    for record in records:
        flux = record["flux"][:-2]
        assert float(flux[0]) > 0 and float(flux[1]) < 0
        assert float((flux * masses).sum()) == pytest.approx(0.0, abs=1e-15)
    end, report = model._step(
        state, 1e-6, model.default_openings, [0.0] * spec.cells, [196.2] * spec.cells
    )
    assert state.to_dict() == before
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    top = dict(end.nodes)[spec.cell_name(spec.cells - 1)]
    assert top.component_moles[0] / sum(top.component_moles) > 0.5
    # Local equilibrium does not instantly transfer the column inventory
    # to pure top/bottom layers in a single time step.
    assert all(len(s.phase_roots) == 2 for _, s in end.nodes)


def test_terminal_response_controls_rate_not_a_postprocessed_efficiency():
    _, slow, a = separator(response=1e-6)
    _, fast, b = separator(response=2e-6)

    def rate(model, state):
        data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
        return model._fluxes(data, model.default_openings)[1][0]["flux"][0]

    assert float(rate(fast, b)) == pytest.approx(2 * float(rate(slow, a)), rel=1e-10)


def test_resolved_separator_lateral_nozzle_and_external_graph_share_restart():
    spec, body, _ = separator()
    height = spec.cell_height_m / 2
    plenum = PREquilibriumNode("product", 1e-5, elevation_m=spec.cell_elevation_m(1))
    external = PREquilibriumTransportNetwork(
        body.fluid,
        [plenum],
        gravity_m_s2=body.gravity_m_s2,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
        minimum_step_s=1e-10,
        dense_limit=1,
    )
    nozzle = PRLateralNozzle("draw", spec.cell_name(1), "product", 0.001, height)
    model = join_pr_transport_graphs([body, external], nozzles=[nozzle])
    state = initialize_grouped(
        model,
        {
            "separator": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
            "product": {
                "pressure_pa": 999000.0,
                "temperature_k": 196.2,
                "mole_fractions": [0.5, 0.5],
            },
        },
    )
    n = len(model.nodes)
    end, report = model._step(state, 1e-7, model.default_openings, [0.0] * n, [196.2] * n)
    assert report["links"]["draw"]["flux"][0] > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    restored = type(model).from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PREquilibriumTransportState.from_dict(json.loads(json.dumps(end.to_dict())))
    expected, _ = model._step(end, 1e-7, model.default_openings, [0.0] * n, [196.2] * n)
    actual, _ = restored._step(checkpoint, 1e-7, restored.default_openings, [0.0] * n, [196.2] * n)
    assert actual.to_dict() == expected.to_dict()


def test_horizontal_stopped_cell_retains_liquid_below_a_finite_gas_pickup():
    fluid = binary()
    diameter, length = 0.02, 0.1
    body = PREquilibriumNode(
        "pocket",
        math.pi * diameter**2 / 4 * length,
        "pipe_cell",
        length_m=length,
        diameter_m=diameter,
    )
    receiver = PREquilibriumNode("gas", 1e-5, elevation_m=0.007)
    empty = PREquilibriumTransportNetwork(fluid, [body, receiver], gravity_m_s2=9.80665)
    nozzle = PRLateralNozzle("gas_draw", "pocket", "gas", 0.001, 0.017)
    model = join_pr_transport_graphs([empty], nozzles=[nozzle])
    state = model.initialize(
        {
            "pocket": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
            "gas": {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]},
        }
    )
    index = model.names.index("pocket")
    data = model._state_data(index, dict(state.nodes)["pocket"])
    _end, weights, _normal, _extra = model._endpoint(index, data, "source", nozzle.link())
    gas = int(data["q"]["density_kg_m3"].argmin())
    assert float(weights[gas]) == pytest.approx(1.0)
    assert float(data["kinetic"]) == 0
    assert float(weights.sum() - weights[gas]) == 0
    # All apertures isolated: no component leaves the retained pocket.
    end, _report = model._step(state, 1e-6, {"gas_draw": False}, [0.0, 0.0], [196.2, 196.2])
    assert dict(end.nodes)["pocket"].component_moles == pytest.approx(
        dict(state.nodes)["pocket"].component_moles, rel=0.0, abs=1e-14
    )


def test_graph_join_rejects_incompatible_physical_policy():
    _, a, _ = separator()
    b = PREquilibriumTransportNetwork(a.fluid, [PREquilibriumNode("other", 0.01)])
    with pytest.raises(ValueError, match="identical"):
        join_pr_transport_graphs([a, b])
    with pytest.raises(ValueError, match="Duplicate"):
        join_pr_transport_graphs([a, a])


def test_low_head_guard_prevents_unqualified_stratified_pickup():
    fluid = binary()
    node = PREquilibriumNode("large", math.pi / 4, "pipe_cell", length_m=1.0, diameter_m=1.0)
    sink = PREquilibriumNode("sink", 0.01)
    base = PREquilibriumTransportNetwork(fluid, [node, sink], gravity_m_s2=9.80665)
    model = join_pr_transport_graphs(
        [base], nozzles=[PRLateralNozzle("tap", "large", "sink", 0.01, 0.5)]
    )
    with pytest.raises(PropertyModelError, match="rho\\*g\\*height"):
        model.initialize(
            {
                n: {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": [0.5, 0.5]}
                for n in model.names
            }
        )


@pytest.mark.parametrize("liquid_fraction", [0.35, 0.45, 0.5, 0.55, 0.65])
def test_layer_interface_crosses_finite_lateral_aperture_by_exact_disk_area(liquid_fraction):
    fluid = binary()
    tie = fluid.flash_pt(1e6, 196.2, [0.5, 0.5])["phases"]
    phases = sorted(tie, key=lambda p: float(p["density_kg_m3"]), reverse=True)
    amounts = [
        liquid_fraction / phases[0]["molar_volume_m3_mol"],
        (1 - liquid_fraction) / phases[1]["molar_volume_m3_mol"],
    ]
    composition = sum(n * p["mole_fractions"] for n, p in zip(amounts, phases)) / sum(amounts)
    body = PREquilibriumNode(
        "cell",
        math.pi * 0.01**2 / 4 * 0.01,
        "pipe_cell",
        length_m=0.01,
        diameter_m=0.01,
        elevation_change_m=0.01,
    )
    sink = PREquilibriumNode("sink", 1e-5)
    base = PREquilibriumTransportNetwork(fluid, [body, sink], gravity_m_s2=9.80665)
    nozzle = PRLateralNozzle("draw", "cell", "sink", 0.002, 0.005)
    model = join_pr_transport_graphs([base], nozzles=[nozzle])
    state = model.initialize(
        {
            n: {"pressure_pa": 1e6, "temperature_k": 196.2, "mole_fractions": composition.tolist()}
            for n in model.names
        }
    )
    index = model.names.index("cell")
    data = model._state_data(index, dict(state.nodes)["cell"])
    _endpoint, weights, _, _ = model._endpoint(index, data, "source", nozzle.link())
    liquid = int(data["q"]["density_kg_m3"].argmax())
    y = max(-1.0, min(1.0, (liquid_fraction * 0.01 - 0.005) / 0.001))
    expected = (math.acos(-y) + y * math.sqrt(max(0.0, 1 - y * y))) / math.pi
    assert float(weights[liquid]) == pytest.approx(expected, abs=1e-8)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    if liquid_fraction == 0.5:
        seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
        y, _, _, evaluate, *_ = model._problem(
            state, seeds, 1e-7, model.default_openings, [0.0] * len(model.nodes)
        )
        direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
        jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
        eps = 1e-4
        finite = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
        torch.testing.assert_close(jac @ direction, finite, rtol=1e-5, atol=1e-7)


def test_resolved_separator_independent_local_balance_audit_and_damage_detection(monkeypatch):
    import copy

    spec, model, _ = separator()
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {
                    "groups": {
                        spec.name: {
                            "pressure_pa": 1e6,
                            "temperature_k": 196.2,
                            "mole_fractions": [0.5, 0.5],
                        }
                    }
                },
                "schedule": [{"duration_s": 1e-6}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    damaged = copy.deepcopy(report)
    damaged["intervals"][0]["steps"][0]["state"]["nodes"][spec.cell_name(0)]["component_moles"][
        0
    ] *= 1.001
    with pytest.raises(ValueError):
        audit(config, damaged)


def test_three_phase_resolved_separation_and_joint_derivatives():
    from test_surface_oil_water_drift import three_phase_pipe

    original, _, initial = three_phase_pipe()
    spec = PRResolvedSeparator("three_phase", 0.01, 0.02, 2, 1e-6)
    model = spec.build(original.fluid, dense_limit=1)
    state = initialize_grouped(model, {spec.name: initial["0"]})
    assert all(len(s.phase_roots) == 3 for _, s in state.nodes)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-7, model.default_openings, [0.0, 0.0])
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    finite = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, finite, rtol=1e-5, atol=1e-7)
    _, report = model._step(state, 1e-7, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert max(map(abs, report["conservation_residuals"])) < 1e-6


@pytest.mark.parametrize("measurement", ["pressure", "inventory_level"])
def test_controlled_resolved_separator_needs_no_dummy_equilibrium_column(measurement, monkeypatch):
    from surface.actuator import ConnectionActuator
    from surface.pr_equilibrium_transport import PREquilibriumLink
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_surface_control import PRControlledSurface, PRControlValve, PRSurfacePI

    spec, body, _ = separator(cells=2)
    height = spec.cell_elevation_m(1)
    diameter = 0.001
    valve_node = PREquilibriumNode(
        "valve_body",
        math.pi * diameter**2 / 4 * 0.01,
        "pipe_cell",
        length_m=0.01,
        diameter_m=diameter,
        elevation_m=height,
    )
    product = PREquilibriumNode("product", 1e-5, elevation_m=height)
    external = PREquilibriumTransportNetwork(
        body.fluid,
        [valve_node, product],
        [PREquilibriumLink("outlet", "valve_body", "product", diameter)],
        gravity_m_s2=body.gravity_m_s2,
        initial_step_s=1e-6,
        maximum_step_s=1e-4,
        minimum_step_s=1e-10,
        dense_limit=1,
    )
    nozzle = PRLateralNozzle(
        "draw", spec.cell_name(1), "valve_body", diameter, spec.cell_height_m / 2
    )
    transport = join_pr_transport_graphs([body, external], nozzles=[nozzle])
    plant = PRGravityTransportNetwork(transport, {}, [])
    valve = PRControlValve("v", "draw", 1.0, ConnectionActuator(1.0, 1.0, 0.0, time_constant_s=0.1))
    controller = PRSurfacePI(
        "v",
        1e6 if measurement == "pressure" else 0.001,
        1e-6 if measurement == "pressure" else 1.0,
        measurement=measurement,
        component=None if measurement == "pressure" else body.fluid.names[1],
        sensor=None if measurement == "pressure" else spec.name,
    )
    model = PRControlledSurface(
        plant, [valve], [controller], initial_step_s=1e-7, maximum_step_s=1e-7, minimum_step_s=1e-10
    )
    initial = {
        name: {
            "pressure_pa": 1e6 if name.startswith("separator") else 999000.0,
            "temperature_k": 196.2,
            "mole_fractions": [0.5, 0.5],
        }
        for name in transport.names
    }
    config = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {
                    "plant": {"transport": initial, "columns": {}, "wall_temperatures": {}},
                    "positions": {"v": 0.5},
                    "integrals": {},
                },
                "schedule": [{"duration_s": 1e-7}],
            }
        )
    )
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_controlled_surface import audit
    from run_pr_controlled_surface import run

    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    assert report["initial"]["plant"]["columns"] == {}
    assert report["intervals"][0]["steps"][0]["measurements"]["v"] > 0


def test_column_free_wrapper_keeps_finite_thermal_links_without_phantom_stocks():
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_heat_link import PRHeatLink

    transport = PREquilibriumTransportNetwork(
        binary(), [PREquilibriumNode("hot", 0.001), PREquilibriumNode("cold", 0.001)]
    )
    initial = transport.initialize(
        {
            "hot": {"pressure_pa": 1e6, "temperature_k": 360.0, "mole_fractions": [1.0, 0.0]},
            "cold": {"pressure_pa": 1e6, "temperature_k": 350.0, "mole_fractions": [1.0, 0.0]},
        }
    )
    model = PRGravityTransportNetwork(
        transport, {}, [], heat_links=[PRHeatLink("contact", "hot", "cold", 1.0)]
    )
    state = model.initialize(initial, {})
    end, report = model._step(state, 1e-4, {"contact": True}, {}, {}, {})
    assert end.columns == () and end.walls == ()
    assert report["columns"]["heat_links"]["contact"]["heat_rate_w"] > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-8
    assert dict(end.transport.nodes)["hot"].temperature_k < 360.0
    assert dict(end.transport.nodes)["cold"].temperature_k > 350.0
