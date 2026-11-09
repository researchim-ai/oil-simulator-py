"""Shared phase faces exchange equal and opposite finite equipment stocks."""

import json

import pytest
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.pr_equilibrium_euler_plant import PREquilibriumEulerPlant, PREquilibriumEulerPlantState
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork


def plant(pressure=1e6):
    pipe, v = setup(count=2)
    graph = PREquilibriumTransportNetwork(
        pipe.fluid,
        [PREquilibriumNode("feed", 0.001), PREquilibriumNode("sink", 0.001)],
        wave_speed_bound_m_s=4000.0,
    )
    gs = graph.initialize(
        {
            n: {"pressure_pa": p, "temperature_k": 320.0, "mole_fractions": [0.5, 0.5]}
            for n, p in [("feed", pressure), ("sink", 1e6)]
        }
    )
    cell = {"gas": phase(pipe.fluid, 1e6, 320.0, "vapor", v, x=(0.5, 0.5))}
    ps = pipe.initialize([cell, cell])
    model = PREquilibriumEulerPlant(
        graph,
        {"line": {"pipe": pipe, "boundary_nodes": ("feed", "sink"), "direction": (1.0, 0.0, 0.0)}},
    )
    return model, model.initialize(gs, {"line": ps})


@pytest.mark.parametrize("pressure", [1e6, 1.001e6, 0.999e6])
def test_finite_nodes_shared_faces_and_restart(pressure):
    model, old = plant(pressure)
    new, r = model.step(old, 0.01 * model.maximum_step_s)
    assert len(r["interfaces"]) == 2
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert r["entropy_production_j_k"] >= -1e-6
    nc = len(model.fluid.names)
    for j in range(nc):
        pipe_delta = sum(
            b[j] - a[j]
            for a, b in zip(
                dict(old.pipes)["line"].bulk_stocks, dict(new.pipes)["line"].bulk_stocks
            )
        )
        graph_delta = sum(
            b.component_moles[j] - a.component_moles[j]
            for (_, a), (_, b) in zip(old.transport.nodes, new.transport.nodes)
        )
        assert pipe_delta + graph_delta == pytest.approx(0.0, abs=1e-8)
    restored = PREquilibriumEulerPlant.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PREquilibriumEulerPlantState.from_dict(json.loads(json.dumps(new.to_dict())))
    restored.inspect(checkpoint)
    again, rr = restored.step(checkpoint, 0.01 * model.maximum_step_s)
    assert again.time_s > new.time_s
    assert max(map(abs, rr["conservation_residuals"])) < 1e-5


def test_active_pipe_couples_existing_pump_and_finite_rotor():
    import math

    from test_surface_pr_pump_connection import setup as equipment

    from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe

    graph, gs, _ = equipment(driven=True, dense_limit=1)
    diameter = 0.05
    volume = 0.0001
    pipe = PRActiveMultifluidPipe(
        graph.fluid,
        [{"name": "liquid", "root": "liquid"}, {"name": "gas", "root": "vapor"}],
        [volume / (math.pi * diameter**2 / 4)] * 2,
        diameter,
    )
    model = PREquilibriumEulerPlant(
        graph,
        {
            "line": {
                "pipe": pipe,
                "boundary_nodes": ("feed", "return"),
                "direction": (1.0, 0.0, 0.0),
            }
        },
    )
    node = dict(gs.nodes)["feed"]
    cell = {
        "liquid": phase(
            graph.fluid,
            node.pressure_pa,
            node.temperature_k,
            "liquid",
            volume,
            x=tuple(v / sum(node.component_moles) for v in node.component_moles),
        )
    }
    ps = pipe.initialize([cell, cell])
    old = model.initialize(gs, {"line": ps})
    new, r = model.step(old, 0.01 * model.maximum_step_s)
    assert r["transport"]["pump_connections"]["booster"]["shaft_power_w"] > 0
    assert dict(new.transport.rotors)["booster"] != dict(old.transport.rotors)["booster"]
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_active_plant_full_interval_and_atomic_rejection():
    model, old = plant(1.001e6)
    duration = 2.25 * model.maximum_step_s
    result = model.advance(old, duration)
    assert result["state"]["transport"]["time_s"] == duration
    assert len(result["steps"]) >= 3
    assert max(map(abs, result["conservation_residuals"])) < 1e-5
    snapshot = old.to_dict()
    with pytest.raises(ValueError, match="reservoir temperature"):
        model.step(old, 0.1 * model.maximum_step_s, pipe_heat_w={"line": [1.0, 1.0]})
    assert old.to_dict() == snapshot
