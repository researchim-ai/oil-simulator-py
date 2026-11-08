"""Finite equipment nodes receive exactly the opposite Euler interface N/E."""

import json

import pytest
from test_surface_pr_equilibrium_euler_pipe import cooling_energy, liquid_pipe

from surface.pr_equilibrium_euler_plant import PREquilibriumEulerPlant, PREquilibriumEulerPlantState
from surface.pr_equilibrium_transport import PREquilibriumNode, PREquilibriumTransportNetwork


def setup():
    pipe, ps = liquid_pipe(mode="walls")
    graph = PREquilibriumTransportNetwork(
        pipe.fluid, [PREquilibriumNode("plenum", 0.001)], wave_speed_bound_m_s=4000.0
    )
    gs = graph.initialize(
        {"plenum": {"pressure_pa": 2e6, "temperature_k": 250.0, "mole_fractions": [0.0, 1.0]}}
    )
    model = PREquilibriumEulerPlant(
        graph,
        {
            "branch": {
                "pipe": pipe,
                "boundary_nodes": ("plenum", None),
                "direction": (1.0, 0.0, 0.0),
            }
        },
    )
    return model, model.initialize(gs, {"branch": ps})


def test_finite_plenum_exchange_phase_birth_and_coupled_restart():
    model, old = setup()
    pipe = model.pipes["branch"]["pipe"]
    dt = 0.1 * model.maximum_step_s
    heat = cooling_energy(pipe, dict(old.pipes)["branch"])
    new, r = model.step(
        old,
        dt,
        pipe_heat_w={"branch": [heat / dt, 0.0, 0.0]},
        pipe_heat_temperatures_k={"branch": [100.0, None, None]},
    )
    assert len(dict(new.pipes)["branch"].cells[0].phases) == 2
    end, r = model.step(new, dt)
    assert (
        abs(
            dict(end.transport.nodes)["plenum"].total_energy_j
            - dict(old.transport.nodes)["plenum"].total_energy_j
        )
        > 1e-6
    )
    assert max(map(abs, r["conservation_residuals"])) < 1e-5
    assert max(map(abs, model.ledger(old, end, heat, r["external_impulse_kg_m_s"]))) < 1e-5
    recovered = PREquilibriumEulerPlant.from_dict(json.loads(json.dumps(model.to_dict())))
    checkpoint = PREquilibriumEulerPlantState.from_dict(json.loads(json.dumps(end.to_dict())))
    recovered.inspect(checkpoint)


def connected(graph, gs, node, other=None):
    from surface.pr_equilibrium_euler_pipe import PREquilibriumEulerPipe

    volume = 0.0001
    diameter = 0.05
    import math

    pipe = PREquilibriumEulerPipe(
        graph.fluid, [volume / (math.pi * diameter**2 / 4)] * 2, diameter, boundary_mode="walls"
    )
    index = graph.names.index(node)
    d = graph._state_data(index, dict(gs.nodes)[node])
    scale = volume / graph.nodes[index].volume_m3
    specs = [
        {
            "component_moles": [scale * x for x in dict(gs.nodes)[node].component_moles],
            "total_energy_j": scale * float(d["internal"].cpu()),
            "momentum_kg_m_s": 0.0,
            "pressure_guess_pa": float(d["p"].cpu()),
            "temperature_guess_k": float(d["t"].cpu()),
        }
        for _ in range(2)
    ]
    ps = pipe.initialize(specs)
    model = PREquilibriumEulerPlant(
        graph,
        {"branch": {"pipe": pipe, "boundary_nodes": (node, other), "direction": (1.0, 0.0, 0.0)}},
    )
    return model, model.initialize(gs, {"branch": ps})


def test_existing_pump_connection_and_finite_rotor_remain_coupled():
    from test_surface_pr_pump_connection import setup as equipment

    graph, gs, _ = equipment(driven=True, dense_limit=1)
    model, old = connected(graph, gs, "feed", "return")
    end, r = model.step(old, 0.1 * model.maximum_step_s)
    assert "booster" in r["transport"]["pump_connections"]
    assert r["transport"]["pump_connections"]["booster"]["shaft_power_w"] > 0
    assert dict(end.transport.rotors)["booster"] != dict(old.transport.rotors)["booster"]
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_existing_compressor_pr_work_is_in_the_joint_energy_ledger():
    from test_surface_pr_compressor_unit import setup as equipment

    graph, gs, _ = equipment(pressure=1e6, dense_limit=1)
    model, old = connected(graph, gs, "suction")
    _end, r = model.step(old, 0.1 * model.maximum_step_s)
    assert r["transport"]["compressors"]["compressor"]["shaft_power_w"] > 0
    assert r["external_energy_j"] == pytest.approx(r["transport"]["external_work_j"], abs=1e-8)
    assert max(map(abs, r["conservation_residuals"])) < 1e-5


def test_equipment_failure_cannot_publish_only_the_pipe_substep(monkeypatch):
    from surface.liquid import HydraulicConvergenceError

    model, old = setup()
    saved = old.to_dict()

    def rejected(*args, **kwargs):
        raise HydraulicConvergenceError("Equipment graph rejected")

    monkeypatch.setattr(model.transport, "_step", rejected)
    with pytest.raises(HydraulicConvergenceError, match="graph rejected"):
        model.step(old, 0.1 * model.maximum_step_s)
    assert old.to_dict() == saved


def test_independent_interface_audit_rejects_changed_flux_without_runtime_physics(monkeypatch):
    import copy

    from check_pr_equilibrium_euler_plant import audit_pipe_step

    from surface.pr_equilibrium_euler_pipe import PREquilibriumEulerPipe
    from surface.pr_phase_equilibration import PRPhaseEquilibration

    model, old = setup()
    dt = 0.1 * model.maximum_step_s
    _, step = model.step(old, dt)

    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime physics called by independent audit")

    monkeypatch.setattr(PRPhaseEquilibration, "recover", forbidden)
    monkeypatch.setattr(PREquilibriumEulerPipe, "_packets", forbidden)
    monkeypatch.setattr(model.transport, "_state_data", forbidden)
    assert audit_pipe_step(model, old, step, "branch")["accepted"]
    changed = copy.deepcopy(step)
    changed["interfaces"][0]["flux"][-1] += 1.0
    with pytest.raises((ValueError, AssertionError)):
        audit_pipe_step(model, old, changed, "branch")


@pytest.mark.parametrize("equipment_kind", ["plenum", "pump", "compressor"])
def test_independent_equipment_and_spatial_audit(equipment_kind):
    from check_pr_equilibrium_euler_plant import audit_step

    if equipment_kind == "plenum":
        model, old = setup()
    elif equipment_kind == "pump":
        from test_surface_pr_pump_connection import setup as equipment

        graph, gs, _ = equipment(driven=True, dense_limit=1)
        model, old = connected(graph, gs, "feed", "return")
    else:
        from test_surface_pr_compressor_unit import setup as equipment

        graph, gs, _ = equipment(pressure=1e6, dense_limit=1)
        model, old = connected(graph, gs, "suction")
    _, step = model.step(old, 0.1 * model.maximum_step_s)
    assert audit_step(model, old, step)["accepted"]


def test_parallel_pipes_share_one_finite_plenum_without_double_counting():
    from check_pr_equilibrium_euler_plant import audit_step

    base, old = setup()
    pipe = base.pipes["branch"]["pipe"]
    state = dict(old.pipes)["branch"]
    model = PREquilibriumEulerPlant(
        base.transport,
        {
            "east": {
                "pipe": pipe,
                "boundary_nodes": ("plenum", None),
                "direction": (1.0, 0.0, 0.0),
            },
            "north": {
                "pipe": pipe,
                "boundary_nodes": ("plenum", None),
                "direction": (0.0, 1.0, 0.0),
            },
        },
    )
    before = model.initialize(old.transport, {"east": state, "north": state})
    dt = 0.1 * model.maximum_step_s
    heat = cooling_energy(pipe, state)
    commands = {
        "pipe_heat_w": {"east": [heat / dt, 0.0, 0.0]},
        "pipe_heat_temperatures_k": {"east": [100.0, None, None]},
    }
    heated, _ = model.step(before, dt, **commands)
    end, report = model.step(heated, dt)
    assert any(
        abs(a - b) > 1e-9
        for a, b in zip(
            dict(end.pipes)["east"].bulk_stocks[0], dict(end.pipes)["north"].bulk_stocks[0]
        )
    )
    assert audit_step(model, heated, report)["accepted"]


def test_cli_complete_schedule_and_restart_match_the_same_physics():
    from check_pr_multifluid_pipe import audit
    from run_pr_multifluid_pipe import run

    model, state = setup()
    dt = 0.1 * model.maximum_step_s
    command = {"duration_s": 2 * dt, "maximum_step_s": dt}
    config = json.loads(
        json.dumps(
            {"model": model.to_dict(), "initial": {"state": state.to_dict()}, "schedule": [command]}
        )
    )
    report = json.loads(json.dumps(run(config)))
    assert audit(config, report)["accepted"]
    copied = PREquilibriumEulerPlant.from_dict(config["model"])
    checkpoint = PREquilibriumEulerPlantState.from_dict(report["state"])
    uninterrupted = model.advance(
        PREquilibriumEulerPlantState.from_dict(report["state"]), dt, maximum_step_s=dt
    )
    resumed = copied.advance(checkpoint, dt, maximum_step_s=dt)
    assert resumed["state"] == uninterrupted["state"]
