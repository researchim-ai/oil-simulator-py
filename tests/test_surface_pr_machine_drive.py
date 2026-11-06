"""Coupled fluid/shaft stocks, live load feedback, isolation and restart."""

import json
from dataclasses import asdict

import pytest
import torch
from test_surface_pr_compressor_unit import setup as compressor_setup
from test_surface_pr_pump_unit import setup as pump_setup

from surface.pr_equilibrium_transport import (
    PREquilibriumTransportNetwork,
    PREquilibriumTransportState,
)
from surface.pr_machine_drive import PRMachineDrive, isolated_motion
from surface.rotating_drive import RotatingDrive


def setup(kind="pump", dense_limit=128, initial_speed=None, torque=2.0):
    base, _, initial = (
        pump_setup(dense_limit=dense_limit)
        if kind == "pump"
        else compressor_setup(dense_limit=dense_limit)
    )
    name = "booster" if kind == "pump" else "compressor"
    nominal = 100.0 if kind == "pump" else 150.0
    drive = PRMachineDrive(
        name,
        RotatingDrive(
            0.1,
            300.0,
            30.0,
            30.0,
            viscous_drag_nm_s=0.001,
            motor_efficiency=0.9,
            regeneration_efficiency=0.8,
        ),
        nominal,
        torque,
        initial_speed,
    )
    data = base.to_dict()
    data["drives"] = [asdict(drive)]
    model = PREquilibriumTransportNetwork.from_dict(data)
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("kind", ["pump", "compressor"])
@pytest.mark.parametrize("dense_limit", [1, 128])
def test_running_machine_and_rotor_exchange_power_in_same_newton(kind, dense_limit):
    model, old, _ = setup(kind, dense_limit)
    new, report = model._step(
        old, 1e-5, model.default_openings, [0.0] * len(model.nodes), [350.0] * len(model.nodes)
    )
    name = model.drives[0].machine
    d = report["drives"][name]
    assert dict(new.rotors)[name] > dict(old.rotors)[name]
    assert abs(d["energy_residual_j"]) < 1e-8
    assert d["shaft_work_j"] > 0
    assert report["external_work_j"] == pytest.approx(d["electrical_energy_j"], abs=1e-12)
    assert report["ambient_heat_j"] == pytest.approx(
        -d["drag_heat_j"] - d["motor_heat_j"], abs=1e-12
    )
    fluid = sum(s.total_energy_j for _, s in new.nodes) - sum(
        s.total_energy_j for _, s in old.nodes
    )
    assert fluid == pytest.approx(d["shaft_work_j"], abs=1e-7)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    restored = PREquilibriumTransportState.from_dict(json.loads(json.dumps(new.to_dict())))
    type(model).from_dict(json.loads(json.dumps(model.to_dict()))).inspect(restored)


def test_rotor_jacobian_has_two_way_hydraulic_coupling_and_sparse_support():
    model, state, _ = setup(dense_limit=1)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, offsets, *_ = model._problem(
        state, seeds, 1e-5, model.default_openings, [0.0] * len(model.nodes)
    )
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=1e-5, atol=1e-7)
    assert bool((jac[:-1, -1].abs() > 1e-12).any()) and bool((jac[-1, :-1].abs() > 1e-12).any())
    base = model._sparsity(offsets, model.default_openings)
    support = set(zip(*base)) | evaluate.additional_sparsity
    mask = torch.zeros_like(jac, dtype=torch.bool)
    for row, col in support:
        mask[row, col] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())


@pytest.mark.parametrize(
    "torque,drag,duration", [(2.0, 0.0, 0.2), (-20.0, 0.2, 6.0), (0.0, 10.0, 0.1)]
)
def test_isolated_torch_shaft_matches_independent_existing_exact_drive(torque, drag, duration):
    drive = RotatingDrive(1.0, 300.0, 30.0, 30.0, viscous_drag_nm_s=drag)
    old = torch.tensor(100.0, dtype=torch.float64)
    end, motor, heat = isolated_motion(old, old.new_tensor(torque), duration, drive)
    ref = drive.advance(drive.initialize(100.0), duration, torque)
    assert float(end) == pytest.approx(ref["state"]["angular_velocity_rad_s"], rel=1e-12, abs=1e-12)
    assert float(motor) == pytest.approx(ref["motor_work_j"], rel=1e-12, abs=1e-10)
    assert float(heat) == pytest.approx(ref["drag_heat_j"], rel=1e-12, abs=1e-10)


def test_isolated_machine_brakes_to_rest_without_fluid_inventory_loss():
    model, state, _ = setup(torque=-20.0)
    # Real stop event occurs inside the long interval; the machine is isolated.
    end, report = model._step(
        state, 2.0, {**{k: False for k in model.default_openings}}, [0.0] * 3, [350.0] * 3
    )
    assert dict(end.rotors)["booster"] == 0.0
    d = report["drives"]["booster"]
    assert d["shaft_work_j"] == 0 and d["electrical_energy_j"] < 0
    for (_, a), (_, b) in zip(state.nodes, end.nodes):
        assert b.component_moles == pytest.approx(a.component_moles, abs=1e-11)
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("kind", ["pump", "compressor"])
def test_independent_cli_audit_includes_rotor_and_electrical_balances(kind, monkeypatch):
    import copy
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    model, _, initial = setup(kind)
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-5},
                    {
                        "duration_s": 1e-5,
                        "openings": {k: False for k in model.default_openings},
                        "drive_torque_nm": {model.drives[0].machine: -1.0},
                    },
                ],
            }
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    broken = copy.deepcopy(result)
    broken["intervals"][0]["steps"][0]["state"]["rotors"][0][1] *= 1.01
    with pytest.raises(ValueError):
        audit(cfg, broken)


def test_drive_energy_is_preserved_in_joined_graph_and_gravity_wrapper():
    from surface.pr_gravity_transport import PRGravityTransportNetwork
    from surface.pr_resolved_separator import join_pr_transport_graphs

    model, state, _ = setup()
    joined = join_pr_transport_graphs([model])
    assert joined.model_id == model.model_id
    graph = PRGravityTransportNetwork(joined, {}, [])
    old = graph.initialize(state, {})
    new, report = graph._step(old, 1e-5, model.default_openings, {}, {}, {})
    assert new.transport.rotors
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("kind", ["pump", "compressor"])
def test_isolated_drive_can_start_from_rest(kind):
    model, state, _ = setup(kind, initial_speed=0.0)
    name = model.drives[0].machine
    end, report = model._step(
        state,
        0.1,
        {**{k: False for k in model.default_openings}},
        [0.0] * len(model.nodes),
        [350.0] * len(model.nodes),
    )
    assert dict(end.rotors)[name] > 0
    assert report["drives"][name]["shaft_work_j"] == 0.0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_motor_losses_heat_finite_coolant_without_external_heat_sink(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    base, _, initial = setup()
    data = base.to_dict()
    data["drives"][0]["heat_sink_node"] = "feed"
    model = type(base).from_dict(data)
    state = model.initialize(initial)
    end, report = model._step(
        state, 1e-4, {**{k: False for k in model.default_openings}}, [0.0] * 3, [350.0] * 3
    )
    d = report["drives"]["booster"]
    assert d["heat_to_node_j"] > 0 and d["heat_to_environment_j"] == 0.0
    assert report["ambient_heat_j"] == 0.0
    fluid_change = sum(s.total_energy_j for _, s in end.nodes) - sum(
        s.total_energy_j for _, s in state.nodes
    )
    assert fluid_change == pytest.approx(d["heat_to_node_j"], abs=1e-7)
    cfg = json.loads(
        json.dumps(
            {
                "model": model.to_dict(),
                "initial": {"nodes": initial},
                "schedule": [
                    {"duration_s": 1e-5, "openings": {k: False for k in model.default_openings}}
                ],
            }
        )
    )
    assert audit(cfg, json.loads(json.dumps(run(cfg))))["accepted"]
