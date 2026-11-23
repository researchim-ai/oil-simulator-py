"""Restart regression: a check valve cannot switch thermal equations in Newton."""

import json
from pathlib import Path

import pytest
import torch
from test_surface_pr_pump_connection import setup

from surface.liquid import HydraulicConvergenceError
from surface.pr_gravity_transport import (
    PRGravityTransportNetwork,
    PRGravityTransportState,
    _ColumnCoupling,
)
from surface.pr_pump_connection import PumpConnectionBundle

DATA = Path(__file__).parent / "data/surface/pr_restart_check_valve.json"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def captured_branch(device="cpu"):
    data = json.loads(DATA.read_text())
    model = PRGravityTransportNetwork.from_dict(data["model"], device=device)
    state = PRGravityTransportState.from_dict(data["state"])
    t = model.transport
    coupling = _ColumnCoupling(
        model, state, data["dt_s"], data["order"], data["openings"], data["heat_w"], data["utility"]
    )
    seeds = [t._seed(i, s) for i, (_, s) in enumerate(state.transport.nodes)]
    y, lo, hi, evaluate, offsets, *_ = t._problem(
        state.transport,
        seeds,
        data["dt_s"],
        {n: data["openings"][n] for n in t.default_openings},
        [data["heat_w"].get(n, 0.0) for n in t.names],
        data["losses"],
        external_ports=coupling.external_ports,
        active_override=coupling.active,
        guesses_override=coupling.ordinary_guesses,
        drive_torque_nm=data["drive_torque_nm"],
    )
    y, lo, hi, evaluate, _ = coupling.prepare(y, lo, hi, evaluate, offsets, data["dt_s"], None)
    return model, state, data, y, evaluate


@pytest.mark.parametrize("device", DEVICES)
def test_fixed_branch_removes_actual_restart_thermal_jump(device):
    model, _, data, initial, evaluate = captured_branch(device)
    f = model.transport.fluid
    torch.testing.assert_close(initial, f.tensor(data["initial_y"]), rtol=0, atol=1e-13)
    with torch.no_grad():
        y = f.tensor(data["failing_y"])
        residual = evaluate(y)[0]
        jacobian = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
        direction = torch.linalg.solve(jacobian, -residual)
        # The old residual jumped by 0.2205 even at alpha=2**-24.
        # On the same fixed branch the full Newton update is now smooth.
        changed = evaluate(y + direction)[0]
    assert float(changed.abs().max().cpu()) < 1e-6


def test_actual_restart_slow_step_opens_valve_and_balances():
    data = json.loads(DATA.read_text())
    model = PRGravityTransportNetwork.from_dict(data["model"])
    old = PRGravityTransportState.from_dict(data["state"])
    before = old.to_dict()
    new, report = model._step(
        old,
        data["dt_s"],
        data["openings"],
        data["heat_w"],
        data["utility"],
        data["losses"],
        drive_torque_nm=data["drive_torque_nm"],
    )
    assert old.to_dict() == before
    model.inspect(new)
    assert report["quadrature_error"] <= 1
    for connection in model.transport.pump_connections:
        row = report["transport"]["pump_connections"][connection.name]
        assert not row["valve_closed"]
        assert row["flow_m3_s"] > 0
        assert row["entropy_production_w_k"] >= -1e-8
        receiver = dict(new.transport.nodes)[connection.target]
        gap = (
            receiver.pressure_pa
            + connection.outlet_loss_pa_s2_m6 * row["flow_m3_s"] ** 2
            - row["pump_pressure_pa"]
        )
        assert abs(gap) < 1e-5
    assert max(map(abs, report["conservation_residuals"][:-1])) < 1e-8
    assert abs(report["conservation_residuals"][-1]) < 1e-5


@pytest.mark.parametrize("closed", [False, True])
def test_incompatible_active_branch_requests_switch_without_acceptance(closed):
    model, state, _ = setup()
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(
        state,
        seeds,
        1e-5,
        model.default_openings,
        [0.0, 0.0],
        pump_valve_states={"booster": closed},
    )
    bundle = evaluate.pump_connections
    unit, _a, b, offset, *_ = bundle.entries[0]
    y[offset] = 0.0
    target = dict(state.nodes)[model.names[b]].pressure_pa
    y[offset + 1] = y.new_tensor(target * (1.01 if closed else 0.99)).log()
    _, physical = evaluate(y)
    assert bundle.check_valve_updates(y, physical["data"]) == {unit.name: not closed}


def test_legacy_policy_retains_discontinuous_reference(monkeypatch):
    monkeypatch.setattr(PumpConnectionBundle, "_fixed_check_valves", False)
    model, _, data, _, evaluate = captured_branch()
    y = model.transport.fluid.tensor(data["failing_y"])
    with torch.no_grad():
        residual = evaluate(y)[0]
        jacobian = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
        direction = torch.linalg.solve(jacobian, -residual)
        changed = evaluate(y + 2**-24 * direction)[0]
    assert float(changed.abs().max()) > 0.2
    assert float(residual.abs().max()) < 0.002


def test_active_set_retry_budget_cannot_accept_inconsistent_valve(monkeypatch):
    model, old, _ = setup()
    before = old.to_dict()
    monkeypatch.setattr(
        PumpConnectionBundle, "check_valve_updates", lambda *a, **kw: {"booster": True}
    )
    with pytest.raises(HydraulicConvergenceError, match="check-valve active-set budget"):
        model._step(
            old,
            1e-5,
            model.default_openings,
            [0.0, 0.0],
            [350.0, 350.0],
            _pump_valve_attempt=model.maximum_branch_attempts,
        )
    assert old.to_dict() == before
