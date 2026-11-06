"""Synthetic signed actual-flow closure; no field calibration claim."""

import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from test_surface_pr_pump_connection import setup as forward_setup

from surface.pr_pump_connection import PRPumpConnection
from surface.pr_pump_four_quadrant import SignedPumpMap


def setup(speed=100.0, backpressure=1e6):
    base, _, initial = forward_setup(backpressure=backpressure)
    unit = replace(
        base.pump_connections[0],
        card=replace(base.pump_connections[0].card, check_valve=False),
        signed_map=SignedPumpMap(100.0, 200.0, 1e-3, 5e-4, 1e8, 1e-9, 1e-4),
        mapped_speed_rad_s=speed,
    )
    config = base.to_dict()
    config["pump_connections"] = [asdict(unit)]
    model = type(base).from_dict(config)
    return model, model.initialize(initial), initial


@pytest.mark.parametrize("speed", [-100.0, 0.0, 100.0])
@pytest.mark.parametrize("backpressure", [9e5, 1e6, 1.1e6])
def test_quadrants_conserve_and_independent_audit(speed, backpressure, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    model, _old, initial = setup(speed, backpressure)
    cfg = {
        "model": model.to_dict(),
        "initial": {"nodes": initial},
        "schedule": [{"duration_s": 1e-5}],
    }
    cfg = json.loads(json.dumps(cfg))
    result = json.loads(json.dumps(run(cfg)))
    row = result["intervals"][0]["steps"][0]["pump_connections"]["booster"]
    assert row["shaft_power_w"] == pytest.approx(row["load_torque_nm"] * speed)
    assert row["shaft_power_w"] - row["hydraulic_power_w"] == pytest.approx(
        row["dissipation_w"], abs=1e-10
    )
    assert row["entropy_production_w_k"] >= -1e-8
    assert max(map(abs, result["conservation_residuals"])) < 1e-7
    assert audit(cfg, result)["accepted"]
    bad = copy.deepcopy(result)
    bad["intervals"][0]["steps"][0]["pump_connections"]["booster"]["load_torque_nm"] += 1.0
    with pytest.raises(ValueError):
        audit(cfg, bad)
    if speed == 0 and backpressure != 1e6:
        assert row["load_torque_nm"] != 0
        assert row["shaft_power_w"] == 0


@pytest.mark.parametrize("flow", [-1e-4, 0.0, 1e-4])
def test_common_jacobian_fd_and_sparsity(flow):
    model, state, _ = setup(0.0 if flow == 0 else 100.0)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, offsets, *_ = model._problem(
        state, seeds, 1e-5, model.default_openings, [0.0, 0.0]
    )
    offset = evaluate.pump_connections.entries[0][3]
    y[offset] = flow / 1e-3
    jac = torch.func.jacfwd(lambda z: evaluate(z)[0])(y)
    direction = torch.linspace(-1e-4, 1e-4, len(y), dtype=y.dtype)
    eps = 1e-4
    if flow == 0:
        # Upwind donor selection has a directional derivative at reversal.
        direction[offset] = abs(direction[offset])
        fd = (evaluate(y + eps * direction)[0] - evaluate(y)[0]) / eps
    else:
        fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    torch.testing.assert_close(jac @ direction, fd, rtol=2e-4, atol=1e-7)
    support = (
        set(zip(*model._sparsity(offsets, model.default_openings))) | evaluate.additional_sparsity
    )
    mask = torch.zeros_like(jac, dtype=torch.bool)
    for r, c in support:
        mask[r, c] = True
    assert not bool((jac[~mask].abs() > 1e-12).any())


def test_opt_in_requires_map_and_no_valve():
    model, _, _ = forward_setup()
    unit = model.pump_connections[0]
    with pytest.raises(ValueError, match="requires"):
        replace(unit, mapped_speed_rad_s=-1.0)
    with pytest.raises(ValueError, match="check_valve"):
        replace(unit, signed_map=SignedPumpMap(100.0, 200.0, 1e-3, 5e-4, 1e8))
    signed, _, _ = setup()
    unit = signed.pump_connections[0]
    assert PRPumpConnection.from_dict(asdict(unit)) == unit


@pytest.mark.parametrize("flow", [-1e-4, 1e-4])
def test_live_donor_composition_enthalpy_and_density(flow):
    model, _, initial = setup()
    initial["feed"]["mole_fractions"] = [0.001, 0.999, 0.0]
    initial["return"]["mole_fractions"] = [0.002, 0.998, 0.0]
    initial["return"]["temperature_k"] = 355.0
    state = model.initialize(initial)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-5, model.default_openings, [0.0, 0.0])
    bundle = evaluate.pump_connections
    unit, a, b, offset, *_ = bundle.entries[0]
    y[offset] = flow / unit.signed_map.maximum_flow_m3_s
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    net, _, records, traces, _ = bundle.evaluate(y, data, {})
    donor = data[a if flow > 0 else b]
    row = records[unit.name]
    mw = (donor["x"][0] * model.fluid.molar_masses).sum()
    torch.testing.assert_close(row["mass_rate_kg_s"], row["flow_m3_s"] * donor["rho"])
    torch.testing.assert_close(-net[a][:-2], flow * donor["rho"] / mw * donor["x"][0])
    torch.testing.assert_close(traces[0][6], donor["x"][0])
    energy = net[a][-1] + net[b][-1]
    # At this arbitrary Newton trial the port ledger follows the live PH trace,
    # rather than asserting the still-unsolved thermodynamic residual is zero.
    hin = donor["q"]["molar_enthalpy_j_mol"][0]
    pout = data[b if flow > 0 else a]["p"]
    hd = model.fluid.tensor_homogeneous(
        pout, y[offset + 4] * model.fluid.tref, donor["x"][0], "liquid"
    )["molar_enthalpy_j_mol"]
    heat = donor["rho"] * 9.80665 * unit.signed_map.viscous_torque_coefficient_m4_s * 100.0**2
    torch.testing.assert_close(energy, abs(flow) * donor["rho"] / mw * (hd - hin) + heat)


@pytest.mark.parametrize("speed", [-100.0, 100.0])
def test_reversal_switches_actual_donor_with_changed_composition(speed):
    model, _, initial = setup(speed)
    initial["feed"]["mole_fractions"] = [0.001, 0.999, 0.0]
    initial["return"]["mole_fractions"] = [0.002, 0.998, 0.0]
    state = model.initialize(initial)
    new, report = model._step(state, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    row = report["pump_connections"]["booster"]
    assert row["flow_m3_s"] * speed > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7
    donor = "feed" if speed > 0 else "return"
    before, after = dict(state.nodes)[donor], dict(new.nodes)[donor]
    removed = torch.tensor(before.component_moles, dtype=torch.float64) - torch.tensor(
        after.component_moles, dtype=torch.float64
    )
    assert float(removed[0] / removed[1]) == pytest.approx(
        initial[donor]["mole_fractions"][0] / initial[donor]["mole_fractions"][1], rel=1e-2
    )


@pytest.mark.parametrize("flow", [-1e-4, 1e-4])
def test_validity_uses_live_donor_and_absolute_reverse_head(flow):
    model, state, _ = setup(-100.0)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-5, model.default_openings, [0.0, 0.0])
    bundle = evaluate.pump_connections
    unit, a, b, offset, *_ = bundle.entries[0]
    y[offset] = flow / unit.signed_map.maximum_flow_m3_s
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    donor = a if flow > 0 else b
    # Keep trace temperatures at donor temperature, making entropy nonnegative.
    base = bundle.evaluate(y, data, {})[-1]
    assert bool(base)
    bad = [dict(d) for d in data]
    bad[donor]["rho"] = y.new_tensor(1.0)
    assert not bool(bundle.evaluate(y, bad, {})[-1])
    bad = [dict(d) for d in data]
    bad[donor]["q"] = dict(data[donor]["q"])
    bad[donor]["q"]["frozen_sound_speed_m_s"] = y.new_ones(1)
    assert not bool(bundle.evaluate(y, bad, {})[-1])


def test_example_changed_donor_independent_audit_and_disabled(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/surface_pr_signed_pump_connection.json").read_text())
    for enabled in (True, False):
        cfg["schedule"][0]["openings"] = {"booster": enabled}
        result = json.loads(json.dumps(run(cfg)))
        assert audit(cfg, result)["accepted"]
        row = result["intervals"][0]["steps"][0]["pump_connections"]["booster"]
        if not enabled:
            assert row["load_torque_nm"] == row["shaft_power_w"] == row["dissipation_w"] == 0


def test_reverse_new_component_activates_receiver_support():
    model, _, initial = setup(-100.0)
    initial["return"]["mole_fractions"] = [0.002, 0.998, 0.0]
    state = model.initialize(initial)
    new, report = model._step(state, 1e-5, model.default_openings, [0.0, 0.0], [350.0, 350.0])
    assert report["pump_connections"]["booster"]["flow_m3_s"] < 0
    assert dict(new.nodes)["feed"].component_moles[0] > 0
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_signed_connection_rotor_crosses_zero_with_explicit_load(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    from surface.pr_machine_drive import PRMachineDrive
    from surface.rotating_drive import RotatingDrive

    base, _, initial = setup(100.0, 1.1e6)
    config = base.to_dict()
    config["drives"] = [
        asdict(
            PRMachineDrive("booster", RotatingDrive(0.1, 200.0, 30.0, 30.0), 100.0, -20.0, 0.001)
        )
    ]
    cfg = json.loads(
        json.dumps(
            {"model": config, "initial": {"nodes": initial}, "schedule": [{"duration_s": 1e-5}]}
        )
    )
    result = json.loads(json.dumps(run(cfg)))
    assert audit(cfg, result)["accepted"]
    assert dict(result["intervals"][0]["state"]["rotors"])["booster"] < 0
    for row in result["intervals"][0]["steps"]:
        pump, drive = row["pump_connections"]["booster"], row["drives"]["booster"]
        assert not pump["valve_closed"]
        assert pump["shaft_power_w"] == pytest.approx(
            pump["load_torque_nm"] * pump["mapped_speed_rad_s"]
        )
        assert abs(drive["energy_residual_j"]) < 1e-8
        assert max(map(abs, row["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("speed", [-100.0, 100.0])
def test_zero_flow_rotating_viscous_work_is_finite_donor_heat(speed):
    model, state, _ = setup(speed)
    seeds = [model._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
    y, _, _, evaluate, *_ = model._problem(state, seeds, 1e-5, model.default_openings, [0.0, 0.0])
    bundle = evaluate.pump_connections
    unit, a, b, offset, *_ = bundle.entries[0]
    y[offset] = 0.0
    data = [model._state_data(i, s) for i, (_, s) in enumerate(state.nodes)]
    net, _, records, _, _ = bundle.evaluate(y, data, {})
    row = records[unit.name]
    assert row["mass_rate_kg_s"] == 0
    assert row["shaft_power_w"] > 0
    assert row["entropy_production_w_k"] > 0
    torch.testing.assert_close(row["shaft_power_w"], row["load_torque_nm"] * speed)
    torch.testing.assert_close(net[a][-1] + net[b][-1], row["shaft_power_w"])
    assert bool(
        torch.isfinite(torch.func.jacfwd(lambda z: bundle.evaluate(z, data, {})[1][0])(y)).all()
    )
