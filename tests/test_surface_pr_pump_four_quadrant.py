"""Synthetic constitutive tests, not industrial pump validation."""

from dataclasses import asdict, replace

import pytest
import torch
from test_surface_pr_pump_unit import setup

from surface.pr_pump_four_quadrant import SignedPumpMap
from surface.pr_pump_unit import PRPumpUnit


def mapping():
    return SignedPumpMap(100.0, 200.0, 1e-3, 5e-4, 1e8, 1e-9, 1e-4)


@pytest.mark.parametrize("speed", [-100.0, 0.0, 100.0])
@pytest.mark.parametrize("flow", [-1e-4, 0.0, 1e-4])
def test_signed_energy_and_finite_derivatives(flow, speed):
    q = torch.tensor(flow, dtype=torch.float64, requires_grad=True)
    omega = torch.tensor(speed, dtype=torch.float64, requires_grad=True)
    head, torque, shaft, hydraulic, loss, valid = mapping().evaluate(q, omega, 800.0)
    assert valid
    assert loss >= 0
    assert float(shaft - hydraulic) == pytest.approx(float(loss), abs=1e-12)
    assert all(torch.isfinite(v) for v in torch.autograd.grad(head + torque + shaft, (q, omega)))
    if speed == 0 and flow != 0:
        assert shaft == 0
        assert hydraulic < 0
        assert torque != 0


def test_map_rejects_envelope_and_bad_coefficients():
    q = torch.tensor(2e-3)
    assert not mapping().evaluate(q, 100.0, 800.0)[-1]
    assert not mapping().evaluate(q * 0, -201.0, 800.0)[-1]
    with pytest.raises(ValueError):
        replace(mapping(), flow_loss_coefficient_s2_m5=-1.0)


@pytest.mark.parametrize("speed,velocity", [(-100.0, -0.03), (100.0, -0.03), (-100.0, 0.03)])
def test_finite_cell_signed_work_conserves_and_roundtrips(speed, velocity):
    model, _, initial = setup(velocity=velocity)
    unit = replace(model.pumps[0], signed_map=mapping(), mapped_speed_rad_s=speed)
    assert PRPumpUnit.from_dict(asdict(unit)) == unit
    config = model.to_dict()
    config["pumps"] = [asdict(unit)]
    model = type(model).from_dict(config)
    old = model.initialize(initial)
    _, report = model._step(old, 1e-5, model.default_openings, [0.0] * 3, [350.0] * 3)
    row = report["pumps"]["booster"]
    assert row["shaft_power_w"] == pytest.approx(row["load_torque_nm"] * speed)
    assert row["shaft_power_w"] - row["hydraulic_power_w"] == pytest.approx(row["dissipation_w"])
    assert report["external_work_j"] == pytest.approx(1e-5 * row["shaft_power_w"])
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_reverse_flow_drives_positive_rotor_with_conservative_load():
    from surface.pr_machine_drive import PRMachineDrive
    from surface.rotating_drive import RotatingDrive

    base, _, initial = setup(velocity=-0.03)
    unit = replace(base.pumps[0], signed_map=mapping())
    drive = PRMachineDrive("booster", RotatingDrive(0.1, 200.0, 30.0, 30.0), 100.0)
    config = base.to_dict()
    config.update(pumps=[asdict(unit)], drives=[asdict(drive)])
    model = type(base).from_dict(config)
    old = model.initialize(initial)
    new, report = model._step(old, 1e-5, model.default_openings, [0.0] * 3, [350.0] * 3)
    assert dict(new.rotors)["booster"] > dict(old.rotors)["booster"]
    pump, rotor = report["pumps"]["booster"], report["drives"]["booster"]
    assert pump["shaft_power_w"] < 0
    assert rotor["shaft_work_j"] < 0
    assert abs(rotor["energy_residual_j"]) < 1e-8
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


@pytest.mark.parametrize("drag", [0.0, 0.2, 10.0])
@pytest.mark.parametrize("old,torque", [(1.0, -2.0), (-1.0, 2.0), (-1.0, -2.0), (0.0, -2.0)])
def test_signed_isolated_exact_motion_and_electrical_split(drag, old, torque):
    from scipy.integrate import quad, solve_ivp
    from scipy.optimize import brentq

    from surface.pr_machine_drive import isolated_motion, signed_motor_work
    from surface.rotating_drive import RotatingDrive

    drive = RotatingDrive(1.0, 300.0, 30.0, 30.0, viscous_drag_nm_s=drag)
    w, tau = torch.tensor(old, dtype=torch.float64), torch.tensor(torque, dtype=torch.float64)
    end, motor, heat = isolated_motion(w, tau, 2.0, drive, signed=True)
    ref = solve_ivp(
        lambda t, y: [(torque - drag * y[0]), y[0], y[0] ** 2],
        (0.0, 2.0),
        [old, 0.0, 0.0],
        rtol=1e-12,
        atol=1e-13,
        method="DOP853",
        dense_output=True,
    )
    assert float(end) == pytest.approx(ref.y[0, -1], abs=1e-10)
    assert float(motor) == pytest.approx(torque * ref.y[1, -1], abs=1e-10)
    assert float(heat) == pytest.approx(drag * ref.y[2, -1], abs=1e-10)
    assert float(motor - heat) == pytest.approx(0.5 * (float(end) ** 2 - old**2), abs=1e-10)
    positive, negative = signed_motor_work(w, end, tau, 2.0, drive, isolated=True)
    points = [brentq(lambda t: ref.sol(t)[0], 0.0, 2.0)] if old * float(end) < 0 else []
    for sign, actual in ((1.0, positive), (-1.0, negative)):
        expected = quad(
            lambda t, sign=sign: max(0.0, sign * torque * ref.sol(t)[0]),
            0.0,
            2.0,
            points=points,
            epsabs=1e-11,
        )[0]
        assert float(actual) == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("pressure_delta", [0.0, 100.0])
def test_open_signed_rotor_crosses_zero_and_preserves_energy(pressure_delta):
    from surface.pr_machine_drive import PRMachineDrive
    from surface.rotating_drive import RotatingDrive

    base, _, initial = setup()
    initial["return"]["pressure_pa"] += pressure_delta
    unit = replace(base.pumps[0], signed_map=mapping())
    drive = PRMachineDrive(
        "booster",
        RotatingDrive(0.1, 200.0, 30.0, 30.0, motor_efficiency=0.8, regeneration_efficiency=0.7),
        100.0,
        -20.0,
        0.001,
    )
    config = base.to_dict()
    config.update(pumps=[asdict(unit)], drives=[asdict(drive)])
    model = type(base).from_dict(config)
    old = model.initialize(initial)
    new, report = model._step(old, 1e-5, model.default_openings, [0.0] * 3, [350.0] * 3)
    assert dict(old.rotors)["booster"] > 0 > dict(new.rotors)["booster"]
    pump, rotor = report["pumps"]["booster"], report["drives"]["booster"]
    if pressure_delta:
        assert pump["flow_m3_s"] < 0
    else:
        assert abs(rotor["load_evaluation_speed_rad_s"]) < 1e-12
    assert rotor["peak_electrical_draw_w"] > 0
    assert rotor["peak_electrical_export_w"] > 0
    assert rotor["motor_heat_j"] > 0
    assert abs(rotor["energy_residual_j"]) < 1e-8
    assert max(map(abs, report["conservation_residuals"])) < 1e-7


def test_signed_reversal_schedule_independent_audit_and_tamper_rejection(monkeypatch):
    import copy
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    from check_pr_equilibrium_transport import audit
    from run_pr_equilibrium_transport import run

    config = json.loads((root / "configs/surface_pr_signed_pump_reversal.json").read_text())
    result = json.loads(json.dumps(run(config)))
    assert audit(config, result)["accepted"]
    endpoints = [dict(interval["state"]["rotors"])["booster"] for interval in result["intervals"]]
    assert endpoints[0] < 0 < endpoints[1] and endpoints[2] < 0
    assert result["intervals"][0]["steps"][0]["pumps"]["booster"]["flow_m3_s"] < 0
    for key in ("load_torque_nm", "dissipation_w"):
        corrupted = copy.deepcopy(result)
        corrupted["intervals"][0]["steps"][0]["pumps"]["booster"][key] += 1.0
        with pytest.raises(ValueError):
            audit(config, corrupted)


@pytest.mark.parametrize(
    "torque,limit", [(-20.0, "max_electrical_draw_w"), (20.0, "max_electrical_export_w")]
)
def test_reverse_initial_speed_checks_signed_electrical_ratings(torque, limit):
    import json
    from pathlib import Path

    from surface.liquid import EquipmentOperatingError

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/surface_pr_signed_pump_reversal.json").read_text())
    card = config["model"]["drives"][0]
    card.update(initial_speed_rad_s=-1.0, motor_torque_nm=torque)
    card["drive"][limit] = 1.0
    base, _, _ = setup()
    model = type(base).from_dict(config["model"])
    state = model.initialize(config["initial"]["nodes"])
    with pytest.raises(EquipmentOperatingError, match="electrical rating"):
        model._step(state, 1e-5, model.default_openings, [0.0] * 3, [350.0] * 3)


def test_legacy_pump_rejects_negative_initial_speed_in_network():
    from surface.liquid import PropertyModelError
    from surface.pr_machine_drive import PRMachineDrive
    from surface.rotating_drive import RotatingDrive

    base, _, initial = setup()
    drive = PRMachineDrive(
        "booster", RotatingDrive(0.1, 200.0, 30.0, 30.0), 100.0, initial_speed_rad_s=-1.0
    )
    config = base.to_dict()
    config["drives"] = [asdict(drive)]
    model = type(base).from_dict(config)
    with pytest.raises(PropertyModelError, match="Rotor speed"):
        model.initialize(initial)


def test_forward_legacy_config_keeps_checkpoint_model_identity():
    import json
    from pathlib import Path

    from surface.operation import _digest
    from surface.pr_equilibrium_transport import PREquilibriumTransportNetwork

    path = Path(__file__).resolve().parents[1] / "configs/surface_pr_pump_unit.json"
    old = json.loads(path.read_text())["model"]
    model = PREquilibriumTransportNetwork.from_dict(old)
    assert model.model_id == _digest(old)
    assert "signed_map" not in model.to_dict()["pumps"][0]
