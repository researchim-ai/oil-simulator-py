"""Finite aperture conservative gravitational vessel dynamics."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.component_energy import ComponentEnergyStream
from surface.compositional_separator import EquilibriumOutlet
from surface.stratified_pr import StratifiedPRColumn, StratifiedPRState
from surface.stratified_pr_flow import StratifiedFeed, StratifiedPRFlow
from surface.stratified_pr_thermal import StratifiedThermalState
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def setup(device="cpu", two=False):
    cfg = json.loads(
        (
            ROOT
            / (
                "configs/surface_stratified_pr.json"
                if two
                else "configs/surface_stratified_pr_gas.json"
            )
        ).read_text()
    )
    cfg["system"]["geometry"]["length_m"] = 2.0
    c = StratifiedPRColumn.from_dict(cfg["system"], device=device)
    s = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    return c, s


@pytest.mark.parametrize("device", DEVICES)
def test_withdrawal_balance_entropy_and_restart(device):
    c, s = setup(device)
    outlet = EquilibriumOutlet("export", 990000.0, 1e-10, height_m=1.0, diameter_m=0.05)
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0, 0.0), outlets=[outlet], initial_step_s=0.02)
    initial = model.initialize(s, s.temperature_k)
    final, report = model._step(initial, 0.02)
    assert all(b < a for a, b in zip(s.component_moles, final.column.component_moles))
    for a, b, out in zip(
        s.component_moles, final.column.component_moles, report["outlet_component_moles"]
    ):
        assert b - a == pytest.approx(-out, abs=1e-10)
    assert final.column.total_energy_j - s.total_energy_j == pytest.approx(
        -report["outlet_energy_j"], abs=1e-8
    )
    assert report["entropy_production_j_k"] >= -1e-7
    assert report["outlets"][0]["phase_area_fractions"] == pytest.approx([1.0])
    restored = StratifiedPRFlow.from_dict(json.loads(json.dumps(model.to_dict())), device=device)
    assert restored.model_id == model.model_id
    restored.inspect(StratifiedThermalState.from_dict(final.to_dict()))


def test_split_aperture_and_monolithic_jacobian():
    c, s = setup(two=True)
    interface = s.interface_heights_m[0]
    outlet = EquilibriumOutlet("split", 995000.0, 1e-11, height_m=interface, diameter_m=0.05)
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0), outlets=[outlet])
    initial = model.initialize(s, s.temperature_k)
    seed = c._seed(c.fluid.tensor(s.component_moles), s.total_energy_j, 1e6, s.temperature_k)
    y, _low, _high, evaluate, *_ = model._problem(initial, 0.01, seed, 4)
    jac = torch.func.jacfwd(lambda v: evaluate(v)[0])(y)
    direction = torch.linspace(-0.01, 0.01, len(y), dtype=y.dtype)
    eps = 1e-4
    fd = (evaluate(y + eps * direction)[0] - evaluate(y - eps * direction)[0]) / (2 * eps)
    assert torch.isfinite(jac).all()
    assert torch.max(torch.abs(jac @ direction - fd)) < 1e-6
    _final, report = model._step(initial, 0.01)
    assert report["outlets"][0]["phase_area_fractions"] == pytest.approx([0.5, 0.5], abs=0.005)
    assert abs(report["energy_residual_j"]) < 1e-6


def test_prescribed_feed_and_full_interval_ledger():
    c, s = setup()
    feed = ComponentEnergyStream.from_pt(
        c.fluid, [1e-4, 2e-4], 1010000.0, 320.0, elevation_m=c.elevation_m + 1.0
    )
    model = StratifiedPRFlow(
        c, VesselWall(1000.0, 0.0), feeds=[StratifiedFeed(1.0, feed)], initial_step_s=0.02
    )
    initial = model.initialize(s, s.temperature_k)
    report = model.advance(initial, 0.02)
    final = StratifiedThermalState.from_dict(report["state"])
    assert final.time_s == 0.02
    assert final.column.component_moles[0] - s.component_moles[0] == pytest.approx(2e-6, abs=1e-10)
    assert max(map(abs, report["component_residuals_mol"])) < 1e-10
    assert abs(report["energy_residual_j"]) < 1e-6
    assert report["feed_energy_j"] == pytest.approx(0.02 * feed.total_energy_rate_w)


def test_reverse_pressure_blocks_outlet_and_bad_feed_is_atomic():
    from surface.liquid import EquipmentOperatingError

    c, s = setup()
    outlet = EquilibriumOutlet("check_valve", 1100000.0, 1e-10, height_m=1.0, diameter_m=0.05)
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0), outlets=[outlet])
    initial = model.initialize(s, s.temperature_k)
    final, report = model._step(initial, 0.02)
    assert final.column.component_moles == s.component_moles
    assert report["outlet_energy_j"] == 0.0
    assert report["outlets"][0]["volume_rate_m3_s"] == 0.0
    feed = ComponentEnergyStream.from_pt(
        c.fluid, [1e-4, 1e-4], 900000.0, 320.0, elevation_m=c.elevation_m + 1.0
    )
    bad = StratifiedPRFlow(c, model.wall, feeds=[StratifiedFeed(1.0, feed)])
    original = bad.initialize(s, s.temperature_k)
    before = original.to_dict()
    with pytest.raises(EquipmentOperatingError, match="Feed pressure"):
        bad.advance(original, 0.02)
    assert original.to_dict() == before


def test_pressure_and_level_feedback_remain_in_joint_residual():
    from surface.phase_geometry import EquilibriumLevelControl

    c, s = setup()
    pressure = EquilibriumOutlet(
        "pressure",
        990000.0,
        1e-10,
        height_m=1.0,
        diameter_m=0.05,
        pressure_setpoint_pa=999000.0,
        opening_gain_per_pa=0.0001,
    )
    # Missing rich phase must fail closed, even though a gas layer spans the port.
    level = EquilibriumOutlet(
        "level",
        990000.0,
        1e-10,
        height_m=1.0,
        diameter_m=0.05,
        level_control=EquilibriumLevelControl("Ethane", 0.5, 1.0, 1.0, 0.9),
    )
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0), outlets=[pressure, level])
    final, report = model._step(model.initialize(s, s.temperature_k), 0.02)
    assert 0 < report["outlets"][0]["opening"] < 1
    assert report["outlets"][1]["opening"] == 0
    assert report["outlets"][1]["volume_rate_m3_s"] == 0
    assert final.column.component_moles != s.component_moles


def test_ideal_gas_discharge_matches_independent_scalar_balance():
    from scipy.optimize import brentq

    from surface.peng_robinson import R

    cfg = json.loads((ROOT / "configs/surface_stratified_pr_gas.json").read_text())
    cfg["system"]["geometry"]["length_m"] = 1.0
    cfg["system"]["gravity_m_s2"] = 0.0
    cfg["system"]["fluid"]["min_pressure_pa"] = 1e-7
    cfg["initial"]["pressure_pa"] = 0.01
    c = StratifiedPRColumn.from_dict(cfg["system"])
    old = StratifiedPRState.from_dict(c.initialize(**cfg["initial"])["state"])
    outlet = EquilibriumOutlet("ideal", 0.0099, 10.0, height_m=0.5, diameter_m=0.05)
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0), outlets=[outlet])
    dt = 0.1
    actual, report = model._step(model.initialize(old, old.temperature_k), dt)
    n0 = sum(old.component_moles)
    cv = sum(
        n / n0 * (component.ideal_cp_j_mol_k - R)
        for n, component in zip(old.component_moles, c.fluid.components)
    )

    def temperature(loss):
        return n0 * cv * old.temperature_k / (n0 * cv + loss * R)

    def residual(fraction):
        loss = fraction * n0
        t = temperature(loss)
        p = (n0 - loss) * R * t / c.geometry.volume_m3
        return fraction - dt * outlet.volume_conductance_m3_s_pa * max(
            p - outlet.back_pressure_pa, 0
        ) * p / (R * t * n0)

    fraction = brentq(residual, 0.0, 0.1, xtol=1e-14)
    assert actual.column.temperature_k == pytest.approx(temperature(fraction * n0), abs=2e-6)
    assert sum(actual.column.component_moles) / n0 == pytest.approx(1 - fraction, abs=1e-8)
    assert report["outlet_energy_j"] != 0


def test_phase_disappearance_and_reappearance_with_isolated_finite_port():
    c, s = setup(two=True)
    closed = EquilibriumOutlet(
        "isolated", 990000.0, 1e-10, height_m=1.0, diameter_m=0.05, opening=0.0
    )
    model = StratifiedPRFlow(
        c, VesselWall(1e6, 1000.0, max_temperature_change_k=100.0), outlets=[closed]
    )
    hot, heating = model._step(model.initialize(s, 320.0), 100.0)
    assert len(s.phase_roots) == 2
    assert len(hot.column.phase_roots) == 1
    cold, cooling = model._step(model.initialize(hot.column, 180.0), 100.0)
    assert len(cold.column.phase_roots) == 2
    assert cold.column.component_moles == s.component_moles
    for result in (heating, cooling):
        assert abs(result["energy_residual_j"]) < 1e-6
        assert result["entropy_production_j_k"] >= 0
        assert result["outlet_component_moles"] == [0.0, 0.0]
        assert sum(result["outlets"][0]["phase_area_fractions"]) == pytest.approx(1.0)


def test_new_feed_component_activates_previously_absent_inventory():
    c, _ = setup()
    old = StratifiedPRState.from_dict(c.initialize(1e6, 320.0, [1.0, 0.0])["state"])
    stream = ComponentEnergyStream.from_pt(
        c.fluid, [0.0, 1e-4], 1010000.0, 320.0, elevation_m=c.elevation_m + 1.0
    )
    model = StratifiedPRFlow(c, VesselWall(1000.0, 0.0), feeds=[StratifiedFeed(1.0, stream)])
    final, report = model._step(model.initialize(old, old.temperature_k), 0.02)
    assert final.column.component_moles[1] == pytest.approx(2e-6, abs=1e-11)
    assert abs(report["energy_residual_j"]) < 1e-6


def test_independent_numpy_audit_rejects_modified_export_rate():
    from dataclasses import asdict

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_stratified_pr_flow import audit

    c, s = setup()
    port = EquilibriumOutlet("audit", 990000.0, 1e-10, height_m=1.0, diameter_m=0.05)
    wall = VesselWall(1000.0, 0.0)
    model = StratifiedPRFlow(c, wall, outlets=[port], initial_step_s=0.02)
    result = model.advance(model.initialize(s, s.temperature_k), 0.02)
    cfg = {
        "column": c.to_dict(),
        "wall": asdict(wall),
        "outlets": [asdict(port)],
        "feeds": [],
        "wall_temperature_k": s.temperature_k,
        "durations_s": [0.02],
    }
    payload = {"config": cfg, "initial": {"state": s.to_dict()}, "intervals": [result]}
    assert audit(cfg, payload)["accepted"]
    broken = json.loads(json.dumps(payload))
    broken["intervals"][0]["steps"][0]["outlets"][0]["component_rates_mol_s"][0] *= 2
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(cfg, broken)

    broken = json.loads(json.dumps(payload))
    broken["intervals"][0]["steps"][0]["column_state"]["solution_variables"][-1] += 0.01
    with pytest.raises(ValueError, match="audit mismatch"):
        audit(cfg, broken)


def test_combined_feed_and_withdrawal_independent_audit():
    from dataclasses import asdict

    sys.path.insert(0, str(ROOT / "scripts"))
    from check_stratified_pr_flow import audit

    c, s = setup()
    feed = StratifiedFeed(
        1.0,
        ComponentEnergyStream.from_pt(
            c.fluid, [1e-4, 2e-4], 1010000.0, 320.0, elevation_m=c.elevation_m + 1.0
        ),
    )
    port = EquilibriumOutlet("combined", 990000.0, 1e-10, height_m=1.0, diameter_m=0.05)
    wall = VesselWall(1000.0, 0.0)
    model = StratifiedPRFlow(c, wall, outlets=[port], feeds=[feed], initial_step_s=0.02)
    result = model.advance(model.initialize(s, s.temperature_k), 0.02)
    assert result["feed_energy_j"] != 0
    assert result["outlet_energy_j"] != 0
    assert max(map(abs, result["component_residuals_mol"])) < 1e-10
    cfg = {
        "column": c.to_dict(),
        "wall": asdict(wall),
        "outlets": [asdict(port)],
        "feeds": [asdict(feed)],
        "wall_temperature_k": s.temperature_k,
        "durations_s": [0.02],
    }
    payload = {"config": cfg, "initial": {"state": s.to_dict()}, "intervals": [result]}
    assert audit(cfg, payload)["accepted"]
