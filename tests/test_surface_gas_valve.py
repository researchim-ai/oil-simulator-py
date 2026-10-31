"""Ideal-gas sonic capacity, finite-area export control and facility restart."""

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_surface_facility import load_facility

from surface import (
    EquipmentOperatingError,
    FacilityCapacityControl,
    FacilityPeriod,
    GasExportValve,
    GasState,
    IdealGas,
    ideal_gas_nozzle_flow,
    run_facility_schedule,
)


def gas():
    return IdealGas(287.05, 1.4)


@pytest.mark.parametrize("ratio", [0.2, 0.5, 0.7, 0.95, 0.999999])
def test_nozzle_matches_independent_mach_area_formula(ratio):
    g = gas()
    p, t, area, cd, opening = 2e6, 320.0, 1e-4, 0.8, 0.6
    r = ideal_gas_nozzle_flow(p, t, p * ratio, g, area, cd, opening)
    gamma = g.heat_capacity_ratio
    critical = (2 / (gamma + 1)) ** (gamma / (gamma - 1))
    mach = (
        1
        if ratio <= critical
        else math.sqrt(2 / (gamma - 1) * (ratio ** (-(gamma - 1) / gamma) - 1))
    )
    expected = (
        cd
        * area
        * opening
        * p
        / math.sqrt(t)
        * math.sqrt(gamma / g.specific_gas_constant_j_kg_k)
        * mach
        * (1 + (gamma - 1) / 2 * mach**2) ** (-(gamma + 1) / (2 * (gamma - 1)))
    )
    assert r["mass_rate_kg_s"] == pytest.approx(expected, rel=5e-10)
    assert r["throat_mach"] == pytest.approx(mach, rel=1e-9)
    assert abs(r["mass_residual_kg_s"]) < 1e-12
    assert abs(r["nozzle_energy_residual_j_kg"]) < 1e-8
    assert r["regime"] == ("choked" if ratio <= critical else "subcritical")


@pytest.mark.parametrize("back", [0.4e6, 1.6e6])
def test_nozzle_all_derivatives_against_finite_differences(back):
    values = {
        "pressure_pa": 2e6,
        "temperature_k": 320.0,
        "back_pressure_pa": back,
        "gas": gas(),
        "throat_area_m2": 1e-4,
        "discharge_coefficient": 0.8,
        "opening": 0.6,
    }
    r = ideal_gas_nozzle_flow(**values)
    for argument, key, h in [
        ("pressure_pa", "inlet_pressure_pa", 2.0),
        ("back_pressure_pa", "back_pressure_pa", 2.0),
        ("temperature_k", "temperature_k", 0.001),
        ("opening", "opening", 1e-5),
        ("throat_area_m2", "throat_area_m2", 1e-8),
        ("discharge_coefficient", "discharge_coefficient", 1e-5),
    ]:
        a, b = dict(values), dict(values)
        a[argument] += h
        b[argument] -= h
        fd = (
            ideal_gas_nozzle_flow(**a)["mass_rate_kg_s"]
            - ideal_gas_nozzle_flow(**b)["mass_rate_kg_s"]
        ) / (2 * h)
        assert r["mass_derivatives"][key] == pytest.approx(fd, rel=2e-7, abs=1e-12)


def test_critical_plateau_near_equal_pressure_and_closed_boundary():
    g = gas()
    args = (2e6, 300.0)
    a = ideal_gas_nozzle_flow(*args, 0.2e6, g, 1e-4)
    b = ideal_gas_nozzle_flow(*args, 0.8e6, g, 1e-4)
    assert a["mass_rate_kg_s"] == b["mass_rate_kg_s"]
    assert a["mass_derivatives"]["back_pressure_pa"] == 0
    pc = 2e6 * a["critical_pressure_ratio"]
    lo = ideal_gas_nozzle_flow(*args, pc * (1 - 1e-7), g, 1e-4)
    hi = ideal_gas_nozzle_flow(*args, pc * (1 + 1e-7), g, 1e-4)
    assert lo["mass_rate_kg_s"] == pytest.approx(hi["mass_rate_kg_s"], rel=1e-12)
    near = ideal_gas_nozzle_flow(*args, math.nextafter(2e6, 0), g, 1e-4)
    assert near["mass_rate_kg_s"] > 0 and math.isfinite(
        near["mass_derivatives"]["back_pressure_pa"]
    )
    equal = ideal_gas_nozzle_flow(*args, 2e6, g, 1e-4)
    assert equal["mass_rate_kg_s"] == 0 and not equal["pressure_derivative_defined"]
    assert equal["mass_derivatives"]["back_pressure_pa"] is None
    closed = ideal_gas_nozzle_flow(*args, 0.8e6, g, 1e-4, opening=0)
    assert closed["regime"] == "closed" and closed["throat_mach"] is None
    json.dumps([near, equal, closed], allow_nan=False)
    with pytest.raises(EquipmentOperatingError, match="reverse"):
        ideal_gas_nozzle_flow(*args, 3e6, g, 1e-4)


def test_required_opening_derivatives_capacity_and_no_fake_zero_flow_ports():
    g = gas()
    v = GasExportValve("trim", 0.8e6, 1e-4, 0.8, 0.7)
    inlet = GasState(0.1, 2e6, 320)
    r = v.run(inlet, g)
    assert r["opening"] < 0.7 and r["nozzle"]["regime"] == "choked"
    assert r["outlet"]["temperature_k"] == 320
    for key, h in [("pressure_pa", 2.0), ("temperature_k", 0.001), ("mass_rate_kg_s", 1e-5)]:
        a = v.run(replace(inlet, **{key: getattr(inlet, key) + h}), g)
        b = v.run(replace(inlet, **{key: getattr(inlet, key) - h}), g)
        expected = (a["opening"] - b["opening"]) / (2 * h)
        k = (
            "inlet_pressure_pa"
            if key == "pressure_pa"
            else ("temperature_k" if key == "temperature_k" else "mass_rate_kg_s")
        )
        assert r["opening_derivatives"][k] == pytest.approx(expected, rel=1e-7)
    with pytest.raises(EquipmentOperatingError, match="capacity"):
        replace(v, max_opening=0.01).run(inlet, g)
    zero = v.run(replace(inlet, mass_rate_kg_s=0), g)
    assert zero["status"] == "no_flow" and zero["outlet"] is None and zero["nozzle"] is None


def plant_config():
    c = json.loads((ROOT / "configs/surface_gas_facility.json").read_text())
    c["gas_export"]["export_valve"] = {
        "name": "export_trim",
        "outlet_pressure_pa": 2e6,
        "throat_area_m2": 2e-5,
        "discharge_coefficient": 0.8,
        "max_opening": 1.0,
    }
    for limit in c["limits"]:
        if limit["name"] == "gas_delivery":
            limit["minimum"] = 1.9e6
    return c


def test_facility_finite_capacity_dispatch_balances_and_atomic_controls():
    p = load_facility(plant_config())
    r = p.run()
    assert r["feasible"]
    assert r["gas_export"]["valve"]["nozzle"]["regime"] == "choked"
    assert r["gas_export"]["outlet"]["pressure_pa"] == 2e6
    assert abs(r["gas_export"]["energy_residual_w"]) < 1e-8
    base = p.gas_export
    with pytest.raises(EquipmentOperatingError, match="capacity"):
        p.run(gas_valve_max_opening=0.1)
    assert p.gas_export == base
    dispatch = FacilityCapacityControl(scan_points=5, max_evaluations=40).solve(
        p, gas_valve_max_opening=0.1
    )
    assert dispatch["status"] == "limited" and 0 < dispatch["fraction"] < 1
    final = dispatch["facility"]
    assert final["feasible"]
    assert final["gas_export"]["valve"]["utilization"] <= 1
    assert max(abs(v) for v in final["component_mass_residuals_kg_s"].values()) < 1e-12
    assert p.gas_export == base


def test_schedule_valve_controls_and_serialized_resume():
    p = load_facility(plant_config())
    periods = [
        FacilityPeriod("normal", 300),
        FacilityPeriod("limited", 300, gas_valve_max_opening=0.1),
        FacilityPeriod("backpressure", 300, gas_valve_outlet_pressure_pa=5e6),
        FacilityPeriod("shutdown", 60, shutdown=True),
        FacilityPeriod("restart", 300, gas_valve_max_opening=1.0),
    ]
    control = FacilityCapacityControl(scan_points=5, max_evaluations=40)
    full = run_facility_schedule(p, periods, control)
    partial = run_facility_schedule(p, periods, control, stop_after=2)
    resumed = run_facility_schedule(
        p, periods, control, checkpoint=json.loads(json.dumps(partial["checkpoint"]))
    )
    assert full["checkpoint"] == resumed["checkpoint"] and full["totals"] == resumed["totals"]
    ledger = full["checkpoint"]["ledger"]
    assert ledger[0]["facility"]["gas_export"]["valve"]["nozzle"]["regime"] == "choked"
    assert ledger[2]["facility"]["gas_export"]["valve"]["nozzle"]["regime"] == "subcritical"
    assert ledger[3]["facility"] is None
    absent = plant_config()
    absent["gas_export"].pop("export_valve")
    p = load_facility(absent)
    with pytest.raises(ValueError, match="require"):
        run_facility_schedule(
            p, [FacilityPeriod("bad", 10, shutdown=True, gas_valve_max_opening=0.2)]
        )


@pytest.mark.parametrize("back", [0.4e6, 1.6e6])
def test_required_supply_pressure_and_implicit_tangents(back):
    g = gas()
    v = GasExportValve("trim", back, 1e-4, 0.8, 0.7)
    flow = ideal_gas_nozzle_flow(2e6, 320, back, g, 1e-4, 0.8, 0.6)
    mass = flow["mass_rate_kg_s"]
    r = v.required_supply_pressure(mass, 320, g, opening=0.6)
    assert r["required_pressure_pa"] == pytest.approx(2e6, abs=1e-5)
    h = mass * 1e-5
    fd = (
        v.required_supply_pressure(mass + h, 320, g, 0.6)["required_pressure_pa"]
        - v.required_supply_pressure(mass - h, 320, g, 0.6)["required_pressure_pa"]
    ) / (2 * h)
    assert r["pressure_derivatives"]["mass_rate_kg_s"] == pytest.approx(fd, rel=1e-7)
    fd = (
        replace(v, outlet_pressure_pa=back + 2).required_supply_pressure(mass, 320, g, 0.6)[
            "required_pressure_pa"
        ]
        - replace(v, outlet_pressure_pa=back - 2).required_supply_pressure(mass, 320, g, 0.6)[
            "required_pressure_pa"
        ]
    ) / 4
    assert r["pressure_derivatives"]["back_pressure_pa"] == pytest.approx(fd, rel=1e-7, abs=1e-8)


def test_back_pressure_inverse_and_nonunique_choked_plateau():
    g = gas()
    v = GasExportValve("trim", 0.8e6, 1e-4, 0.8)
    rate = ideal_gas_nozzle_flow(2e6, 320, 1.6e6, g, 1e-4, 0.8)["mass_rate_kg_s"]
    inlet = GasState(rate, 2e6, 320)
    r = v.required_back_pressure(inlet, g)
    assert r["back_pressure_pa"] == pytest.approx(1.6e6, abs=1e-5)
    h = rate * 1e-5
    fd = (
        v.required_back_pressure(replace(inlet, mass_rate_kg_s=rate + h), g)["back_pressure_pa"]
        - v.required_back_pressure(replace(inlet, mass_rate_kg_s=rate - h), g)["back_pressure_pa"]
    ) / (2 * h)
    assert r["pressure_derivatives"]["mass_rate_kg_s"] == pytest.approx(fd, rel=1e-7)
    choked = ideal_gas_nozzle_flow(2e6, 320, 0.5e6, g, 1e-4, 0.8)
    plateau = v.required_back_pressure(replace(inlet, mass_rate_kg_s=choked["mass_rate_kg_s"]), g)
    assert plateau["status"] == "critical_plateau" and plateau["back_pressure_pa"] is None
    assert plateau["max_back_pressure_pa"] == pytest.approx(2e6 * choked["critical_pressure_ratio"])
    with pytest.raises(EquipmentOperatingError, match="critical"):
        v.required_back_pressure(replace(inlet, mass_rate_kg_s=choked["mass_rate_kg_s"] * 1.01), g)


@pytest.mark.parametrize("gamma", [1 + 2**-40, 1.01, 1.4, 1.67])
def test_caloric_limit_and_weak_pressure_drop_are_numerically_stable(gamma):
    g = IdealGas(287.05, gamma)
    r = ideal_gas_nozzle_flow(2e6, 300, 1.6e6, g, 1e-4)
    assert math.isfinite(r["mass_rate_kg_s"]) and r["mass_rate_kg_s"] > 0
    assert abs(r["nozzle_energy_residual_j_kg"]) < 1e-8
    fd = (
        ideal_gas_nozzle_flow(2e6, 300, 1.6e6 + 2, g, 1e-4)["mass_rate_kg_s"]
        - ideal_gas_nozzle_flow(2e6, 300, 1.6e6 - 2, g, 1e-4)["mass_rate_kg_s"]
    ) / 4
    assert r["mass_derivatives"]["back_pressure_pa"] == pytest.approx(fd, rel=1e-7)
