"""Shared utility envelopes, extensive replay and scheduled command integrity."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    EquipmentOperatingError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    ThermalResourceLimits,
    ThermalStream,
    run_thermal_capacity,
    run_thermal_flowsheet,
    run_thermal_schedule,
)


def case():
    root = Path(__file__).resolve().parents[1]
    d = json.loads((root / "configs/surface_thermal_process.json").read_text())
    f = PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in d["fluid"]["components"]),
        d["fluid"]["binary_interactions"],
    )
    feed = {
        "pressure_pa": 1e6,
        "temperature_k": 300,
        "molar_rate_mol_s": 2,
        "mole_fractions": [0.5, 0.5],
    }
    units = [
        {
            "name": "heat",
            "type": "heat_exchanger",
            "inputs": ["feed"],
            "outputs": ["hot"],
            "parameters": {"outlet_temperature_k": 360, "utility_temperature_k": 400},
        },
        {
            "name": "cool",
            "type": "heat_exchanger",
            "inputs": ["hot"],
            "outputs": ["product"],
            "parameters": {"outlet_temperature_k": 320, "utility_temperature_k": 290},
        },
    ]
    return f, feed, units


def test_shared_heating_and_cooling_use_gross_duties_and_replay():
    f, feed, units = case()
    feeds = {"feed": ThermalStream.from_pt(f, **feed)}
    potential = run_thermal_flowsheet(f, feeds, units)
    r = run_thermal_capacity(f, feeds, units, ThermalResourceLimits(max_heating_w=1000))
    control = r["capacity_control"]
    fraction = control["feed_fraction"]
    assert fraction == pytest.approx(1000 / potential["units"][0]["heat_duty_w"], rel=1e-9)
    assert control["resources"]["heating_w"]["actual"] == pytest.approx(1000, abs=1e-6)
    assert control["resources"]["cooling_w"]["actual"] > 0
    assert control["resources"]["heating_w"]["actual"] > r["balances"]["heat_duty_w"]
    selected = {**feed, "molar_rate_mol_s": feed["molar_rate_mol_s"] * fraction}
    direct = run_thermal_flowsheet(f, {"feed": ThermalStream.from_pt(f, **selected)}, units)
    assert direct["product_states"] == r["product_states"]
    assert direct["balances"] == r["balances"]
    outgoing = r["streams"]["product"]["component_molar_rates_mol_s"]
    leftover = control["unprocessed_component_molar_rates_mol_s"]
    assert [a + b for a, b in zip(outgoing, leftover)] == pytest.approx([1, 1], abs=1e-10)


def test_equipment_limit_and_zero_shared_budget():
    f, feed, units = case()
    feeds = {"feed": ThermalStream.from_pt(f, **feed)}
    units[0]["parameters"]["max_abs_duty_w"] = 500
    r = run_thermal_capacity(f, feeds, units, ThermalResourceLimits())
    assert r["units"][0]["heat_duty_w"] == pytest.approx(500, abs=1e-6)
    assert r["capacity_control"]["active_constraints"] == ["equipment:heat:max_abs_duty_w"]
    closed = run_thermal_capacity(f, feeds, units, ThermalResourceLimits(max_cooling_w=0))
    assert closed["capacity_control"]["feed_fraction"] == 0
    assert closed["product_states"]["product"] is None
    assert closed["capacity_control"]["unprocessed_component_molar_rates_mol_s"] == [1, 1]


def test_physical_and_numeric_failures_are_not_capacity_fallbacks(monkeypatch):
    f, feed, units = case()
    feeds = {"feed": ThermalStream.from_pt(f, **feed)}
    units[0]["parameters"]["utility_temperature_k"] = 330
    with pytest.raises(EquipmentOperatingError):
        run_thermal_capacity(f, feeds, units, ThermalResourceLimits(max_heating_w=1))

    def failed(*args, **kwargs):
        raise RuntimeError("numerical failure")

    monkeypatch.setattr(f.__class__, "flash_ph", failed)
    with pytest.raises(RuntimeError, match="numerical failure"):
        run_thermal_capacity(f, feeds, units, ThermalResourceLimits(max_heating_w=1))


def test_period_commands_resources_and_restart_preserve_requested_mass():
    f, feed, units = case()
    periods = [
        {
            "duration_s": 10,
            "feeds": {"feed": feed},
            "unit_controls": {"heat": {"outlet_temperature_k": 350}},
        },
        {"duration_s": 20, "feeds": {"feed": feed}, "resource_limits": {"max_heating_w": 0}},
    ]
    saved = copy.deepcopy((periods, units))
    limits = ThermalResourceLimits(max_heating_w=1000)
    full = run_thermal_schedule(f, units, periods, resource_limits=limits)
    partial = run_thermal_schedule(f, units, periods, stop_after=1, resource_limits=limits)
    resumed = run_thermal_schedule(
        f, units, periods, checkpoint=json.loads(json.dumps(partial)), resource_limits=limits
    )
    assert full == resumed and (periods, units) == saved
    assert full["records"][1]["result"]["capacity_control"]["feed_fraction"] == 0
    assert [
        a + b for a, b in zip(full["product_component_moles"], full["unprocessed_component_moles"])
    ] == pytest.approx([30, 30], abs=1e-9)
    with pytest.raises(ValueError, match="fingerprint"):
        run_thermal_schedule(
            f,
            units,
            periods,
            checkpoint=partial,
            resource_limits=ThermalResourceLimits(max_heating_w=999),
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"missing": {"fraction": 0.5}},
        {"heat": {"inputs": ["x"]}},
        {"heat": {"outlet_temperature_k": 900}},
    ],
)
def test_bad_period_commands_rejected_before_shutdown(patch):
    f, feed, units = case()
    feed["molar_rate_mol_s"] = 0
    with pytest.raises(ValueError):
        run_thermal_schedule(
            f,
            units,
            [{"duration_s": 10, "feeds": {"feed": feed}, "unit_controls": patch}],
            stop_after=0,
        )


def test_shared_shaft_budget_and_record_component_audit():
    from surface.operation import _digest

    f, feed, _ = case()
    units = [
        {
            "name": "booster",
            "type": "pressure_raiser",
            "inputs": ["feed"],
            "outputs": ["product"],
            "parameters": {"outlet_pressure_pa": 2e6, "isentropic_efficiency": 0.8},
        }
    ]
    periods = [{"duration_s": 10, "feeds": {"feed": feed}}]
    limits = ThermalResourceLimits(max_shaft_power_w=1000)
    checkpoint = run_thermal_schedule(f, units, periods, resource_limits=limits)
    control = checkpoint["records"][0]["result"]["capacity_control"]
    assert control["resources"]["shaft_power_w"]["actual"] == pytest.approx(1000, abs=1e-6)
    assert checkpoint["requested_component_moles"] == [10, 10]
    corrupt = copy.deepcopy(checkpoint)
    corrupt["records"][0]["product_component_moles"][0] += 1
    corrupt["product_component_moles"][0] += 1
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ValueError, match="component accounting"):
        run_thermal_schedule(f, units, periods, checkpoint=corrupt, resource_limits=limits)
