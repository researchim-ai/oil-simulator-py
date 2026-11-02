"""Physical process balances, explicit domains and restart transaction checks."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    EquipmentOperatingError,
    MixtureFlashSeparator,
    MixtureHeatExchanger,
    MixturePressureRaiser,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PropertyModelError,
    ThermalStream,
    let_down_thermal_stream,
    mix_thermal_streams,
    run_thermal_flowsheet,
    run_thermal_schedule,
)
from surface.operation import _digest
from surface.peng_robinson import R

ROOT = Path(__file__).resolve().parents[1]


def data():
    return json.loads((ROOT / "configs/surface_thermal_process.json").read_text())


def fluid():
    d = data()["fluid"]
    return PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in d["components"]), d["binary_interactions"]
    )


def test_stream_model_order_and_reference_are_checked():
    f = fluid()
    s = ThermalStream.from_pt(f, 1e6, 300, 2, [0.4, 0.6])
    for changed in [
        PengRobinsonMixture(f.components[::-1], f.binary_interactions),
        PengRobinsonMixture(f.components, ((0, 0.01), (0.01, 0))),
        PengRobinsonMixture(
            tuple(replace(c, reference_temperature_k=310) for c in f.components),
            f.binary_interactions,
        ),
    ]:
        with pytest.raises(PropertyModelError, match="mismatch"):
            s.state(changed)
    restored = ThermalStream(**json.loads(json.dumps(s.to_dict())))
    assert restored == s
    assert sum(s.component_mass_rates(f)) > 0
    with pytest.raises(PropertyModelError):
        ThermalStream((1, 0), 1e6, 0, s.model_id)


def test_adiabatic_mixing_conserves_components_and_enthalpy():
    f = fluid()
    a = ThermalStream.from_pt(f, 1e6, 280, 2, [0.3, 0.7])
    b = ThermalStream.from_pt(f, 1e6, 330, 3, [0.7, 0.3])
    r = mix_thermal_streams([a, b], f)
    s = r["outputs"][0]
    assert s.mole_fractions == pytest.approx([0.54, 0.46])
    assert s.enthalpy_rate_w == pytest.approx(a.enthalpy_rate_w + b.enthalpy_rate_w)
    assert r["entropy_generation_w_k"] > 0
    assert 280 < s.state(f)["temperature_k"] < 330
    split = s.split(0.3)
    assert sum(x.enthalpy_rate_w for x in split) == pytest.approx(s.enthalpy_rate_w)
    assert mix_thermal_streams(split, f)["outputs"][0].mole_fractions == pytest.approx(
        s.mole_fractions
    )
    with pytest.raises(PropertyModelError, match="equal"):
        mix_thermal_streams([a, replace(b, pressure_pa=2e6)], f)


def test_cooling_condensation_and_equilibrium_separation():
    f = fluid()
    feed = ThermalStream.from_pt(f, 3e6, 260, 10, [0.5, 0.5])
    cooling = MixtureHeatExchanger("cold", 220, 180, 1e5).run(feed, f)
    cold = cooling["outputs"][0]
    assert cooling["heat_duty_w"] < 0 and cold.state(f)["phase_count"] == 2
    r = MixtureFlashSeparator("separator", 2.8e6).run(cold, f)
    assert r["phase_count"] == 2 and r["phase_roles"] == ["liquid", "vapor"]
    out = r["outputs"]
    assert np.sum([s.component_molar_rates_mol_s for s in out], axis=0) == pytest.approx(
        feed.component_molar_rates_mol_s, abs=1e-8
    )
    assert abs(r["energy_residual_w"]) < 1e-6
    assert all(s.state(f)["phase_count"] == 1 for s in out)
    assert out[1].mole_fractions[0] > out[0].mole_fractions[0]
    with pytest.raises(EquipmentOperatingError):
        MixtureHeatExchanger("impossible", 220, 250).run(feed, f)
    with pytest.raises(EquipmentOperatingError):
        MixtureHeatExchanger("small", 220, 180, max_abs_duty_w=1).run(feed, f)


def test_single_phase_separator_requires_explicit_routing():
    f = fluid()
    s = ThermalStream.from_pt(f, 1e6, 300, 2, [0.5, 0.5])
    with pytest.raises(PropertyModelError, match="single_phase_outlet"):
        MixtureFlashSeparator("ambiguous", 1e6).run(s, f)
    r = MixtureFlashSeparator("route", 1e6, single_phase_outlet=1).run(s, f)
    assert r["outputs"][0].molar_rate_mol_s == 0
    assert r["phase_roles"] == ["absent", "single"]
    assert r["outputs"][1].molar_rate_mol_s == 2


def test_pressure_raiser_independent_ideal_gas_limit_and_limits():
    f = fluid()
    z = [0.3, 0.7]
    n = 2.0
    t = 300.0
    s = ThermalStream.from_pt(f, 100, t, n, z)
    eta = 0.8
    r = MixturePressureRaiser("raise", 200, eta).run(s, f)
    cp = sum(x * c.ideal_cp_j_mol_k for x, c in zip(z, f.components))
    tout = t + (t * 2 ** (R / cp) - t) / eta
    assert r["outputs"][0].state(f)["temperature_k"] == pytest.approx(tout, rel=2e-5)
    assert r["shaft_power_w"] == pytest.approx(n * cp * (tout - t), rel=2e-5)
    assert r["entropy_generation_w_k"] > 0
    for equipment in [
        MixturePressureRaiser("limited", 200, eta, max_power_w=1),
        MixturePressureRaiser("hot", 200, eta, max_outlet_temperature_k=310),
    ]:
        with pytest.raises(EquipmentOperatingError):
            equipment.run(s, f)
    wet = ThermalStream.from_pt(f, 3e6, 220, 2, [0.5, 0.5])
    with pytest.raises(EquipmentOperatingError, match="two-phase inlet"):
        MixturePressureRaiser("wet", 4e6, eta).run(wet, f)
    with pytest.raises(PropertyModelError):
        MixturePressureRaiser("wrong", 50, eta).run(s, f)


def test_shutdown_does_not_invent_thermodynamic_state():
    f = fluid()
    s = ThermalStream.from_pt(f, 3e6, 260, 0, [0.5, 0.5])
    assert s.state(f) is None
    operations = [
        MixtureHeatExchanger("cold", 220, 180),
        MixturePressureRaiser("raise", 4e6, 0.8),
        MixtureFlashSeparator("separator", 1e6),
    ]
    for operation in operations:
        r = operation.run(s, f)
        assert r["heat_duty_w"] == r["shaft_power_w"] == r["entropy_generation_w_k"] == 0
        assert all(q.state(f) is None for q in r["outputs"])
    assert let_down_thermal_stream(s, 1e6, f)["outputs"][0].state(f) is None


def test_flowsheet_prevents_double_consumption_and_unknown_fields():
    f = fluid()
    s = ThermalStream.from_pt(f, 1e6, 300, 2, [0.5, 0.5])
    u = {
        "name": "split",
        "type": "split",
        "inputs": ["feed"],
        "outputs": ["a", "b"],
        "parameters": {"fraction": 0.5},
    }
    second = {
        "name": "twice",
        "type": "mix",
        "inputs": ["feed", "a"],
        "outputs": ["bad"],
        "parameters": {},
    }
    with pytest.raises(ValueError, match="consumed only once"):
        run_thermal_flowsheet(f, {"feed": s}, [u, second])
    u["typo"] = 1
    with pytest.raises(ValueError, match="fields"):
        run_thermal_flowsheet(f, {"feed": s}, [u])


def test_recipe_whole_plant_balance_and_json_restart():
    d = data()
    f = fluid()
    before = copy.deepcopy(d)
    full = run_thermal_schedule(f, d["units"], d["periods"])
    partial = run_thermal_schedule(f, d["units"], d["periods"], stop_after=1)
    serialized = json.loads(json.dumps(partial, allow_nan=False))
    resumed = run_thermal_schedule(f, d["units"], d["periods"], checkpoint=serialized)
    assert resumed == full
    assert d == before and partial == serialized
    assert full["next_period"] == 3 and full["elapsed_s"] == 7800
    expected = [
        sum(
            p["duration_s"]
            * sum(
                feed["molar_rate_mol_s"] * feed["mole_fractions"][i] for feed in p["feeds"].values()
            )
            for p in d["periods"]
        )
        for i in range(2)
    ]
    assert full["product_component_moles"] == pytest.approx(expected, abs=1e-5)
    for record in full["records"]:
        b = record["result"]["balances"]
        assert max(abs(x) for x in b["component_molar_residuals_mol_s"]) < 1e-8
        assert abs(b["energy_residual_w"]) < 1e-5
        assert b["entropy_generation_w_k"] >= -1e-6
    assert all(s is None for s in full["records"][1]["result"]["product_states"].values())


def test_checkpoint_corruption_model_change_and_failure_rollback():
    d = data()
    f = fluid()
    checkpoint = run_thermal_schedule(f, d["units"], d["periods"], stop_after=0)
    pristine = copy.deepcopy(checkpoint)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["elapsed_s"] = 100
    with pytest.raises(ValueError, match="checksum"):
        run_thermal_schedule(f, d["units"], d["periods"], checkpoint=corrupt)
    corrupt = copy.deepcopy(checkpoint)
    corrupt["product_component_moles"] = [1, 0]
    corrupt["checksum"] = _digest({k: v for k, v in corrupt.items() if k != "checksum"})
    with pytest.raises(ValueError, match="accounting"):
        run_thermal_schedule(f, d["units"], d["periods"], checkpoint=corrupt)
    units = copy.deepcopy(d["units"])
    units[2]["parameters"]["max_abs_duty_w"] = 1
    with pytest.raises(ValueError, match="fingerprint"):
        run_thermal_schedule(f, units, d["periods"], checkpoint=checkpoint)
    failing = run_thermal_schedule(f, units, d["periods"], stop_after=0)
    saved = copy.deepcopy(failing)
    with pytest.raises(EquipmentOperatingError):
        run_thermal_schedule(f, units, d["periods"], checkpoint=failing)
    assert failing == saved and checkpoint == pristine


def test_thermal_commands_are_validated_even_when_stopped():
    d = data()
    f = fluid()
    for p in d["periods"]:
        for feed in p["feeds"].values():
            feed["molar_rate_mol_s"] = 0
    d["units"][2]["parameters"]["outlet_temperature_k"] = 900
    with pytest.raises(PropertyModelError, match="Temperature outside"):
        run_thermal_schedule(f, d["units"], d["periods"], stop_after=0)


def test_cli_failure_preserves_existing_checkpoint(tmp_path):
    import os
    import subprocess

    config = tmp_path / "config.json"
    output = tmp_path / "checkpoint.json"
    config.write_text(json.dumps(data()))
    command = [
        sys.executable,
        str(ROOT / "scripts/run_surface_process.py"),
        str(config),
        "--output",
        str(output),
        "--stop-after",
        "0",
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    good = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert good.returncode == 0, good.stderr
    original = output.read_bytes()
    broken = data()
    broken["units"][0]["type"] = "typo"
    config.write_text(json.dumps(broken))
    bad = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30, check=False)
    assert bad.returncode == 2
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_heater_duty_and_entropy_independent_ideal_limit():
    f = fluid()
    z = [0.3, 0.7]
    n = 2
    s = ThermalStream.from_pt(f, 100, 300, n, z)
    r = MixtureHeatExchanger("warm", 350, 400).run(s, f)
    cp = sum(x * c.ideal_cp_j_mol_k for x, c in zip(z, f.components))
    expected_heat = n * cp * 50
    expected_entropy = n * cp * np.log(350 / 300) - expected_heat / 400
    assert r["heat_duty_w"] == pytest.approx(expected_heat, rel=2e-5)
    assert r["entropy_generation_w_k"] == pytest.approx(expected_entropy, rel=2e-5)
