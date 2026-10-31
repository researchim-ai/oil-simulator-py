"""End-to-end gathering/flash/process accounting and actual pressure constraints."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from run_surface_gathering import load_gathering

from surface import FacilityCapacityControl, PropertyModelError
from surface.facility import FacilityLimit
from surface.process import ComponentStream


def config():
    return json.loads((ROOT / "configs/surface_gathering_facility.json").read_text())


def test_full_handoff_and_restart_guess():
    f = load_gathering(config())
    r = f.run()
    assert r["feasible"]
    assert r["gathering_network"]["receiving_stream"] == pytest.approx(r["feed"])
    assert max(abs(v) for v in r["component_mass_residuals_kg_s"].values()) < 1e-15
    assert abs(r["gathering_network"]["mechanical_balance_error_w"]) < 1e-5
    assert r["gas_export"]["mass_rate_kg_s"] == pytest.approx(0.072)
    assert r["water_handoff"]["mass_rate_kg_s"] == pytest.approx(2.0)
    rr = f.run(initial=f.network_initial(r))
    assert (
        rr["gathering_network"]["node_pressures_pa"] == r["gathering_network"]["node_pressures_pa"]
    )
    assert rr["feed"] == r["feed"]
    assert rr["gas_export"]["shaft_power_w"] == r["gas_export"]["shaft_power_w"]


def test_gathering_pressure_limit_drives_capacity_control():
    f = load_gathering(config())
    f.limits = (*f.limits, FacilityLimit("A_rating", "gathering_pressure:A", "Pa", maximum=1.3e6))
    requested = f.run()
    assert not requested["feasible"]
    r = FacilityCapacityControl(scan_points=5, max_evaluations=40, fraction_tolerance=1e-4).solve(f)
    assert r["status"] == "limited" and 0 < r["fraction"] < 1
    accepted = r["facility"]
    assert accepted["feasible"]
    assert accepted["gathering_network"]["node_pressures_pa"]["A"] <= 1.3e6
    assert accepted["gathering_network"]["node_pressures_pa"]["A"] > 1.299e6
    expected = ComponentStream.mix(s.split(r["fraction"])[0] for s in f.feeds)
    assert accepted["feed"] == pytest.approx(expected.__dict__)
    assert accepted["gas_export"]["shaft_power_w"] < requested["gas_export"]["shaft_power_w"]


def test_incompatible_pvt_pressures_and_bindings_fail_explicitly():
    cfg = config()
    cfg["gathering"]["nodes"][-1]["pressure_pa"] = 1.1e6
    with pytest.raises(PropertyModelError, match="pressure"):
        load_gathering(cfg)
    cfg = config()
    cfg["gathering"]["pvt"]["Bg"][1] *= 1.01
    with pytest.raises(PropertyModelError, match="PVT"):
        load_gathering(cfg)
    cfg = config()
    cfg["feed_nodes"] = ["A"]
    with pytest.raises(ValueError, match="Every"):
        load_gathering(cfg)
    f = load_gathering(config())
    with pytest.raises(ValueError, match="scale"):
        f.run(feed_scales=[1, 2])


def test_schedule_restart_and_gathering_fingerprint():
    from surface.operation import FacilityPeriod, run_facility_schedule

    cfg = config()
    plant = load_gathering(cfg)
    periods = [
        FacilityPeriod(**p)
        for p in json.loads((ROOT / "configs/surface_gathering_schedule.json").read_text())[
            "periods"
        ]
    ]
    full = run_facility_schedule(plant, periods)
    part = run_facility_schedule(plant, periods, stop_after=2)
    resumed = run_facility_schedule(plant, periods, checkpoint=part["checkpoint"])
    assert full["checkpoint"] == resumed["checkpoint"]
    assert full["totals"] == resumed["totals"]
    assert full["totals"]["energies_j"]["gathering_friction_dissipation_j"] > 0
    assert full["checkpoint"]["ledger"][2]["facility"] is None
    changed = config()
    changed["gathering"]["pipes"][0]["diameter_m"] *= 1.01
    with pytest.raises(ValueError, match="model or operating plan"):
        run_facility_schedule(load_gathering(changed), periods, checkpoint=part["checkpoint"])


@pytest.mark.parametrize("maximum,exit_code", [(2000000, 0), (1300000, 3)])
def test_cli_reports_real_pressure_limit_status(tmp_path, maximum, exit_code):
    import os
    import subprocess

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    output = tmp_path / "gathering.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_surface_gathering.py"),
            str(ROOT / "configs/surface_gathering_facility.json"),
            "--max-pressure",
            "A",
            str(maximum),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    data = json.loads(output.read_text())
    assert data["feasible"] == (exit_code == 0)
    assert data["gathering_network"]["node_pressures_pa"]["A"] > 1.39e6


def test_cli_rejects_unknown_pressure_node(tmp_path):
    import os
    import subprocess

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    output = tmp_path / "invalid.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_surface_gathering.py"),
            str(ROOT / "configs/surface_gathering_facility.json"),
            "--max-pressure",
            "missing",
            "1300000",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Unknown gathering pressure-limit node" in result.stderr
    assert not output.exists()


def test_dispatch_recognizes_subcritical_flow_capacity():
    # Synthetic mathematical stress case with wide declared PVT coverage;
    # it is not a pipe design or an experimental multiphase benchmark.
    cfg = config()
    for feed in cfg["facility"]["feeds"]:
        feed["gas_m3_s"] *= 20
    for node in cfg["gathering"]["nodes"]:
        node["elevation_m"] = 0
    for pipe in cfg["gathering"]["pipes"]:
        pipe.update(length_m=0.1, diameter_m=0.01)
    for limit in cfg["facility"]["limits"]:
        if limit["unit"] == "W":
            limit["maximum"] = 1e6
        if limit["metric"].startswith("separator:"):
            limit["maximum"] = 1
    for part in (cfg["gathering"], cfg["facility"]):
        pvt = part["pvt"]
        pvt.pop("extended", None)
        pvt["pressure_MPa"].append(100)
        for key in ("Bo", "Bw", "mu_o_cP", "mu_w_cP", "mu_g_cP", "Rs_m3m3", "Rv_m3m3"):
            pvt[key].append(pvt[key][-1])
        pvt["Bg"].append(0.001)
    cfg["limits"] = []
    r = FacilityCapacityControl(scan_points=7, max_evaluations=60, min_fraction=0.01).solve(
        load_gathering(cfg)
    )
    assert r["status"] == "limited"
    assert 0 < r["fraction"] < 1
    assert r["facility"]["feasible"]
    assert any(
        "subcritical" in reason.get("physical_domain", "") for reason in r["limiting_constraints"]
    )
