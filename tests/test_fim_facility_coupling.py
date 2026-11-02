"""Actual reservoir response to facility bottlenecks, replay and rollback."""

import copy
import json
import pickle
import sys

import pytest
import torch
from test_fim_blackoil_surface import ROOT, config, report_rates, solver

sys.path.insert(0, str(ROOT / "scripts"))
from simulator.facility_coupling import FacilityCoupledFIM
from surface import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    SurfaceFacility,
)


def facility_spec():
    c = json.loads((ROOT / "configs/surface_gas_valve_facility.json").read_text())
    network = config()["simulation"]["physical_surface_network"]["network"]
    c["pvt"] = copy.deepcopy(network["pvt"])
    c["stock_densities"] = copy.deepcopy(network["stock_densities"])
    c["separator_stages"] = [
        {"name": "high_pressure", "pressure_pa": 1e6},
        {"name": "stock_tank", "pressure_pa": 1e5},
    ]
    c["gas_export"]["export_valve"]["throat_area_m2"] = 2e-7
    return c


def build(limited=True, device="cpu"):
    c = facility_spec()
    if not limited:
        c["gas_export"]["export_valve"]["throat_area_m2"] = 2e-5
    if device == "cpu":
        s = solver()
    else:
        from test_fim_surface_cuda import build as device_solver

        s = device_solver(device)
    return FacilityCoupledFIM(
        s,
        SurfaceFacility.from_dict(c),
        {"min_fraction": 0.01, "scan_points": 3, "fraction_tolerance": 1e-3, "max_evaluations": 30},
    )


def masses(s):
    return s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)


def test_facility_limit_changes_actual_reservoir_and_conserves_components():
    free, limited = build(False), build()
    old = masses(limited.solver)
    caps = [w.control_value for w in limited.solver.wells.wells]
    assert free.step(86400) and limited.step(86400)
    s = limited.solver
    r = limited.report
    assert r["fraction"] < 1 and free.report["fraction"] == 1
    assert report_rates(s).sum() < report_rates(free.solver).sum()
    assert s.fluid.pressure.mean() > free.solver.fluid.pressure.mean()
    assert (old - masses(s) - report_rates(s).sum(0) * 86400).abs().max() < 0.01
    feed = r["facility"]["feed"]
    rho = limited.facility.densities
    from surface import ComponentStream

    actual = torch.tensor(
        list(ComponentStream(**feed).mass_rates(rho).values()), dtype=torch.float64
    )
    assert torch.allclose(actual, report_rates(s).sum(0), rtol=1e-12, atol=1e-14)
    assert r["facility"]["gas_export"]["valve"]["utilization"] <= 1
    assert [w.control_value for w in s.wells.wells] == caps
    # Replay must not add production from every rejected trial to cumulatives.
    direct = solver()
    for w, cap in zip(direct.wells.wells, r["applied_rate_caps_m3_day"]):
        w.control_value = cap
    assert direct.step(86400)
    assert torch.equal(s.fluid.pressure, direct.fluid.pressure)
    for a, b in zip(s.wells.wells, direct.wells.wells):
        for name, value in a.__dict__.items():
            if name.startswith("cum_"):
                assert value == b.__dict__[name]


def test_failure_restores_state_caps_report_and_restart(monkeypatch):
    c = build()
    assert c.step(86400)
    restored = pickle.loads(pickle.dumps(c))
    assert c.step(86400) and restored.step(86400)
    assert torch.equal(c.solver.fluid.pressure, restored.solver.fluid.pressure)
    old = c.solver.fluid.pressure.clone()
    previous = copy.deepcopy(c.report)
    values = [w.control_value for w in c.solver.wells.wells]

    def fail(*args, **kwargs):
        raise HydraulicConvergenceError("network numerical failure")

    monkeypatch.setattr(SurfaceFacility, "run", fail)
    assert not c.step(86400)
    assert torch.equal(c.solver.fluid.pressure, old) and c.report == previous
    assert [w.control_value for w in c.solver.wells.wells] == values


def test_infeasible_property_and_feed_scaling_are_not_successes():
    c = build()
    old = c.solver.fluid.pressure.clone()
    c.controls = {"gas_valve_max_opening": 0}
    with pytest.raises(EquipmentOperatingError, match="No sampled"):
        c.step(86400)
    assert torch.equal(c.solver.fluid.pressure, old) and c.report is None
    c.controls = {"feed_scales": [0.1, 0.1]}
    with pytest.raises(ValueError, match="scaling"):
        c.step(86400)
    assert torch.equal(c.solver.fluid.pressure, old)
    spec = facility_spec()
    spec["separator_stages"][0]["pressure_pa"] = 1.2e6
    with pytest.raises(PropertyModelError, match="receiver"):
        FacilityCoupledFIM(solver(), SurfaceFacility.from_dict(spec))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_actual_capacity_feedback_cpu_cuda():
    a, b = build(), build(device="cuda")
    assert a.step(86400) and b.step(86400)
    assert a.report["fraction"] == b.report["fraction"]
    assert torch.allclose(
        a.solver.fluid.pressure, b.solver.fluid.pressure.cpu(), rtol=1e-9, atol=0.001
    )
    assert torch.allclose(report_rates(a.solver), report_rates(b.solver), rtol=1e-7, atol=1e-9)
    assert b.solver._graphs is not None


def test_simulator_uses_coupled_stepper_and_pickles_one_shared_solver():
    from simulator.fluid import Fluid
    from simulator.reservoir import Reservoir
    from simulator.simulation import Simulator
    from simulator.well import WellManager

    c = config()
    c["simulation"]["coupled_facility"] = {
        "facility": facility_spec(),
        "control": {
            "min_fraction": 0.01,
            "scan_points": 3,
            "fraction_tolerance": 1e-3,
            "max_evaluations": 30,
        },
    }
    device = torch.device("cpu")
    r = Reservoir(c["reservoir"], device=device)
    f = Fluid(c["fluid"], r, device=device)
    sim = Simulator(r, f, WellManager(c["wells"], r), c["simulation"], device=device)
    assert sim.run_step(86400)
    assert sim.facility_coupling.report["fraction"] < 1
    saved = pickle.loads(pickle.dumps(sim))
    assert saved.facility_coupling.solver is saved.fim_solver
    assert sim.run_step(86400) and saved.run_step(86400)
    assert sim.current_time == saved.current_time == 172800
    assert torch.equal(sim.fluid.pressure, saved.fluid.pressure)


def test_integrated_material_account_capacity_change_and_exact_restart():
    c = build(False)
    initial = masses(c.solver).tolist()
    assert c.step(86400)
    assert c.report["fraction"] == 1
    previous_produced = c.report["material_account"]["produced_kg"].copy()
    c.controls = {"gas_valve_max_opening": 0.01}
    saved = pickle.loads(pickle.dumps(c))
    assert c.step(43200) and saved.step(43200)
    assert c.report == saved.report
    assert c.report["fraction"] < 1
    account = c.report["material_account"]
    assert account["elapsed_s"] == 129600
    assert list(account["initial_inventory_kg"].values()) == initial
    step = account["step"]
    rates = report_rates(c.solver).sum(0).tolist()
    for i, component in enumerate(("oil", "water", "gas")):
        assert step["produced_kg"][component] == pytest.approx(rates[i] * 43200, rel=1e-12)
        assert account["produced_kg"][component] == pytest.approx(
            previous_produced[component] + step["produced_kg"][component], rel=1e-12
        )
        assert sum(p[component] for p in account["products_kg"].values()) == pytest.approx(
            account["produced_kg"][component], rel=1e-12, abs=1e-12
        )
        assert abs(account["end_to_end_residual_kg"][component]) < 0.02
    c.controls = {}
    assert c.step(21600)
    assert c.report["fraction"] == 1
    assert c.report["material_account"]["elapsed_s"] == 151200


def test_material_account_rejects_corrupted_products_and_rolls_back(monkeypatch):
    c = build(False)
    assert c.step(86400)
    previous = copy.deepcopy(c.report)
    inventory = masses(c.solver).clone()
    original = SurfaceFacility.run

    def corrupt(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["products"]["oil"]["oil_m3_s"] *= 1.01
        return result

    monkeypatch.setattr(SurfaceFacility, "run", corrupt)
    assert not c.step(86400)
    assert "products" in c.last_attempt["reason"]
    assert c.report == previous
    assert torch.equal(masses(c.solver), inventory)
