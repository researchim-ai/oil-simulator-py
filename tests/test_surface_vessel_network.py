"""Monolithic transfer, boundary conservation, upwinding and checked restart."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicVessel,
    DynamicVesselNetwork,
    HydraulicConvergenceError,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PropertyModelError,
    VesselConnection,
    run_network_schedule,
)


def setup():
    d = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    f = PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in d["components"]), d["binary_interactions"]
    )
    v = {name: DynamicVessel(name, 1, 1, 1e7) for name in ("a", "b")}
    s = {
        "a": v["a"].initialize(f, 200, 300, [0.3, 0.7]),
        "b": v["b"].initialize(f, 100, 300, [0.7, 0.3]),
    }
    c = VesselConnection("line", "a", "b", 1e-5, 500, 0.01)
    return f, v, s, c


def test_transfer_uses_new_pressure_and_upstream_composition_without_lag():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,))
    r = net.advance(s, f, 1, initial_dt_s=1)
    step = r["steps"][0]
    flow = step["connections"][0]
    a, b = r["states"]["a"], r["states"]["b"]
    assert flow["molar_rate_mol_s"] == pytest.approx(
        c.molar_conductance_mol_s_pa * (a["pressure_pa"] - b["pressure_pa"]), abs=1e-12
    )
    assert flow["molar_rate_mol_s"] < 0.001
    assert np.array(a["component_moles"]) + flow["component_rates_mol_s"] == pytest.approx(
        s["a"].component_moles, abs=1e-9
    )
    assert np.array(b["component_moles"]) - flow["component_rates_mol_s"] == pytest.approx(
        s["b"].component_moles, abs=1e-9
    )
    assert a["internal_energy_j"] - s["a"].internal_energy_j == pytest.approx(
        -flow["enthalpy_rate_w"], abs=1e-7
    )
    assert b["internal_energy_j"] - s["b"].internal_energy_j == pytest.approx(
        flow["enthalpy_rate_w"], abs=1e-7
    )
    assert max(abs(x) for x in step["component_residuals_mol"]) < 1e-9
    assert abs(step["energy_residual_j"]) < 1e-7


def test_reverse_flow_is_upwind_and_check_connection_stays_closed():
    f, v, s, c = setup()
    c = replace(c, source="b", target="a")
    closed = DynamicVesselNetwork(v, (c,)).advance(s, f, 0.5, initial_dt_s=0.5)
    assert closed["steps"][0]["connections"][0]["molar_rate_mol_s"] == 0
    assert closed["states"]["a"]["pressure_pa"] == pytest.approx(200, abs=1e-8)
    reverse = DynamicVesselNetwork(v, (replace(c, allow_reverse=True),)).advance(
        s, f, 0.5, initial_dt_s=0.5
    )
    link = reverse["steps"][0]["connections"][0]
    assert link["molar_rate_mol_s"] < 0 and link["source"] == "a" and link["target"] == "b"
    assert link["component_rates_mol_s"][1] > link["component_rates_mol_s"][0]


def test_cycle_boundary_conservation_and_order_independence():
    f, v, s, c = setup()
    v["c"] = DynamicVessel("c", 1, 1, 1e7)
    s["c"] = v["c"].initialize(f, 50, 300, [0.5, 0.5])
    connections = [
        replace(c, molar_conductance_mol_s_pa=1e-6),
        VesselConnection("bc", "b", "c", 1e-6, 500, 0.01),
        VesselConnection("ca", "c", "a", 1e-6, 500, 0.01, allow_reverse=True),
    ]
    a = DynamicVesselNetwork(v, connections).advance(s, f, 0.5, initial_dt_s=0.5)
    b = DynamicVesselNetwork(dict(reversed(list(v.items()))), connections[::-1]).advance(
        s, f, 0.5, initial_dt_s=0.5
    )
    for name in v:
        assert a["states"][name]["component_moles"] == pytest.approx(
            b["states"][name]["component_moles"], abs=1e-10
        )
        assert a["states"][name]["pressure_pa"] == pytest.approx(
            b["states"][name]["pressure_pa"], abs=1e-7
        )
    assert np.sum([q["component_moles"] for q in a["states"].values()], axis=0) == pytest.approx(
        np.sum([q.component_moles for q in s.values()], axis=0), abs=1e-9
    )


def test_isolated_node_agrees_with_single_vessel():
    f, v, s, _ = setup()
    node = v["a"]
    state = s["a"]
    expected = node.advance(state, f, 1, heat_w=0.1, utility_temperature_k=400, initial_dt_s=0.5)
    actual = DynamicVesselNetwork({"a": node}).advance(
        {"a": state},
        f,
        1,
        heat={"a": {"heat_w": 0.1, "utility_temperature_k": 400}},
        initial_dt_s=0.5,
    )
    for field in ("pressure_pa", "temperature_k", "internal_energy_j"):
        assert actual["states"]["a"][field] == pytest.approx(
            expected["state"][field], rel=1e-9, abs=1e-8
        )


def test_two_phase_transfer_carries_selected_vapor_components():
    f, _, _, _ = setup()
    v = {n: DynamicVessel(n, 0.1, 1e5, 1e7) for n in ("a", "b")}
    s = {
        "a": v["a"].initialize(f, 3e6, 220, [0.5, 0.5]),
        "b": v["b"].initialize(f, 2.9e6, 220, [0.5, 0.5]),
    }
    c = VesselConnection("vapor", "a", "b", 1e-7, 2e5, 0.1, source_port="phase_1")
    r = DynamicVesselNetwork(v, (c,)).advance(s, f, 0.1, initial_dt_s=0.1)
    flow = r["steps"][0]["connections"][0]
    assert flow["component_rates_mol_s"][0] / sum(flow["component_rates_mol_s"]) > 0.7
    assert r["states"]["b"]["component_moles"][0] / sum(r["states"]["b"]["component_moles"]) > 0.5
    assert abs(r["steps"][0]["energy_residual_j"]) < 1e-4


def test_calibration_and_numerical_failures_do_not_mutate_states(monkeypatch):
    f, v, s, c = setup()
    saved = {n: q.to_dict() for n, q in s.items()}
    with pytest.raises(PropertyModelError, match="calibrated"):
        DynamicVesselNetwork(v, (replace(c, max_pressure_difference_pa=10),)).advance(s, f, 1)

    def fail(*args, **kwargs):
        raise HydraulicConvergenceError("forced numerical failure")

    monkeypatch.setattr(DynamicVesselNetwork, "_step", fail)
    with pytest.raises(HydraulicConvergenceError, match="minimum dt"):
        DynamicVesselNetwork(v, (c,)).advance(s, f, 0.1, initial_dt_s=0.1, minimum_dt_s=0.05)
    assert {n: q.to_dict() for n, q in s.items()} == saved


def test_adaptive_network_step_covers_full_interval():
    f, v, s, c = setup()
    v = {n: replace(q, max_fractional_inventory_change=0.001) for n, q in v.items()}
    r = DynamicVesselNetwork(v, (c,)).advance(s, f, 0.2, initial_dt_s=0.2)
    assert r["failed_steps"] > 0
    assert sum(x["dt_s"] for x in r["steps"]) == pytest.approx(0.2)
    assert all(x["time_s"] == 0.2 for x in r["states"].values())


def test_network_schedule_json_restart_and_only_external_accounting():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,))
    periods = [
        {"duration_s": 0.5},
        {"duration_s": 0.5, "heat": {"b": {"heat_w": 0.1, "utility_temperature_k": 400}}},
        {"duration_s": 0.5, "withdrawals": {"b": {"bulk": 0.0001}}},
    ]
    full = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5})
    partial = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5}, stop_after=1)
    saved = copy.deepcopy(partial)
    resumed = run_network_schedule(
        f,
        net,
        s,
        periods,
        stepper={"initial_dt_s": 0.5},
        checkpoint=json.loads(json.dumps(partial)),
    )
    assert full == resumed and partial == saved
    assert full["totals"]["inlet_component_moles"] == [0, 0]
    assert sum(full["totals"]["outlet_component_moles"]) == pytest.approx(0.00005)
    assert full["totals"]["heat_j"] == pytest.approx(0.05)
    corrupt = copy.deepcopy(partial)
    corrupt["states"]["b"]["internal_energy_j"] += 1
    with pytest.raises(ValueError, match="checksum"):
        run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5}, checkpoint=corrupt)
    bad = [{"duration_s": 1, "heat": {"missing": {"heat_w": 1}}}]
    with pytest.raises(ValueError, match="command"):
        run_network_schedule(f, net, s, bad, stop_after=0)


def test_connection_factor_commands_close_and_reopen_without_mutating_network():
    f, v, s, c = setup()
    net = DynamicVesselNetwork(v, (c,))
    closed = net.advance(s, f, 0.5, connection_factors={"line": 0}, initial_dt_s=0.5)
    assert closed["steps"][0]["connections"][0]["molar_rate_mol_s"] == 0
    periods = [
        {"duration_s": 0.5, "connection_factors": {"line": 0}},
        {"duration_s": 0.5, "connection_factors": {"line": 0.5}},
    ]
    r = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.5})
    assert r["records"][0]["steps"][0]["connections"][0]["molar_rate_mol_s"] == 0
    assert r["records"][1]["steps"][0]["connections"][0]["molar_rate_mol_s"] > 0
    assert c.molar_conductance_mol_s_pa == 1e-5
    with pytest.raises(ValueError, match="factor"):
        run_network_schedule(
            f, net, s, [{"duration_s": 1, "connection_factors": {"line": 2}}], stop_after=0
        )
