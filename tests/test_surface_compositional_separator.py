"""Monolithic separator dynamics, independent targets and phase transitions."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface.component_energy import ComponentEnergyStream
from surface.compositional_separator import (
    CompositionalSeparator,
    CompositionalSeparatorState,
    EquilibriumOutlet,
)
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
)
from surface.peng_robinson import PengRobinsonFluid, R
from surface.peng_robinson_mixture import PengRobinsonMixture
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def binary(device="cpu"):
    return EquilibriumFluid(
        [PengRobinsonFluid(**c) for c in DATA["components"]], [[0, 0], [0, 0]], device=device
    )


def ternary(device="cpu"):
    return EquilibriumFluid.from_dict(
        json.loads((ROOT / "configs/surface_equilibrium_process.json").read_text())["fluid"],
        device=device,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [320.0, 196.2])
def test_closed_cell_and_exact_bulk_backward_euler_inventory(device, temperature):
    f = binary(device)
    m = CompositionalSeparator(f, 0.001, temporal_error_control=False)
    state = m.initialize(1e6, temperature, [0.5, 0.5])
    old = state.to_dict()
    result = m.advance(state, 0.02)
    inv = result["state"]["inventory"]
    assert inv["component_moles"] == pytest.approx(state.inventory.component_moles, abs=1e-12)
    assert inv["internal_energy_j"] == pytest.approx(state.inventory.internal_energy_j, abs=1e-7)
    assert inv["temperature_k"] == pytest.approx(temperature, abs=1e-9)
    assert inv["time_s"] == 0.02
    assert state.to_dict() == old
    port = EquilibriumOutlet("bulk", 0.99e6, 1e-9)
    m = CompositionalSeparator(f, 0.001, [port], temporal_error_control=False)
    state = m.initialize(1e6, temperature, [0.5, 0.5])
    result = m.advance(state, 0.01)
    inv = result["state"]["inventory"]
    record = result["steps"][0]
    coefficient = (
        1
        + 0.01
        * port.volume_conductance_m3_s_pa
        * (inv["pressure_pa"] - port.back_pressure_pa)
        / m.volume_m3
    )
    assert inv["component_moles"] == pytest.approx(
        np.array(state.inventory.component_moles) / coefficient, rel=1e-11
    )
    assert inv["internal_energy_j"] - state.inventory.internal_energy_j == pytest.approx(
        -0.01 * record["outlet_energy_rate_w"], abs=1e-7
    )
    assert record["entropy_production_j_k"] >= -1e-6
    assert sum(record["phase_volume_fractions"]) == pytest.approx(1.0, abs=1e-10)
    assert (
        m.inspect(CompositionalSeparatorState.from_dict(result["state"]))["phase_count"]
        == record["phase_count"]
    )


@pytest.mark.parametrize("device", DEVICES)
def test_three_real_eos_phases_and_component_selective_withdrawals(device):
    f = ternary(device)
    ports = [
        EquilibriumOutlet("water", 0.99e6, 1e-10, "component_rich", "water"),
        EquilibriumOutlet("oil", 0.99e6, 1e-10, "component_rich", "hexane"),
        EquilibriumOutlet("gas", 0.99e6, 1e-10, "lightest"),
    ]
    m = CompositionalSeparator(f, 0.001, ports, temporal_error_control=False)
    state = m.initialize(1e6, 350, [0.4, 0.3, 0.3])
    result = m.advance(state, 0.01)
    s = result["steps"][0]
    assert s["phase_count"] == 3
    assert (
        s["ports"]["water"]["component_molar_rates_mol_s"][2]
        > s["ports"]["water"]["component_molar_rates_mol_s"][0] * 1000
    )
    assert (
        s["ports"]["oil"]["component_molar_rates_mol_s"][1]
        > s["ports"]["oil"]["component_molar_rates_mol_s"][0] * 10
    )
    assert s["ports"]["gas"]["component_molar_rates_mol_s"][0] > 0
    assert s["ports"]["gas"]["component_molar_rates_mol_s"][2] > 0
    assert max(map(abs, s["component_residuals_mol"])) < 1e-9
    assert abs(s["fluid_energy_residual_j"]) < 1e-5
    assert m.inspect(CompositionalSeparatorState.from_dict(result["state"]))["phase_count"] == 3


@pytest.mark.parametrize("device", DEVICES)
def test_manufactured_phase_appearance_and_disappearance(device):
    f = binary(device)
    reference = PengRobinsonMixture(f.components, f.binary_interactions)
    z = np.array([0.19, 0.81])
    volume = 0.001
    temperature = 196.2
    # Independent CPU PR equilibrium supplies the exact target, not the
    # monolithic solver under test. Bulk discharge has an analytic BE balance.
    a = reference.flash_pt(1.1e6, temperature, z)
    b = reference.flash_pt(1e6, temperature, z)
    vm = lambda q: sum(p["molar_fraction"] * p["molar_volume_m3_mol"] for p in q["phases"])
    na, nb = volume / vm(a), volume / vm(b)
    ua = na * a["molar_enthalpy_j_mol"] - 1.1e6 * volume
    ub = nb * b["molar_enthalpy_j_mol"] - 1e6 * volume
    g = (na / nb - 1) * volume / (1e6 - 0.98e6)
    heat = ub - ua + (na - nb) * b["molar_enthalpy_j_mol"]
    m = CompositionalSeparator(
        f,
        volume,
        [EquilibriumOutlet("out", 0.98e6, g, maximum_pressure_drop_fraction=0.2)],
        maximum_dt_s=1.0,
        temporal_error_control=False,
        maximum_fractional_inventory_change=0.9,
        maximum_temperature_change_k=50.0,
    )
    state = m.initialize(1.1e6, temperature, z)
    result = m.advance(state, 1.0, heat_w=heat, utility_temperature_k=350)
    inv = result["state"]["inventory"]
    assert result["steps"][0]["phase_count"] == 2
    assert result["steps"][0]["phase_branch_attempts"] >= 2
    assert inv["pressure_pa"] == pytest.approx(1e6, abs=0.1)
    assert inv["temperature_k"] == pytest.approx(temperature, abs=1e-6)
    assert inv["component_moles"] == pytest.approx(nb * z, rel=1e-8)
    source = ComponentEnergyStream.from_pt(
        f, ((na - nb) * z).tolist(), 1.2e6, temperature, gravity_m_s2=0.0
    )
    heat = ua - ub - source.total_energy_rate_w
    m = CompositionalSeparator(
        f,
        volume,
        maximum_dt_s=1.0,
        temporal_error_control=False,
        maximum_fractional_inventory_change=0.9,
        maximum_temperature_change_k=50.0,
    )
    result = m.advance(
        m.initialize(1e6, temperature, z),
        1.0,
        feeds=[source],
        heat_w=heat,
        utility_temperature_k=170,
    )
    inv = result["state"]["inventory"]
    assert result["steps"][0]["phase_count"] == 1
    assert result["steps"][0]["phase_branch_attempts"] >= 2
    assert inv["pressure_pa"] == pytest.approx(1.1e6, abs=0.1)
    assert inv["component_moles"] == pytest.approx(na * z, rel=1e-8)


@pytest.mark.parametrize("device", DEVICES)
def test_new_component_enters_from_exact_zero_and_dry_pickup_is_closed(device):
    f = binary(device)
    m = CompositionalSeparator(
        f,
        0.001,
        [EquilibriumOutlet("ethane", 0.99e6, 1e-9, "component_rich", "Ethane")],
        temporal_error_control=False,
    )
    state = m.initialize(1e6, 320, [1.0, 0.0])
    dry = m.advance(state, 0.01)
    assert dry["steps"][0]["ports"]["ethane"]["volume_rate_m3_s"] == 0
    assert dry["state"]["inventory"]["component_moles"][1] == 0
    feed = ComponentEnergyStream.from_pt(f, [0.001, 0.001], 1.02e6, 320, gravity_m_s2=0.0)
    result = m.advance(state, 0.01, feeds=[feed])
    assert result["state"]["inventory"]["component_moles"][1] == pytest.approx(1e-5, abs=1e-12)
    assert result["steps"][0]["ports"]["ethane"]["volume_rate_m3_s"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_real_monolithic_jacobian_and_no_nested_flash_in_newton(device, monkeypatch):
    f = binary(device)
    m = CompositionalSeparator(
        f, 0.001, [EquilibriumOutlet("out", 0.99e6, 1e-9, "lightest")], temporal_error_control=False
    )
    state = m.initialize(1e6, 196.2, [0.5, 0.5])
    original = m.newton.solve
    flash = f.flash_pt
    calls = []

    def forbid(*args, **kwargs):
        raise AssertionError("Nested flash in a monolithic Newton trial")

    def checked(function, x, lower, upper, **kwargs):
        monkeypatch.setattr(f, "flash_pt", forbid)
        try:
            jac = torch.func.jacfwd(function)(x).cpu().numpy()
            columns = []
            for i in range(x.numel()):
                h = 1e-6
                delta = torch.zeros_like(x)
                delta[i] = h
                columns.append(
                    ((function(x + delta) - function(x - delta)) / (2 * h)).cpu().numpy()
                )
            assert jac == pytest.approx(np.array(columns).T, rel=3e-5, abs=3e-7)
            calls.append(x.numel())
            return original(function, x, lower, upper, **kwargs)
        finally:
            monkeypatch.setattr(f, "flash_pt", flash)

    monkeypatch.setattr(m.newton, "solve", checked)
    result = m.advance(state, 0.005)
    assert calls and result["steps"][0]["phase_count"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_wall_backward_euler_refinement_against_independent_ode(device):
    f = binary(device)
    z = np.array([0.5, 0.5])
    V = 0.001
    wall = VesselWall(1000, 4, 2, 300)
    models = [
        CompositionalSeparator(f, V, wall=wall, maximum_dt_s=h, temporal_error_control=False)
        for h in [0.1, 0.05]
    ]
    states = [m.initialize(1e6, 320, z, wall_temperature_k=300) for m in models]
    n = sum(states[0].inventory.component_moles)
    v = V / n
    old = PengRobinsonMixture(f.components, f.binary_interactions)

    def values(t):
        a, _, b, _, _ = old._parameters(t, z)
        p = R * t / (v - b) - a / (v * v + 2 * b * v - b * b)
        return old.state(p, t, z)["molar_enthalpy_j_mol"] - p * v

    def rhs(time, y):
        t, tw = y
        cv = (values(t + 0.0001) - values(t - 0.0001)) / 0.0002
        return [
            wall.fluid_ua_w_k * (tw - t) / (n * cv),
            (wall.ambient_ua_w_k * (wall.ambient_temperature_k - tw) - wall.fluid_ua_w_k * (tw - t))
            / wall.heat_capacity_j_k,
        ]

    reference = solve_ivp(rhs, [0, 0.1], [320, 300], rtol=1e-11, atol=1e-11, method="DOP853").y[
        :, -1
    ]
    errors = []
    for m, state in zip(models, states):
        result = m.advance(state, 0.1)
        inv = result["state"]["inventory"]
        tw = result["state"]["wall_temperature_k"]
        errors.append(abs(inv["temperature_k"] - reference[0]))
        stored = (
            inv["internal_energy_j"]
            - state.inventory.internal_energy_j
            + wall.energy(tw)
            - wall.energy(state.wall_temperature_k)
        )
        ambient = sum(s["dt_s"] * s["ambient_heat_w"] for s in result["steps"])
        assert stored == pytest.approx(ambient, abs=1e-6)
    assert errors[1] < errors[0] * 0.6


@pytest.mark.parametrize("device", DEVICES)
def test_pressure_feedback_exact_interval_guards_and_atomic_failure(device):
    f = binary(device)
    port = EquilibriumOutlet(
        "out", 0.99e6, 1e-9, pressure_setpoint_pa=1e6, opening_bias=0.4, opening_gain_per_pa=1e-5
    )
    m = CompositionalSeparator(f, 0.001, [port], maximum_dt_s=0.01)
    state = m.initialize(1e6, 320, [0.5, 0.5])
    before = state.to_dict()
    result = m.advance(state, 0.025)
    assert result["state"]["inventory"]["time_s"] == 0.025
    assert sum(s["dt_s"] for s in result["steps"]) == pytest.approx(0.025, abs=1e-15)
    for s in result["steps"]:
        assert s["ports"]["out"]["opening"] == pytest.approx(
            max(0, min(1, 0.4 + 1e-5 * (s["pressure_pa"] - 1e6))), abs=1e-12
        )
    stopped = m.advance(state, 0.01, openings={"out": 0})
    assert stopped["state"]["inventory"]["component_moles"] == pytest.approx(
        state.inventory.component_moles, abs=1e-12
    )
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(state, 0.02, max_substeps=1)
    assert state.to_dict() == before
    with pytest.raises(ValueError):
        m.advance(state, 0.01, openings={"missing": 1})
    with pytest.raises(PropertyModelError):
        m.inspect(replace(state, model_id="0" * 64))
    with pytest.raises(ValueError, match="utility"):
        m.advance(state, 0.01, heat_w=1)
    stream = ComponentEnergyStream.from_pt(f, [0.01, 0.01], 0.9e6, 320, gravity_m_s2=0.0)
    with pytest.raises(EquipmentOperatingError, match="pressure"):
        m.advance(state, 0.01, feeds=[stream])


@pytest.mark.parametrize("device", DEVICES)
def test_shared_total_energy_reference_and_configuration_snapshot(device):
    f = binary(device)
    m = CompositionalSeparator(
        f, 0.001, elevation_m=100, gravity_m_s2=9.80665, temporal_error_control=False
    )
    state = m.initialize(1e6, 320, [0.5, 0.5])
    feed = ComponentEnergyStream.from_pt(
        f, [0.001, 0.001], 1.02e6, 320, elevation_m=50, velocity_m_s=2
    )
    result = m.advance(state, 0.01, feeds=[feed])
    final = CompositionalSeparatorState.from_dict(result["state"])
    change = final.inventory.total_energy_j(f) - state.inventory.total_energy_j(f)
    assert change == pytest.approx(0.01 * feed.total_energy_rate_w, abs=1e-7)
    assert abs(result["energy_residual_j"]) < 1e-7
    assert max(map(abs, result["component_residuals_mol"])) < 1e-10
    assert CompositionalSeparator.from_dict(m.to_dict(), device=device).model_id == m.model_id
    m.outlets = (EquilibriumOutlet("changed", 0.99e6, 1e-9),)
    with pytest.raises(PropertyModelError, match="configuration changed"):
        m.advance(state, 0.01)


@pytest.mark.parametrize("device", DEVICES)
def test_schedule_physical_replay_restart_and_complete_preflight(device):
    import copy

    from surface.compositional_separator_schedule import run_compositional_separator_schedule
    from surface.operation import _digest

    f = binary(device)
    m = CompositionalSeparator(
        f, 0.001, [EquilibriumOutlet("out", 0.99e6, 1e-9)], temporal_error_control=False
    )
    state = m.initialize(1e6, 320, [0.5, 0.5])
    periods = [
        {"duration_s": 0.01},
        {
            "duration_s": 0.01,
            "feeds": [
                {
                    "from_pt": {
                        "component_molar_rates_mol_s": [0.001, 0.001],
                        "pressure_pa": 1.02e6,
                        "temperature_k": 320,
                    }
                }
            ],
        },
        {"duration_s": 0.01, "openings": {"out": 0}},
    ]
    original = copy.deepcopy(periods)
    full = run_compositional_separator_schedule(m, state, periods)
    prefix = run_compositional_separator_schedule(m, state, periods, stop_after=2)
    resumed = run_compositional_separator_schedule(m, state, periods, checkpoint=prefix)
    assert resumed == full
    assert periods == original
    assert full["state"]["inventory"]["time_s"] == 0.03
    bad = copy.deepcopy(prefix)
    bad["records"][0]["steps"][0]["fluid_energy_residual_j"] += 1.0
    bad.pop("checksum")
    bad["checksum"] = _digest(bad)
    with pytest.raises(ConservationError, match="physical replay"):
        run_compositional_separator_schedule(m, state, periods, checkpoint=bad)
    bad_periods = copy.deepcopy(periods)
    bad_periods[-1]["openings"] = {"missing": 0}
    with pytest.raises(ValueError, match="opening"):
        run_compositional_separator_schedule(m, state, bad_periods, stop_after=0)


@pytest.mark.parametrize("device", DEVICES)
def test_pure_latent_equilibrium_inventory_and_native_torch_execution(device, monkeypatch):
    import scipy.optimize

    from surface.equilibrium_inventory import EquilibriumInventory

    f = binary(device)
    p = f.components[1].saturation_at_temperature(200)["pressure_pa"]
    l = f.homogeneous(p, 200, [0, 1], "liquid")
    v = f.homogeneous(p, 200, [0, 1], "vapor")
    n = 10.0
    energy = n * float(
        (0.7 * l["molar_internal_energy_j_mol"] + 0.3 * v["molar_internal_energy_j_mol"]).cpu()
    )
    volume = n * float((0.7 * l["molar_volume_m3_mol"] + 0.3 * v["molar_volume_m3_mol"]).cpu())
    m = CompositionalSeparator(f, volume, temporal_error_control=False)
    inv = EquilibriumInventory((0, 10), energy, volume, p, 200, f.model_id, gravity_m_s2=0.0)
    state = CompositionalSeparatorState(inv, None, m.maximum_dt_s, m.model_id)

    def forbid(*args, **kwargs):
        raise AssertionError("SciPy execution in monolithic Torch separator")

    for name in ["root", "minimize", "least_squares", "brentq"]:
        monkeypatch.setattr(scipy.optimize, name, forbid)
    result = m.advance(state, 0.01)
    assert result["steps"][0]["phase_count"] == 2
    assert result["state"]["inventory"]["internal_energy_j"] == pytest.approx(energy, abs=1e-6)
    assert result["steps"][0]["phase_moles"][1] / sum(
        result["steps"][0]["phase_moles"]
    ) == pytest.approx(0.3, abs=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_pure_vapor_condensation_from_conserved_heat(device):
    f = binary(device)
    c = f.components[1]
    p = c.saturation_at_temperature(200)["pressure_pa"]
    liquid = f.homogeneous(p, 200, [0, 1], "liquid")
    vapor = f.homogeneous(p, 200, [0, 1], "vapor")
    vm = float((0.4 * liquid["molar_volume_m3_mol"] + 0.6 * vapor["molar_volume_m3_mol"]).cpu())
    target_u = 10 * float(
        (
            0.4 * liquid["molar_internal_energy_j_mol"] + 0.6 * vapor["molar_internal_energy_j_mol"]
        ).cpu()
    )
    a, _ = c._a(220)
    initial_p = R * 220 / (vm - c.b) - a / (vm * vm + 2 * c.b * vm - c.b * c.b)
    m = CompositionalSeparator(
        f,
        10 * vm,
        maximum_dt_s=1.0,
        temporal_error_control=False,
        maximum_temperature_change_k=50,
    )
    state = m.initialize(initial_p, 220, [0, 1])
    assert m.inspect(state)["phase_count"] == 1
    heat = target_u - state.inventory.internal_energy_j
    result = m.advance(state, 1.0, heat_w=heat, utility_temperature_k=170)
    end = result["state"]["inventory"]
    assert end["temperature_k"] == pytest.approx(200, abs=1e-5)
    assert end["pressure_pa"] == pytest.approx(p, rel=1e-7)
    assert end["internal_energy_j"] == pytest.approx(target_u, abs=1e-5)
    assert result["steps"][-1]["phase_count"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_out_of_domain_pressure_is_rejected_before_newton(device, monkeypatch):
    f = binary(device)
    m = CompositionalSeparator(
        f, 0.001, [EquilibriumOutlet("unsupported", 0.5e6, 1e-9)], temporal_error_control=False
    )
    state = m.initialize(1e6, 320, [0.5, 0.5])

    def forbid(*args, **kwargs):
        raise AssertionError("Known-invalid operating point entered Newton")

    monkeypatch.setattr(m.newton, "solve", forbid)
    with pytest.raises(EquipmentOperatingError, match="calibrated pressure-drop"):
        m.advance(state, 0.01)
    # Closing the outlet removes flow and the conductance calibration domain.
    monkeypatch.undo()
    result = m.advance(state, 0.01, openings={"unsupported": 0})
    assert result["steps"][0]["ports"]["unsupported"]["volume_rate_m3_s"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_pressure_error_control_resolves_stiff_liquid_drain(device):
    f = binary(device)
    z = np.array([0.19, 0.81])
    volume = 0.001
    port = EquilibriumOutlet("liquid", 1.09e6, 1e-9)
    models = [
        CompositionalSeparator(
            f,
            volume,
            [port],
            maximum_dt_s=0.01,
            relative_tolerance=1e-5,
            temporal_error_control=mode,
        )
        for mode in [False, True]
    ]
    states = [m.initialize(1.1e6, 196.2, z) for m in models]
    old = PengRobinsonMixture(f.components, f.binary_interactions)

    def properties(n, t):
        v = volume / n
        a, at, b, _, _ = old._parameters(t, z)
        p = R * t / (v - b) - a / (v * v + 2 * b * v - b * b)

        def u(temp):
            aa, _, _, _, _ = old._parameters(temp, z)
            pp = R * temp / (v - b) - aa / (v * v + 2 * b * v - b * b)
            return old.state(pp, temp, z)["molar_enthalpy_j_mol"] - pp * v

        cv = (u(t + 0.0001) - u(t - 0.0001)) / 0.0002
        dpdt = R / (v - b) - at / (v * v + 2 * b * v - b * b)
        return p, cv, dpdt

    def rhs(time, y):
        n, t = y
        p, cv, dpdt = properties(n, t)
        dn = -port.volume_conductance_m3_s_pa * max(0.0, p - port.back_pressure_pa) * n / volume
        return [dn, dn * t * (volume / n) * dpdt / (n * cv)]

    start = [sum(states[0].inventory.component_moles), 196.2]
    reference = solve_ivp(rhs, [0, 0.01], start, method="DOP853", rtol=1e-11, atol=1e-12).y[:, -1]
    pref = properties(*reference)[0]
    results = [m.advance(state, 0.01) for m, state in zip(models, states)]
    raw, adaptive = results
    assert adaptive["failed_steps"] > 0
    assert any("temporal" in reason for reason in adaptive["retry_reasons"])
    assert abs(adaptive["state"]["inventory"]["pressure_pa"] - pref) < 0.4 * abs(
        raw["state"]["inventory"]["pressure_pa"] - pref
    )
    assert all(s["temporal_error_ratio"] <= 1 for s in adaptive["steps"])
