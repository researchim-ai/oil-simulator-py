"""Coupled PR conservation, upwinding, Jacobian, phase events and restart."""

import copy
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
from surface.compositional_network import (
    CompositionalConnection,
    CompositionalNetwork,
    CompositionalNetworkState,
)
from surface.compositional_separator import CompositionalSeparator, EquilibriumOutlet
from surface.equilibrium_properties import EquilibriumFluid
from surface.liquid import EquipmentOperatingError, HydraulicConvergenceError, PropertyModelError
from surface.peng_robinson import PengRobinsonFluid, R
from surface.peng_robinson_mixture import PengRobinsonMixture
from surface.vessel_wall import VesselWall

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def binary(device):
    c = json.loads((ROOT / "tests/data/surface/pr_mixture_coolprop.json").read_text())
    return EquilibriumFluid(
        [PengRobinsonFluid(**x) for x in c["components"]], [[0, 0], [0, 0]], device=device
    )


def pair(device, *, reverse=False, allow_reverse=True, conductance=1e-10, temporal=False, dt=0.01):
    f = binary(device)
    m = CompositionalNetwork(
        {"a": CompositionalSeparator(f, 0.001), "b": CompositionalSeparator(f, 0.002)},
        [CompositionalConnection("ab", "a", "b", conductance, allow_reverse=allow_reverse)],
        maximum_dt_s=dt,
        temporal_error_control=temporal,
    )
    s = m.initialize(
        {
            "a": {
                "pressure_pa": 0.99e6 if reverse else 1e6,
                "temperature_k": 320,
                "mole_fractions": [0.5, 0.5],
            },
            "b": {
                "pressure_pa": 1e6 if reverse else 0.99e6,
                "temperature_k": 315,
                "mole_fractions": [0.4, 0.6],
            },
        }
    )
    return m, s


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reverse", [False, True])
def test_bulk_transfer_analytic_be_and_independent_upwind_enthalpy(device, reverse):
    m, s = pair(device, reverse=reverse)
    r = m.advance(s, 0.01)
    end = CompositionalNetworkState.from_dict(r["state"])
    body = r["steps"][0]["vessels"]
    link = r["steps"][0]["connections"]["ab"]
    rate = np.array(link["component_molar_rates_mol_s"])
    assert np.all(rate < 0) if reverse else np.all(rate > 0)
    src, dst = (1, 0) if reverse else (0, 1)
    old = s.vessels[src][1].inventory
    new = end.vessels[src][1].inventory
    donor_name, recipient_name = m.names[src], m.names[dst]
    delta_p = body[donor_name]["pressure_pa"] - body[recipient_name]["pressure_pa"]
    factor = (
        1 + 0.01 * m.connections[0].volume_conductance_m3_s_pa * delta_p / m.vessels[src].volume_m3
    )
    assert np.array(new.component_moles) == pytest.approx(
        np.array(old.component_moles) / factor, rel=1e-10
    )
    reference = PengRobinsonMixture(m.fluid.components, m.fluid.binary_interactions)
    x = np.array(new.component_moles) / sum(new.component_moles)
    h = reference.state(new.pressure_pa, new.temperature_k, x)["molar_enthalpy_j_mol"]
    assert link["energy_rate_w"] == pytest.approx(rate.sum() * h, rel=1e-9, abs=1e-9)
    for i, name in enumerate(m.names):
        direction = -1 if i == 0 else 1
        ni = np.array(s.vessels[i][1].inventory.component_moles)
        nf = np.array(end.vessels[i][1].inventory.component_moles)
        assert nf - ni == pytest.approx(direction * 0.01 * rate, abs=1e-11)
        assert end.vessels[i][1].inventory.internal_energy_j - s.vessels[i][
            1
        ].inventory.internal_energy_j == pytest.approx(
            direction * 0.01 * link["energy_rate_w"], abs=1e-7
        )
    assert max(map(abs, r["component_residuals_mol"])) < 1e-10
    assert abs(r["energy_residual_j"]) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_closed_and_check_valve_and_isolated_component(device):
    for reverse, allow, opening in [(True, False, 1), (False, True, 0)]:
        m, s = pair(device, reverse=reverse, allow_reverse=allow)
        r = m.advance(s, 0.01, connection_openings={"ab": opening})
        for (name, a), (_, b) in zip(
            s.vessels, CompositionalNetworkState.from_dict(r["state"]).vessels
        ):
            assert b.inventory.component_moles == pytest.approx(
                a.inventory.component_moles, abs=1e-12
            )
            assert b.inventory.internal_energy_j == pytest.approx(
                a.inventory.internal_energy_j, abs=1e-8
            )
        assert r["steps"][0]["connections"]["ab"]["energy_rate_w"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_new_component_transport_from_exact_zero(device):
    f = binary(device)
    m = CompositionalNetwork(
        {n: CompositionalSeparator(f, 0.001) for n in ["a", "b", "isolated"]},
        [CompositionalConnection("ab", "a", "b", 1e-10, allow_reverse=False)],
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            "a": {"pressure_pa": 1e6, "temperature_k": 320, "mole_fractions": [1, 0]},
            "b": {"pressure_pa": 0.99e6, "temperature_k": 320, "mole_fractions": [0, 1]},
            "isolated": {"pressure_pa": 0.99e6, "temperature_k": 320, "mole_fractions": [0, 1]},
        }
    )
    r = m.advance(s, 0.01)
    end = dict(CompositionalNetworkState.from_dict(r["state"]).vessels)
    assert len(r["steps"]) == 1
    assert end["b"].inventory.component_moles[0] > 0
    assert end["isolated"].inventory.component_moles[0] == 0
    assert end["a"].inventory.component_moles[1] == 0
    assert max(map(abs, r["component_residuals_mol"])) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_full_coupled_jacobian_has_cross_vessel_blocks(device, monkeypatch):
    m, s = pair(device)
    states = [v for _, v in s.vessels]
    seeds = m.inspect(s)
    commands = m.prepare(s)
    active = m._active(states, [c[0] for c in commands[0]], commands[1])
    branches = tuple((q, tuple(range(q["phase_count"]))) for q in seeds)
    problems, offsets, evaluate, pattern = m._problem(states, branches, 0.01, commands, active)
    x = torch.cat([p["initial"] for p in problems])

    # No nested flash and no SciPy within a trial/Jacobian.
    def forbid(*args, **kwargs):
        raise AssertionError("Flash in coupled residual")

    monkeypatch.setattr(m.fluid, "flash_pt", forbid)
    fun = lambda y: evaluate(y)[0]
    ad = torch.func.jacfwd(fun)(x).cpu().numpy()
    fd = []
    for j in range(len(x)):
        d = torch.zeros_like(x)
        d[j] = 2e-6
        fd.append(((fun(x + d) - fun(x - d)) / 4e-6).cpu().numpy())
    np.testing.assert_allclose(ad, np.array(fd).T, rtol=3e-5, atol=2e-7)
    k = offsets[1]
    assert abs(ad[:k, k:]).max() > 1e-6
    assert abs(ad[k:, :k]).max() > 1e-6
    support = set(zip(*pattern))
    assert all((i, j) in support for i, j in zip(*np.nonzero(abs(ad) > 1e-12)))


@pytest.mark.parametrize("device", DEVICES)
def test_global_wall_feed_heat_and_external_outlet_ledger(device):
    f = binary(device)
    wall = VesselWall(2000, 10, 2, 300)
    vessels = {
        "a": CompositionalSeparator(f, 0.001, wall=wall),
        "b": CompositionalSeparator(f, 0.002, [EquilibriumOutlet("export", 0.98e6, 1e-11)]),
    }
    m = CompositionalNetwork(
        vessels,
        [CompositionalConnection("ab", "a", "b", 1e-10)],
        temporal_error_control=True,
        maximum_dt_s=0.01,
    )
    s = m.initialize(
        {
            "a": {
                "pressure_pa": 1e6,
                "temperature_k": 320,
                "mole_fractions": [0.5, 0.5],
                "wall_temperature_k": 325,
            },
            "b": {"pressure_pa": 0.99e6, "temperature_k": 315, "mole_fractions": [0.4, 0.6]},
        }
    )
    feed = ComponentEnergyStream.from_pt(f, [0.001, 0.001], 1.1e6, 330, gravity_m_s2=0)
    r = m.advance(
        s, 0.025, feeds={"a": [feed]}, heat_w={"b": 100}, utility_temperature_k={"b": 400}
    )
    assert all(v["inventory"]["time_s"] == 0.025 for v in r["state"]["vessels"].values())
    assert abs(r["energy_residual_j"]) < 1e-6
    assert max(map(abs, r["component_residuals_mol"])) < 1e-9
    assert all(row["temporal_error_ratio"] <= 1 for row in r["steps"])


@pytest.mark.parametrize("device", DEVICES)
def test_three_phase_selective_link(device):
    cfg = json.loads((ROOT / "configs/surface_equilibrium_process.json").read_text())
    f = EquilibriumFluid.from_dict(cfg["fluid"], device=device)
    m = CompositionalNetwork(
        {n: CompositionalSeparator(f, 0.001) for n in ["a", "b"]},
        [
            CompositionalConnection(
                "water",
                "a",
                "b",
                1e-10,
                source_selection="component_rich",
                source_component="water",
                target_selection="lightest",
            )
        ],
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            "a": {"pressure_pa": 1e6, "temperature_k": 350, "mole_fractions": [0.4, 0.3, 0.3]},
            "b": {"pressure_pa": 0.99e6, "temperature_k": 350, "mole_fractions": [0.4, 0.3, 0.3]},
        }
    )
    r = m.advance(s, 0.01)
    assert all(v["phase_count"] == 3 for v in r["steps"][0]["vessels"].values())
    rate = r["steps"][0]["connections"]["water"]["component_molar_rates_mol_s"]
    assert rate[2] > 1000 * rate[0]
    assert max(map(abs, r["component_residuals_mol"])) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_pure_phase_birth_in_coupled_system(device):
    f = binary(device)
    c = f.components[1]
    p = c.saturation_at_temperature(200)["pressure_pa"]
    l = f.homogeneous(p, 200, [0, 1], "liquid")
    v = f.homogeneous(p, 200, [0, 1], "vapor")
    vm = float((0.4 * l["molar_volume_m3_mol"] + 0.6 * v["molar_volume_m3_mol"]).cpu())
    u = 10 * float(
        (0.4 * l["molar_internal_energy_j_mol"] + 0.6 * v["molar_internal_energy_j_mol"]).cpu()
    )
    a, _ = c._a(220)
    pi = R * 220 / (vm - c.b) - a / (vm * vm + 2 * c.b * vm - c.b * c.b)
    m = CompositionalNetwork(
        {
            "cool": CompositionalSeparator(f, 10 * vm, maximum_temperature_change_k=50),
            "other": CompositionalSeparator(f, 0.001),
        },
        temporal_error_control=False,
        maximum_dt_s=1,
    )
    s = m.initialize(
        {
            "cool": {"pressure_pa": pi, "temperature_k": 220, "mole_fractions": [0, 1]},
            "other": {"pressure_pa": 1e6, "temperature_k": 320, "mole_fractions": [0.5, 0.5]},
        }
    )
    heat = u - dict(s.vessels)["cool"].inventory.internal_energy_j
    r = m.advance(s, 1, heat_w={"cool": heat}, utility_temperature_k={"cool": 170})
    end = r["state"]["vessels"]["cool"]["inventory"]
    assert end["temperature_k"] == pytest.approx(200, abs=1e-5)
    assert end["pressure_pa"] == pytest.approx(p, rel=1e-7)
    assert r["steps"][-1]["vessels"]["cool"]["phase_count"] == 2


@pytest.mark.parametrize("device", DEVICES)
def test_temporal_refinement_against_independent_closed_ode(device):
    f = binary(device)
    ref = PengRobinsonMixture(f.components, f.binary_interactions)
    z = np.array([0.5, 0.5])
    vols = [0.001, 0.002]
    conductance = 1e-8

    def props(n, t, volume):
        v = volume / n
        a, at, b, _, _ = ref._parameters(t, z)
        p = R * t / (v - b) - a / (v * v + 2 * b * v - b * b)
        q = ref.state(p, t, z)
        h = q["molar_enthalpy_j_mol"]

        def u(tt):
            aa, *_ = ref._parameters(tt, z)
            pp = R * tt / (v - b) - aa / (v * v + 2 * b * v - b * b)
            return ref.state(pp, tt, z)["molar_enthalpy_j_mol"] - pp * v

        cv = (u(t + 0.0001) - u(t - 0.0001)) / 0.0002
        dpdt = R / (v - b) - at / (v * v + 2 * b * v - b * b)
        return p, h, cv, h - t * v * dpdt

    def rhs(time, y):
        qa = props(y[0], y[2], vols[0])
        qb = props(y[1], y[3], vols[1])
        donor = 0 if qa[0] >= qb[0] else 1
        rate = conductance * (qa[0] - qb[0]) * y[donor] / vols[donor]
        h = (qa, qb)[donor][1]
        return [
            -rate,
            rate,
            (-rate * h + qa[3] * rate) / (y[0] * qa[2]),
            (rate * h - qb[3] * rate) / (y[1] * qb[2]),
        ]

    errors = []
    for dt in [0.02, 0.01, 0.005]:
        m = CompositionalNetwork(
            {n: CompositionalSeparator(f, v) for n, v in zip(["a", "b"], vols)},
            [CompositionalConnection("ab", "a", "b", conductance)],
            maximum_dt_s=dt,
            temporal_error_control=False,
        )
        s = m.initialize(
            {
                "a": {"pressure_pa": 1e6, "temperature_k": 320, "mole_fractions": z},
                "b": {"pressure_pa": 0.99e6, "temperature_k": 320, "mole_fractions": z},
            }
        )
        y0 = [
            sum(s.vessels[0][1].inventory.component_moles),
            sum(s.vessels[1][1].inventory.component_moles),
            320,
            320,
        ]
        reference = solve_ivp(rhs, [0, 0.04], y0, method="DOP853", rtol=1e-11, atol=1e-12).y[:, -1]
        r = m.advance(s, 0.04)
        end = CompositionalNetworkState.from_dict(r["state"])
        actual = np.array(
            [sum(x.inventory.component_moles) for _, x in end.vessels]
            + [x.inventory.temperature_k for _, x in end.vessels]
        )
        errors.append(np.linalg.norm((actual - reference) / np.array([1, 1, 320, 320])))
    assert errors[1] < 0.7 * errors[0]
    assert errors[2] < 0.7 * errors[1]


@pytest.mark.parametrize("device", DEVICES)
def test_invalid_domain_budget_identity_and_transaction(device):
    m, s = pair(device)
    original = copy.deepcopy(s.to_dict())
    with pytest.raises(HydraulicConvergenceError, match="budget"):
        m.advance(s, 0.1, max_substeps=1)
    assert s.to_dict() == original
    with pytest.raises(ValueError, match="Unknown connection"):
        m.advance(s, 0.01, connection_openings={"missing": 0})
    with pytest.raises(ValueError, match="utility"):
        m.advance(s, 0.01, heat_w={"a": 1})
    bad = replace(s, model_id="0" * 64)
    with pytest.raises(PropertyModelError):
        m.advance(bad, 0.01)
    badclock = replace(
        s,
        vessels=(
            s.vessels[0],
            ("b", replace(s.vessels[1][1], inventory=replace(s.vessels[1][1].inventory, time_s=1))),
        ),
    )
    with pytest.raises(ValueError, match="clocks"):
        m.advance(badclock, 0.01)
    wide = CompositionalNetwork(
        dict(zip(m.names, m.vessels)),
        [CompositionalConnection("ab", "a", "b", 1e-10, maximum_pressure_drop_fraction=0.001)],
    )
    w = replace(s, model_id=wide.model_id)
    with pytest.raises(EquipmentOperatingError):
        wide.advance(w, 0.01)
    assert wide.advance(w, 0.01, connection_openings={"ab": 0})["failed_steps"] == 0


@pytest.mark.parametrize("device", DEVICES)
def test_sparse_coupled_path_agrees_with_direct(device, monkeypatch):
    import surface.compositional_network as implementation
    from surface.torch_linear import TorchNewton

    m, s = pair(device)
    direct = m.advance(s, 0.01)

    def sparse(*args, **kwargs):
        solver = TorchNewton(*args, **kwargs)
        solver.dense_limit = 0
        return solver

    monkeypatch.setattr(implementation, "TorchNewton", sparse)
    result = m.advance(s, 0.01)
    for name in m.names:
        a = direct["state"]["vessels"][name]["inventory"]
        b = result["state"]["vessels"][name]["inventory"]
        assert b["pressure_pa"] == pytest.approx(a["pressure_pa"], abs=1e-3)
        assert b["temperature_k"] == pytest.approx(a["temperature_k"], abs=1e-7)
        assert b["component_moles"] == pytest.approx(a["component_moles"], abs=1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_manufactured_phase_birth_with_actual_finite_receiver(device):
    from scipy.optimize import root

    f = binary(device)
    ref = PengRobinsonMixture(f.components, f.binary_interactions)
    z = np.array([0.19, 0.81])
    temperature = 196.2
    va = 0.001
    vb = 0.1

    def values(p, t, volume):
        q = ref.flash_pt(p, t, z)
        vm = sum(x["molar_fraction"] * x["molar_volume_m3_mol"] for x in q["phases"])
        n = volume / vm
        return n, n * q["molar_enthalpy_j_mol"] - p * volume, q["molar_enthalpy_j_mol"]

    na, ua, _ = values(1.1e6, temperature, va)
    nf, uf, h = values(1e6, temperature, va)
    nb, ub, _ = values(0.98e6, temperature, vb)
    transfer = na - nf
    nold, uold = nb - transfer, ub - transfer * h

    def residual(y):
        n, u, _ = values(np.exp(y[0]), y[1], vb)
        return [(n - nold) / nold, (u - uold) / max(1, abs(uold))]

    sol = root(residual, [np.log(0.97e6), temperature])
    assert sol.success and np.max(np.abs(residual(sol.x))) < 1e-8
    pold, told = np.exp(sol.x[0]), sol.x[1]
    g = transfer * va / (0.02e6 * nf)
    m = CompositionalNetwork(
        {
            "a": CompositionalSeparator(f, va, maximum_fractional_inventory_change=0.5),
            "b": CompositionalSeparator(f, vb, maximum_fractional_inventory_change=0.5),
        },
        [CompositionalConnection("ab", "a", "b", g, maximum_pressure_drop_fraction=0.2)],
        maximum_dt_s=1,
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            "a": {"pressure_pa": 1.1e6, "temperature_k": temperature, "mole_fractions": z},
            "b": {"pressure_pa": pold, "temperature_k": told, "mole_fractions": z},
        }
    )
    assert m.inspect(s)[0]["phase_count"] == 1
    r = m.advance(s, 1, heat_w={"a": uf - ua + transfer * h}, utility_temperature_k={"a": 400})
    for name, p in [("a", 1e6), ("b", 0.98e6)]:
        end = r["state"]["vessels"][name]["inventory"]
        assert end["pressure_pa"] == pytest.approx(p, rel=1e-7)
        assert end["temperature_k"] == pytest.approx(temperature, abs=1e-6)
    assert r["steps"][-1]["vessels"]["a"]["phase_count"] == 2
    assert len(r["steps"]) == 1


@pytest.mark.parametrize("device", DEVICES)
def test_multihop_new_component_arrives_within_same_implicit_step(device):
    f = binary(device)
    m = CompositionalNetwork(
        {n: CompositionalSeparator(f, 0.001) for n in ["a", "b", "c"]},
        [
            CompositionalConnection("ab", "a", "b", 1e-9, allow_reverse=False),
            CompositionalConnection("bc", "b", "c", 1e-9, allow_reverse=False),
        ],
        temporal_error_control=False,
    )
    s = m.initialize(
        {
            n: {"pressure_pa": p, "temperature_k": 320, "mole_fractions": z}
            for n, p, z in [("a", 1e6, [1, 0]), ("b", 0.99e6, [0, 1]), ("c", 0.98e6, [0, 1])]
        }
    )
    r = m.advance(s, 0.01)
    assert len(r["steps"]) == 1
    assert r["state"]["vessels"]["c"]["inventory"]["component_moles"][0] > 1e-10
    assert max(map(abs, r["component_residuals_mol"])) < 1e-9


@pytest.mark.parametrize("device", DEVICES)
def test_network_checkpoint_prefix_replay_and_future_preflight(device):
    from surface.compositional_network_schedule import run_compositional_network_schedule
    from surface.liquid import ConservationError
    from surface.operation import _digest

    m, s = pair(device)
    periods = [
        {"duration_s": 0.01},
        {"duration_s": 0.01, "connection_openings": {"ab": 0}},
        {"duration_s": 0.01, "heat_w": {"b": 10}, "utility_temperature_k": {"b": 400}},
    ]
    full = run_compositional_network_schedule(m, s, periods)
    prefix = run_compositional_network_schedule(m, s, periods, stop_after=1)
    resumed = run_compositional_network_schedule(m, s, periods, checkpoint=prefix)
    assert resumed == full
    forged = copy.deepcopy(prefix)
    forged["records"][0]["steps"][0]["connections"]["ab"]["energy_rate_w"] += 1
    forged.pop("checksum")
    forged["checksum"] = _digest(forged)
    with pytest.raises(ConservationError, match="physical replay"):
        run_compositional_network_schedule(m, s, periods, checkpoint=forged)
    bad = copy.deepcopy(periods)
    bad[-1]["connection_openings"] = {"typo": 0}
    with pytest.raises(ValueError, match="Unknown connection"):
        run_compositional_network_schedule(m, s, bad, stop_after=0)


def test_common_eos_device_and_elevation_are_explicit():
    f = binary("cpu")
    a = CompositionalSeparator(f, 0.001)
    b = CompositionalSeparator(f, 0.001, elevation_m=1)
    with pytest.raises(PropertyModelError, match="elevation"):
        CompositionalNetwork({"a": a, "b": b})
    with pytest.raises(ValueError, match="endpoint"):
        CompositionalNetwork({"a": a}, [CompositionalConnection("bad", "a", "missing", 1e-10)])
    with pytest.raises(ValueError, match="self loop"):
        CompositionalConnection("bad", "a", "a", 1e-10)
