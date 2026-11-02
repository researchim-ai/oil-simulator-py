"""Two-phase HEM transfer: exact caching, Jacobian tangents and dynamics."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    DynamicVessel,
    DynamicVesselNetwork,
    HEMNozzleConnection,
    MixtureEquilibriumNozzle,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PRNozzleConnection,
    PropertyModelError,
    VesselState,
    run_network_schedule,
    vessel_connection_from_dict,
)


def case(temperature=220.0):
    c = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    f = PengRobinsonMixture(
        tuple(PengRobinsonFluid(**x) for x in c["components"]), c["binary_interactions"]
    )
    v = {n: DynamicVessel(n, 0.01, 1e5, 5e6) for n in ("a", "b")}
    s = {
        "a": v["a"].initialize(f, 3e6, temperature, [0.5, 0.5]),
        "b": v["b"].initialize(f, 1e6, 220, [0.5, 0.5]),
    }
    c = HEMNozzleConnection("hem", "a", "b", 1e-8, 5e5, 3e6, 0.1, scan_points=9)
    return f, v, s, c


def values(fluid, vessels, states):
    return {
        name: (
            np.array(s.component_moles),
            s.pressure_pa,
            s.temperature_k,
            vessels[name].inspect(s, fluid)["equilibrium"],
            None,
            None,
            None,
        )
        for name, s in states.items()
    }


def test_exact_capacity_cache_reuses_choked_path_and_invalidates_source(monkeypatch):
    f, _, _, _ = case()
    model = MixtureEquilibriumNozzle("reference", 1e-8, 5e5, scan_points=9)
    original = f.__class__.flash_ps
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(f.__class__, "flash_ps", counted)
    cache = {}
    a = model.run(3e6, 220, 5e5, [0.5, 0.5], f, with_outlet=False, _cache=cache)
    number = len(calls)
    b = model.run(3e6, 220, 8e5, [0.5, 0.5], f, with_outlet=False, _cache=cache)
    assert a["regime"] == b["regime"] == "choked"
    assert a["mass_rate_kg_s"] == b["mass_rate_kg_s"] and len(calls) == number
    doubled = MixtureEquilibriumNozzle("area", 2e-8, 5e5, scan_points=9).run(
        3e6, 220, 8e5, [0.5, 0.5], f, with_outlet=False, _cache=cache
    )
    assert doubled["mass_rate_kg_s"] == 2 * a["mass_rate_kg_s"] and len(calls) == number
    changed = model.run(3e6, 221, 8e5, [0.5, 0.5], f, with_outlet=False, _cache=cache)
    assert len(calls) > number and changed["mass_rate_kg_s"] != a["mass_rate_kg_s"]
    assert changed["outlet"] is None and changed["outlet_energy_residual_w"] is None


def test_single_phase_feed_can_generate_two_phase_throat_without_fallback():
    f, v, s, c = case(260)
    x = values(f, v, s)
    p = {n: q.pressure_pa for n, q in s.items()}
    cache = {}
    rate = c.flow(p, values=x, fluid=f, cache=cache)
    diag = cache[("diagnostic", "hem")]
    assert rate > 0 and diag["upstream_port_phase_count"] == 1 and diag["throat_phase_count"] == 2
    with pytest.raises(PropertyModelError):
        PRNozzleConnection("single", "a", "b", 1e-8, 3e6, 0.1).flow(p, values=x, fluid=f)


def test_smooth_critical_tangent_matches_full_search_derivative():
    f, v, s, c = case()
    x = values(f, v, s)
    p = {n: q.pressure_pa for n, q in s.items()}
    cache = {}
    base = c.flow(p, values=x, fluid=f, cache=cache)
    tangent = c.linearized_flow(p, x, f, 1.0, cache)
    assert tangent is not None
    dp = 100.0
    rates = []
    linear = []
    for sign in (-1, 1):
        changed = dict(s)
        changed["a"] = v["a"].initialize(f, 3e6 + sign * dp, 220, [0.5, 0.5])
        xx = values(f, v, changed)
        pp = {n: q.pressure_pa for n, q in changed.items()}
        rates.append(c.flow(pp, values=xx, fluid=f))
        linear.append(tangent(pp, xx))
    assert (linear[1] - linear[0]) / (2 * dp) == pytest.approx(
        (rates[1] - rates[0]) / (2 * dp), rel=2e-3
    )
    pback = dict(p)
    pback["b"] += 1000
    assert tangent(pback, x) == pytest.approx(base, rel=1e-12)


def test_two_phase_network_conserves_mass_energy_and_replays_physical_flow():
    f, v, s, c = case()
    r = DynamicVesselNetwork(v, (c,)).advance(s, f, 0.01, initial_dt_s=0.01)
    step = r["steps"][0]
    diag = step["connections"][0]
    assert diag["upstream_port_phase_count"] == diag["throat_phase_count"] == 2
    assert "throat_mach" not in diag
    n0 = np.sum([q.component_moles for q in s.values()], axis=0)
    n1 = np.sum([q["component_moles"] for q in r["states"].values()], axis=0)
    assert n1 == pytest.approx(n0, abs=1e-7)
    assert abs(step["energy_residual_j"]) < 1e-5
    new = {n: VesselState(**q) for n, q in r["states"].items()}
    xx = values(f, v, new)
    pp = {n: q.pressure_pa for n, q in new.items()}
    exact = c.flow(pp, values=xx, fluid=f)
    assert diag["molar_rate_mol_s"] == pytest.approx(exact, rel=1e-9)


def test_hem_reverse_closed_factory_and_restart_without_doubling_transfer():
    f, v, s, c = case()
    reverse = replace(c, source="b", target="a", allow_reverse=True)
    x = values(f, v, s)
    p = {n: q.pressure_pa for n, q in s.items()}
    assert reverse.flow(p, values=x, fluid=f) < 0
    assert reverse.flow(p, factor=0, values=x, fluid=f) == 0
    assert vessel_connection_from_dict({"model": "pr_hem_nozzle", **asdict(reverse)}) == reverse
    # A closed interval still evolves vessel inventory/energy correctly and
    # exercises the same new model fingerprint/checkpoint without an extra costly transfer.
    periods = [
        {"duration_s": 0.01, "connection_factors": {"hem": 0}},
        {"duration_s": 0.01, "connection_factors": {"hem": 0}},
    ]
    net = DynamicVesselNetwork(v, (c,))
    full = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.01})
    partial = run_network_schedule(f, net, s, periods, stepper={"initial_dt_s": 0.01}, stop_after=1)
    resumed = run_network_schedule(
        f,
        net,
        s,
        periods,
        stepper={"initial_dt_s": 0.01},
        checkpoint=json.loads(json.dumps(partial)),
    )
    assert resumed == full and full["totals"]["outlet_component_moles"] == [0.0, 0.0]


def test_ps_temperature_hint_changes_only_search_not_equilibrium():
    f, _, _, _ = case()
    reference = f.flash_pt(3e6, 220, [0.5, 0.5])
    sm = reference["molar_entropy_j_mol_k"]
    cold = f.flash_ps(2e6, sm, [0.5, 0.5])
    warm = f.flash_ps(2e6, sm, [0.5, 0.5], temperature_guess_k=cold["temperature_k"])
    assert warm["temperature_k"] == pytest.approx(cold["temperature_k"], abs=1e-7)
    assert warm["density_kg_m3"] == pytest.approx(cold["density_kg_m3"], rel=1e-8)
    with pytest.raises(PropertyModelError):
        f.flash_ps(2e6, sm, [0.5, 0.5], temperature_guess_k=999)
