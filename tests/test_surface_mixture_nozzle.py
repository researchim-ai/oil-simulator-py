"""HEM mixture nozzle: limiting physics, balances, resolution and domains."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from surface import (
    IdealGas,
    MixtureEquilibriumNozzle,
    PengRobinsonFluid,
    PengRobinsonMixture,
    PropertyModelError,
    ideal_gas_nozzle_flow,
)
from surface.peng_robinson import R


def fluid():
    spec = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/surface_thermal_mixture.json").read_text()
    )["fluid"]
    return PengRobinsonMixture(
        tuple(PengRobinsonFluid(**c) for c in spec["components"]), spec["binary_interactions"]
    )


def test_ideal_gas_mixture_limit():
    f = fluid()
    z = np.array([0.3, 0.7])
    cp = sum(x * c.ideal_cp_j_mol_k for x, c in zip(z, f.components))
    mw = sum(x * c.molar_mass_kg_mol for x, c in zip(z, f.components))
    n = MixtureEquilibriumNozzle(
        "dilute", 1e-4, 10.0, 0.8, scan_points=9, pressure_tolerance_pa=1e-6
    )
    r = n.run(100.0, 400.0, 10.0, z, f)
    ideal = ideal_gas_nozzle_flow(100.0, 400.0, 10.0, IdealGas(R / mw, cp / (cp - R)), 1e-4, 0.8)
    assert r["regime"] == "choked"
    assert r["mass_rate_kg_s"] == pytest.approx(ideal["mass_rate_kg_s"], rel=2e-5)
    assert r["critical_pressure_pa"] / 100 == pytest.approx(
        ideal["critical_pressure_ratio"], rel=2e-5
    )
    assert sum(r["component_mass_rates_kg_s"]) == pytest.approx(r["mass_rate_kg_s"])
    assert abs(r["outlet_energy_residual_w"]) < 1e-6


def test_two_phase_choking_and_mesh_resolution():
    f = fluid()
    z = [0.5, 0.5]
    nozzle = MixtureEquilibriumNozzle("mixture", 1e-5, 5e5, 0.8, scan_points=9)
    r = nozzle.run(3e6, 220, 5e5, z, f)
    dense = MixtureEquilibriumNozzle("dense", 1e-5, 5e5, 0.8, scan_points=17).run(
        3e6, 220, 5e5, z, f
    )
    assert r["regime"] == "choked"
    assert r["inlet"]["phase_count"] == r["throat"]["phase_count"] == 2
    assert r["mass_rate_kg_s"] == pytest.approx(dense["mass_rate_kg_s"], rel=2e-5)
    assert r["critical_pressure_pa"] == pytest.approx(dense["critical_pressure_pa"], rel=2e-5)
    back = (r["critical_pressure_pa"] + 5e5) / 2
    plateau = nozzle.run(3e6, 220, back, z, f)
    assert plateau["mass_rate_kg_s"] == pytest.approx(r["mass_rate_kg_s"], rel=1e-10)
    sub = nozzle.run(3e6, 220, 2.9e6, z, f)
    assert sub["regime"] == "subcritical" and sub["mass_rate_kg_s"] < r["mass_rate_kg_s"]
    for q in [r, dense, plateau, sub]:
        assert abs(q["isentropic_energy_residual_w"]) < 1e-6
        assert abs(q["outlet_energy_residual_w"]) < 1e-6
        assert q["entropy_generation_w_k"] >= -1e-6
        assert q["throat"]["component_balance_error"] < 1e-8
        assert sum(q["component_mass_rates_kg_s"]) == pytest.approx(q["mass_rate_kg_s"])


def test_no_flow_invalid_and_unresolved_critical():
    f = fluid()
    n = MixtureEquilibriumNozzle("domain", 1e-5, 95.0, scan_points=9)
    assert n.run(100.0, 400.0, 95.0, [0.5, 0.5], f, opening=0)["outlet"] is None
    assert n.run(100.0, 400.0, 100.0, [0.5, 0.5], f)["mass_rate_kg_s"] == 0
    r = n.run(100.0, 400.0, 95.0, [0.5, 0.5], f)
    assert r["regime"] == "subcritical"
    assert not r["critical_pressure_resolved"] and r["critical_pressure_pa"] is None
    with pytest.raises(PropertyModelError):
        n.run(100.0, 400.0, 101.0, [0.5, 0.5], f)
    with pytest.raises(ValueError):
        n.run(100.0, 400.0, 95.0, [0.5, 0.5], f, opening=1.1)


def test_single_phase_feed_can_flash_inside_nozzle():
    f = fluid()
    r = MixtureEquilibriumNozzle("flashing", 1e-5, 5e5, scan_points=9).run(
        3e6, 260, 5e5, [0.5, 0.5], f
    )
    assert r["inlet"]["phase_count"] == 1
    assert r["throat"]["phase_count"] == 2
    assert r["regime"] == "choked"
    assert abs(r["outlet_energy_residual_w"]) < 1e-6
    assert r["throat"]["log_fugacity_error"] < 1e-8
    assert r["entropy_generation_w_k"] > 0
