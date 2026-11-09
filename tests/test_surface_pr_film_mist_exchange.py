"""Real morphology stocks, surface energy, momentum and kinetic funding."""

import math

import numpy as np
import pytest
import torch
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_film_mist_exchange import PRFilmMistExchange


def case(*, gas_u=30.0, film_u=0.0, mist=False, film=True, gas=True, device="cpu", efficiency=0.5):
    base, v = setup(count=2)
    raw = base.to_dict()
    raw["phase_slots"] = [
        {"name": "film", "root": "liquid"},
        {"name": "mist", "root": "liquid"},
        {"name": "gas", "root": "vapor"},
    ]
    p = PRActiveMultifluidPipe.from_dict(raw, device=device)
    pressure = p.fluid.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cells = []
    for _ in range(2):
        c = {}
        vf = 0.02 * v if film else 0.0
        vd = 0.002 * v if mist else 0.0
        if not gas:
            vf = v - vd if film else 0.0
            if not film:
                vd = v
        if film:
            c["film"] = phase(p.fluid, pressure, 200.0, "liquid", vf, film_u)
        if mist:
            c["mist"] = phase(p.fluid, pressure, 200.0, "liquid", vd, film_u)
        if gas:
            c["gas"] = phase(p.fluid, pressure, 200.0, "vapor", v - vf - vd, gas_u)
        cells.append(c)
    s = p.initialize(cells)
    law = PRFilmMistExchange(
        "film", "mist", "gas", 1e-5, 0.02, efficiency, 0.75, "planar_complete_wetting"
    )
    return p, s, law


@pytest.mark.parametrize("speed", [-30.0, 30.0])
@pytest.mark.parametrize("efficiency", [0.5, 1.0])
def test_actual_drop_birth_preserves_components_momentum_and_surface_plus_bulk_energy(
    speed, efficiency
):
    from reference_pr_active_multifluid import packets

    p, s, law = case(gas_u=speed, efficiency=efficiency)
    end, r = law.apply(p, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[0.0, 0.0])
    assert "mist" in dict(end.cells[0])
    assert end.cells[1] == s.cells[1]
    a, b = p._totals(s), p._totals(end)
    b[-2] += sum(r["surface_energy_after_j"]) - sum(r["surface_energy_before_j"])
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-12)
    assert (r["gas_to_mist_impulse_kg_m_s"][0] > 0) == (speed > 0)
    assert r["kinetic_energy_removed_j"][0] == pytest.approx(
        r["surface_creation_j"][0] / efficiency
    )
    assert r["irreversible_heat_j"][0] == pytest.approx(
        r["kinetic_energy_removed_j"][0] - r["surface_creation_j"][0]
    )
    assert r["droplet_number"][0] == pytest.approx(
        6 * dict(end.cells[0])["mist"].volume_m3 / (math.pi * law.droplet_diameter_m**3)
    )
    packets(p, end.to_dict())


def test_deposition_releases_surface_energy_and_creates_a_real_film_without_gas():
    p, s, law = case(mist=True, film=False, gas=False)
    amount = dict(s.cells[0])["mist"].volume_m3
    end, r = law.apply(p, s, 1.0, entrainment_m3_s=[0.0, 0.0], deposition_m3_s=[amount, 0.0])
    assert tuple(n for n, _ in end.cells[0]) == ("film",)
    assert r["surface_energy_after_j"][0] == 0.0
    assert r["droplet_number"][0] == 0.0
    assert dict(end.cells[0])["film"].temperature_k > dict(s.cells[0])["mist"].temperature_k
    assert r["deposition_heat_j"][0] == pytest.approx(r["surface_energy_before_j"][0])
    assert max(map(abs, r["conservation_residuals"])) < 1e-10


def test_insufficient_energy_and_excess_inventory_are_rejected_atomically():
    from surface.liquid import PropertyModelError

    p, s, law = case(gas_u=0.0)
    snapshot = s.to_dict()
    with pytest.raises(PropertyModelError, match="kinetic"):
        law.apply(p, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[0.0, 0.0])
    with pytest.raises(PropertyModelError, match="volume"):
        law.apply(p, s, 1.0, entrainment_m3_s=[1.0, 0.0], deposition_m3_s=[0.0, 0.0])
    assert s.to_dict() == snapshot


def test_gross_exchange_still_costs_energy_when_net_surface_area_is_unchanged():
    p, s, law = case(mist=True)
    end, r = law.apply(p, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[1e-7, 0.0])
    assert r["surface_energy_after_j"] == pytest.approx(r["surface_energy_before_j"])
    assert r["kinetic_energy_removed_j"][0] > 0
    assert r["deposition_heat_j"][0] > 0
    assert r["entropy_production_j_k"] >= -1e-12
    assert max(map(abs, p.ledger(s, end))) < 1e-10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_morphology_exchange_cuda_parity():
    a, s, law = case(mist=True)
    b, sg, _ = case(mist=True, device="cuda")
    u, ru = law.apply(a, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[2e-8, 0.0])
    v, rv = law.apply(b, sg, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[2e-8, 0.0])
    for ca, cb in zip(u.cells, v.cells):
        for (_, x), (_, y) in zip(ca, cb):
            assert y.temperature_k == pytest.approx(x.temperature_k, abs=1e-9)
            assert y.momentum_kg_m_s == pytest.approx(x.momentum_kg_m_s, abs=1e-16)
    assert rv["joint_energy_residual_j"] == pytest.approx(ru["joint_energy_residual_j"], abs=1e-12)


def test_creation_with_unit_efficiency_uses_kinetic_energy_without_cooling_the_phases():
    p, s, law = case(efficiency=1.0)
    end, r = law.apply(p, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[0.0, 0.0])
    assert r["irreversible_heat_j"] == [0.0, 0.0]
    for name, p0 in end.cells[0]:
        assert p0.temperature_k == pytest.approx(200.0, abs=1e-10)
    # A quiescent source is an exact identity, including P/T cache fields.
    idle, _ = law.apply(p, s, 0.01, entrainment_m3_s=[0.0, 0.0], deposition_m3_s=[0.0, 0.0])
    assert idle.to_dict() == s.to_dict()


def test_exchange_is_galilean_invariant():
    p, s, law = case(gas_u=30.0, film_u=0.0)
    shifted, ss, _ = case(gas_u=130.0, film_u=100.0)
    a, ra = law.apply(p, s, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[0.0, 0.0])
    b, rb = law.apply(shifted, ss, 0.01, entrainment_m3_s=[1e-7, 0.0], deposition_m3_s=[0.0, 0.0])
    assert rb["gas_to_mist_impulse_kg_m_s"] == pytest.approx(
        ra["gas_to_mist_impulse_kg_m_s"], rel=1e-12
    )
    for (_, x), (_, y) in zip(a.cells[0], b.cells[0]):
        mass = sum(n * c.molar_mass_kg_mol for n, c in zip(x.component_moles, p.fluid.components))
        assert y.temperature_k == pytest.approx(x.temperature_k, abs=1e-9)
        assert y.momentum_kg_m_s - x.momentum_kg_m_s == pytest.approx(100 * mass, rel=1e-12)
        assert y.total_energy_j - x.total_energy_j == pytest.approx(
            100 * x.momentum_kg_m_s + 5000 * mass, rel=1e-11, abs=1e-12
        )


def test_film_can_disappear_without_a_ghost_stock_in_the_declared_complete_wetting_limit():
    p, s, law = case(gas_u=100.0, efficiency=1.0)
    amount = dict(s.cells[0])["film"].volume_m3
    end, r = law.apply(p, s, 1.0, entrainment_m3_s=[amount, 0.0], deposition_m3_s=[0.0, 0.0])
    assert tuple(n for n, _ in end.cells[0]) == ("mist", "gas")
    assert r["droplet_number"][0] > 0
    assert abs(r["joint_energy_residual_j"]) < 1e-10


@pytest.mark.parametrize(
    "change",
    [
        {"atomization_efficiency": 0.0},
        {"atomization_efficiency": True},
        {"heat_to_mist_fraction": float("nan")},
        {"film_surface_model": "partial_wetting"},
    ],
)
def test_missing_or_invalid_physical_assumptions_are_not_silently_filled(change):
    _, _, law = case()
    with pytest.raises(ValueError):
        PRFilmMistExchange.from_dict({**law.to_dict(), **change})


def test_independent_com_audit_rejects_forged_surface_work_and_momentum():
    import json
    from copy import deepcopy

    from reference_pr_film_mist_exchange import audit

    p, s, law = case(mist=True)
    e, d = [1e-7, 0.0], [2e-8, 0.0]
    after, r = law.apply(p, s, 0.01, entrainment_m3_s=e, deposition_m3_s=d)
    before, after, report = (json.loads(json.dumps(o)) for o in (s.to_dict(), after.to_dict(), r))
    assert audit(p, before, after, report, law.to_dict(), e, d, 0.01)["accepted"]
    for key in ("surface_creation_j", "gas_to_mist_impulse_kg_m_s"):
        bad = deepcopy(report)
        bad[key][0] *= 1.1
        with pytest.raises(ValueError):
            audit(p, before, after, bad, law.to_dict(), e, d, 0.01)


def test_component_compositions_are_carried_through_both_gross_transfers():
    from reference_pr_film_mist_exchange import audit

    p, s, law = case(mist=True)
    cells = []
    for cell in s.cells:
        c = dict(cell)
        for n, x in [("film", (0.01, 0.99)), ("mist", (0.02, 0.98))]:
            old = c[n]
            c[n] = phase(p.fluid, old.pressure_pa, 200.0, "liquid", old.volume_m3, x=x)
        cells.append(c)
    s = p.initialize(cells)
    e, d = [1e-7, 0.0], [2e-8, 0.0]
    end, r = law.apply(p, s, 0.01, entrainment_m3_s=e, deposition_m3_s=d)
    assert audit(p, s.to_dict(), end.to_dict(), r, law.to_dict(), e, d, 0.01)["accepted"]
    assert p._totals(end)[:2] == pytest.approx(p._totals(s)[:2], rel=1e-12, abs=1e-18)


def test_source_schedule_independent_audit_and_restart():
    import json

    from check_pr_film_mist_exchange import audit
    from run_pr_film_mist_exchange import run

    p, s, law = case()
    period = {
        "duration_s": 0.002,
        "maximum_step_s": 0.001,
        "entrainment_m3_s": [1e-7, 0.0],
        "deposition_m3_s": [0.0, 0.0],
    }
    config = json.loads(
        json.dumps(
            {
                "pipe": p.to_dict(),
                "exchange": law.to_dict(),
                "initial": s.to_dict(),
                "schedule": [period, period],
            }
        )
    )
    r = json.loads(json.dumps(run(config)))
    assert audit(config, r)["checked_steps"] == 4
    restart = {**config, "initial": r["intervals"][0]["state"], "schedule": [period]}
    rr = json.loads(json.dumps(run(restart)))
    assert audit(restart, rr)["checked_steps"] == 2
    assert rr["final"] == r["final"]


def test_uninvolved_liquid_slot_is_preserved_exactly_and_independently_checked():
    from reference_pr_film_mist_exchange import audit

    base, s, law = case()
    raw = base.to_dict()
    raw["phase_slots"] = [*raw["phase_slots"], {"name": "passive_liquid", "root": "liquid"}]
    p = PRActiveMultifluidPipe.from_dict(raw)
    cells = []
    for cell in s.cells:
        c = dict(cell)
        g = c["gas"]
        extra = 0.05 * base.volumes[0]
        c["gas"] = phase(
            p.fluid, g.pressure_pa, g.temperature_k, "vapor", g.volume_m3 - extra, 30.0
        )
        c["passive_liquid"] = phase(p.fluid, g.pressure_pa, 200.0, "liquid", extra)
        cells.append(c)
    before = p.initialize(cells)
    e, d = [1e-7, 0.0], [0.0, 0.0]
    after, r = law.apply(p, before, 0.01, entrainment_m3_s=e, deposition_m3_s=d)
    for ca, cb in zip(before.cells, after.cells):
        assert dict(ca)["passive_liquid"] == dict(cb)["passive_liquid"]
    assert audit(p, before.to_dict(), after.to_dict(), r, law.to_dict(), e, d, 0.01)["accepted"]
