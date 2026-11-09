"""Conservative material/thermal splitting, not a complete momentum coupling."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_surface_pr_active_multifluid_pipe import phase, setup

from surface.liquid import ConservationError
from surface.pr_active_multifluid_pipe import PRActiveMultifluidPipe
from surface.pr_phase_volume_transport import remap_phase_volumes, transfer_film_increment
from surface.thin_film_dynamics import ThinFilmDynamics


def state_with_phases(*, absent=False):
    pipe, v = setup(mode="periodic", count=4)
    p = pipe.fluid.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cells = []
    for i in range(4):
        alpha = 0.0 if absent and i == 0 else 0.02 + 0.002 * i
        cell = {"gas": phase(pipe.fluid, p, 200.0, "vapor", v * (1 - alpha), 0.1 * i)}
        if alpha:
            cell["liquid"] = phase(pipe.fluid, p, 200.0, "liquid", v * alpha, -0.01 * i)
        cells.append(cell)
    return pipe, pipe.initialize(cells)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_parcel_flux_carries_identical_donor_fraction_of_all_pr_stocks(sign):
    pipe, old = state_with_phases()
    q = np.array([1e-7, -0.5e-7, 0.2e-7, -0.4e-7]) * sign
    end, r = remap_phase_volumes(pipe, old, 0.01, {"liquid": q, "gas": -q})
    for j, name in enumerate(pipe.names):
        stocks = np.array(
            [
                [
                    *dict(c)[name].component_moles,
                    dict(c)[name].volume_m3,
                    dict(c)[name].momentum_kg_m_s,
                    dict(c)[name].total_energy_j,
                ]
                for c in old.cells
            ]
        )
        flux = q if name == "liquid" else -q
        expected = []
        for i, v in enumerate(flux):
            donor = stocks[i if v >= 0 else (i + 1) % 4]
            expected.append(donor * v / donor[2])
        expected = np.array(expected)
        updated = stocks - 0.01 * (expected - np.roll(expected, 1, 0))
        actual = np.array(
            [
                [
                    *dict(c)[name].component_moles,
                    dict(c)[name].volume_m3,
                    dict(c)[name].momentum_kg_m_s,
                    dict(c)[name].total_energy_j,
                ]
                for c in end.cells
            ]
        )
        np.testing.assert_allclose(actual, updated, rtol=1e-13, atol=1e-18)
        np.testing.assert_allclose(
            np.array(r["face_stock_fluxes"])[:, j, :], expected, rtol=1e-13, atol=1e-18
        )
    assert end.time_s == old.time_s
    assert max(map(abs, r["conservation_residuals"])) < 1e-10


def test_birth_in_absent_receiver_has_real_donor_and_invalid_outflow_is_atomic():
    from surface.liquid import HydraulicConvergenceError, PropertyModelError

    pipe, s = state_with_phases(absent=True)
    old = s.to_dict()
    q = np.array([-1e-6, 0.0, 0.0, 0.0])
    end, _ = remap_phase_volumes(pipe, s, 0.01, {"liquid": q, "gas": -q})
    assert dict(end.cells[0])["liquid"].volume_m3 > 0
    with pytest.raises(PropertyModelError, match="Absent"):
        remap_phase_volumes(pipe, s, 0.01, {"liquid": -q, "gas": q})
    with pytest.raises(HydraulicConvergenceError, match="inventory"):
        remap_phase_volumes(pipe, s, 10.0, {"liquid": q, "gas": -q})
    assert s.to_dict() == old


def film_setup(device="cpu"):
    base, _ = setup(mode="periodic", count=16)
    raw = base.to_dict()
    raw["diameter_m"] = 0.002
    raw["lengths_m"] = [0.001 / 16] * 16
    pipe = PRActiveMultifluidPipe.from_dict(raw, device=device)
    film = ThinFilmDynamics(
        (16,),
        (0.001,),
        0.02,
        0.001,
        600.0,
        1e-6,
        5.0,
        1e-8,
        transverse_width_m=math.pi * 0.002,
        device=device,
    )
    h = torch.tensor(
        [3e-6 * (1 + 0.2 * math.cos(2 * math.pi * i / 16)) for i in range(16)], dtype=torch.float64
    )
    fs = film.initialize(h)
    p = pipe.fluid.components[1].saturation_at_temperature(200.0)["pressure_pa"]
    cells = []
    for i, height in enumerate(h.tolist()):
        vl = height * film.cell_area
        cells.append(
            {
                "liquid": phase(pipe.fluid, p, 200.0, "liquid", vl),
                "gas": phase(pipe.fluid, p, 200.0, "vapor", pipe.volumes[i] - vl),
            }
        )
    return pipe, pipe.initialize(cells), film, fs


def test_film_material_and_surface_energy_reach_actual_pr_stocks_once():
    from reference_pr_active_multifluid import packets

    pipe, old, film, fs = film_setup()
    initial = pipe._totals(old)
    initial[-2] += float(film.free_energy(film.tensor(fs.thickness_m)).cpu())
    end, newfs, r = transfer_film_increment(
        pipe, old, film, fs, 0.02, liquid_slot="liquid", gas_slot="gas"
    )
    final = pipe._totals(end)
    final[-2] += float(film.free_energy(film.tensor(newfs.thickness_m)).cpu())
    np.testing.assert_allclose(final, initial, atol=1e-18, rtol=1e-11)
    assert max(abs(a - b) for a, b in zip(newfs.thickness_m, fs.thickness_m)) > 0
    assert sum(r["phase_heat_j"]) == pytest.approx(r["film"]["dissipation_j"], rel=1e-14)
    assert abs(r["joint_energy_residual_j"]) < 1e-18
    # Independent scalar PR packets validate recovered T/P/E/V, not runtime UV.
    packets(pipe, end.to_dict())
    for a, b in zip(old.cells, end.cells):
        assert dict(b)["liquid"].temperature_k >= dict(a)["liquid"].temperature_k - 1e-10
    assert end.time_s == newfs.time_s == 0.02
    bad = replace(fs, thickness_m=tuple(v * 1.1 for v in fs.thickness_m))
    with pytest.raises(ConservationError, match="stock"):
        transfer_film_increment(pipe, old, film, bad, 0.02, liquid_slot="liquid", gas_slot="gas")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_transfer_cuda_parity():
    p, s, f, fs = film_setup()
    pg, sg, fg, fsg = film_setup("cuda")
    a, af, _ = transfer_film_increment(p, s, f, fs, 0.02, liquid_slot="liquid", gas_slot="gas")
    b, bf, _ = transfer_film_increment(pg, sg, fg, fsg, 0.02, liquid_slot="liquid", gas_slot="gas")
    torch.testing.assert_close(
        f.tensor(af.thickness_m), f.tensor(bf.thickness_m), atol=1e-18, rtol=1e-10
    )
    for ca, cb in zip(a.cells, b.cells):
        for (_, u), (_, v) in zip(ca, cb):
            assert v.temperature_k == pytest.approx(u.temperature_k, abs=1e-9)
            assert v.pressure_pa == pytest.approx(u.pressure_pa, rel=1e-9)


def test_independent_full_transfer_audit_restart_and_corrupted_heat_detection():
    import json
    from copy import deepcopy

    from check_pr_film_transfer import audit
    from run_pr_film_transfer import run

    p, s, f, fs = film_setup()
    c = json.loads(
        json.dumps(
            {
                "pipe": p.to_dict(),
                "film": f.to_dict(),
                "initial": s.to_dict(),
                "initial_film": fs.to_dict(),
                "duration_s": 0.04,
                "maximum_step_s": 0.02,
                "liquid_slot": "liquid",
                "gas_slot": "gas",
            }
        )
    )
    r = json.loads(json.dumps(run(c)))
    assert audit(c, r)["checked_steps"] == 2
    restart = {
        **c,
        "initial": r["steps"][0]["state"],
        "initial_film": r["steps"][0]["film_state"],
        "duration_s": 0.02,
    }
    rr = json.loads(json.dumps(run(restart)))
    assert audit(restart, rr)["checked_steps"] == 1
    assert rr["final"] == r["final"]
    assert rr["final_film"] == r["final_film"]
    bad = deepcopy(r)
    bad["steps"][0]["phase_heat_j"][0] += 1e-8
    with pytest.raises(ValueError, match="heat"):
        audit(c, bad)
