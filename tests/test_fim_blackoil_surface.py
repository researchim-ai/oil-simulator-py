"""Actual three-component reservoir/tree coupling, full Schur, and rollback."""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.blackoil_fim import BlackOilFIMSolver
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.well import WellManager
from surface import HydraulicConvergenceError, PropertyModelError


def config():
    c = json.loads((ROOT / "configs/fim_blackoil_surface.json").read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    return c


def solver(c=None, initial_rs=None):
    c = config() if c is None else c
    r = Reservoir(c["reservoir"], device=torch.device("cpu"))
    f = Fluid(c["fluid"], r, device=torch.device("cpu"))
    if initial_rs is not None:
        f.s_g.zero_()
        f.s_o = 1 - f.s_w
        f.rs = torch.full_like(f.pressure, initial_rs)
    return BlackOilFIMSolver(r, f, WellManager(c["wells"], r), c["simulation"])


def context(s):
    p = s.fluid.pressure.flatten().clone()
    sw = s.fluid.s_w.flatten().clone()
    x = s.x.clone()
    old = s.masses(p, sw, x, s.sat, s.ug)
    s.wells.begin_step(*s.cell_state(p, sw, x, s.sat, s.ug), 0)
    pb = torch.tensor([2.4e6, 2.4e6], dtype=torch.float64)

    def assemble(pp=p, ww=sw, xx=x, pw=pb):
        return s.assemble_bo(pp, ww, xx, s.sat, old, 86400, pw, ug=s.ug)

    return p, sw, x, pb, assemble


@pytest.mark.parametrize("initial_rs", [None, 25.0])
def test_all_component_and_cross_well_jacobian_blocks(initial_rs):
    s = solver(initial_rs=initial_rs)
    p, sw, x, pb, assemble = context(s)
    _, _, ws = assemble()
    B = ws["B_full"].clone()
    D = ws["D_full"].clone()
    assert abs(D[0, 1]) > 0.01 and abs(D[1, 0]) > 0.01
    for j in range(2):
        a = pb.clone()
        b = pb.clone()
        a[j] += 2
        b[j] -= 2
        *_, plus = assemble(pw=a)
        *_, minus = assemble(pw=b)
        assert torch.allclose((plus["r_w"] - minus["r_w"]) / 4, D[:, j], rtol=3e-5, atol=1e-5)
    for cell in (0, 2):
        for k, (base, h) in enumerate(
            zip((p, sw, x), (2.0, 1e-6, 1e-6 if initial_rs is None else 0.001))
        ):
            a = [p.clone(), sw.clone(), x.clone()]
            b = [p.clone(), sw.clone(), x.clone()]
            a[k][cell] = base[cell] + h
            b[k][cell] = base[cell] - h
            *_, plus = assemble(*a)
            *_, minus = assemble(*b)
            expected = B[:, s.wells.conn_cell == cell, k].sum(1)
            assert torch.allclose(
                (plus["r_w"] - minus["r_w"]) / (2 * h), expected, rtol=3e-5, atol=0.003
            )


def test_three_by_three_schur_matches_full_matrix():
    s = solver()
    *_, assemble = context(s)
    R, J, ws = assemble()
    n = s.n_cells
    nb = 3
    nw = s.wells.n_w
    A = torch.zeros((n * nb, n * nb), dtype=torch.float64)
    for row in range(n):
        for pos in range(int(s.builder.indptr[row]), int(s.builder.indptr[row + 1])):
            col = int(s.builder.indices[pos])
            A[nb * row : nb * (row + 1), nb * col : nb * (col + 1)] += J[pos]
    C = A.new_zeros((n * nb, nw))
    B = A.new_zeros((nw, n * nb))
    for k, cell in enumerate(s.wells.conn_cell.tolist()):
        C[nb * cell : nb * (cell + 1), int(s.wells.conn_well[k])] += ws["Cv"][k]
        B[:, nb * cell : nb * (cell + 1)] += ws["B_full"][:, k]
    D = ws["D_full"]
    S = A - C @ torch.linalg.solve(D, B)
    s.wells.add_local_schur(ws, J)
    x = torch.linspace(0.1, 1, n * nb, dtype=torch.float64).view(n, nb)
    y = torch.zeros_like(x)
    for row in range(n):
        for pos in range(int(s.builder.indptr[row]), int(s.builder.indptr[row + 1])):
            y[row] += J[pos] @ x[int(s.builder.indices[pos])]
    y += s.wells.schur_correction(ws, x)
    assert torch.allclose(y.flatten(), S @ x.flatten(), rtol=1e-10, atol=1e-10)
    assert torch.allclose(
        s.wells.recover_dpw(ws, x),
        torch.linalg.solve(D, -ws["r_w"] - B @ x.flatten()),
        rtol=1e-10,
        atol=1e-7,
    )
    full = torch.cat([torch.cat([A, C], 1), torch.cat([B, D], 1)], 0)
    direct = torch.linalg.solve(full, -torch.cat([R, ws["r_w"]]))
    reduced = torch.linalg.solve(S, -R + C @ torch.linalg.solve(D, ws["r_w"]))
    assert torch.allclose(direct[: n * nb], reduced, rtol=1e-8, atol=1e-6)


def report_rates(s):
    return torch.tensor(
        [w["component_mass_rates_kg_s"] for w in s.wells.surface_report["wells"].values()],
        dtype=torch.float64,
    )


def run_case(outlet=1e6, rate=100, bhp=None):
    c = config()
    c["simulation"]["physical_surface_network"]["network"]["nodes"][-1]["pressure_pa"] = outlet
    for w in c["wells"]:
        w["control_value"] = rate
        if bhp is not None:
            w["bhp_min"] = bhp
    s = solver(c)
    old = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
    assert s.step(86400)
    new = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
    error = old - new - report_rates(s).sum(0) * 86400
    assert error.abs().max() < 0.001
    return s


def test_actual_feedback_rate_and_bhp_limits_and_mass_conservation():
    low = run_case(0.8e6)
    high = run_case(1.2e6)
    assert report_rates(low).sum() > report_rates(high).sum()
    assert all(w["mode"] == "surface" for w in low.wells.surface_report["wells"].values())
    capped = run_case(rate=1)
    assert all(w["mode"] == "rate" for w in capped.wells.surface_report["wells"].values())
    assert torch.allclose(
        torch.tensor(
            [
                w["standard_control_rate_m3_s"]
                for w in capped.wells.surface_report["wells"].values()
            ],
            dtype=torch.float64,
        ),
        torch.full((2,), 1 / 86400, dtype=torch.float64),
        rtol=1e-4,
    )
    bounded = run_case(bhp=2.6)
    assert all(w["mode"] == "bhp_limit" for w in bounded.wells.surface_report["wells"].values())


def test_hydraulic_failure_rolls_back(monkeypatch):
    s = run_case()
    p = s.fluid.pressure.clone()
    sw = s.fluid.s_w.clone()
    x = s.x.clone()
    report = s.wells.surface_report

    def fail(_sources):
        raise HydraulicConvergenceError("injected black-oil gathering failure")

    monkeypatch.setattr(s.wells.surface_network, "solve", fail)
    assert not s.step(86400, 86400)
    assert torch.equal(s.fluid.pressure, p) and torch.equal(s.fluid.s_w, sw) and torch.equal(s.x, x)
    assert s.wells.surface_report is report
    assert "injected" in s.last_surface_failure


def test_unsupported_controls_and_incompatible_properties_are_rejected():
    c = config()
    c["simulation"]["physical_surface_network"]["network"]["stock_densities"]["gas_kg_m3"] = 1.3
    with pytest.raises(PropertyModelError, match="densities"):
        solver(c)
    c = config()
    c["wells"][0]["control_type"] = "bhp"
    with pytest.raises(PropertyModelError, match="rate-controlled"):
        solver(c)
    c = config()
    c["wells"][0]["control_value"] = 0
    s = solver(c)
    old = s.fluid.pressure.clone()
    with pytest.raises(PropertyModelError, match="positive rate"):
        s.step(86400)
    assert torch.equal(old, s.fluid.pressure) and s.wells.surface_report is None


def test_reservoir_undersaturated_pvto_matches_surface_values_and_derivatives():
    s = solver(initial_rs=25.0)
    p = torch.tensor([2.4e6, 4.2e6], dtype=torch.float64)
    rs = torch.tensor([25.0, 55.0], dtype=torch.float64)
    b, mu = s.pvt.oil(p, rs, torch.zeros(2, dtype=torch.bool))
    model = s.wells.surface_network.model
    for j in range(2):
        state = model.state(float(p[j]), [0.85, 0, float(rs[j]) * 0.001 * 1.2])
        assert 1 / float(b[j]) == pytest.approx(
            state["phase_volume_rates_m3_s"][0] / 0.001, rel=1e-12
        )
        assert float(mu[j]) == pytest.approx(state["viscosity_pa_s"], rel=1e-12)
    from torch.func import jvp

    def values(pp, rr):
        bo, m = s.pvt.oil(pp, rr, torch.zeros(2, dtype=torch.bool))
        return torch.stack([bo, m])

    for variable, h in [(0, 1.0), (1, 1e-4)]:
        tangent = [torch.zeros_like(p), torch.zeros_like(rs)]
        tangent[variable].fill_(1.0)
        _, grad = jvp(values, (p, rs), tuple(tangent))
        plus = [p.clone(), rs.clone()]
        minus = [p.clone(), rs.clone()]
        plus[variable] += h
        minus[variable] -= h
        fd = (values(*plus) - values(*minus)) / (2 * h)
        assert torch.allclose(grad, fd, rtol=1e-6, atol=1e-12)


def test_undersaturated_coupled_interval_conserves_all_components():
    s = solver(initial_rs=25.0)
    old = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
    assert s.step(86400)
    new = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug).sum(1)
    assert (old - new - report_rates(s).sum(0) * 86400).abs().max() < 0.001


def test_warm_boundary_change_and_return():
    s = run_case()
    s.wells.set_surface_boundary_pressures({"outlet": 1.2e6})
    assert s.step(86400, 86400)
    higher = report_rates(s).sum()
    s.wells.set_surface_boundary_pressures({"outlet": 0.8e6})
    assert s.step(86400, 172800)
    assert report_rates(s).sum() > higher
    with pytest.raises(ValueError, match="receiver"):
        s.wells.set_surface_boundary_pressures({"bottom_1": 1e6})


def test_extended_pvto_with_different_pressure_axes_is_interpolated_exactly():
    import numpy as np

    from simulator.blackoil_props import BlackOilPVT
    from simulator.pvt import PVTTable

    s = solver()
    data = json.loads((ROOT / "configs/pvt/pvt_surface_coupled.json").read_text())
    table = data["extended"]["oil"]["tables"]["20.0"]
    old_axis = table["pressure_MPa"][:]
    new_axis = [0.1, 1.0, 2.0, 5.0]
    for key in ("Bo", "mu_o_cP"):
        table[key] = np.interp(new_axis, old_axis, table[key]).tolist()
    table["pressure_MPa"] = new_axis
    s.fluid.pvt = PVTTable(data)
    pvt = BlackOilPVT(s.fluid, torch.device("cpu"))
    p = torch.tensor([2.4e6, 3.3e6, 4.2e6], dtype=torch.float64)
    rs = torch.tensor([25.0, 35.0, 55.0], dtype=torch.float64)
    b, mu = pvt.oil(p, rs, torch.zeros_like(p, dtype=torch.bool))
    expected = s.fluid.pvt.eval_oil(p.numpy() / 1e6, rs.numpy())
    assert np.allclose(1 / b.numpy(), expected["Bo"], rtol=1e-14)
    assert np.allclose(mu.numpy(), expected["mu_o"] * 1e-3, rtol=1e-14)


@pytest.mark.parametrize("field", ["Bo", "pressure_MPa"])
def test_invalid_extended_pvto_is_rejected(field):
    from simulator.blackoil_props import BlackOilPVT
    from simulator.pvt import PVTTable

    s = solver()
    data = json.loads((ROOT / "configs/pvt/pvt_surface_coupled.json").read_text())
    table = data["extended"]["oil"]["tables"]["20.0"]
    if field == "Bo":
        table[field][1] = -1.0
    else:
        table[field][1] = table[field][0]
    s.fluid.pvt = PVTTable(data)
    with pytest.raises(ValueError, match="Invalid extended PVTO"):
        BlackOilPVT(s.fluid, torch.device("cpu"))


@pytest.mark.parametrize("mixing", [True, False])
def test_optimized_producer_path_rejects_reverse_perforation_flow(mixing):
    c = config()
    c["wells"][0]["perforations"] = [[0, 0, 0], [1, 0, 0]]
    s = solver(c)
    p, sw, x, pb, _ = context(s)
    s.wells.wellbore_mixing = mixing
    p[1] = 2.0e6  # One completion receives fluid while the other produces.
    state = s.cell_state(p, sw, x, s.sat, s.ug)
    old = s.masses(p, sw, x, s.sat, s.ug)
    error = PropertyModelError if mixing else HydraulicConvergenceError
    match = "perforation crossflow" if mixing else "producing-flow domain"
    with pytest.raises(error, match=match):
        s.assemble_bo(p, sw, x, s.sat, old, 86400, pb, state, s.ug)
