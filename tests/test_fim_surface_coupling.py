"""Full cross-well Newton/Schur blocks and actual reservoir/surface feedback."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulator.fluid import Fluid
from simulator.manual_assembly_solver import ManualFIMSolver
from simulator.reservoir import Reservoir
from simulator.well import WellManager
from surface import HydraulicConvergenceError, PropertyModelError

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/fim_water_surface.json").read_text())


def solver(cfg=None):
    cfg = config() if cfg is None else cfg
    device = torch.device("cpu")
    r = Reservoir(cfg["reservoir"], device=device)
    f = Fluid(cfg["fluid"], r, device=device)
    return ManualFIMSolver(r, f, WellManager(cfg["wells"], r), cfg["simulation"])


def assemble_context(s):
    p = s.fluid.pressure.reshape(-1).double()
    sw = s.fluid.s_w.reshape(-1).double()
    old = s.calc_masses(p, sw)
    s.wells.begin_step(p, sw, s.cell_props(p, sw), 0)
    pb = torch.tensor([18.4e6, 18.5e6], dtype=torch.float64)

    def assemble(pp=p, ss=sw, pw=pb):
        R, J = s.assemble_system(pp, ss, *old, 86400, p_bh=pw)
        return R, J, s._last_well_sys

    return p, sw, pb, assemble


def test_all_well_and_reservoir_cross_blocks_match_finite_differences():
    s = solver()
    p, _sw, pb, assemble = assemble_context(s)
    _R, _J, sys = assemble()
    B = sys["B_full"].clone()
    D = sys["D_full"].clone()
    C = sys["Cv"].clone()
    assert abs(D[0, 1]) > 0.01 and abs(D[1, 0]) > 0.01
    h = 10.0
    for j in range(2):
        plus = pb.clone()
        minus = pb.clone()
        plus[j] += h
        minus[j] -= h
        Rp, _, sp = assemble(pw=plus)
        rm_plus = sp["r_w"].clone()
        Rm, _, sm = assemble(pw=minus)
        assert torch.allclose((rm_plus - sm["r_w"]) / (2 * h), D[:, j], rtol=3e-7, atol=1e-8)
        expected = torch.zeros((s.n_cells, 2), dtype=torch.float64)
        mask = s.wells.conn_well == j
        expected.index_add_(0, s.wells.conn_cell[mask], C[mask])
        assert torch.allclose(((Rp - Rm) / (2 * h)).view(-1, 2), expected, rtol=1e-7, atol=1e-12)
    for cell in (0, 2):
        pp = p.clone()
        pm = p.clone()
        pp[cell] += h
        pm[cell] -= h
        _, _, sp = assemble(pp=pp)
        rplus = sp["r_w"].clone()
        _, _, sm = assemble(pp=pm)
        expected = B[:, s.wells.conn_cell == cell, 0].sum(1)
        assert torch.allclose((rplus - sm["r_w"]) / (2 * h), expected, rtol=3e-7, atol=1e-8)


@pytest.mark.parametrize("shared_connections", [False, True])
def test_full_schur_operator_and_recovery_match_explicit_dense_system(shared_connections):
    cfg = config()
    if shared_connections:
        cfg["wells"][0]["perforations"] = [[0, 0, 0], [1, 0, 0]]
        cfg["wells"][1]["perforations"] = [[1, 0, 0], [2, 0, 0]]
    s = solver(cfg)
    _p, _sw, _pb, assemble = assemble_context(s)
    R, J, ws = assemble()
    n = s.n_cells
    nb = 2
    nw = s.wells.n_w
    A = torch.zeros((n * nb, n * nb), dtype=torch.float64)
    for row in range(n):
        for pos in range(int(s.builder.indptr[row]), int(s.builder.indptr[row + 1])):
            col = int(s.builder.indices[pos])
            A[2 * row : 2 * row + 2, 2 * col : 2 * col + 2] += J[pos]
    C = A.new_zeros((n * nb, nw))
    B = A.new_zeros((nw, n * nb))
    for k, cell in enumerate(s.wells.conn_cell.tolist()):
        C[2 * cell : 2 * cell + 2, int(s.wells.conn_well[k])] += ws["Cv"][k]
        B[:, 2 * cell : 2 * cell + 2] += ws["B_full"][:, k]
    D = ws["D_full"]
    S = A - C @ torch.linalg.solve(D, B)
    original = J.clone()
    s.wells.add_local_schur(ws, J)
    x = torch.linspace(0.1, 1, n * nb, dtype=torch.float64).view(n, nb)
    y = torch.zeros_like(x)
    for row in range(n):
        for pos in range(int(s.builder.indptr[row]), int(s.builder.indptr[row + 1])):
            y[row] += J[pos] @ x[int(s.builder.indices[pos])]
    y += s.wells.schur_correction(ws, x, local_included=True)
    assert torch.allclose(y.reshape(-1), S @ x.reshape(-1), rtol=1e-10, atol=1e-12)
    assert torch.allclose(
        s.wells.schur_rhs(ws, n).reshape(-1),
        C @ torch.linalg.solve(D, ws["r_w"]),
        rtol=1e-10,
        atol=1e-12,
    )
    recovered = s.wells.recover_dpw(ws, x)
    assert torch.allclose(
        recovered, torch.linalg.solve(D, -ws["r_w"] - B @ x.reshape(-1)), rtol=1e-10, atol=1e-9
    )
    full = torch.cat([torch.cat([A, C], 1), torch.cat([B, D], 1)], 0)
    rhs = -torch.cat([R, ws["r_w"]])
    solution = torch.linalg.solve(full, rhs)
    reduced = torch.linalg.solve(S, -R + C @ torch.linalg.solve(D, ws["r_w"]))
    assert torch.allclose(solution[: n * nb], reduced, rtol=1e-8, atol=1e-7)
    assert not torch.equal(original, J)


def run_case(outlet_mpa=8, rate=100, bhp_min=None):
    cfg = config()
    cfg["simulation"]["physical_surface_network"]["network"]["nodes"][-1]["pressure_pa"] = (
        outlet_mpa * 1e6
    )
    for w in cfg["wells"]:
        w["control_value"] = rate
        if bhp_min is not None:
            w["bhp_min"] = bhp_min
    s = solver(cfg)
    assert s.step(86400)
    return s


def total_rate(s):
    return sum(v["water_mass_rate_kg_s"] for v in s.wells.surface_report["wells"].values())


def test_reservoir_surface_feedback_rate_cap_bhp_bound_and_stop():
    low = run_case(6)
    normal = run_case(8)
    high = run_case(10)
    assert total_rate(low) > total_rate(normal) > total_rate(high) > 0
    for s in (low, normal, high):
        report = s.wells.surface_report
        assert total_rate(s) == pytest.approx(
            -report["network"]["boundary_mass_rates_kg_s"]["outlet"], abs=1e-9
        )
        assert max(map(abs, report["network"]["node_mass_residuals_kg_s"].values())) < 1e-9
        assert all(row["mode"] == "surface" for row in report["wells"].values())
    limited = run_case(8, rate=5)
    assert all(row["mode"] == "rate" for row in limited.wells.surface_report["wells"].values())
    assert total_rate(limited) == pytest.approx(2 * 5 / 86400 * 1000, rel=1e-5)
    bounded = run_case(8, bhp_min=19)
    assert all(row["mode"] == "bhp_limit" for row in bounded.wells.surface_report["wells"].values())
    stopped = run_case(12)
    assert abs(total_rate(stopped)) < 1e-8
    assert all(
        row["mode"] == "surface_stop" for row in stopped.wells.surface_report["wells"].values()
    )


def test_unsupported_mobile_oil_is_rejected_and_rolls_back():
    cfg = config()
    cfg["fluid"]["s_w"] = 0.4
    s = solver(cfg)
    p = s.fluid.pressure.clone()
    sw = s.fluid.s_w.clone()
    with pytest.raises(PropertyModelError, match="water-only"):
        s.step(86400)
    assert torch.equal(s.fluid.pressure, p) and torch.equal(s.fluid.s_w, sw)
    assert s.wells.surface_report is None


def test_surface_numerical_failure_requests_substep_retry_and_preserves_state(monkeypatch):
    s = run_case()
    p = s.fluid.pressure.clone()
    pb = s.wells.p_bh.clone()
    report = copy.deepcopy(s.wells.surface_report)

    def fail(**kwargs):
        raise HydraulicConvergenceError("injected surface solver failure")

    monkeypatch.setattr(s.wells.surface_network, "solve", fail)
    assert not s.step(86400)
    assert "injected" in s.last_surface_failure
    assert torch.equal(s.fluid.pressure, p) and torch.equal(s.wells.p_bh, pb)
    assert s.wells.surface_report == report


def test_surface_backpressure_can_stop_and_restart_warm_wells():
    s = run_case(8)
    s.wells.set_surface_boundary_pressures({"outlet": 12e6})
    assert s.step(86400, current_time=86400)
    assert abs(total_rate(s)) < 1e-8
    s.wells.set_surface_boundary_pressures({"outlet": 8e6})
    assert s.step(86400, current_time=2 * 86400)
    assert total_rate(s) > 0
    assert all(row["mode"] == "surface" for row in s.wells.surface_report["wells"].values())


def test_binding_rejects_wrong_reference_height_and_blackoil():
    from simulator.blackoil_fim import BlackOilFIMSolver

    cfg = config()
    cfg["simulation"]["physical_surface_network"]["network"]["nodes"][0]["elevation_m"] = 0
    with pytest.raises(ValueError, match="reference depth"):
        solver(cfg)
    s = solver()
    with pytest.raises(NotImplementedError, match="oil_water"):
        BlackOilFIMSolver(s.reservoir, s.fluid, s.well_manager, config()["simulation"])
