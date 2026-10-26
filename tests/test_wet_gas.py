"""Жирный газ (VAPOIL / PVTG, Rv) на SPE1: PVDG заменён на PVTG."""
import contextlib
import io
import os
import re
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck, apply_controls  # noqa: E402
from simulator.simulation import Simulator  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
DAY = 86400.0


def _wet_deck(tmp_path, rv_per_psi, vapoil=True, vappars=None, name="WET.DATA"):
    """PVTG из PVDG SPE1: Rv_sat = rv_per_psi·P [STB/Mscf], недонасыщенная ветвь до Rv = 0 (Bg +0.5%, μg −2%)."""
    text = open(SPE1).read()
    m = re.search(r"^PVDG\s*$", text, re.M)
    nxt = re.search(r"^[A-Z][A-Z0-9]{1,7}\s*$", text[m.end():], re.M)
    body = text[m.end():m.end() + nxt.start()]
    rows = [list(map(float, ln.split()[:3])) for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("--") and re.match(r"^\s*[\d.]", ln)]
    recs = []
    for p, bg, mu in rows:
        rv = rv_per_psi * p
        if rv > 0:
            recs.append(f" {p} {rv:.8f} {bg} {mu}\n     0.0 {bg * 1.005} {mu * 0.98} /")
        else:
            recs.append(f" {p} 0.0 {bg} {mu} /")
    pvtg = "PVTG\n" + "\n".join(recs) + "\n/\n\n"
    text = text[:m.start()] + pvtg + text[m.end() + nxt.start():]
    if vapoil:
        text = text.replace("\nDISGAS", "\nDISGAS\nVAPOIL", 1)
    if vappars is not None:
        # после DRSDT: из двух действует заданный последним
        m = re.search(r"^DRSDT\s*\n[^/]*/[^\n]*\n", text, re.M)
        text = text[:m.end()] + f"VAPPARS\n {vappars} /\n" + text[m.end():]
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_zero_rv_reproduces_dry_gas(tmp_path):
    base, _ = run_deck(SPE1, max_days=300)
    wet, _ = run_deck(_wet_deck(tmp_path, 0.0), max_days=300)
    for key in ("FOPR", "FGPR", "WBHP:PROD", "BPR:10,10,3", "BGSAT:10,10,1"):
        assert np.allclose(wet[key][1:], base[key][1:], rtol=1e-5, atol=1e-12), key


def test_pvto_saturated_branch_is_continuous():
    _, fluid, _, _ = EclipseCase(SPE1).build()
    pvt = fluid.blackoil_pvt
    p = torch.linspace(1e6, 4e7, 500, dtype=torch.float64)
    rs_s = pvt.rs_sat(p)
    yes = torch.ones_like(p, dtype=torch.bool)
    b_sat, mu_sat = pvt.oil(p, rs_s, yes)
    b_us, mu_us = pvt.oil(p, rs_s, ~yes)
    assert torch.equal(b_sat, b_us) and torch.equal(mu_sat, mu_us)


def test_runs_are_bitwise_reproducible():
    a, _ = run_deck(SPE1, max_days=150)
    b, _ = run_deck(SPE1, max_days=150)
    for key in ("FOPR", "FGPR", "WBHP:PROD"):
        assert np.array_equal(a[key][1:], b[key][1:]), key


def test_pvtg_saturated_branch_is_continuous(tmp_path):
    case = EclipseCase(_wet_deck(tmp_path, 4e-5))
    _, fluid, _, _ = case.build()
    pvt = fluid.blackoil_pvt
    assert pvt.wet
    p = torch.linspace(2e6, 3.5e7, 200, dtype=torch.float64)
    rv_s = pvt.rv_sat(p)
    b_sat, mu_sat = pvt.gas(p, rv_s, torch.ones_like(p, dtype=torch.bool))
    b_us, mu_us = pvt.gas(p, rv_s, torch.zeros_like(p, dtype=torch.bool))
    # точное совпадение ветвей на линии насыщения: иначе переключения состояния дают скачки массы
    assert torch.equal(b_sat, b_us) and torch.equal(mu_sat, mu_us)
    b0, _ = pvt.gas(p, 0.0 * p, torch.zeros_like(p, dtype=torch.bool))
    assert torch.all(b0 < b_sat)  # меньше конденсата — больше Bg (по таблице +0.5%)


def _run(path, days):
    case = EclipseCase(path)
    with contextlib.redirect_stdout(io.StringIO()):
        res, fluid, wm, prm = case.build()
        sim = Simulator(res, fluid, wm, prm, device=torch.device("cpu"))
    s = sim.fim_solver

    def mass():
        p, sw = fluid.pressure.reshape(-1).double(), fluid.s_w.reshape(-1).double()
        return s.cell_values(p, sw, s.x, s.sat, s.ug)[0:3].sum(1)

    m0 = mass()
    t, n_ug = 0.0, 0
    for n, d in enumerate(case.report_days):
        if t + d > days:
            break
        apply_controls(case, n, wm, s)
        with contextlib.redirect_stdout(io.StringIO()):
            assert sim.run_step(d * DAY)
        t += d
        n_ug = max(n_ug, int(s.ug.sum()))
    rho = s.pvt.rho_sc
    cum = np.zeros(3)
    for w in wm.wells:
        c = w.cum_surface
        cum += np.array([c["oil"] * rho[0], c["water"] * rho[1], c["gas"] * rho[2]])
    return m0.numpy(), mass().numpy(), cum, n_ug, s


def test_wet_gas_mass_balance_and_vaporization(tmp_path):
    m0, m1, cum, n_ug, s = _run(_wet_deck(tmp_path, 1e-4), 1500)
    for comp in (0, 2):
        assert abs((m0[comp] - m1[comp]) - cum[comp]) <= 1e-5 * max(abs(m0[comp]), 1.0), comp
    assert n_ug > 0  # закачанный сухой газ испаряет нефть: ячейки без нефти (X = Rv)
    rv = s.fluid.rv.reshape(-1)
    assert float(rv.max()) > 0 and float(rv.min()) >= 0


_NO_DRSDT_WET = pytest.mark.skip(reason="синтетическая колода некорректна без DRSDT (OPM Flow тоже закрывает INJ "
                                         "на t = 0); VAPOIL+VAPPARS без DRSDT проверяет "
                                         "test_eclipse_benchmarks::test_vappars_wet_gas_matches_opm")


@_NO_DRSDT_WET
def test_vappars_zero_is_neutral_and_factor_formula(tmp_path):
    base, _ = run_deck(_wet_deck(tmp_path, 1e-4), max_days=300)
    zero, _ = run_deck(_wet_deck(tmp_path, 1e-4, vappars="0 0", name="V0.DATA"), max_days=300)
    for key in ("FOPR", "FGPR", "WBHP:PROD"):
        assert np.array_equal(zero[key][1:], base[key][1:]), key
    from simulator.blackoil_fim import BlackOilFIMSolver
    res, fluid, wm, params = EclipseCase(_wet_deck(tmp_path, 1e-4, vappars="0.5 0", name="V1.DATA")).build()
    s = BlackOilFIMSolver(res, fluid, wm, params)
    p = torch.full((s.n_cells,), 2.5e7, dtype=torch.float64)
    s.so_max = torch.full_like(p, 0.8)
    so = torch.linspace(0.0, 0.9, s.n_cells, dtype=torch.float64)
    rv = s.rv_sat_eff(p, so)
    expect = s.pvt.rv_sat(p) * torch.where(so < 0.8, (so.clamp_min(1e-3) / 0.8) ** 0.5, torch.ones_like(so))
    assert torch.allclose(rv, expect, rtol=1e-14)


@_NO_DRSDT_WET
def test_vappars_reduces_vaporization(tmp_path):
    m0, m1, cum, n_ug, s = _run(_wet_deck(tmp_path, 1e-4, vappars="0.5 0", name="V2.DATA"), 1500)
    for comp in (0, 2):
        assert abs((m0[comp] - m1[comp]) - cum[comp]) <= 1e-5 * max(abs(m0[comp]), 1.0), comp
    rv = s.fluid.rv.reshape(-1)
    p = s.fluid.pressure.reshape(-1).double()
    assert torch.all(rv <= s.pvt.rv_sat(p) * (1 + 1e-10))
    _, _, _, _, s0 = _run(_wet_deck(tmp_path, 1e-4, name="V3.DATA"), 1500)
    assert float(rv.sum()) < float(s0.fluid.rv.reshape(-1).sum())
