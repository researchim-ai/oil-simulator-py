"""Пороговые давления между регионами равновесия (EQLOPTS THPRES, THPRES)."""
import os
import re
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.blackoil_fim import BlackOilFIMSolver, I_ACC  # noqa: E402
from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")


def _deck(tmp_path, thpres, eqlopts="'THPRES'", name="T.DATA"):
    """SPE1: EQLNUM 1 — верхний слой (нагнетательная), 2 — слои 2–3 (добывающая)."""
    text = open(SPE1).read()
    for kw in ("EQUIL", "RSVD"):
        m = re.search(rf"^{kw}\s*$", text, re.M)
        nxt = re.search(r"^[A-Z][A-Z0-9]{1,7}\s*$", text[m.end():], re.M)
        a, b = m.end(), m.end() + nxt.start()
        text = text[:a] + text[a:b].rstrip() + "\n" + text[a:b] + text[b:]
    if eqlopts is not None:
        text = text.replace("\nGRID", f"\nEQLOPTS\n {eqlopts} /\nGRID", 1)
    text = text.replace("\nSOLUTION", "\nREGIONS\nEQLNUM\n 100*1 200*2 /\nSOLUTION", 1)
    if thpres is not None:
        text = text.replace("\nSUMMARY", f"\nTHPRES\n{thpres}/\nSUMMARY", 1)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_zero_threshold_reproduces_no_threshold(tmp_path):
    base, _ = run_deck(_deck(tmp_path, None, eqlopts=None, name="B.DATA"), max_days=300)
    zero, _ = run_deck(_deck(tmp_path, " 1 2 0.0 /\n"), max_days=300)
    for key in ("FOPR", "FGPR", "WBHP:PROD", "BPR:1,1,1"):
        assert np.array_equal(zero[key][1:], base[key][1:]), key


def test_large_threshold_isolates_regions(tmp_path):
    base, _ = run_deck(_deck(tmp_path, None, eqlopts=None, name="B.DATA"), max_days=300)
    iso, _ = run_deck(_deck(tmp_path, " 1 2 100000 /\n"), max_days=300)
    # газ остаётся в верхнем слое: давление там растёт сильнее, у добывающей — падает быстрее
    assert iso["BPR:1,1,1"][-1] > base["BPR:1,1,1"][-1] + 100.0
    assert iso["WBHP:PROD"][-1] < base["WBHP:PROD"][-1] - 100.0


def _solver_state(case):
    res, fluid, wm, params = case.build()
    s = BlackOilFIMSolver(res, fluid, wm, params)
    p = s.fluid.pressure.reshape(-1).double().clone()
    sw = s.fluid.s_w.reshape(-1).double().clone()
    return s, p, sw, s.x.clone(), s.sat.clone()


def test_default_threshold_from_equilibrium(tmp_path):
    s, p, sw, x, sat = _solver_state(EclipseCase(_deck(tmp_path, " 1 2 /\n")))
    th_lr, th_rl = s._face_thresholds(s.cell_state(p, sw, x, sat)[0])
    assert torch.equal(th_lr, th_rl)
    assert float(th_lr.max()) < 1e3   # равновесие: разности потенциалов через границу малы, Па


def test_thpres_requires_eqlopts(tmp_path):
    with pytest.raises(ValueError, match="EQLOPTS"):
        EclipseCase(_deck(tmp_path, " 1 2 1.0 /\n", eqlopts=None)).build()


def test_thpres_jacobian_matches_finite_differences(tmp_path):
    s, p, sw, x, sat = _solver_state(EclipseCase(_deck(tmp_path, " 1 2 15 /\n")))
    # перепад между слоями 1 и 2 около порога: часть граней сверх него, часть — ниже
    k = torch.arange(s.n_cells) % s.nz
    p = p + torch.where(k == 0, 1.0e5 * (1.0 + 0.5 * torch.sin(torch.arange(s.n_cells, dtype=torch.float64))), 0.0)
    acc_old = s.cell_values(p * 0.999, sw, x, sat)[I_ACC:I_ACC + 3]
    st = s.cell_state(p, sw, x, sat)
    s.wells.begin_step(st[0], st[1], 0.0)
    p_bh = s.wells.p_bh.clone() if s.wells.p_bh is not None else torch.full((s.wells.n_w,), 2.0e7, dtype=torch.float64)
    s._th_cache = None
    R0, J, _ = s.assemble_bo(p, sw, x, sat, acc_old, 86400.0, p_bh)
    th = s._face_thresholds(st[0])
    assert float(th[0].max()) == pytest.approx(15 * 6894.757293168)
    rows = torch.repeat_interleave(torch.arange(s.n_cells), s.builder.indptr[1:] - s.builder.indptr[:-1])
    cols = s.builder.indices
    worst = 0.0
    for c in range(0, s.n_cells, 3):
        h = max(1.0, 1e-7 * float(p[c]))
        p2 = p.clone()
        p2[c] += h
        R1, _, _ = s.assemble_bo(p2, sw, x, sat, acc_old, 86400.0, p_bh)
        fd = (R1 - R0).view(-1, 3) / h
        an = torch.zeros_like(fd)
        m = cols == c
        an[rows[m]] = J[m][:, :, 0]
        scale = an.abs().max().clamp_min(1e-30)
        worst = max(worst, float((fd - an).abs().max() / scale))
    assert worst < 1e-4
