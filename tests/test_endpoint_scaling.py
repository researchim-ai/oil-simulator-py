"""Масштабирование концевых точек (ENDSCALE), SWATINIT и гистерезис ОФП (SATOPTS HYSTER)."""
import os
import re
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402
from simulator.endpoint_scaling import piecewise_map  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data")
SPE1 = os.path.join(DATA, "spe1", "SPE1CASE1.DATA")
SPE9 = os.path.join(DATA, "spe9", "SPE9_CP.DATA")


def _deck(tmp_path, src, runspec="", props="", name="X.DATA"):
    text = open(src).read()
    text = text.replace("\nGRID", "\n" + runspec + "GRID", 1)
    anchor = "REGIONS" if "\nREGIONS" in text else "SOLUTION"
    text = text.replace("\n" + anchor, "\n" + props + anchor, 1)
    if "INCLUDE" in text:
        for f in os.listdir(os.path.dirname(src)):
            if f.upper().endswith((".GRDECL", ".INC")) or f.startswith("PERM"):
                shutil.copy(os.path.join(os.path.dirname(src), f), tmp_path / f)
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def test_piecewise_map_is_monotone_and_hits_nodes():
    s = torch.linspace(0, 1, 101, dtype=torch.float64)
    a = torch.full_like(s, 0.2)
    b = torch.full_like(s, 0.8)
    out = piecewise_map(s, [a, b], [0.1, 0.9])
    assert torch.all(out[1:] >= out[:-1])
    assert abs(float(out[20]) - 0.1) < 1e-12 and abs(float(out[80]) - 0.9) < 1e-12


@pytest.mark.parametrize("scalecrs", ["", "SCALECRS\n YES /\n"])
def test_endscale_without_arrays_reproduces_tables(tmp_path, scalecrs):
    base, _ = run_deck(SPE1, max_days=200)
    sc, _ = run_deck(_deck(tmp_path, SPE1, "ENDSCALE\n/\n" + scalecrs), max_days=200)
    for key in ("FOPR", "FGPR", "WBHP:PROD", "BPR:10,10,3", "BGSAT:10,10,1"):
        assert np.allclose(sc[key][1:], base[key][1:], rtol=1e-8, atol=1e-12), key


def test_swl_shifts_initial_water_and_krw(tmp_path):
    case = EclipseCase(_deck(tmp_path, SPE1, "ENDSCALE\n/\n", "SWL\n 300*0.2 /\nSWCR\n 300*0.25 /\n"))
    _, fluid, _, _ = case.build()
    sw = fluid.s_w.reshape(-1)
    assert torch.allclose(sw, torch.full_like(sw, 0.2))
    kr = fluid.blackoil_relperm
    krw, _, _ = kr(torch.full_like(sw, 0.25), torch.zeros_like(sw))
    assert float(krw.abs().max()) < 1e-14
    krw2, _, _ = kr(torch.full_like(sw, 0.26), torch.zeros_like(sw))
    assert float(krw2.min()) > 0


def test_swatinit_honoured_with_scaled_pc(tmp_path):
    base = EclipseCase(SPE9)
    _, f0, _, _ = base.build()
    sw0 = f0.s_w.reshape(-1).numpy()
    target = np.clip(sw0 + 0.05, 0, 1)
    ecl_order = target.reshape(f0.s_w.shape).transpose(2, 1, 0).ravel()
    vals = " ".join(f"{v:.12f}" for v in ecl_order)
    case = EclipseCase(_deck(tmp_path, SPE9, props=f"SWATINIT\n {vals} /\n"))
    _, fluid, _, _ = case.build()
    sw = fluid.s_w.reshape(-1).numpy()
    trans = (sw0 > 0.2) & (sw0 < 0.8)
    assert trans.sum() > 10
    kr = fluid.blackoil_relperm
    pcow0, _ = f0.blackoil_relperm.pc(f0.s_w.reshape(-1), f0.s_g.reshape(-1))
    pc_t = np.interp(target, case.region_tables["swof"][0]["sw"], case.region_tables["swof"][0]["pcow"])
    ok = trans & (pcow0.numpy() > 0) & (pc_t > 0)
    assert ok.sum() > 10 and (~ok & trans).sum() > 0
    assert np.allclose(sw[ok], target[ok], atol=1e-10)
    assert np.allclose(sw[trans & ~ok], sw0[trans & ~ok], atol=1e-10)
    # Pc с новым PCW в начальном Sw воспроизводит po − pw (равновесие, нет перетоков)
    pcow, _ = kr.pc(fluid.s_w.reshape(-1), fluid.s_g.reshape(-1))
    assert np.allclose(pcow.numpy()[trans], pcow0.numpy()[trans], rtol=1e-6, atol=1.0)


def test_unsupported_hysteresis_rejected(tmp_path):
    with pytest.raises(NotImplementedError, match="EHYSTR модель 2"):
        EclipseCase(_deck(tmp_path, SPE1, "SATOPTS\n HYSTER /\n", "EHYSTR\n 0.1 2 /\n"))


def _hyst_deck(tmp_path, isgcr=None):
    text = open(SPE1).read()
    for kw in ("SWOF", "SGOF"):
        m = re.search(rf"^{kw}\s*$", text, re.M)
        nxt = re.search(r"^[A-Z][A-Z0-9]{1,7}\s*$", text[m.end():], re.M)
        a, b = m.end(), m.end() + nxt.start()
        body = text[a:b]
        text = text[:a] + body.rstrip() + "\n" + body + text[b:]
    runspec = "SATOPTS\n HYSTER /\n" + ("ENDSCALE\n/\n" if isgcr is not None else "")
    props = "EHYSTR\n 0.1 0 0.1 1* KR /\n" + (f"ISGCR\n 300*{isgcr} /\n" if isgcr is not None else "")
    path = tmp_path / "H.DATA"
    path.write_text(text)
    return _deck(tmp_path, str(path), runspec, props + "IMBNUM\n 300*2 /\n", name="H2.DATA")


def test_hysteresis_identical_curves_reproduce_base(tmp_path):
    base, _ = run_deck(SPE1, max_days=200)
    h, _ = run_deck(_hyst_deck(tmp_path), max_days=200)
    for key in ("FOPR", "FGPR", "WBHP:PROD", "BGSAT:10,10,1"):
        assert np.allclose(h[key][1:], base[key][1:], rtol=1e-6, atol=1e-10), key


def test_carlson_scanning_curve(tmp_path):
    case = EclipseCase(_hyst_deck(tmp_path, isgcr=0.15))
    _, fluid, _, _ = case.build()
    kr = fluid.blackoil_relperm
    hy = kr.hyst
    n = fluid.s_w.numel()
    sw = torch.full((n,), 0.12, dtype=torch.float64)
    hy.commit(sw, torch.full((n,), 0.5, dtype=torch.float64))
    sg = torch.linspace(0.0, 0.6, n, dtype=torch.float64)
    krg_h = kr(sw, sg)[2]
    krg_d = kr.krg_curve(sg)
    above = sg >= 0.5
    assert torch.allclose(krg_h[above], krg_d[above])
    below = sg < 0.5
    assert torch.all(krg_h[below] <= krg_d[below] + 1e-14)
    # непрерывность в точке разворота
    eps = torch.tensor([0.5 - 1e-7, 0.5], dtype=torch.float64).repeat(n // 2)
    k = kr(torch.full_like(eps, 0.12), eps)[2].view(-1, 2)
    assert float((k[:, 0] - k[:, 1]).abs().max()) < 1e-5
    # защемлённый газ: krg = 0 ниже Sgt > SGCR дренирования
    sgt = hy.trapped_gas()
    assert float(sgt.min()) > 0.03
    k0 = kr(sw, sgt - 1e-6)[2]
    assert float(k0.abs().max()) < 1e-12


def test_hysteresis_changes_gas_injection_response(tmp_path):
    base, _ = run_deck(SPE1, max_days=1200)
    h, _ = run_deck(_hyst_deck(tmp_path, isgcr=0.15), max_days=1200)
    assert np.isfinite(h["FOPR"][1:]).all()
    assert np.abs(h["FGPR"][1:] - base["FGPR"][1:]).max() > 1e-4 * base["FGPR"][1:].max()
