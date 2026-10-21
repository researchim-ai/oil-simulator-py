"""Регионы SATNUM / PVTNUM / EQLNUM на SPE1 (10×10×3): верхний слой — регион 1, нижние два — регион 2."""
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.eclipse_deck import EclipseCase  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

SPE1 = os.path.join(os.path.dirname(__file__), "data", "spe1", "SPE1CASE1.DATA")
REGIONS = "REGIONS\nSATNUM\n 100*1 200*2 /\nPVTNUM\n 100*1 200*2 /\nEQLNUM\n 100*1 200*2 /\n"


def _keyword_body(text, kw):
    m = re.search(rf"^{kw}\s*$", text, re.M)
    nxt = re.search(r"^[A-Z][A-Z0-9]{1,7}\s*$", text[m.end():], re.M)
    return m.end(), m.end() + nxt.start()


def _two_region_deck(tmp_path, equil2=None):
    text = open(SPE1).read()
    for kw in ("PVTW", "ROCK", "SWOF", "SGOF", "DENSITY", "PVDG", "PVTO", "EQUIL", "RSVD"):
        a, b = _keyword_body(text, kw)
        body = text[a:b]
        second = body if (kw != "EQUIL" or equil2 is None) else f"\n {equil2} /\n"
        text = text[:a] + body.rstrip() + "\n" + second + text[b:]
    text = text.replace("\nSOLUTION", "\n" + REGIONS + "SOLUTION", 1)
    path = tmp_path / "SPE1_REG.DATA"
    path.write_text(text)
    return str(path)


def test_identical_regions_reproduce_single_region(tmp_path):
    base, _ = run_deck(SPE1, max_days=200)
    reg, _ = run_deck(_two_region_deck(tmp_path), max_days=200)
    for key in ("FOPR", "FGPR", "WBHP:PROD", "BPR:10,10,3", "BGSAT:10,10,1"):
        a, b = reg[key][1:], base[key][1:]
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12), key


def test_eqlnum_contacts_per_region(tmp_path):
    # регион 2: ВНК на 8370 ft — нижний слой (центр ~8400 ft) под водой, средний (~8360) — в нефти
    case = EclipseCase(_two_region_deck(tmp_path, equil2="8400 4800 8370 0 8300 0 1 0 0"))
    _, fluid, _, _ = case.build()
    sw = fluid.s_w.numpy()
    swc = 0.12
    assert np.allclose(sw[:, :, 0], swc) and np.allclose(sw[:, :, 1], swc)
    assert np.allclose(sw[:, :, 2], 1.0)
    base_case = EclipseCase(SPE1)
    _, f0, _, _ = base_case.build()
    assert np.allclose(f0.s_w.numpy()[:, :, 2], swc)


def test_region_count_mismatch_is_reported(tmp_path):
    text = open(SPE1).read().replace("\nSOLUTION", "\n" + REGIONS + "SOLUTION", 1)
    path = tmp_path / "BAD.DATA"
    path.write_text(text)
    with pytest.raises(ValueError, match="регионов"):
        EclipseCase(str(path)).build()
