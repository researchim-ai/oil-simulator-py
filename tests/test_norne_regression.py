"""Регрессия на Norne против Eclipse 2014.2: концевые точки ОФП по ячейкам и первый год истории."""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

_CANDIDATES = [os.environ.get("NORNE_DIR"), "/tmp/ot2/norne",
               os.path.join(os.path.dirname(__file__), "..", ".scratch", "ot2", "norne")]
NORNE = next((p for p in _CANDIDATES if p and os.path.exists(os.path.join(p, "ECL.2014.2", "NORNE_ATW2013.INIT"))),
             None)
DECK = os.path.join(NORNE, "NORNE_ATW2013.DATA") if NORNE else None
ECL = os.path.join(NORNE, "ECL.2014.2", "NORNE_ATW2013") if NORNE else None
needs_norne = pytest.mark.skipif(NORNE is None, reason="нет данных Norne (OPM opm-tests); задайте NORNE_DIR")


@needs_norne
def test_norne_endpoints_match_eclipse_init():
    """Масштабированные концевые точки дренирования и пропитки (ENDSCALE, SCALECRS, HYSTER) — как в INIT Eclipse."""
    import contextlib
    import io

    from simulator.ecl_io import read_ecl_arrays
    from simulator.eclipse_deck import EclipseCase

    case = EclipseCase(DECK)
    with contextlib.redirect_stdout(io.StringIO()):
        res, fluid, wm, sp = case.build(device=torch.device("cpu"))
    from simulator.simulation import Simulator
    with contextlib.redirect_stdout(io.StringIO()):
        kr = Simulator(res, fluid, wm, sp, device=torch.device("cpu")).fim_solver.kr
    nx, ny, nz = case.dims
    init = dict(read_ecl_arrays(ECL + ".INIT"))
    kk, jj, ii = np.nonzero(init["PORV"].reshape(nz, ny, nx) > 0)
    o = np.lexsort((ii, jj, kk))
    idx = torch.tensor(ii[o] * ny * nz + jj[o] * nz + kk[o])
    for prefix, model in (("", kr), ("I", kr.hyst.imb)):
        for key in ("SWL", "SWCR", "SWU", "SOWCR", "SOGCR", "SGL", "SGCR", "SGU"):
            ecl = init[prefix + key].astype(np.float64)
            ours = np.asarray(model.endpoint(key, idx), dtype=np.float64)
            given = ecl > -1e19        # −1e20 в INIT: не задано, берётся из таблицы
            assert np.allclose(ours[given], ecl[given], atol=1e-6), prefix + key


@needs_norne
@pytest.mark.skipif(not torch.cuda.is_available(), reason="первый год Norne на CPU слишком долгий")
def test_norne_first_year_matches_eclipse():
    from simulator.ecl_io import EclSummary
    from simulator.eclipse_run import run_deck

    s, info = run_deck(DECK, device=torch.device("cuda"), max_days=360)
    e = EclSummary(ECL)
    t = float(s["TIME"][-1])
    assert abs(t - 360.0) < 1e-6
    fgit = float((np.nan_to_num(s["FGIR"][1:]) * np.diff(s["TIME"])).sum())
    for key, ours, tol in (("FOPT", float(s["FOPT"][-1]), 0.5), ("FGPT", float(s["FGPT"][-1]), 1.0),
                           ("FGIT", fgit, 0.5)):
        ev = float(np.interp(t, e.days, e.get(key)))
        assert abs(ours - ev) <= tol / 100 * abs(ev), (key, ours, ev)
