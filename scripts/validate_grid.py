#!/usr/bin/env python
"""
Статическая проверка сетки против Eclipse: активные ячейки, PORV, TRANX/TRANY/TRANZ и NNC (TRANNNC).

    python scripts/validate_grid.py DECK.DATA CASE.INIT CASE.EGRID
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulator.ecl_io import read_ecl_arrays  # noqa: E402
from simulator.eclipse_deck import EclipseCase  # noqa: E402

# Eclipse METRIC: Tecl [cP·rm³/(сут·бар)] = 0.00852702 · k[мД]·A/L
CDARCY_METRIC = 0.00852702 / 9.869233e-16


def _first(arrs, name):
    for k, v in arrs:
        if k == name:
            return v
    return None


def _stats(name, ours, ref, floor):
    big = np.maximum(np.abs(ref), np.abs(ours)) > floor
    rel = np.abs(ours - ref)[big] / np.maximum(np.abs(ref[big]), 1e-300)
    wrong = int((rel > 1e-3).sum())
    print(f"  {name:8s} n={big.sum():7d}  медиана отн.ош. {np.median(rel) if rel.size else 0:.2e}  "
          f"макс {rel.max() if rel.size else 0:.2e}  >0.1%: {wrong}")
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("init")
    ap.add_argument("egrid")
    a = ap.parse_args()

    case = EclipseCase(a.deck, grid_only=True)
    if case.units_name != "METRIC":
        raise SystemExit("пока только METRIC")
    nx, ny, nz = case.dims
    s = case.static_model()
    act = s["active"]
    init, egrid = read_ecl_arrays(a.init), read_ecl_arrays(a.egrid)

    ecl_act = (_first(init, "PORV") > 0).reshape(nz, ny, nx).transpose(2, 1, 0)
    print(f"активные: наши {act.sum()}, Eclipse {ecl_act.sum()}, расхождений {(act != ecl_act).sum()}")
    gidx = np.arange(nx * ny * nz).reshape(nz, ny, nx).transpose(2, 1, 0)   # [i,j,k] -> глобальный Eclipse
    our2g = gidx.reshape(-1)                                                 # наш flat (i,j,k) -> Eclipse
    g2act = -np.ones(nx * ny * nz, dtype=np.int64)
    eclg_act = np.flatnonzero(_first(init, "PORV") > 0)
    g2act[eclg_act] = np.arange(eclg_act.size)

    pv_ref = _first(init, "PORV")
    pv_ours = s["pv"].transpose(2, 1, 0).reshape(-1)
    both = (pv_ref > 0) & (pv_ours > 0)
    _stats("PORV", pv_ours[both], pv_ref[both], 0.0)

    c = s["conn"]
    T = c["T"] * CDARCY_METRIC
    g1, g2 = our2g[c["c1"]], our2g[c["c2"]]
    for ax, name in enumerate(("TRANX", "TRANY", "TRANZ")):
        ref = _first(init, name).astype(np.float64)
        ours = np.zeros_like(ref)
        sel = (c["axis"] == ax) & ~c["nnc"]
        np.add.at(ours, g2act[g1[sel]], T[sel])
        _stats(name, ours, ref, 1e-6 * ref.max())

    n1, n2 = _first(egrid, "NNC1").astype(np.int64), _first(egrid, "NNC2").astype(np.int64)
    tn = _first(init, "TRANNNC").astype(np.float64)
    key = lambda p, q: np.minimum(p, q) * (nx * ny * nz) + np.maximum(p, q)
    ref = dict(zip(key(n1 - 1, n2 - 1).tolist(), tn))
    sel = c["nnc"]
    ours = {}
    for kk, t in zip(key(g1[sel], g2[sel]).tolist(), T[sel]):
        ours[kk] = ours.get(kk, 0.0) + t
    keys = sorted(set(ref) | set(ours))
    r = np.array([ref.get(k, 0.0) for k in keys])
    o = np.array([ours.get(k, 0.0) for k in keys])
    only_ref = sum(1 for k in keys if k not in ours and ref[k] > 0)
    only_ours = sum(1 for k in keys if k not in ref and ours[k] > 1e-6 * tn.max())
    print(f"NNC: Eclipse {len(ref)}, наши {len(ours)}; только у Eclipse (T>0) {only_ref}, только у нас {only_ours}")
    _stats("TRANNNC", o, r, 1e-6 * tn.max())
    tot = r.sum()
    print(f"  суммарная T NNC: наша/Eclipse = {o.sum() / tot:.6f}")


if __name__ == "__main__":
    main()
