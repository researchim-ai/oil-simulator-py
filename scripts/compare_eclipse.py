"""
Сравнение расчёта deck Eclipse с эталонными сводками (Eclipse и/или OPM Flow).

    python scripts/compare_eclipse.py tests/data/spe1/SPE1CASE1.DATA \
        --ref eclipse=tests/data/spe1/eclipse/SPE1CASE1 --ref opm=tests/data/spe1/opm/SPE1CASE1 \
        --plot results/spe1.png
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

VECTORS = [("FOPR", None, None), ("FGOR", None, None), ("WBHP", "PROD", None), ("WBHP", "INJ", None),
           ("BPR", None, (1, 1, 1)), ("BPR", None, (10, 10, 3)), ("BGSAT", None, (1, 1, 1)),
           ("BGSAT", None, (10, 10, 3))]


def ref_series(ref, kw, well, cell):
    if cell is not None:
        return ref.days, ref.get(kw, num=ref.cell_num(*cell))
    return ref.days, ref.get(kw, well)


def our_key(kw, well, cell):
    if cell is not None:
        return f"{kw}:{cell[0]},{cell[1]},{cell[2]}"
    return f"{kw}:{well}" if well else kw


def compare(summary, refs):
    """Средняя относительная ошибка по времени (без t = 0) для каждого вектора и эталона."""
    t = summary["TIME"][1:]
    rows = []
    for kw, well, cell in VECTORS:
        key = our_key(kw, well, cell)
        if key not in summary:
            continue
        ours = summary[key][1:]
        row = {"vector": key}
        for name, ref in refs.items():
            try:
                td, v = ref_series(ref, kw, well, cell)
            except KeyError:
                continue
            rv = np.interp(t, td, v)
            scale = max(np.abs(rv).max(), 1e-12) if kw == "BGSAT" else np.abs(rv).clip(1e-12)
            row[name] = float(np.mean(np.abs(ours - rv) / scale))
        if len(refs) == 2:
            (n1, r1), (n2, r2) = refs.items()
            a = np.interp(t, *ref_series(r1, kw, well, cell))
            b = np.interp(t, *ref_series(r2, kw, well, cell))
            scale = max(np.abs(a).max(), 1e-12) if kw == "BGSAT" else np.abs(a).clip(1e-12)
            row[f"{n1}↔{n2}"] = float(np.mean(np.abs(a - b) / scale))
        rows.append(row)
    return rows


def plot(summary, refs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, (kw, well, cell) in zip(axes.flat, VECTORS):
        key = our_key(kw, well, cell)
        for (name, ref), st in zip(refs.items(), ("k-", "b--")):
            td, v = ref_series(ref, kw, well, cell)
            ax.plot(td, v, st, lw=1.5, label=name)
        ax.plot(summary["TIME"], summary[key], "r.", ms=4, label="этот симулятор")
        ax.set_title(key)
        ax.set_xlabel("сутки")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--ref", action="append", default=[], help="имя=путь/без/расширения")
    ap.add_argument("--plot")
    ap.add_argument("--days", type=float)
    ap.add_argument("--cuda", action="store_true")
    a = ap.parse_args()
    import torch
    refs = {r.split("=", 1)[0]: EclSummary(r.split("=", 1)[1]) for r in a.ref}
    cells = sorted({c for _, _, c in VECTORS if c is not None})
    summary, info = run_deck(a.deck, device=torch.device("cuda" if a.cuda else "cpu"), cells=cells,
                             max_days=a.days)
    print(f"Расчёт: {info['wall_s']:.1f} с, подшагов {info['substeps']}, итераций Ньютона {info['newton_iters']}")
    rows = compare(summary, refs)
    names = [k for k in rows[0] if k != "vector"] if rows else []
    print(f"{'вектор':22s}" + "".join(f"{n:>16s}" for n in names) + "   (средняя отн. ошибка)")
    for r in rows:
        print(f"{r['vector']:22s}" + "".join(f"{100 * r.get(n, np.nan):15.2f}%" for n in names))
    if a.plot:
        plot(summary, refs, a.plot)
        print(f"График: {a.plot}")


if __name__ == "__main__":
    main()
