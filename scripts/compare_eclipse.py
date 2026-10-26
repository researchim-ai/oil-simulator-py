"""
Сравнение расчёта deck Eclipse с эталонными сводками (Eclipse и/или OPM Flow).

    python scripts/compare_eclipse.py tests/data/spe1/SPE1CASE1.DATA \
        --ref eclipse=tests/data/spe1/eclipse/SPE1CASE1 --ref opm=tests/data/spe1/opm/SPE1CASE1 \
        --plot results/spe1.png

Сравниваются векторы из секции SUMMARY deck. Ошибка вектора — среднее по отчётным шагам
|наш − эталон| / max|эталон| (нормировка на масштаб вектора устойчива к нулевым дебитам).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from simulator.ecl_io import EclSummary  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402


def ref_series(ref, key):
    """(сутки, значения) эталона для ключа вида 'FOPR', 'WBHP:PROD', 'BPR:1,1,1'."""
    kw, _, arg = key.partition(":")
    if kw[0] == "B":
        i, j, k = (int(v) for v in arg.split(","))
        return ref.days, ref.get(kw, num=ref.cell_num(i, j, k))
    v = ref.get(kw, arg or None)
    if kw.endswith("GOR") and not ref.unit(kw, arg or None).strip():
        # сводки без единиц (OPM flow_legacy) пишут GOR в ст.м³/ст.м³; deck предполагается в FIELD
        v = v / 178.1076
    return ref.days, v


def _scale(ref, key, rv):
    """Масштаб вектора; для дебитов скважин — не меньше 1% от месторожденческого (W*PR → F*PR)."""
    s = max(np.abs(rv).max(), 1e-12)
    kw = key.split(":")[0]
    if kw.endswith("SAT") or kw.endswith("WCT"):
        s = max(s, 0.01)
    if kw[0] in "FW" and kw[1] == "W" and kw[2:] in ("PR", "PT"):
        try:
            s = max(s, 0.01 * np.abs(ref.get("FO" + kw[2:])).max())
        except (KeyError, ValueError):
            pass
    if kw[0] == "W" and kw not in ("WBHP", "WGOR"):
        fallback = ("FGPR",) if kw[1] == "G" else ("FOPR",)
        for fk in ("F" + kw[1:],) + fallback:
            try:
                s = max(s, 0.01 * np.abs(ref.get(fk)).max())
                break
            except (KeyError, ValueError):
                continue
    return s


def compare(summary, refs):
    """[{'vector': ключ, имя_эталона: ошибка, 'a↔b': ошибка между эталонами}, ...]"""
    t = summary["TIME"][1:]
    rows = []
    for key in summary:
        if key == "TIME":
            continue
        ours = summary[key][1:]
        row = {"vector": key}
        series = {}
        for name, ref in refs.items():
            try:
                series[name] = np.interp(t, *ref_series(ref, key))
            except (KeyError, ValueError):
                continue
        if not series:
            continue
        for name, rv in series.items():
            row[name] = float(np.mean(np.abs(ours - rv)) / _scale(refs[name], key, rv))
        if len(series) == 2:
            (n1, a), (n2, b) = series.items()
            row[f"{n1}↔{n2}"] = float(np.mean(np.abs(a - b)) / _scale(refs[n1], key, a))
        rows.append(row)
    return rows


def family_table(rows):
    """Свод по семействам: для W* — максимум по скважинам."""
    fam = {}
    for r in rows:
        kw = r["vector"].split(":")[0]
        key = r["vector"] if kw[0] != "W" else f"{kw} (макс. по скважинам)"
        cur = fam.setdefault(key, {})
        for k, v in r.items():
            if k != "vector":
                cur[k] = max(cur.get(k, 0.0), v)
    return fam


def plot(summary, refs, path, keys=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if keys is None:
        keys = [k for k in summary if k != "TIME" and (k[0] in "FB")]
        keys += [k for k in summary if k.startswith("WBHP:")][:2]
    keys = keys[:12]
    ncol = 4
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 4 * nrow), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        for (name, ref), st in zip(refs.items(), ("k-", "b--")):
            try:
                ax.plot(*ref_series(ref, key), st, lw=1.5, label=name)
            except (KeyError, ValueError):
                pass
        ax.plot(summary["TIME"], summary[key], "r.", ms=4, label="этот симулятор")
        ax.set_title(key)
        ax.set_xlabel("сутки")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
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
    ap.add_argument("--cell", action="append", default=[], help="i,j,k — добавить BPR/BGSAT ячейки")
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--all", action="store_true", help="печатать все векторы, а не свод по семействам")
    a = ap.parse_args()
    import torch
    refs = {r.split("=", 1)[0]: EclSummary(r.split("=", 1)[1]) for r in a.ref}
    cells = [tuple(int(v) for v in c.split(",")) for c in a.cell]
    summary, info = run_deck(a.deck, device=torch.device("cuda" if a.cuda else "cpu"), cells=cells,
                             max_days=a.days)
    print(f"Расчёт: {info['wall_s']:.1f} с, подшагов {info['substeps']}, итераций Ньютона {info['newton_iters']}")
    rows = compare(summary, refs)
    table = {r["vector"]: r for r in rows} if a.all else family_table(rows)
    names = sorted({k for r in table.values() for k in r if k != "vector"}, key=lambda s: ("↔" in s, s))
    print(f"{'вектор':30s}" + "".join(f"{n:>16s}" for n in names) + "   (средняя ошибка / масштаб)")
    for key, r in table.items():
        print(f"{key:30s}" + "".join(f"{100 * r.get(n, np.nan):15.2f}%" for n in names))
    if a.plot:
        plot(summary, refs, a.plot)
        print(f"График: {a.plot}")


if __name__ == "__main__":
    main()
