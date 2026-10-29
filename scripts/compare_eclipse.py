"""
Сравнение расчёта deck Eclipse с эталонными сводками (Eclipse и/или OPM Flow).

    python scripts/compare_eclipse.py tests/data/spe1/SPE1CASE1.DATA \
        --ref eclipse=tests/data/spe1/eclipse/SPE1CASE1 --ref opm=tests/data/spe1/opm/SPE1CASE1 \
        --plot results/spe1.png

Сравниваются векторы из секции SUMMARY deck. Ошибка вектора — среднее по отчётным шагам
|наш − эталон| / max|эталон| (нормировка на масштаб вектора устойчива к нулевым дебитам).
"""
import argparse
import json
import os
import sys
import warnings

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


def _valid_samples(key, ours, reference):
    valid = np.isfinite(ours) & np.isfinite(reference)
    # 0/0 при остановленной добыче — неопределённый GOR/обводнённость, а не отказ расчёта.
    if not valid.all() and key.split(":")[0] not in ("FGOR", "FWCT", "GPR", "GPRB"):
        raise ValueError(f"{key}: нечисловые результаты или эталон в выбранном интервале")
    return valid


def compare(summary, refs, metric="mean", start_day=None, end_day=None):
    """[{'vector': ключ, имя_эталона: ошибка, 'a↔b': ошибка между эталонами}, ...]"""
    if metric not in ("mean", "max", "final"):
        raise ValueError("metric: ожидается mean, max или final")
    all_t = np.asarray(summary["TIME"], dtype=float)[1:]
    mask = np.ones(all_t.shape, dtype=bool)
    if start_day is not None:
        mask &= all_t >= start_day
    if end_day is not None:
        mask &= all_t <= end_day
    t = all_t[mask]
    if not len(t):
        raise ValueError("В выбранном интервале нет отчётных дат")
    reduce = {"mean": np.mean, "max": np.max, "final": lambda x: x[-1]}[metric]
    rows = []
    for key in summary:
        if key == "TIME":
            continue
        ours = np.asarray(summary[key])[1:][mask]
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
            valid = _valid_samples(key, ours, rv)
            if valid.any():
                row[name] = float(reduce(np.abs(ours[valid] - rv[valid])) / _scale(refs[name], key, rv[valid]))
            else:
                warnings.warn(f"{key}: нет определённых значений для сравнения с {name}; вектор пропущен",
                              RuntimeWarning, stacklevel=2)
        if len(series) == 2:
            (n1, a), (n2, b) = series.items()
            valid = _valid_samples(key, ours, a) & _valid_samples(key, ours, b)
            if valid.any():
                row[f"{n1}↔{n2}"] = float(reduce(np.abs(a[valid] - b[valid])) / _scale(refs[n1], key, a[valid]))
        if len(row) > 1:
            rows.append(row)
    return rows


def comparison_report(summary, refs, start_day=None, end_day=None):
    """Полные векторы, без агрегации скважин: пики, итог и абсолютная ошибка в единицах SUMMARY."""
    report = {m: compare(summary, refs, m, start_day, end_day) for m in ("mean", "max", "final")}
    t = np.asarray(summary["TIME"])[1:]
    mask = ((t >= start_day) if start_day is not None else np.ones(t.shape, bool))
    if end_day is not None:
        mask &= t <= end_day
    t = t[mask]
    peaks = []
    for row in report["max"]:
        key = row["vector"]
        ours = np.asarray(summary[key])[1:][mask]
        for name, ref in refs.items():
            if name not in row:
                continue
            rv = np.interp(t, *ref_series(ref, key))
            valid = _valid_samples(key, ours, rv)
            ov, ev, tv = ours[valid], rv[valid], t[valid]
            delta = ov - ev
            k = int(np.argmax(np.abs(delta)))
            peaks.append(dict(vector=key, reference=name, day=float(tv[k]),
                              samples_compared=int(valid.sum()), samples_undefined=int((~valid).sum()),
                              last_compared_day=float(tv[-1]),
                              ours=float(ov[k]), reference_value=float(ev[k]),
                              absolute_error=float(abs(delta[k])), final_signed_error=float(delta[-1]),
                              final_ours=float(ov[-1]), final_reference=float(ev[-1]),
                              final_relative_error=(float(delta[-1] / abs(ev[-1]))
                                                    if abs(ev[-1]) > 1e-12 else None)))
    return dict(start_day=float(t[0]), end_day=float(t[-1]),
                normalization="reference scale with small-vector floors; not pointwise relative error",
                metrics=report, peaks=peaks)


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
    ap.add_argument("--metric", choices=("mean", "max", "final"), default="mean")
    ap.add_argument("--start-day", type=float, help="начало окна сравнения, сутки")
    ap.add_argument("--end-day", type=float, help="конец окна сравнения, сутки")
    ap.add_argument("--report-json", help="полный отчёт: среднее, максимум, итог и даты пиков по скважинам")
    ap.add_argument("--convergence-profile", choices=("strict", "relaxed"), default="strict")
    a = ap.parse_args()
    import torch
    refs = {r.split("=", 1)[0]: EclSummary(r.split("=", 1)[1]) for r in a.ref}
    cells = [tuple(int(v) for v in c.split(",")) for c in a.cell]
    summary, info = run_deck(a.deck, device=torch.device("cuda" if a.cuda else "cpu"), cells=cells,
                             max_days=a.days, sim_overrides={"convergence_profile": a.convergence_profile})
    print(f"Расчёт: {info['wall_s']:.1f} с, подшагов {info['substeps']}, итераций Ньютона {info['newton_iters']}")
    rows = compare(summary, refs, a.metric, a.start_day, a.end_day)
    table = {r["vector"]: r for r in rows} if a.all else family_table(rows)
    names = sorted({k for r in table.values() for k in r if k != "vector"}, key=lambda s: ("↔" in s, s))
    print(f"{'вектор':30s}" + "".join(f"{n:>16s}" for n in names) + f"   ({a.metric} / масштаб)")
    for key, r in table.items():
        print(f"{key:30s}" + "".join(f"{100 * r.get(n, np.nan):15.2f}%" for n in names))
    if a.report_json:
        report = comparison_report(summary, refs, a.start_day, a.end_day)
        report["convergence_profile"] = a.convergence_profile
        report["convergence"] = info["convergence"]
        with open(a.report_json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    if a.plot:
        plot(summary, refs, a.plot)
        print(f"График: {a.plot}")


if __name__ == "__main__":
    main()
