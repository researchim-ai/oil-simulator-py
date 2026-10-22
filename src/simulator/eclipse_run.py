"""Расчёт deck Eclipse по расписанию TSTEP со сводкой (векторы SUMMARY) в единицах deck."""
import contextlib
import io
import time

import numpy as np
import torch

from .eclipse_deck import EclipseCase, DAY
from .simulation import Simulator

_PHASE = {"O": "oil", "W": "water", "G": "gas"}


def apply_controls(case, step, wm, solver):
    """Переводит скважины на режимы отчётного шага step (смена WCONPROD/WCONINJE по расписанию)."""
    from .groups import GroupTree
    controls = case.steps[step][1]
    fw = getattr(solver, "wells", None)
    if fw is not None and hasattr(fw, "groups"):
        fw.groups = (GroupTree(case.group_parent, case.well_group), case.group_constraints(step))
    for idx, w in enumerate(wm.wells):
        cfg = case.control_cfg(w.name, controls.get(w.name))
        if cfg["type"] != w.type:
            raise NotImplementedError(f"{w.name}: смена типа скважины не поддерживается")
        new = (cfg["control_type"], cfg["control_value"], cfg.get("surface_phase"), cfg.get("bhp_min"),
               cfg.get("bhp_max"), cfg.get("hist_rates"))
        old = (w.control_type, w.control_value, w.surface_phase, w.bhp_min, w.bhp_max,
               getattr(w, "hist_rates", None))
        w.grup = bool(cfg.get("grup", False))
        if fw is not None and getattr(w, "connections", None) is not None and hasattr(fw, "set_connections"):
            fw.set_connections(fw.wells.index(w), case.well_connections(w.name, step))
        if fw is not None and hasattr(fw, "set_open"):
            # скважина без открытых соединений закрыта (как в Eclipse/OPM)
            is_open = cfg["control_type"] != "shut"
            if hasattr(fw, "has_open_connections"):
                is_open = is_open and fw.has_open_connections(fw.wells.index(w))
            fw.set_open(fw.wells.index(w), is_open)
        if new == old:
            continue
        if cfg.get("surface_phase") != w.surface_phase and fw is not None and hasattr(fw, "rate_w"):
            fw.set_surface_phase(fw.wells.index(w), cfg.get("surface_phase"))
        w.control_type, w.control_value, w.surface_phase, w.bhp_min, w.bhp_max, w.hist_rates = new
        if fw is not None and hasattr(fw, "mode"):
            fw.mode[fw.wells.index(w)] = w.control_type


def _requests(case, cells):
    """Список векторов: (ключ, тип, аргумент). Берётся из SUMMARY deck и дополняется cells."""
    out = []
    names = [w for w in case.wells]
    for kw, recs in case.summary_requests:
        if kw[0] == "F" and kw in ("FOPR", "FWPR", "FGPR", "FGOR", "FWCT", "FOPT", "FWPT", "FGPT", "FWIR", "FGIR"):
            out.append((kw, "F", None))
        elif kw[0] == "W" and kw[1:] in ("BHP", "OPR", "WPR", "GPR", "WIR", "GIR", "OIR", "OPT", "WPT", "GPT",
                                         "OIT", "WIT", "GIT", "GOR"):
            listed = [str(v) for r in recs for v in r if v is not None] or names
            for n in listed:
                for m in case._match_wells(n):
                    out.append((f"{kw}:{m}", "W", (kw, m)))
        elif kw[0] == "G" and kw[1:] in ("OPR", "WPR", "GPR", "WIR", "GIR", "OPT", "WPT", "GPT", "WIT", "GIT"):
            listed = [str(v) for r in recs for v in r if v is not None] or ["FIELD"]
            for g in listed:
                out.append((f"{kw}:{g}", "G", (kw, g)))
        elif kw in ("BPR", "BGSAT", "BWSAT", "BOSAT"):
            for r in recs:
                if r:
                    ijk = tuple(int(v) for v in r[:3])
                    out.append((f"{kw}:{ijk[0]},{ijk[1]},{ijk[2]}", "B", (kw, ijk)))
    for kw in ("BPR", "BGSAT"):
        for ijk in cells:
            key = f"{kw}:{ijk[0]},{ijk[1]},{ijk[2]}"
            if all(k != key for k, _, _ in out):
                out.append((key, "B", (kw, tuple(ijk))))
    for kw in ("FOPR", "FWPR", "FGPR", "FGOR"):
        if all(k != kw for k, _, _ in out):
            out.append((kw, "F", None))
    return out


def run_deck(path, device=None, sim_overrides=None, cells=(), quiet=True, max_days=None, progress=None):
    """
    Возвращает (summary, info): summary — словарь массивов по отчётным шагам (t = 0 — NaN для дебитов);
    ключи как в Eclipse: 'TIME', 'FOPR', 'WBHP:PROD', 'BPR:i,j,k' (i,j,k с единицы), ...
    """
    case = EclipseCase(path)
    u = case.u
    out = io.StringIO()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        res, fluid, wm, sim_params = case.build(device=device, sim_overrides=sim_overrides)
        sim = Simulator(res, fluid, wm, sim_params, device=device or torch.device("cpu"))
    solver = sim.fim_solver
    rate_u = {"oil": u["liq"], "water": u["liq"], "gas": u["gas"]}
    requests = _requests(case, cells)
    from .groups import GroupTree
    tree = GroupTree(case.group_parent, case.well_group)
    wells = {w.name: w for w in wm.wells}
    summary = {"TIME": [0.0]}

    def rates(w):
        r = getattr(w, "last_surface_rates", None) or {}
        return {ph: r.get(ph, 0.0) / rate_u[ph] for ph in ("oil", "water", "gas")}

    def record():
        f = {"oil": 0.0, "water": 0.0, "gas": 0.0}
        fi = {"water": 0.0, "gas": 0.0}
        wr = {}
        for w in wm.wells:
            q = rates(w)
            wr[w.name] = q
            for ph in f:
                if w.type == "producer":
                    f[ph] += q[ph]
                elif ph in fi:
                    fi[ph] += -q[ph]
        vals = {"FOPR": f["oil"], "FWPR": f["water"], "FGPR": f["gas"], "FWIR": fi["water"], "FGIR": fi["gas"],
                "FGOR": f["gas"] / f["oil"] if f["oil"] > 0 else np.nan,
                "FWCT": f["water"] / (f["water"] + f["oil"]) if f["water"] + f["oil"] > 0 else np.nan}
        cum = {w.name: {ph: (getattr(w, "cum_surface", None) or {}).get(ph, 0.0) / rate_u[ph]
                        for ph in ("oil", "water", "gas")} for w in wm.wells}
        for key, kind, arg in requests:
            if kind == "F":
                if key.endswith("PT"):
                    ph = _PHASE[key[1]]
                    v = sum(cum[w.name][ph] for w in wm.wells if w.type == "producer")
                else:
                    v = vals[key]
            elif kind == "W":
                kw, name = arg
                w = wells[name]
                if kw == "WBHP":
                    v = np.nan if getattr(w, "bhp_mpa", None) is None else w.bhp_mpa * 1e6 / u["p"]
                elif kw == "WGOR":
                    q = wr[name]
                    v = q["gas"] / q["oil"] if w.type == "producer" and q["oil"] > 0 else 0.0
                else:
                    ph = _PHASE[kw[1]]
                    prod = kw[2] == "P"
                    q = wr[name][ph]
                    rate = (q if w.type == "producer" else 0.0) if prod else (-q if w.type == "injector" else 0.0)
                    if kw.endswith("T"):
                        c = cum[name][ph]
                        v = (c if w.type == "producer" else 0.0) if prod else (-c if w.type == "injector" else 0.0)
                    else:
                        v = rate
            elif kind == "G":
                kw, g = arg
                ph = _PHASE[kw[1]]
                prod = kw[2] == "P"
                v = 0.0
                for w in wm.wells:
                    if g not in tree.ancestors(case.well_group.get(w.name, "FIELD")):
                        continue
                    if (w.type == "producer") != prod:
                        continue
                    x = cum[w.name][ph] if kw.endswith("T") else wr[w.name][ph]
                    v += x if prod else -x
            else:
                kw, (i, j, k) = arg
                c = (i - 1, j - 1, k - 1)
                v = {"BPR": lambda: float(fluid.pressure[c]) / u["p"], "BGSAT": lambda: float(fluid.s_g[c]),
                     "BWSAT": lambda: float(fluid.s_w[c]),
                     "BOSAT": lambda: 1.0 - float(fluid.s_w[c]) - float(fluid.s_g[c])}[kw]()
            summary.setdefault(key, []).append(v)

    def record_initial():
        for key, kind, arg in requests:
            if kind == "B":
                kw, (i, j, k) = arg
                c = (i - 1, j - 1, k - 1)
                v = {"BPR": float(fluid.pressure[c]) / u["p"], "BGSAT": float(fluid.s_g[c]),
                     "BWSAT": float(fluid.s_w[c]), "BOSAT": 1.0 - float(fluid.s_w[c]) - float(fluid.s_g[c])}[kw]
            elif key.split(":")[0].endswith(("PT", "IT")):
                v = 0.0
            else:
                v = np.nan
            summary[key] = [v]

    record_initial()
    t0 = time.perf_counter()
    newton = 0
    substeps = 0
    t = 0.0
    for n, d in enumerate(case.report_days):
        if max_days is not None and t + d > max_days + 1e-9:
            break
        apply_controls(case, n, wm, solver)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            ok = sim.run_step(d * DAY)
        log = buf.getvalue()
        newton += log.count("Iter ") - log.count("Converged!")
        substeps += log.count("Converged!")
        if not ok:
            raise RuntimeError(f"расчёт остановлен на шаге {n} (t = {t:.1f} сут):\n{log[-4000:]}")
        t += d
        summary["TIME"].append(t)
        record()
        if progress:
            progress(n, t, summary)
    summary = {k: np.array(v, dtype=np.float64) for k, v in summary.items()}
    info = {"wall_s": time.perf_counter() - t0, "newton_iters": newton, "substeps": substeps,
            "units": case.units_name, "case": case, "solver": solver}
    return summary, info
