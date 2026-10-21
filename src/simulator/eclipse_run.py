"""Расчёт deck Eclipse по расписанию TSTEP со сводкой (FOPR, FGOR, WBHP, BPR, BGSAT…) в единицах deck."""
import contextlib
import io
import time

import numpy as np
import torch

from .eclipse_deck import EclipseCase, DAY
from .simulation import Simulator


def run_deck(path, device=None, sim_overrides=None, cells=(), quiet=True, max_days=None, progress=None):
    """
    Возвращает (summary, info): summary — словарь массивов по отчётным шагам
    (ключи 'TIME', 'FOPR', 'FGOR', 'WBHP:PROD', 'BPR:i,j,k', ...; i,j,k с единицы).
    """
    case = EclipseCase(path)
    u = case.u
    out = io.StringIO()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        res, fluid, wm, sim_params = case.build(device=device, sim_overrides=sim_overrides)
        sim = Simulator(res, fluid, wm, sim_params, device=device or torch.device("cpu"))
    solver = sim.fim_solver
    rate_u = {"oil": u["liq"], "water": u["liq"], "gas": u["gas"]}
    nx, ny, _ = case.dims
    summary = {"TIME": [0.0]}

    def record():
        q = {"oil": 0.0, "water": 0.0, "gas": 0.0}
        for w in wm.wells:
            r = getattr(w, "last_surface_rates", None) or {"oil": 0.0, "water": 0.0, "gas": 0.0}
            prod = w.type == "producer"
            for ph in q:
                val = r.get(ph, 0.0) / rate_u[ph]
                summary.setdefault(f"W{ph[0].upper()}{'P' if prod else 'I'}R:{w.name}", []).append(
                    val if prod else -val)
                if prod:
                    q[ph] += val
            bhp = getattr(w, "bhp_mpa", None)
            summary.setdefault(f"WBHP:{w.name}", []).append(np.nan if bhp is None else bhp * 1e6 / u["p"])
        summary.setdefault("FOPR", []).append(q["oil"])
        summary.setdefault("FWPR", []).append(q["water"])
        summary.setdefault("FGPR", []).append(q["gas"])
        summary.setdefault("FGOR", []).append(q["gas"] / q["oil"] if q["oil"] > 0 else np.nan)
        for (i, j, k) in cells:
            summary.setdefault(f"BPR:{i},{j},{k}", []).append(float(fluid.pressure[i - 1, j - 1, k - 1]) / u["p"])
            summary.setdefault(f"BGSAT:{i},{j},{k}", []).append(float(fluid.s_g[i - 1, j - 1, k - 1]))

    t0 = time.perf_counter()
    newton = 0
    substeps = 0
    t = 0.0
    for n, d in enumerate(case.report_days):
        if max_days is not None and t + d > max_days + 1e-9:
            break
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
    for k in list(summary):
        if k != "TIME":
            summary[k] = np.array([np.nan] + summary[k])
    summary["TIME"] = np.array(summary["TIME"])
    info = {"wall_s": time.perf_counter() - t0, "newton_iters": newton, "substeps": substeps,
            "units": case.units_name, "case": case, "solver": solver}
    return summary, info
