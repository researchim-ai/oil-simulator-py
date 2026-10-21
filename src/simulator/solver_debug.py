"""
Режим отладки нелинейного/линейного решателя FIM.

Включается через ``simulation.debug`` в конфиге:

    "debug": {
        "enabled": true,
        "level": 1,                 # 1 — сводка по итерациям, 2 — + истории FGMRES/AMG/line search/скважины,
                                    # 3 — + проверка Якобиана конечными разностями и дампы на каждой итерации
        "output_dir": "results/solver_debug",
        "console": true,            # печатать компактную таблицу итераций
        "dump_on_failure": true,    # сохранить p, Sw, R, J при несходимости шага
        "top_k_cells": 5,           # сколько худших ячеек по невязке сохранять
        "fd_check_cells": 8,        # уровень 3: число ячеек для проверки Якобиана
        "capture_amg_output": true  # не печатать лог построения AMG, а складывать в журнал (level>=2)
    }

Журнал — JSON Lines (одна запись на событие) в ``<output_dir>/<run_id>/events.jsonl``.
"""
import atexit
import contextlib
import datetime
import io
import json
import os
import time

import numpy as np
import torch

EQ_NAMES = ("oil", "water")
VAR_NAMES = ("p", "sw")


def _f(x):
    if isinstance(x, torch.Tensor):
        x = x.item() if x.numel() == 1 else x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    return x


class _Timer:
    def __init__(self, sync):
        self.sync = sync
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, name):
        if self.sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                torch.cuda.synchronize()
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0


class _NullTimer:
    @contextlib.contextmanager
    def __call__(self, name):
        yield


class SolverDebugger:
    """Сборщик диагностики. При выключенном режиме все методы — дешёвые no-op."""

    def __init__(self, cfg, dims, device):
        cfg = cfg if isinstance(cfg, dict) else {"enabled": bool(cfg)}
        self.enabled = bool(cfg.get("enabled", False))
        self.level = int(cfg.get("level", 1)) if self.enabled else 0
        self.console = bool(cfg.get("console", True))
        self.dump_on_failure = bool(cfg.get("dump_on_failure", True))
        self.top_k = int(cfg.get("top_k_cells", 5))
        self.fd_cells = int(cfg.get("fd_check_cells", 8))
        self.capture_amg = bool(cfg.get("capture_amg_output", True))
        self.dims = tuple(int(d) for d in dims)
        self.device = device
        self.step_idx = 0
        self.iter_rec = None
        self.step_rec = None
        self.summary = {"steps": 0, "failed_steps": 0, "newton_iters": 0, "linear_iters": 0,
                        "linear_not_converged": 0, "time": {}}
        if not self.enabled:
            return
        run_id = cfg.get("run_id") or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(cfg.get("output_dir", "results/solver_debug"), run_id)
        os.makedirs(self.dir, exist_ok=True)
        self._fh = open(os.path.join(self.dir, "events.jsonl"), "w")
        atexit.register(self.close)
        self.event("debug_start", level=self.level, dims=list(self.dims), device=str(device))
        print(f"[solver-debug] level={self.level}, журнал: {self.dir}/events.jsonl")

    # ------------------------------------------------------------------ utils
    def event(self, kind, **data):
        if not self.enabled:
            return
        rec = {"event": kind, "step": self.step_idx, "wall": time.time()}
        rec.update({k: _f(v) if not isinstance(v, (dict, list)) else v for k, v in data.items()})
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=_f) + "\n")
        self._fh.flush()

    def timer(self):
        if not self.enabled:
            return _NullTimer()
        return _Timer(sync=(isinstance(self.device, torch.device) and self.device.type == "cuda"))

    def ijk(self, cell):
        _, ny, nz = self.dims
        return [int(cell // (ny * nz)), int((cell // nz) % ny), int(cell % nz)]

    def _state_stats(self, p, sw):
        p_mpa = p / 1e6
        bad = (~torch.isfinite(p)).sum() + (~torch.isfinite(sw)).sum()
        return {"p_mpa": [p_mpa.min().item(), p_mpa.mean().item(), p_mpa.max().item()],
                "sw": [sw.min().item(), sw.mean().item(), sw.max().item()],
                "n_negative_p": int((p < 0).sum().item()),
                "n_sw_out_of_range": int(((sw < 0) | (sw > 1)).sum().item()),
                "n_nonfinite": int(bad.item())}

    @contextlib.contextmanager
    def quiet(self, tag):
        """Перехватывает stdout (лог AMG) и складывает его в журнал."""
        if not (self.enabled and self.capture_amg):
            yield
            return
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            yield
        if self.level >= 2:
            self.event("captured_output", tag=tag, text=buf.getvalue())

    # ---------------------------------------------------------------- шаг
    def begin_step(self, dt, t, p, sw):
        if not self.enabled:
            return
        self.step_rec = {"dt_days": dt / 86400.0, "t_days": (t or 0.0) / 86400.0, "iters": [], "t0": time.perf_counter()}
        self.event("step_start", dt_days=dt / 86400.0, t_days=(t or 0.0) / 86400.0, **self._state_stats(p, sw))
        if self.console:
            print(f"[solver-debug] шаг {self.step_idx}: t={self.step_rec['t_days']:.4f} сут, dt={dt/86400.0:.4g} сут")
            print("   it |   CNV oil    CNV wat | худшая ячейка(eq)  | MB       | lin it  lin rel   | max dP,bar max dSw | скв.err | t,s")

    def end_step(self, converged, reason, p=None, sw=None, R=None, J_vals=None, builder=None):
        if not self.enabled:
            return
        el = time.perf_counter() - self.step_rec["t0"]
        self.summary["steps"] += 1
        if not converged:
            self.summary["failed_steps"] += 1
        extra = self._state_stats(p, sw) if p is not None else {}
        self.event("step_end", converged=converged, reason=reason, newton_iters=len(self.step_rec["iters"]),
                   time_s=el, **extra)
        if self.console:
            print(f"[solver-debug] шаг {self.step_idx}: {'OK' if converged else 'FAIL'} ({reason}), "
                  f"{len(self.step_rec['iters'])} итер., {el:.2f} с")
        if not converged and self.dump_on_failure and R is not None:
            self.dump(f"fail_step{self.step_idx}", p, sw, R, J_vals, builder)
        self.step_idx += 1

    def dt_chop(self, attempt, dt_old, dt_new):
        self.event("dt_chop", attempt=attempt, dt_old_days=dt_old / 86400.0, dt_new_days=dt_new / 86400.0)

    # ------------------------------------------------------------ итерация
    def begin_iter(self, k):
        if not self.enabled:
            return
        self.iter_rec = {"iter": k, "time": {}}

    def residual(self, R, dt, mass_total):
        """Нормы невязки по уравнениям, худшие ячейки, ошибка материального баланса."""
        if not self.enabled:
            return
        Rb = R.view(-1, 2)
        absR = Rb.abs()
        rec = self.iter_rec
        rec["res_inf"] = absR.max(dim=0).values.tolist()
        rec["res_l2"] = Rb.norm(dim=0).tolist()
        # глобальный баланс: потоки между ячейками взаимно сокращаются, остаётся накопление + скважины
        mb = (Rb.sum(dim=0) * dt).tolist()
        rec["mb_err_kg"] = mb
        rec["mb_err_rel"] = [abs(m) / max(mt, 1e-30) for m, mt in zip(mb, mass_total)]
        flat = absR.view(-1)
        k = min(self.top_k, flat.numel())
        vals, idx = torch.topk(flat, k)
        rec["worst"] = [{"cell": int(i // 2), "ijk": self.ijk(int(i // 2)), "eq": EQ_NAMES[int(i % 2)],
                         "res": float(v)} for v, i in zip(vals.tolist(), idx.tolist())]
        rec["nan_in_R"] = bool(torch.isnan(R).any().item() or torch.isinf(R).any().item())

    def jacobian(self, J_vals, diag_map):
        if self.level < 2:
            return
        d = J_vals[diag_map]
        det = d[:, 0, 0] * d[:, 1, 1] - d[:, 0, 1] * d[:, 1, 0]
        self.iter_rec["jac"] = {
            "nan": bool(torch.isnan(J_vals).any().item()),
            "diag_abs_min": [d[:, 0, 0].abs().min().item(), d[:, 1, 1].abs().min().item()],
            "diag_abs_max": [d[:, 0, 0].abs().max().item(), d[:, 1, 1].abs().max().item()],
            "diag_det_abs_min": det.abs().min().item(),
            "n_near_singular_blocks": int((det.abs() < 1e-14 * det.abs().max()).sum().item()),
        }

    def linear(self, stats):
        if not self.enabled:
            return
        self.iter_rec["linear"] = stats if self.level >= 2 else {k: v for k, v in stats.items() if k != "history"}
        self.summary["linear_iters"] += int(stats.get("iters", 0))
        if not stats.get("converged", True):
            self.summary["linear_not_converged"] += 1

    def update(self, dp, dsw, scale):
        if not self.enabled:
            return
        adp, ads = dp.abs().view(-1), dsw.abs().view(-1)
        ip, js = int(torch.argmax(adp)), int(torch.argmax(ads))
        self.iter_rec["update"] = {"max_dp_bar": adp[ip].item() / 1e5, "max_dp_ijk": self.ijk(ip),
                                   "max_dsw": ads[js].item(), "max_dsw_ijk": self.ijk(js),
                                   "scale": scale}

    def line_search(self, trials, alpha, accepted):
        if not self.enabled:
            return
        self.iter_rec["line_search"] = {"alpha": alpha, "accepted": accepted}
        if self.level >= 2:
            self.iter_rec["line_search"]["trials"] = trials

    def wells(self, records):
        if self.level >= 2 and records:
            self.iter_rec["wells"] = records

    def end_iter(self, timings):
        if not self.enabled:
            return
        rec = self.iter_rec
        rec["time"] = timings
        self.step_rec["iters"].append(rec)
        self.summary["newton_iters"] += 1
        for k, v in timings.items():
            self.summary["time"][k] = self.summary["time"].get(k, 0.0) + v
        self.event("newton_iter", **rec)
        if self.console:
            self._print_iter(rec)
        self.iter_rec = None

    def converged_iter(self):
        """Последняя оценка невязки на сошедшейся итерации (без решения СЛАУ)."""
        if not self.enabled or self.iter_rec is None:
            return
        self.event("newton_iter", **self.iter_rec)
        if self.console:
            self._print_iter(self.iter_rec)
        self.iter_rec = None

    def _print_iter(self, r):
        w = r["worst"][0] if r.get("worst") else {"ijk": "-", "eq": "-"}
        lin = r.get("linear", {})
        up = r.get("update", {})
        crit = r.get("criteria", {})
        cnv = crit.get("cnv", r["res_inf"])
        mb = max(crit.get("mb", r["mb_err_rel"]))
        t = sum(r.get("time", {}).values())
        print(f"  {r['iter']:3d} | {cnv[0]:10.3e} {cnv[1]:10.3e} | "
              f"{str(w['ijk']):>12s}({w['eq'][0]}) | {mb:8.1e} | "
              f"{lin.get('iters', '-'):>5} {lin.get('true_rel_res', float('nan')):9.2e} | "
              f"{up.get('max_dp_bar', float('nan')):9.3g} {up.get('max_dsw', float('nan')):7.3g} | "
              f"{crit.get('well_rate_rel_err', 0.0):7.1e} | {t:.3f}"
              + (f"  ⇄ {'; '.join(r['control_switch'])}" if r.get("control_switch") else ""))

    # -------------------------------------------------------- диагностика
    def fd_check(self, solver, p, sw, m_o_old, m_w_old, dt, t, J_vals):
        """Уровень 3: сравнение столбцов Якобиана с центральными разностями."""
        if self.level < 3:
            return
        n = solver.n_cells
        rng = np.random.default_rng(self.step_idx)
        cells = set(rng.choice(n, min(self.fd_cells, n), replace=False).tolist())
        cells.update(w.cell_index_flat for w in solver.well_manager.wells)
        rows = torch.repeat_interleave(torch.arange(n, device=J_vals.device),
                                       (solver.builder.indptr[1:] - solver.builder.indptr[:-1]))
        cols = solver.builder.indices
        # изломы относительных проницаемостей: центральная разность через излом даёт ложное расхождение
        kinks = (float(getattr(solver, "sw_irr", 0.0)), 1.0 - float(getattr(solver, "so_r", 0.0)))
        out = []
        for c in sorted(cells):
            for var, h in ((0, max(1.0, 1e-6 * float(p.view(-1)[c].abs()))), (1, 1e-6)):
                signs = (1.0, -1.0)
                if var == 1:
                    s0 = float(sw.view(-1)[c])
                    if any(abs(s0 - kk) < h for kk in kinks):
                        signs = (1.0, 0.0) if s0 >= kinks[0] and s0 < kinks[1] else (0.0, -1.0)
                Rs = []
                for sgn in signs:
                    pp, ss = p.clone().view(-1), sw.clone().view(-1)
                    (pp if var == 0 else ss)[c] += sgn * h
                    Rk, _ = solver.assemble_system(pp.view_as(p), ss.view_as(sw), m_o_old, m_w_old, dt, current_time=t)
                    Rs.append(Rk)
                fd = (Rs[0] - Rs[1]) / ((signs[0] - signs[1]) * h)
                an = torch.zeros_like(fd)
                m = cols == c
                an[2 * rows[m]] = J_vals[m, 0, var]
                an[2 * rows[m] + 1] = J_vals[m, 1, var]
                err = ((fd - an).abs().max() / fd.abs().max().clamp_min(1e-300)).item()
                out.append({"cell": c, "ijk": self.ijk(c), "var": VAR_NAMES[var], "rel_err": err})
        worst = max(o["rel_err"] for o in out)
        self.iter_rec["fd_check"] = {"worst_rel_err": worst, "cells": out}
        if self.console and worst > 1e-5:
            print(f"  [solver-debug] ВНИМАНИЕ: Якобиан расходится с конечными разностями, rel_err={worst:.2e}")

    def dump(self, tag, p, sw, R, J_vals, builder):
        if not self.enabled:
            return
        path = os.path.join(self.dir, f"{tag}.npz")
        arrays = {"p": p.detach().cpu().numpy(), "sw": sw.detach().cpu().numpy(), "R": R.detach().cpu().numpy(),
                  "dims": np.array(self.dims)}
        if J_vals is not None and builder is not None:
            arrays.update(J_vals=J_vals.detach().cpu().numpy(), indptr=builder.indptr.cpu().numpy(),
                          indices=builder.indices.cpu().numpy())
        np.savez_compressed(path, **arrays)
        self.event("dump", path=path)
        print(f"[solver-debug] дамп системы: {path}")

    def maybe_dump_iter(self, p, sw, R, J_vals, builder, k):
        if self.level >= 3:
            self.dump(f"step{self.step_idx}_iter{k}", p, sw, R, J_vals, builder)

    def close(self):
        if not self.enabled:
            return
        s = self.summary
        self.event("summary", **s)
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        self._fh.close()
        self.enabled = False
        tt = ", ".join(f"{k}={v:.2f}s" for k, v in sorted(s["time"].items(), key=lambda kv: -kv[1]))
        print(f"[solver-debug] итог: шагов {s['steps']} (неудачных {s['failed_steps']}), итераций Ньютона "
              f"{s['newton_iters']}, итераций FGMRES {s['linear_iters']} (несошедшихся СЛАУ "
              f"{s['linear_not_converged']}); время: {tt}")
