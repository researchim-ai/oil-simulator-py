"""Полный прогон Norne (NORNE_ATW2013, 3312 суток) с записью для отчёта: сводка, поля раз в полгода,
статистика подшагов и Ньютона, meta.json. Всё пишется после каждого отчётного шага.

    git clone --filter=blob:none https://github.com/OPM/opm-tests.git
    NORNE_DIR=opm-tests/norne NORNE_OUT=results/norne python scripts/run_norne.py [сутки]

NORNE_DEVICE=cuda:0   — устройство (по умолчанию cuda, если есть);
NORNE_SNAP_AT=1272,1637 — дополнительные снимки полей в эти сутки (например, в даты рестартов OPM);
NORNE_CKPT=путь      — сохранять полное состояние после каждых NORNE_CKPT_EVERY (1) отчётных шагов;
NORNE_RESUME=путь    — продолжить расчёт с такого файла (история подшагов/полей подхватывается из NORNE_OUT).
"""
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import simulator.blackoil_fim as B  # noqa: E402
from simulator.eclipse_run import run_deck  # noqa: E402

OUT = os.environ.get("NORNE_OUT", "results/norne")
os.makedirs(OUT, exist_ok=True)
DECK = os.path.join(os.environ.get("NORNE_DIR", "opm-tests/norne"), "NORNE_ATW2013.DATA")
SNAP_AT = [float(d) for d in os.environ.get("NORNE_SNAP_AT", "").split(",") if d.strip()]
CKPT = os.environ.get("NORNE_CKPT")
RESUME = os.environ.get("NORNE_RESUME")
PROFILE = os.environ.get("NORNE_CONVERGENCE_PROFILE", "strict")
t0 = time.time()
H = {"solver": None, "sub": [], "snap_t": [], "snap": {k: [] for k in ("p", "sw", "sg", "so", "rs")},
     "next_snap": 0.0, "summary": None, "pv": None, "status": "running", "error": None}

if RESUME:
    t_res = float(torch.load(RESUME, map_location="cpu", weights_only=False)["t"])
    if os.path.exists(f"{OUT}/substeps.npz"):
        H["sub"] = [tuple(r) for r in np.load(f"{OUT}/substeps.npz")["data"] if r[0] < t_res - 1e-9]
    if os.path.exists(f"{OUT}/fields.npz"):
        z = np.load(f"{OUT}/fields.npz")
        keep = z["t"] <= t_res + 1e-9
        H["snap_t"] = list(z["t"][keep])
        H["snap"] = {k: list(z[k][keep]) for k in H["snap"]}
        if H["snap_t"]:
            H["next_snap"] = H["snap_t"][-1] + 182.0

orig_init, orig_step = B.BlackOilFIMSolver.__init__, B.BlackOilFIMSolver.step


def init(self, *a, **k):
    orig_init(self, *a, **k)
    H["solver"] = self


def step(self, dt, current_time=None):
    H["solver"] = self
    w0 = time.time()
    ok = orig_step(self, dt, current_time)
    H["sub"].append((float(current_time or 0.0) / 86400.0, dt / 86400.0, bool(ok),
                     int(getattr(self, "last_newton_iters", -1)) if ok else -1, time.time() - w0))
    return ok


def savez(name, **arrays):
    tmp = f"{OUT}/{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, f"{OUT}/{name}.npz")


def full_dims(sol):
    return getattr(sol.reservoir, "full_dims", None) or (sol.nx, sol.ny, sol.nz)


def to_full(sol, a):
    """Поле расчётных ячеек → [nx, ny, nz] полной сетки (неактивные — 0)."""
    a = a.detach().reshape(-1).cpu().numpy()
    cells = getattr(sol.reservoir, "active_cells", None)
    if cells is not None:
        out = np.zeros(int(np.prod(full_dims(sol))), dtype=a.dtype)
        out[cells] = a
        a = out
    return a.reshape(full_dims(sol))


def save_fields():
    sol = H["solver"]
    if H["pv"] is None:
        p0 = sol.fluid.pressure.reshape(-1).double()
        H["pv"] = to_full(sol, sol._pv(p0))
    savez("fields", t=np.array(H["snap_t"]), pv=H["pv"], **{k: np.stack(v) for k, v in H["snap"].items()})


def save_meta(info=None):
    sol = H["solver"]
    s = H["summary"]
    meta = {"status": H["status"], "error": H["error"], "wall_s": time.time() - t0,
            "cells_total": int(np.prod(full_dims(sol))), "dims": list(full_dims(sol)), "cells_computed": sol.n_cells,
            "cells_active": int((H["pv"] > 0).sum()) if H["pv"] is not None else None,
            "wells": len(sol.wells.wells),
            "substeps": len(H["sub"]), "failed_substeps": sum(1 for r in H["sub"] if not r[2]),
            "newton_iters": int(sum(r[3] for r in H["sub"] if r[2])),
            "days": float(s["TIME"][-1]) if s is not None else 0.0,
            "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
            "device": torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else "cpu",
            "torch": torch.__version__, "python": platform.python_version(),
            "convergence": {"tol_cnv": sol.tol_cnv, "tol_mb": sol.tol_mb,
                            "max_strict_iter": sol.max_strict_iter,
                            "relaxed_max_pv_fraction": sol.relaxed_max_pv_fraction,
                            "tol_cnv_relaxed": sol.tol_cnv_relaxed, "tol_mb_relaxed": sol.tol_mb_relaxed},
            "well_events": [list(map(str, e)) for e in (info or {}).get("well_events", [])]}
    try:
        meta["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    tmp = f"{OUT}/meta.json.tmp"
    json.dump(meta, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, f"{OUT}/meta.json")
    return meta


def save_partial():
    if H["summary"] is not None:
        savez("summary", **{k: np.array(v, dtype=np.float64) for k, v in H["summary"].items()})
    savez("substeps", data=np.array(H["sub"], dtype=np.float64).reshape(-1, 5))
    if H["solver"] is not None:
        save_meta()


def snapshot(t):
    f = H["solver"].fluid
    H["snap_t"].append(t)
    for k, a in (("p", f.pressure / 1e5), ("sw", f.s_w), ("sg", f.s_g), ("so", f.s_o), ("rs", f.rs)):
        H["snap"][k].append(to_full(H["solver"], a).astype(np.float32))
    save_fields()


def progress(n, t, s):
    print(f"step {n:4d}  t={t:8.1f}  FOPR={s['FOPR'][-1]:10.1f}  FWPR={s['FWPR'][-1]:10.1f}  "
          f"FGPR={s['FGPR'][-1]:12.1f}  FGIR={s['FGIR'][-1]:12.1f}  FWIR={s['FWIR'][-1]:10.1f}  "
          f"wall={time.time() - t0:7.1f}s", flush=True)
    H["summary"] = s
    if t >= H["next_snap"]:
        snapshot(t)
        H["next_snap"] = t + 182.0
    elif any(abs(t - d) < 0.5 for d in SNAP_AT):
        snapshot(t)
    save_partial()


B.BlackOilFIMSolver.__init__, B.BlackOilFIMSolver.step = init, step
DEV = torch.device(os.environ.get("NORNE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
try:
    s, info = run_deck(DECK, device=DEV, progress=progress, quiet=False,
                       sim_overrides={"convergence_profile": PROFILE},
                       max_days=float(sys.argv[1]) if len(sys.argv) > 1 else None,
                       checkpoint=CKPT, checkpoint_every=int(os.environ.get("NORNE_CKPT_EVERY", "1")),
                       resume=RESUME)
except BaseException as e:
    H["status"], H["error"] = "failed", f"{type(e).__name__}: {str(e)[:300]}"
    save_partial()
    raise

H["status"], H["summary"] = "done", s
if not H["snap_t"] or H["snap_t"][-1] != s["TIME"][-1]:
    snapshot(float(s["TIME"][-1]))
savez("summary", **s)
savez("substeps", data=np.array(H["sub"], dtype=np.float64).reshape(-1, 5))
meta = save_meta(info)
print(json.dumps({k: v for k, v in meta.items() if k != "well_events"}, ensure_ascii=False))
