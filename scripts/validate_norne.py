"""Run strict Norne from frozen sources, auditing every accepted substep.

NORNE_DIR and a new NORNE_OUT directory are required. NORNE_DAYS defaults
to 3312; shorter checks must end on a deck report date. NORNE_DEVICE and
NORNE_THREADS have the same meaning as in run_norne.py. A rolling checkpoint
is saved every ten reports by default. This validates the isothermal reservoir
deck, without attaching the separately qualified surface facility model.

Based on the audited launcher used for the complete 2026-10-31 GPU run.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("NORNE_RESUME"):
    raise ValueError("This audit requires a fresh run to establish initial component masses")
requested_days = float(os.environ.get("NORNE_DAYS", "3312"))
if not math.isfinite(requested_days) or requested_days <= 0:
    raise ValueError("NORNE_DAYS must be finite and positive")
OUT = (ROOT / os.environ["NORNE_OUT"]).resolve()
OUT.mkdir(parents=True, exist_ok=False)
SNAPSHOT = OUT / "source_snapshot"
DECK_DIR = (ROOT / os.environ["NORNE_DIR"]).resolve()
manifest_path = OUT / "manifest.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if manifest_path.exists():
    raise RuntimeError(f"Refusing to overwrite an existing validation run: {OUT}")
sources = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "scripts/run_norne.py", Path(__file__).resolve()]
source_hashes = {}
for source in sources:
    relative = source.relative_to(ROOT)
    target = SNAPSHOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hashes[str(relative)] = digest(target)
inputs = [DECK_DIR / "NORNE_ATW2013.DATA"] + sorted(p for p in (DECK_DIR / "INCLUDE").rglob("*") if p.is_file())
references = [DECK_DIR / "ECL.2014.2" / ("NORNE_ATW2013" + ext) for ext in (".SMSPEC", ".UNSMRY")]
sys.path.insert(0, str(SNAPSHOT / "src"))
from simulator.norne_execution import configure_norne_threads
device = torch.device(os.environ.get("NORNE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"))
thread_policy = configure_norne_threads(device)
os.environ["NORNE_DEVICE"] = str(device)
os.environ["NORNE_CONVERGENCE_PROFILE"] = "strict"
os.environ.setdefault("NORNE_CKPT", str(OUT / "latest_checkpoint.pt"))
os.environ.setdefault("NORNE_CKPT_EVERY", "10")
manifest = {
    "started_unix": time.time(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
    "source_files": source_hashes, "input_files": {str(p.relative_to(DECK_DIR)): digest(p) for p in inputs},
    "reference_files": {str(p.relative_to(DECK_DIR)): digest(p) for p in references},
    "profile": "strict", "device": os.environ["NORNE_DEVICE"],
    "threads": torch.get_num_threads(), "torch": torch.__version__, "python": sys.version,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "requested_days": requested_days, "thread_policy": thread_policy,
    "cpr_reuse_graph": os.environ.get("NORNE_CPR_REUSE_GRAPH", "0"),
    "amg_rebuild": os.environ.get("NORNE_AMG_REBUILD", "step"),
    "newton_line_search_fallback": os.environ.get("NORNE_NEWTON_FALLBACK", "0"),
    "scope": "isothermal Norne reservoir deck; no additional physical surface network",
}
manifest_path.write_text(json.dumps(manifest, indent=2))
shutil.copy2(__file__, OUT / "validation_launcher.py")
(OUT / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT))
sys.path.insert(0, str(SNAPSHOT / "src"))
from simulator.blackoil_fim import BlackOilFIMSolver, I_ACC

original_save = torch.save

def save_with_selected_checkpoints(obj, destination, *args, **kwargs):
    original_save(obj, destination, *args, **kwargs)
    if isinstance(obj, dict) and 'sim' in obj and 't' in obj:
        day = float(obj['t'])
        if any(abs(day - target) < 1e-7 for target in (1395, 1821, 2105, 2431)):
            original_save(obj, OUT / f'checkpoint_day_{round(day)}.pt')

torch.save = save_with_selected_checkpoints
original_init = BlackOilFIMSolver.__init__
original_step = BlackOilFIMSolver.step
original_convergence = BlackOilFIMSolver.convergence
audit = {"initial_mass": None, "candidate": None, "accepted": 0, "failed": 0,
         "max_cnv": [0., 0., 0.], "max_mb": [0., 0., 0.]}
audit_file = (OUT / "accepted_substeps.csv").open("w", newline="", buffering=1)
writer = csv.writer(audit_file)
writer.writerow(["day", "dt_days", "newton", *[f"cnv_{p}" for p in ("oil", "water", "gas")],
                 *[f"mb_{p}" for p in ("oil", "water", "gas")], "min_sw", "min_sg", "min_so",
                 *[f"mass_{p}_kg" for p in ("oil", "water", "gas")],
                 *[f"net_surface_{p}_m3" for p in ("oil", "water", "gas")],
                 *[f"balance_rel_{p}" for p in ("oil", "water", "gas")]])


def init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    p = self.fluid.pressure.reshape(-1).double()
    sw = self.fluid.s_w.reshape(-1).double()
    audit["initial_mass"] = self.masses(p, sw, self.x, self.sat, self.ug).sum(1).cpu().numpy()
    assert self.tol_cnv == 1e-3 and self.tol_mb == 1e-7 and self.relaxed_max_pv_fraction == 0


def convergence(self, residual, dt, pv, wsys, p_bh, v=None):
    ok, detail = original_convergence(self, residual, dt, pv, wsys, p_bh, v)
    if ok:
        if detail["cnv_relaxed"] or max(detail["cnv"]) > 1e-3 or max(detail["mb"]) > 1e-7:
            raise RuntimeError("An accepted substep violates strict convergence")
        audit["candidate"] = (detail, v[I_ACC:I_ACC + 3].sum(1).detach().cpu().numpy())
    return ok, detail


def step(self, dt, current_time=None):
    audit["candidate"] = None
    ok = original_step(self, dt, current_time=current_time)
    if not ok:
        audit["failed"] += 1
        return False
    detail, mass = audit["candidate"]
    fields = (self.fluid.s_w, self.fluid.s_g, self.fluid.s_o)
    mins = [float(x.min()) for x in fields]
    if not all(bool(torch.isfinite(x).all()) for x in fields) or min(mins) < -1e-10:
        raise RuntimeError(f"Nonphysical saturation at day {((current_time or 0) + dt) / 86400}: {mins}")
    if max(float(x.max()) for x in fields) > 1 + 1e-10:
        raise RuntimeError("Saturation exceeds one")
    if float((sum(fields) - 1).abs().max()) > 1e-10:
        raise RuntimeError("Phase saturations do not sum to one")
    for name in ("pressure", "rs", "rv"):
        if not bool(torch.isfinite(getattr(self.fluid, name)).all()):
            raise RuntimeError(f"Nonfinite accepted field: {name}")
    net = np.array([sum((getattr(w, "cum_surface", None) or {}).get(ph, 0.0) for w in self.wells.wells)
                    for ph in ("oil", "water", "gas")])
    initial = audit["initial_mass"]
    balance = (mass - initial + net * np.asarray(self.pvt.rho_sc)) / np.maximum(np.abs(initial), 1.0)
    if not np.isfinite(balance).all() or max(abs(balance)) > 1e-4:
        raise RuntimeError(f"Accumulated component mass balance exceeds 1e-4: {balance}")
    audit["accepted"] += 1
    audit["max_cnv"] = np.maximum(audit["max_cnv"], detail["cnv"]).tolist()
    audit["max_mb"] = np.maximum(audit["max_mb"], detail["mb"]).tolist()
    day = ((current_time or 0.0) + dt) / 86400
    writer.writerow([day, dt / 86400, self.last_newton_iters, *detail["cnv"], *detail["mb"],
                     *mins, *mass, *net, *balance])
    status = {k: v for k, v in audit.items() if k not in ("candidate", "initial_mass")}
    from simulator.cpr_gpu import CPRPreconditionerGPU
    status.update(status="running", day=day, balance_rel=balance.tolist(), last_cnv=detail["cnv"], last_mb=detail["mb"],
                  cpr_graphs_enabled=CPRPreconditionerGPU.use_graphs,
                  cuda_reserved_mb=torch.cuda.memory_reserved() / 2**20 if self.fluid.pressure.is_cuda else None)
    workspace = getattr(self, "_cpr_replay_workspace", None)
    status["cpr_replay_workspace"] = workspace.report() if workspace is not None else None
    if self.fluid.pressure.is_cuda:
        status["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 2**20
        status["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    with (OUT / "execution_samples.jsonl").open("a") as samples:
        samples.write(json.dumps({k: status[k] for k in ("day", "cpr_replay_workspace", "cuda_reserved_mb")}
                                 | {k: status[k] for k in ("cuda_allocated_mb", "cuda_peak_allocated_mb") if k in status}) + "\n")
    diagnostic_days = (1668, 1742, 1821, 2002, 2072, 2105, 2177, 2265, 2431, 2552, 2672, 2764)
    if any(abs(day - target) < 1e-6 for target in diagnostic_days):
        arrays = {key: getattr(self.fluid, attr).detach().cpu().numpy()
                  for key, attr in (("p", "pressure"), ("sw", "s_w"), ("sg", "s_g"),
                                    ("so", "s_o"), ("rs", "rs"), ("rv", "rv"))}
        arrays.update({key: getattr(self.wells, key).detach().cpu().numpy()
                       for key in ("conn_cell", "conn_well", "conn_wi", "conn_head")})
        arrays["well_names"] = np.array([w.name for w in self.wells.wells])
        np.savez_compressed(OUT / f"diagnostic_day_{round(day)}.npz", **arrays)
        (OUT / f"well_rates_day_{round(day)}.json").write_text(json.dumps([
            {"name": w.name, "net": getattr(w, "last_surface_rates", {}),
             "gross_production": getattr(w, "last_surface_prod", {})}
            for w in self.wells.wells], indent=2))
    tmp = OUT / "audit_status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2, allow_nan=False))
    tmp.replace(OUT / "audit_status.json")
    return True


BlackOilFIMSolver.__init__ = init
BlackOilFIMSolver.convergence = convergence
BlackOilFIMSolver.step = step
sys.argv = [str(SNAPSHOT / "scripts/run_norne.py"), str(requested_days)]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
    meta = json.loads((OUT / "meta.json").read_text())
    if meta["status"] != "done" or abs(meta["days"] - requested_days) > 1e-7:
        raise RuntimeError("Run did not reach the requested report date")
    for name, expected in source_hashes.items():
        if digest(SNAPSHOT / name) != expected:
            raise RuntimeError(f"Frozen source changed during the run: {name}")
    for path in inputs:
        if digest(path) != manifest["input_files"][str(path.relative_to(DECK_DIR))]:
            raise RuntimeError(f"Deck input changed during the run: {path}")
    for path in references:
        if digest(path) != manifest["reference_files"][str(path.relative_to(DECK_DIR))]:
            raise RuntimeError(f"Reference changed during the run: {path}")
    meta = json.loads((OUT / "meta.json").read_text())
    if meta["status"] != "done" or abs(meta["days"] - requested_days) > 1e-6:
        raise RuntimeError("Requested audit horizon was not completed")
    for source, expected in source_hashes.items():
        if digest(SNAPSHOT / source) != expected:
            raise RuntimeError(f"Frozen source changed: {source}")
    for source, expected in manifest["input_files"].items():
        if digest(DECK_DIR / source) != expected:
            raise RuntimeError(f"Deck input changed during the run: {source}")
    status = json.loads((OUT / "audit_status.json").read_text())
    status.update(status="complete", requested_days=requested_days,
                  source_and_input_hashes_verified=True, wall_s=meta["wall_s"])
    (OUT / "audit_status.json").write_text(json.dumps(status, indent=2, allow_nan=False))
except BaseException as error:
    (OUT / "audit_failure.json").write_text(json.dumps({"status": "failed", "error": repr(error)}, indent=2))
    raise
finally:
    torch.save = original_save
    BlackOilFIMSolver.__init__ = original_init
    BlackOilFIMSolver.convergence = original_convergence
    BlackOilFIMSolver.step = original_step
    audit_file.close()
