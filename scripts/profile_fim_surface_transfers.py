"""Count CUDA host interactions in assembly/Schur; profiling is not a speed benchmark."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simulator.blackoil_fim import BlackOilFIMSolver
from simulator.fluid import Fluid
from simulator.reservoir import Reservoir
from simulator.well import WellManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    c = json.loads((ROOT / "configs/fim_blackoil_surface.json").read_text())
    c["fluid"]["pvt_path"] = str(ROOT / c["fluid"]["pvt_path"])
    dev = torch.device("cuda")
    r = Reservoir(c["reservoir"], device=dev)
    f = Fluid(c["fluid"], r, device=dev)
    s = BlackOilFIMSolver(r, f, WellManager(c["wells"], r), c["simulation"])
    p = f.pressure.flatten().clone()
    sw = f.s_w.flatten().clone()
    x = s.x.clone()
    state = s.cell_state(p, sw, x, s.sat, s.ug)
    old = s.masses(p, sw, x, s.sat, s.ug)
    s.wells.begin_step(*state, 0)
    pb = p.new_full((2,), 2.4e6)
    original = s.wells.assemble

    def wrapped(*a, **kw):
        with record_function("oil.surface_assembly"):
            return original(*a, **kw)

    s.wells.assemble = wrapped

    def execute():
        R, J, ws = s.assemble_bo(p, sw, x, s.sat, old, 86400, pb, state, s.ug)
        with record_function("oil.convergence"):
            s.convergence(R, 86400, s._pv(p), ws, pb, state[0])
        with record_function("oil.schur_prepare"):
            s.wells.add_local_schur(ws, J)
        with record_function("oil.well_report"):
            s._store_well_results(ws, pb, 86400)
        with record_function("oil.schur_matvec"):
            for _ in range(3):
                s.wells.schur_correction(ws, torch.ones_like(p).repeat_interleave(3).view(-1, 3))

    execute()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
    ) as prof:
        execute()
        with record_function("oil.profiler_drain"):
            torch.cuda.synchronize()
    counters = {
        name: Counter()
        for name in [
            "surface",
            "convergence",
            "schur_prepare",
            "well_report",
            "schur_matvec",
            "other_assembly",
        ]
    }
    paths = Counter()
    for event in prof.events():
        chain = []
        parent = event
        while parent is not None:
            chain.append(parent.name)
            parent = parent.cpu_parent
        if "oil.profiler_drain" in chain:
            continue
        group = (
            "surface"
            if "oil.surface_assembly" in chain
            else "convergence"
            if "oil.convergence" in chain
            else "schur_prepare"
            if "oil.schur_prepare" in chain
            else "well_report"
            if "oil.well_report" in chain
            else "schur_matvec"
            if "oil.schur_matvec" in chain
            else "other_assembly"
        )
        name = event.name
        if name in (
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyAsync",
            "cudaMemcpy",
            "aten::_local_scalar_dense",
            "aten::item",
        ):
            counters[group][name] += 1
            if name in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaMemcpyAsync"):
                paths[" > ".join(reversed(chain[:8]))] += 1
    sources = ["src/simulator/blackoil_fim.py", "src/simulator/device_exchange.py", "src/simulator/blackoil_surface_coupling.py", "src/simulator/full_well_schur.py"]
    result = {
        "device": torch.cuda.get_device_name(0),
        "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "counts": {k: dict(v) for k, v in counters.items()},
        "cuda_api_call_paths": dict(paths),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "scope": "one warmed assembly and three Schur matvecs on 3-cell case; counts include profiler API instrumentation; not a wall-time benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.trace:
        prof.export_chrome_trace(str(args.trace))
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
