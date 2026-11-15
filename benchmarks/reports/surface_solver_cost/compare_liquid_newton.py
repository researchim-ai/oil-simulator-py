"""Alternating old/new CPU Newton on the same audited liquid integration."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts")]
import torch
from check_liquid_transient import coupled_model
from surface.torch_liquid_bridge import TorchLiquidBridge
from surface.torch_linear import TorchNewton as Legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("positive pair count required")
    torch.set_num_threads(1)
    def run(legacy, horizon):
        m = coupled_model(.001)
        initial = m.initialize({"h":550000., "t":500000.}, {"line":.002}, {"a":100., "b":110.})
        m._torch_bridge = TorchLiquidBridge(m, "cpu")
        e = m._torch_bridge.equations
        if legacy:
            e.newton = Legacy("cpu")
        solve = e.newton.solve
        evaluations = []
        def measured(*a, **kw):
            answer = solve(*a, **kw)
            evaluations.append(answer[3])
            return answer
        e.newton.solve = measured
        start = time.perf_counter()
        result = m.advance(initial, horizon, motor_torques_nm={"a":5., "b":7.})
        elapsed = time.perf_counter()-start
        return dict(policy="legacy" if legacy else "modern", elapsed_s=elapsed,
                    nonlinear_evaluations=sum(evaluations), nonlinear_solves=len(evaluations),
                    max_energy_residual_j=max(abs(s["energy_residual_j"]) for s in result["steps"]),
                    endpoint=result["state"],
                    accepted_state_sha256=hashlib.sha256(json.dumps([
                        [s["final_pressures_pa"], s["final_pipe_volume_rates_m3_s"], s["final_speeds_rad_s"]]
                        for s in result["steps"]], sort_keys=True).encode()).hexdigest())
    run(True, .005)
    run(False, .005)
    samples = []
    for pair in range(args.pairs):
        for legacy in ((True, False) if pair%2 == 0 else (False, True)):
            row = run(legacy, 1.)
            row["pair"] = pair
            samples.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "endpoint"}), flush=True)
    assert len({r["accepted_state_sha256"] for r in samples}) == 1
    medians = {p:statistics.median(r["elapsed_s"] for r in samples if r["policy"] == p)
               for p in ("legacy", "modern")}
    output = dict(scope="Same audited one-second liquid midpoint integration, CPU float64, one thread",
                  medians_s=medians, results=samples,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT/"src/surface").glob("*.py"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
