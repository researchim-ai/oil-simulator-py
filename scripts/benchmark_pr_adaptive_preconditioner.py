"""End-to-end adaptive Radau3 A/B, including rejected attempts and report intervals.

CPU thermal PR ring and open binary hydraulic graph only. Equipment/gravity
plant adapters and GPU/full-facility performance are outside this protocol.
"""

import argparse
import gc
import gzip
import hashlib
import inspect
import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from audit_pr_collocation import audit_gauss_step
from benchmark_journal import BenchmarkJournal
from benchmark_pr_block_preconditioner import fixture, trace_linear_solves
from check_pr_collocation import ROOT, IsochoricReference
from check_pr_open_transport import audit as open_audit
from scipy.integrate import solve_ivp

from surface import PRCollocation, PREquilibriumTransportState
from surface.pr_collocation import TorchNewton
from surface.pr_open_transport import PROpenTransportNetwork
from surface.torch_backend import runtime_fingerprint
from surface.torch_linear import ColoredJacobian

PAIR_BUDGETS = {
    "pressure_pa": 0.5,
    "temperature_k": 1e-4,
    "wall_temperature_k": 1e-4,
    "component_moles": 1e-9,
    "total_energy_j": 1e-3,
    "momentum_kg_m_s": 1e-9,
}
REFERENCE_BUDGETS = {"temperature_k": 0.005, "wall_temperature_k": 0.005, "pressure_pa": 25.0}
OPEN_CONFIG = ROOT / "configs/surface_pr_open_relaxation.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare(case, enabled):
    if case == "thermal32":
        model, state = fixture(32)
        schedule = [{"duration_s": 0.02}]
        model._collocation_block_preconditioning = enabled
    elif case == "open6":
        config = json.loads(OPEN_CONFIG.read_bytes())
        model = PROpenTransportNetwork.from_dict(config["model"])
        state = model.transport.initialize(config["initial"]["nodes"])
        schedule = config["schedule"]
        model.transport._collocation_block_preconditioning = enabled
    else:
        raise ValueError("Unknown adaptive benchmark")
    return model, state, schedule


@contextmanager
def monitored_solves(case, mode, pair):
    with trace_linear_solves() as rows:
        original = TorchNewton.solve

        def observed(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if len(rows) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "integrating",
                            "case": case,
                            "mode": mode,
                            "pair": pair,
                            "nonlinear_solves_so_far": len(rows),
                            "jacobian_builds_so_far": sum(r["jacobian_builds"] for r in rows),
                        }
                    ),
                    flush=True,
                )
            return result

        with patch.object(TorchNewton, "solve", observed):
            yield rows


def integrate(model, state, schedule):
    closed = not isinstance(model, PROpenTransportNetwork)
    solver = PRCollocation(model, scheme="radau3", phase_fallback=False) if closed else model
    results = []
    for command in schedule:
        before = state
        kwargs = {} if closed else {"integrator": "radau3"}
        report = solver.advance(state, **command, **kwargs)
        state = PREquilibriumTransportState.from_dict(report["state"])
        assert state.time_s == before.time_s + command["duration_s"]
        assert all(s["error_ratio"] <= 1 for s in report["steps"])
        results.append((before, report))
    return results


class ThermalReference(IsochoricReference):
    def __init__(self, model, state):
        super().__init__(model, state)
        assert not model.links and all(n.kind == "vessel" for n in model.nodes)
        assert all(sum(s.component_moles[1:]) == 0 for _, s in state.nodes)
        self.c = np.array([n.wall.heat_capacity_j_k for n in model.nodes])
        self.conductance = np.array([n.wall.fluid_ua_w_k for n in model.nodes])
        assert all(n.wall.ambient_ua_w_k == 0 for n in model.nodes)
        self.ends = [
            (model.names.index(h.source), model.names.index(h.target), h.conductance_w_k)
            for h in model.heat_links
        ]
        self.count = len(model.nodes)
        self.initial = np.array(
            [s.temperature_k for _, s in state.nodes]
            + [s.wall_temperature_k for _, s in state.nodes]
        )

    def rhs(self, _time, y):
        tf, tw = y[: self.count], y[self.count :]
        q = self.conductance * (tw - tf)
        net = np.zeros(self.count)
        for i, j, ua in self.ends:
            heat = ua * (tf[i] - tf[j])
            net[i] -= heat
            net[j] += heat
        return np.r_[(net + q) / (self.n * self.cv(tf)), -q / self.c]

    def solve(self, end):
        solution = solve_ivp(
            self.rhs,
            (0, end),
            self.initial,
            method="Radau",
            rtol=1e-11,
            atol=1e-10,
            dense_output=True,
        )
        tighter = solve_ivp(
            self.rhs,
            (0, end),
            self.initial,
            method="Radau",
            rtol=3e-12,
            atol=3e-11,
            dense_output=True,
        )
        assert solution.success and tighter.success
        probes = np.r_[0.0, np.geomspace(1e-9, end, 200)]
        defect = float(abs(solution.sol(probes) - tighter.sol(probes)).max())
        assert defect < 2e-7, defect
        self.solution = tighter
        self.refinement_defect_k = defect

    def errors(self, results):
        states = [
            PREquilibriumTransportState.from_dict(step["state"])
            for _, report in results
            for step in report["steps"]
        ]
        times = np.array([s.time_s for s in states])
        expected = self.solution.sol(times)
        tf = np.array([[s.temperature_k for _, s in state.nodes] for state in states])
        tw = np.array([[s.wall_temperature_k for _, s in state.nodes] for state in states])
        pressure = np.array([[s.pressure_pa for _, s in state.nodes] for state in states])
        pp = np.stack([self.pressure(row) for row in expected[: self.count].T])
        errors = {
            "temperature_k": float(abs(tf - expected[: self.count].T).max()),
            "wall_temperature_k": float(abs(tw - expected[self.count :].T).max()),
            "pressure_pa": float(abs(pressure - pp).max()),
        }
        assert all(errors[k] <= REFERENCE_BUDGETS[k] for k in errors), errors
        return errors, {
            "time_s": times.tolist(),
            "fluid_temperature_k": expected[: self.count].T.tolist(),
            "wall_temperature_k": expected[self.count :].T.tolist(),
            "pressure_pa": pp.tolist(),
        }


def audit_results(case, model, results, schedule):
    checked = 0
    for command, (before, report) in zip(schedule, results, strict=True):
        if case == "open6":
            assert open_audit(
                model.to_dict(),
                before.to_dict(),
                report,
                **{k: v for k, v in command.items() if k != "duration_s"},
            )["accepted"]
            checked += len(report["steps"])
        else:
            old = before.to_dict()
            for step in report["steps"]:
                assert audit_gauss_step(
                    model.to_dict(), old, step, model.fluid, {"openings": report["openings"]}
                )["accepted"]
                old = step["state"]
                checked += 1
    return checked


def pair_errors(left, right):
    error = {k: 0.0 for k in PAIR_BUDGETS}
    for (_, a), (_, b) in zip(left, right, strict=True):
        assert a["state"]["time_s"] == b["state"]["time_s"]
        assert set(a["state"]["nodes"]) == set(b["state"]["nodes"])
        for name, x in a["state"]["nodes"].items():
            y = b["state"]["nodes"][name]
            for key, previous in error.items():
                xx, yy = x[key], y[key]
                if xx is None or yy is None:
                    assert xx is None and yy is None
                    delta = 0.0
                elif key == "component_moles":
                    delta = max(abs(u - v) for u, v in zip(xx, yy, strict=True))
                else:
                    delta = abs(xx - yy)
                error[key] = max(previous, delta)
    assert all(error[k] <= PAIR_BUDGETS[k] for k in error), error
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", choices=["thermal32", "open6"], default=["thermal32", "open6"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        torch.get_num_threads() != 1
        or not 1 <= args.repeats <= 5
        or len(set(args.cases)) != len(args.cases)
    ):
        parser.error("One CPU thread, unique cases and 1..5 repeats required")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    assert Path(__file__).resolve().parents[1] == ROOT, "Benchmark helper source root differs"
    assert Path(inspect.getfile(PRCollocation)).resolve() == ROOT / "src/surface/pr_collocation.py"
    assert Path(inspect.getfile(ColoredJacobian)).resolve() == ROOT / "src/surface/torch_linear.py"
    probe = ColoredJacobian(1, [0], [0], "cpu", chunk_size=1)
    assert probe.color_count == 1 and not hasattr(probe, "seeds"), "Bounded JVP backend required"
    paths = [
        *sorted((ROOT / "src/surface").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        OPEN_CONFIG,
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    runtime = runtime_fingerprint()
    protocol = {
        "format": "pr_adaptive_preconditioner_v1",
        "scope": __doc__,
        "cases": args.cases,
        "repeats": args.repeats,
        "pair_budgets": PAIR_BUDGETS,
        "independent_ode_budgets": REFERENCE_BUDGETS,
        "runtime": runtime,
        "cpu_threads": 1,
        "source_sha256": hashes,
        "source_root": str(ROOT),
    }
    journal = BenchmarkJournal(args.output, protocol)
    reference = None
    if "thermal32" in args.cases:
        model, state, schedule = prepare("thermal32", True)
        reference = ThermalReference(model, state)
        reference.solve(sum(s["duration_s"] for s in schedule))
        print(
            json.dumps(
                {
                    "phase": "independent_reference_ready",
                    "refinement_defect_k": reference.refinement_defect_k,
                }
            ),
            flush=True,
        )
    rows = []
    comparisons = []
    certificates = {}
    for case in args.cases:
        # Short warmup on separate models, never on a timed model.
        for enabled in (False, True):
            m, s, _commands = prepare(case, enabled)
            integrate(m, s, [{"duration_s": 1e-6}])
        for pair in range(args.repeats):
            paired = {}
            for enabled in (False, True) if pair % 2 == 0 else (True, False):
                mode = "block" if enabled else "diagonal"
                model, state, schedule = prepare(case, enabled)
                gc.collect()
                with monitored_solves(case, mode, pair) as trace:
                    start = time.perf_counter()
                    results = integrate(model, state, schedule)
                    elapsed = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "phase": "integration_finished",
                            "case": case,
                            "pair": pair,
                            "mode": mode,
                            "elapsed_s": elapsed,
                        }
                    ),
                    flush=True,
                )
                payload = {
                    "case": case,
                    "model": model.to_dict(),
                    "initial": state.to_dict(),
                    "schedule": schedule,
                    "reports": [r for _, r in results],
                }
                fingerprint = digest(payload)
                reused = fingerprint in certificates
                if not reused:
                    checked = audit_results(case, model, results, schedule)
                    error, refstates = (
                        reference.errors(results) if case == "thermal32" else (None, None)
                    )
                    certificates[fingerprint] = {
                        "first_run": f"{case}_pair{pair}_{mode}",
                        "independently_audited_steps": checked,
                        "errors_vs_independent_ode": error,
                        "reference_states": refstates,
                    }
                cert = certificates[fingerprint]
                row = {
                    "case": case,
                    "mode": mode,
                    "pair": pair,
                    "elapsed_s": elapsed,
                    "model_time_s": sum(s["duration_s"] for s in schedule),
                    "accepted_substeps": sum(len(r["steps"]) for _, r in results),
                    "attempts": sum(r["attempts"] for _, r in results),
                    "rejected_attempts": sum(r["rejected_attempts"] for _, r in results),
                    "fallback_intervals": sum(r["fallback_intervals"] for _, r in results),
                    "jacobian_builds": sum(r["jacobian_builds"] for r in trace),
                    "nonlinear_solves": len(trace),
                    "trajectory_sha256": fingerprint,
                    "audit_certificate": fingerprint,
                    "audit_exact_replay": reused,
                    "errors_vs_independent_ode": cert["errors_vs_independent_ode"],
                }
                rows.append(row)
                paired[mode] = results
                with gzip.open(args.output / f"{case}_pair{pair}_{mode}.json.gz", "wt") as f:
                    json.dump(
                        {
                            **payload,
                            "measurement": row,
                            "linear_solves": trace,
                            "protocol": protocol,
                        },
                        f,
                        allow_nan=False,
                    )
                journal.checkpoint(rows, comparisons, certificates)
                print(json.dumps(row), flush=True)
            comparisons.append(
                {
                    "case": case,
                    "pair": pair,
                    "errors_at_report_times": pair_errors(paired["diagonal"], paired["block"]),
                }
            )
            journal.checkpoint(rows, comparisons, certificates)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    medians = {
        case: {
            mode: statistics.median(
                r["elapsed_s"] for r in rows if r["case"] == case and r["mode"] == mode
            )
            for mode in ("diagonal", "block")
        }
        for case in args.cases
    }
    summary = {
        "scope": __doc__,
        "repeats": args.repeats,
        "median_elapsed_s": medians,
        "speedup": {case: v["diagonal"] / v["block"] for case, v in medians.items()},
        "pair_budgets": PAIR_BUDGETS,
        "independent_ode_budgets": REFERENCE_BUDGETS,
        "reference_refinement_defect_k": None
        if reference is None
        else reference.refinement_defect_k,
        "runs": rows,
        "comparisons": comparisons,
        "audit_certificates": certificates,
        "runtime": runtime,
        "cpu_threads": 1,
        "source_sha256": hashes,
        "source_root": str(ROOT),
    }
    journal.complete(summary)
    print(json.dumps({"medians": medians, "speedup": summary["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
