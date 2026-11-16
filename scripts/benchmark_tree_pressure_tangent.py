"""Physical dense/compact tree comparison plus a separate deep algebraic probe.

Local black-oil pipe laws are identical. This does not qualify the dense FIM
well-block limit or claim a multi-GPU implementation.
"""

import argparse
import copy
import gzip
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from benchmark_journal import BenchmarkJournal

from surface.blackoil_network import BlackOilProductionNetwork
from surface.process import ComponentStream
from surface.tree_pressure_tangent import TreePressureTangent

ROOT = Path(__file__).resolve().parents[1]


def tensor_storage(operator):
    seen = set()

    def size(value):
        if torch.is_tensor(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            return value.numel() * value.element_size()
        if isinstance(value, (tuple, list)):
            return sum(size(x) for x in value)
        return 0

    return sum(size(v) for v in vars(operator).values())


def physical_fixture(config, count):
    c = copy.deepcopy(config["simulation"]["physical_surface_network"]["network"])
    c["nodes"] = [{"name": f"P{i}"} for i in range(count)] + [
        {"name": "header"},
        {"name": "outlet", "pressure_pa": 1e6},
    ]
    c["pipes"] = [
        {
            "name": f"branch{i}",
            "start": f"P{i}",
            "end": "header",
            "length_m": 100,
            "diameter_m": 0.025,
            "roughness_m": 1e-5,
        }
        for i in range(count)
    ]
    c["pipes"].append(
        {
            "name": "collector",
            "start": "header",
            "end": "outlet",
            "length_m": 700,
            "diameter_m": 0.04,
            "roughness_m": 1e-5,
        }
    )
    c.pop("valves", None)
    sources = {
        f"P{i}": ComponentStream(0.00007 / count, 0.00003 / count, 0.002 / count)
        for i in range(count)
    }
    return c, sources


def deep_probe(count):
    spec = {
        "model": "blackoil_tree_pressure_tangent_v1",
        "node_names": [f"n{i}" for i in range(count)],
        "parent_names": [f"n{i + 1}" for i in range(count - 1)] + [None],
        "source_names": [f"n{i}" for i in range(count)],
        "pressure_coefficients": [1.0] * (count - 1) + [0.0],
        "mass_coefficients": [[1.0, 2.0, 3.0]] * (count - 1) + [[0.0, 0.0, 0.0]],
        "valve_names": [],
        "valve_nodes": [],
        "valve_coefficients": [],
    }
    started = time.perf_counter()
    t = TreePressureTangent(spec)
    construction = time.perf_counter() - started
    q = torch.zeros(count, 3, dtype=torch.float64)
    q[0, 1] = 1
    expected = 2 * torch.arange(count - 1, -1, -1, dtype=torch.float64)
    torch.testing.assert_close(t.matvec(q), expected, atol=0, rtol=0)
    z = torch.ones(count, dtype=torch.float64)
    torch.testing.assert_close(
        (t.matvec(q) * z).sum(), (q * t.rmatvec(z)).sum(), rtol=1e-12, atol=1e-8
    )
    samples = []
    for _ in range(10):
        start = time.perf_counter()
        t.matvec(q)
        t.rmatvec(z)
        samples.append(time.perf_counter() - start)
    return {
        "scope": "Algebra only, no PVT solve or reservoir integration",
        "nodes": count,
        "sources": count,
        "tensor_storage_bytes": tensor_storage(t),
        "hypothetical_dense_float64_bytes": count * count * 3 * 8,
        "dense_allocated": False,
        "construction_s": construction,
        "median_jvp_plus_vjp_s": statistics.median(samples),
        "closed_form_and_adjoint_accepted": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=256)
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--deep-nodes", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sources < 2 or args.deep_nodes < 2 or torch.get_num_threads() != 1:
        parser.error("At least two sources/nodes and OMP_NUM_THREADS=1 required")
    cfg = ROOT / "configs/fim_blackoil_facility_coupled.json"
    config = json.loads(cfg.read_text())
    spec, sources = physical_fixture(config, args.sources)
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__), cfg]
    fingerprints = lambda: {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths
    }
    before = fingerprints()
    protocol = {
        "scope": __doc__,
        "source_sha256": before,
        "physical_sources": args.sources,
        "repeats": args.repeats,
        "device": "cpu",
        "cpu_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "timing": "Steady physical solve + compact tangent construction; JSON/gzip/row extraction/audits excluded",
    }
    journal = BenchmarkJournal(args.output, protocol)
    rows, comparisons = [], []
    for pair in range(args.repeats):
        matched = {}
        for mode in ("dense", "operator") if pair % 2 == 0 else ("operator", "dense"):
            net = BlackOilProductionNetwork.from_dict(spec)
            print(json.dumps({"event": "run", "pair": pair, "mode": mode}), flush=True)
            start = time.perf_counter()
            r = net.solve(sources, sensitivity_mode=mode)
            tangent = (
                TreePressureTangent(r["pressure_linearization"]) if mode == "operator" else None
            )
            elapsed = time.perf_counter() - start
            path = args.output / f"pair{pair}_{mode}.json.gz"
            with gzip.open(path, "wt") as handle:
                json.dump(
                    {
                        "network": spec,
                        "source_streams": {n: vars(s) for n, s in sources.items()},
                        "result": r,
                    },
                    handle,
                    allow_nan=False,
                )
            matched[mode] = (r, tangent)
            row = {
                "pair": pair,
                "mode": mode,
                "elapsed_s": elapsed,
                "archive": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "maximum_component_residual_kg_s": max(
                    abs(v)
                    for d in r["node_component_mass_residuals_kg_s"].values()
                    for v in d.values()
                ),
            }
            if tangent is not None:
                row["operator_tensor_bytes"] = tensor_storage(tangent)
            rows.append(row)
            journal.checkpoint(rows, comparisons, {})
            print(json.dumps(row), flush=True)
        dense, _ = matched["dense"]
        compact, tangent = matched["operator"]
        assert dense["node_pressures_pa"] == compact["node_pressures_pa"]
        selected = [
            f"P{i}" for i in sorted({0, args.sources // 3, 2 * args.sources // 3, args.sources - 1})
        ]
        actual = tangent.jacobian_rows(selected)
        expected = torch.tensor(
            [
                [
                    [
                        dense["pressure_derivatives_per_kg_s"][n][s][c]
                        for c in ("oil", "water", "gas")
                    ]
                    for s in tangent.source_names
                ]
                for n in selected
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-7)
        comparison = {
            "pair": pair,
            "accepted": True,
            "pressures_identical": True,
            "selected_rows_maximum_absolute_difference": float((actual - expected).abs().max()),
            "relative_budget": 1e-11,
            "absolute_budget": 1e-7,
        }
        comparisons.append(comparison)
        journal.checkpoint(rows, comparisons, {})
    probe = deep_probe(args.deep_nodes)
    if fingerprints() != before:
        raise RuntimeError("Source changed during benchmark")
    summary = {
        **protocol,
        "runs": rows,
        "comparisons": comparisons,
        "deep_probe": probe,
        "median_elapsed_s": {
            m: statistics.median(r["elapsed_s"] for r in rows if r["mode"] == m)
            for m in ("dense", "operator")
        },
        "sources_unchanged": True,
    }
    journal.complete(summary)
    print(json.dumps({k: summary[k] for k in ("median_elapsed_s", "deep_probe")}, indent=2))


if __name__ == "__main__":
    main()
