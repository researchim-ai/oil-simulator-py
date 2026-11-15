"""CPU memory/work comparison of legacy all-color and bounded sparse JVPs.

Synthetic sparse operators isolate derivative workspace, not facility physics.
Largest dispatched tensor is an allocation-shape check, NOT process peak RSS.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from surface.torch_backend import runtime_fingerprint
from surface.torch_linear import ColoredJacobian

ROOT = Path(__file__).resolve().parents[1]


class AllocationShapes(TorchDispatchMode):
    def __init__(self):
        self.largest_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))

        def visit(value):
            if torch.is_tensor(value):
                self.largest_bytes = max(self.largest_bytes, value.numel() * value.element_size())
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(result)
        return result


def case(kind, size):
    rows, cols = list(range(size)), list(range(size))
    x = torch.linspace(0.1, 1.1, size, dtype=torch.float64)
    if kind == "hub":
        rows = [0] * size + list(range(1, size))
        cols = list(range(size)) + list(range(1, size))

        def fn(y):
            return torch.cat((y.sin().sum().reshape(1), y[1:].square() + 2 * y[1:]))

        expected = torch.cat((x.cos(), 2 * x[1:] + 2))
    elif kind == "line":
        rows += list(range(size - 1))
        cols += list(range(1, size))

        def fn(y):
            return y.square() + torch.cat((0.2 * y[1:], y.new_zeros(1)))

        expected = torch.cat((2 * x, x.new_full((size - 1,), 0.2)))
    else:
        raise ValueError("Unknown graph")
    return rows, cols, x, fn, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if torch.get_num_threads() != 1 or not 1 <= args.repeats <= 5:
        parser.error("One CPU thread and 1..5 repeats required")
    if args.output.exists():
        parser.error("Choose a fresh output file")
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    records = []
    for kind, size in (("hub", 512), ("hub", 2048), ("line", 10000), ("line", 100000)):
        rows, cols, x, fn, expected = case(kind, size)
        plan = ColoredJacobian(size, rows, cols, "cpu")
        # Legacy constructor materialized all seed directions. Preparation is
        # outside assembly timing for both; include its shape in allocation audit.
        with AllocationShapes() as legacy_allocations:
            seeds = torch.nn.functional.one_hot(plan.color, plan.color_count).T.to(x.dtype)

            def legacy(plan=plan, x=x, fn=fn, seeds=seeds):
                compressed = torch.vmap(lambda seed: torch.func.jvp(fn, (x,), (seed,))[1])(seeds)
                return compressed[plan.color[plan.c], plan.r]

            old = legacy()
        with AllocationShapes() as bounded_allocations:
            new = plan.values(fn, x)
        assert torch.equal(new, expected) and torch.equal(old, expected)
        times = {"legacy": [], "bounded": []}
        for pair in range(args.repeats):
            modes = ("legacy", "bounded") if pair % 2 == 0 else ("bounded", "legacy")
            for mode in modes:
                gc.collect()
                start = time.perf_counter()
                actual = legacy() if mode == "legacy" else plan.values(fn, x)
                times[mode].append(time.perf_counter() - start)
                assert torch.equal(actual, expected)
        record = {
            "graph": kind,
            "unknowns": size,
            "nonzeros": len(rows),
            "colors": plan.color_count,
            "direction_batch_limit": plan.chunk_size,
            "direction_batches": len(plan.packs),
            "legacy_persistent_seed_entries": seeds.numel(),
            "bounded_persistent_seed_entries": 0,
            "legacy_largest_dispatched_tensor_bytes": legacy_allocations.largest_bytes,
            "bounded_largest_dispatched_tensor_bytes": bounded_allocations.largest_bytes,
            "maximum_direction_buffer_entries": min(plan.color_count, plan.chunk_size) * size,
            "all_values_exact_analytic_and_legacy_match": True,
            "assembly_s": times,
            "median_assembly_s": {k: statistics.median(v) for k, v in times.items()},
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del legacy, seeds, old, new, actual
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result = {
        "scope": __doc__,
        "runtime": runtime_fingerprint(),
        "cpu_threads": 1,
        "repeats": args.repeats,
        "cases": records,
        "source_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
