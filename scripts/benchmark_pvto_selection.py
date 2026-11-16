"""Synthetic ragged PVTO lookup timing; not a pipe or reservoir simulation.

Exact scalar interpolation ordering, two-column mixture and first derivatives.
Reports an actually allocated all-table matrix, not process peak/RSS savings.
"""

import argparse
import hashlib
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_journal import BenchmarkJournal
from pvto_reference import AllTablesPVTO

from surface.blackoil_properties import BlackOilMixture
from surface.blackoil_pvto_batch import PackedPVTO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if torch.get_num_threads() != 1:
        parser.error("One Torch CPU thread required")
    tables = {}
    for i in range(64):
        axis = torch.linspace(1e5, 5e6, 3 + i % 5, dtype=torch.float64)
        tables[float(i)] = (axis, 1.0 + i * 0.003 + axis * 1e-9, 4.0 - axis * 1e-8 + i * 0.01)
    model = SimpleNamespace(
        oil_tables=tables,
        rho=torch.ones(3, dtype=torch.float64),
        interpolate=BlackOilMixture.interpolate,
    )
    rng = torch.Generator().manual_seed(73)
    p = (1e5 + torch.rand(64, 256, dtype=torch.float64, generator=rng) * 4.9e6).requires_grad_(True)
    weight = torch.rand(64, 1, dtype=torch.float64, generator=rng, requires_grad=True)
    selected = torch.arange(64) % 63
    paths = [
        ROOT / "src/surface/blackoil_pvto_batch.py",
        ROOT / "src/surface/blackoil_properties.py",
        Path(__file__).resolve(),
        Path(__file__).with_name("pvto_reference.py").resolve(),
    ]
    hashes = lambda: {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "seed": 73,
            "device": "cpu",
            "threads": 1,
            "streams": 64,
            "query_points_per_stream": 256,
            "tables": 64,
            "repeats": 5,
            "synthetic_table_recipe": "axis linspace(1e5,5e6,3+i%5); Bo=1+i*.003+p*1e-9; mu=4-p*1e-8+i*.01",
        },
    )

    def evaluate(kernel):
        values = []
        for column in (1, 2):
            lo = kernel.interpolate(p, selected, column)
            hi = kernel.interpolate(p, selected + 1, column)
            values.append(lo * (1 - weight) + hi * weight)
        loss = (values[0].square() + values[1].square()).sum()
        gradients = torch.autograd.grad(loss, (p, weight))
        return values, gradients

    old, new = AllTablesPVTO(model), PackedPVTO(model)
    expected = evaluate(old)
    actual = evaluate(new)
    for a, b in zip(expected[0] + list(expected[1]), actual[0] + list(actual[1])):
        torch.testing.assert_close(a, b, rtol=1e-13, atol=1e-13)
    samples = {"all_tables": [], "selected": []}
    for _ in range(5):
        for name, kernel in [("all_tables", old), ("selected", new)]:
            started = time.perf_counter()
            evaluate(kernel)
            samples[name].append(time.perf_counter() - started)
    if hashes() != before:
        raise RuntimeError("PVTO benchmark source changed")
    medians = {k: statistics.median(v) for k, v in samples.items()}
    journal.complete(
        {
            "scope": __doc__,
            "source_sha256": before,
            "sources_unchanged": True,
            "samples_s": samples,
            "medians_s": medians,
            "speedup_ratio": medians["all_tables"] / medians["selected"],
            "all_table_matrix_allocated_bytes": old.largest_full_table_bytes,
            "selected_output_tensor_bytes": p.numel() * p.element_size(),
            "packed_property_payload_bytes": sum(
                v.numel() * v.element_size() for v in [new.axis, *new.columns, new.lengths]
            ),
            "values_and_derivatives_accepted": True,
            "scope_memory": "Matrix payloads only, not total allocator/autograd/process peak memory.",
        }
    )


if __name__ == "__main__":
    main()
