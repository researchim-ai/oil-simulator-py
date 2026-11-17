"""Forward-autograd retained tensor storages, shared versus repeated lookups.

One real example PVT snapshot, mixed gas/oil regimes. This is not process RSS,
allocator peak, GPU memory or an end-to-end FIM speed benchmark.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from benchmark_journal import BenchmarkJournal

from surface import blackoil_pipe_batch, blackoil_pvto_batch
from surface.blackoil_network import BlackOilProductionNetwork
from surface.blackoil_pressure_lookup import PressureLookup

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ROOT / "configs/fim_blackoil_surface.json"
    spec = json.loads(config.read_text())["simulation"]["physical_surface_network"]["network"]
    model = BlackOilProductionNetwork.from_dict(spec).model
    model.validate(2.4e6, [0.85, 2, 0.06])
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), config, Path(__file__).resolve()]
    hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    journal = BenchmarkJournal(
        args.output,
        {
            "scope": __doc__,
            "source_sha256": before,
            "device": "cpu",
            "streams": 128,
            "points_per_stream": 256,
        },
    )

    class Repeated:
        def __init__(self, *a, **kw):
            self.a, self.kw = a, kw

        def interpolate(self, values):
            return PressureLookup(*self.a, **self.kw).interpolate(values)

    saved = (blackoil_pipe_batch.PressureLookup, blackoil_pvto_batch.PressureLookup)
    records = {}
    outputs = {}
    gradients = {}
    try:
        for name, kind in [("repeated", Repeated), ("shared", PressureLookup)]:
            blackoil_pipe_batch.PressureLookup = kind
            blackoil_pvto_batch.PressureLookup = kind
            p = (
                torch.linspace(1.1e6, 4.5e6, 256, dtype=torch.float64)[None, None, :]
                .repeat(128, 1, 1)
                .requires_grad_(True)
            )
            rates = (
                torch.tensor([[0.85, 2, 0.06]], dtype=torch.float64)
                .repeat(128, 1)
                .requires_grad_(True)
            )
            storages = {}

            def pack(tensor, storages=storages):
                storage = tensor.untyped_storage()
                storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                state = blackoil_pipe_batch.tensor_state(model, p, rates)
                loss = (
                    state["specific_volume"].square().sum() + state["viscosity_pa_s"].square().sum()
                )
            records[name] = {
                "unique_saved_storages": len(storages),
                "unique_saved_storage_bytes_including_borrowed_inputs": sum(storages.values()),
            }
            outputs[name] = {k: v.detach().clone() for k, v in state.items()}
            gradients[name] = tuple(v.detach() for v in torch.autograd.grad(loss, (p, rates)))
        for key in outputs["repeated"]:
            torch.testing.assert_close(
                outputs["repeated"][key], outputs["shared"][key], rtol=0, atol=0
            )
        for a, b in zip(gradients["repeated"], gradients["shared"]):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    finally:
        blackoil_pipe_batch.PressureLookup, blackoil_pvto_batch.PressureLookup = saved
    if hashes() != before:
        raise RuntimeError("Source changed during retained-storage check")
    journal.complete(
        {
            "scope": __doc__,
            "source_sha256": before,
            "sources_unchanged": True,
            "records": records,
            "field_values_bitwise_equal": True,
            "first_derivatives_accepted": True,
            "memory_scope": "Unique storage bytes requested to be saved by forward autograd hooks, including borrowed inputs. Not total peak/RSS/GPU.",
        }
    )


if __name__ == "__main__":
    main()
