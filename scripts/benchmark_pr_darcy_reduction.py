"""Conservative reduced Darcy / refined inertial PR comparison after startup.

One synthetic pure-methane case, not a whole-facility speed qualification.
The zero-holdup model does not reproduce the initial acoustic/inertial layer.
"""

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from benchmark_journal import BenchmarkJournal
from check_pr_collocation import thermal_fixture
from check_pr_equilibrium_transport import audit as transport_audit
from check_pr_gravity_transport import audit as gravity_audit
from reduce_pr_recycle import merge_pipe_stocks
from run_pr_equilibrium_transport import run as run_transport
from run_pr_gravity_transport import run as run_gravity

from surface.pr_darcy_connection import PRDarcyConnection
from surface.pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)
from surface.pr_gravity_transport import PRGravityTransportNetwork
from surface.torch_backend import runtime_fingerprint

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {
    "pressure_pa": 2.0,
    "temperature_k": 0.001,
    "component_moles": 1e-5,
    "energy_j": 0.05,
    "coarse_flow_relative": 0.03,
    "refined_flow_relative": 0.01,
}


def fixture(count):
    base, _ = thermal_fixture()
    volume = math.pi * 0.001**2 / 4 * 0.2
    nodes = [PREquilibriumNode("a", 0.1), PREquilibriumNode("b", 0.1)]
    names = [f"p{i}" for i in range(count)]
    nodes += [
        PREquilibriumNode(
            n,
            volume / count,
            "pipe_cell",
            length_m=0.2 / count,
            diameter_m=0.001,
            dynamic_viscosity_pa_s=1e-4,
        )
        for n in names
    ]
    path = ["a", *names, "b"]
    links = [
        PREquilibriumLink(f"face{i}", a, b, 0.001) for i, (a, b) in enumerate(zip(path, path[1:]))
    ]
    model = PREquilibriumTransportNetwork(
        base.fluid,
        nodes,
        links,
        reservoir_ports="stagnation_single_phase",
        initial_step_s=1e-4,
        maximum_step_s=0.002,
    )
    state = model.initialize(
        {
            n.name: {
                "pressure_pa": 1.001e6 if n.name == "a" else 1e6 if n.name == "b" else 1.0005e6,
                "temperature_k": 350.0,
                "mole_fractions": [1.0, 0.0],
            }
            for n in nodes
        }
    )
    return model, state, volume


def requests():
    configs = {}
    for count in (1, 4, 8):
        model, state, volume = fixture(count)
        configs[f"pipe{count}"] = {
            "model": model.to_dict(),
            "initial": {"state": state.to_dict()},
            "schedule": [{"duration_s": 0.02}],
        }
    original, state, volume = fixture(1)
    t = PREquilibriumTransportNetwork(
        original.fluid,
        [PREquilibriumNode("a", 0.1), PREquilibriumNode("b", 0.1 + volume)],
        initial_step_s=1e-4,
        maximum_step_s=0.002,
    )
    old = dict(state.nodes)
    merged = merge_pipe_stocks(t, "b", old["b"], [old["p0"]])
    ts = replace(state, nodes=(("a", old["a"]), ("b", merged)), model_id=t.model_id)
    graph = PRGravityTransportNetwork(
        t, {}, [], darcy_connections=[PRDarcyConnection("line", "a", "b", 0.2, 0.001, 1e-4)]
    )
    gs = graph.initialize(ts, {})
    configs["darcy"] = {
        "model": graph.to_dict(),
        "initial": {"state": gs.to_dict()},
        "schedule": [{"duration_s": 0.02}],
    }
    return configs


def stocks(state):
    nodes = state.get("transport", state)["nodes"]
    return (
        [math.fsum(s["component_moles"][i] for s in nodes.values()) for i in (0, 1)],
        math.fsum(s["total_energy_j"] for s in nodes.values()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Reuse immutable detailed trajectories, with exact request/hash checks and a fresh independent audit",
    )
    args = parser.parse_args()
    if torch.get_num_threads() != 1:
        parser.error("One CPU thread required")
    paths = [*sorted((ROOT / "src/surface").glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = {
        "scope": __doc__,
        "budgets": BUDGETS,
        "source_sha256": hashes,
        "runtime": runtime_fingerprint(),
    }
    reference = None
    if args.reference_dir is not None:
        metadata = args.reference_dir / "metadata.json"
        reference = json.loads(metadata.read_text())["runs"]
        protocol["reference_metadata_sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
        protocol["reference_timing"] = (
            "Original measured CPU times, not rerun times; original manifests in reference metadata"
        )
    journal = BenchmarkJournal(args.output, protocol)
    configs = requests()
    results = {}
    rows = []
    for name in ("pipe1", "pipe4", "pipe8", "darcy"):
        cfg = json.loads(json.dumps(configs[name]))
        cached = reference is not None and name != "darcy"
        if cached:
            archive = args.reference_dir / f"{name}.json.gz"
            assert (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                == reference[name]["archive_sha256"]
            )
            result = json.loads(gzip.decompress(archive.read_bytes()))
            assert result["config"] == cfg, "Reference request differs"
            elapsed = reference[name]["measurement"]["elapsed_s"]
        else:
            start = time.perf_counter()
            result = (run_gravity if name == "darcy" else run_transport)(cfg)
            elapsed = time.perf_counter() - start
            result = json.loads(json.dumps(result, allow_nan=False))
        with gzip.open(args.output / f"{name}.json.gz", "wt") as handle:
            json.dump(result, handle, allow_nan=False)
        checked = (gravity_audit if name == "darcy" else transport_audit)(cfg, result)
        assert checked["accepted"]
        results[name] = result
        last = result["intervals"][-1]["steps"][-1]
        flow = (
            sum(last["transport"]["coupling"]["darcy_connections"]["line"]["component_rates_mol_s"])
            if name == "darcy"
            else sum(last["links"][f"face{name[4:]}"]["flux"][:-2])
        )
        row = {
            "mode": name,
            "elapsed_s": elapsed,
            "accepted_substeps": sum(len(r["steps"]) for r in result["intervals"]),
            "outlet_molar_rate_mol_s": flow,
            "audit": checked,
            "reused_reference": cached,
        }
        rows.append(row)
        journal.checkpoint(rows, [], {})
        print(json.dumps(row), flush=True)
    n0, e0 = stocks(results["pipe1"]["initial"])
    nr, er = stocks(results["darcy"]["initial"])
    assert max(abs(a - b) for a, b in zip(n0, nr)) < 1e-9 and abs(e0 - er) < 1e-6
    comparisons = []
    reduced = results["darcy"]["intervals"][-1]["state"]["transport"]["nodes"]
    q = rows[-1]["outlet_molar_rate_mol_s"]
    for i, name in enumerate(("pipe1", "pipe4", "pipe8")):
        nodes = results[name]["intervals"][-1]["state"]["nodes"]
        error = {
            "pressure_pa": max(
                abs(nodes[n]["pressure_pa"] - reduced[n]["pressure_pa"]) for n in ("a", "b")
            ),
            "temperature_k": max(
                abs(nodes[n]["temperature_k"] - reduced[n]["temperature_k"]) for n in ("a", "b")
            ),
            "component_moles": max(
                abs(a - b)
                for a, b in zip(nodes["a"]["component_moles"], reduced["a"]["component_moles"])
            ),
            "energy_j": abs(nodes["a"]["total_energy_j"] - reduced["a"]["total_energy_j"]),
            "flow_relative": abs(rows[i]["outlet_molar_rate_mol_s"] - q) / abs(q),
        }
        assert all(error[k] <= BUDGETS[k] for k in error if k != "flow_relative"), error
        assert (
            error["flow_relative"]
            <= BUDGETS["coarse_flow_relative" if i == 0 else "refined_flow_relative"]
        ), error
        comparisons.append({"reference": name, "errors": error})
    assert comparisons[1]["errors"]["flow_relative"] < comparisons[0]["errors"]["flow_relative"]
    assert comparisons[2]["errors"]["flow_relative"] < comparisons[1]["errors"]["flow_relative"]
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    journal.complete(
        {
            **protocol,
            "runs": rows,
            "comparisons": comparisons,
            "initial_global_inventory_preserved": True,
        }
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
