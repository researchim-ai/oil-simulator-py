"""Export a completed facility forecast to the local engineering viewer.

Never reruns physics, invents intermediate reservoir fields, or modifies sources.
Time sampling selects actual accepted states; component rates remain kg/s.
"""

import argparse
import hashlib
import json
from pathlib import Path

from check_facility_refinement_pair import completed
from check_fim_facility_forecast import audit


def validate_topology(config):
    """The first viewer adapter draws this declared facility family only."""
    surface = config["simulation"]["physical_surface_network"]
    network = surface["network"]
    plant = config["simulation"]["coupled_facility"]["facility"]
    wells = config["wells"]
    if (
        network["model"] != "homogeneous_blackoil_tree"
        or len(plant["separator_stages"]) != 2
        or len(plant["water_network"]["pumps"]) != 1
        or "gas_export" not in plant
        or not 1 <= len(wells) <= 16
    ):
        raise ValueError(
            "Viewer v1 supports up to 16 producers, two separator stages, one water pump and gas export"
        )
    expected = {(node, "header") for node in surface["well_nodes"].values()} | {
        ("header", "receiver")
    }
    actual = {(p["start"], p["end"]) for p in network["pipes"]}
    if (
        actual != expected
        or len(network["pipes"]) != len(expected)
        or set(surface["well_nodes"]) != {w["name"] for w in wells}
        or any(w["type"] != "producer" for w in wells)
    ):
        raise ValueError("Viewer v1 requires the producer/header/receiver gathering tree")


def build(directory, max_frames=160):
    if type(max_frames) is not int or max_frames < 2:
        raise ValueError("max_frames must be an integer >= 2")
    directory = Path(directory)
    meta = json.loads((directory / "protocol.json").read_text())
    summary, bundle, _ = completed(directory, meta["days"])
    c, r = bundle["config"], bundle["result"]
    checked = audit(
        c, r, expected_schedule=bundle["schedule"], expected_days=bundle["requested_days"]
    )
    validate_topology(c)
    steps = r["accepted_substeps"]
    if not steps:
        raise ValueError("No accepted states")
    # Uniform in accepted-state index preserves startup resolution. Always keep
    # adjacent states at control events and changes of well operating mode.
    selected = {
        round(i * (len(steps) - 1) / (min(max_frames, len(steps)) - 1))
        for i in range(min(max_frames, len(steps)))
    }
    previous = None
    for i, step in enumerate(steps):
        modes = tuple((n, w["mode"]) for n, w in step["surface"]["wells"].items())
        if previous is not None and modes != previous:
            selected.update((i - 1, i))
        previous = modes
    for event in bundle["schedule"]:
        t = event["time_days"]
        i = next((i for i, s in enumerate(steps) if s["time_days"] >= t), len(steps) - 1)
        selected.update((max(0, i - 1), i))
    frames = []
    for i in sorted(selected):
        s = steps[i]
        coupling = s["facility_coupling"]
        f = coupling["facility"]
        net = s["surface"]["network"]
        frames.append(
            {
                "time_days": s["time_days"],
                "dt_s": s["dt_s"],
                "source_step": i,
                "newton_iterations": s["newton_iterations"],
                "wells": s["surface"]["wells"],
                "node_pressures_pa": net["node_pressures_pa"],
                "pipes": {
                    k: {
                        key: v
                        for key, v in p.items()
                        if key
                        in (
                            "inlet_pressure_pa",
                            "outlet_pressure_pa",
                            "component_mass_rates_kg_s",
                            "inlet_state",
                            "outlet_state",
                            "friction_dissipation_w",
                            "mechanical_balance_error_w",
                            "model",
                        )
                    }
                    for k, p in net["pipes"].items()
                },
                "facility": {
                    k: f[k]
                    for k in (
                        "stages",
                        "products",
                        "gas_export",
                        "network",
                        "measurements",
                        "limits",
                        "feasible",
                        "violations",
                        "component_mass_residuals_kg_s",
                    )
                },
                "control": {
                    "fraction": coupling["fraction"],
                    "evaluations": coupling["evaluations"],
                    "reservoir_evaluations": coupling.get(
                        "reservoir_evaluations", coupling["evaluations"]
                    ),
                    "trace": [
                        {
                            k: v
                            for k, v in row.items()
                            if k
                            in (
                                "fraction",
                                "feasible",
                                "constraint_margins",
                                "certified_from_trace_index",
                                "capacity_proposal",
                            )
                        }
                        for row in coupling["trace"]
                    ],
                },
                "withdrawn_component_mass_kg": s["withdrawn_component_mass_kg"],
            }
        )
    return {
        "schema": "oil-engineering-viewer/v1",
        "title": "Связанный прогноз · " + "×".join(map(str, c["reservoir"]["dimensions"])),
        "provenance": {
            "run": "/".join(directory.parts[-2:]),
            "trajectory_sha256": summary["trajectory_sha256"],
            "status": summary["status"],
            "source_guard": summary["sources_unchanged"],
            "audit": checked,
            "device": summary["device"],
            "solve_wall_s": summary["solve_wall_s"],
            "accepted_steps": len(steps),
            "displayed_frames": len(frames),
            "sampling": "Actual accepted states; index sampling plus operating mode/event boundaries",
            "scope": "Synthetic isothermal black-oil + homogeneous gathering tree + quasi-steady plant. Not Norne.",
        },
        "topology": {
            "gathering": c["simulation"]["physical_surface_network"]["network"],
            "well_nodes": c["simulation"]["physical_surface_network"]["well_nodes"],
            "wells": c["wells"],
            "reservoir": c["reservoir"],
            "facility": c["simulation"]["coupled_facility"]["facility"],
        },
        "frames": frames,
        "field": {
            "time_days": r["elapsed_days"],
            "dimensions": c["reservoir"]["dimensions"],
            "layout": "C-order [i,j,k]; flat=(i*ny+j)*nz+k",
            "pressure_pa": r["final_pressure_pa"],
            "s_w": r["final_s_w"],
            "s_g": r["final_s_g"],
            "rs": r["final_rs"],
        },
        "totals": {
            "initial_component_mass_kg": r["initial_component_mass_kg"],
            "final_component_mass_kg": r["final_component_mass_kg"],
            "withdrawn_component_mass_kg": r["withdrawn_component_mass_kg"],
            "mass_balance_error_kg": r["mass_balance_error_kg"],
        },
        "schedule": bundle["schedule"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("forecast", type=Path)
    p.add_argument("--output", type=Path, default=Path("apps/fieldroom/data/engineering-case.json"))
    p.add_argument("--max-frames", type=int, default=160)
    args = p.parse_args()
    data = build(args.forecast, args.max_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    args.output.write_text(raw + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": len(raw.encode()),
                "frames": len(data["frames"]),
                "sha256": hashlib.sha256((raw + "\n").encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
