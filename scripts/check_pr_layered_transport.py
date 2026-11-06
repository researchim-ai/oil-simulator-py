"""Independent per-cell layered N/E/momentum audit; no runtime transport solve."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from check_stratified_pr_flow import close
from reference_pr_layered_transport import reconstruct


def audit(config, report):
    expected_model = dict(config["model"])
    expected_model.setdefault("dynamic_viscosity_pa_s", 0.0)
    expected_model.setdefault("roughness_m", 0.0)
    if config != report["config"] or expected_model != report["model"]:
        raise ValueError("Layered config/model differs")
    if len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Layered schedule length differs")
    model = report["model"]
    old = report["initial"]
    if set(config["initial"]) == {"state"} and config["initial"]["state"] != old:
        raise ValueError("Restart checkpoint differs")
    checked = 0
    max_source = 0.0
    # Initial conserved profile is independently checked as well.
    reconstruct(model, old)
    for command, interval in zip(config["schedule"], report["intervals"]):
        elapsed = 0.0
        heat = command.get("heat_w", [0.0] * len(model["columns"]))
        if isinstance(heat, (float, int)):
            heat = [heat] * len(model["columns"])
        heat = np.array(heat)
        for step in interval["steps"]:
            new = step["state"]
            dt = step["dt_s"]
            if not dt > 0:
                raise ValueError("Nonpositive layered step")
            close(new["time_s"] - old["time_s"], dt, atol=1e-12, rtol=1e-10)
            decoded = reconstruct(model, new)
            close(step["heat_j"], dt * heat, atol=1e-9, rtol=1e-12)
            close(decoded["source"], step["source_n"], atol=2e-7, rtol=1e-5)
            close(decoded["source"], decoded["gravity"], atol=2e-7, rtol=1e-4)
            max_source = max(
                max_source, float(np.max(np.abs(decoded["source"] - decoded["gravity"])))
            )
            rates = decoded["rates"]
            close(decoded["friction_force_n"], step["friction_force_n"], atol=1e-8, rtol=1e-6)
            wall_pressure = decoded["wall_pressure"]
            close(wall_pressure, step["wall_pressure"], atol=2e-7, rtol=1e-5)
            for i, (a, b) in enumerate(zip(old["cells"], new["cells"])):
                na = np.asarray(a["profile"]["component_moles"])
                nb = np.asarray(b["profile"]["component_moles"])
                close(nb - na, dt * rates[i, :-2], atol=1e-8, rtol=2e-5)
                ea = a["total_energy_j"]
                eb = b["total_energy_j"]
                absolute_rate = (
                    rates[i, -1]
                    + rates[i, :-2]
                    @ decoded["molar_masses"]
                    * decoded["gravity_acceleration"]
                    * decoded["datum"]
                )
                close(eb - ea, dt * (absolute_rate + heat[i]), atol=2e-5, rtol=2e-5)
                kinetic = 0.5 * b["momentum_kg_m_s"] ** 2 / decoded["mass"][i]
                close(eb, b["profile"]["total_energy_j"] + kinetic, atol=1e-7, rtol=1e-12)
                wall = wall_pressure[i, 0] - wall_pressure[i, 1]
                impulse = dt * (
                    rates[i, -2] + wall + decoded["source"][i] + decoded["friction_force_n"][i]
                )
                close(b["momentum_kg_m_s"] - a["momentum_kg_m_s"], impulse, atol=2e-8, rtol=2e-5)
            old = new
            elapsed += dt
            checked += 1
        close(elapsed, command["duration_s"], atol=1e-12, rtol=1e-10)
        # Adaptive next-step metadata may differ from the last accepted half.
        if {k: v for k, v in old.items() if k != "next_step_s"} != {
            k: v for k, v in interval["state"].items() if k != "next_step_s"
        }:
            raise ValueError("Interval state differs from accepted halves")
        old = interval["state"]
    if old != report["state"]:
        raise ValueError("Final layered state differs")
    return {
        "passed": True,
        "accepted": True,
        "checked_steps": checked,
        "maximum_source_quadrature_error_n": max_source,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    if not report.get("sources_unchanged_during_run") or not report.get("source_sha256"):
        raise ValueError("Missing source-bound runtime provenance")
    for name, digest in report["source_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Runtime source hash differs")
    print(json.dumps(audit(config, report), indent=2))


if __name__ == "__main__":
    main()
