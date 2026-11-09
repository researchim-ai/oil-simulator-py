"""Audit saved scaling checkpoints against scalar PR and finer gravity profiles.

This validates FINAL states, source identity and reported global-ledger status.
It does not reconstruct the large-mesh time trajectory or turn a short scaling
run into a long-horizon physical validation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from benchmark_pr_active_surface import build
from check_pr_active_surface import close, column_stock
from check_stratified_pr import Reference
from reference_pr_active_multifluid import packets

from surface.pr_active_surface import PRActiveSurfaceState

ROOT = Path(__file__).resolve().parents[1]


def audit(report):
    for name, digest in report["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Scaling benchmark source/config changed")
    checked = []
    for case in report["cases"]:
        model, _initial = build(case["cells"], case["branches"], "cpu")
        state = PRActiveSurfaceState.from_dict(case["final_state"])
        if (
            state.time_s != case["simulated_duration_s"]
            or len(case["step_seconds"]) * case["physical_step_s"] != state.time_s
        ):
            raise ValueError("Scaling benchmark physical duration differs")
        count = 0
        for name, s in state.pipes:
            packets(model.pipes[name]["pipe"], s.to_dict())
            count += sum(len(c) for c in s.cells)
        for name, s in state.columns:
            column_stock(model.columns[name], s)
        for node, (_, s) in zip(model.equipment.nodes, state.equipment.nodes):
            volumes = []
            energies = []
            for amounts, tag in zip(s.phase_component_moles, s.phase_roots):
                n = np.array(amounts)
                q = Reference(model.fluid, np.flatnonzero(n > 0)).state(
                    s.pressure_pa, s.temperature_k, n / n.sum(), tag
                )
                volumes.append(n.sum() * q["vm"])
                energies.append(n.sum() * q["u"])
            close(sum(volumes), node.volume_m3, atol=1e-9, rtol=1e-9)
            mass = sum(
                v * c.molar_mass_kg_mol for v, c in zip(s.component_moles, model.fluid.components)
            )
            potential = (
                mass * model.gravity * (node.elevation_m - model.equipment.reference_elevation_m)
            )
            close(
                sum(energies) + potential + 0.5 * s.momentum_kg_m_s**2 / mass,
                s.total_energy_j,
                atol=1e-5,
                rtol=1e-9,
            )
        if (
            not case["global_ledger_accepted"]
            or abs(case["independent_energy_residual_j"]) > case["energy_roundoff_limit_j"]
        ):
            raise ValueError("Benchmark declared global energy ledger failed")
        checked.append(
            {
                "cells": case["cells"],
                "independent_phase_eos_checks": count,
                "columns_checked_at_quadrature_order": 32,
                "accepted": True,
            }
        )
    return {
        "accepted": True,
        "scope": "final phase EOS and finer column profiles; source-bound reported global ledger; not trajectory reconstruction",
        "cases": checked,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(json.loads(args.report.read_text()))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
