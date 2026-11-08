"""Independent spatial path balances and composed PR source-stage audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_pr_interphase_relaxation import audit as audit_source
from check_stratified_pr_flow import close
from reference_pr_multifluid_pipe import packets, reconstruct

from surface.pr_multifluid_pipe import PRMultifluidPipe


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Multifluid request differs")
    if config["model"].get("model") == "pr_equilibrium_euler_plant_v1":
        from check_pr_equilibrium_euler_plant import audit as audit_plant

        return audit_plant(config, report)
    if config["model"].get("model") == "pr_equilibrium_euler_pipe_v1":
        from check_pr_equilibrium_euler_pipe import audit as audit_active

        return audit_active(config, report)
    from surface.pr_multifluid_network import PRMultifluidNetwork

    graph = config["model"].get("model") in (
        "pr_multifluid_network_v1",
        "pr_multifluid_network_v2",
        "pr_multifluid_network_v3",
        "pr_multifluid_network_v4",
    )
    factory = PRMultifluidNetwork if graph else PRMultifluidPipe
    request_model = config["model"]
    balanced = request_model.get("model") == "pr_multifluid_network_v3"
    if balanced:
        request_model = {k: v for k, v in request_model.items() if k != "hydrostatic_reference"}
        request_model["model"] = "pr_multifluid_network_v2"
    parsed = factory.from_dict(request_model)
    if graph:
        from reference_pr_multifluid_network import reconstruct as reconstruct_spatial
    else:
        reconstruct_spatial = reconstruct
    model = json.loads(json.dumps(parsed.to_dict()))
    if balanced:
        from surface.operation import _digest

        model.update(
            model="pr_multifluid_network_v3",
            hydrostatic_reference=config["model"]["hydrostatic_reference"],
        )
        parsed.model_id = _digest(model)
        from reference_pr_multifluid_network import hydrostatic_state

        hydrostatic_state(model, parsed.fluid)
    if model != report["model"] or len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Multifluid model/schedule differs")
    nc = len(parsed.fluid.names)
    nph = len(parsed.phase_roots)
    ncell = len(parsed.sources)
    impulse_width = 3 if graph else 1
    supports = set(model.get("supported_nodes", ()))
    directions = []
    if graph:
        directions = [
            np.zeros(3)
            if name in supports
            else np.array(
                next(face["direction"] for face in model["faces"] if face["source"] == name)
            )
            for name in model["cell_names"]
        ]
    elevations = np.array(model.get("elevations_m", [0.0] * ncell))
    elevations -= elevations[0]
    gravity = model.get("gravity_m_s2", 0.0)
    mw = np.array([c.molar_mass_kg_mol for c in parsed.fluid.components])

    def stored_energy(state):
        return math.fsum(
            p["total_energy_j"] + gravity * z * float(np.array(p["component_moles"]) @ mw)
            for c, z in zip(state["cells"], elevations)
            for p in c["phases"]
        )

    def metadata(state):
        if (
            set(state) != {"cells", "time_s", "next_step_s", "model_id"}
            or state["model_id"] != parsed.model_id
            or len(state["cells"]) != ncell
        ):
            raise ValueError("Multifluid state registry/model differs")
        if (
            not math.isfinite(state["time_s"])
            or state["time_s"] < 0
            or not math.isfinite(state["next_step_s"])
            or state["next_step_s"] <= 0
        ):
            raise ValueError("Invalid multifluid clock/step")
        for cell, source in zip(state["cells"], parsed.sources):
            if set(cell) != {"phases", "time_s", "model_id", "next_step_s"}:
                raise ValueError("Unknown/missing source checkpoint fields")
            if (
                cell["model_id"] != source.model_id
                or cell["time_s"] != state["time_s"]
                or not math.isfinite(cell["next_step_s"])
                or cell["next_step_s"] <= 0
            ):
                raise ValueError("Cell/source clock or identity differs")
        if graph:
            for name, cell in zip(model["cell_names"], state["cells"]):
                if name in supports and any(p["momentum_kg_m_s"] != 0.0 for p in cell["phases"]):
                    raise ValueError("Moving supported junction")
        return packets(model, state, parsed.fluid)

    def ledger(a, b, wall, work=0.0):
        aa = [p for cell in a["cells"] for p in cell["phases"]]
        bb = [p for cell in b["cells"] for p in cell["phases"]]
        components = [
            math.fsum(p["component_moles"][k] for p in bb)
            - math.fsum(p["component_moles"][k] for p in aa)
            for k in range(nc)
        ]
        if graph:
            dp = [
                math.fsum(
                    (
                        math.fsum(p["momentum_kg_m_s"] for p in cb["phases"])
                        - math.fsum(p["momentum_kg_m_s"] for p in ca["phases"])
                    )
                    * direction[j]
                    for ca, cb, direction in zip(a["cells"], b["cells"], directions)
                )
                - wall[j]
                for j in range(3)
            ]
        else:
            dp = [
                math.fsum(p["momentum_kg_m_s"] for p in bb)
                - math.fsum(p["momentum_kg_m_s"] for p in aa)
                - wall
            ]
        return components + [stored_energy(b) - stored_energy(a) - work] + dp

    def verify_ledger(a, b, wall, saved, work=0.0):
        for k in range(nph):
            for c in range(nc):
                delta = math.fsum(
                    cell["phases"][k]["component_moles"][c] for cell in b["cells"]
                ) - math.fsum(cell["phases"][k]["component_moles"][c] for cell in a["cells"])
                close(delta, 0.0, atol=1e-8, rtol=0.0)
        residual = ledger(a, b, wall, work)
        close(
            residual,
            np.zeros(nc + 1 + impulse_width),
            atol=np.array([1e-8] * nc + [1e-6] + [1e-8] * impulse_width),
            rtol=0.0,
        )
        close(
            saved,
            residual,
            atol=np.array([1e-12] * nc + [1e-9] + [1e-11] * impulse_width),
            rtol=0.0,
        )

    old = report["initial"]
    metadata(old)
    first = old
    if set(config["initial"]) == {"state"}:
        if old != config["initial"]["state"]:
            raise ValueError("Requested pipe restart differs")
    elif set(config["initial"]) == {"cells"}:
        if old["time_s"] != 0.0 or len(config["initial"]["cells"]) != ncell:
            raise ValueError("Cold pipe clock/grid differs")
        for specs, cell in zip(config["initial"]["cells"], old["cells"]):
            if len(specs) != nph:
                raise ValueError("Cold phase registry differs")
            for spec, p in zip(specs, cell["phases"]):
                if set(spec) != {
                    "pressure_pa",
                    "temperature_k",
                    "volume_m3",
                    "mole_fractions",
                    "velocity_m_s",
                }:
                    raise ValueError("Unknown/missing phase initialization fields")
                n = np.array(p["component_moles"])
                mw = np.array([c.molar_mass_kg_mol for c in parsed.fluid.components])
                for key in ("pressure_pa", "temperature_k", "volume_m3"):
                    close(p[key], spec[key], atol=1e-9, rtol=1e-10)
                close(n / n.sum(), spec["mole_fractions"], atol=1e-12, rtol=1e-12)
                close(p["momentum_kg_m_s"] / (n @ mw), spec["velocity_m_s"], atol=1e-12, rtol=1e-10)
    else:
        raise ValueError("Unknown multifluid initialization")
    checked = 0
    wall_total = np.zeros(3) if graph else 0.0
    maximum_energy = 0.0
    work_total = 0.0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        if set(command) - {
            "duration_s",
            "relative_tolerance",
            "velocity_tolerance_m_s",
            "component_tolerance_mol",
            "minimum_step_s",
            "maximum_step_s",
            "maximum_attempts",
        }:
            raise ValueError("Unknown multifluid schedule field")
        start = old
        wall_interval = np.zeros(3) if graph else 0.0
        work_interval = 0.0
        if not interval["steps"]:
            raise ValueError("Missing accepted multifluid steps")
        for step in interval["steps"]:
            dt = step["dt_s"]
            transport = step["transport"]
            moved = transport["state"]
            new = step["state"]
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Invalid multifluid step")
            close(
                new["time_s"] - old["time_s"],
                dt,
                atol=8 * max(math.ulp(new["time_s"]), math.ulp(old["time_s"])),
                rtol=0.0,
            )
            close(transport["dt_s"], dt, atol=1e-15, rtol=0.0)
            old_packets = metadata(old)
            moving_packets = metadata(moved)
            final_packets = metadata(new)
            if moved["time_s"] != new["time_s"]:
                raise ValueError("Spatial/source subflows double-counted physical time")
            spatial = reconstruct_spatial(model, moved, parsed.fluid)
            for key, sourcekey in (
                ("face_fluxes", "flux"),
                ("phase_path_integrals", "phase_path"),
                ("alpha_path_integrals", "alpha_path"),
            ):
                expected = [face[sourcekey].tolist() for face in spatial["faces"]]
                if expected:
                    close(
                        transport[key],
                        expected,
                        atol=np.array([1e-5] * nc + [1e-5, 1e-3]) if sourcekey == "flux" else 2e-6,
                        rtol=2e-7,
                    )
                elif transport[key]:
                    raise ValueError("Unexpected spatial face diagnostics")
            if spatial["wall_fluxes"]:
                close(transport["wall_fluxes"], spatial["wall_fluxes"], atol=2e-5, rtol=2e-6)
            elif transport["wall_fluxes"]:
                raise ValueError("Periodic pipe has wall flux")
            for i, (a, b) in enumerate(zip(old["cells"], moved["cells"])):
                for k, (pa, pb) in enumerate(zip(a["phases"], b["phases"])):
                    difference = np.r_[
                        np.array(pb["component_moles"]) - pa["component_moles"],
                        pb["momentum_kg_m_s"] - pa["momentum_kg_m_s"],
                        pb["total_energy_j"] - pa["total_energy_j"],
                    ]
                    expected = -dt * spatial["phase_rate"][i, k]
                    if graph and model["cell_names"][i] in supports:
                        if difference[-2] != 0.0:
                            raise ValueError("Supported junction momentum changed")
                        difference = np.r_[difference[:nc], difference[-1]]
                        expected = np.r_[expected[:nc], expected[-1]]
                        tolerances = np.array([2e-10] * nc + [3e-7])
                    else:
                        tolerances = np.array([2e-10] * nc + [2e-10, 3e-7])
                    close(
                        difference,
                        expected,
                        atol=tolerances,
                        rtol=2e-6,
                    )
                a0 = old_packets[i]["alpha"]
                a1 = moving_packets[i]["alpha"]
                close(a1[1:] - a0[1:], -dt * spatial["alpha_rate"][i], atol=1e-10, rtol=2e-6)
            wall = dt * spatial["wall_force_n"]
            close(transport["wall_impulse_kg_m_s"], wall, atol=1e-10, rtol=1e-7)
            entropy_transport = sum(p["entropy"] for p in moving_packets) - sum(
                p["entropy"] for p in old_packets
            )
            if entropy_transport < -1e-7:
                raise ValueError("Independent spatial entropy decrease")
            close(transport["entropy_change_j_k"], entropy_transport, atol=1e-8, rtol=1e-6)
            work = dt * spatial.get("external_power_w", 0.0)
            close(transport.get("external_work_j", 0.0), work, atol=1e-9, rtol=1e-7)
            verify_ledger(old, moved, wall, transport["conservation_residuals"], work)
            if len(step["sources"]) != ncell or len(step["source_inputs"]) != ncell:
                raise ValueError("Source stage registry differs")
            for i, (source_input, source_report) in enumerate(
                zip(step["source_inputs"], step["sources"])
            ):
                expected_input = {**moved["cells"][i], "time_s": old["time_s"]}
                if source_input != expected_input or source_report["state"] != new["cells"][i]:
                    raise ValueError("Source composition changed or bypassed spatial stocks")
                ss = model["sources"][i]
                request = {
                    "model": ss,
                    "initial": {"state": source_input},
                    "schedule": [{"duration_s": dt}],
                }
                source_ledger = [0.0] * nc + [
                    sum(p["volume_m3"] for p in new["cells"][i]["phases"])
                    - sum(p["volume_m3"] for p in source_input["phases"]),
                    sum(p["total_energy_j"] for p in new["cells"][i]["phases"])
                    - sum(p["total_energy_j"] for p in source_input["phases"]),
                    sum(p["momentum_kg_m_s"] for p in new["cells"][i]["phases"])
                    - sum(p["momentum_kg_m_s"] for p in source_input["phases"])
                    - source_report["wall_impulse_kg_m_s"],
                ]
                local = {
                    "config": request,
                    "model": ss,
                    "initial": source_input,
                    "state": new["cells"][i],
                    "intervals": [
                        {
                            "state": new["cells"][i],
                            "steps": [source_report],
                            "conservation_residuals": source_ledger,
                        }
                    ],
                    "conservation_residuals": source_ledger,
                }
                audit_source(request, local)
                wall = wall + source_report["wall_impulse_kg_m_s"] * (
                    directions[i] if graph else 1.0
                )
            close(step["wall_impulse_kg_m_s"], wall, atol=1e-10, rtol=1e-7)
            entropy = sum(p["entropy"] for p in final_packets) - sum(
                p["entropy"] for p in old_packets
            )
            if entropy < -1e-7:
                raise ValueError("Independent composed entropy decrease")
            close(step["entropy_change_j_k"], entropy, atol=1e-8, rtol=1e-6)
            close(step.get("external_work_j", 0.0), work, atol=1e-9, rtol=1e-7)
            verify_ledger(old, new, wall, step["conservation_residuals"], work)
            maximum_energy = max(maximum_energy, abs(ledger(old, new, wall, work)[nc]))
            old = new
            checked += 1
            wall_interval += wall
            work_interval += work
        published = interval["state"]
        metadata(published)
        if {k: v for k, v in old.items() if k != "next_step_s"} != {
            k: v for k, v in published.items() if k != "next_step_s"
        }:
            raise ValueError("Published pipe checkpoint differs from fine steps")
        close(
            published["time_s"],
            start["time_s"] + command["duration_s"],
            atol=8 * math.ulp(published["time_s"]),
            rtol=0.0,
        )
        verify_ledger(
            start, published, wall_interval, interval["conservation_residuals"], work_interval
        )
        wall_total += wall_interval
        work_total += work_interval
        old = published
    if report["state"] != old:
        raise ValueError("Final multifluid state differs")
    verify_ledger(first, old, wall_total, report["conservation_residuals"], work_total)
    return {
        "accepted": True,
        "checked_substeps": checked,
        "cells": ncell,
        "phases": nph,
        "maximum_energy_residual_j": maximum_energy,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    result = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    if not result.get("sources_unchanged_during_run") or not result.get("source_sha256"):
        raise ValueError("Missing multifluid source provenance")
    for path, h in result["source_sha256"].items():
        source = (root / path).resolve()
        if not source.is_relative_to(root) or hashlib.sha256(source.read_bytes()).hexdigest() != h:
            raise ValueError("Changed multifluid sources")
    print(json.dumps(audit(json.loads(args.config.read_text()), result), indent=2))


if __name__ == "__main__":
    main()
