"""Independent multi-branch PR graph flux, stock and entropy audit."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from check_pr_equilibrium_transport import audit as ordinary_audit
from check_pr_equilibrium_transport import decode
from check_pr_layered_plant import audit_initial, audit_profile_step
from check_stratified_pr_flow import close
from reference_pr_layered_plant import exchanges

from surface.pr_profile_graph import PRProfileGraphNetwork


def audit(config, report):
    if config != report["config"]:
        raise ValueError("Profile graph request differs")
    parsed = PRProfileGraphNetwork.from_dict(config["model"])
    model = json.loads(json.dumps(parsed.to_dict()))
    if report["model"] != model or len(config["schedule"]) != len(report["intervals"]):
        raise ValueError("Profile graph model/schedule differs")
    t = model["transport"]
    nodes = {n["name"]: n for n in t["nodes"]}
    names = list(nodes)
    mw = np.array([c["molar_mass_kg_mol"] for c in t["fluid"]["components"]])
    nc = len(mw)
    g, datum = t["gravity_m_s2"], t["reference_elevation_m"]

    persistent = model["model"] == "pr_profile_graph_v2"
    endpoint_switches = {
        f"profile:{n}:{side}": (n, i)
        for n, spec in model["profiles"].items()
        for i, side in enumerate(("left", "right"))
        if persistent and spec["boundary_nodes"][i] is not None
    }

    def metadata(state):
        required = {"transport", "profiles", "model_id"}
        if persistent:
            required.add("profile_openings")
        if set(state) != required:
            raise ValueError("Unknown/missing graph checkpoint fields")
        for s in [state["transport"], *state["profiles"].values()]:
            if not math.isfinite(s["time_s"]) or s["time_s"] < 0:
                raise ValueError("Invalid graph clock")
            if not math.isfinite(s["next_step_s"]) or s["next_step_s"] <= 0:
                raise ValueError("Invalid adaptive step metadata")
        if state["model_id"] != parsed.model_id or set(state["profiles"]) != set(parsed.names):
            raise ValueError("Profile checkpoint registry/model differs")
        if state["transport"]["model_id"] != parsed.transport.model_id:
            raise ValueError("Ordinary checkpoint model differs")
        flags = state.get("profile_openings", {})
        if (persistent and set(flags) != set(parsed.names)) or (not persistent and flags):
            raise ValueError("Persistent endpoint registry differs")
        for n, ff in flags.items():
            if len(ff) != 2 or any(type(v) is not bool for v in ff):
                raise ValueError("Invalid persistent endpoint flags")
            if any(on and e is None for on, e in zip(ff, model["profiles"][n]["boundary_nodes"])):
                raise ValueError("Geometrically closed endpoint was opened")
        for n, s in state["profiles"].items():
            if (
                s["model_id"] != parsed.bridges[n].layered.model_id
                or s["time_s"] != state["transport"]["time_s"]
            ):
                raise ValueError("Profile checkpoint model/clock differs")

    def stocks(state):
        n = np.zeros(nc)
        e = 0.0
        for name, node in nodes.items():
            s = state["transport"]["nodes"][name]
            n += s["component_moles"]
            e += s["total_energy_j"]
            if node["wall"] is not None:
                e += node["wall"]["heat_capacity_j_k"] * s["wall_temperature_k"]
        drives = {d["machine"]: d for d in t.get("drives", [])}
        e += sum(
            0.5 * drives[name]["drive"]["inertia_kg_m2"] * w * w
            for name, w in state["transport"].get("rotors", [])
        )
        for s in state["profiles"].values():
            for cell in s["cells"]:
                nn = np.array(cell["profile"]["component_moles"])
                n += nn
                e += cell["total_energy_j"] - nn @ mw * g * datum
        return n, e

    def balance(a, b, heat, work):
        na, ea = stocks(a)
        nb, eb = stocks(b)
        return np.r_[nb - na, eb - ea - heat - work]

    old = report["initial"]
    metadata(old)
    initial = config["initial"]
    if set(initial) == {"state"}:
        if old != initial["state"]:
            raise ValueError("Requested graph restart differs")
    elif set(initial) == {"transport", "profiles"} and set(initial["profiles"]) == set(
        parsed.names
    ):
        if persistent:
            expected = {
                n: [e is not None for e in spec["boundary_nodes"]]
                for n, spec in model["profiles"].items()
            }
            if old["profile_openings"] != expected:
                raise ValueError("Cold initialization fabricated endpoint commands")
        for n, b in parsed.bridges.items():
            audit_initial(
                b.to_dict(),
                {"transport": old["transport"], "layered": old["profiles"][n]},
                {"transport": initial["transport"], "layered_equilibrium": initial["profiles"][n]},
                parsed.transport.fluid,
            )
    else:
        raise ValueError("Invalid graph initial condition registry")
    first = old
    ordinary_intervals, ordinary_commands, cache = [], [], {}
    total_heat = total_work = 0.0
    checked = 0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        allowed = {
            "duration_s",
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
            "drive_torque_nm",
            "restriction_openings",
            "profile_heat_w",
            "profile_utility_temperature_k",
            "profile_openings",
            "relative_tolerance",
            "interface_tolerance_m",
            "maximum_attempts",
        }
        if set(command) - allowed:
            raise ValueError("Unknown graph schedule command")
        for field in ("profile_heat_w", "profile_utility_temperature_k", "profile_openings"):
            if set(command.get(field, {})) - set(parsed.names):
                raise ValueError("Unknown requested profile branch")
        start = old
        if not interval["steps"]:
            raise ValueError("No accepted profile graph steps")
        ih = iw = oh = ow = os = 0.0
        ordinary_steps = []
        for step in interval["steps"]:
            new, dt = step["state"], step["dt_s"]
            metadata(new)
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Nonpositive/nonfinite graph step")
            close(
                new["transport"]["time_s"] - old["transport"]["time_s"],
                dt,
                atol=8
                * max(math.ulp(new["transport"]["time_s"]), math.ulp(old["transport"]["time_s"])),
                rtol=0,
            )
            if set(step["profiles"]) != set(parsed.names):
                raise ValueError("Accepted branch registry differs")
            ordinary = step["transport"]
            if (
                ordinary["state"] != new["transport"]
                or ordinary["coupling"]["profiles"] != step["profiles"]
            ):
                raise ValueError("Published ordinary/profile iterate differs")
            data = {
                n: decode(new["transport"]["nodes"][n], nodes[n], t, parsed.transport.fluid)
                for n in names
            }
            net = {n: np.zeros(nc + 2) for n in names}
            support = {}
            incoming = production = qheat = 0.0
            expected_ends = {}
            for n, spec in model["profiles"].items():
                flags = command.get("profile_openings", {}).get(
                    n, old.get("profile_openings", {}).get(n, (True, True))
                )
                if len(flags) != 2 or any(type(v) is not bool for v in flags):
                    raise ValueError("Invalid requested endpoint flags")
                flags = list(flags)
                for switch, (branch, i) in endpoint_switches.items():
                    if branch == n and switch in command.get("openings", {}):
                        value = command["openings"][switch]
                        if type(value) is not bool:
                            raise ValueError("Invalid endpoint switch flag")
                        if n in command.get("profile_openings", {}) and value != flags[i]:
                            raise ValueError("Conflicting requested endpoint commands")
                        flags[i] = value
                if persistent:
                    close_flags = [
                        on and e is not None for on, e in zip(flags, spec["boundary_nodes"])
                    ]
                    if new["profile_openings"][n] != close_flags:
                        raise ValueError("Persistent endpoint state differs from accepted command")
                ends = [e if on else None for e, on in zip(spec["boundary_nodes"], flags)]
                expected_ends[n] = ends
                view_model = {"transport": t, "layered": spec["pipe"], "boundary_nodes": ends}
                a, b = {"layered": old["profiles"][n]}, {"layered": new["profiles"][n]}
                if step["profiles"][n]["state"] != b["layered"]:
                    raise ValueError("Accepted native iterate differs")
                d = exchanges(view_model, b, data)
                cmd = {"profile_heat_w": command.get("profile_heat_w", {}).get(n, 0.0)}
                if n in command.get("profile_utility_temperature_k", {}):
                    cmd["profile_utility_temperature_k"] = command["profile_utility_temperature_k"][
                        n
                    ]
                entropy, heat = audit_profile_step(
                    view_model, a, b, step["profiles"][n]["report"], cmd, d
                )
                close(step["profiles"][n]["entropy_production_j_k"], entropy, atol=2e-6, rtol=2e-6)
                branch_support = np.array([d["support"][name] for name in names])
                close(
                    step["profiles"][n]["vessel_support_force_n"],
                    branch_support,
                    atol=2e-6,
                    rtol=2e-6,
                )
                support[n] = branch_support
                incoming += d["incoming_entropy"]
                production += entropy
                qheat += dt * sum(heat)
                for name in names:
                    net[name] += d["ordinary_net"][name]
            if json.loads(json.dumps(step["boundary_nodes"])) != expected_ends:
                raise ValueError("Actual endpoint isolation differs from requested command")
            if set(step["vessel_support_force_n"]) != set(parsed.names):
                raise ValueError("Reaction basis registry differs")
            if set(ordinary["coupling"]["vessel_support_force_n"]) != set(parsed.names):
                raise ValueError("Ordinary reaction basis registry differs")
            for n, force in support.items():
                close(step["vessel_support_force_n"][n], force, atol=2e-6, rtol=2e-6)
                close(
                    ordinary["coupling"]["vessel_support_force_n"][n], force, atol=2e-6, rtol=2e-6
                )
            key = new["transport"]["time_s"]
            if key in cache:
                raise ValueError("Duplicate accepted graph time")
            cache[key] = {"net": net, "incoming_entropy": incoming}
            sh = ordinary["ambient_heat_j"] + ordinary["external_heat_j"] + qheat
            sw = ordinary["external_work_j"]
            close(step["total_heat_j"], sh, atol=1e-8, rtol=2e-7)
            close(step["total_work_j"], sw, atol=1e-10, rtol=2e-7)
            close(
                step["entropy_production_j_k"],
                ordinary["entropy_production_j_k"] + production,
                atol=2e-6,
                rtol=2e-6,
            )
            residual = balance(old, new, sh, sw)
            close(residual, np.zeros(nc + 1), atol=2e-5, rtol=0)
            close(step["conservation_residuals"], residual, atol=1e-6, rtol=2e-7)
            ordinary_steps.append(ordinary)
            ih += sh
            iw += sw
            oh += ordinary["ambient_heat_j"] + ordinary["external_heat_j"]
            ow += sw
            os += ordinary["entropy_production_j_k"]
            old = new
            checked += 1
        close(
            old["transport"]["time_s"],
            start["transport"]["time_s"] + command["duration_s"],
            atol=8 * math.ulp(old["transport"]["time_s"]),
            rtol=0,
        )
        published = interval["state"]
        metadata(published)
        if published.get("profile_openings", {}) != old.get("profile_openings", {}):
            raise ValueError("Adaptive checkpoint changed endpoint flags")
        for a, b in [
            (old["transport"], published["transport"]),
            *((old["profiles"][n], published["profiles"][n]) for n in parsed.names),
        ]:
            if {k: v for k, v in a.items() if k != "next_step_s"} != {
                k: v for k, v in b.items() if k != "next_step_s"
            }:
                raise ValueError("Published adaptive state differs from fine solves")
        old = published
        close(interval["total_heat_j"], ih, atol=1e-7, rtol=2e-7)
        close(interval["total_work_j"], iw, atol=1e-10, rtol=2e-7)
        close(interval["conservation_residuals"], balance(start, old, ih, iw), atol=1e-6, rtol=2e-7)
        total_heat += ih
        total_work += iw
        allowed = {
            "duration_s",
            "openings",
            "heat_w",
            "utility_temperature_k",
            "loss_coefficients",
            "drive_torque_nm",
            "restriction_openings",
        }
        ordinary_command = {k: v for k, v in command.items() if k in allowed}
        if "openings" in ordinary_command:
            ordinary_command["openings"] = {
                k: v for k, v in ordinary_command["openings"].items() if k not in endpoint_switches
            }
        ordinary_commands.append(ordinary_command)
        opened = dict(parsed.transport.default_openings)
        opened.update(
            {k: v for k, v in command.get("openings", {}).items() if k not in endpoint_switches}
        )
        ordinary_intervals.append(
            {
                "state": old["transport"],
                "steps": ordinary_steps,
                "openings": opened,
                "loss_coefficients": {
                    n: command.get("loss_coefficients", {}).get(
                        n, v.get("local_loss_coefficient", 0.0)
                    )
                    for n, v in nodes.items()
                    if v["kind"] == "pipe_cell"
                },
                "total_heat_j": oh,
                "total_work_j": ow,
                "entropy_production_j_k": os,
                "conservation_residuals": [0.0] * (nc + 1),
            }
        )
    ordinary_config = {
        "model": t,
        "initial": {"state": first["transport"]},
        "schedule": ordinary_commands,
    }
    ordinary_report = {
        "config": ordinary_config,
        "model": t,
        "initial": first["transport"],
        "intervals": ordinary_intervals,
        "total_heat_j": sum(i["total_heat_j"] for i in ordinary_intervals),
        "total_work_j": sum(i["total_work_j"] for i in ordinary_intervals),
        "conservation_residuals": [0.0] * (nc + 1),
    }
    result = ordinary_audit(
        ordinary_config, ordinary_report, external_exchange=lambda a, b, d, dt: cache[b["time_s"]]
    )
    if report["state"] != old:
        raise ValueError("Final profile graph checkpoint differs")
    close(report["total_heat_j"], total_heat, atol=1e-7, rtol=2e-7)
    close(report["total_work_j"], total_work, atol=1e-10, rtol=2e-7)
    close(
        report["conservation_residuals"],
        balance(first, old, total_heat, total_work),
        atol=1e-6,
        rtol=2e-7,
    )
    return {
        "accepted": True,
        "checked_substeps": checked,
        "profile_branches": len(parsed.names),
        "ordinary": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[1]
    if not report.get("sources_unchanged_during_run") or not report.get("source_sha256"):
        raise ValueError("Missing source provenance")
    for path, digest in report["source_sha256"].items():
        source = (root / path).resolve()
        if (
            not source.is_relative_to(root)
            or hashlib.sha256(source.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Changed profile graph source")
    print(json.dumps(audit(json.loads(args.config.read_text()), report), indent=2))


if __name__ == "__main__":
    main()
