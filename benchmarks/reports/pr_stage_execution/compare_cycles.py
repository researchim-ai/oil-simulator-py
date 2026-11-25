"""Compare identical full-cycle requests; interpolated observations are diagnostic only."""

import bisect
import copy
import gzip
import json
import math
from pathlib import Path

from check_pr_controlled_surface import actuator

from surface.pr_surface_control import PRControlledSurface, PRControlledSurfaceState

ROOT = Path(__file__).resolve().parent


def load(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def linear(a, b, w):
    if a is None and b is None:
        return None
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return a + w * (b - a)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [linear(x, y, w) for x, y in zip(a, b, strict=True)]
    if isinstance(a, dict) and a.keys() == b.keys():
        return {k: linear(v, b[k], w) for k, v in a.items()}
    if a == b:
        return copy.deepcopy(a)
    raise ValueError("Cannot interpolate changing topology/registry")


def interpolate(a, b, w):
    # Interpolate only observed fields in the original temporal norm plus
    # phase inventories for dataclass consistency. Keep collocation guesses,
    # quadrature hints and adapter IDs out of this observation comparison.
    state = copy.deepcopy(a)
    ta, tb = a["plant"]["transport"], b["plant"]["transport"]
    state["plant"]["transport"]["time_s"] = linear(ta["time_s"], tb["time_s"], w)
    for name, node in ta["nodes"].items():
        other = tb["nodes"][name]
        if node["phase_roots"] != other["phase_roots"]:
            raise ValueError("Node phase transition")
        for key in (
            "component_moles",
            "total_energy_j",
            "momentum_kg_m_s",
            "pressure_pa",
            "temperature_k",
            "wall_temperature_k",
            "phase_component_moles",
        ):
            state["plant"]["transport"]["nodes"][name][key] = linear(node[key], other[key], w)
    state["plant"]["transport"]["rotors"] = [
        [n, linear(x, dict(tb["rotors"])[n], w)] for n, x in ta.get("rotors", [])
    ]
    for name, col in a["plant"]["columns"].items():
        other = b["plant"]["columns"][name]
        if col["phase_roots"] != other["phase_roots"]:
            raise ValueError("Column phase transition")
        for key in ("component_moles", "total_energy_j", "temperature_k", "interface_heights_m"):
            state["plant"]["columns"][name][key] = linear(col[key], other[key], w)
    state["plant"]["walls"] = linear(a["plant"]["walls"], b["plant"]["walls"], w)
    for key in ("positions", "integrals"):
        state[key] = linear(a[key], b[key], w)
    return state


def trace(interval, initial):
    return [initial] + [s["state"] for s in interval["steps"]]


def clock(state):
    return state["plant"]["transport"]["time_s"]


def observations(model, target, source, command, *, event=False):
    times = [clock(s) for s in source]
    errors = []
    skipped = []
    for state in target:
        t = clock(state)
        j = bisect.bisect_left(times, t)
        if j < len(times) and times[j] == t:
            other = source[j]
        elif j == 0 or j == len(times):
            skipped.append({"time_s": t, "reason": "outside bracket"})
            continue
        elif event and j == 1:
            skipped.append(
                {
                    "time_s": t,
                    "reason": "initial layer after command change; exact-time replay required",
                }
            )
            continue
        else:
            w = (t - times[j - 1]) / (times[j] - times[j - 1])
            try:
                other = interpolate(source[j - 1], source[j], w)
                specs = model.to_dict()
                for channel in (*specs.get("valves", []), *specs.get("motors", [])):
                    name = channel["name"]
                    if name in command.get("targets", {}):
                        other["positions"][name] = actuator(
                            source[0]["positions"][name],
                            command["targets"][name],
                            t - clock(source[0]),
                            channel["actuator"],
                            command.get("actuator_modes", {}).get(name, "normal"),
                        )
            except ValueError as exc:
                skipped.append({"time_s": t, "reason": str(exc)})
                continue
        error = model._error(
            PRControlledSurfaceState.from_dict(state), PRControlledSurfaceState.from_dict(other)
        )
        errors.append({"time_s": t, "error_in_original_norm": error})
    return {
        "checked": len(errors),
        "maximum_error_in_original_norm": max(
            (x["error_in_original_norm"] for x in errors), default=None
        ),
        "samples": errors,
        "skipped": skipped,
    }


def operating(report):
    intervals = []
    for interval in report["intervals"]:
        ports = {}
        for record in interval["steps"]:
            for step in record["plant"].get("steps", [record["plant"]]):
                for name, port in step["columns"]["ports"].items():
                    values = port.get("integrated_flux", [step["dt_s"] * v for v in port["flux"]])
                    ports[name] = [
                        a + b for a, b in zip(ports.get(name, [0.0] * len(values)), values)
                    ]
        intervals.append(
            {name: {"component_moles": v[:-2], "energy_j": v[-1]} for name, v in ports.items()}
        )
    assert all(
        all(x == 0 for x in p["component_moles"]) and p["energy_j"] == 0
        for p in intervals[1].values()
    )
    assert all(
        any(x != 0 for x in p["component_moles"])
        for i in (intervals[0], intervals[2])
        for p in i.values()
    )
    return {
        "accepted": True,
        "interval_port_transfers": intervals,
        "scope": "all five separator ports active in startup/restart, exactly zero N/E transfers during isolation; reflecting momentum retained",
    }


def main():
    a, b = [
        load(ROOT / f"release-cycle-{s}/trajectory.json.gz") for s in ("reference", "optimized")
    ]
    ma, mb = [
        json.loads((ROOT / f"release-cycle-{s}/summary.json").read_text())
        for s in ("reference", "optimized")
    ]
    assert (
        ma["accepted"]
        and mb["accepted"]
        and a["config"] == b["config"]
        and ma["source_sha256"] == mb["source_sha256"]
    )
    assert a["initial"] == b["initial"]
    for r in (a, b):
        assert not r["numerical_policy"]["collocation_stage_predictor"]
        assert r["numerical_policy"]["collocation_dense_cpu_jacobian_mode"] == "reverse"
    assert not a["numerical_policy"]["collocation_stage_tangent_reuse"]
    assert b["numerical_policy"]["collocation_stage_tangent_reuse"]
    assert not a["numerical_policy"]["surface_command_step_restart"]
    assert b["numerical_policy"]["surface_command_step_restart"]
    model = PRControlledSurface.from_dict(a["config"]["model"])
    endpoints = []
    waves = []
    old_a = a["initial"]
    old_b = b["initial"]
    previous_command = None
    for command, ia, ib in zip(
        a["config"]["schedule"], a["intervals"], b["intervals"], strict=True
    ):
        content = {k: v for k, v in command.items() if k != "duration_s"}
        event = previous_command is not None and content != previous_command
        previous_command = content
        error = model._error(
            PRControlledSurfaceState.from_dict(ia["state"]),
            PRControlledSurfaceState.from_dict(ib["state"]),
        )
        assert math.isfinite(error) and error <= 1
        endpoints.append({"time_s": clock(ia["state"]), "error_in_original_norm": error})
        waves.append(
            {
                "reference_times": observations(
                    model, trace(ia, old_a), trace(ib, old_b), command, event=event
                ),
                "optimized_times": observations(
                    model, trace(ib, old_b), trace(ia, old_a), command, event=event
                ),
            }
        )
        old_a, old_b = ia["state"], ib["state"]
    assert clock(old_a) == clock(old_b) == 1.0
    checks = {"reference": operating(a), "optimized": operating(b)}
    replay = json.loads((ROOT / "event-replay-validation.json").read_text())
    assert replay["accepted"]
    covered = {(x["interval"], x["time_s"]) for x in replay["exact_time_replays"]}
    for i, wave in enumerate(waves):
        for direction in wave.values():
            assert direction["maximum_error_in_original_norm"] <= 1
            assert all((i, item["time_s"]) in covered for item in direction["skipped"])
    result = {
        "accepted": True,
        "event_replay_validation": replay,
        "paired_full_cycle_wall_s": {
            "reference": ma["integration_wall_s"],
            "optimized": mb["integration_wall_s"],
        },
        "speedup_single_pair": ma["integration_wall_s"] / mb["integration_wall_s"],
        "interval_stats": {
            label: [
                {
                    "end_time_s": clock(i["state"]),
                    "records": len(i["steps"]),
                    "rejected_attempts": i["rejected_attempts"],
                    "fallback_intervals": i["fallback_intervals"],
                    "command_step_restart": i["command_step_restart"],
                    "initial_step_hint_s": i["initial_step_hint_s"],
                    "linear_work": i["collocation_linear_work"],
                }
                for i in report["intervals"]
            ]
            for label, report in (("reference", a), ("optimized", b))
        },
        "endpoint_errors": endpoints,
        "observation_diagnostics": waves,
        "operating_checks": checks,
        "scope": "one sequential identical 1 s full-cycle pair including initialization/journal, audits/serialization excluded; endpoint error uses unchanged original norm; intermediate slow fields interpolated only for diagnosis; manual actuator positions use the independent exact law; initial command layers checked by exact-time replay; reconstructed observations are not accepted thermodynamic states or external truth",
    }
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "accepted",
                    "paired_full_cycle_wall_s",
                    "speedup_single_pair",
                    "endpoint_errors",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
