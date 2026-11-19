"""Audit split stages with existing independent transport/controller checkers.

No runtime integration or projection/merge helper is invoked. Stock transfer
and the complete finite network are reconstructed from serialized stages.
"""

import copy
import json
import math

from check_pr_equilibrium_transport import audit as audit_transport
from check_stratified_pr_flow import close

from surface.pr_surface_control import PRControlledSurface


def _transport(state):
    return state["plant"]["transport"]


def _clock(state):
    return _transport(state)["time_s"]


def _same_state(actual, expected, *, schedule_hint=False):
    actual, expected = copy.deepcopy(actual), copy.deepcopy(expected)
    close(_clock(actual), _clock(expected), atol=8 * max(math.ulp(_clock(expected)), 1e-30), rtol=0)
    _transport(actual)["time_s"] = _clock(expected)
    if schedule_hint:
        actual["next_step_s"] = expected["next_step_s"]
    if actual != expected:
        raise ValueError("Split projection/merge changed or lost a physical/controller state")


def _ledger(model, before, after, heat_work):
    t = model.plant.transport
    old, new = _transport(before), _transport(after)
    first = [*old["nodes"].values(), *before["plant"]["columns"].values()]
    last = [*new["nodes"].values(), *after["plant"]["columns"].values()]
    residual = [
        math.fsum(v["component_moles"][i] for v in last)
        - math.fsum(v["component_moles"][i] for v in first)
        for i in range(len(t.fluid.names))
    ]
    energy = [b["total_energy_j"] - a["total_energy_j"] for a, b in zip(first, last, strict=True)]
    for name, old_column in before["plant"]["columns"].items():
        new_column = after["plant"]["columns"][name]
        energy.append(
            -t.gravity_m_s2
            * t.reference_elevation_m
            * math.fsum(
                (b - a) * c.molar_mass_kg_mol
                for a, b, c in zip(
                    old_column["component_moles"], new_column["component_moles"], t.fluid.components
                )
            )
        )
    for node in t.nodes:
        if node.wall:
            energy.append(
                node.wall.heat_capacity_j_k
                * (
                    new["nodes"][node.name]["wall_temperature_k"]
                    - old["nodes"][node.name]["wall_temperature_k"]
                )
            )
    for name, wall in model.plant.walls.items():
        energy.append(
            wall.heat_capacity_j_k
            * (after["plant"]["walls"][name] - before["plant"]["walls"][name])
        )
    a, b = dict(old.get("rotors", [])), dict(new.get("rotors", []))
    for name, drive in t._drives.items():
        energy.append(0.5 * drive.drive.inertia_kg_m2 * (b[name] ** 2 - a[name] ** 2))
    residual.append(math.fsum(energy) - heat_work)
    for i, value in enumerate(residual[:-1]):
        if abs(value) > 1e-8 + 1e-9 * math.fsum(v["component_moles"][i] for v in first):
            raise ValueError("Split component conservation failed")
    if abs(residual[-1]) > 1e-5 + 1e-9 * max(1.0, abs(heat_work)):
        raise ValueError("Split energy conservation failed")
    return residual


def _reported_ledger(saved, actual, before, after):
    close(saved[:-1], actual[:-1], atol=1e-8, rtol=2e-7)
    # Runtime reports subtract absolute total energies; this checker sums
    # local changes. Account for their floating summation order, not for a
    # larger physical error: _ledger above still enforces the original bound.
    rounding = 0.0
    for state in (before, after):
        stocks = [*_transport(state)["nodes"].values(), *state["plant"]["columns"].values()]
        rounding += math.fsum(math.ulp(v["total_energy_j"]) for v in stocks)
    close(saved[-1], actual[-1], atol=1e-10 + 16 * rounding, rtol=2e-7)


def audit(config, report):
    from check_pr_controlled_surface import audit as audit_control

    if config != report["config"] or set(config["initial"]) != {"state"}:
        raise ValueError("Subcycle audit requires the exact explicit initial checkpoint")
    model = PRControlledSurface.from_dict(config["model"])
    if json.loads(json.dumps(model.to_dict())) != report["model"]:
        raise ValueError("Subcycled model differs")
    if report["initial"] != config["initial"]["state"]:
        raise ValueError("Subcycle initial stocks differ")
    current = copy.deepcopy(report["initial"])
    if current["model_id"] != model.model_id:
        raise ValueError("Subcycle checkpoint identity differs")
    initial = copy.deepcopy(current)
    heat_total = work_total = 0.0
    fast_steps = slow_steps = macro_steps = 0
    for command, interval in zip(config["schedule"], report["intervals"], strict=True):
        interval_initial = copy.deepcopy(current)
        end = _clock(current) + command["duration_s"]
        interval_heat = interval_work = 0.0
        commands = {
            k: v
            for k, v in command.items()
            if k not in ("duration_s", "targets", "actuator_modes", "reset_trips")
        }
        # Independent partition of the user commands, not the saved claims.
        fast_commands, slow_commands = {}, copy.deepcopy(commands)
        for key, selected in (
            ("openings", model.fast_links),
            ("heat_w", model.pipe_nodes),
            ("utility_temperature_k", model.pipe_nodes),
            ("loss_coefficients", model.pipe_nodes),
        ):
            values = commands.get(key) or {}
            fast_commands[key] = {k: v for k, v in values.items() if k in selected}
            slow_commands[key] = {k: v for k, v in values.items() if k not in selected}
        for step in interval["steps"]:
            before = copy.deepcopy(current)
            start, duration = _clock(current), step["dt_s"]
            details = step["local_subcycling"]
            if (
                details["fast_commands"] != fast_commands
                or details["slow_commands"] != slow_commands
            ):
                raise ValueError("Split commands differ from the requested schedule")
            stages = details["stages"]
            if [s["operator"] for s in stages] != ["fast", "slow", "fast"]:
                raise ValueError("Missing or reordered split operator stage")
            # Floating-point half-interval convention, independently derived.
            middle = start + duration / 2
            clocks, durations = (
                (start, start, middle),
                (middle - start, duration, start + duration - middle),
            )
            step_heat = step_work = 0.0
            for stage, clock, dt in zip(stages, clocks, durations, strict=True):
                saved = stage["report"]
                if stage["operator"] == "fast":
                    expected = copy.deepcopy(_transport(current))
                    expected["nodes"] = {
                        n: s for n, s in expected["nodes"].items() if n in model.fast.names
                    }
                    expected.update(time_s=clock, model_id=model.fast.model_id)
                    expected.pop("rotors", None)
                    if stage["initial"] != expected:
                        raise ValueError("Fast stage did not receive the preceding finite stocks")
                    request = {
                        "model": json.loads(json.dumps(model.fast.to_dict())),
                        "initial": {"state": expected},
                        "schedule": [{"duration_s": dt, **fast_commands}],
                    }
                    audit_transport(
                        request,
                        {
                            "config": request,
                            "model": request["model"],
                            "initial": expected,
                            "intervals": [saved],
                            "total_heat_j": saved["total_heat_j"],
                            "total_work_j": saved["total_work_j"],
                            "conservation_residuals": saved["conservation_residuals"],
                        },
                    )
                    result = saved["state"]
                    if set(stage["boundary_transfers"]) != set(model.boundaries):
                        raise ValueError("Boundary exchange registry differs")
                    for name in model.boundaries:
                        a, b = expected["nodes"][name], result["nodes"][name]
                        close(
                            stage["boundary_transfers"][name]["component_moles"],
                            [y - x for x, y in zip(a["component_moles"], b["component_moles"])],
                            atol=1e-12,
                            rtol=1e-10,
                        )
                        close(
                            stage["boundary_transfers"][name]["total_energy_j"],
                            b["total_energy_j"] - a["total_energy_j"],
                            atol=1e-10,
                            rtol=1e-10,
                        )
                    _transport(current)["nodes"].update(result["nodes"])
                    _transport(current).update(
                        time_s=result["time_s"], next_step_s=result["next_step_s"]
                    )
                    fast_steps += len(saved["steps"])
                else:
                    expected = copy.deepcopy(current)
                    _transport(expected)["nodes"] = {
                        n: s
                        for n, s in _transport(expected)["nodes"].items()
                        if n in model.slow.plant.transport.names
                    }
                    _transport(expected).update(
                        time_s=clock, model_id=model.slow.plant.transport.model_id
                    )
                    expected["plant"]["model_id"], expected["model_id"] = (
                        model.slow.plant.model_id,
                        model.slow.model_id,
                    )
                    if stage["initial"] != expected:
                        raise ValueError("Slow stage did not receive the updated boundary stocks")
                    request = {
                        "model": json.loads(json.dumps(model.slow.to_dict())),
                        "initial": {"state": expected},
                        "schedule": [
                            {
                                "duration_s": dt,
                                **slow_commands,
                                "targets": command.get("targets", {}),
                                "actuator_modes": command.get("actuator_modes", {}),
                            }
                        ],
                    }
                    residual = saved["plant"]["conservation_residuals"]
                    audit_control(
                        request,
                        {
                            "config": request,
                            "model": request["model"],
                            "initial": expected,
                            "intervals": [
                                {
                                    "steps": [saved],
                                    "state": saved["state"],
                                    "total_heat_j": saved["total_heat_j"],
                                    "total_work_j": saved["total_work_j"],
                                    "conservation_residuals": residual,
                                }
                            ],
                            "total_heat_j": saved["total_heat_j"],
                            "total_work_j": saved["total_work_j"],
                            "conservation_residuals": residual,
                        },
                    )
                    result = copy.deepcopy(saved["state"])
                    full_nodes = copy.deepcopy(_transport(current)["nodes"])
                    full_nodes.update(_transport(result)["nodes"])
                    _transport(result).update(
                        nodes=full_nodes,
                        model_id=model.plant.transport.model_id,
                        next_step_s=_transport(current)["next_step_s"],
                    )
                    result["plant"]["model_id"], result["model_id"] = (
                        model.plant.model_id,
                        model.model_id,
                    )
                    current = result
                    slow_steps += len(saved["plant"]["steps"])
                step_heat += saved["total_heat_j"]
                step_work += saved["total_work_j"]
            _same_state(step["state"], current)
            residual = _ledger(model, before, current, step_heat + step_work)
            _reported_ledger(details["conservation_residuals"], residual, before, current)
            close(step["total_heat_j"], step_heat, atol=1e-8, rtol=2e-7)
            close(step["total_work_j"], step_work, atol=1e-8, rtol=2e-7)
            interval_heat += step_heat
            interval_work += step_work
            macro_steps += 1
        close(_clock(current), end, atol=8 * math.ulp(end), rtol=0)
        _same_state(interval["state"], current, schedule_hint=True)
        current = copy.deepcopy(interval["state"])
        residual = _ledger(model, interval_initial, current, interval_heat + interval_work)
        _reported_ledger(interval["conservation_residuals"], residual, interval_initial, current)
        close(interval["total_heat_j"], interval_heat, atol=1e-8, rtol=2e-7)
        close(interval["total_work_j"], interval_work, atol=1e-8, rtol=2e-7)
        heat_total += interval_heat
        work_total += interval_work
    residual = _ledger(model, initial, current, heat_total + work_total)
    _reported_ledger(report["conservation_residuals"], residual, initial, current)
    close(report["total_heat_j"], heat_total, atol=1e-8, rtol=2e-7)
    close(report["total_work_j"], work_total, atol=1e-8, rtol=2e-7)
    return {
        "accepted": True,
        "checked_split_steps": macro_steps,
        "checked_local_steps": fast_steps,
        "checked_slow_steps": slow_steps,
        "conservation_residuals": residual,
    }
