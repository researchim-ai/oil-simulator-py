"""Transactional dynamic-vessel network schedule with boundary-only accounting."""

import copy
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from .dynamic_vessel import VesselState
from .liquid import ConservationError, _number
from .operation import _digest
from .separator_geometry import LevelWithdrawal
from .thermal_process import ThermalStream
from .vessel_schedule import _totals


def _commands(period, fluid, names, connection_names, construct=False):
    if "duration_s" not in period or set(period) - {
        "duration_s",
        "feeds",
        "withdrawals",
        "heat",
        "connection_factors",
        "reset_trips",
        "ambient_temperatures",
        "actuator_modes",
    }:
        raise ValueError("Invalid dynamic network period fields")
    _number(period["duration_s"], "duration_s", positive=True)
    feeds = period.get("feeds", {})
    withdrawals = period.get("withdrawals", {})
    heat = period.get("heat", {})
    if (
        not all(isinstance(x, dict) for x in (feeds, withdrawals, heat))
        or (set(feeds) | set(withdrawals) | set(heat)) - names
    ):
        raise ValueError("Unknown or invalid network vessel command")
    for incoming in feeds.values():
        if not isinstance(incoming, dict):
            raise TypeError("Network feeds must map names to stream specifications")
        for feed in incoming.values():
            if set(feed) != {"pressure_pa", "temperature_k", "molar_rate_mol_s", "mole_fractions"}:
                raise ValueError("Invalid network feed fields")
            _number(feed["molar_rate_mol_s"], "molar_rate_mol_s", nonnegative=True)
            fluid._composition(feed["mole_fractions"])
            for c in fluid.components:
                c._validate(feed["pressure_pa"], feed["temperature_k"])
    for ports in withdrawals.values():
        if not isinstance(ports, dict) or set(ports) - {"bulk", "phase_0", "phase_1"}:
            raise ValueError("Unknown network phase port")
        for port, rate in ports.items():
            if isinstance(rate, dict):
                if port != "phase_0":
                    raise ValueError("Level withdrawal only supports liquid phase_0")
                LevelWithdrawal(**rate)
            else:
                _number(rate, "outlet_molar_rate_mol_s", nonnegative=True)
    for command in heat.values():
        if not isinstance(command, dict) or set(command) - {"heat_w", "utility_temperature_k"}:
            raise ValueError("Unknown network heat command")
        if _number(command.get("heat_w", 0.0), "heat_w"):
            _number(command.get("utility_temperature_k"), "utility_temperature_k", positive=True)
    factors = period.get("connection_factors", {})
    if not isinstance(factors, dict) or set(factors) - connection_names:
        raise ValueError("Unknown connection factor command")
    for value in factors.values():
        if not 0 <= _number(value, "conductance_factor") <= 1:
            raise ValueError("Conductance factor must be in [0,1]")
    if construct:
        withdrawals = {
            name: {
                port: LevelWithdrawal(**rate) if isinstance(rate, dict) else rate
                for port, rate in ports.items()
            }
            for name, ports in withdrawals.items()
        }
        feeds = {
            name: tuple(ThermalStream.from_pt(fluid, **feed) for feed in incoming.values())
            for name, incoming in feeds.items()
        }
    ambient = period.get("ambient_temperatures", {})
    if not isinstance(ambient, dict) or set(ambient) - names:
        raise ValueError("Unknown ambient temperature command")
    for t in ambient.values():
        _number(t, "ambient_temperature_k", positive=True)
    return feeds, withdrawals, heat, factors


def run_network_schedule(
    fluid,
    network,
    initial_states,
    periods,
    stepper=None,
    checkpoint=None,
    stop_after=None,
    protections=(),
    event_time_tolerance_s=1e-7,
    initial_actuator_positions=None,
):
    vessels = dict(network.vessels)
    names = set(vessels)
    if set(initial_states) != names or len({s.time_s for s in initial_states.values()}) != 1:
        raise ValueError("Initial network states/clocks must match vessels")
    for name, v in network.vessels:
        v.inspect(initial_states[name], fluid)
    network.inspect_walls(initial_states)
    initial_positions, _ = network.inspect_actuators(initial_actuator_positions)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Network periods must be a nonempty sequence")
    from .network_events import advance_protected_network

    protections = tuple(protections)
    rule_map = {r.name: r for r in protections}
    if len(rule_map) != len(protections):
        raise ValueError("Duplicate protection name")
    _number(event_time_tolerance_s, "event_time_tolerance_s", positive=True)
    for rule in protections:
        rule.validate(network, fluid, require_action=True)
    for period in periods:
        _commands(period, fluid, names, {c.name for c in network.connections})
        network.inspect_actuators(initial_positions, period.get("actuator_modes"))
        for name, ports in period.get("withdrawals", {}).items():
            for rate in ports.values():
                if isinstance(rate, dict):
                    geometry = dict(network.geometries).get(name)
                    if geometry is None or rate["setpoint_m"] > geometry.height_m:
                        raise ValueError("Level control needs matching separator geometry")
        if set(period.get("ambient_temperatures", {})) - set(dict(network.walls)):
            raise ValueError("Ambient command requires a configured wall")
        reset = period.get("reset_trips", [])
        if (
            not isinstance(reset, (tuple, list))
            or len(set(reset)) != len(reset)
            or set(reset) - set(rule_map)
        ):
            raise ValueError("Unknown or duplicate protection reset")
    stepper = copy.deepcopy(stepper or {})
    if set(stepper) - {"initial_dt_s", "maximum_dt_s", "minimum_dt_s", "max_substeps"}:
        raise ValueError("Unknown network stepper option")
    for key, value in stepper.items():
        if key == "max_substeps":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("max_substeps must be positive integer")
        elif value is not None or key != "maximum_dt_s":
            _number(value, key, positive=True)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "revision": "monolithic_vessel_network_v5",
            "initial_actuator_positions": initial_positions,
            "protections": [asdict(r) for r in protections],
            "event_time_tolerance_s": event_time_tolerance_s,
            "fluid": asdict(fluid),
            "network": asdict(network),
            "initial": {name: initial_states[name].to_dict() for name, _ in network.vessels},
            "periods": periods,
            "stepper": stepper,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "implementation": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "vessel_network.py",
                    "vessel_wall.py",
                    "actuator.py",
                    "separator_geometry.py",
                    "nozzle_connection.py",
                    "single_phase_nozzle.py",
                    "mixture_nozzle.py",
                    "thermal_equipment.py",
                    "network_schedule.py",
                    "network_events.py",
                    "dynamic_vessel.py",
                    "vessel_schedule.py",
                    "peng_robinson.py",
                    "peng_robinson_mixture.py",
                    "thermal_process.py",
                    "operation.py",
                    "liquid.py",
                )
            },
        }
    )
    n = len(fluid.components)
    start = next(iter(initial_states.values())).time_s
    state = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "states": {name: initial_states[name].to_dict() for name, _ in network.vessels},
        "actuator_positions": initial_positions,
        "records": [],
        "latched_trips": [],
        "blocked_feed_component_moles": [0.0] * n,
        "totals": _totals([], n),
    }
    if checkpoint is not None:
        candidate = copy.deepcopy(checkpoint)
        if set(candidate) != set(state) | {"checksum"}:
            raise ValueError("Invalid network checkpoint fields")
        checksum = candidate.pop("checksum")
        if checksum != _digest(candidate) or candidate["fingerprint"] != fingerprint:
            raise ValueError("Network checkpoint checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Network checkpoint index mismatch")
        for i, record in enumerate(candidate["records"]):
            if record["period"] != i or not math.isclose(
                math.fsum(s["dt_s"] for s in record["steps"]),
                periods[i]["duration_s"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Network checkpoint timeline mismatch")
        if candidate["totals"] != _totals(candidate["records"], n):
            raise ValueError("Network checkpoint accumulation mismatch")
        state = candidate
    if set(state["states"]) != names:
        raise ValueError("Checkpoint vessel set mismatch")
    current = {name: VesselState(**state["states"][name]) for name, _ in network.vessels}
    initial_moles = np.sum(
        [initial_states[name].component_moles for name, _ in network.vessels], axis=0
    )
    initial_energy = math.fsum(
        initial_states[name].internal_energy_j + (initial_states[name].wall_energy_j or 0.0)
        for name, _ in network.vessels
    )

    def audit():
        positions = dict(initial_positions)
        actuator_models = dict(network.actuators)
        latches = set()
        for i, record in enumerate(state["records"]):
            period = periods[i]
            latches.difference_update(period.get("reset_trips", []))
            time = start + math.fsum(p["duration_s"] for p in periods[:i])
            events = list(record.get("events", []))
            for step in record["steps"]:
                # A localized event acts only AFTER the step ending at it.
                # Include initial events before the first step of the period.
                while events and events[0]["time_s"] <= time + 1e-12:
                    latches.add(events.pop(0)["name"])
                commands = dict(period.get("connection_factors", {}))
                for name in latches:
                    if name not in rule_map:
                        raise ValueError("Unknown actuator protection event")
                    for connection in rule_map[name].close_connections:
                        commands[connection] = 0.0
                if set(step.get("actuators", {})) != set(actuator_models):
                    raise ValueError("Checkpoint actuator history mismatch")
                for name, model in actuator_models.items():
                    expected = model.step(
                        positions[name],
                        commands.get(name, 1.0),
                        step["dt_s"],
                        period.get("actuator_modes", {}).get(name, "normal"),
                    )
                    if step["actuators"][name] != expected:
                        raise ValueError("Checkpoint actuator trajectory/command mismatch")
                    positions[name] = expected["final_position"]
                time = math.fsum([time, step["dt_s"]])
            if record.get("actuator_positions", {}) != positions:
                raise ValueError("Checkpoint period actuator positions mismatch")
            latches.update(event["name"] for event in events)
        network.inspect_actuators(state["actuator_positions"])
        if positions != state["actuator_positions"]:
            raise ValueError("Checkpoint final actuator positions mismatch")
        blocked = [
            math.fsum(r.get("blocked_feed_component_moles", [0.0] * n)[i] for r in state["records"])
            for i in range(n)
        ]
        if state["blocked_feed_component_moles"] != blocked:
            raise ValueError("Checkpoint blocked feed accumulation mismatch")
        for i, record in enumerate(state["records"]):
            requested = np.array(
                [
                    periods[i]["duration_s"]
                    * math.fsum(
                        feed["molar_rate_mol_s"] * feed["mole_fractions"][j]
                        for node in periods[i].get("feeds", {}).values()
                        for feed in node.values()
                    )
                    for j in range(n)
                ]
            )
            actual = (
                np.sum([s["inlet_component_moles"] for s in record["steps"]], axis=0)
                if record["steps"]
                else np.zeros(n)
            )
            if not np.allclose(
                actual + record.get("blocked_feed_component_moles", [0.0] * n),
                requested,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ConservationError("Protection requested/admitted/blocked feed balance failed")
        network.inspect_walls(current)
        expected_latches = set()
        for i, record in enumerate(state["records"]):
            expected_latches.difference_update(periods[i].get("reset_trips", []))
            previous_time = start + math.fsum(p["duration_s"] for p in periods[:i])
            end_time = previous_time + periods[i]["duration_s"]
            for event in record.get("events", []):
                if event["name"] not in rule_map or event["name"] in expected_latches:
                    raise ValueError("Checkpoint event/latch mismatch")
                rule = rule_map[event["name"]]
                if (
                    not previous_time - event_time_tolerance_s
                    <= event["time_s"]
                    <= end_time + event_time_tolerance_s
                    or event["metric"] != rule.metric
                    or event["vessel"] != rule.vessel
                    or event["threshold"] != rule.threshold
                    or event["direction"] != rule.direction
                ):
                    raise ValueError("Checkpoint protection event mismatch")
                gap = (
                    rule.threshold - event["value"]
                    if rule.direction == "above"
                    else event["value"] - rule.threshold
                )
                if (
                    event["initially_triggered"]
                    and gap > 0
                    or not event["initially_triggered"]
                    and abs(gap) > rule.value_tolerance
                ):
                    raise ValueError("Checkpoint event value mismatch")
                previous_time = event["time_s"]
                expected_latches.add(event["name"])
        if state["latched_trips"] != sorted(expected_latches):
            raise ValueError("Checkpoint protection latch differs from event history")
        time = start + math.fsum(p["duration_s"] for p in periods[: state["next_period"]])
        for name, v in network.vessels:
            v.inspect(current[name], fluid)
            if not math.isclose(current[name].time_s, time, rel_tol=0, abs_tol=1e-9):
                raise ValueError("Checkpoint network clocks mismatch")
        mass = (
            np.sum([current[name].component_moles for name, _ in network.vessels], axis=0)
            - initial_moles
            - np.array(state["totals"]["inlet_component_moles"])
            + state["totals"]["outlet_component_moles"]
        )
        energy = (
            math.fsum(
                current[name].internal_energy_j + (current[name].wall_energy_j or 0.0)
                for name, _ in network.vessels
            )
            - initial_energy
            - state["totals"]["inlet_enthalpy_j"]
            + state["totals"]["outlet_enthalpy_j"]
            - state["totals"]["heat_j"]
        )
        if max(abs(mass)) > 1e-8 * max(1.0, sum(initial_moles)) or abs(energy) > 1e-5 + 1e-8 * max(
            1.0, abs(initial_energy), abs(state["totals"]["inlet_enthalpy_j"])
        ):
            raise ConservationError("Network checkpoint boundary balance failed")

    audit()
    end = (
        len(periods) if stop_after is None else min(len(periods), state["next_period"] + stop_after)
    )
    for index in range(state["next_period"], end):
        feeds, withdrawals, heat, factors = _commands(
            periods[index], fluid, names, {c.name for c in network.connections}, construct=True
        )
        active_latches = set(state["latched_trips"]) - set(periods[index].get("reset_trips", []))
        if protections:
            result = advance_protected_network(
                network,
                current,
                fluid,
                periods[index]["duration_s"],
                protections,
                latched=active_latches,
                feeds=feeds,
                withdrawals=withdrawals,
                heat=heat,
                connection_factors=factors,
                time_tolerance_s=event_time_tolerance_s,
                ambient_temperatures=periods[index].get("ambient_temperatures"),
                actuator_positions=state["actuator_positions"],
                actuator_modes=periods[index].get("actuator_modes"),
                **stepper,
            )
        else:
            result = network.advance(
                current,
                fluid,
                periods[index]["duration_s"],
                feeds=feeds,
                withdrawals=withdrawals,
                heat=heat,
                connection_factors=factors,
                ambient_temperatures=periods[index].get("ambient_temperatures"),
                actuator_positions=state["actuator_positions"],
                actuator_modes=periods[index].get("actuator_modes"),
                **stepper,
            )
        state["latched_trips"] = result.get("latched_trips", [])
        state["actuator_positions"] = result["actuator_positions"]
        current = {name: VesselState(**result["states"][name]) for name, _ in network.vessels}
        state["records"].append({"period": index, **result})
        state["totals"] = _totals(state["records"], n)
        state["blocked_feed_component_moles"] = [
            math.fsum(r.get("blocked_feed_component_moles", [0.0] * n)[i] for r in state["records"])
            for i in range(n)
        ]
        state["next_period"] = index + 1
        state["states"] = {name: s.to_dict() for name, s in current.items()}
        audit()
    return dict(state, checksum=_digest(state))
