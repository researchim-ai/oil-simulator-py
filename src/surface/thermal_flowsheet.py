"""Acyclic PR-mixture process recipes and transactional steady-period restart.

Reuses canonical fingerprints from the existing surface schedule machinery;
the molar/enthalpy basis is deliberately distinct from stock black-oil streams.
No recycle iteration, pressure-network or dynamic equipment inventory.
"""

import copy
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from .liquid import _name, _number
from .operation import _digest
from .thermal_process import (
    MixtureFlashSeparator,
    MixtureHeatExchanger,
    MixturePressureRaiser,
    ThermalStream,
    _result,
    let_down_thermal_stream,
    mix_thermal_streams,
)

MODELS = {
    "heat_exchanger": MixtureHeatExchanger,
    "pressure_raiser": MixturePressureRaiser,
    "separator": MixtureFlashSeparator,
}


def _names(names):
    if not isinstance(names, (tuple, list)) or not names or len(set(names)) != len(names):
        raise ValueError("Expected a nonempty list of unique stream names")
    for name in names:
        _name(name)
    return tuple(names)


def _compile(units, feeds):
    if not isinstance(units, (list, tuple)):
        raise TypeError("Units must be a sequence")
    for name in feeds:
        _name(name)
    available = set(feeds)
    used = set()
    unit_names = set()
    compiled = []
    for spec in units:
        if set(spec) != {"name", "type", "inputs", "outputs", "parameters"}:
            raise ValueError("Unit fields must be name/type/inputs/outputs/parameters")
        name, kind = spec["name"], spec["type"]
        _name(name)
        if name in unit_names:
            raise ValueError("Duplicate equipment name")
        unit_names.add(name)
        inputs, outputs = _names(spec["inputs"]), _names(spec["outputs"])
        if set(inputs) - available or set(inputs) & used:
            raise ValueError("Inputs must exist and be consumed only once; use an explicit split")
        if set(outputs) & available:
            raise ValueError("Output name would overwrite an existing stream")
        if kind != "mix" and len(inputs) != 1:
            raise ValueError("Unit requires one input")
        if len(outputs) != (2 if kind in ("split", "separator") else 1):
            raise ValueError("Invalid output count")
        parameters = copy.deepcopy(spec["parameters"])
        if kind in MODELS:
            model = MODELS[kind](name=name, **parameters)
        elif kind == "split":
            if set(parameters) != {"fraction"}:
                raise ValueError("Split requires only fraction")
            f = _number(parameters["fraction"], "fraction", nonnegative=True)
            if f > 1:
                raise ValueError("Split fraction must not exceed one")
            model = f
        elif kind == "letdown":
            if set(parameters) != {"pressure_pa"}:
                raise ValueError("Letdown requires only pressure_pa")
            model = _number(parameters["pressure_pa"], "pressure_pa", positive=True)
        elif kind == "mix":
            if parameters:
                raise ValueError("Mixer takes no equipment parameters")
            model = None
        else:
            raise ValueError(f"Unknown thermal unit: {kind}")
        available.update(outputs)
        used.update(inputs)
        compiled.append((name, kind, inputs, outputs, model))
    return compiled, sorted(available - used)


def _validate_equipment(fluid, compiled):
    for _, kind, _, _, model in compiled:
        if kind == "heat_exchanger":
            for c in fluid.components:
                c._validate(1.0, model.outlet_temperature_k)


def run_thermal_flowsheet(fluid, feeds, units):
    """Evaluate a DAG once; local state only, no mutation on failed operation."""
    if not isinstance(feeds, dict) or not feeds:
        raise ValueError("Feeds must be a nonempty mapping")
    for name, stream in feeds.items():
        _name(name)
        if not isinstance(stream, ThermalStream):
            raise TypeError("Feeds must be ThermalStreams")
        stream._validate_fluid(fluid)
    compiled, terminal = _compile(units, feeds)
    _validate_equipment(fluid, compiled)
    streams = dict(feeds)
    reports = []
    for name, kind, inputs, outputs, model in compiled:
        incoming = tuple(streams[x] for x in inputs)
        if kind == "mix":
            result = mix_thermal_streams(incoming, fluid)
        elif kind == "split":
            result = _result(incoming, incoming[0].split(model), fluid)
        elif kind == "letdown":
            result = let_down_thermal_stream(incoming[0], model, fluid)
        else:
            result = model.run(incoming[0], fluid)
        streams.update(zip(outputs, result["outputs"]))
        reports.append(
            {
                "name": name,
                "type": kind,
                "outputs": list(outputs),
                **{k: v for k, v in result.items() if k != "outputs"},
            }
        )
    totals = _result(
        tuple(feeds.values()),
        tuple(streams[x] for x in terminal),
        fluid,
        heat_w=math.fsum(r["heat_duty_w"] for r in reports),
        shaft_power_w=math.fsum(r["shaft_power_w"] for r in reports),
        heat_entropy_transfer_w_k=math.fsum(r["heat_entropy_transfer_w_k"] for r in reports),
    )
    return {
        "streams": {name: stream.to_dict() for name, stream in streams.items()},
        "products": terminal,
        "units": reports,
        "balances": {k: v for k, v in totals.items() if k != "outputs"},
        "product_states": {name: streams[name].state(fluid) for name in terminal},
    }


def thermal_process_fingerprint(fluid, units, periods, resource_limits=None):
    root = Path(__file__).resolve().parent
    return _digest(
        {
            "revision": "thermal_molar_process_v2",
            "resource_limits": asdict(resource_limits) if resource_limits is not None else None,
            "fluid": asdict(fluid),
            "units": units,
            "periods": periods,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "implementation": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "thermal_process.py",
                    "thermal_flowsheet.py",
                    "thermal_capacity.py",
                    "peng_robinson.py",
                    "peng_robinson_mixture.py",
                    "operation.py",
                    "liquid.py",
                )
            },
        }
    )


def run_thermal_schedule(
    fluid, units, periods, checkpoint=None, stop_after=None, resource_limits=None
):
    """Independent steady periods; checkpoint advances only after full success."""
    from .thermal_capacity import ThermalResourceLimits, period_thermal_units, run_thermal_capacity

    if resource_limits is not None and not isinstance(resource_limits, ThermalResourceLimits):
        raise TypeError("Expected ThermalResourceLimits")
    effective = []
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Periods must be a nonempty sequence")
    for period in periods:
        if (
            not {"duration_s", "feeds"} <= set(period)
            or set(period) - {"duration_s", "feeds", "unit_controls", "resource_limits"}
            or not isinstance(period["feeds"], dict)
            or not period["feeds"]
        ):
            raise ValueError("Each period requires duration_s and nonempty feeds")
        _number(period["duration_s"], "duration_s", positive=True)
        commanded = period_thermal_units(units, period.get("unit_controls", {}))
        limits = (
            ThermalResourceLimits(**period["resource_limits"])
            if "resource_limits" in period
            else resource_limits
        )
        effective.append((commanded, limits))
        compiled, _ = _compile(commanded, period["feeds"])
        _validate_equipment(fluid, compiled)
        for feed in period["feeds"].values():
            if set(feed) != {"pressure_pa", "temperature_k", "molar_rate_mol_s", "mole_fractions"}:
                raise ValueError("Feeds require P/T, molar rate and composition")
            _number(feed["molar_rate_mol_s"], "molar_rate_mol_s", nonnegative=True)
            fluid._composition(feed["mole_fractions"])
            for c in fluid.components:
                c._validate(feed["pressure_pa"], feed["temperature_k"])
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    fingerprint = thermal_process_fingerprint(fluid, units, periods, resource_limits)
    initial = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "elapsed_s": 0.0,
        "records": [],
        "product_component_moles": [0.0] * len(fluid.components),
        "unprocessed_component_moles": [0.0] * len(fluid.components),
        "requested_component_moles": [0.0] * len(fluid.components),
        "heat_energy_j": 0.0,
        "shaft_energy_j": 0.0,
    }
    if checkpoint is None:
        state = initial
    else:
        state = copy.deepcopy(checkpoint)
        if set(state) != set(initial) | {"checksum"}:
            raise ValueError("Invalid thermal checkpoint fields")
        checksum = state.pop("checksum")
        if checksum != _digest(state) or state["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint checksum or process fingerprint mismatch")
        index = state["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(state["records"]) != index
        ):
            raise ValueError("Checkpoint period index mismatch")
        if state["elapsed_s"] != math.fsum(p["duration_s"] for p in periods[:index]):
            raise ValueError("Checkpoint time mismatch")
        # Check cumulative totals against saved accepted records, independently
        # of the checksum (which only detects accidental changes).
        for period_index, record in enumerate(state["records"]):
            expected = [
                periods[period_index]["duration_s"]
                * math.fsum(
                    feed["molar_rate_mol_s"] * feed["mole_fractions"][i]
                    for feed in periods[period_index]["feeds"].values()
                )
                for i in range(len(fluid.components))
            ]
            admitted = (
                np.array(record["product_component_moles"]) + record["unprocessed_component_moles"]
            )
            if (
                record["period"] != period_index
                or not np.allclose(
                    record["requested_component_moles"], expected, rtol=1e-12, atol=1e-9
                )
                or not np.allclose(admitted, expected, rtol=1e-9, atol=1e-9)
            ):
                raise ValueError("Checkpoint component accounting differs from requested feed")
        expected_moles = [
            math.fsum(r["product_component_moles"][i] for r in state["records"])
            for i in range(len(fluid.components))
        ]
        for key, expected in [
            ("product_component_moles", expected_moles),
            (
                "requested_component_moles",
                [
                    math.fsum(r["requested_component_moles"][i] for r in state["records"])
                    for i in range(len(fluid.components))
                ],
            ),
            (
                "unprocessed_component_moles",
                [
                    math.fsum(r["unprocessed_component_moles"][i] for r in state["records"])
                    for i in range(len(fluid.components))
                ],
            ),
            ("heat_energy_j", math.fsum(r["heat_energy_j"] for r in state["records"])),
            ("shaft_energy_j", math.fsum(r["shaft_energy_j"] for r in state["records"])),
        ]:
            if state[key] != expected:
                raise ValueError("Checkpoint cumulative accounting mismatch")
    end = (
        len(periods) if stop_after is None else min(len(periods), state["next_period"] + stop_after)
    )
    for index in range(state["next_period"], end):
        period = periods[index]
        feeds = {
            name: ThermalStream.from_pt(fluid, **spec) for name, spec in period["feeds"].items()
        }
        commanded, limits = effective[index]
        result = (
            run_thermal_capacity(fluid, feeds, commanded, limits)
            if limits is not None
            else run_thermal_flowsheet(fluid, feeds, commanded)
        )
        dt = period["duration_s"]
        moles = [
            dt
            * math.fsum(
                result["streams"][s]["component_molar_rates_mol_s"][i] for s in result["products"]
            )
            for i in range(len(fluid.components))
        ]
        record = {
            "period": index,
            "unit_controls": copy.deepcopy(period.get("unit_controls", {})),
            "resource_limits": asdict(limits) if limits is not None else None,
            "requested_component_moles": [
                dt * math.fsum(s.component_molar_rates_mol_s[i] for s in feeds.values())
                for i in range(len(fluid.components))
            ],
            "result": result,
            "product_component_moles": moles,
            "unprocessed_component_moles": [
                dt * x
                for x in result.get("capacity_control", {}).get(
                    "unprocessed_component_molar_rates_mol_s", [0.0] * len(fluid.components)
                )
            ],
            "heat_energy_j": dt * result["balances"]["heat_duty_w"],
            "shaft_energy_j": dt * result["balances"]["shaft_power_w"],
        }
        state["records"].append(record)
        state["next_period"] = index + 1
        state["elapsed_s"] = math.fsum(p["duration_s"] for p in periods[: index + 1])
        state["product_component_moles"] = [
            math.fsum(r["product_component_moles"][i] for r in state["records"])
            for i in range(len(fluid.components))
        ]
        state["unprocessed_component_moles"] = [
            math.fsum(r["unprocessed_component_moles"][i] for r in state["records"])
            for i in range(len(fluid.components))
        ]
        state["requested_component_moles"] = [
            math.fsum(r["requested_component_moles"][i] for r in state["records"])
            for i in range(len(fluid.components))
        ]
        for key in ("heat_energy_j", "shaft_energy_j"):
            state[key] = math.fsum(r[key] for r in state["records"])
    return dict(state, checksum=_digest(state))
