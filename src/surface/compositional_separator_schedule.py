"""Transactional monolithic-separator schedules with physical prefix replay."""

import copy
import hashlib
from pathlib import Path

from .component_energy import ComponentEnergyStream
from .compositional_separator import CompositionalSeparatorState
from .liquid import ConservationError, _number
from .operation import _digest
from .torch_backend import runtime_fingerprint

SOURCES = (
    "phase_geometry.py",
    "separator_geometry.py",
    "three_phase_properties.py",
    "compositional_separator.py",
    "compositional_separator_schedule.py",
    "equilibrium_properties.py",
    "equilibrium_inventory.py",
    "component_energy.py",
    "peng_robinson.py",
    "peng_robinson_mixture.py",
    "torch_linear.py",
    "vessel_wall.py",
    "liquid.py",
    "operation.py",
    "torch_backend.py",
)


def run_compositional_separator_schedule(
    system, initial, periods, checkpoint=None, stop_after=None
):
    system.inspect(initial)
    root = Path(__file__).resolve().parent
    source_hashes = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Nonempty separator periods required")
    periods = copy.deepcopy(periods)
    commands = []
    for period in periods:
        if (
            not isinstance(period, dict)
            or "duration_s" not in period
            or set(period)
            - {"duration_s", "feeds", "heat_w", "utility_temperature_k", "openings", "max_substeps"}
        ):
            raise ValueError("Unknown or missing compositional separator period field")
        _number(period["duration_s"], "duration_s", positive=True)
        if period.get("heat_w", 0) and "utility_temperature_k" not in period:
            raise ValueError("Nonzero heat requires utility temperature")
        _number(period.get("heat_w", 0), "heat_w")
        if "utility_temperature_k" in period:
            _number(period["utility_temperature_k"], "utility_temperature_k", positive=True)
        budget = period.get("max_substeps", 10000)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("Invalid period work budget")
        openings = period.get("openings", {})
        if not isinstance(openings, dict) or set(openings) - {p.name for p in system.outlets}:
            raise ValueError("Unknown separator opening")
        for value in openings.values():
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Opening outside [0,1]")
        specs = period.get("feeds", ())
        if not isinstance(specs, (list, tuple)):
            raise TypeError("Separator feeds require a sequence")
        streams = []
        canonical = []
        for spec in specs:
            if isinstance(spec, ComponentEnergyStream):
                stream = spec
                raw = {"stream": stream.to_dict()}
            elif isinstance(spec, dict) and set(spec) == {"stream"}:
                stream = ComponentEnergyStream.from_dict(spec["stream"])
                raw = copy.deepcopy(spec)
            elif isinstance(spec, dict) and set(spec) == {"from_pt"}:
                args = dict(spec["from_pt"])
                args.setdefault("gravity_m_s2", system.gravity_m_s2)
                args.setdefault("elevation_m", system.elevation_m)
                stream = ComponentEnergyStream.from_pt(system.fluid, **args)
                raw = copy.deepcopy(spec)
            else:
                raise ValueError("Feed requires exactly stream or from_pt")
            stream.validate_fluid(system.fluid)
            if stream.gravity_m_s2 != system.gravity_m_s2:
                raise ValueError("Feed/model gravity references differ")
            streams.append(stream)
            canonical.append(raw)
        period["feeds"] = canonical
        command = {**period, "feeds": tuple(streams)}
        commands.append(command)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    fingerprint = _digest(
        {
            "system": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "device": str(system.fluid.device),
            "runtime": runtime_fingerprint(),
            "source": source_hashes,
        }
    )
    saved = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "state": initial.to_dict(),
        "records": [],
    }
    current = initial
    if checkpoint is not None:
        candidate = copy.deepcopy(checkpoint)
        if set(candidate) != set(saved) | {"checksum"}:
            raise ValueError("Invalid separator checkpoint fields")
        checksum = candidate.pop("checksum")
        if checksum != _digest(candidate) or candidate["fingerprint"] != fingerprint:
            raise ValueError("Separator checkpoint checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Invalid separator checkpoint index")
        for i, record in enumerate(candidate["records"]):
            replay = {
                "period": i,
                "initial_state": current.to_dict(),
                **system.advance(current, **commands[i]),
            }
            if _digest(replay) != _digest(record):
                raise ConservationError("Separator checkpoint physical replay mismatch")
            current = CompositionalSeparatorState.from_dict(replay["state"])
        if current.to_dict() != candidate["state"]:
            raise ConservationError("Separator checkpoint final state mismatch")
        saved = candidate
    start = saved["next_period"]
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for i in range(start, end):
        record = {
            "period": i,
            "initial_state": current.to_dict(),
            **system.advance(current, **commands[i]),
        }
        saved["records"].append(record)
        saved["next_period"] = i + 1
        saved["state"] = copy.deepcopy(record["state"])
        current = CompositionalSeparatorState.from_dict(record["state"])
    if source_hashes != {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES}:
        raise ConservationError("Source files changed during separator schedule")
    return {**saved, "checksum": _digest(saved)}
