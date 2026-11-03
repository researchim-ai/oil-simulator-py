"""Source-bound three-phase separator schedules with full physical prefix replay."""

import copy
import hashlib
from pathlib import Path

import numpy as np

from .liquid import ConservationError, _number
from .operation import _digest
from .three_phase_properties import ThreePhaseStream
from .three_phase_separator import ThreePhaseState
from .torch_backend import runtime_fingerprint

SOURCES = (
    "three_phase_properties.py",
    "three_phase_separator.py",
    "three_phase_schedule.py",
    "torch_linear.py",
    "separator_geometry.py",
    "vessel_wall.py",
    "liquid.py",
    "operation.py",
)


def run_three_phase_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    system.inspect(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Nonempty separator periods required")
    periods = copy.deepcopy(periods)
    allowed = {
        "duration_s",
        "feeds",
        "openings",
        "heat_duties_w",
        "utility_temperatures_k",
        "max_substeps",
    }
    commands = []
    for item in periods:
        if not isinstance(item, dict) or "duration_s" not in item or set(item) - allowed:
            raise ValueError("Unknown or missing three-phase period field")
        _number(item["duration_s"], "duration_s", positive=True)
        command = copy.deepcopy(item)
        feeds = {}
        for name, specs in command.get("feeds", {}).items():
            if not isinstance(specs, (list, tuple)):
                raise TypeError("Feed rows require stream lists")
            feeds[name] = []
            for spec in specs:
                if isinstance(spec, ThreePhaseStream):
                    stream = spec
                else:
                    if (
                        not isinstance(spec, dict)
                        or not {"phase_mass_rates_kg_s", "pressure_pa", "temperature_k"}
                        <= set(spec)
                        or set(spec)
                        - {"phase_mass_rates_kg_s", "pressure_pa", "temperature_k", "elevation_m"}
                    ):
                        raise ValueError("Invalid three-phase feed stream fields")
                    stream = ThreePhaseStream.create(system.fluid, **spec)
                feeds[name].append(stream)
        command["feeds"] = feeds
        commands.append(command)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "system": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "numpy": np.__version__,
            "torch": runtime_fingerprint(),
            "source": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in SOURCES},
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
            current = ThreePhaseState.from_dict(replay["state"])
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
        current = ThreePhaseState.from_dict(record["state"])
    return {**saved, "checksum": _digest(saved)}
