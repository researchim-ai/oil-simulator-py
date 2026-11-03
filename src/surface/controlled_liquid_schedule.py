"""Source-bound controlled liquid restart with deterministic physics replay.

Replay is deliberately paid once on resume, not after every new period. It
audits the complete stored trajectory, including acoustic/shaft energy, the persistent controller clock and
trip/availability memory, rather than trusting a self-consistent checksum alone.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import scipy

from .controlled_liquid import ControlledLiquidState
from .liquid import ConservationError, _number
from .operation import _digest
from .torch_backend import TORCH_SOURCES, runtime_fingerprint


def run_controlled_liquid_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    system._check(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Controlled liquid schedule requires nonempty periods")
    periods = copy.deepcopy(periods)
    allowed = {
        "duration_s",
        "mode",
        "pressure_setpoint_pa",
        "speed_ratio",
        "running_pumps",
        "motor_torques_nm",
        "unavailable",
        "boundary_pressures_pa",
        "valve_openings",
        "actuator_modes",
        "reset_trip",
        "redispatch",
        "max_substeps",
        "hydraulic_state_transfer",
        "hydraulic_journal",
        "hydraulic_journal_branches",
        "max_control_samples",
    }
    for period in periods:
        if not isinstance(period, dict) or "duration_s" not in period or set(period) - allowed:
            raise ValueError("Invalid controlled liquid period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        system.physics._controls(period.get("boundary_pressures_pa"), period.get("valve_openings"))
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
            "scipy": scipy.__version__,
            "torch_runtime": runtime_fingerprint(),
            "source": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    *TORCH_SOURCES,
                    "dynamic_station.py",
                    "dynamic_station_schedule.py",
                    "liquid_transient.py",
                    "liquid_relief.py",
                    "liquid_transient_schedule.py",
                    "controlled_liquid.py",
                    "controlled_liquid_schedule.py",
                    "station.py",
                    "liquid.py",
                    "curves.py",
                    "control.py",
                    "pid.py",
                    "rotating_drive.py",
                    "actuator.py",
                    "operation.py",
                )
            },
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
            raise ValueError("Invalid liquid transient checkpoint fields")
        checksum = candidate.pop("checksum")
        if checksum != _digest(candidate) or candidate["fingerprint"] != fingerprint:
            raise ValueError("Controlled liquid checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Controlled liquid checkpoint index mismatch")
        for i, record in enumerate(candidate["records"]):
            replay = {
                "period": i,
                "initial_state": current.to_dict(),
                **system.advance(current, **periods[i]),
            }
            if _digest(record) != _digest(replay):
                raise ConservationError("Controlled liquid checkpoint physics replay mismatch")
            current = ControlledLiquidState.from_dict(replay["state"])
        if current.to_dict() != candidate["state"]:
            raise ConservationError("Controlled liquid checkpoint final memory mismatch")
        saved = candidate
    start = saved["next_period"]
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for index in range(start, end):
        record = {
            "period": index,
            "initial_state": current.to_dict(),
            **system.advance(current, **periods[index]),
        }
        saved["records"].append(record)
        saved["next_period"] = index + 1
        saved["state"] = copy.deepcopy(record["state"])
        current = ControlledLiquidState.from_dict(record["state"])
    return {**saved, "checksum": _digest(saved)}
