"""Source-bound protected liquid restart with deterministic physics replay.

Replay is deliberately paid once on resume, not after every new period. It
audits the complete stored trajectory, including acoustic/shaft energy, the persistent controller clock and
trip/availability memory, rather than trusting a self-consistent checksum alone.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import scipy

from .liquid import ConservationError, _number
from .liquid_protection import ProtectedLiquidState
from .operation import _digest
from .torch_backend import TORCH_SOURCES, runtime_fingerprint


def run_protected_liquid_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    system._check(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Protected liquid schedule requires nonempty periods")
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
        "reset_trips",
        "redispatch",
        "max_substeps",
        "hydraulic_state_transfer",
        "hydraulic_journal",
        "hydraulic_journal_branches",
        "max_control_samples",
        "max_protection_samples",
    }
    for period in periods:
        if not isinstance(period, dict) or "duration_s" not in period or set(period) - allowed:
            raise ValueError("Invalid protected liquid period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        system.controller.physics._controls(
            period.get("boundary_pressures_pa"), period.get("valve_openings")
        )
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
                    "liquid_protection.py",
                    "liquid_protection_schedule.py",
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
            raise ValueError("Protected liquid checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Protected liquid checkpoint index mismatch")
        for i, record in enumerate(candidate["records"]):
            replay = {
                "period": i,
                "initial_state": current.to_dict(),
                **system.advance(current, **periods[i]),
            }
            if _digest(record) != _digest(replay):
                raise ConservationError("Protected liquid checkpoint physics replay mismatch")
            current = ProtectedLiquidState.from_dict(replay["state"])
        if current.to_dict() != candidate["state"]:
            raise ConservationError("Protected liquid checkpoint final memory mismatch")
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
        current = ProtectedLiquidState.from_dict(record["state"])
    return {**saved, "checksum": _digest(saved)}
