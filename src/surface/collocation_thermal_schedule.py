"""Source-bound period changes and audited restart of collocation controls."""

import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import scipy

from .collocation_thermal_station import CollocationThermalState, CollocationThermalStation
from .liquid import _number
from .operation import _digest
from .thermal_liquid_schedule import _run_replay_schedule
from .torch_backend import runtime_fingerprint


def run_collocation_thermal_schedule(system, initial, periods, checkpoint=None, stop_after=None):
    if not isinstance(system, CollocationThermalStation):
        raise TypeError("Expected CollocationThermalStation")
    system._check(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Nonempty control periods required")
    periods = copy.deepcopy(periods)
    allowed = set(inspect.signature(system.advance).parameters) - {"state"}
    for period in periods:
        if not isinstance(period, dict) or "duration_s" not in period or set(period) - allowed:
            raise ValueError("Invalid collocation control period fields")
        _number(period["duration_s"], "period duration", positive=True)
        system.physics._controls(period.get("boundary_pressures_pa"), period.get("valve_openings"))
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "model": system.to_dict(),
            "initial": initial.to_dict(),
            "periods": periods,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "runtime": runtime_fingerprint(),
            "source": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(root.glob("*.py"))
            },
        }
    )
    return _run_replay_schedule(
        system, initial, periods, checkpoint, stop_after, fingerprint, CollocationThermalState
    )
