"""Checked restart and signed boundary accounting for a distributed pipe."""

import copy
import hashlib
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy

from .liquid import ConservationError, _number
from .operation import _digest


def run_pipe_schedule(pipe, initial, periods, stepper=None, checkpoint=None, stop_after=None):
    pipe.inspect(initial)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Pipe schedule needs periods")
    for period in periods:
        if "duration_s" not in period or set(period) - {"duration_s", "left", "right"}:
            raise ValueError("Unknown pipe period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        pipe._boundary(period.get("left"))
        pipe._boundary(period.get("right"))
    stepper = copy.deepcopy(stepper or {})
    if set(stepper) - {"initial_dt_s", "maximum_dt_s", "minimum_dt_s", "max_substeps"}:
        raise ValueError("Unknown pipe stepper fields")
    for key, value in stepper.items():
        if key == "max_substeps":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Invalid pipe substep budget")
        elif value is not None or key != "maximum_dt_s":
            _number(value, key, positive=True)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be nonnegative integer")
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "model": asdict(pipe),
            "model_type": type(pipe).__name__,
            "initial": initial.to_dict(),
            "periods": periods,
            "stepper": stepper,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "source": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
                    "transient_pipe.py",
                    "thermal_pipe.py",
                    "thermal_flux.py",
                    "pipe_schedule.py",
                    "liquid.py",
                    "operation.py",
                )
            },
        }
    )
    state = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "state": initial.to_dict(),
        "records": [],
    }
    if checkpoint is not None:
        saved = copy.deepcopy(checkpoint)
        if set(saved) != set(state) | {"checksum"}:
            raise ValueError("Invalid pipe checkpoint fields")
        checksum = saved.pop("checksum")
        if checksum != _digest(saved) or saved["fingerprint"] != fingerprint:
            raise ValueError("Pipe checkpoint checksum/fingerprint mismatch")
        index = saved["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(saved["records"]) != index
        ):
            raise ValueError("Pipe checkpoint index mismatch")
        state = saved

    def audit():
        current = type(initial)(**state["state"])
        pipe.inspect(current)
        time = initial.time_s
        thermal = hasattr(initial, "total_energy_j")
        transfer = np.zeros(4 if thermal else 3)
        impulse = 0.0
        heat = 0.0
        for i, record in enumerate(state["records"]):
            if record["period"] != i or not math.isclose(
                math.fsum(s["dt_s"] for s in record["steps"]),
                periods[i]["duration_s"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Pipe checkpoint period timeline mismatch")
            for step in record["steps"]:
                _number(step["dt_s"], "dt_s", positive=True)
                transfer += (
                    np.asarray(step["left_boundary_transfer"]) - step["right_boundary_transfer"]
                )
                impulse += step["body_impulse_kg_m_s"]
                if thermal:
                    heat += step["heat_j"]
            time = initial.time_s + math.fsum(p["duration_s"] for p in periods[: i + 1])
        if not math.isclose(current.time_s, time, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Pipe checkpoint clock mismatch")
        old = np.array(
            [sum(initial.liquid_mass_kg), sum(initial.gas_mass_kg), sum(initial.momentum_kg_m_s)]
        )
        new = np.array(
            [sum(current.liquid_mass_kg), sum(current.gas_mass_kg), sum(current.momentum_kg_m_s)]
        )
        sources = [0, 0, impulse]
        if thermal:
            old = np.append(old, sum(initial.total_energy_j))
            new = np.append(new, sum(current.total_energy_j))
            sources.append(heat)
        residual = new - old - transfer - sources
        scale = np.maximum(1.0, abs(old) + abs(transfer) + np.abs(sources))
        if not np.isfinite(residual).all() or np.max(abs(residual) / scale) > 1e-7:
            raise ConservationError("Pipe checkpoint accumulated balance failed")
        return current

    current = audit()
    end = (
        len(periods) if stop_after is None else min(len(periods), state["next_period"] + stop_after)
    )
    for i in range(state["next_period"], end):
        period = periods[i]
        result = pipe.advance(current, **period, **stepper)
        state["records"].append({"period": i, **result})
        state["state"] = result["state"]
        state["next_period"] = i + 1
        current = audit()
    return dict(state, checksum=_digest(state))
