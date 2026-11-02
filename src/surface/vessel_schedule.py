"""Checked conservative dynamic-vessel schedule and JSON restart."""

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
from .thermal_process import ThermalStream

SCALARS = ("inlet_enthalpy_j", "outlet_enthalpy_j", "heat_j")
VECTORS = ("inlet_component_moles", "outlet_component_moles")


def _totals(records, n):
    steps = [s for r in records for s in r["steps"]]
    result = {k: math.fsum(s[k] for s in steps) for k in SCALARS}
    result.update({k: [math.fsum(s[k][i] for s in steps) for i in range(n)] for k in VECTORS})
    return result


def run_vessel_schedule(
    fluid, vessel, initial_state, periods, stepper=None, checkpoint=None, stop_after=None
):
    stepper = copy.deepcopy(stepper or {})
    if set(stepper) - {"initial_dt_s", "minimum_dt_s", "maximum_dt_s", "max_substeps"}:
        raise ValueError("Unknown vessel stepper option")
    for key, value in stepper.items():
        if key == "max_substeps":
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("max_substeps must be a positive integer")
        elif value is not None or key != "maximum_dt_s":
            _number(value, key, positive=True)
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("Vessel periods must be a nonempty sequence")
    for period in periods:
        if "duration_s" not in period or set(period) - {
            "duration_s",
            "feeds",
            "withdrawals",
            "heat_w",
            "utility_temperature_k",
        }:
            raise ValueError("Unknown or missing vessel period fields")
        _number(period["duration_s"], "duration_s", positive=True)
        if not isinstance(period.get("feeds", {}), dict):
            raise TypeError("Vessel feeds must be a mapping")
        for feed in period.get("feeds", {}).values():
            if set(feed) != {"pressure_pa", "temperature_k", "molar_rate_mol_s", "mole_fractions"}:
                raise ValueError("Invalid vessel feed fields")
            _number(feed["molar_rate_mol_s"], "molar_rate_mol_s", nonnegative=True)
            fluid._composition(feed["mole_fractions"])
            for c in fluid.components:
                c._validate(feed["pressure_pa"], feed["temperature_k"])
        heat = _number(period.get("heat_w", 0.0), "heat_w")
        if heat:
            _number(period.get("utility_temperature_k"), "utility_temperature_k", positive=True)
        withdrawals = period.get("withdrawals", {})
        if not isinstance(withdrawals, dict) or set(withdrawals) - {"bulk", "phase_0", "phase_1"}:
            raise ValueError("Invalid vessel withdrawal ports")
        for value in withdrawals.values():
            _number(value, "outlet_rate", nonnegative=True)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0
    ):
        raise ValueError("stop_after must be a nonnegative integer")
    vessel.inspect(initial_state, fluid)
    root = Path(__file__).resolve().parent
    fingerprint = _digest(
        {
            "revision": "rigid_equilibrium_vessel_v1",
            "fluid": asdict(fluid),
            "vessel": asdict(vessel),
            "initial": initial_state.to_dict(),
            "periods": periods,
            "stepper": stepper,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "implementation": {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in (
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
    state = {
        "fingerprint": fingerprint,
        "next_period": 0,
        "state": initial_state.to_dict(),
        "records": [],
        "totals": _totals([], len(fluid.components)),
    }
    if checkpoint is not None:
        candidate = copy.deepcopy(checkpoint)
        if set(candidate) != set(state) | {"checksum"}:
            raise ValueError("Invalid vessel checkpoint fields")
        checksum = candidate.pop("checksum")
        if checksum != _digest(candidate) or candidate["fingerprint"] != fingerprint:
            raise ValueError("Vessel checkpoint checksum/fingerprint mismatch")
        index = candidate["next_period"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index <= len(periods)
            or len(candidate["records"]) != index
        ):
            raise ValueError("Vessel checkpoint index mismatch")
        for i, record in enumerate(candidate["records"]):
            if record["period"] != i or not math.isclose(
                math.fsum(s["dt_s"] for s in record["steps"]),
                periods[i]["duration_s"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Vessel checkpoint step timeline mismatch")
        if candidate["totals"] != _totals(candidate["records"], len(fluid.components)):
            raise ValueError("Vessel checkpoint accumulation mismatch")
        state = candidate
    current = VesselState(**state["state"])
    vessel.inspect(current, fluid)
    expected_time = initial_state.time_s + math.fsum(
        p["duration_s"] for p in periods[: state["next_period"]]
    )
    if not math.isclose(current.time_s, expected_time, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Vessel checkpoint time mismatch")

    def check_balance(now, totals):
        residual = (
            np.array(now.component_moles)
            - initial_state.component_moles
            - np.array(totals["inlet_component_moles"])
            + totals["outlet_component_moles"]
        )
        energy = (
            now.internal_energy_j
            - initial_state.internal_energy_j
            - totals["inlet_enthalpy_j"]
            + totals["outlet_enthalpy_j"]
            - totals["heat_j"]
        )
        if max(abs(residual)) > 1e-8 * max(1.0, sum(initial_state.component_moles)) or abs(
            energy
        ) > 1e-5 + 1e-8 * max(
            1.0, abs(initial_state.internal_energy_j), abs(totals["inlet_enthalpy_j"])
        ):
            raise ConservationError("Vessel checkpoint global component/energy balance failed")
        return {"component_residuals_mol": residual.tolist(), "energy_residual_j": energy}

    check_balance(current, state["totals"])
    end = (
        len(periods) if stop_after is None else min(len(periods), state["next_period"] + stop_after)
    )
    for index in range(state["next_period"], end):
        period = periods[index]
        feeds = tuple(
            ThermalStream.from_pt(fluid, **feed) for feed in period.get("feeds", {}).values()
        )
        result = vessel.advance(
            current,
            fluid,
            period["duration_s"],
            feeds=feeds,
            withdrawals=period.get("withdrawals"),
            heat_w=period.get("heat_w", 0.0),
            utility_temperature_k=period.get("utility_temperature_k"),
            **stepper,
        )
        current = VesselState(**result["state"])
        state["records"].append({"period": index, **result})
        state["totals"] = _totals(state["records"], len(fluid.components))
        check_balance(current, state["totals"])
        state["next_period"] = index + 1
        state["state"] = current.to_dict()
    # Balance diagnostics live outside the checksum payload until every accepted
    # period has passed. Restart verifies them by recomputation.
    return dict(state, checksum=_digest(state))
