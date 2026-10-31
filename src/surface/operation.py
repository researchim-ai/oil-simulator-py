"""Steady operating periods, throughput/energy accounting and checked restart.

Each period holds a steady state. This is not separator inventory, linepack,
startup transients or reservoir time integration. Shutdown pressures are absent.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import scipy

from .dispatch import FacilityCapacityControl
from .liquid import EquipmentOperatingError, NetworkResult, _name, _number
from .process import ComponentStream

COMPONENTS = ("oil_m3_s", "water_m3_s", "gas_m3_s")


def _canonical(value):
    if hasattr(value, "tolist"):
        return _canonical(value.tolist())
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            _canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def facility_fingerprint(facility):
    from .gathering import GatheringFacility

    if isinstance(facility, GatheringFacility):
        import torch

        source_root = Path(__file__).resolve().parent
        return _digest(
            {
                "revision": "homogeneous_blackoil_gathering_schedule_v1",
                "plant": facility_fingerprint(facility.facility),
                "gathering": facility.gathering_network.to_dict(),
                "pressure_control": (
                    asdict(facility.gathering_pressure_control)
                    if facility.gathering_pressure_control is not None
                    else None
                ),
                "feed_nodes": facility.feed_nodes,
                "limits": [asdict(limit) for limit in facility.limits],
                "torch": torch.__version__,
                "implementation": {
                    name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                    for name in (
                        "gathering.py",
                        "blackoil_network.py",
                        "blackoil_control.py",
                        "blackoil_pipe.py",
                        "blackoil_valve.py",
                        "blackoil_properties.py",
                        "multiphase.py",
                    )
                },
            }
        )
    stages = []
    for stage in facility.stages:
        pvt = stage.pvt
        stages.append(
            {
                "name": stage.name,
                "pressure_pa": stage.pressure_pa,
                "pvt": {
                    key: getattr(pvt, key)
                    for key in ("P", "Bo", "Bw", "Bg", "Rs", "Rv", "oil_rs", "oil_tables")
                },
            }
        )
    source_root = Path(__file__).resolve().parent
    implementation = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in (
            "liquid.py",
            "gas.py",
            "gas_valve.py",
            "process.py",
            "facility.py",
            "curves.py",
            "dispatch.py",
            "operation.py",
            "station.py",
            "control.py",
        )
    }
    pvt_source = source_root.parent / "simulator" / "pvt.py"
    if pvt_source.exists():
        implementation["simulator/pvt.py"] = hashlib.sha256(pvt_source.read_bytes()).hexdigest()
    return _digest(
        {
            "implementation": implementation,
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "revision": "steady_surface_operation_v1",
            "densities": asdict(facility.densities),
            "feeds": [asdict(s) for s in facility.feeds],
            "stages": stages,
            "water_network": facility.water_network.to_dict(),
            "water_feed_node": facility.water_feed_node,
            "limits": [asdict(limit) for limit in facility.limits],
            "gas_export": None if facility.gas_export is None else asdict(facility.gas_export),
            "station_control": None
            if facility.station_control is None
            else asdict(facility.station_control),
        }
    )


@dataclass(frozen=True)
class FacilityPeriod:
    name: str
    duration_s: float
    feed_scales: tuple | None = None
    pump_speed_ratios: dict = field(default_factory=dict)
    valve_openings: dict = field(default_factory=dict)
    gas_header_pressure_pa: float | None = None
    gas_cooling_temperature_k: float | None = None
    shutdown: bool = False
    unavailable_pumps: tuple | None = None
    station_target_pressure_pa: float | None = None
    gathering_valve_openings: dict = field(default_factory=dict)
    gathering_target_pressure_pa: float | None = None
    gas_valve_max_opening: float | None = None
    gas_valve_outlet_pressure_pa: float | None = None

    def __post_init__(self):
        _name(self.name)
        _number(self.duration_s, "duration_s", positive=True)
        if not isinstance(self.shutdown, bool):
            raise TypeError("shutdown must be boolean")
        if self.unavailable_pumps is not None:
            if isinstance(self.unavailable_pumps, str):
                raise TypeError("Unavailable pumps must be a sequence of names")
            names = tuple(self.unavailable_pumps)
            for name in names:
                _name(name)
            if len(set(names)) != len(names):
                raise ValueError("Duplicate unavailable pump")
            object.__setattr__(self, "unavailable_pumps", names)
        if self.feed_scales is not None:
            values = tuple(self.feed_scales)
            for value in values:
                if not 0 <= _number(value, "feed scale", nonnegative=True) <= 1:
                    raise ValueError("Feed scales must be in [0,1]")
            object.__setattr__(self, "feed_scales", values)
        if not isinstance(self.gathering_valve_openings, dict):
            raise TypeError("gathering_valve_openings must be a mapping")
        for name, value in self.gathering_valve_openings.items():
            _name(name)
            if not 0 < _number(value, "gathering valve opening", positive=True) <= 1:
                raise ValueError("Gathering valve opening must be in (0,1]")
        if (
            self.gas_valve_max_opening is not None
            and not 0
            <= _number(self.gas_valve_max_opening, "gas valve max opening", nonnegative=True)
            <= 1
        ):
            raise ValueError("Gas valve max opening must be in [0,1]")
        if self.gas_valve_outlet_pressure_pa is not None:
            _number(self.gas_valve_outlet_pressure_pa, "gas valve outlet pressure", positive=True)
        if self.gathering_target_pressure_pa is not None:
            _number(self.gathering_target_pressure_pa, "gathering target", positive=True)
        for value in self.pump_speed_ratios.values():
            _number(value, "pump speed", positive=True)
        for value in self.valve_openings.values():
            if not 0 <= _number(value, "valve opening", nonnegative=True) <= 1:
                raise ValueError("Valve opening must be in [0,1]")
        for name in (
            "gas_header_pressure_pa",
            "gas_cooling_temperature_k",
            "station_target_pressure_pa",
        ):
            if getattr(self, name) is not None:
                _number(getattr(self, name), name, positive=True)


def run_facility_schedule(
    facility, periods, capacity_control=None, checkpoint=None, stop_after=None
):
    periods = tuple(periods)
    if not periods or not all(isinstance(p, FacilityPeriod) for p in periods):
        raise ValueError("Schedule requires FacilityPeriods")
    if len({p.name for p in periods}) != len(periods):
        raise ValueError("Duplicate operating-period names")
    if capacity_control is not None and not isinstance(capacity_control, FacilityCapacityControl):
        raise TypeError("capacity_control must be FacilityCapacityControl")
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 1
    ):
        raise ValueError("stop_after must be a positive number of additional periods")
    signature = _digest(
        {
            "periods": [asdict(p) for p in periods],
            "capacity": None if capacity_control is None else asdict(capacity_control),
        }
    )
    plant_signature = facility_fingerprint(facility)
    if checkpoint is None:
        state = {
            "version": 1,
            "facility_fingerprint": plant_signature,
            "schedule_fingerprint": signature,
            "next_period": 0,
            "elapsed_s": 0.0,
            "controls": {
                "feed_scales": [1.0] * len(facility.feeds),
                "pump_speed_ratios": {},
                "valve_openings": {},
                "gas_header_pressure_pa": None,
                "gas_cooling_temperature_k": None,
                "gas_valve_max_opening": None,
                "gas_valve_outlet_pressure_pa": None,
                "unavailable_pumps": [],
                "station_target_pressure_pa": None,
            },
            "station_running": [],
            "last_network_guess": None,
            "ledger": [],
        }
    else:
        state = copy.deepcopy(checkpoint)
        checksum = state.pop("checksum", None)
        if checksum != _digest(state):
            raise ValueError("Schedule checkpoint checksum mismatch")
        if (
            state.get("version") != 1
            or state.get("facility_fingerprint") != plant_signature
            or state.get("schedule_fingerprint") != signature
        ):
            raise ValueError("Schedule checkpoint model or operating plan mismatch")
        index = state.get("next_period")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= len(periods):
            raise ValueError("Invalid schedule restart position")
        if len(state.get("ledger", [])) != index or not math.isclose(
            state["elapsed_s"],
            math.fsum(p.duration_s for p in periods[:index]),
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError("Schedule checkpoint timeline mismatch")
    # All writes below are to local state. A failed period does not alter the caller's checkpoint.
    start = state["next_period"]
    end = len(periods) if stop_after is None else min(len(periods), start + stop_after)
    for index in range(start, end):
        period = periods[index]
        controls = state["controls"]
        if period.feed_scales is not None:
            controls["feed_scales"] = list(period.feed_scales)
        if len(controls["feed_scales"]) != len(facility.feeds):
            raise ValueError("Period feed count mismatch")
        controls["pump_speed_ratios"].update(period.pump_speed_ratios)
        controls["valve_openings"].update(period.valve_openings)
        if period.unavailable_pumps is not None:
            controls["unavailable_pumps"] = list(period.unavailable_pumps)
        for name in (
            "gas_header_pressure_pa",
            "gas_cooling_temperature_k",
            "station_target_pressure_pa",
        ):
            if getattr(period, name) is not None:
                controls[name] = getattr(period, name)
        if hasattr(facility, "validate_gathering_controls"):
            controls.setdefault("gathering_valve_openings", {}).update(
                period.gathering_valve_openings
            )
            if period.gathering_target_pressure_pa is not None:
                controls["gathering_target_pressure_pa"] = period.gathering_target_pressure_pa
            facility.validate_gathering_controls(
                controls["gathering_valve_openings"], controls.get("gathering_target_pressure_pa")
            )
        elif period.gathering_valve_openings or period.gathering_target_pressure_pa is not None:
            raise ValueError("Gathering controls require a gathering facility")
        for name in ("gas_valve_max_opening", "gas_valve_outlet_pressure_pa"):
            if getattr(period, name) is not None:
                controls[name] = getattr(period, name)
        if any(
            controls[name] is not None
            for name in ("gas_valve_max_opening", "gas_valve_outlet_pressure_pa")
        ) and (facility.gas_export is None or facility.gas_export.export_valve is None):
            raise ValueError("Scheduled gas valve controls require an export valve")
        # Validate actuator names even in a shutdown period, where hydraulics is skipped.
        if set(controls["pump_speed_ratios"]) - {p.name for p in facility.water_network.pumps}:
            raise ValueError("Unknown scheduled pump control")
        if set(controls["valve_openings"]) - {v.name for v in facility.water_network.valves}:
            raise ValueError("Unknown scheduled valve control")
        station = facility.station_control
        if station is None:
            if controls["unavailable_pumps"] or controls["station_target_pressure_pa"] is not None:
                raise ValueError("Scheduled station controls require a pump station")
        else:
            if set(controls["unavailable_pumps"]) - set(station.pumps):
                raise ValueError("Unknown scheduled unavailable pump")
            if controls["pump_speed_ratios"]:
                raise ValueError("Pump station owns scheduled pump speeds")
        if facility.gas_export is None and any(
            controls[key] is not None
            for key in ("gas_header_pressure_pa", "gas_cooling_temperature_k")
        ):
            raise ValueError("Scheduled gas controls require a gas export system")
        potential = ComponentStream.mix(
            s.split(f)[0] for s, f in zip(facility.feeds, controls["feed_scales"])
        )
        dispatch = None
        result = None
        if not period.shutdown:
            guess = (
                None
                if state["last_network_guess"] is None
                else NetworkResult.from_dict(state["last_network_guess"])
            )
            if capacity_control is None:
                result = facility.run(
                    initial=guess, previous_running=state["station_running"], **controls
                )
            else:
                dispatch = capacity_control.solve(
                    facility, initial=guess, previous_running=state["station_running"], **controls
                )
                result = dispatch["facility"]
                if result is None:
                    raise EquipmentOperatingError(
                        f"{period.name}: no feasible sampled operating point"
                    )
            if not result["feasible"]:
                raise EquipmentOperatingError(f"{period.name}: facility capacity limits violated")
            state["last_network_guess"] = result["network"]
            state["station_running"] = (
                []
                if result.get("pump_station") is None
                else list(result["pump_station"]["running_pumps"])
            )
        else:
            state["station_running"] = []
        accepted = ComponentStream() if result is None else ComponentStream(**result["feed"])
        product_streams = (
            [] if result is None else [ComponentStream(**s) for s in result["products"].values()]
        )
        product = ComponentStream.mix(product_streams)
        dt = period.duration_s

        def integrate(stream, interval_s=dt):
            return {key.removesuffix("_s"): getattr(stream, key) * interval_s for key in COMPONENTS}

        accepted_volumes = integrate(accepted)
        product_volumes = integrate(product)
        curtailed = {
            key.removesuffix("_s"): max(0.0, getattr(potential, key) - getattr(accepted, key)) * dt
            for key in COMPONENTS
        }
        energies = {}
        for metric in (
            "shaft_power_total_w",
            "gas_drive_power_w",
            "gas_cooling_duty_w",
            "water_letdown_power_w",
            "gathering_friction_dissipation_w",
        ):
            value = (
                0.0
                if result is None
                else result["measurements"].get(metric, {"value": 0.0})["value"]
            )
            energies[metric.removesuffix("_w") + "_j"] = value * dt
        energies["facility_mechanical_energy_j"] = (
            energies["shaft_power_total_j"] + energies["gas_drive_power_j"]
        )
        dispatch_audit = (
            None
            if dispatch is None
            else {key: value for key, value in dispatch.items() if key != "facility"}
        )
        state["ledger"].append(
            {
                "name": period.name,
                "duration_s": dt,
                "shutdown": period.shutdown,
                "controls": copy.deepcopy(controls),
                "facility": result,
                "dispatch": dispatch_audit,
                "accepted_standard_volumes_m3": accepted_volumes,
                "product_standard_volumes_m3": product_volumes,
                "curtailed_potential_standard_volumes_m3": curtailed,
                "energies_j": energies,
            }
        )
        state["next_period"] = index + 1
        state["elapsed_s"] = math.fsum(p.duration_s for p in periods[: index + 1])
    ledger = state["ledger"]
    for index, row in enumerate(ledger):
        if row["name"] != periods[index].name or row["duration_s"] != periods[index].duration_s:
            raise ValueError("Schedule ledger does not match operating periods")
        for category in (
            "accepted_standard_volumes_m3",
            "product_standard_volumes_m3",
            "curtailed_potential_standard_volumes_m3",
        ):
            if set(row[category]) != {"oil_m3", "water_m3", "gas_m3"}:
                raise ValueError("Invalid component volume ledger")
            for value in row[category].values():
                _number(value, "ledger volume", nonnegative=True)
        for key in ("oil_m3", "water_m3", "gas_m3"):
            if not math.isclose(
                row["accepted_standard_volumes_m3"][key],
                row["product_standard_volumes_m3"][key],
                rel_tol=1e-12,
                abs_tol=1e-14,
            ):
                raise ValueError("Schedule component-volume conservation failed")
        for value in row["energies_j"].values():
            _number(value, "ledger energy", nonnegative=True)
    totals = {
        category: {
            key: math.fsum(row[category][key] for row in ledger)
            for key in ("oil_m3", "water_m3", "gas_m3")
        }
        for category in (
            "accepted_standard_volumes_m3",
            "product_standard_volumes_m3",
            "curtailed_potential_standard_volumes_m3",
        )
    }
    totals["energies_j"] = {
        key: math.fsum(row["energies_j"][key] for row in ledger)
        for key in (
            "shaft_power_total_j",
            "gas_drive_power_j",
            "gas_cooling_duty_j",
            "water_letdown_power_j",
            "gathering_friction_dissipation_j",
            "facility_mechanical_energy_j",
        )
    }
    state["checksum"] = _digest(state)
    return {
        "model": "piecewise_steady_facility_schedule",
        "completed": state["next_period"] == len(periods),
        "processed_periods": state["next_period"] - start,
        "elapsed_s": state["elapsed_s"],
        "totals": totals,
        "checkpoint": state,
    }
