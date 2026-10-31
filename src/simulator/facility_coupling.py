"""Transactional rate-cap feedback from a steady facility to black-oil FIM.

Every capacity trial re-solves the SAME implicit reservoir step. Actual well
component masses feed the facility; no post-solve scaling of produced material.
This is an outer constrained solve, not a monolithic facility Jacobian.
"""

import math
from copy import copy
from dataclasses import dataclass

import numpy as np

from .blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from .fim_transaction import _StepState

try:
    from ..surface import (
        ComponentStream,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        SurfaceFacility,
    )
    from ..surface.liquid import InvalidPressureError
except ImportError:
    from surface import (
        ComponentStream,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        SurfaceFacility,
    )
    from surface.liquid import InvalidPressureError


@dataclass(frozen=True)
class CoupledFacilityControl:
    min_fraction: float = 0.001
    fraction_tolerance: float = 1e-4
    scan_points: int = 9
    max_evaluations: int = 60

    def __post_init__(self):
        if not 0 < self.min_fraction <= 1 or not math.isfinite(self.min_fraction):
            raise ValueError("min_fraction must be in (0,1]")
        if not 0 < self.fraction_tolerance < 1 or not math.isfinite(self.fraction_tolerance):
            raise ValueError("fraction_tolerance must be in (0,1)")
        for key, minimum in (("scan_points", 2), ("max_evaluations", 3)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")


class FacilityCoupledFIM:
    def __init__(self, solver, facility, control=None):
        if not isinstance(solver.wells, SurfaceCoupledBlackOilWells):
            raise PropertyModelError(
                "Facility feedback requires a physical black-oil gathering FIM"
            )
        if not isinstance(facility, SurfaceFacility):
            raise TypeError("Expected SurfaceFacility")
        if any(
            w.control_type != "rate"
            or w.type != "producer"
            or not math.isfinite(w.control_value)
            or w.control_value <= 0
            or getattr(w, "ramping_days", 0) > 0
            for w in solver.wells.wells
        ):
            raise PropertyModelError(
                "Facility feedback requires positive producer rate caps without ramps"
            )
        net = solver.wells.surface_network
        if net.model.densities != facility.densities:
            raise PropertyModelError("Reservoir/gathering/facility stock densities differ")
        if net.by_name[net.root].pressure_pa != facility.stages[0].pressure_pa:
            raise PropertyModelError("Gathering receiver must match first separator pressure")
        pvt = net.model.pvt
        other = facility.stages[0].pvt
        for key in ("P", "Bo", "Bw", "Bg", "Rs", "Rv", "mu_o", "mu_w", "mu_g", "oil_rs"):
            if not np.array_equal(getattr(pvt, key), getattr(other, key)):
                raise PropertyModelError("Gathering/facility PVT differs")
        if set(pvt.oil_tables) != set(other.oil_tables) or any(
            not np.array_equal(value, other.oil_tables[rs].get(key))
            for rs, table in pvt.oil_tables.items()
            for key, value in table.items()
        ):
            raise PropertyModelError("Gathering/facility PVTO differs")
        self.solver, self.facility = solver, facility
        self.control = (
            CoupledFacilityControl()
            if control is None
            else (CoupledFacilityControl(**control) if isinstance(control, dict) else control)
        )
        if not isinstance(self.control, CoupledFacilityControl):
            raise TypeError("Expected CoupledFacilityControl")
        self.report = None
        self.last_attempt = None
        self.controls = {}

    def step(self, dt, current_time=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        allowed_controls = {
            "pump_speed_ratios",
            "valve_openings",
            "gas_header_pressure_pa",
            "gas_cooling_temperature_k",
            "unavailable_pumps",
            "station_target_pressure_pa",
            "gas_valve_max_opening",
            "gas_valve_outlet_pressure_pa",
        }
        if set(self.controls) - allowed_controls:
            raise ValueError("Unsupported coupled facility controls; feed scaling is forbidden")
        solver = self.solver
        net = solver.wells.surface_network
        if net.by_name[net.root].pressure_pa != self.facility.stages[0].pressure_pa:
            raise PropertyModelError("Gathering receiver must match first separator pressure")
        if any(
            w.control_type != "rate"
            or w.type != "producer"
            or not math.isfinite(w.control_value)
            or w.control_value <= 0
            or getattr(w, "ramping_days", 0) > 0
            for w in solver.wells.wells
        ):
            raise PropertyModelError(
                "Coupled producer control changed outside supported rate-cap mode"
            )
        # A new capture after each restore keeps the saved well arrays from
        # becoming the live mutable arrays of the next trial.
        initial = _StepState(solver)
        previous = self.report
        caps = [w.control_value for w in solver.wells.wells]
        trace = []
        accepted = False

        def reset():
            nonlocal initial
            initial.restore()
            initial = _StepState(solver)

        def trial(fraction):
            if len(trace) >= self.control.max_evaluations:
                raise HydraulicConvergenceError("Coupled facility evaluation budget exhausted")
            reset()
            for w, cap in zip(solver.wells.wells, caps):
                w.control_value = cap * fraction
            if not solver.step(dt, current_time=current_time):
                raise HydraulicConvergenceError(
                    "Reservoir step failed during facility capacity trial"
                )
            report = solver.wells.surface_report
            streams = [
                ComponentStream.from_mass_rates(
                    *report["wells"][w.name]["component_mass_rates_kg_s"], self.facility.densities
                )
                for w in solver.wells.wells
            ]
            plant = copy(self.facility)
            plant.feeds = tuple(streams)
            try:
                running = (
                    ()
                    if previous is None or previous["facility"].get("pump_station") is None
                    else previous["facility"]["pump_station"]["running_pumps"]
                )
                result = plant.run(previous_running=running, **self.controls)
                expected = ComponentStream.mix(streams).mass_rates(self.facility.densities)
                actual = ComponentStream(**result["feed"]).mass_rates(self.facility.densities)
                if any(
                    not math.isclose(actual[c], expected[c], rel_tol=1e-12, abs_tol=1e-14)
                    for c in expected
                ):
                    raise PropertyModelError(
                        "Facility feed does not match actual reservoir production"
                    )
                feasible = result["feasible"]
                reason = result["violations"]
            except PropertyModelError:
                raise
            except (EquipmentOperatingError, InvalidPressureError) as error:
                feasible = False
                result = None
                reason = [{"physical_domain": str(error)}]
            trace.append({"fraction": fraction, "feasible": feasible, "reason": reason})
            return feasible, result

        try:
            ok, plant = trial(1.0)
            chosen = 1.0
            if not ok:
                grid = [
                    self.control.min_fraction
                    + (1 - self.control.min_fraction) * i / (self.control.scan_points - 1)
                    for i in range(self.control.scan_points - 1)
                ]
                outcomes = [trial(f)[0] for f in grid]
                good = [i for i, x in enumerate(outcomes) if x]
                if not good:
                    raise EquipmentOperatingError(
                        "No sampled feasible coupled reservoir/facility rate cap"
                    )
                if good != list(range(good[0], good[-1] + 1)):
                    raise EquipmentOperatingError(
                        "Disconnected feasible rate-cap regions require a different optimizer"
                    )
                i = good[-1]
                low = grid[i]
                high = grid[i + 1] if i + 1 < len(grid) else 1.0
                while high - low > self.control.fraction_tolerance:
                    mid = (low + high) / 2
                    if trial(mid)[0]:
                        low = mid
                    else:
                        high = mid
                chosen = low
                # Commit a fresh solve at the selected cap, never a scaled copy
                # or a cached intermediate reservoir state from another trial.
                ok, plant = trial(chosen)
                if not ok:
                    raise HydraulicConvergenceError("Selected coupled facility mode failed replay")
            self.report = {
                "model": "outer_rate_cap_coupled_blackoil_fim_facility",
                "fraction": chosen,
                "dt_s": dt,
                "evaluations": len(trace),
                "trace": trace,
                "facility": plant,
                "applied_rate_caps_m3_day": [cap * chosen for cap in caps],
                "scope": "Common producer rate-cap multiplier; each trial solves the same FIM step. No monolithic facility Jacobian.",
            }
            self.last_attempt = {"success": True, "evaluations": len(trace)}
            accepted = True
            return True
        except HydraulicConvergenceError as error:
            self.last_attempt = {"success": False, "reason": str(error), "trace": trace}
            return False
        except Exception as error:
            self.last_attempt = {"success": False, "reason": str(error), "trace": trace}
            raise
        finally:
            if not accepted:
                reset()
                self.report = previous
            # Preserve operator requests; applied caps are recorded separately.
            for w, cap in zip(solver.wells.wells, caps):
                w.control_value = cap
