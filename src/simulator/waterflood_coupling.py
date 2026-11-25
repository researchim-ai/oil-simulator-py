"""Transactional production/treated-water/injection closure of one FIM step.

Dispatch first admits a producer cap at zero injection, then an injection cap
at that fixed producer cap. This is an explicit feasible dispatch policy, not a
global production/injection optimizer. Every trial solves the SAME old state.
"""

import math
from copy import copy, deepcopy

import numpy as np

try:
    from ..surface import ComponentStream, SurfaceFacility
    from ..surface.liquid import (
        ConservationError,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        Pump,
    )
except ImportError:
    from surface import ComponentStream, SurfaceFacility
    from surface.liquid import (
        ConservationError,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        Pump,
    )

from .fim_transaction import _StepState
from .mixed_surface_wells import MixedSurfaceBlackOilWells


class WaterfloodCoupledFIM:
    def __init__(self, solver, spec):
        if set(spec) - {
            "facility",
            "makeup_water_max_kg_s",
            "fraction_tolerance",
            "dispatch_bhp_tolerance_pa",
            "min_production_fraction",
            "max_evaluations",
            "max_total_drive_power_w",
        }:
            raise ValueError("Unknown coupled_waterflood fields")
        if not isinstance(solver.wells, MixedSurfaceBlackOilWells):
            raise PropertyModelError("Waterflood coupling requires mixed physical surface wells")
        self.solver = solver
        self.facility = SurfaceFacility.from_dict(spec["facility"])
        self.makeup_max = spec.get("makeup_water_max_kg_s", 0.0)
        self.tolerance = spec.get("fraction_tolerance", 1e-4)
        self.bhp_tolerance = spec.get("dispatch_bhp_tolerance_pa", 10.0)
        if (
            isinstance(self.bhp_tolerance, bool)
            or not math.isfinite(self.bhp_tolerance)
            or self.bhp_tolerance <= 0
        ):
            raise ValueError("Positive finite dispatch BHP tolerance required")
        self.min_production = spec.get("min_production_fraction", 1e-6)
        self.maximum_evaluations = spec.get("max_evaluations", 90)
        self.max_power = spec.get("max_total_drive_power_w", math.inf)
        if (
            any(
                isinstance(v, bool) or not math.isfinite(v)
                for v in [self.makeup_max, self.tolerance, self.min_production]
            )
            or self.makeup_max < 0
            or not 0 < self.tolerance < 1
            or not 0 < self.min_production <= 1
        ):
            raise ValueError("Invalid waterflood availability/search settings")
        if (
            type(self.maximum_evaluations) is not int
            or self.maximum_evaluations < 5
            or isinstance(self.max_power, bool)
            or self.max_power <= 0
            or math.isnan(self.max_power)
        ):
            raise ValueError("Invalid waterflood search/power budget")
        wells = solver.wells
        net = wells.production.surface_network
        if (
            net.model.densities != self.facility.densities
            or net.by_name[net.root].pressure_pa != self.facility.stages[0].pressure_pa
        ):
            raise PropertyModelError("Production gathering and facility boundaries differ")
        for stage in self.facility.stages:
            for key in ["P", "Bo", "Bw", "Bg", "Rs", "Rv", "mu_o", "mu_w", "mu_g", "oil_rs"]:
                if not np.array_equal(getattr(stage.pvt, key), getattr(net.model.pvt, key)):
                    raise PropertyModelError("Waterflood facility must share reservoir PVT")
            if set(stage.pvt.oil_tables) != set(net.model.pvt.oil_tables) or any(
                not np.array_equal(value, net.model.pvt.oil_tables[rs][key])
                for rs, table in stage.pvt.oil_tables.items()
                for key, value in table.items()
            ):
                raise PropertyModelError("Waterflood facility PVTO differs")
        boundary = next(n for n in self.facility.water_network.nodes if n.pressure_pa is not None)
        tree = wells.injection.tree
        if (
            not math.isclose(
                boundary.pressure_pa, tree.by_name[tree.root].pressure_pa, rel_tol=0, abs_tol=1e-6
            )
            or not math.isclose(
                boundary.elevation_m, tree.by_name[tree.root].elevation_m, rel_tol=0, abs_tol=1e-6
            )
            or not math.isclose(
                self.facility.water_network.fluid.density_kg_m3,
                tree.fluid.density_kg_m3,
                rel_tol=1e-12,
            )
        ):
            raise PropertyModelError(
                "Treated-water handoff and injection supply pressure/elevation/density differ"
            )
        self.controls = {
            "injection_enabled": True,
            "pump_speed_ratios": {},
            "makeup_water_max_kg_s": self.makeup_max,
            "facility_controls": {},
        }
        self.report = None
        self.last_attempt = None

    def set_controls(self, **controls):
        if set(controls) - set(self.controls):
            raise ValueError("Unknown waterflood operating control")
        updated = {**self.controls, **deepcopy(controls)}
        q = updated["makeup_water_max_kg_s"]
        if (
            type(updated["injection_enabled"]) is not bool
            or isinstance(q, bool)
            or not math.isfinite(q)
            or q < 0
        ):
            raise ValueError("Invalid waterflood operating control")
        # Validate equipment speed controls atomically without changing a FIM state.
        from dataclasses import replace

        pumps = {p.name: p for p in self.solver.wells.injection.tree.order if isinstance(p, Pump)}
        if not isinstance(updated["pump_speed_ratios"], dict) or set(
            updated["pump_speed_ratios"]
        ) - set(pumps):
            raise ValueError("Unknown injection pump speed")
        for name, speed in updated["pump_speed_ratios"].items():
            replace(pumps[name], speed_ratio=speed)
        from .facility_coupling import FacilityCoupledFIM

        if (
            not isinstance(updated["facility_controls"], dict)
            or set(updated["facility_controls"]) - FacilityCoupledFIM.control_fields
        ):
            raise ValueError("Unknown production facility control")
        self.controls = updated

    def _inventory(self):
        s = self.solver
        f = s.fluid
        return (
            s.masses(f.pressure.flatten().double(), f.s_w.flatten().double(), s.x, s.sat, s.ug)
            .sum(1)
            .detach()
            .cpu()
            .tolist()
        )

    def step(self, dt, current_time=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite waterflood dt required")
        s = self.solver
        w = s.wells
        initial = _StepState(s)
        previous = self.report
        caps = [well.control_value for well in w.wells]
        if any(
            well.control_type not in (("rate",) if well.type == "producer" else ("rate", "shut"))
            or not math.isfinite(cap)
            or cap < 0
            or (well.type == "producer" and cap == 0)
            for well, cap in zip(w.wells, caps)
        ):
            raise PropertyModelError(
                "Waterflood requires positive producer caps and nonnegative water-injection caps"
            )
        before = self._inventory()
        trace = []
        success = False
        w.injection.enabled = self.controls["injection_enabled"]
        w.injection.speed_ratios = dict(self.controls["pump_speed_ratios"])
        w.injection.allow_infeasible_trial = True

        def trial(pfrac, ifrac):
            if len(trace) >= self.maximum_evaluations:
                raise HydraulicConvergenceError("Waterflood dispatch evaluation budget exhausted")
            initial.restore(reusable=True)
            for i, well in enumerate(w.wells):
                well.control_value = caps[i] * (pfrac if well.type == "producer" else ifrac)
            if not s.step(dt, current_time=current_time):
                raise HydraulicConvergenceError("FIM failed inside waterflood trial")
            pressures = {
                name: row["bhp_pa"]
                for name, row in w.surface_report["wells"].items()
                if row.get("bhp_status") != "isolated_gauge"
            }
            streams = [
                ComponentStream.from_mass_rates(
                    *row["component_mass_rates_kg_s"], self.facility.densities
                )
                for row in w.production.surface_report["wells"].values()
            ]
            plant = copy(self.facility)
            plant.feeds = tuple(streams)
            try:
                result = plant.run(**self.controls["facility_controls"])
            except PropertyModelError:
                raise
            except EquipmentOperatingError as error:
                trace.append(
                    {
                        "production_fraction": pfrac,
                        "injection_fraction": ifrac,
                        "feasible": False,
                        "reason": str(error),
                    }
                )
                return False, {"well_bhp_pa": pressures}
            injection = w.injection.injection_report
            available = ComponentStream(**result["products"]["water"]).mass_rates(
                self.facility.densities
            )
            if available["oil"] or available["gas"]:
                raise PropertyModelError("First waterflood requires pure separated water")
            produced = available["water"]
            injected = injection["supply_mass_rate_kg_s"]
            makeup = max(0.0, injected - produced)
            disposed = max(0.0, produced - injected)
            power = (
                result["measurements"]["facility_drive_power_total_w"]["value"]
                + injection["shaft_power_w"]
            )
            feasible = (
                result["feasible"]
                and injection["feasible"]
                and makeup <= self.controls["makeup_water_max_kg_s"] + 1e-12
                and power <= self.max_power
            )
            data = {
                "well_bhp_pa": pressures,
                "facility": result,
                "injection_network": injection,
                "water_rates_kg_s": {
                    "produced": produced,
                    "injected": injected,
                    "recycled": min(produced, injected),
                    "makeup": makeup,
                    "disposed": disposed,
                },
                "total_drive_power_w": power,
            }
            trace.append(
                {
                    "production_fraction": pfrac,
                    "injection_fraction": ifrac,
                    "feasible": bool(feasible),
                    "water_rates_kg_s": data["water_rates_kg_s"],
                    "total_drive_power_w": power,
                    "facility_violations": result["violations"],
                    "injection_violations": injection["violations"],
                }
            )
            return bool(feasible), data

        def pressure_span(a, b):
            left, right = a["well_bhp_pa"], b["well_bhp_pa"]
            return max(
                (abs(left[name] - right[name]) for name in left.keys() & right.keys()), default=0.0
            )

        def bracket(low, high, a, b):
            return {
                "low_fraction": low,
                "high_fraction": high,
                "low_bhp_pa": a["well_bhp_pa"],
                "high_bhp_pa": b["well_bhp_pa"],
                "maximum_common_bhp_gap_pa": pressure_span(a, b),
            }

        try:
            feasible, data = trial(1.0, 0.0)
            full_data = data
            pfrac = 1.0
            production_bracket = bracket(1.0, 1.0, data, data)
            if not feasible:
                feasible, low_data = trial(self.min_production, 0.0)
                if not feasible:
                    raise HydraulicConvergenceError(
                        "No feasible production anchor for waterflood dispatch"
                    )
                low, high = self.min_production, 1.0
                high_data = full_data
                while (
                    high - low > self.tolerance
                    or pressure_span(low_data, high_data) > self.bhp_tolerance
                ):
                    middle = (low + high) / 2
                    feasible, middle_data = trial(middle, 0.0)
                    if feasible:
                        low, low_data = middle, middle_data
                    else:
                        high, high_data = middle, middle_data
                pfrac = low
                data = low_data
                production_bracket = bracket(low, high, low_data, high_data)
            ifrac = 0.0
            injection_anchor = data
            injection_bracket = bracket(0.0, 0.0, data, data)
            if self.controls["injection_enabled"] and any(
                caps[i] > 0 for i in w.injection_well_indices
            ):
                feasible, data = trial(pfrac, 1.0)
                if feasible:
                    ifrac = 1.0
                    injection_bracket = bracket(1.0, 1.0, data, data)
                else:
                    low, high = 0.0, 1.0
                    low_data, high_data = injection_anchor, data
                    while (
                        high - low > self.tolerance
                        or pressure_span(low_data, high_data) > self.bhp_tolerance
                    ):
                        middle = (low + high) / 2
                        feasible, middle_data = trial(pfrac, middle)
                        if feasible:
                            low, low_data = middle, middle_data
                        else:
                            high, high_data = middle, middle_data
                    ifrac = low
                    injection_bracket = bracket(low, high, low_data, high_data)
            feasible, data = trial(pfrac, ifrac)
            if not feasible:
                raise HydraulicConvergenceError("Final waterflood dispatch failed certification")
            after = self._inventory()
            products = {
                name: ComponentStream(**stream).mass_rates(self.facility.densities)
                for name, stream in data["facility"]["products"].items()
            }
            exports = [
                math.fsum(product[c] for product in products.values())
                for c in ["oil", "water", "gas"]
            ]
            water = data["water_rates_kg_s"]
            exports[1] = water["disposed"]
            external_in = [0.0, water["makeup"] * dt, 0.0]
            external_out = [q * dt for q in exports]
            residual = [
                a + b - c - d for a, b, c, d in zip(before, external_in, after, external_out)
            ]
            mass_budgets = [0.01 + 1e-8 * max(a, b) for a, b in zip(external_in, external_out)]
            if any(abs(r) > budget for r, budget in zip(residual, mass_budgets)):
                raise ConservationError(
                    "Accepted waterflood step violates external component accounting"
                )
            report = {
                "model": "coupled_waterflood_v1",
                "dt_s": dt,
                "production_fraction": pfrac,
                "injection_fraction": ifrac,
                "dispatch_resolution": {
                    "fraction_tolerance": self.tolerance,
                    "bhp_tolerance_pa": self.bhp_tolerance,
                    "production": production_bracket,
                    "injection": injection_bracket,
                    "scope": "Resolved endpoint span of the feasible/infeasible search bracket, excluding isolated gauges; not a global nonlinear error bound",
                },
                "operator_controls": deepcopy(self.controls),
                "trace": trace,
                **data,
                "material_account": {
                    "component_order": ["oil", "water", "gas"],
                    "inventory_before_kg": before,
                    "inventory_after_kg": after,
                    "external_in_kg": external_in,
                    "external_out_kg": external_out,
                    "residual_kg": residual,
                    "absolute_budgets_kg": mass_budgets,
                },
                "scope": "Isothermal black-oil, producer tree, ideal separation/treated-water manifold, external makeup at handoff pressure, incompressible-water injection tree. No surface storage/thermal reservoir, experimental calibration or global dispatch optimum.",
            }
            if (
                abs(water["produced"] + water["makeup"] - water["injected"] - water["disposed"])
                > 1e-10
            ):
                raise ConservationError("Waterflood handoff does not conserve water")
            self.report = report
            self.last_attempt = {"success": True, "evaluations": len(trace)}
            success = True
            return True
        except HydraulicConvergenceError as error:
            self.last_attempt = {"success": False, "reason": str(error), "trace": trace}
            return False
        except Exception as error:
            self.last_attempt = {
                "success": False,
                "reason": f"{type(error).__name__}: {error}",
                "trace": trace,
            }
            raise
        finally:
            if not success:
                initial.restore()
                self.report = previous
            for well, cap in zip(w.wells, caps):
                well.control_value = cap
            w.injection.allow_infeasible_trial = False
