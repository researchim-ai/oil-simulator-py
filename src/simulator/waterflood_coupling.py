"""Transactional production/treated-water/injection closure of one FIM step.

Dispatch first admits a producer cap at zero injection, then an injection cap
at that fixed producer cap. This is an explicit feasible dispatch policy, not a
global production/injection optimizer. Every trial solves the SAME old state.
"""

import math
from copy import copy, deepcopy

import numpy as np
import torch

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

from .facility_temporal_control import FacilityTemporalControl, temporal_step
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
            "temporal_control",
            "dispatch_solver",
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
        try:
            from ..surface.waterflood_search import WaterfloodDispatchControl
        except ImportError:
            from surface.waterflood_search import WaterfloodDispatchControl
        self.dispatch = WaterfloodDispatchControl(**spec.get("dispatch_solver", {}))
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
        self.accepted_stages = ()
        options = spec.get("temporal_control")
        self.temporal_control = None
        if options is not None:
            options = dict(options)
            if "solution_ratio_tolerance" in options:
                raise ValueError("Use component inventory tolerances across phase changes, not solution ratios")
            self.component_rtol = options.pop("component_relative_tolerance", 1e-3)
            self.component_atol = options.pop("component_absolute_tolerance_kg", 1e-6)
            self.energy_rtol = options.pop("drive_energy_relative_tolerance", .002)
            self.energy_atol = options.pop("drive_energy_absolute_tolerance_j", 1.0)
            for value in [self.component_rtol, self.component_atol, self.energy_rtol, self.energy_atol]:
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("Positive finite waterflood temporal tolerances required")
            if self.component_rtol >= 1 or self.energy_rtol >= 1:
                raise ValueError("Relative temporal tolerances must be below one")
            self.temporal_control = FacilityTemporalControl(**options)
            if self.temporal_control.withdrawal_error_scale != "increment":
                raise ValueError("Waterflood temporal control requires increment scaling")
            bhp = self.temporal_control.well_bhp_tolerance_pa
            if bhp is not None and self.bhp_tolerance > .1 * bhp:
                raise ValueError("Dispatch BHP resolution must be <= one tenth of temporal BHP tolerance")

    def limit_initial_time_step_s(self, proposed_s, *, clipped=False):
        return self._limit_time_step_s(proposed_s, clipped=clipped)

    @property
    def limit_next_time_step_s(self):
        return self._limit_time_step_s if self.temporal_control is not None else None

    def _limit_time_step_s(self, proposed_s, *, clipped=False):
        if self.temporal_control is None:
            return proposed_s
        row = (self.report or {}).get("temporal_control") if (
            self.report is not None and self.report["operator_controls"] == self.controls
        ) else None
        return self.temporal_control.limit(
            proposed_s, None if row is None else row["macro_dt_s"],
            None if row is None else row["indicator"], clipped=clipped,
        )

    def limit_rejected_time_step_s(self, proposed_s, *, failed_dt_s):
        errors = (self.last_attempt or {}).get("temporal_errors")
        if self.temporal_control is None or not errors:
            return proposed_s
        return self.temporal_control.rejected_step(proposed_s, failed_dt_s, max(errors.values()))

    @staticmethod
    def _temporal_fields(solver):
        f = solver.fluid
        return {
            "pressure": f.pressure.detach().clone(),
            "s_w": f.s_w.detach().clone(),
            "s_g": f.s_g.detach().clone(),
            "component_mass": solver.masses(
                f.pressure.flatten(), f.s_w.flatten(), solver.x, solver.sat, solver.ug
            ).detach().clone(),
        }

    @staticmethod
    def _temporal_stage(coupled, duration, end):
        surface = deepcopy(coupled.solver.wells.surface_report)
        # Preserve well AND component identity. Production must not cancel
        # injection when estimating time integration of the internal recycle.
        rates = torch.tensor([
            row["component_mass_rates_kg_s"] for row in surface["wells"].values()
        ], dtype=torch.float64)
        return {
            "time_days": end / 86400, "dt_s": duration,
            "withdrawn_component_mass_kg": (duration * rates).tolist(),
            "newton_iterations": coupled.solver.last_newton_iters,
            "surface": surface, "facility_coupling": deepcopy(coupled.report),
            "well_order": list(surface["wells"]),
        }

    @staticmethod
    def _transfers(stage):
        r = stage["facility_coupling"]
        account = r["material_account"]
        return torch.tensor(
            account["external_in_kg"] + account["external_out_kg"]
            + [r["water_rates_kg_s"][name] * stage["dt_s"]
               for name in ["produced", "injected", "recycled", "makeup", "disposed"]],
            dtype=torch.float64,
        )

    def _temporal_defect(self, coarse_fields, fine_fields, coarse, first, second):
        policy = self.temporal_control
        cm = torch.tensor(coarse["withdrawn_component_mass_kg"], dtype=torch.float64)
        fm = (torch.tensor(first["withdrawn_component_mass_kg"], dtype=torch.float64)
              + torch.tensor(second["withdrawn_component_mass_kg"], dtype=torch.float64))
        errors = policy.defect(coarse_fields, fine_fields, cm, fm)
        c, f = coarse_fields["component_mass"], fine_fields["component_mass"]
        scale = self.component_atol + self.component_rtol * torch.maximum(c.abs(), f.abs())
        errors["component_inventory"] = float(((c - f).abs() / scale).max().cpu())
        ct, ft = self._transfers(coarse), self._transfers(first) + self._transfers(second)
        scale = policy.withdrawal_absolute_tolerance_kg + policy.withdrawal_relative_tolerance * torch.maximum(ct.abs(), ft.abs())
        errors["resource_transfers"] = float(((ct - ft).abs() / scale).max())
        ce = coarse["facility_coupling"]["total_drive_power_w"] * coarse["dt_s"]
        fe = sum(stage["facility_coupling"]["total_drive_power_w"] * stage["dt_s"] for stage in [first, second])
        errors["drive_energy"] = abs(ce - fe) / (self.energy_atol + self.energy_rtol * max(abs(ce), abs(fe)))
        def bhp(stage):
            return {name: row["bhp_pa"] for name, row in stage["surface"]["wells"].items()
                    if row.get("bhp_status") != "isolated_gauge"}
        errors.update(policy.well_defect(bhp(coarse), bhp(second)))
        if any(not math.isfinite(v) for v in errors.values()):
            raise HydraulicConvergenceError("Nonfinite waterflood temporal defect")
        probe = {
            "well_order": coarse["well_order"],
            "coarse_well_withdrawal_kg": cm.tolist(),
            "coarse_transfers_kg": ct.tolist(),
            "coarse_drive_energy_j": ce,
            "coarse_well_bhp_pa": bhp(coarse),
        }
        second["facility_coupling"]["temporal_probe"] = probe
        self.report["temporal_probe"] = probe
        return errors

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
        self.accepted_stages = ()
        if self.temporal_control is not None:
            s = self.solver
            settings = {name: getattr(s, name) for name in (
                "tol_cnv", "tol_mb", "min_newton", "lin_tol_max", "lin_tol_min",
                "lin_accept", "relaxed_max_pv_fraction", "tol_well_bhp",
            )}
            s.tol_cnv = min(s.tol_cnv, .1 * self.temporal_control.saturation_tolerance,
                            .1 * self.component_rtol)
            s.tol_mb = min(s.tol_mb, 1e-8)
            s.min_newton = max(s.min_newton, 1)
            s.lin_tol_max, s.lin_tol_min = min(s.lin_tol_max, 1e-8), min(s.lin_tol_min, 1e-9)
            s.lin_accept = min(s.lin_accept, 1e-7)
            s.relaxed_max_pv_fraction = 0.
            if self.temporal_control.well_bhp_tolerance_pa is not None:
                s.tol_well_bhp = min(s.tol_well_bhp, .1 * self.temporal_control.well_bhp_tolerance_pa)
            try:
                return temporal_step(
                    self, dt, current_time, stage_factory=self._temporal_stage,
                    field_factory=self._temporal_fields, defect_evaluator=self._temporal_defect,
                )
            finally:
                for name, value in settings.items():
                    setattr(s, name, value)
        return self._step_once(dt, current_time)

    def _step_once(self, dt, current_time=None):
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
        reservoir_evaluations = 0
        selected_state = None
        from .fim_initial_cell_cache import InitialCellStateCache
        try:
            from ..surface.capacity_bracket import facility_constraint_margins
            from ..surface.waterflood_search import pressure_span, resolve_boundary
        except ImportError:
            from surface.capacity_bracket import facility_constraint_margins
            from surface.waterflood_search import pressure_span, resolve_boundary
        cells = InitialCellStateCache(
            s, dt, current_time, self.dispatch.initial_cell_state_cache_bytes
        ) if self.dispatch.reuse_initial_cell_state else None
        water_scale = max(sum(caps[i] for i in w.injection_well_indices)
                          * self.facility.densities.water_kg_m3 / 86400, 1e-8)
        success = False
        w.injection.enabled = self.controls["injection_enabled"]
        w.injection.speed_ratios = dict(self.controls["pump_speed_ratios"])
        w.injection.allow_infeasible_trial = True

        def trial(pfrac, ifrac):
            nonlocal reservoir_evaluations, selected_state
            if len(trace) >= self.maximum_evaluations:
                raise HydraulicConvergenceError("Waterflood dispatch evaluation budget exhausted")
            initial.restore(reusable=True)
            for i, well in enumerate(w.wells):
                well.control_value = caps[i] * (pfrac if well.type == "producer" else ifrac)
            reservoir_evaluations += 1
            from contextlib import nullcontext
            with cells.installed() if cells is not None else nullcontext():
                converged = s.step(dt, current_time=current_time)
            if not converged:
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
                try:
                    from ..surface.gas_valve import GasValveCapacityError
                except ImportError:
                    from surface.gas_valve import GasValveCapacityError
                margins = ({"facility:gas_export.valve": error.normalized_margin}
                           if isinstance(error, GasValveCapacityError) else {})
                trace.append(
                    {
                        "production_fraction": pfrac,
                        "injection_fraction": ifrac,
                        "feasible": False,
                        "reason": str(error),
                        "constraint_margins": margins,
                    }
                )
                return False, {"well_bhp_pa": pressures, "constraint_margins": margins}
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
            margins = {"facility:" + key: value for key, value in facility_constraint_margins(result).items()}
            if self.controls["injection_enabled"]:
                margins.update({"injection:" + key: value for key, value in injection.get("constraint_margins", {}).items()})
            margins["water_resource"] = (
                injected - produced - self.controls["makeup_water_max_kg_s"] - 1e-12
            ) / water_scale
            if math.isfinite(self.max_power):
                margins["shared_drive_power"] = (power - self.max_power) / self.max_power
            # An infeasible opaque equipment state must not borrow the sign
            # of unrelated feasible facets for a capacity proposal.
            if not feasible and max((v for v in margins.values() if v is not None), default=-1) <= 0:
                margins = {}
            data["constraint_margins"] = margins
            trace.append(
                {
                    "production_fraction": pfrac,
                    "injection_fraction": ifrac,
                    "feasible": bool(feasible),
                    "water_rates_kg_s": data["water_rates_kg_s"],
                    "total_drive_power_w": power,
                    "facility_violations": result["violations"],
                    "injection_violations": injection["violations"],
                    "constraint_margins": margins,
                }
            )
            if feasible and self.dispatch.commit == "state_restore":
                selected_state = (
                    (pfrac, ifrac), _StepState(s), data, len(trace) - 1,
                    getattr(getattr(s, "debug", None), "step_rec", None),
                )
            return bool(feasible), data

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
                low, high, low_data, high_data = resolve_boundary(
                    lambda point: trial(point, 0.), self.min_production, 1., low_data,
                    full_data, self.tolerance, self.bhp_tolerance, self.dispatch.method,
                )
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
                    low, high, low_data, high_data = resolve_boundary(
                        lambda point: trial(pfrac, point), 0., 1., injection_anchor,
                        data, self.tolerance, self.bhp_tolerance, self.dispatch.method,
                    )
                    ifrac = low
                    injection_bracket = bracket(low, high, low_data, high_data)
            if self.dispatch.commit == "state_restore":
                if selected_state is None or selected_state[0] != (pfrac, ifrac):
                    raise RuntimeError("Missing solved state for selected waterflood controls")
                selected_state[1].restore()
                data = selected_state[2]
                if getattr(s, "debug", None) is not None:
                    s.debug.step_rec = selected_state[4]
                commit = {"method": "state_restore", "solved_trace_index": selected_state[3]}
            else:
                feasible, data = trial(pfrac, ifrac)
                if not feasible:
                    raise HydraulicConvergenceError("Final waterflood dispatch failed certification")
                commit = {"method": "replay", "solved_trace_index": len(trace) - 1}
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
                "model": "coupled_waterflood_v2",
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
                "dispatch_commit": commit,
                "reservoir_evaluations": reservoir_evaluations,
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
            if cells is not None:
                report["initial_cell_state_cache"] = cells.report()
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
            if self.last_attempt is not None:
                self.last_attempt["reservoir_evaluations"] = reservoir_evaluations
            if not success:
                initial.restore()
                self.report = previous
            for well, cap in zip(w.wells, caps):
                well.control_value = cap
            w.injection.allow_infeasible_trial = False
