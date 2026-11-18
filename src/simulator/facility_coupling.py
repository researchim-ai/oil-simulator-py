"""Transactional rate-cap feedback from a steady facility to black-oil FIM.

Every capacity trial re-solves the SAME implicit reservoir step. Actual well
component masses feed the facility; no post-solve scaling of produced material.
This is an outer constrained solve, not a monolithic facility Jacobian.
"""

import math
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np
import torch

from .blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from .capacity_amg_reuse import CapacityAMGReuse
from .facility_tangent import FacilityTangentCapture, plant_margin_tangent
from .facility_temporal_control import FacilityTemporalControl, temporal_step
from .facility_time_control import FacilityTimeControl
from .fim_initial_cell_cache import InitialCellStateCache
from .fim_transaction import _StepState
from .rate_cap_invariance import certificate_covers, make_inactive_cap_certificate

try:
    from ..surface import (
        ComponentStream,
        ConservationError,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        SurfaceFacility,
    )
    from ..surface.blackoil_pipe import PipePressureCoverageError
    from ..surface.capacity_bracket import (
        CapacityBracket,
        facility_constraint_margin,
        facility_constraint_margins,
    )
    from ..surface.gas_valve import GasValveCapacityError
    from ..surface.liquid import InvalidPressureError
except ImportError:
    from surface import (
        ComponentStream,
        ConservationError,
        EquipmentOperatingError,
        HydraulicConvergenceError,
        PropertyModelError,
        SurfaceFacility,
    )
    from surface.blackoil_pipe import PipePressureCoverageError
    from surface.capacity_bracket import (
        CapacityBracket,
        facility_constraint_margin,
        facility_constraint_margins,
    )
    from surface.gas_valve import GasValveCapacityError
    from surface.liquid import InvalidPressureError


@dataclass(frozen=True)
class CoupledFacilityControl:
    min_fraction: float = 0.001
    fraction_tolerance: float = 1e-4
    scan_points: int = 9
    max_evaluations: int = 60
    search_method: str = "safeguarded"
    commit_method: str = "replay"
    reuse_inactive_caps: bool = False
    reuse_initial_cell_state: bool = False
    initial_cell_state_cache_bytes: int = 64 * 1024**2
    reuse_capacity_amg: bool = False
    sensitivity_sparse_budget_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if type(self.reuse_capacity_amg) is not bool:
            raise ValueError('reuse_capacity_amg must be boolean')
        if type(self.sensitivity_sparse_budget_bytes) is not int or self.sensitivity_sparse_budget_bytes <= 0:
            raise ValueError('sensitivity_sparse_budget_bytes must be a positive integer')
        if type(self.reuse_initial_cell_state) is not bool:
            raise ValueError('reuse_initial_cell_state must be boolean')
        if type(self.initial_cell_state_cache_bytes) is not int or self.initial_cell_state_cache_bytes <= 0:
            raise ValueError('initial_cell_state_cache_bytes must be a positive integer')
        if type(self.reuse_inactive_caps) is not bool:
            raise ValueError("reuse_inactive_caps must be boolean")
        if self.commit_method not in ("replay", "state_restore"):
            raise ValueError("commit_method must be replay or state_restore")
        if self.search_method not in ("safeguarded", "facet_safeguarded", "sensitivity_safeguarded", "bisect"):
            raise ValueError("Unknown capacity search_method")
        if not 0 < self.min_fraction <= 1 or not math.isfinite(self.min_fraction):
            raise ValueError("min_fraction must be in (0,1]")
        if not 0 < self.fraction_tolerance < 1 or not math.isfinite(self.fraction_tolerance):
            raise ValueError("fraction_tolerance must be in (0,1)")
        for key, minimum in (("scan_points", 2), ("max_evaluations", 3)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")


class FacilityCoupledFIM:
    control_fields = frozenset(
        {
            "pump_speed_ratios",
            "valve_openings",
            "gas_header_pressure_pa",
            "gas_cooling_temperature_k",
            "unavailable_pumps",
            "station_target_pressure_pa",
            "gas_valve_max_opening",
            "gas_valve_outlet_pressure_pa",
        }
    )

    def __init__(
        self,
        solver,
        facility,
        control=None,
        *,
        energy_process=None,
        energy_controls=None,
        time_control=None,
        temporal_control=None,
    ):
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
        if energy_process is not None:
            from .accepted_well_energy import AcceptedWellEnergyProcess

            if not isinstance(energy_process, AcceptedWellEnergyProcess):
                raise TypeError("Expected AcceptedWellEnergyProcess")
        self.energy_process = energy_process
        self.time_control = (
            FacilityTimeControl(**time_control) if isinstance(time_control, dict) else time_control
        )
        if self.time_control is not None and not isinstance(self.time_control, FacilityTimeControl):
            raise TypeError("Expected FacilityTimeControl")
        self.temporal_control = (
            FacilityTemporalControl(**temporal_control) if isinstance(temporal_control, dict) else temporal_control
        )
        if self.temporal_control is not None and not isinstance(self.temporal_control, FacilityTemporalControl):
            raise TypeError("Expected FacilityTemporalControl")
        if self.temporal_control is not None and self.time_control is not None:
            raise ValueError("Choose temporal_control or endpoint-rate time_control")
        self.accepted_stages = ()
        self.temporal_work = {"macro_attempts": 0, "rejected_macro_steps": 0,
                              "coupled_solves": 0, "capacity_evaluations": 0}
        self.energy_controls = {} if energy_controls is None else dict(energy_controls)
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

    def _continuous_controls(self):
        return self.report is not None and self.report.get("operator_controls") == self.controls

    def limit_initial_time_step_s(self, proposed_s, *, clipped=False):
        return self._limit_time_step_s(proposed_s, clipped=clipped)

    def limit_rejected_time_step_s(self, proposed_s, *, failed_dt_s):
        policy = getattr(self, "temporal_control", None)
        errors = (self.last_attempt or {}).get("temporal_errors")
        if policy is None or not errors:
            return proposed_s
        return policy.rejected_step(proposed_s, failed_dt_s, max(errors.values()))

    @property
    def limit_next_time_step_s(self):
        return self._limit_time_step_s if (getattr(self, "time_control", None) is not None
                                          or getattr(self, "temporal_control", None) is not None) else None

    def _limit_time_step_s(self, proposed_s, *, clipped=False):
        temporal = getattr(self, "temporal_control", None)
        if temporal is not None:
            row = self.report.get("temporal_control") if self._continuous_controls() else None
            return temporal.limit(proposed_s, None if row is None else row["macro_dt_s"],
                                  None if row is None else row["indicator"], clipped=clipped)
        policy = getattr(self, "time_control", None)
        if policy is None:
            return proposed_s
        row = self.report.get("rate_time_control") if self._continuous_controls() else None
        return policy.limit(
            proposed_s,
            None if row is None else self.report["dt_s"],
            None if row is None else row["indicator"],
            clipped=clipped,
        )

    def _inventory(self):
        """Reduce on the reservoir device; transfer only three component totals."""
        s = self.solver
        with torch.no_grad():
            values = (
                s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug)
                .sum(1)
                .detach()
                .cpu()
                .numpy()
            )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ConservationError("Invalid coupled reservoir component inventory")
        return dict(zip(("oil", "water", "gas"), map(float, values)))

    def _material_account(self, before, plant, dt):
        """Audit actual accepted material, never requested caps or rejected trials."""
        after = self._inventory()
        rho = self.facility.densities
        feed = ComponentStream(**plant["feed"]).mass_rates(rho)
        products = {
            name: ComponentStream(**stream).mass_rates(rho)
            for name, stream in plant["products"].items()
        }
        outgoing = {c: math.fsum(stream[c] for stream in products.values()) for c in before}
        if any(
            not math.isclose(feed[c], outgoing[c], rel_tol=1e-12, abs_tol=1e-14) for c in before
        ):
            raise ConservationError("Coupled facility products do not match actual feed")
        produced = {c: feed[c] * dt for c in before}
        delivered = {name: {c: rate[c] * dt for c in before} for name, rate in products.items()}
        previous = None if self.report is None else self.report.get("material_account")
        initial = before if previous is None else previous["initial_inventory_kg"]
        cumulative = {
            c: math.fsum([produced[c], 0.0 if previous is None else previous["produced_kg"][c]])
            for c in before
        }
        cumulative_products = {
            name: {
                c: math.fsum(
                    [
                        delivered[name][c],
                        0.0
                        if previous is None
                        else previous["products_kg"].get(name, {}).get(c, 0.0),
                    ]
                )
                for c in before
            }
            for name in delivered
        }
        return {
            "scope": "Stock-component masses, closed producer-only reservoir; no surface storage. Reservoir residuals are diagnostics, not a replacement for FIM CNV/MB.",
            "step": {
                "inventory_before_kg": before,
                "inventory_after_kg": after,
                "produced_kg": produced,
                "products_kg": delivered,
                "reservoir_residual_kg": {c: before[c] - after[c] - produced[c] for c in before},
                "surface_residual_kg": {c: (feed[c] - outgoing[c]) * dt for c in before},
            },
            "elapsed_s": dt + (0.0 if previous is None else previous["elapsed_s"]),
            "initial_inventory_kg": dict(initial),
            "current_inventory_kg": after,
            "produced_kg": cumulative,
            "products_kg": cumulative_products,
            "end_to_end_residual_kg": {
                c: initial[c] - after[c] - math.fsum(p[c] for p in cumulative_products.values())
                for c in before
            },
        }

    def step(self, dt, current_time=None):
        if getattr(self, "temporal_control", None) is not None:
            return temporal_step(self, dt, current_time)
        self.accepted_stages = ()
        return self._step_once(dt, current_time)

    def _step_once(self, dt, current_time=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if set(self.controls) - self.control_fields:
            raise ValueError("Unsupported coupled facility controls; feed scaling is forbidden")
        policy = getattr(self, "time_control", None)
        if (
            policy is not None
            and not self._continuous_controls()
            and dt > policy.startup_step_days * 86400 * (1 + 1e-12)
        ):
            self.last_attempt = {
                "success": False,
                "reason": "Facility startup/event time-step limit",
                "trace": [],
            }
            return False
        if getattr(self, "energy_controls", {}) and getattr(self, "energy_process", None) is None:
            raise PropertyModelError("Caloric controls require an explicit energy process")
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
        inventory_before = self._inventory()
        caps = [w.control_value for w in solver.wells.wells]
        trace = []
        trial_calls = 0
        sample_calls = 0
        inactive_sources = []
        sensitivities = {}
        tangent_work = {'linear_solves': 0, 'plant_evaluations': 0, 'proposals': 0}
        # Smooth continuing intervals already have a close warm endpoint.
        # Spend auxiliary solves at startup/events, or after slow search.
        request_tangent = (not self._continuous_controls()
                           or previous.get('requested_rate_caps_m3_day') != caps)
        initial_cells = (InitialCellStateCache(solver, dt, current_time,
                        self.control.initial_cell_state_cache_bytes)
                         if self.control.reuse_initial_cell_state else None)
        selected_state = None
        amg_scope = CapacityAMGReuse(solver, dt, current_time, enabled=self.control.reuse_capacity_amg)
        accepted = False

        def reset():
            nonlocal initial
            initial.restore()
            initial = _StepState(solver)

        def trial(fraction):
            nonlocal trial_calls, sample_calls, selected_state
            if len(trace) >= self.control.max_evaluations:
                raise HydraulicConvergenceError("Coupled facility evaluation budget exhausted")
            sample_calls += 1
            if getattr(self.control, "reuse_inactive_caps", False):
                for source_index in inactive_sources:
                    source = trace[source_index]
                    if certificate_covers(source['inactive_cap_certificate'], fraction):
                        trace.append({
                            'fraction': fraction, 'feasible': False,
                            'reason': deepcopy(source['reason']),
                            'normalized_margin': source['normalized_margin'],
                            'constraint_margins': deepcopy(source.get('constraint_margins')),
                            'certified_from_trace_index': source_index,
                        })
                        return False, None
            reset()
            trial_calls += 1
            for w, cap in zip(solver.wells.wells, caps):
                w.control_value = cap * fraction
            capture = (FacilityTangentCapture(solver, caps, self.control.sensitivity_sparse_budget_bytes,
                                             fraction=fraction)
                       if self.control.search_method == 'sensitivity_safeguarded' and request_tangent else None)
            try:
                with (initial_cells.installed() if initial_cells is not None else nullcontext()), \
                     (amg_scope.installed() if amg_scope is not None else nullcontext()), \
                     (capture.installed() if capture is not None else nullcontext()):
                    converged = solver.step(dt, current_time=current_time)
            except PipePressureCoverageError as error:
                # Reject only this declared operating-envelope violation.
                # Invalid PVT tables and other PropertyModelError instances
                # still propagate; no pressure extrapolation is permitted.
                trace.append(
                    {
                        "fraction": fraction,
                        "feasible": False,
                        "reason": [{"pipe_pressure_coverage": str(error)}],
                        "normalized_margin": None,
                    }
                )
                return False, None
            if not converged:
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
            margin = None
            constraint_margins = None
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
                margin = facility_constraint_margin(result)
                constraint_margins = facility_constraint_margins(result)
            except PropertyModelError:
                raise
            except GasValveCapacityError as error:
                feasible, result = False, None
                reason = [{"physical_domain": str(error)}]
                margin = error.normalized_margin
                constraint_margins = {"gas_export.valve": margin}
            except (EquipmentOperatingError, InvalidPressureError) as error:
                feasible = False
                result = None
                reason = [{"physical_domain": str(error)}]
            trace.append(
                {
                    "fraction": fraction,
                    "feasible": feasible,
                    "reason": reason,
                    "normalized_margin": margin,
                    "constraint_margins": constraint_margins,
                }
            )
            if capture is not None and fraction < 1:
                response = None
                try:
                    response = capture.response()
                except (HydraulicConvergenceError, torch.cuda.OutOfMemoryError):
                    capture.reason = 'auxiliary_linear_failure'
                tangent_work['linear_solves'] += capture.linear_solves
                slopes, count, reason = ({}, 0, capture.reason)
                if response is not None:
                    slopes, count, reason = plant_margin_tangent(
                        plant, response, fraction, self.controls, running)
                tangent_work['plant_evaluations'] += count
                sensitivities[fraction] = slopes
                trace[-1]['capacity_sensitivity'] = {'derivatives': slopes, 'reason': reason,
                    'linear_solves': capture.linear_solves, 'plant_evaluations': count}
            if not feasible and getattr(self.control, "reuse_inactive_caps", False):
                proof = make_inactive_cap_certificate(
                    report,
                    {w.name: solver.wells.step_targets[i] for i, w in enumerate(solver.wells.wells)},
                    {w.name: cap for w, cap in zip(solver.wells.wells, caps)},
                    {w.name: solver.wells.bhp_limit(w) for w in solver.wells.wells},
                    fraction,
                    version=2 if self.control.search_method == 'sensitivity_safeguarded' else 1,
                )
                if proof is not None:
                    trace[-1]['inactive_cap_certificate'] = proof
                    inactive_sources.append(len(trace) - 1)
            if (feasible and fraction < 1 and self.control.commit_method == "state_restore"
                    and (selected_state is None or fraction > selected_state[0])):
                selected_state = (fraction, _StepState(solver), result, len(trace) - 1,
                                  getattr(getattr(solver, "debug", None), "step_rec", None))
            return feasible, result

        try:
            final_bracket = None
            capacity_commit = None
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
                margins = {row["fraction"]: row["normalized_margin"] for row in trace}
                facets = ({row["fraction"]: row.get("constraint_margins") for row in trace}
                          if self.control.search_method in ('facet_safeguarded', 'sensitivity_safeguarded') else {})
                bracket = CapacityBracket(
                    low, high, self.control.fraction_tolerance, margins[low], margins[high],
                    low_margins=facets.get(low), high_margins=facets.get(high)
                )
                if self.control.search_method != "bisect" and previous is not None:
                    warm = previous["fraction"]
                    if not bracket.complete and bracket.low < warm < bracket.high:
                        status, _ = trial(warm)
                        bracket.update(warm, status, trace[-1]["normalized_margin"],
                                       margins=trace[-1].get("constraint_margins") if facets else None)
                while not bracket.complete:
                    predicted = (bracket.tangent_proposal(sensitivities.get(bracket.low), sensitivities.get(bracket.high))
                                 if self.control.search_method == 'sensitivity_safeguarded' else None)
                    tangent_work['proposals'] += int(predicted is not None)
                    mid = (
                        (bracket.low + bracket.high) / 2
                        if self.control.search_method == "bisect"
                        else predicted if predicted is not None else bracket.proposal()
                    )
                    request_tangent = request_tangent or bracket.slow_updates >= 2
                    prediction_bracket = (bracket.low, bracket.high)
                    status, _ = trial(mid)
                    if predicted is not None:
                        trace[-1]['capacity_proposal'] = {'low_fraction': prediction_bracket[0],
                                                         'high_fraction': prediction_bracket[1]}
                    bracket.update(mid, status, trace[-1]["normalized_margin"],
                                   margins=trace[-1].get("constraint_margins") if facets else None)
                chosen = bracket.low
                final_bracket = {
                    "feasible_fraction": bracket.low,
                    "infeasible_fraction": bracket.high,
                    "width": bracket.high - bracket.low,
                    "fraction_tolerance": self.control.fraction_tolerance,
                }
                if self.control.commit_method == "state_restore":
                    # Each saved endpoint was independently converged from
                    # this SAME old-time state at exactly this rate cap.
                    # Restore its fields, phase flags, well cumulatives and
                    # reports, never scale another trial's flows or fields.
                    if selected_state is None or selected_state[0] != chosen:
                        raise RuntimeError("Missing solved state for selected rate cap")
                    selected_state[1].restore()
                    if getattr(solver, "debug", None) is not None:
                        solver.debug.step_rec = selected_state[4]
                    plant = selected_state[2]
                    capacity_commit = {"method": "state_restore", "solved_trace_index": selected_state[3]}
                else:
                    ok, plant = trial(chosen)
                    if not ok:
                        raise HydraulicConvergenceError("Selected coupled facility mode failed replay")
            rate_indicator = None
            if policy is not None and self._continuous_controls():
                before_rates = ComponentStream(**previous["facility"]["feed"]).mass_rates(
                    self.facility.densities
                )
                after_rates = ComponentStream(**plant["feed"]).mass_rates(self.facility.densities)
                rate_indicator = policy.indicator(before_rates.values(), after_rates.values())
                if not math.isfinite(rate_indicator) or rate_indicator > 1:
                    raise HydraulicConvergenceError(
                        f"Facility rate-change indicator {rate_indicator:g} exceeds 1"
                    )
            material_account = self._material_account(inventory_before, plant, dt)
            energy = None
            if getattr(self, "energy_process", None) is not None:
                energy = self.energy_process.run(
                    solver.wells.surface_report["wells"],
                    dt,
                    None if previous is None else previous.get("energy_process"),
                    controls=getattr(self, "energy_controls", {}),
                )
            self.report = {
                "model": "outer_rate_cap_coupled_blackoil_fim_facility",
                "fraction": chosen,
                "dt_s": dt,
                "evaluations": len(trace),
                "reservoir_evaluations": trial_calls,
                "certified_evaluations": len(trace) - trial_calls,
                "trace": trace,
                "search_method": self.control.search_method,
                "capacity_bracket": final_bracket,
                "operator_controls": deepcopy(self.controls),
                "facility": plant,
                "material_account": material_account,
                "applied_rate_caps_m3_day": [cap * chosen for cap in caps],
                "requested_rate_caps_m3_day": list(caps),
                "scope": (
                    "Common producer rate-cap multiplier; converged infeasible states may certify "
                    "other inactive caps within the same FIM step. No monolithic facility Jacobian."
                    if self.control.reuse_inactive_caps else
                    "Common producer rate-cap multiplier; each trial solves the same FIM step. No monolithic facility Jacobian."
                ),
            }
            if capacity_commit is not None:
                self.report["capacity_commit"] = capacity_commit
            if self.control.search_method == 'sensitivity_safeguarded':
                self.report['sensitivity_work'] = tangent_work
            if initial_cells is not None:
                self.report['initial_cell_state_cache'] = initial_cells.report()
            if amg_scope is not None:
                self.report['capacity_amg_work'] = amg_scope.report()
            if policy is not None:
                self.report["rate_time_control"] = {
                    "indicator": rate_indicator,
                    "rate_relative_tolerance": policy.rate_relative_tolerance,
                    "rate_absolute_tolerance_kg_s": policy.rate_absolute_tolerance_kg_s,
                    "scope": "Endpoint-rate variation heuristic; requires independent time refinement",
                }
            if energy is not None:
                self.report["energy_process"] = energy
            self.last_attempt = {"success": True, "evaluations": len(trace),
                                 "reservoir_evaluations": trial_calls, "sensitivity_work": dict(tangent_work)}
            accepted = True
            return True
        except HydraulicConvergenceError as error:
            self.last_attempt = {"success": False, "reason": str(error), "trace": trace,
                                 "evaluations": sample_calls, "reservoir_evaluations": trial_calls,
                                 "sensitivity_work": dict(tangent_work)}
            return False
        except Exception as error:
            self.last_attempt = {"success": False, "reason": str(error), "trace": trace,
                                 "evaluations": sample_calls, "reservoir_evaluations": trial_calls,
                                 "sensitivity_work": dict(tangent_work)}
            raise
        finally:
            if amg_scope is not None and self.last_attempt is not None:
                self.last_attempt['capacity_amg_work'] = amg_scope.report()
            if not accepted:
                reset()
                self.report = previous
            # Preserve operator requests; applied caps are recorded separately.
            for w, cap in zip(solver.wells.wells, caps):
                w.control_value = cap
