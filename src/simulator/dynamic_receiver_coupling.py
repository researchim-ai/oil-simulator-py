"""Two-way END-pressure coupling of isothermal FIM and a finite PR receiver.

Each outer trial repeats the same reservoir and vessel interval. The accepted
receiver has finite component/energy storage, finite apertures and a thermal
wall. Whole-system step doubling controls splitting/time error. This is a
producer-only partitioned solve, not a thermal reservoir or PR pipe momentum
model. Stock-to-species characterization and well thermal boundaries are
explicit physical inputs.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import torch

from .blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from .fim_transaction import _StepState
from .thermal_surface_boundary import characterize_well_thermal_boundaries

try:
    from ..surface.component_energy import StockComponentAssay
    from ..surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from ..surface.operation import _digest
    from ..surface.stratified_pr_flow import (
        ReceiverFeedPressureError,
        StratifiedFeed,
        StratifiedPRFlow,
    )
    from ..surface.stratified_pr_thermal import StratifiedThermalState
    from ..surface.time_interval import split_step_s
except ImportError:
    from surface.component_energy import StockComponentAssay
    from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from surface.operation import _digest
    from surface.stratified_pr_flow import (
        ReceiverFeedPressureError,
        StratifiedFeed,
        StratifiedPRFlow,
    )
    from surface.stratified_pr_thermal import StratifiedThermalState
    from surface.time_interval import split_step_s


@dataclass(frozen=True)
class DynamicReceiverControl:
    maximum_pressure_iterations: int = 24
    pressure_tolerance_pa: float = 0.01
    pressure_relative_tolerance: float = 1e-9
    temporal_pressure_tolerance_pa: float = 1.0
    temporal_temperature_tolerance_k: float = 0.002
    temporal_relative_tolerance: float = 1e-5
    temporal_reservoir_pressure_tolerance_pa: float = 10.0
    temporal_saturation_tolerance: float = 1e-5

    def __post_init__(self):
        if (
            type(self.maximum_pressure_iterations) is not int
            or self.maximum_pressure_iterations < 2
        ):
            raise ValueError("At least two pressure iterations required")
        for key, value in asdict(self).items():
            if key != "maximum_pressure_iterations" and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")


class DynamicReceiverCoupledFIM:
    def __init__(
        self,
        solver,
        receiver,
        receiver_state,
        assay,
        thermal_conditions,
        inlet_height_m,
        control=None,
    ):
        if not isinstance(solver.wells, SurfaceCoupledBlackOilWells):
            raise PropertyModelError("Dynamic receiver requires physical black-oil gathering FIM")
        if not isinstance(receiver, StratifiedPRFlow) or receiver.feeds:
            raise TypeError("An unforced StratifiedPRFlow receiver is required")
        if not isinstance(assay, StockComponentAssay):
            raise TypeError("Explicit StockComponentAssay required")
        f = receiver.column.fluid
        if assay.fluid_id != f.model_id or assay.component_names != f.names:
            raise PropertyModelError("Receiver EOS and well assay differ")
        receiver.inspect(receiver_state)
        if str(f.device) != str(solver.device):
            raise PropertyModelError("Reservoir and PR receiver must use the same device")
        self.solver, self.receiver, self.receiver_state = solver, receiver, receiver_state
        self.assay = assay
        self.thermal_conditions = deepcopy(thermal_conditions)
        self.inlet_height_m = float(inlet_height_m)
        if (
            isinstance(inlet_height_m, bool)
            or not math.isfinite(self.inlet_height_m)
            or not (0 <= self.inlet_height_m <= receiver.column.geometry.height_m)
        ):
            raise ValueError("Receiver inlet height outside vessel")
        self.control = (
            DynamicReceiverControl(**control)
            if isinstance(control, dict)
            else (DynamicReceiverControl() if control is None else control)
        )
        if not isinstance(self.control, DynamicReceiverControl):
            raise TypeError("Expected DynamicReceiverControl")
        self.outlet_openings = {}
        self.report, self.last_attempt = None, None
        self._validate_binding()
        self.model_id = _digest(self.to_dict())

    def _validate_binding(self):
        s, c = self.solver, self.receiver.column
        net = s.wells.surface_network
        if not math.isclose(
            c.elevation_m + self.inlet_height_m,
            net.by_name[net.root].elevation_m,
            abs_tol=1e-9,
            rel_tol=0,
        ):
            raise PropertyModelError("Receiver inlet must match gathering receiver elevation")
        if c.gravity_m_s2 != 9.80665:
            raise PropertyModelError("Well energy boundary and receiver gravity differ")
        if set(self.thermal_conditions) != {w.name for w in s.wells.wells}:
            raise ValueError("Explicit thermal conditions required for every producer")
        for w, node in zip(s.wells.wells, s.wells.surface_nodes):
            if (
                w.type != "producer"
                or w.control_type != "rate"
                or not math.isfinite(w.control_value)
                or w.control_value <= 0
                or getattr(w, "stopped", False)
                or getattr(w, "ramping_days", 0) > 0
            ):
                raise PropertyModelError("Dynamic receiver requires positive producer rate caps")
            if not math.isclose(
                self.thermal_conditions[w.name].get("elevation_m", math.inf),
                net.by_name[node].elevation_m,
                abs_tol=1e-9,
                rel_tol=0,
            ):
                raise PropertyModelError(
                    "Well thermal boundary must use the physical BHP elevation"
                )

    def to_dict(self):
        return {
            "model": "dynamic_pr_receiver_fim_v1",
            "receiver": self.receiver.to_dict(),
            "assay": asdict(self.assay),
            "thermal_conditions": deepcopy(self.thermal_conditions),
            "inlet_height_m": self.inlet_height_m,
            "control": asdict(self.control),
        }

    @classmethod
    def from_dict(cls, solver, raw, *, device="cpu"):
        if (
            set(raw)
            != {
                "model",
                "receiver",
                "initial_state",
                "assay",
                "thermal_conditions",
                "inlet_height_m",
                "control",
            }
            or raw["model"] != "dynamic_pr_receiver_fim_v1"
        ):
            raise ValueError("Explicit dynamic receiver specification required")
        return cls(
            solver,
            StratifiedPRFlow.from_dict(raw["receiver"], device=device),
            StratifiedThermalState.from_dict(raw["initial_state"]),
            StockComponentAssay(**raw["assay"]),
            raw["thermal_conditions"],
            raw["inlet_height_m"],
            raw["control"],
        )

    def _pressure(self, state):
        return self.receiver.column.sample(state.column, self.inlet_height_m, side="below")[
            "pressure_pa"
        ]

    def _inventory(self):
        s = self.solver
        with torch.no_grad():
            v = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug)
            return v.sum(1).detach().cpu().tolist()

    def _flow(self, streams):
        base = self.receiver
        keys = (
            "initial_step_s",
            "minimum_step_s",
            "maximum_step_s",
            "temperature_tolerance_k",
            "relative_tolerance",
            "maximum_attempts",
        )
        return StratifiedPRFlow(
            base.column,
            base.wall,
            outlets=[
                replace(p, opening=self.outlet_openings.get(p.name, p.opening))
                for p in base.outlets
            ],
            feeds=[StratifiedFeed(self.inlet_height_m, stream) for stream in streams],
            **{key: getattr(base, key) for key in keys},
        )

    def _backward_euler(self, old, dt, current_time):
        s, net = self.solver, self.solver.wells.surface_network
        initial = _StepState(s)
        before = self._inventory()
        trace, accepted = [], False
        point, lower, upper = self._pressure(old), None, None
        domain = net.model.pressure_bounds_pa
        fluid = self.receiver.column.fluid
        lo, hi = max(domain[0], fluid.min_pressure_pa), min(domain[1], fluid.max_pressure_pa)

        def reset():
            nonlocal initial
            initial.restore()
            initial = _StepState(s)
            s.wells.surface_network = net

        try:
            for iteration in range(self.control.maximum_pressure_iterations):
                reset()
                if not math.isfinite(point) or not lo <= point <= hi:
                    raise HydraulicConvergenceError(
                        "Coupled receiver pressure outside common domain"
                    )
                s.wells.set_surface_boundary_pressures({net.root: point})
                if not s.step(dt, current_time=current_time):
                    raise HydraulicConvergenceError(
                        "Reservoir failed during dynamic receiver trial"
                    )
                boundary = characterize_well_thermal_boundaries(
                    s.wells.surface_report["wells"], self.thermal_conditions, fluid, self.assay
                )
                streams = [
                    stream.transport(
                        fluid,
                        point,
                        elevation_m=self.receiver.column.elevation_m + self.inlet_height_m,
                        velocity_m_s=0.0,
                        temperature_guess_k=old.column.temperature_k,
                    )
                    for stream in boundary["streams"].values()
                ]
                flow = self._flow(streams)
                trial = replace(old, model_id=flow.model_id)
                try:
                    candidate, report = flow._step(trial, dt)
                    pressure = self._pressure(candidate)
                    physical = True
                except ReceiverFeedPressureError as exc:
                    pressure, physical = exc.receiver_pressure_pa, False
                defect = pressure - point
                tolerance = (
                    self.control.pressure_tolerance_pa
                    + self.control.pressure_relative_tolerance * max(point, pressure)
                )
                trace.append(
                    {
                        "pressure_pa": point,
                        "receiver_pressure_pa": pressure,
                        "pressure_defect_pa": defect,
                        "physical_trial": physical,
                    }
                )
                if physical and abs(defect) <= tolerance:
                    # The temporary model binds actual accepted feeds/commands;
                    # the persistent checkpoint uses the unforced receiver ID.
                    end = replace(candidate, model_id=self.receiver.model_id)
                    self.receiver.inspect(end)
                    accepted = True
                    return end, {
                        "dt_s": dt,
                        "accepted_newton_iterations": s.last_newton_iters,
                        "pressure_trials": trace,
                        "flow": report,
                        "well_boundary": boundary["boundaries"],
                        "accepted_wells": deepcopy(s.wells.surface_report["wells"]),
                        "inlet_well_names": list(boundary["streams"]),
                        "inlet_streams": [stream.to_dict() for stream in streams],
                        "reservoir_inventory_before_kg": before,
                        "reservoir_inventory_after_kg": self._inventory(),
                        "actual_stock_component_mass_kg": [
                            dt
                            * math.fsum(
                                w["component_mass_rates_kg_s"][j]
                                for w in s.wells.surface_report["wells"].values()
                            )
                            for j in range(3)
                        ],
                    }
                if defect > 0:
                    lower = (point, defect)
                else:
                    upper = (point, defect)
                if lower is not None and upper is not None:
                    a, fa = lower
                    b, fb = upper
                    proposal = (a + b) / 2 if iteration % 4 == 3 else (a - fa * (b - a) / (fb - fa))
                    if not a < proposal < b:
                        proposal = (a + b) / 2
                    point = proposal
                else:
                    point = min(hi, max(lo, point + defect + 0.25 * tolerance))
            raise HydraulicConvergenceError("Dynamic receiver pressure iteration budget exhausted")
        finally:
            if not accepted:
                reset()

    def _temporal_error(self, coarse, fine, coarse_reservoir):
        c = self.control
        nc, nf = coarse.column.component_moles, fine.column.component_moles
        scale = max(1.0, math.fsum(nf))
        pressure_c, pressure_f = self._pressure(coarse), self._pressure(fine)
        error = max(
            abs(pressure_c - pressure_f)
            / (c.temporal_pressure_tolerance_pa + c.temporal_relative_tolerance * pressure_f),
            abs(coarse.column.temperature_k - fine.column.temperature_k)
            / c.temporal_temperature_tolerance_k,
            abs(coarse.wall_temperature_k - fine.wall_temperature_k)
            / c.temporal_temperature_tolerance_k,
            max(abs(a - b) for a, b in zip(nc, nf))
            / (1e-9 + c.temporal_relative_tolerance * scale),
            abs(coarse.column.total_energy_j - fine.column.total_energy_j)
            / (
                1e-5
                + c.temporal_relative_tolerance
                * max(
                    1.0,
                    abs(fine.column.total_energy_j),
                    scale * 8.314462618 * fine.column.temperature_k,
                )
            ),
        )
        f = self.solver.fluid
        terms = [
            (
                (f.pressure - coarse_reservoir["pressure"]).abs()
                / (
                    c.temporal_reservoir_pressure_tolerance_pa
                    + c.temporal_relative_tolerance * f.pressure.abs()
                )
            ).max()
        ]
        for name in ("s_w", "s_g"):
            previous = coarse_reservoir[name]
            terms.append(
                ((getattr(f, name) - previous).abs() / c.temporal_saturation_tolerance).max()
            )
        for name in ("rs", "rv"):
            if name in coarse_reservoir:
                value = getattr(f, name)
                terms.append(
                    (
                        (value - coarse_reservoir[name]).abs()
                        / (1e-8 + c.temporal_relative_tolerance * value.abs().clamp_min(1.0))
                    ).max()
                )
        error = max(error, float(torch.stack(terms).max().detach().cpu()))
        if len(coarse.column.phase_roots) != len(fine.column.phase_roots):
            error = max(error, 2.0)
        else:
            for a, b in zip(coarse.column.interface_heights_m, fine.column.interface_heights_m):
                error = max(
                    error,
                    abs(a - b)
                    / (
                        1e-8
                        + c.temporal_relative_tolerance * self.receiver.column.geometry.height_m
                    ),
                )
        return error

    def _account(self, old, end, stages):
        flows = [s["flow"] for s in stages]
        n = len(end.column.component_moles)
        incoming = [math.fsum(f["feed_component_moles"][j] for f in flows) for j in range(n)]
        outgoing = [math.fsum(f["outlet_component_moles"][j] for f in flows) for j in range(n)]
        energy_in = math.fsum(f["feed_energy_j"] for f in flows)
        energy_out = math.fsum(f["outlet_energy_j"] for f in flows)
        ambient = math.fsum(f["ambient_heat_j"] for f in flows)
        balance = [
            b - a - i + o
            for a, b, i, o in zip(
                old.column.component_moles, end.column.component_moles, incoming, outgoing
            )
        ]
        energy_balance = math.fsum(
            (
                end.column.total_energy_j,
                -old.column.total_energy_j,
                self.receiver.wall.heat_capacity_j_k
                * (end.wall_temperature_k - old.wall_temperature_k),
                -energy_in,
                energy_out,
                -ambient,
            )
        )
        if max(map(abs, balance)) > 1e-8 + 1e-9 * math.fsum(end.column.component_moles):
            raise ConservationError("Coupled receiver component ledger rejected")
        if abs(energy_balance) > 1e-5 + 1e-9 * max(
            1, abs(energy_in), abs(energy_out), abs(ambient)
        ):
            raise ConservationError("Coupled receiver energy ledger rejected")
        prev = {} if self.report is None else self.report["cumulative"]
        account = {}
        for key, values in (
            ("feed_component_moles", incoming),
            ("export_component_moles", outgoing),
        ):
            account[key] = [math.fsum((a, b)) for a, b in zip(prev.get(key, [0.0] * n), values)]
        for key, value in (
            ("thermal_boundary_energy_j", energy_in),
            ("export_energy_j", energy_out),
            ("ambient_heat_j", ambient),
        ):
            account[key] = math.fsum((prev.get(key, 0.0), value))
        stock = [
            math.fsum(s["actual_stock_component_mass_kg"][j] for s in stages) for j in range(3)
        ]
        account["produced_stock_component_kg"] = [
            math.fsum((a, b))
            for a, b in zip(prev.get("produced_stock_component_kg", [0.0] * 3), stock)
        ]
        account["elapsed_s"] = end.time_s
        before, after = (
            stages[0]["reservoir_inventory_before_kg"],
            stages[-1]["reservoir_inventory_after_kg"],
        )
        return {
            "component_residuals_mol": balance,
            "energy_residual_j": energy_balance,
            "reservoir_residual_kg": [a - b - p for a, b, p in zip(before, after, stock)],
            "cumulative": account,
        }

    def step(self, dt, current_time=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Coupled receiver step must be finite and positive")
        self._validate_binding()
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Dynamic receiver model changed; explicit restart required")
        self.receiver.inspect(self.receiver_state)
        if current_time is not None and current_time != self.receiver_state.time_s:
            raise ValueError("Reservoir and receiver clocks differ")
        if set(self.outlet_openings) - {p.name for p in self.receiver.outlets}:
            raise ValueError("Unknown receiver outlet command")
        # Validate commands before starting any reservoir trial.
        self._flow(())
        old, initial = self.receiver_state, _StepState(self.solver)
        network, previous = self.solver.wells.surface_network, self.report
        accepted = False
        try:
            coarse, _ = self._backward_euler(old, dt, old.time_s)
            f = self.solver.fluid
            coarse_reservoir = {
                name: getattr(f, name).detach().clone()
                for name in ("pressure", "s_w", "s_g", "rs", "rv")
                if hasattr(f, name)
            }
            initial.restore()
            initial = _StepState(self.solver)
            self.solver.wells.surface_network = network
            a, b = split_step_s(old.time_s, dt)
            middle, first = self._backward_euler(old, a, old.time_s)
            end, second = self._backward_euler(middle, b, middle.time_s)
            error = self._temporal_error(coarse, end, coarse_reservoir)
            if not math.isfinite(error) or error > 1:
                raise HydraulicConvergenceError("Whole-system temporal error above tolerance")
            account = self._account(old, end, (first, second))
            # The next full-interval dt controller must see the full change,
            # not just the accepted second half-step.
            sw0, sg0 = initial.old["s_w"], initial.fluid_bindings["s_g"]
            self.solver.last_max_dsw = float(
                torch.stack(
                    (
                        (f.s_w - sw0).abs().max(),
                        (f.s_g - sg0).abs().max(),
                        ((f.s_w + f.s_g) - (sw0 + sg0)).abs().max(),
                    )
                )
                .max()
                .detach()
                .cpu()
            )
            self.solver.last_max_dp = float(
                (f.pressure - initial.old["pressure"]).abs().max().detach().cpu()
            )
            self.solver.last_newton_iters = max(
                first["accepted_newton_iterations"], second["accepted_newton_iterations"]
            )
            self.report = {
                "model": "dynamic_pr_receiver_fim_v1",
                "model_id": self.model_id,
                "state": end.to_dict(),
                "initial_state": old.to_dict(),
                "dt_s": dt,
                "temporal_error_ratio": error,
                "stages": [first, second],
                "outlet_openings": dict(self.outlet_openings),
                **account,
                "reservoir_energy_equation_solved": False,
                "pressure_feedback_from_caloric_process": True,
                "pvt_eos_characterization_calibrated": False,
                "scope": "Producer-only two-way finite-receiver pressure feedback. Steady PVT gathering, conservative adiabatic energy transfer, explicit thermal boundary; no pipe storage or reservoir energy equation.",
            }
            self.receiver_state = end
            self.last_attempt = {"success": True}
            accepted = True
            return True
        except HydraulicConvergenceError as exc:
            self.last_attempt = {"success": False, "reason": str(exc)}
            return False
        finally:
            if not accepted:
                initial.restore()
                self.solver.wells.surface_network = network
                self.receiver_state, self.report = old, previous
