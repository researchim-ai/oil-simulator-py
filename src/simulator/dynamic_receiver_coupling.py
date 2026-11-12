"""Two-way END-pressure coupling of isothermal FIM and a finite PR receiver.

Each outer trial repeats the same reservoir and vessel interval. The accepted
receiver has finite component/energy storage, finite apertures and a thermal
wall. Whole-system step doubling controls splitting/time error. This is a
producer-only partitioned solve, not a thermal reservoir or PR pipe momentum
model. Stock-to-species characterization and well thermal boundaries are
explicit physical inputs.
"""

import math
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass

import torch

from .blackoil_surface_coupling import SurfaceCoupledBlackOilWells
from .fim_transaction import _StepState
from .pr_surface_boundary import make_surface_boundary
from .thermal_surface_boundary import characterize_well_thermal_boundaries

try:
    from ..surface.component_energy import StockComponentAssay
    from ..surface.equilibrium_cache import cached_flashes
    from ..surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from ..surface.operation import _digest
    from ..surface.stratified_pr_flow import (
        ReceiverFeedPressureError,
        StratifiedPRFlow,
    )
    from ..surface.stratified_pr_thermal import StratifiedThermalState
    from ..surface.time_interval import split_step_s
except ImportError:
    from surface.component_energy import StockComponentAssay
    from surface.equilibrium_cache import cached_flashes
    from surface.liquid import ConservationError, HydraulicConvergenceError, PropertyModelError
    from surface.operation import _digest
    from surface.stratified_pr_flow import (
        ReceiverFeedPressureError,
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
        *,
        inlet_vessel=None,
    ):
        if not isinstance(solver.wells, SurfaceCoupledBlackOilWells):
            raise PropertyModelError("Dynamic receiver requires physical black-oil gathering FIM")
        self.inlet_vessel = inlet_vessel
        boundary = make_surface_boundary(receiver, inlet_vessel)
        if not isinstance(assay, StockComponentAssay):
            raise TypeError("Explicit StockComponentAssay required")
        f = boundary.fluid
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
            or not (0 <= self.inlet_height_m <= boundary.maximum_inlet_height_m)
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
        self.connection_openings = {}
        self.report, self.last_attempt = None, None
        self._validate_binding()
        self.model_id = _digest(self.to_dict())

    @property
    def boundary(self):
        return make_surface_boundary(self.receiver, getattr(self, "inlet_vessel", None))

    def _validate_binding(self):
        s, c = self.solver, self.boundary
        net = s.wells.surface_network
        if not math.isclose(
            c.inlet_elevation(self.inlet_height_m),
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
        result = {
            "model": self.boundary.model_tag,
            "receiver": self.receiver.to_dict(),
            "assay": asdict(self.assay),
            "thermal_conditions": deepcopy(self.thermal_conditions),
            "inlet_height_m": self.inlet_height_m,
            "control": asdict(self.control),
        }
        if self.boundary.network:
            result["inlet_vessel"] = self.inlet_vessel
        return result

    @classmethod
    def from_dict(cls, solver, raw, *, device="cpu"):
        capillary = raw.get("model") == "dynamic_pr_capillary_surface_fim_v1"
        resolved = capillary or raw.get("model") == "dynamic_pr_gravity_transport_fim_v1"
        distributed = resolved or raw.get("model") == "dynamic_pr_transport_fim_v1"
        graph = distributed or raw.get("model") == "dynamic_pr_vessel_network_fim_v1"
        keys = {
            "model",
            "receiver",
            "initial_state",
            "assay",
            "thermal_conditions",
            "inlet_height_m",
            "control",
        }
        if graph:
            keys.add("inlet_vessel")
        if set(raw) != keys or raw["model"] not in (
            "dynamic_pr_receiver_fim_v1",
            "dynamic_pr_vessel_network_fim_v1",
            "dynamic_pr_transport_fim_v1",
            "dynamic_pr_gravity_transport_fim_v1",
            "dynamic_pr_capillary_surface_fim_v1",
        ):
            raise ValueError("Explicit dynamic receiver specification required")
        if capillary:
            try:
                from ..surface.pr_capillary_surface_plant import (
                    PRCapillarySurfacePlant,
                    PRCapillarySurfaceState,
                )
            except ImportError:
                from surface.pr_capillary_surface_plant import (
                    PRCapillarySurfacePlant,
                    PRCapillarySurfaceState,
                )
            model = PRCapillarySurfacePlant.from_dict(raw["receiver"], device=device)
            state = PRCapillarySurfaceState.from_dict(raw["initial_state"])
        elif resolved:
            try:
                from ..surface.pr_gravity_transport import PRGravityTransportState
                from ..surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
            except ImportError:
                from surface.pr_gravity_transport import PRGravityTransportState
                from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
            model = PROpenGravityTransportNetwork.from_dict(raw["receiver"], device=device)
            state = PRGravityTransportState.from_dict(raw["initial_state"])
        elif distributed:
            try:
                from ..surface.pr_equilibrium_transport import PREquilibriumTransportState
                from ..surface.pr_open_transport import PROpenTransportNetwork
            except ImportError:
                from surface.pr_equilibrium_transport import PREquilibriumTransportState
                from surface.pr_open_transport import PROpenTransportNetwork
            model = PROpenTransportNetwork.from_dict(raw["receiver"], device=device)
            state = PREquilibriumTransportState.from_dict(raw["initial_state"])
        elif graph:
            try:
                from ..surface.stratified_pr_network import (
                    StratifiedNetworkState,
                    StratifiedPRNetwork,
                )
            except ImportError:
                from surface.stratified_pr_network import (
                    StratifiedNetworkState,
                    StratifiedPRNetwork,
                )
            model = StratifiedPRNetwork.from_dict(raw["receiver"], device=device)
            state = StratifiedNetworkState.from_dict(raw["initial_state"])
        else:
            model = StratifiedPRFlow.from_dict(raw["receiver"], device=device)
            state = StratifiedThermalState.from_dict(raw["initial_state"])
        return cls(
            solver,
            model,
            state,
            StockComponentAssay(**raw["assay"]),
            raw["thermal_conditions"],
            raw["inlet_height_m"],
            raw["control"],
            inlet_vessel=raw.get("inlet_vessel"),
        )

    def _pressure(self, state):
        return self.boundary.pressure(state, self.inlet_height_m)

    def _inventory(self):
        s = self.solver
        with torch.no_grad():
            v = s.masses(s.fluid.pressure.flatten(), s.fluid.s_w.flatten(), s.x, s.sat, s.ug)
            return v.sum(1).detach().cpu().tolist()

    def _flow(self, streams):
        return self.boundary.build(
            streams,
            self.inlet_height_m,
            self.outlet_openings,
            getattr(self, "connection_openings", {}),
        )

    def _backward_euler(self, old, dt, current_time, *, pressure_guess_pa=None):
        s, net = self.solver, self.solver.wells.surface_network
        initial = _StepState(s)
        before = self._inventory()
        trace, accepted = [], False
        point, lower, upper = self._pressure(old), None, None
        domain = net.model.pressure_bounds_pa
        fluid = self.boundary.fluid
        lo, hi = max(domain[0], fluid.min_pressure_pa), min(domain[1], fluid.max_pressure_pa)
        if (
            pressure_guess_pa is not None
            and math.isfinite(pressure_guess_pa)
            and lo <= pressure_guess_pa <= hi
        ):
            point = pressure_guess_pa

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
                        elevation_m=self.boundary.inlet_elevation(self.inlet_height_m),
                        velocity_m_s=0.0,
                        temperature_guess_k=self.boundary.temperature(old),
                    )
                    for stream in boundary["streams"].values()
                ]
                flow = self._flow(streams)
                trial = self.boundary.rebind(old, flow)
                try:
                    candidate, report = self.boundary.step(
                        flow, trial, dt, getattr(self, "connection_openings", {})
                    )
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
                    end = self.boundary.rebind(candidate, self.receiver)
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
        error = self.boundary.temporal_error(coarse, fine, c, self.inlet_height_m)
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
        return error

    def _account(self, old, end, stages):
        flows = [s["flow"] for s in stages]
        old_n, old_energy = self.boundary.totals(old)
        end_n, end_energy = self.boundary.totals(end)
        n = len(end_n)
        incoming = [math.fsum(f["feed_component_moles"][j] for f in flows) for j in range(n)]
        outgoing = [math.fsum(f["outlet_component_moles"][j] for f in flows) for j in range(n)]
        energy_in = math.fsum(f["feed_energy_j"] for f in flows)
        energy_out = math.fsum(f["outlet_energy_j"] for f in flows)
        ambient = math.fsum(f["ambient_heat_j"] for f in flows)
        external_energy = math.fsum(f.get("external_energy_j", 0.0) for f in flows)
        balance = [b - a - i + o for a, b, i, o in zip(old_n, end_n, incoming, outgoing)]
        energy_balance = math.fsum(
            (end_energy, -old_energy, -energy_in, energy_out, -ambient, -external_energy)
        )
        if max(map(abs, balance)) > 1e-8 + 1e-9 * math.fsum(end_n):
            raise ConservationError("Coupled receiver component ledger rejected")
        if abs(energy_balance) > 1e-5 + 1e-9 * max(
            1, abs(energy_in), abs(energy_out), abs(ambient), abs(external_energy)
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
        if self.boundary.model_tag in (
            "dynamic_pr_transport_fim_v1",
            "dynamic_pr_gravity_transport_fim_v1",
            "dynamic_pr_capillary_surface_fim_v1",
        ):
            account["surface_external_energy_j"] = math.fsum(
                (prev.get("surface_external_energy_j", 0.0), external_energy)
            )
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
        # Scratch lives for this complete coarse/two-half transaction only.
        # Exact keys include model tensor versions, so no old phase region or
        # rejected inventory can become a stale physical boundary.
        with ExitStack() as stack:
            inspections = stack.enter_context(self.boundary.transaction())
            caches = [stack.enter_context(cached_flashes(f)) for f in self.boundary.fluids()]
            accepted = self._step_impl(dt, current_time)
            if accepted:
                if inspections is not None:
                    self.report["boundary_inspections"] = inspections.statistics()
                stats = [cache.statistics() for cache in caches]
                if len(stats) == 1:
                    self.report["property_evaluations"] = stats[0]
                else:
                    keys = set().union(*(s.keys() for s in stats)) - {"enabled"}
                    total = {k: sum(s.get(k, 0) for s in stats) for k in sorted(keys)}
                    total["enabled"] = all(s["enabled"] for s in stats)
                    total["fluid_caches"] = [{"index": i, **s} for i, s in enumerate(stats)]
                    self.report["property_evaluations"] = total
            return accepted

    def limit_next_time_step_s(self, proposed_s, *, clipped=False):
        """Bound a reservoir proposal by the accepted whole-system time error.

        Full versus two-half backward Euler has an O(dt**2) local defect.
        This is a predictor only: every proposed step is solved and checked
        again. A small low-error report tail must not erase a useful step.
        """
        if isinstance(proposed_s, bool) or not math.isfinite(proposed_s) or proposed_s <= 0:
            raise ValueError("Positive finite next coupled time step required")
        if type(clipped) is not bool:
            raise ValueError("Explicit reporting-tail switch required")
        if self.report is None:
            return proposed_s
        error, dt = self.report["temporal_error_ratio"], self.report["dt_s"]
        if not math.isfinite(error) or not 0 <= error <= 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid accepted coupled temporal estimate")
        if error == 0 or (clipped and error <= 0.25):
            return proposed_s
        return min(proposed_s, dt * 0.9 / math.sqrt(error))

    def _step_impl(self, dt, current_time=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Coupled receiver step must be finite and positive")
        self._validate_binding()
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Dynamic receiver model changed; explicit restart required")
        self.receiver.inspect(self.receiver_state)
        if current_time is not None and current_time != self.receiver_state.time_s:
            raise ValueError("Reservoir and receiver clocks differ")
        if set(self.outlet_openings) - self.boundary.outlet_names():
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
            # Use the already solved coarse endpoint as a pressure PREDICTOR
            # for the half steps; their states, flows and error are still
            # solved and checked independently from the same old reservoir.
            p0, pc = self._pressure(old), self._pressure(coarse)
            middle, first = self._backward_euler(
                old, a, old.time_s, pressure_guess_pa=p0 + (pc - p0) * (a / dt)
            )
            end, second = self._backward_euler(middle, b, middle.time_s, pressure_guess_pa=pc)
            error = self._temporal_error(coarse, end, coarse_reservoir)
            if not math.isfinite(error) or error > 1:
                raise HydraulicConvergenceError(
                    f"Whole-system temporal error above tolerance: ratio={error:.8g}"
                )
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
                "model": self.to_dict()["model"],
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
            if self.boundary.network:
                self.report["connection_openings"] = dict(self.connection_openings)
                self.report["scope"] = (
                    "Producer-only two-way pressure feedback into a finite gravitational PR vessel graph with direct massless connectors and finite outlets. Steady PVT gathering; explicit thermal boundary; no distributed pipe storage, pipe momentum, injection or reservoir energy equation."
                )
            if self.boundary.model_tag in (
                "dynamic_pr_transport_fim_v1",
                "dynamic_pr_gravity_transport_fim_v1",
                "dynamic_pr_capillary_surface_fim_v1",
            ):
                self.report["scope"] = self.boundary.scope
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
