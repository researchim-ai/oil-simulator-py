"""Exact reduced Newton coupling of pure-water FIM producers to passive pipes.

The network nodes are at well BHP reference elevations. Hydrostatics from these
nodes to the surface belong to the network, not a second well lift correction.
Rate/BHP limits act as ideal chokes; zero production closes the well interface.
All cross-well pressure derivatives enter B and D and the full Schur operator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from functools import wraps

import torch

from .fim_wells import DAY, FIMWells
from .full_well_schur import FullWellSchur

try:
    from ..surface.control import PumpPressureControl
    from ..surface.liquid import HydraulicConvergenceError, LiquidNetwork, PropertyModelError
except ImportError:
    from surface.control import PumpPressureControl
    from surface.liquid import HydraulicConvergenceError, LiquidNetwork, PropertyModelError


def surface_step_failure(function):
    """The inner atomic_fim_step restores physical state before this retry signal."""

    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if not getattr(self.wells, "physical_surface", False):
            return function(self, *args, **kwargs)
        self.last_surface_failure = None
        try:
            return function(self, *args, **kwargs)
        except HydraulicConvergenceError as error:
            self.last_surface_failure = str(error)
            self._p_bh_cur = None
            return False

    return wrapped


class SurfaceCoupledWaterWells(FullWellSchur, FIMWells):
    def __init__(self, well_manager, reservoir, fluid, dims, device, spec):
        super().__init__(well_manager, reservoir, fluid, dims, device)
        if set(spec) - {"network", "well_nodes", "controls", "pressure_control"}:
            raise ValueError("Unknown physical_surface_network fields")
        self.surface_network = LiquidNetwork.from_dict(spec["network"])
        self.surface_pressure_control = None
        self.surface_station_policy = None
        self.surface_station_base = None
        self.surface_station_running = ()
        self.set_surface_controls(**spec.get("controls", {}))
        self.set_surface_pressure_control(spec.get("pressure_control"))
        if self.surface_pressure_control is not None:
            controls = spec.get("controls", {})
            owned = {self.surface_pressure_control.pump, *self.surface_pressure_control.linked_pumps}
            if owned & (set(controls.get("pump_speed_ratios", {})) | set(controls.get("pump_status", {}))):
                raise ValueError("Pressure controller owns the regulated pump speed/status")
        network = self.surface_network
        if getattr(fluid, "pvt", None) is not None:
            raise PropertyModelError(
                "Physical FIM pipe coupling currently requires analytic oil_water properties, not black-oil PVT tables"
            )
        if any(n.source_mass_rate_kg_s != 0 for n in network.nodes):
            raise ValueError("FIM surface sources must come exclusively from bound wells")
        if not math.isclose(network.fluid.density_kg_m3, fluid.rho_water_ref, rel_tol=1e-10):
            raise ValueError("FIM surface water density must match reference water density")
        if not math.isclose(network.fluid.viscosity_pa_s, fluid.mu_water, rel_tol=1e-10):
            raise ValueError("FIM surface viscosity must match water viscosity")
        nodes = {n.name: n for n in network.nodes}
        names = {w.name: i for i, w in enumerate(self.wells)}
        bindings = spec["well_nodes"]
        if not isinstance(bindings, dict) or not bindings or set(bindings) - set(names):
            raise ValueError("Unknown or empty FIM well bindings")
        self.surface_well_indices = tuple(names[name] for name in bindings)
        self.surface_nodes = tuple(bindings[name] for name in bindings)
        if self.n_w > 128 or self.conn_cell.numel() > 10000:
            raise ValueError(
                "Dense FIM surface coupling currently supports <=128 wells and <=10000 connections"
            )
        for i, node in zip(self.surface_well_indices, self.surface_nodes):
            well = self.wells[i]
            if (
                well.type != "producer"
                or well.control_type != "rate"
                or (well.surface_phase or "liquid").lower() not in ("water", "liquid")
            ):
                raise ValueError(
                    "Physical FIM surface binding requires a water/liquid rate-controlled producer"
                )
            if node not in nodes or nodes[node].pressure_pa is not None:
                raise ValueError("Bound well must connect to a free-pressure network node")
            if not math.isclose(nodes[node].elevation_m, -self.ref_depth[i], abs_tol=1e-6):
                raise ValueError(
                    "Surface node elevation must equal negative well BHP reference depth"
                )
            if (
                getattr(well, "vfp_table", None) is not None
                or getattr(well, "msw", None) is not None
            ):
                raise ValueError("Physical pipes cannot be combined with VFP/MSW on the same well")
        self.physical_surface = True
        self.surface_report = None

    @staticmethod
    def _validate_surface_configuration(network, control):
        for component in set(network._components):
            if not any(n.pressure_pa is not None for i, n in enumerate(network.nodes)
                       if network._components[i] == component):
                raise ValueError("FIM surface equipment configuration disconnects a pressure boundary")
        if control is not None:
            pumps = {p.name: p for p in network.pumps}
            nodes = {n.name: n for n in network.nodes}
            group = (control.pump, *control.linked_pumps)
            if any(name not in pumps or not pumps[name].open for name in group):
                raise ValueError("Pressure control requires a known open pump")
            if control.node not in nodes or nodes[control.node].pressure_pa is not None:
                raise ValueError("Pressure control requires a known free-pressure node")

    def set_surface_controls(self, pump_speed_ratios=None, pump_status=None, valve_openings=None):
        """Atomically change equipment setpoints between reservoir steps."""
        control = self.surface_pressure_control
        if self.surface_station_policy is not None and set(self.surface_station_policy.pumps) & (
                set(pump_speed_ratios or {}) | set(pump_status or {})):
            raise ValueError("Station dispatch owns pump speed/status; clear or redispatch it")
        if control is not None and {control.pump, *control.linked_pumps} & (
                set(pump_speed_ratios or {}) | set(pump_status or {})):
            raise ValueError("Pressure controller owns the regulated pump speed/status")
        network = self.surface_network.with_controls(pump_speed_ratios, pump_status, valve_openings)
        self._validate_surface_configuration(network, control)
        if self.surface_station_base is not None:
            self.surface_station_base = self.surface_station_base.with_controls(valve_openings=valve_openings)
        self.surface_network = network

    def set_surface_pressure_control(self, control):
        """Set/remove a bounded pressure controller; settings are immutable in Newton."""
        if self.surface_station_policy is not None:
            raise ValueError("Clear or redispatch the station before changing its pressure controller")
        if isinstance(control, dict):
            control = PumpPressureControl(**{
                "method": "newton", "response_direction": "auto",
                "pressure_tolerance_pa": 1e-4, **control})
        if control is not None and not isinstance(control, PumpPressureControl):
            raise TypeError("Expected PumpPressureControl, dictionary or None")
        self._validate_surface_configuration(self.surface_network, control)
        self.surface_pressure_control = control

    @property
    def station_network(self):
        """Original equipment availability for the next explicit duty dispatch."""
        return self.surface_station_base or self.surface_network

    def set_surface_station(self, selection):
        """Apply a safe station recipe between steps; recheck its limits in Newton."""
        try:
            from ..surface.station import PumpStationResult, station_network_fingerprint
        except ImportError:
            from surface.station import PumpStationResult, station_network_fingerprint
        if selection is None:
            if self.surface_station_base is not None:
                self.surface_network = self.surface_station_base
            self.surface_station_policy, self.surface_station_base = None, None
            self.surface_station_running = ()
            self.surface_pressure_control = None
            return
        if not isinstance(selection, PumpStationResult):
            raise TypeError("Expected PumpStationResult or None")
        base = self.station_network
        if station_network_fingerprint(base) != selection.model_fingerprint:
            raise ValueError("Station recipe does not match this network configuration")
        group = (selection.control.pump, *selection.control.linked_pumps)
        if set(group) != set(selection.running_pumps) or selection.regulation.status != "target":
            raise ValueError("Station recipe must name the regulated running pumps")
        network = base.with_controls(**selection.network.operating_controls)
        control = replace(selection.control,
                          pressure_tolerance_pa=min(selection.control.pressure_tolerance_pa, 1e-4))
        self._validate_surface_configuration(network, control)
        selection.policy._audit(selection.network, selection.running_pumps)
        self.surface_network, self.surface_pressure_control = network, control
        self.surface_station_base = base
        self.surface_station_policy = selection.policy
        self.surface_station_running = selection.running_pumps

    def set_surface_boundary_pressures(self, pressures_pa):
        """Change existing outlet pressure setpoints without altering graph topology."""
        boundaries = {
            node.name for node in self.surface_network.nodes if node.pressure_pa is not None
        }
        if set(pressures_pa) - boundaries:
            raise ValueError("Only existing surface pressure boundaries may be changed")
        nodes = [
            replace(node, pressure_pa=pressures_pa.get(node.name, node.pressure_pa))
            for node in self.surface_network.nodes
        ]
        if any(node.pressure_pa is None for node in nodes if node.name in boundaries):
            raise ValueError("Cannot release a surface pressure boundary during a FIM run")
        if any(
            node.pressure_pa is not None
            and node.pressure_pa <= self.surface_network.fluid.vapor_pressure_pa
            for node in nodes
        ):
            raise ValueError("Surface boundary must exceed vapor pressure")
        network = LiquidNetwork(self.surface_network.fluid, nodes, self.surface_network.pipes,
                                self.surface_network.pumps, self.surface_network.valves)
        self._validate_surface_configuration(network, self.surface_pressure_control)
        if self.surface_station_base is not None:
            base = self.surface_station_base
            self.surface_station_base = LiquidNetwork(base.fluid, nodes, base.pipes, base.pumps, base.valves)
        self.surface_network = network

    def update_controls(self, p_bh, q_surf):
        original = self.step_targets
        self.step_targets = [
            ("physical_surface", v) if i in self.surface_well_indices else (m, v)
            for i, (m, v) in enumerate(original)
        ]
        try:
            return super().update_controls(p_bh, q_surf)
        finally:
            self.step_targets = original

    def assemble(self, R, J_vals, diag_map, p, sw, props, p_bh):
        sys = super().assemble(R, J_vals, diag_map, p, sw, props, p_bh)
        c, cw = self.conn_cell, self.conn_well
        nconn = c.numel()
        nw = self.n_w
        # Refuse mobile oil even when the current drawdown accidentally gives zero flow.
        selected = torch.zeros(nw, dtype=torch.bool, device=self.device)
        selected[list(self.surface_well_indices)] = True
        mask = selected[cw]
        if bool(
            (sys["Cv"][mask, 0].abs() > 1e-12 * sys["Cv"][mask, 1].abs().clamp_min(1e-30)).any()
        ):
            raise PropertyModelError(
                "FIM physical pipe binding supports water-only mobility; oil/gas requires a multiphase model"
            )
        wi = self.conn_wi * self.conn_sched
        pc = p.view(-1)[c]
        pw = p_bh[cw] + self.rho_wb[cw] * 9.80665 * self.conn_dz
        water_dq = torch.stack(
            [
                wi * (props["dlam_w_dp"][c] * (pc - pw) + props["lam_w"][c]),
                wi * props["dlam_w_ds"][c] * (pc - pw),
            ],
            dim=1,
        )
        for i in self.surface_well_indices:
            if not bool((sys["Cv"][cw == i, 1].abs() > 0).any()):
                raise PropertyModelError(
                    "Physical FIM binding requires nonzero water productivity; closed completions are not supported"
                )
        water_mass = torch.zeros(nw, dtype=torch.float64, device=self.device).index_add_(
            0, cw, sys["q_w"]
        )
        water_dp = torch.zeros_like(water_mass).index_add_(0, cw, sys["Cv"][:, 1])
        positive = water_mass.clamp_min(0)
        sources = {n.name: 0.0 for n in self.surface_network.nodes if n.pressure_pa is None}
        for i, node in zip(self.surface_well_indices, self.surface_nodes):
            sources[node] += float(positive[i])
        options = {"source_mass_rates_kg_s": sources, "pressure_tolerance_pa": 1e-6,
                   "mass_tolerance_kg_s": 1e-10}
        control = self.surface_pressure_control
        regulated = None
        if control is None:
            result = self.surface_network.solve(**options)
            sensitivity = self.surface_network.pressure_sensitivities(
                result, self.surface_nodes, self.surface_nodes)
        else:
            regulated = control.solve(self.surface_network, **options)
            result = regulated.network
            if self.surface_station_policy is not None:
                self.surface_station_policy._audit(result, self.surface_station_running)
            sensitivity = control.pressure_sensitivities(
                self.surface_network, regulated, self.surface_nodes, self.surface_nodes)
        K = torch.as_tensor(sensitivity, dtype=torch.float64, device=self.device)
        B = torch.zeros((nw, nconn, 2), dtype=torch.float64, device=self.device)
        B[cw, torch.arange(nconn, device=self.device)] = sys["Bv"]
        D = torch.diag(sys["D"])
        rate_scale = sys["target"].abs().clamp_min(1e-12)
        states = {}
        rho = self.fluid.rho_water_ref
        for row, (i, node) in enumerate(zip(self.surface_well_indices, self.surface_nodes)):
            ctrl, qmax = self.step_targets[i]
            if ctrl != "rate" or not math.isfinite(qmax) or qmax < 0:
                raise ValueError("Physical FIM surface well requires a nonnegative rate target")
            q = water_mass[i] / rho
            surface_pressure = result.pressures_pa[node]
            floor = self.bhp_limit(self.wells[i])
            required = max(surface_pressure, floor)
            qs = max(qmax, 0.01 / DAY)
            ps = max(required, 1e5)
            trial = float(q) + qs / ps * (float(p_bh[i]) - required)
            own = cw == i
            B[i].zero_()
            D[i].zero_()
            if trial <= 0 or trial >= qmax:
                target = 0.0 if trial <= 0 else qmax
                sys["r_w"][i] = q - target
                sys["is_rate"][i] = True
                sys["target"][i] = target
                B[i, own] = water_dq[own] / rho
                D[i, i] = water_dp[i] / rho
                mode = "surface_stop" if target == 0 else "rate"
                rate_scale[i] = max(abs(target), 0.01 / DAY)
            else:
                sys["r_w"][i] = p_bh[i] - required
                sys["is_rate"][i] = False
                sys["target"][i] = required
                D[i, i] = 1.0
                if surface_pressure >= floor:
                    mode = "surface"
                    for col, j in enumerate(self.surface_well_indices):
                        active = 1.0 if float(water_mass[j]) >= 0 else 0.0
                        own_j = cw == j
                        B[i, own_j] -= K[row, col] * active * water_dq[own_j]
                        D[i, j] -= K[row, col] * active * water_dp[j]
                else:
                    mode = "bhp_limit"
            self.step_mode[i] = mode
            states[self.wells[i].name] = {
                "node": node,
                "mode": mode,
                "bhp_pa": float(p_bh[i]),
                "network_reference_pressure_pa": surface_pressure,
                "water_mass_rate_kg_s": float(water_mass[i]),
                "ideal_choke_dissipation_w": max(0.0, float(q))
                * max(0.0, float(p_bh[i]) - surface_pressure),
            }
        sys["B_full"] = B
        sys["D_full"] = D
        sys["D"] = torch.diagonal(D)
        sys["rate_normalization"] = rate_scale
        self.surface_report = {"network": result.to_dict(), "wells": states,
                               "station": None if self.surface_station_policy is None else {
                                   "running_pumps": list(self.surface_station_running),
                                   "policy": asdict(self.surface_station_policy)},
                               "pressure_control": None if regulated is None else {
                                   **asdict(control), "status": regulated.status,
                                   "speed_ratio": regulated.speed_ratio,
                                   "pressure_error_pa": regulated.pressure_error_pa,
                                   "evaluations": regulated.evaluations}}
        return sys

    def commit(self, p_bh):
        super().commit(p_bh)
        for i in self.surface_well_indices:
            well = self.wells[i]
            state = self.surface_report["wells"][well.name]
            well.surface_mode = state["mode"]
            well.surface_reference_pressure_mpa = state["network_reference_pressure_pa"] / 1e6
