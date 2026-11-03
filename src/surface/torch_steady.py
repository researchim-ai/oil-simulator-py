"""Torch steady preview used by the transient station dispatcher.

The preview shares equipment laws with the transient Torch equations. Python
only enumerates candidate configurations; pressure/flow solves stay in Torch.
"""

import copy
import math
from dataclasses import replace
from types import SimpleNamespace

import torch

from .liquid import (
    EquipmentOperatingError,
    HydraulicConvergenceError,
    InvalidPressureError,
    NetworkResult,
)
from .rotating_drive import RotatingDrive
from .torch_linear import TorchNewton
from .torch_liquid import TorchLiquidEquations
from .torch_liquid_bridge import TorchLiquidBridge


class TorchNetworkView:
    def __init__(self, network, device="cpu", compiled=None):
        self.network = network
        self.device = device
        self._compiled = [] if compiled is None else compiled

    def __getattr__(self, name):
        return getattr(self.network, name)

    def with_controls(self, pump_speed_ratios=None, pump_status=None, valve_openings=None):
        speeds, status, openings = pump_speed_ratios or {}, pump_status or {}, valve_openings or {}
        names = {p.name for p in self.pumps}
        if (set(speeds) | set(status)) - names or set(openings) - {v.name for v in self.valves}:
            raise ValueError("Unknown pump or valve controls")
        network = copy.copy(self.network)
        network.pumps = tuple(
            replace(
                p, speed_ratio=speeds.get(p.name, p.speed_ratio), open=status.get(p.name, p.open)
            )
            for p in self.pumps
        )
        network.valves = tuple(
            replace(v, opening=openings.get(v.name, v.opening)) for v in self.valves
        )
        network._branches = network.pipes + network.pumps + network.valves
        return TorchNetworkView(network, self.device, self._compiled)

    def solve(
        self,
        source_mass_rates_kg_s=None,
        boundary_pressures_pa=None,
        initial=None,
        pressure_tolerance_pa=1e-3,
        mass_tolerance_kg_s=1e-8,
        max_iterations=60,
        pump_speed_ratios=None,
        pump_status=None,
        valve_openings=None,
    ):
        if pump_speed_ratios or pump_status or valve_openings:
            return self.with_controls(pump_speed_ratios, pump_status, valve_openings).solve(
                source_mass_rates_kg_s,
                boundary_pressures_pa,
                initial,
                pressure_tolerance_pa,
                mass_tolerance_kg_s,
                max_iterations,
            )
        if any(n.elevation_m or n.source_mass_rate_kg_s for n in self.nodes) or any(
            (source_mass_rates_kg_s or {}).values()
        ):
            raise ValueError(
                "Torch transient station preview requires horizontal pressure-driven geometry"
            )
        if not self._compiled:
            self._compiled.append(TorchSteadyEquations(self.network, self.device))
        return self._compiled[0].solve(
            self.network,
            boundary_pressures_pa,
            initial,
            pressure_tolerance_pa,
            mass_tolerance_kg_s,
            max_iterations,
        )

    def pump_speed_sensitivities(self, result, pump_names, pressure_nodes):
        # Explicit finite differences of Torch solves; no hidden SciPy tangent.
        columns = []
        controls = result.operating_controls
        for name in pump_names:
            speed = controls["pump_speed_ratios"][name]
            h = 1e-5 * max(1.0, speed)
            plus = self.with_controls(
                **{
                    **controls,
                    "pump_speed_ratios": {**controls["pump_speed_ratios"], name: speed + h},
                }
            ).solve(initial=result)
            minus = self.with_controls(
                **{
                    **controls,
                    "pump_speed_ratios": {**controls["pump_speed_ratios"], name: speed - h},
                }
            ).solve(initial=result)
            columns.append(
                [(plus.pressures_pa[n] - minus.pressures_pa[n]) / (2 * h) for n in pressure_nodes]
            )
        return torch.tensor(columns, dtype=torch.float64).T.numpy()


class TorchSteadyEquations:
    def __init__(self, network, device):
        free = tuple(n.name for n in network.nodes if n.pressure_pa is None)
        branches = network.pipes + network.pumps + network.valves
        names = tuple(p.name for p in network.pumps)
        physics = SimpleNamespace(
            network=network,
            free=free,
            names=names,
            _nodes=tuple(n.name for n in network.nodes),
            _branches=branches,
            storage_volumes_m3={n: 1.0 for n in free},
            bulk_modulus_pa=1e12,
            reference_pressure_pa=1e6,
            max_pressure_pa=1e12,
            max_compression_fraction=0.05,
            _pscale=1e6,
            _qscale=max(
                [0.001]
                + [p.max_flow_m3_s for p in network.pumps]
                + [math.pi * p.diameter_m**2 / 4 for p in network.pipes]
            ),
            nominal_speeds_rad_s={n: 1.0 for n in names},
            no_flow_power_w_at_nominal={n: 0.0 for n in names},
            drives={n: RotatingDrive(1.0, 1e6, 1e9, 1e9) for n in names},
            max_station_electrical_draw_w=None,
            max_station_electrical_export_w=None,
            relief_valves={},
            actuated_valves={},
        )
        self.e = TorchLiquidEquations(physics, device)
        self.network = network
        self.size = self.e.nf + self.e.nb
        self.newton = TorchNewton(device, self.e.newton.maximum_iterations)
        self.newton.host_convergence = True
        self.pattern = tuple(
            zip(*[(r, c) for r, c in zip(*self.e.pattern) if r < self.size and c < self.size])
        )
        self.protected = torch.tensor(
            [b.check_valve for b in branches], dtype=torch.bool, device=self.e.device
        )
        self.flag_bridge = SimpleNamespace(equations=self.e, physics=physics)

    def evaluate(self, x, fixed, omega, openings, opened):
        e = self.e
        nf, npip, npump = e.nf, e.npipe, e.npump
        pressure = fixed.scatter(0, e.free, x[:nf] * e.ps)
        flow = x[nf:] * e.qs
        drive = pressure[e.start] - pressure[e.end]
        pipe_loss = e.pipe_loss(flow[:npip])
        pump_loss, _, power, _, ratio, eta = e.pump_quantities(
            flow[npip : npip + npump], omega, drive[npip : npip + npump]
        )
        qv = flow[npip + npump :]
        valve_loss = e.rho / (2 * (e.valve_cv * openings.clamp_min(1e-30)).square()) * qv * qv.abs()
        loss = torch.cat((pipe_loss, pump_loss, valve_loss))
        zero_loss = torch.cat(
            (
                torch.zeros_like(pipe_loss),
                -e.rho * 9.80665 * e.shutoff * ratio.square(),
                torch.zeros_like(valve_loss),
            )
        )
        blocked = self.protected & (drive <= zero_loss)
        force = (drive - loss) / e.ps
        branch_r = torch.where(opened, torch.where(blocked, flow / e.qs, force), flow / e.qs)
        continuity = e.incidence(flow)[e.free] * e.rho / (e.rho * e.qs)
        return torch.cat((continuity, branch_r)), (pressure, flow, loss, power, eta, blocked)

    def solve(self, network, boundaries, initial, ptol, mtol, max_iterations):
        e = self.e
        boundary = {n.name: n.pressure_pa for n in network.nodes if n.pressure_pa is not None}
        patch = dict(boundaries or {})
        if set(patch) - set(boundary) or any(v is None for v in patch.values()):
            raise ValueError("Unsupported steady preview boundary override")
        boundary.update(patch)
        for p in boundary.values():
            if not math.isfinite(p) or p <= network.fluid.vapor_pressure_pa:
                raise InvalidPressureError("Liquid pressure is at or below vapor pressure")
        guess = sum(boundary.values()) / len(boundary)
        fixed = e.tensor([boundary.get(n.name, guess) for n in network.nodes])
        omega = e.tensor([p.speed_ratio if p.open else 0.0 for p in network.pumps])
        openings = e.tensor([v.opening for v in network.valves])
        opened = torch.tensor(
            [b.open and getattr(b, "opening", 1) > 0 for b in network._branches],
            dtype=torch.bool,
            device=e.device,
        )
        p0 = e.tensor(
            [initial.pressures_pa[n] if initial is not None else guess for n in e.physics.free]
        )
        pressure = fixed.scatter(0, e.free, p0)
        qp, qv, _ = e.algebraic_seed(pressure, omega, openings)
        qpipe = e.tensor(
            [
                initial.pipe_volume_rates_m3_s.get(p.name, 0.0) if initial is not None else 0.0
                for p in network.pipes
            ]
        )
        flow = torch.cat((qpipe, qp, qv))
        flow = torch.where(opened, flow, 0.0)
        x0 = torch.cat((p0 / e.ps, flow / e.qs))
        lower = torch.cat(
            (
                torch.full_like(
                    p0, math.nextafter(network.fluid.vapor_pressure_pa, math.inf) / e.ps
                ),
                torch.where(self.protected, 0.0, -math.inf),
            )
        )
        upper = torch.full_like(lower, math.inf)
        function = lambda x: self.evaluate(x, fixed, omega, openings, opened)[0]
        self.newton.maximum_iterations = max_iterations
        x, r, ok, count = self.newton.solve(
            function,
            x0,
            lower,
            upper,
            self.pattern,
            tolerance=min(2e-11, ptol / e.ps * 0.1, mtol / (e.rho * e.qs) * 0.1),
        )
        _, data = self.evaluate(x, fixed, omega, openings, opened)
        pressure, flow, loss, power, eta, blocked = data
        flow = torch.where(blocked | ~opened, 0.0, flow)
        x = torch.cat((x[: e.nf], flow / e.qs))
        r, data = self.evaluate(x, fixed, omega, openings, opened)
        pressure, flow, loss, power, eta, blocked = data
        valid = (
            ok
            & (r[: e.nf].abs().max() * (e.rho * e.qs) <= mtol)
            & (r[e.nf :].abs().max() * e.ps <= ptol)
        )
        node_mass = e.rho * e.incidence(flow)
        boundary_mask = torch.ones(e.nn, dtype=e.dtype, device=e.device).scatter(0, e.free, 0.0)
        boundary_mass = node_mass * boundary_mask
        mass_res = node_mass - boundary_mass
        drive = pressure[e.start] - pressure[e.end]
        pump_hydraulic = -flow[e.npipe : e.npipe + e.npump] * drive[e.npipe : e.npipe + e.npump]
        active = opened[e.npipe : e.npipe + e.npump] & ~blocked[e.npipe : e.npipe + e.npump]
        pump_hydraulic = torch.where(active, pump_hydraulic, 0.0)
        power = torch.where(active, power, 0.0)
        pipe_heat = flow[: e.npipe] * loss[: e.npipe]
        valve_heat = flow[e.npipe + e.npump :] * loss[e.npipe + e.npump :]
        valve_heat = torch.where(opened[e.npipe + e.npump :], valve_heat, 0.0)
        balance = (
            (pressure * boundary_mass).sum() / e.rho
            + pump_hydraulic.sum()
            - pipe_heat.sum()
            - valve_heat.sum()
        )
        npsh = (
            pressure[e.start[e.npipe : e.npipe + e.npump]] - network.fluid.vapor_pressure_pa
        ) / (e.rho * 9.80665) + (flow[e.npipe : e.npipe + e.npump] / e.inlet_area).square() / (
            2 * 9.80665
        )
        required = torch.maximum(
            e.required_npsh,
            omega.square()
            * e.interpolate(
                e.maps["npsh_curve"], flow[e.npipe : e.npipe + e.npump] / omega.clamp_min(1e-30)
            ),
        )
        fake = (
            x[: e.nf] * e.ps,
            flow[: e.npipe],
            omega,
            pressure,
            flow,
            torch.where(active, omega, 0.0),
            torch.zeros_like(omega),
            power,
            torch.cat((pipe_heat, valve_heat)),
            torch.ones((), dtype=e.dtype, device=e.device),
            torch.ones((), dtype=e.dtype, device=e.device),
        )
        flags = TorchLiquidBridge.physical_flags(self.flag_bridge, fake)
        tensors = (
            pressure,
            flow,
            r,
            node_mass,
            boundary_mass,
            mass_res,
            pipe_heat,
            valve_heat,
            pump_hydraulic,
            power,
            eta,
            npsh,
            required,
            blocked.to(e.dtype),
            loss,
            drive,
            balance.reshape(1),
            valid.reshape(1).to(e.dtype),
            flags.to(e.dtype),
        )
        lengths = [v.numel() for v in tensors]
        packed = torch.cat([v.reshape(-1) for v in tensors]).detach().cpu().numpy()
        parts = []
        offset = 0
        for length in lengths:
            parts.append(packed[offset : offset + length])
            offset += length
        (
            pressure,
            flow,
            residual,
            node_mass,
            boundary_mass,
            mass_res,
            pipe_heat,
            valve_heat,
            hydraulic,
            power,
            eta,
            npsh,
            required,
            blocked,
            loss,
            drive,
            balance,
            valid,
            flags,
        ) = parts
        if not valid[0]:
            raise HydraulicConvergenceError("Torch steady preview did not converge")
        TorchLiquidBridge.audit(flags, e.physics.names)
        live = [
            bool(b.open and getattr(b, "opening", 1) > 0 and not blocked[i])
            for i, b in enumerate(network._branches)
        ]
        adjacency = {n.name: set() for n in network.nodes}
        for b, active_branch in zip(network._branches, live):
            if active_branch:
                adjacency[b.start].add(b.end)
                adjacency[b.end].add(b.start)
        unseen = set(adjacency)
        while unseen:
            stack = [next(iter(unseen))]
            component = set()
            while stack:
                n = stack.pop()
                if n not in component:
                    component.add(n)
                    stack.extend(adjacency[n] - component)
            unseen -= component
            if not component.intersection(boundary):
                raise EquipmentOperatingError(
                    "Pressure is undetermined in a component isolated by non-return valves"
                )
        if any(v <= network.fluid.vapor_pressure_pa for v in pressure):
            raise InvalidPressureError("Liquid pressure is at or below vapor pressure")
        allbranches = network._branches
        states = {
            b.name: "closed"
            if not b.open or getattr(b, "opening", 1) == 0
            else "blocked"
            if blocked[i]
            else "open"
            for i, b in enumerate(allbranches)
        }
        q = {b.name: float(v) for b, v in zip(allbranches, flow)}
        pmap = {n.name: float(v) for n, v in zip(network.nodes, pressure)}
        active_pump = {p.name: live[e.npipe + i] for i, p in enumerate(network.pumps)}
        return NetworkResult(
            pmap,
            {p.name: q[p.name] * e.rho for p in network.pipes},
            {p.name: q[p.name] for p in network.pipes},
            {n: float(boundary_mass[network._index[n]]) for n in boundary},
            dict(zip(e.physics._nodes, map(float, mass_res))),
            {
                p.name: float(residual[e.nf + i] * e.ps)
                for i, p in enumerate(network.pipes)
                if p.open
            },
            dict(zip((p.name for p in network.pipes), map(float, pipe_heat))),
            float(balance[0]),
            count,
            valve_mass_rates_kg_s={v.name: q[v.name] * e.rho for v in network.valves},
            valve_volume_rates_m3_s={v.name: q[v.name] for v in network.valves},
            valve_equation_residuals_pa={
                v.name: float(residual[e.nf + e.npipe + e.npump + i] * e.ps)
                for i, v in enumerate(network.valves)
                if v.open and v.opening > 0
            },
            valve_dissipation_w=dict(zip((v.name for v in network.valves), map(float, valve_heat))),
            pump_mass_rates_kg_s={p.name: q[p.name] * e.rho for p in network.pumps},
            pump_volume_rates_m3_s={p.name: q[p.name] for p in network.pumps},
            pump_equation_residuals_pa={
                p.name: float(residual[e.nf + e.npipe + i] * e.ps)
                for i, p in enumerate(network.pumps)
                if p.open
            },
            pump_hydraulic_power_w={
                p.name: float(hydraulic[i])
                for i, p in enumerate(network.pumps)
                if active_pump[p.name]
            },
            pump_shaft_power_w={
                p.name: float(power[i]) for i, p in enumerate(network.pumps) if active_pump[p.name]
            },
            pump_npsh_available_m={
                p.name: float(npsh[i])
                for i, p in enumerate(network.pumps)
                if active_pump[p.name] and p.inlet_diameter_m is not None
            },
            pump_npsh_required_m={
                p.name: float(required[i])
                for i, p in enumerate(network.pumps)
                if active_pump[p.name]
                and (p.npsh_required_m is not None or p.npsh_curve is not None)
            },
            pump_npsh_margin_m={
                p.name: float(npsh[i] - required[i])
                for i, p in enumerate(network.pumps)
                if active_pump[p.name]
                and (p.npsh_required_m is not None or p.npsh_curve is not None)
            },
            pump_efficiencies={
                p.name: float(eta[i]) for i, p in enumerate(network.pumps) if active_pump[p.name]
            },
            pump_head_gains_m={
                p.name: float(-loss[e.npipe + i] / (e.rho * 9.80665))
                for i, p in enumerate(network.pumps)
                if active_pump[p.name]
            },
            branch_states=states,
            check_valve_driving_pressure_pa={
                b.name: float((drive - loss)[i])
                for i, b in enumerate(allbranches)
                if b.check_valve and b.open
            },
            operating_controls=network._equipment_controls(),
        )
