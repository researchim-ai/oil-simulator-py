"""Explicit JSON/NumPy compatibility boundary around Torch liquid equations.

One packed upload and download per solve; no host matrix or inner-iteration
array transfer. The tensor equations can also be called without this adapter.
"""

import numpy as np
import torch

from .liquid import EquipmentOperatingError, HydraulicConvergenceError
from .torch_liquid import TorchLiquidEquations


class TorchLiquidBridge:
    def __init__(self, physics, device):
        self.physics = physics
        self.equations = TorchLiquidEquations(physics, device)
        self.upload_batches = 0
        self.download_batches = 0

    def physical_flags(self, data):
        e = self.equations
        p = self.physics
        _, _, _, pressure, flow, omega, _, power, _, _, _ = data
        q = flow[e.npipe : e.npipe + e.npump]
        ratio = omega / e.nominal
        dp = (pressure[e.start] - pressure[e.end])[e.npipe : e.npipe + e.npump]
        active = omega > 0
        reverse = active & ((q < -1e-10) | (-dp * q < -1e-7))
        outside = active & ((q < 0) | (q > ratio * e.qmax))
        suction = pressure[e.start[e.npipe : e.npipe + e.npump]]
        available = (suction - p.network.fluid.vapor_pressure_pa) / (e.rho * 9.80665) + (
            q / e.inlet_area
        ).square() / (2 * 9.80665)
        required = torch.maximum(
            e.required_npsh,
            ratio.square() * e.interpolate(e.maps["npsh_curve"], q / ratio.clamp_min(1e-30)),
        )
        npsh = active & (available < required)
        excessive = active & (power > e.max_shaft)
        masks = torch.stack((reverse, outside, npsh, excessive), dim=1)
        indices = (
            masks.to(torch.long).argmax(dim=0)
            if e.npump
            else torch.zeros(4, dtype=torch.long, device=e.device)
        )
        return torch.cat((masks.any(dim=0).to(e.dtype), indices.to(e.dtype)))

    def pack_result(self, x, residual, data, success, ledger):
        flags = self.physical_flags(data)
        self.last_tensor_data = data
        chunks = [
            x,
            residual,
            *[v.reshape(-1) for v in data],
            success.reshape(1).to(torch.float64),
            flags.to(torch.float64),
            *[v.reshape(-1) for v in ledger.values()],
        ]
        sizes = [v.numel() for v in chunks]
        host = torch.cat(chunks).detach().cpu().numpy()
        self.download_batches += 1
        arrays = []
        offset = 0
        for size in sizes:
            arrays.append(host[offset : offset + size])
            offset += size
        output = []
        for i, value in enumerate(arrays[2:13]):
            output.append(float(value[0]) if i in (9, 10) else value)
        self.metrics = {key: value for key, value in zip(ledger, arrays[15:])}
        return arrays[0], arrays[1], tuple(output), bool(arrays[13][0]), arrays[14]

    @staticmethod
    def audit(flags, pump_names=()):
        for i, (failed, message) in enumerate(
            zip(
                flags[:4],
                (
                    "Pump reverse/turbine flow unsupported",
                    "flow outside forward pump curve",
                    "insufficient NPSH",
                    "Pump total shaft power exceeded",
                ),
            )
        ):
            if failed:
                prefix = (
                    pump_names[int(flags[4 + i])] + ": " if pump_names and len(flags) >= 8 else ""
                )
                raise EquipmentOperatingError(prefix + message)

    def solve(self, oldp, oldq, oldw, dt, commands, boundary, valves):
        e = self.equations
        p = self.physics
        midpoint = [v.at(dt / 2) if hasattr(v, "at") else v for v in valves]
        arrays = [
            oldp,
            oldq,
            oldw,
            np.asarray([dt]),
            commands,
            np.asarray([boundary.get(n, p.reference_pressure_pa) for n in p._nodes]),
            np.asarray([v.opening for v in midpoint]),
        ]
        sizes = [len(a) for a in arrays]
        packed = e.tensor(np.concatenate(arrays))
        self.upload_batches += 1
        args = []
        offset = 0
        for size in sizes:
            args.append(packed[offset : offset + size])
            offset += size
        op, oq, ow, h, cmd, fixed, openings = args
        self.last_tensor_initial = (op, oq, ow)
        h = h[0]
        x, residual, data, success, evaluations = e.solve(op, oq, ow, h, cmd, fixed, openings)
        ledger = e.mechanical_ledger(op, oq, ow, h, data, residual)
        host_x, host_r, host_data, ok, flags = self.pack_result(x, residual, data, success, ledger)
        if not ok:
            raise HydraulicConvergenceError("Monolithic Torch liquid/shaft solve failed")
        cached_x = host_x.copy()

        def evaluate(value, audit=False):
            nonlocal cached_x, host_r, host_data, flags
            if not np.array_equal(value, cached_x):
                xt = e.tensor(value)
                self.upload_batches += 1
                rt, dat = e.evaluate(xt, op, oq, ow, h, cmd, fixed, openings)
                ledger = e.mechanical_ledger(op, oq, ow, h, dat, rt)
                _, host_r, host_data, _, flags = self.pack_result(
                    xt, rt, dat, torch.isfinite(rt).all(), ledger
                )
                cached_x = value.copy()
            if audit:
                self.last_tensor_initial = (op, oq, ow)
                self.audit(flags, self.physics.names)
            return host_r, host_data

        return host_data, host_x, evaluate, evaluations

    def sensors(self, state, boundary_pressures_pa=None):
        p = self.physics
        e = self.equations
        if not p.names:
            return {}
        boundary, _ = p._controls(boundary_pressures_pa)
        pressure = {**boundary, **dict(state.pressures_pa)}
        speeds = dict(state.speeds_rad_s)
        packet = e.tensor(
            [[pressure[b.start], pressure[b.end], speeds[b.name]] for b in p.network.pumps]
        )
        self.upload_batches += 1
        flow, ratio, margin, outside, npsh = e.pump_sensors(
            packet[:, 0], packet[:, 1], packet[:, 2]
        )
        host = (
            torch.stack((flow, ratio, margin, outside.to(e.dtype), npsh.to(e.dtype)), dim=1)
            .detach()
            .cpu()
            .numpy()
        )
        self.download_batches += 1
        result = {}
        for pump, row in zip(p.network.pumps, host):
            if row[3]:
                raise EquipmentOperatingError(
                    f"{pump.name}: sensor flow outside forward pump curve"
                )
            if row[4]:
                raise EquipmentOperatingError(f"{pump.name}: insufficient NPSH")
            required = pump.npsh_required_m is not None or pump.npsh_curve is not None
            result[pump.name] = {
                "flow_m3_s": float(row[0]),
                "speed_ratio": float(row[1]),
                "npsh_margin_m": float(row[2]) if row[1] > 0 and required else None,
            }
        return result

    def finish_step(self, dt, data, evaluations, valves):
        from .liquid import ConservationError

        p = self.physics
        m = self.metrics
        scalar = lambda key: float(m[key][0])
        if (
            scalar("minimum_speed_rad_s") < -1e-10
            or scalar("max_speed_change_rad_s") > p.max_speed_change_rad_s
        ):
            raise HydraulicConvergenceError("Shaft change guard")
        if scalar("max_pressure_change_pa") > p.max_pressure_change_pa:
            raise HydraulicConvergenceError("Pressure change guard")
        for value in data[0]:
            p._pressure(value)
        if not bool(scalar("conservation_valid")):
            raise ConservationError("Transient liquid mass/mechanical energy audit failed")
        p1, q1, w1, pm, qm, _, torque, power, _, _, _ = data
        w1 = np.maximum(w1, 0.0)
        extra = {
            "unit_shaft_energy_residuals_j",
            "boundary_mass_transfers_in_kg",
            "friction_unit_heat_j",
            "pump_unit_heat_j",
            "drag_unit_heat_j",
            "motor_unit_heat_j",
            "max_speed_change_rad_s",
            "minimum_speed_rad_s",
            "max_pressure_change_pa",
            "conservation_valid",
        }
        report = {key: scalar(key) for key in m if key not in extra}
        report.update(
            {
                "dt_s": dt,
                "valve_actuators": {
                    v.name: {
                        **v.actuator.advance(v.position, v.target, dt, v.mode),
                        "midpoint_position": v.at(dt / 2).opening,
                    }
                    for v in valves
                    if hasattr(v, "actuator")
                },
                "midpoint_pressures_pa": dict(zip(p._nodes, map(float, pm))),
                "midpoint_branch_volume_rates_m3_s": {
                    b.name: float(q) for b, q in zip(p._branches, qm)
                },
                "relief_openings": {
                    v.name: p.relief_valves[v.name].opening(
                        float(pm[p._nodes.index(v.start)] - pm[p._nodes.index(v.end)])
                    )
                    for v in valves
                    if v.name in p.relief_valves
                },
                "motor_torques_nm": dict(zip(p.names, map(float, torque))),
                "pump_shaft_power_w": dict(zip(p.names, map(float, power))),
                "unit_shaft_energy_residuals_j": dict(
                    zip(p.names, map(float, m["unit_shaft_energy_residuals_j"]))
                ),
                "boundary_mass_transfers_in_kg": {
                    n: float(m["boundary_mass_transfers_in_kg"][p._nodes.index(n)])
                    for n in p._nodes
                    if n not in p.free
                },
                "nonlinear_evaluations": evaluations,
                "final_pressures_pa": dict(zip(p.free, map(float, p1))),
                "final_pipe_volume_rates_m3_s": {
                    b.name: float(q) for b, q in zip(p.network.pipes, q1)
                },
                "final_speeds_rad_s": dict(zip(p.names, map(float, w1))),
                "dissipation_heat_j": {
                    "friction": {
                        b.name: float(q)
                        for b, q in zip(
                            p.network.pipes + p.network.valves, m["friction_unit_heat_j"]
                        )
                    },
                    "pump": dict(zip(p.names, map(float, m["pump_unit_heat_j"]))),
                    "shaft_drag": dict(zip(p.names, map(float, m["drag_unit_heat_j"]))),
                    "motor": dict(zip(p.names, map(float, m["motor_unit_heat_j"]))),
                },
            }
        )
        state = tuple(v.detach().clone() for v in self.last_tensor_data[:3])
        state = (state[0], state[1], state[2].clamp_min(0))
        return TorchPhysicalStep((p1, q1, w1), report, state, self.last_tensor_initial)


class TorchPhysicalStep(tuple):
    def __new__(cls, coordinates, report, device_state, initial_state):
        result = super().__new__(cls, (*coordinates, report))
        result.device_state = device_state
        result.initial_state = initial_state
        return result
