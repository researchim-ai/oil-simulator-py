"""Resident mechanical stepping with unchanged stop location and accepted logs.

Only controls, scalar statuses and actual stopping indices cross the boundary
on trials. Full reports are materialized after temporal acceptance. The cache
is committed only after a successful public interval and checks input bits.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from .liquid import ConservationError, EquipmentOperatingError, HydraulicConvergenceError
from .torch_linear import bisect_scalar
from .torch_liquid_bridge import TorchPhysicalStep


@dataclass
class LiquidDeviceControls:
    commands: torch.Tensor
    fixed: torch.Tensor
    dt: torch.Tensor
    dt_value: float | None = None
    openings: torch.Tensor | None = None
    opening_values: tuple | None = None


@dataclass(frozen=True)
class DeviceHydraulicStep:
    """Short-lived accepted coordinates, never part of a public journal."""

    model_id: str
    data: tuple
    ledger: dict
    journal: str = "full"


class DeferredLiquidRecord(Mapping):
    def __init__(self, owner, dt, data, ledger, evaluations, valves):
        self.owner, self.dt = owner, dt
        self.data, self.ledger = data, ledger
        self.evaluations, self.valves = evaluations, valves
        self.report = None

    def materialize(self):
        if self.report is None:
            self.report = self.owner.export_record(
                self.dt, self.data, self.ledger, self.evaluations, self.valves
            )
            self.data = self.ledger = None
        return self.report

    def __getitem__(self, key):
        if key == "dt_s":
            return self.dt
        return self.materialize()[key]

    def __iter__(self):
        return iter(self.materialize())

    def __len__(self):
        return len(self.materialize())


class TorchLiquidInterval:
    def __init__(self, bridge, journal="full", branches=()):
        self._layout = None
        self.journal = journal
        self.branches = branches
        self.bridge = bridge
        self.e = bridge.equations
        self.physics = bridge.physics

    def begin(self, state, arrays, commands, boundary):
        b, e, p = self.bridge, self.e, self.physics
        b.assert_model()
        arrays = tuple(np.asarray(a, dtype=np.float64) for a in arrays)
        cache = b._interval_cache
        # Public states may be reconstructed from JSON between calls. Reuse
        # only exactly matching bits, including signed zero, model and clock.
        hit = (
            cache is not None
            and cache[0] == state.time_s
            and all(
                np.array_equal(a.view(np.uint64), old.view(np.uint64))
                for a, old in zip(arrays, cache[1])
            )
        )
        if hit:
            b.cache_hits += 1
            coordinates = cache[2]
        else:
            b.cache_misses += 1
            packet = b.upload(np.concatenate(arrays))
            coordinates = packet.split([len(a) for a in arrays])
            b.state_upload_batches += 1
            b.state_upload_bytes += packet.numel() * packet.element_size()
        values = np.asarray([boundary[n] for n in b.boundary_nodes], dtype=np.float64)
        key = values.tobytes()
        if b._fixed_cache is None or b._fixed_cache[0] != key:
            fixed = torch.full((e.nn,), p.reference_pressure_pa, dtype=e.dtype, device=e.device)
            if len(values):
                indices = b.upload(
                    np.asarray([b.node_index[n] for n in b.boundary_nodes]), dtype=torch.long
                )
                fixed = fixed.scatter(0, indices, b.upload(values))
            b._fixed_cache = key, fixed
        command_values = np.asarray(commands, dtype=np.float64)
        key = command_values.tobytes()
        if b._command_cache is None or b._command_cache[0] != key:
            command_tensor = b.upload(command_values) if len(command_values) else e.tensor([])
            b._command_cache = key, command_tensor
        return tuple(coordinates), LiquidDeviceControls(
            b._command_cache[1],
            b._fixed_cache[1],
            torch.zeros((), dtype=e.dtype, device=e.device),
        )

    def commit(self, state, coordinates):
        p = self.physics
        pressure, rates, speeds = (
            dict(state.pressures_pa),
            dict(state.pipe_volume_rates_m3_s),
            dict(state.speeds_rad_s),
        )
        host = (
            np.asarray([pressure[n] for n in p.free], dtype=np.float64),
            np.asarray([rates[b.name] for b in p.network.pipes], dtype=np.float64),
            np.asarray([speeds[n] for n in p.names], dtype=np.float64),
        )
        self.bridge._interval_cache = (state.time_s, host, tuple(coordinates))

    def solve(self, old, dt, controls, valves):
        e = self.e
        midpoint = [v.at(dt / 2) if hasattr(v, "at") else v for v in valves]
        if controls.dt_value != dt:
            # Only a scalar kernel argument changes. No state-vector upload,
            # allocation or CPU-backed tensor construction on every trial.
            controls.dt.fill_(dt)
            controls.dt_value = dt
            self.bridge.scalar_parameter_updates += 1
            self.bridge.scalar_parameter_bytes += controls.dt.element_size()
        values = tuple(v.opening for v in midpoint)
        if controls.opening_values != values:
            controls.openings = self.bridge.upload(np.asarray(values)) if values else e.tensor([])
            controls.opening_values = values
        args = (*old, controls.dt, controls.commands, controls.fixed, controls.openings)
        return e.solve(*args), args

    def roundoff_mask(self, oldw, data):
        endw, wm, torque, power = data[2], data[5], data[6], data[7]
        loads = torch.where(wm > 0, power / torch.where(wm > 0, wm, 1.0), 0.0)
        deceleration = (loads + self.e.drag * wm - torque) / self.e.J
        return (
            (oldw > 0)
            & (torque < 0)
            & (endw >= 0)
            & (deceleration > 0)
            & (endw / torch.where(deceleration > 0, deceleration, 1.0) <= 1e-12)
        )

    def status(self, old, result, dt_tensor, controls):
        _, residual, data, success, _ = result
        e, p, b = self.e, self.physics, self.bridge
        negative = data[2] < -1e-12
        project = self.roundoff_mask(old[2], data)
        ledger = e.mechanical_ledger(*old, dt_tensor, data, residual)
        shaft = (ledger["minimum_speed_rad_s"] < -1e-10) | (
            ledger["max_speed_change_rad_s"] > p.max_speed_change_rad_s
        )
        change = ledger["max_pressure_change_pa"] > p.max_pressure_change_pa
        finite = (torch.isfinite(data[0]) & (data[0] > 0)).all()
        domain = (
            (data[0] > p.network.fluid.vapor_pressure_pa)
            & (data[0] <= p.max_pressure_pa)
            & (
                (data[0] - p.reference_pressure_pa).abs() / p.bulk_modulus_pa
                <= p.max_compression_fraction
            )
        ).all()
        flags = torch.stack(
            (
                success,
                negative.any(),
                (negative & (controls.commands >= 0)).any(),
                project.any(),
                shaft,
                change,
                finite,
                domain,
                ledger["conservation_valid"].to(torch.bool),
            )
        ).to(e.dtype)
        host = b.download(torch.cat((flags, b.physical_flags(data))), "scalar")
        if not host[0]:
            raise HydraulicConvergenceError("Monolithic Torch liquid/shaft solve failed")
        return host, ledger, project

    def validate(self, flags):
        if flags[4]:
            raise HydraulicConvergenceError("Shaft change guard")
        if flags[5]:
            raise HydraulicConvergenceError("Pressure change guard")
        if not flags[6]:
            raise ValueError("pressure_pa must be finite and positive")
        if not flags[7]:
            raise EquipmentOperatingError(
                "Acoustic pressure/compression domain exceeded; no cavitation model"
            )
        # The legacy path audits pump operating flags before mechanical guards.
        # Caller applies this ordering explicitly before entering validate().
        if not flags[8]:
            raise ConservationError("Transient liquid mass/mechanical energy audit failed")

    def step(self, old, dt, controls, valves):
        b, e, p = self.bridge, self.e, self.physics
        for valve in valves:
            if hasattr(valve, "actuator"):
                valve.actuator.step(valve.position, valve.target, dt, valve.mode)
        result, args = self.solve(old, dt, controls, valves)
        flags, ledger, projection = self.status(old, result, args[3], controls)
        evaluations = result[-1]
        x, residual, data = result[:3]
        had_stop = bool(flags[1])
        if had_stop:
            if flags[2]:
                raise HydraulicConvergenceError("Shaft overshoots positive equilibrium")
            mask = b.download(data[2] < -1e-12, "scalar")
            indices = b.upload(np.flatnonzero(mask), dtype=torch.long)

            def event(h):
                if h == 0:
                    return b.read_scalar(old[2].index_select(0, indices).min())
                out, _ = self.solve(old, h, controls, valves)
                status = b.download(
                    torch.stack((out[3].to(e.dtype), out[2][2].index_select(0, indices).min())),
                    "scalar",
                )
                if not status[0]:
                    raise HydraulicConvergenceError("Monolithic Torch liquid/shaft solve failed")
                return float(status[1])

            dt = bisect_scalar(event, 0.0, dt, xtol=1e-12, rtol=1e-11)
            result, args = self.solve(old, dt, controls, valves)
            evaluations += result[-1]
            flags, ledger, projection = self.status(old, result, args[3], controls)
            x, residual, data = result[:3]
            located = torch.zeros_like(projection).scatter(
                0, indices, data[2].index_select(0, indices).abs() < 1e-8
            )
            projection = projection | located
        if had_stop or flags[3]:
            offset = e.nf + e.nb
            x = torch.cat((x[:offset], torch.where(projection, 0.0, x[offset:])))
            residual, data = e.evaluate(x, *args)
            # Located zeroing follows the original stop tolerance, not a new
            # Newton solve; the unchanged mechanical/physical audits decide.
            flags, ledger, _ = self.status(
                old,
                (x, residual, data, torch.isfinite(residual).all(), evaluations),
                args[3],
                controls,
            )
        b.audit(flags[9:], p.names)
        self.validate(flags)
        coordinates = (data[0], data[1], torch.where(data[2] > 0, data[2], 0.0))
        report = DeferredLiquidRecord(self, dt, data, ledger, evaluations, valves)
        return TorchPhysicalStep(coordinates, report, coordinates, tuple(old))

    def export_record(self, dt, data, ledger, evaluations, valves):
        b = self.bridge
        if self.journal == "compact":
            return self.export_compact_record(dt, data, ledger, evaluations, valves)
        selected = (0, 1, 2, 3, 4, 6, 7)
        chunks = [data[i].reshape(-1) for i in selected] + [v.reshape(-1) for v in ledger.values()]
        host = b.download(torch.cat(chunks), "journal")
        b.state_download_batches += 1
        b.state_download_bytes += sum(v.numel() * v.element_size() for v in data[:3])
        offset = 0
        unpacked = []
        for v in chunks:
            unpacked.append(host[offset : offset + v.numel()])
            offset += v.numel()
        host_data = [None] * 11
        for i, value in zip(selected, unpacked[: len(selected)]):
            host_data[i] = value
        metrics = dict(zip(ledger, unpacked[len(selected) :]))
        return b.finish_step(
            dt, host_data, evaluations, valves, metrics=metrics, return_report_only=True
        )

    def compact_layout(self):
        if self._layout is not None:
            return self._layout
        b, p = self.bridge, self.physics
        if not hasattr(b, "_compact_layouts"):
            b._compact_layouts = OrderedDict()
        if self.branches not in b._compact_layouts:
            branch_index = p._journal_branch_order
            relief_nodes = {
                n for v in p.network.valves if v.name in p.relief_valves for n in (v.start, v.end)
            }
            nodes = tuple(n for n in p._nodes if n in relief_nodes)
            b._compact_layouts[self.branches] = (
                b.upload(
                    np.asarray([branch_index[n] for n in self.branches], dtype=np.int64),
                    dtype=torch.long,
                ),
                b.upload(
                    np.asarray([b.node_index[n] for n in nodes], dtype=np.int64), dtype=torch.long
                ),
                nodes,
            )
            if len(b._compact_layouts) > 4:
                b._compact_layouts.popitem(last=False)
        b._compact_layouts.move_to_end(self.branches)
        self._layout = b._compact_layouts[self.branches]
        return self._layout

    def export_compact_record(self, dt, data, ledger, evaluations, valves):
        from .liquid_journal import METRIC_ARRAYS, METRIC_GUARDS, actuator_report

        b, p = self.bridge, self.physics
        flow_index, pressure_index, pressure_names = self.compact_layout()
        keys = [k for k in ledger if k not in METRIC_ARRAYS | METRIC_GUARDS]
        chunks = [ledger[k].reshape(-1) for k in keys] + [
            data[6],
            data[7],
            ledger["unit_shaft_energy_residuals_j"],
            data[4].index_select(0, flow_index),
            data[3].index_select(0, pressure_index),
        ]
        host = b.download(torch.cat(chunks), "journal")
        values, offset = [], 0
        for chunk in chunks:
            values.append(host[offset : offset + chunk.numel()])
            offset += chunk.numel()
        report = {k: float(v[0]) for k, v in zip(keys, values[: len(keys)])}
        torque, power, unit_error, flows, pressures = values[len(keys) :]
        pressure = dict(zip(pressure_names, pressures))
        report.update(
            {
                "dt_s": dt,
                "valve_actuators": actuator_report(valves, dt),
                "midpoint_branch_volume_rates_m3_s": dict(zip(self.branches, map(float, flows))),
                "relief_openings": {
                    v.name: p.relief_valves[v.name].opening(
                        float(pressure[v.start] - pressure[v.end])
                    )
                    for v in valves
                    if v.name in p.relief_valves
                },
                "motor_torques_nm": dict(zip(p.names, map(float, torque))),
                "pump_shaft_power_w": dict(zip(p.names, map(float, power))),
                "unit_shaft_energy_residuals_j": dict(zip(p.names, map(float, unit_error))),
                "nonlinear_evaluations": evaluations,
            }
        )
        return report

    def export_coordinates(self, coordinates):
        b = self.bridge
        host = b.download(torch.cat([v.reshape(-1) for v in coordinates]), "endpoint")
        b.state_download_batches += 1
        b.state_download_bytes += host.nbytes
        sizes = [v.numel() for v in coordinates]
        return np.split(host, np.cumsum(sizes)[:-1])
