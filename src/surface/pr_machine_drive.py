"""Shaft inventories coupled to PR machine loads in the common Newton.

Running shafts use an energy-conserving midpoint torque balance. Isolated
machines have an exact constant-torque coast/spool trajectory, including rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .liquid import EquipmentOperatingError, _name, _number
from .rotating_drive import RotatingDrive


@dataclass(frozen=True)
class PRMachineDrive:
    machine: str
    drive: RotatingDrive
    nominal_speed_rad_s: float
    motor_torque_nm: float = 0.0
    initial_speed_rad_s: float | None = None
    relative_speed_tolerance: float = 1e-4
    ambient_temperature_k: float = 298.15
    heat_sink_node: str | None = None

    def __post_init__(self):
        _name(self.machine)
        if self.heat_sink_node is not None:
            _name(self.heat_sink_node)
        if not isinstance(self.drive, RotatingDrive):
            raise TypeError("Expected RotatingDrive")
        _number(self.nominal_speed_rad_s, "nominal shaft speed", positive=True)
        _number(self.ambient_temperature_k, "drive ambient temperature", positive=True)
        _number(self.motor_torque_nm, "motor torque")
        if (
            not -self.drive.max_braking_torque_nm
            <= self.motor_torque_nm
            <= self.drive.max_motor_torque_nm
        ):
            raise ValueError("Motor torque outside drive limits")
        if self.initial_speed_rad_s is not None:
            _number(self.initial_speed_rad_s, "initial shaft speed")
            if abs(self.initial_speed_rad_s) > self.drive.max_speed_rad_s:
                raise ValueError("Initial shaft speed exceeds drive limit")
        if not 0 < self.relative_speed_tolerance <= 0.01:
            raise ValueError("Explicit narrow rotor temporal tolerance required")

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "drive": RotatingDrive(**data["drive"])})


def isolated_motion(old, torque, dt, drive, *, signed=False):
    """Exact affine dynamics; signed shafts cross rest, legacy shafts stop there."""
    inertia, drag = drive.inertia_kg_m2, drive.viscous_drag_nm_s
    if signed:
        moving = torch.as_tensor(dt, dtype=old.dtype, device=old.device)
    else:
        stopping = (
            inertia * old / (-torque).clamp_min(1e-30)
            if drag == 0
            else inertia / drag * torch.log1p(drag * old / (-torque).clamp_min(1e-30))
        )
        moving = torch.minimum(
            old.new_tensor(dt), torch.where(torque < 0, stopping, old.new_tensor(math.inf))
        )
    x = drag / inertia * moving
    small = x.clamp(max=0.5)
    phi1 = sum((-small) ** n / math.factorial(n + 1) for n in range(18))
    phi2 = sum((-small) ** n / math.factorial(n + 2) for n in range(18))
    phi3 = sum((-small) ** n * (2 ** (n + 2) - 2) / math.factorial(n + 3) for n in range(18))
    safe = x.clamp_min(1e-30)
    em = -torch.expm1(-x)
    phi1 = torch.where(x < 0.5, phi1, em / safe)
    phi2 = torch.where(x < 0.5, phi2, (x - em) / safe.square())
    phi3 = torch.where(x < 0.5, phi3, (x - 2 * em - torch.expm1(-2 * x) / 2) / safe.pow(3))
    acceleration = (torque - drag * old) / inertia
    end = old + acceleration * moving * phi1
    if not signed:
        end = torch.where((torque < 0) & (stopping <= dt), old * 0, end)
    angle = old * moving + acceleration * moving.square() * phi2
    square = (
        old.square() * moving
        + 2 * old * acceleration * moving.square() * phi2
        + acceleration.square() * moving.pow(3) * phi3
    )
    return end, torque * angle, drag * square


def signed_motor_work(old, end, torque, dt, drive, *, isolated=False):
    """Separate positive/negative shaft work across zero for electrical losses.

    Open-machine speed is the linear midpoint trajectory. Isolated motion uses
    the exact affine trajectory and its zero event. No tensor host reads.
    """
    if isolated:
        end, total, _ = isolated_motion(old, torque, dt, drive, signed=True)
        safe_torque = torch.where(torque != 0, torque, torch.ones_like(torque))
        if drive.viscous_drag_nm_s == 0:
            crossing_time = -drive.inertia_kg_m2 * old / safe_torque
        else:
            ratio = -drive.viscous_drag_nm_s * old / safe_torque
            crossing_time = (
                drive.inertia_kg_m2
                / drive.viscous_drag_nm_s
                * torch.log1p(ratio.clamp_min(-1 + torch.finfo(old.dtype).eps))
            )
        crossing_time = crossing_time.clamp(0, dt)
        _, first, _ = isolated_motion(old, torque, crossing_time, drive, signed=True)
    else:
        total = dt * torque * (old + end) / 2
        crossing_time = dt * old.abs() / (old.abs() + end.abs()).clamp_min(1e-30)
        first = torque * old * crossing_time / 2
    first = torch.where(old * end < 0, first, total)
    second = total - first
    return first.clamp_min(0) + second.clamp_min(0), (-first).clamp_min(0) + (-second).clamp_min(0)


class RotorBundle:
    def __init__(self, model, state, dt, openings, commands, offset, previous_data=None):
        self.model, self.dt = model, dt
        self.entries = []
        self.initial, self.lower, self.upper = [], [], []
        self.isolated = {}
        self.signed_names = {
            u.name
            for u in (*model.pumps, *model.pump_connections, *model.compressors)
            if getattr(u, "signed_map", None) is not None
        }
        old = dict(state.rotors)
        commands = commands or {}
        if set(commands) - set(model._drives):
            raise ValueError("Unknown driven machine torque command")
        units = {u.name: u for u in (*model.pumps, *model.compressors, *model.pump_connections)}
        indices = {name: i for i, name in enumerate(model.names)}
        self.heat_indices = {
            d.machine: indices[d.heat_sink_node]
            for d in model.drives
            if d.heat_sink_node is not None
        }
        self.connection_names = {u.name for u in model.pump_connections}
        self.compressor_names = {u.name for u in model.compressors}
        self.source_indices = {
            u.name: indices[u.source] for u in (*model.compressors, *model.pump_connections)
        }
        previous_checks = []
        for drive in model.drives:
            torque = _number(commands.get(drive.machine, drive.motor_torque_nm), "motor torque")
            physics = drive.drive
            if not -physics.max_braking_torque_nm <= torque <= physics.max_motor_torque_nm:
                raise ValueError("Motor torque command outside limits")
            speed = old[drive.machine]
            unit = units[drive.machine]
            signed = drive.machine in self.signed_names
            mapping = getattr(unit, "signed_map", None)
            if mapping is not None and mapping.reference_speed_rad_s != drive.nominal_speed_rad_s:
                raise ValueError(
                    "Signed machine map reference speed must match drive nominal speed"
                )
            if (
                mapping is not None
                and openings[drive.machine]
                and abs(speed) > mapping.maximum_speed_rad_s
            ):
                raise EquipmentOperatingError(
                    "Opened signed machine speed outside its supplied map"
                )
            if (
                not openings[drive.machine]
                and hasattr(unit, "node")
                and any(
                    openings[link.name]
                    for link in model.links
                    if unit.node in (link.source, link.target)
                )
            ):
                raise EquipmentOperatingError(
                    "A coasting driven pump must be hydraulically isolated; represent bypass flow in an explicit branch"
                )
            if openings[drive.machine] and drive.machine in self.compressor_names and not signed:
                source = state.nodes[model.names.index(unit.source)][1]
                corrected = speed / math.sqrt(
                    source.temperature_k / unit.card.reference_temperature_k
                )
                if (
                    not unit.card.corrected_speed_rad_s[0]
                    <= corrected
                    <= unit.card.corrected_speed_rad_s[-1]
                ):
                    raise EquipmentOperatingError(
                        "Spool compressor up in isolation before opening outside its speed map"
                    )
                if len(source.phase_roots) != 1:
                    raise EquipmentOperatingError(
                        "Compressor suction must be dry gas before opening"
                    )
                if previous_data is not None:
                    previous_checks.append(
                        previous_data[indices[unit.source]]["q"]["compressibility_factor"][0]
                        >= unit.minimum_compressibility_factor
                    )
            if openings[drive.machine] and hasattr(unit, "node") and previous_data is not None:
                for name in (unit.node, unit.card.start):
                    point = previous_data[indices[name]]
                    if len(point["phase_n"]) != 1:
                        raise EquipmentOperatingError(
                            "Liquid pump requires single-phase liquid before opening"
                        )
                    previous_checks.append(point["rho"] >= unit.minimum_density_kg_m3)
            if signed:
                peak_draw = max(0.0, torque * speed) / physics.motor_efficiency
                peak_export = max(0.0, -torque * speed) * physics.regeneration_efficiency
            else:
                peak_draw = max(0.0, torque) * speed / physics.motor_efficiency
                peak_export = max(0.0, -torque) * speed * physics.regeneration_efficiency
            if (
                physics.max_electrical_draw_w is not None
                and peak_draw > physics.max_electrical_draw_w
            ) or (
                physics.max_electrical_export_w is not None
                and peak_export > physics.max_electrical_export_w
            ):
                raise EquipmentOperatingError(
                    "Motor command exceeds electrical rating at initial shaft speed"
                )
            guess = min(
                physics.max_speed_rad_s,
                max(
                    -physics.max_speed_rad_s if signed else 0.0,
                    speed
                    + dt * (torque - physics.viscous_drag_nm_s * speed) / physics.inertia_kg_m2,
                ),
            )
            self.entries.append(
                (
                    drive,
                    offset,
                    model.fluid.tensor(speed),
                    model.fluid.tensor(torque),
                    openings[drive.machine],
                    units[drive.machine],
                )
            )
            if not openings[drive.machine] or drive.machine in self.connection_names:
                self.isolated[drive.machine] = isolated_motion(
                    model.fluid.tensor(speed),
                    model.fluid.tensor(torque),
                    dt,
                    physics,
                    signed=signed,
                )
            self.initial.append(model.fluid.tensor([guess / drive.nominal_speed_rad_s]))
            self.lower.append(
                model.fluid.tensor(
                    [-physics.max_speed_rad_s / drive.nominal_speed_rad_s if signed else 0.0]
                )
            )
            self.upper.append(
                model.fluid.tensor([physics.max_speed_rad_s / drive.nominal_speed_rad_s])
            )
            offset += 1

        if previous_checks and not bool(torch.stack(previous_checks).all().cpu()):
            raise EquipmentOperatingError("Machine inlet is outside its liquid/dry-gas envelope")

    def speeds(self, y):
        return {
            d.machine: (old + y[offset] * d.nominal_speed_rad_s) / 2
            for d, offset, old, *_ in self.entries
        }

    def evaluate(self, y, pumps, compressors, data):
        rows, records = [], {}
        heat_sources = {}
        valid = y.new_ones((), dtype=torch.bool)
        for d, offset, old, torque, enabled, unit in self.entries:
            physics = d.drive
            signed = d.machine in self.signed_names
            end = y[offset] * d.nominal_speed_rad_s
            mid = (end + old) / 2
            power = (pumps if d.machine in pumps else compressors)[d.machine]["shaft_power_w"]
            if enabled:
                machine_record = (pumps if d.machine in pumps else compressors)[d.machine]
                load = (
                    machine_record["load_torque_nm"]
                    if "load_torque_nm" in machine_record
                    else power / mid.clamp_min(1e-30)
                )
                row = (
                    physics.inertia_kg_m2 * (end - old)
                    - self.dt * (torque - load - physics.viscous_drag_nm_s * mid)
                ) / (physics.inertia_kg_m2 * d.nominal_speed_rad_s)
                motor = self.dt * torque * mid
                drag = self.dt * physics.viscous_drag_nm_s * mid.square()
                if d.machine in self.connection_names:
                    closed = pumps[d.machine]["valve_closed"]
                    target, free_motor, free_drag = self.isolated[d.machine]
                    row = torch.where(closed, (end - target) / d.nominal_speed_rad_s, row)
                    motor = torch.where(closed, free_motor, motor)
                    drag = torch.where(closed, free_drag, drag)
                    if not signed:
                        valid = valid & (closed | (mid > 0))
                elif not signed:
                    valid = valid & (mid > 0)
            else:
                target, motor, drag = self.isolated[d.machine]
                row = (end - target) / d.nominal_speed_rad_s
            shaft = self.dt * power
            kinetic = 0.5 * physics.inertia_kg_m2 * (end - old) * (end + old)
            electrical = torch.where(
                motor >= 0,
                motor / physics.motor_efficiency,
                motor * physics.regeneration_efficiency,
            )
            if signed:
                positive_work, negative_work = signed_motor_work(
                    old, end, torque, self.dt, physics, isolated=not enabled
                )
                electrical = (
                    positive_work / physics.motor_efficiency
                    - negative_work * physics.regeneration_efficiency
                )
            losses = electrical - motor + drag
            if d.heat_sink_node is not None:
                i = self.heat_indices[d.machine]
                heat_sources[i] = heat_sources.get(i, 0) + losses / self.dt
            peak = torch.maximum(old, end)
            draw = torque.clamp_min(0) * peak / physics.motor_efficiency
            export = (-torque).clamp_min(0) * peak * physics.regeneration_efficiency
            if signed:
                endpoint_power = torch.stack((torque * old, torque * end))
                draw = endpoint_power.max().clamp_min(0) / physics.motor_efficiency
                export = (-endpoint_power.min()).clamp_min(0) * physics.regeneration_efficiency
            if physics.max_electrical_draw_w is not None:
                valid = valid & (draw <= physics.max_electrical_draw_w)
            if physics.max_electrical_export_w is not None:
                valid = valid & (export <= physics.max_electrical_export_w)
            valid = (
                valid
                & torch.isfinite(row)
                & (end >= (-physics.max_speed_rad_s if signed else 0))
                & (end <= physics.max_speed_rad_s)
            )
            if enabled and signed:
                valid = valid & (end.abs() <= unit.signed_map.maximum_speed_rad_s)
            if enabled and d.machine in self.compressor_names and not signed:
                theta = (
                    data[self.source_indices[d.machine]]["t"] / unit.card.reference_temperature_k
                )
                endpoint = end / theta.sqrt()
                valid = (
                    valid
                    & (endpoint >= unit.card.corrected_speed_rad_s[0])
                    & (endpoint <= unit.card.corrected_speed_rad_s[-1])
                )
            records[d.machine] = {
                "speed_rad_s": end,
                "load_evaluation_speed_rad_s": mid,
                "motor_torque_nm": torque,
                "shaft_work_j": shaft,
                "motor_work_j": motor,
                "drag_heat_j": drag,
                "electrical_energy_j": electrical,
                "motor_heat_j": electrical - motor,
                "kinetic_energy_change_j": kinetic,
                "environment_entropy_j_k": losses / d.ambient_temperature_k
                if d.heat_sink_node is None
                else losses * 0,
                "heat_to_environment_j": losses if d.heat_sink_node is None else losses * 0,
                "heat_to_node_j": losses if d.heat_sink_node is not None else losses * 0,
                "energy_residual_j": motor - shaft - drag - kinetic,
                "peak_electrical_draw_w": draw,
                "peak_electrical_export_w": export,
            }
            rows.append(row.reshape(1))
        return rows, records, valid, heat_sources

    def state(self, y):
        return tuple(
            (d.machine, float((y[o] * d.nominal_speed_rad_s).cpu())) for d, o, *_ in self.entries
        )

    @staticmethod
    def serialize(records):
        result = {}
        for name, row in records.items():
            values = torch.stack(list(row.values())).detach().cpu().tolist()
            record = dict(zip(row, values))
            throughput = sum(
                abs(record[k])
                for k in ("motor_work_j", "shaft_work_j", "drag_heat_j", "kinetic_energy_change_j")
            )
            if abs(record["energy_residual_j"]) > 1e-8 + 1e-9 * throughput:
                from .liquid import ConservationError

                raise ConservationError("Coupled rotor energy balance rejected")
            result[name] = record
        return result

    def pattern(self, offsets, compressor_entries, pump_entries=()):
        result = set()
        comp = {e[0].name: (e, 3) for e in compressor_entries}
        comp.update({e[0].name: (e, 5) for e in pump_entries})
        for d, offset, *_ in self.entries:
            if d.machine in comp:
                e, count = comp[d.machine]
                local = [
                    *range(offsets[e[1]], offsets[e[1] + 1]),
                    *range(offsets[e[2]], offsets[e[2] + 1]),
                    *range(e[3], e[3] + count),
                ]
            else:
                unit = next(u for u in self.model.pumps if u.name == d.machine)
                i = self.model.names.index(unit.node)
                local = list(range(offsets[i], offsets[i + 1]))
            local.append(offset)
            if d.heat_sink_node is not None:
                i = self.model.names.index(d.heat_sink_node)
                local.extend(range(offsets[i], offsets[i + 1]))
            result.update((offset, j) for j in local)
            result.update((j, offset) for j in local)
        return result
