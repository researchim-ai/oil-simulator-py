"""Thermodynamic pump/check-valve connection between explicit finite plenums.

Flow is a common Newton unknown, not a cell-average numerical velocity. PS/PH
traces conserve shaft work. Head denotes reversible specific work divided by g.
The connection has no hidden inventory; inlet/outlet stocks belong to the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .liquid import PropertyModelError, Pump, _number
from .pr_pump_unit import G, PRPumpUnit, pump_curve, pump_head_efficiency


@dataclass(frozen=True)
class PRPumpConnection:
    card: Pump
    outlet_loss_pa_s2_m6: float = 0.0
    minimum_density_kg_m3: float = 400.0
    maximum_compressibility_fraction: float = 0.01
    vapor_pressure_pa: float | None = None
    vapor_pressure_temperature_k: float | None = None
    maximum_vapor_pressure_temperature_departure_k: float = 0.5

    @property
    def name(self):
        return self.card.name

    @property
    def source(self):
        return self.card.start

    @property
    def target(self):
        return self.card.end

    def __post_init__(self):
        if not isinstance(self.card, Pump) or self.card.start == self.card.end:
            raise ValueError("Pump connection needs a card and two distinct finite plenums")
        _number(self.outlet_loss_pa_s2_m6, "pump outlet valve resistance", nonnegative=True)
        PRPumpUnit(
            "validation",
            replace(self.card, check_valve=False),
            self.minimum_density_kg_m3,
            self.maximum_compressibility_fraction,
            vapor_pressure_pa=self.vapor_pressure_pa,
            vapor_pressure_temperature_k=self.vapor_pressure_temperature_k,
            maximum_vapor_pressure_temperature_departure_k=self.maximum_vapor_pressure_temperature_departure_k,
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "card": Pump(**data["card"])})


class PumpConnectionBundle:
    count = 5

    def __init__(self, model, state, openings, offset):
        self.model = model
        self.entries = []
        self.initial, self.lower, self.upper = [], [], []
        f = model.fluid
        speeds = dict(state.rotors)
        for unit in model.pump_connections:
            a, b = model.names.index(unit.source), model.names.index(unit.target)
            old = state.nodes[a][1]
            drive = model._drives.get(unit.name)
            ratio = (
                speeds[unit.name] / drive.nominal_speed_rad_s if drive else unit.card.speed_ratio
            )
            qscale = unit.card.max_flow_m3_s * max(
                1.0, drive.drive.max_speed_rad_s / drive.nominal_speed_rad_s if drive else ratio
            )
            rho = (
                sum(n * c.molar_mass_kg_mol for n, c in zip(old.component_moles, f.components))
                / model.nodes[a].volume_m3
            )
            pressure = min(
                f.max_pressure_pa,
                max(
                    f.min_pressure_pa,
                    old.pressure_pa + rho * G * unit.card.shutoff_head_m * ratio**2,
                ),
            )
            guess = (
                0.0
                if state.nodes[b][1].pressure_pa >= pressure
                else 0.25 * unit.card.max_flow_m3_s * ratio / qscale
            )
            self.initial.append(
                f.tensor(
                    [
                        guess,
                        math.log(pressure),
                        old.temperature_k / f.tref,
                        old.temperature_k / f.tref,
                        state.nodes[b][1].temperature_k / f.tref,
                    ]
                )
            )
            self.lower.append(
                f.tensor(
                    [
                        0.0,
                        math.log(f.min_pressure_pa),
                        f.min_temperature_k / f.tref,
                        f.min_temperature_k / f.tref,
                        f.min_temperature_k / f.tref,
                    ]
                )
            )
            self.upper.append(
                f.tensor(
                    [
                        1.0,
                        math.log(f.max_pressure_pa),
                        f.max_temperature_k / f.tref,
                        f.max_temperature_k / f.tref,
                        f.max_temperature_k / f.tref,
                    ]
                )
            )
            curves = {
                key: None
                if getattr(unit.card, key) is None
                else (
                    f.tensor(getattr(unit.card, key).flow_m3_s),
                    f.tensor(getattr(unit.card, key).values),
                )
                for key in ("head_curve", "efficiency_curve", "npsh_curve")
            }
            self.entries.append(
                (
                    unit,
                    a,
                    b,
                    offset,
                    openings[unit.name],
                    qscale,
                    curves,
                    f.tensor(unit.card.efficiency),
                )
            )
            offset += self.count

    def evaluate(self, y, data, speeds):
        f = self.model.fluid
        net, records = {}, {}
        rows, traces = [], []
        valid = y.new_ones((), dtype=torch.bool)
        for unit, a, b, offset, enabled, qscale, curves, efficiency in self.entries:
            source, target = data[a], data[b]
            q = y[offset] * qscale
            pressure = y[offset + 1].exp()
            tis, tpump, tdel = [y[offset + j] * f.tref for j in (2, 3, 4)]
            x = source["x"][0]
            if not enabled:
                rows.append(
                    torch.stack(
                        (
                            y[offset],
                            (pressure / source["p"]).log(),
                            (tis - source["t"]) / f.tref,
                            (tpump - source["t"]) / f.tref,
                            (tdel - target["t"]) / f.tref,
                        )
                    )
                )
                z = q * 0
                records[unit.name] = {
                    "flow_m3_s": z,
                    "mass_rate_kg_s": z,
                    "head_m": z,
                    "efficiency": efficiency,
                    "shaft_power_w": z,
                    "hydraulic_power_w": z,
                    "pump_pressure_pa": pressure,
                    "isentropic_temperature_k": tis,
                    "pump_temperature_k": tpump,
                    "delivery_temperature_k": tdel,
                    "entropy_production_w_k": z,
                    "valve_pressure_drop_pa": z,
                    "valve_closed": q.new_ones((), dtype=torch.bool),
                    "npsh_available_m": z,
                }
                continue
            speed = (
                speeds[unit.name] / self.model._drives[unit.name].nominal_speed_rad_s
                if unit.name in speeds
                else q.new_tensor(unit.card.speed_ratio)
            )
            head, eta = pump_head_efficiency(unit.card, curves, efficiency, q, speed)
            pressure_residual = (
                target["p"] + unit.outlet_loss_pa_s2_m6 * q.square() - pressure
            ) / source["p"]
            closed = (
                (pressure_residual > y[offset])
                if unit.card.check_valve
                else q.new_zeros((), dtype=torch.bool)
            )
            valve_row = (
                torch.minimum(y[offset], pressure_residual)
                if unit.card.check_valve
                else pressure_residual
            )
            qi = f.tensor_homogeneous(pressure, tis, x, "liquid")
            qp = f.tensor_homogeneous(pressure, tpump, x, "liquid")
            qd = f.tensor_homogeneous(target["p"], tdel, x, "liquid")
            hin = source["q"]["molar_enthalpy_j_mol"][0]
            sin = source["q"]["molar_entropy_j_mol_k"][0]
            mw = (x * f.molar_masses).sum()
            his = qi["molar_enthalpy_j_mol"]
            hout = qp["molar_enthalpy_j_mol"]
            hdel = qd["molar_enthalpy_j_mol"]
            escale = (mw * G * unit.card.shutoff_head_m).clamp_min(1.0)
            rows.append(
                torch.stack(
                    (
                        (his - hin - mw * G * head) / escale,
                        (qi["molar_entropy_j_mol_k"] - sin) / 8.31446261815324,
                        torch.where(
                            closed,
                            (tpump - tis) / f.tref,
                            (hout - hin - (his - hin) / eta.clamp_min(1e-8)) / escale,
                        ),
                        torch.where(closed, (tdel - target["t"]) / f.tref, (hdel - hout) / escale),
                        valve_row,
                    )
                )
            )
            actual = torch.where(closed, q * 0, q)
            mass = actual * source["rho"]
            molar = mass / mw
            components = molar * x
            phi = self.model.gravity_m_s2 * (
                self.model.nodes[a].elevation_m - self.model.reference_elevation_m
            )
            first = torch.cat((components, y.new_zeros(1), (molar * (hin + mw * phi)).reshape(1)))
            last = torch.cat((components, y.new_zeros(1), (molar * (hdel + mw * phi)).reshape(1)))
            net[a] = net.get(a, 0) - first
            net[b] = net.get(b, 0) + last
            power = molar * (hdel - hin)
            entropy = molar * (qd["molar_entropy_j_mol_k"] - sin)
            compressibility = G * head / source["q"]["frozen_sound_speed_m_s"][0].square()
            admissible = (
                (len(source["phase_n"]) == 1)
                & (source["rho"] >= unit.minimum_density_kg_m3)
                & (qi["density_kg_m3"] >= unit.minimum_density_kg_m3)
                & qi["valid"]
            )
            admissible = admissible & torch.where(
                closed,
                eta >= 0,
                (eta > 0)
                & qp["valid"]
                & qd["valid"]
                & (qp["density_kg_m3"] >= unit.minimum_density_kg_m3)
                & (qd["density_kg_m3"] >= unit.minimum_density_kg_m3),
            )
            admissible = (
                admissible
                & (eta <= 1)
                & (q <= speed * unit.card.max_flow_m3_s)
                & (head >= 0)
                & (compressibility <= unit.maximum_compressibility_fraction)
                & (power >= -1e-8)
                & (entropy >= -1e-8)
            )
            reduced = q / speed.clamp_min(1e-12)
            for table in curves.values():
                if table is not None:
                    admissible = admissible & (
                        closed | ((reduced >= table[0][0]) & (reduced <= table[0][-1]))
                    )
            if unit.card.max_shaft_power_w is not None:
                admissible = admissible & (power <= unit.card.max_shaft_power_w)
            npsh = q * 0
            if unit.vapor_pressure_pa is not None:
                npsh = (source["p"] - unit.vapor_pressure_pa) / (source["rho"] * G)
                required = q * 0 + (unit.card.npsh_required_m or 0) * speed.square()
                if curves["npsh_curve"] is not None:
                    required = pump_curve(curves, "npsh_curve", reduced) * speed.square()
                admissible = admissible & (
                    closed
                    | (
                        (npsh >= required)
                        & (
                            (source["t"] - unit.vapor_pressure_temperature_k).abs()
                            <= unit.maximum_vapor_pressure_temperature_departure_k
                        )
                    )
                )
            valid = valid & admissible & torch.isfinite(power)
            records[unit.name] = {
                "flow_m3_s": actual,
                "mass_rate_kg_s": mass,
                "head_m": head,
                "efficiency": eta,
                "shaft_power_w": power,
                "hydraulic_power_w": molar * (his - hin),
                "pump_pressure_pa": pressure,
                "isentropic_temperature_k": tis,
                "pump_temperature_k": tpump,
                "delivery_temperature_k": tdel,
                "entropy_production_w_k": entropy,
                "valve_pressure_drop_pa": torch.where(closed, pressure * 0, pressure - target["p"]),
                "valve_closed": closed,
                "npsh_available_m": npsh,
            }
            traces.append((unit, pressure, tis, tpump, target["p"], tdel, x, closed))
        return net, rows, records, traces, valid

    def validate(self, traces):
        for unit, pressure, tis, tpump, target, tdel, x, closed in traces:
            pairs = (
                [(pressure, tis)]
                if bool(closed.cpu())
                else [(pressure, tis), (pressure, tpump), (target, tdel)]
            )
            for p, t in pairs:
                phase = self.model.fluid.flash_pt(p, t, x)
                if (
                    phase["phase_count"] != 1
                    or float(phase["density_kg_m3"].cpu()) < unit.minimum_density_kg_m3
                ):
                    raise PropertyModelError(
                        "Pump thermodynamic trace leaves the single-phase dense-liquid domain"
                    )

    def serialize(self, records):
        output = {}
        for unit, *_ in self.entries:
            row = records[unit.name]
            values = torch.stack(list(row.values())).detach().cpu().tolist()
            output[unit.name] = {
                key: bool(value) if row[key].dtype == torch.bool else value
                for key, value in zip(row, values)
            }
            output[unit.name]["enabled"] = next(
                e[4] for e in self.entries if e[0].name == unit.name
            )
        return output

    def pattern(self, offsets):
        support = set()
        for _unit, a, b, offset, *_ in self.entries:
            local = [
                *range(offsets[a], offsets[a + 1]),
                *range(offsets[b], offsets[b + 1]),
                *range(offset, offset + self.count),
            ]
            for row in local:
                for col in range(offset, offset + self.count):
                    support.add((row, col))
            for row in range(offset, offset + self.count):
                support.update((row, col) for col in local)
        return support
