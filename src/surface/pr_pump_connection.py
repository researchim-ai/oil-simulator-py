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
from .pr_pump_four_quadrant import SignedPumpMap
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

    signed_map: SignedPumpMap | None = None
    mapped_speed_rad_s: float | None = None

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
        if self.signed_map is not None:
            if not isinstance(self.signed_map, SignedPumpMap):
                raise TypeError("Expected SignedPumpMap")
            if self.card.check_valve:
                raise ValueError("Signed pump connection requires check_valve=False")
            if self.card.npsh_required_m is not None or self.card.npsh_curve is not None:
                raise ValueError("Forward NPSH curves cannot certify signed operation")
        if self.mapped_speed_rad_s is not None:
            _number(self.mapped_speed_rad_s, "mapped shaft speed")
            if self.signed_map is None:
                raise ValueError("Mapped shaft speed requires a signed map")
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
        data = dict(data)
        if data.get("signed_map") is not None:
            data["signed_map"] = SignedPumpMap(**data["signed_map"])
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
            if unit.signed_map is not None:
                qscale = unit.signed_map.maximum_flow_m3_s
                omega = (
                    speeds[unit.name]
                    if drive
                    else (
                        unit.mapped_speed_rad_s
                        if unit.mapped_speed_rad_s is not None
                        else unit.card.speed_ratio * unit.signed_map.reference_speed_rad_s
                    )
                )
                coupling = (
                    unit.signed_map.euler_head_coefficient_m_s2 * abs(omega)
                    + unit.signed_map.linear_head_coefficient_m_s
                )
                delta = old.pressure_pa - state.nodes[b][1].pressure_pa + rho * G * coupling * omega
                resistance = (
                    unit.outlet_loss_pa_s2_m6
                    + rho * G * unit.signed_map.flow_loss_coefficient_s2_m5
                )
                estimate = math.copysign(math.sqrt(abs(delta) / max(resistance, 1e-30)), delta)
                guess = max(-0.9, min(0.9, estimate / qscale))
                donor = old if guess >= 0 else state.nodes[b][1]
                receiver = state.nodes[b][1] if guess >= 0 else old
                pressure = min(
                    f.max_pressure_pa,
                    max(
                        f.min_pressure_pa,
                        receiver.pressure_pa + unit.outlet_loss_pa_s2_m6 * (guess * qscale) ** 2,
                    ),
                )
                old = donor
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
                        -1.0 if unit.signed_map is not None else 0.0,
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
            if unit.signed_map is not None and enabled:
                local_net, local_rows, row, trace, admissible = self._signed(
                    unit, a, b, offset, qscale, y, data, speeds
                )
                for index, flux in local_net.items():
                    net[index] = net.get(index, 0) + flux
                rows.append(local_rows)
                records[unit.name] = row
                traces.append(trace)
                valid = valid & admissible
                continue
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
                if unit.signed_map is not None:
                    records[unit.name].update(
                        load_torque_nm=z, dissipation_w=z, mapped_speed_rad_s=z
                    )
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

    def _signed(self, unit, a, b, offset, qscale, y, data, speeds):
        """Direction-dependent PS/PH traces, with viscous loss heated at donor.

        Map head is oriented a->b. Trace head follows actual flow. Both physical
        port energies include their own potential; their sum is map shaft work.
        """
        f = self.model.fluid
        q = y[offset] * qscale
        forward = q >= 0
        sign = torch.where(forward, 1.0, -1.0)
        left, right = data[a], data[b]

        def select(key):
            return torch.where(forward, left[key], right[key])

        x = torch.where(forward, left["x"][0], right["x"][0])
        rho, pin, tin = select("rho"), select("p"), select("t")
        pout = torch.where(forward, right["p"], left["p"])
        hin = torch.where(
            forward, left["q"]["molar_enthalpy_j_mol"][0], right["q"]["molar_enthalpy_j_mol"][0]
        )
        sin = torch.where(
            forward, left["q"]["molar_entropy_j_mol_k"][0], right["q"]["molar_entropy_j_mol_k"][0]
        )
        sound = torch.where(
            forward, left["q"]["frozen_sound_speed_m_s"][0], right["q"]["frozen_sound_speed_m_s"][0]
        )
        mapping = unit.signed_map
        omega = speeds.get(
            unit.name,
            q.new_tensor(
                unit.mapped_speed_rad_s
                if unit.mapped_speed_rad_s is not None
                else unit.card.speed_ratio * mapping.reference_speed_rad_s
            ),
        )
        head, torque, shaft, hydraulic, loss, valid = mapping.evaluate(q, omega, rho)
        pressure = y[offset + 1].exp()
        tis, tpump, tdel = [y[offset + j] * f.tref for j in (2, 3, 4)]
        qi = f.tensor_homogeneous(pressure, tis, x, "liquid")
        qp = f.tensor_homogeneous(pressure, tpump, x, "liquid")
        qd = f.tensor_homogeneous(pout, tdel, x, "liquid")
        mw = (x * f.molar_masses).sum()
        phi_a, phi_b = [
            self.model.gravity_m_s2
            * (self.model.nodes[i].elevation_m - self.model.reference_elevation_m)
            for i in (a, b)
        ]
        dphi = sign * (phi_b - phi_a)
        coupling = (
            mapping.euler_head_coefficient_m_s2 * omega.abs() + mapping.linear_head_coefficient_m_s
        )
        work = mw * G * sign * coupling * omega
        heat = rho * G * mapping.viscous_torque_coefficient_m4_s * omega.square()
        escale = (mw * G * unit.card.shutoff_head_m).clamp_min(1.0)
        his, hp, hd = [state["molar_enthalpy_j_mol"] for state in (qi, qp, qd)]
        rows = torch.stack(
            (
                (his - hin - mw * (G * sign * head - dphi)) / escale,
                (qi["molar_entropy_j_mol_k"] - sin) / 8.31446261815324,
                (hp - hin - work + mw * dphi) / escale,
                (hd - hp) / escale,
                (sign * (pout - pressure) + unit.outlet_loss_pa_s2_m6 * q * q.abs()) / pin,
            )
        )
        mass = q * rho
        molar = mass / mw
        components = molar * x
        ha = torch.where(forward, hin, hd)
        hb = torch.where(forward, hd, hin)
        first = torch.cat((components, y.new_zeros(1), (molar * (ha + mw * phi_a)).reshape(1)))
        last = torch.cat((components, y.new_zeros(1), (molar * (hb + mw * phi_b)).reshape(1)))
        heating = torch.cat((y.new_zeros(len(x) + 1), heat.reshape(1)))
        net = {
            a: -first + torch.where(forward, heating, heating * 0),
            b: last + torch.where(forward, heating * 0, heating),
        }
        entropy = molar.abs() * (qd["molar_entropy_j_mol_k"] - sin) + heat / tin
        # Live donor selection and absolute head are essential in reverse operation.
        dense = torch.where(
            forward,
            q.new_tensor(len(left["phase_n"]) == 1, dtype=torch.bool),
            q.new_tensor(len(right["phase_n"]) == 1, dtype=torch.bool),
        )
        valid = valid & dense & (rho >= unit.minimum_density_kg_m3)
        for state in (qi, qp, qd):
            valid = valid & state["valid"] & (state["density_kg_m3"] >= unit.minimum_density_kg_m3)
        valid = valid & (G * head.abs() / sound.square() <= unit.maximum_compressibility_fraction)
        valid = valid & (entropy >= -1e-8)
        if unit.card.max_shaft_power_w is not None:
            valid = valid & (shaft.abs() <= unit.card.max_shaft_power_w)
        npsh = q * 0
        if unit.vapor_pressure_pa is not None:
            npsh = (pin - unit.vapor_pressure_pa) / (rho * G)
            valid = (
                valid
                & (npsh >= 0)
                & (
                    (tin - unit.vapor_pressure_temperature_k).abs()
                    <= unit.maximum_vapor_pressure_temperature_departure_k
                )
            )
        closed = q.new_zeros((), dtype=torch.bool)
        row = {
            "flow_m3_s": q,
            "mass_rate_kg_s": mass,
            "head_m": head,
            "efficiency": q.new_ones(()),
            "shaft_power_w": shaft,
            "hydraulic_power_w": hydraulic,
            "pump_pressure_pa": pressure,
            "isentropic_temperature_k": tis,
            "pump_temperature_k": tpump,
            "delivery_temperature_k": tdel,
            "entropy_production_w_k": entropy,
            "valve_pressure_drop_pa": sign * (pressure - pout),
            "valve_closed": closed,
            "npsh_available_m": npsh,
            "load_torque_nm": torque,
            "dissipation_w": loss,
            "mapped_speed_rad_s": omega,
        }
        return net, rows, row, (unit, pressure, tis, tpump, pout, tdel, x, closed, pin, tin), valid

    def validate(self, traces):
        for unit, pressure, tis, tpump, target, tdel, x, closed, *donor in traces:
            pairs = (
                [(pressure, tis)]
                if bool(closed.cpu())
                else [(pressure, tis), (pressure, tpump), (target, tdel)]
            )
            if donor:
                pairs.append(tuple(donor))
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
