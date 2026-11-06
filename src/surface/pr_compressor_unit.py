"""Adiabatic mapped dry-gas compressor between finite PR plenums.

Three algebraic unknowns (map beta, isentropic T, delivery T) share the graph
Newton. The card is a prescribed-speed quasi-steady machine, with explicit
isolation and a map choke limit; rotor inertia and surge cycles are excluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .compressor_map import CompressorMap
from .liquid import PropertyModelError, _name, _number
from .peng_robinson import R


@dataclass(frozen=True)
class PRCompressorUnit:
    name: str
    source: str
    target: str
    card: CompressorMap
    speed_rad_s: float
    reference_mole_fractions: tuple
    maximum_composition_departure: float = 0.01
    minimum_compressibility_factor: float = 0.5
    maximum_shaft_power_w: float | None = None
    open: bool = True

    def __post_init__(self):
        for name in (self.name, self.source, self.target):
            _name(name)
        if self.source == self.target or not isinstance(self.card, CompressorMap):
            raise ValueError("Compressor requires a card and distinct finite plenums")
        _number(self.speed_rad_s, "compressor speed", positive=True)
        x = tuple(
            float(_number(v, "reference composition", nonnegative=True))
            for v in self.reference_mole_fractions
        )
        if not x or not math.isclose(sum(x), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Compressor reference composition must sum to one")
        object.__setattr__(self, "reference_mole_fractions", x)
        if not 0 < self.maximum_composition_departure <= 0.05:
            raise ValueError("A narrow, explicit reference-gas composition envelope is required")
        if not 0 < self.minimum_compressibility_factor <= 1:
            raise ValueError("Dry gas compressibility bound required")
        if self.maximum_shaft_power_w is not None:
            _number(self.maximum_shaft_power_w, "maximum compressor power", positive=True)
        if type(self.open) is not bool:
            raise ValueError("Compressor isolation must be boolean")

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "card": CompressorMap(**data["card"])})


class CompressorBundle:
    def __init__(self, model, state, openings, offset):
        self.model = model
        self.entries = []
        self.initial, self.lower, self.upper = [], [], []
        f = model.fluid
        for unit in model.compressors:
            a, b = model.names.index(unit.source), model.names.index(unit.target)
            enabled = openings[unit.name]
            old = state.nodes[a][1]
            old_speed = dict(state.rotors).get(unit.name, unit.speed_rad_s)
            op = unit.card.operating_point(
                old_speed if old_speed > 0 else unit.speed_rad_s,
                old.pressure_pa,
                old.temperature_k,
                state.nodes[b][1].pressure_pa,
                strict=False,
            )
            x = [v / sum(old.component_moles) for v in old.component_moles]
            cp = sum(v * c.ideal_cp_j_mol_k for v, c in zip(x, f.components))
            ratio = op["pressure_ratio"] if enabled else 1.0
            tis = old.temperature_k * ratio ** (R / cp)
            tout = old.temperature_k + (tis - old.temperature_k) / op["isentropic_efficiency"]
            seed = [op["beta"] if enabled else 0.0, tis / f.tref, tout / f.tref]
            seed[1:] = [
                min(f.max_temperature_k / f.tref, max(f.min_temperature_k / f.tref, v))
                for v in seed[1:]
            ]
            self.initial.append(f.tensor(seed))
            self.lower.append(
                f.tensor([0, f.min_temperature_k / f.tref, f.min_temperature_k / f.tref])
            )
            self.upper.append(
                f.tensor([1, f.max_temperature_k / f.tref, f.max_temperature_k / f.tref])
            )
            tables = [
                f.tensor(getattr(unit.card, key))
                for key in ("corrected_mass_flow_kg_s", "pressure_ratio", "isentropic_efficiency")
            ]
            self.entries.append(
                (
                    unit,
                    a,
                    b,
                    offset,
                    enabled,
                    f.tensor(unit.card.corrected_speed_rad_s),
                    f.tensor(unit.card.beta),
                    tables,
                    f.tensor(unit.reference_mole_fractions),
                    max(1.0, cp * old.temperature_k),
                )
            )
            offset += 3

    @staticmethod
    def interpolate(axis, values, query):
        i = (torch.searchsorted(axis, query.reshape(1), right=True) - 1).clamp(0, len(axis) - 2)
        a, b = axis.index_select(0, i).squeeze(0), axis.index_select(0, i + 1).squeeze(0)
        va, vb = values.index_select(0, i).squeeze(0), values.index_select(0, i + 1).squeeze(0)
        return va + (vb - va) * ((query - a) / (b - a))

    def evaluate(self, y, data, shaft_speeds=None):
        f = self.model.fluid
        net = {}
        residual, records, traces = [], {}, []
        valid = y.new_ones((), dtype=torch.bool)
        for unit, a, b, offset, enabled, speeds, betas, tables, reference, scale in self.entries:
            source, target = data[a], data[b]
            beta, tis, tout = y[offset], y[offset + 1] * f.tref, y[offset + 2] * f.tref
            x = source["x"][0]
            hin, sin = (
                source["q"]["molar_enthalpy_j_mol"][0],
                source["q"]["molar_entropy_j_mol_k"][0],
            )
            if not enabled:
                residual.append(
                    torch.stack((beta, (tis - source["t"]) / f.tref, (tout - source["t"]) / f.tref))
                )
                records[unit.name] = {
                    "beta": beta,
                    "mass_rate_kg_s": beta * 0,
                    "shaft_power_w": beta * 0,
                    "entropy_production_w_k": beta * 0,
                    "isentropic_temperature_k": tis,
                    "delivery_temperature_k": tout,
                    "pressure_ratio": beta * 0 + 1,
                    "efficiency": beta * 0 + 1,
                    "map_choked": beta.new_zeros((), dtype=torch.bool),
                }
                continue
            theta = source["t"] / unit.card.reference_temperature_k
            speed = (shaft_speeds or {}).get(unit.name, unit.speed_rad_s) / theta.sqrt()
            # Interpolate beta first, then corrected speed: all tables remain
            # device tensors and all state/temperature derivatives remain live.
            lines = [
                torch.stack([self.interpolate(betas, row, beta) for row in table])
                for table in tables
            ]
            flow, ratio, eta = [self.interpolate(speeds, line, speed) for line in lines]
            surge = self.interpolate(speeds, tables[1][:, 0], speed)
            choke = self.interpolate(speeds, tables[1][:, -1], speed)
            requested = target["p"] / source["p"]
            choked = requested < choke
            discharge = source["p"] * ratio
            safe_pressure = discharge.clamp_min(f.min_pressure_pa)
            qi = f.tensor_homogeneous(safe_pressure, tis, x, "vapor")
            qo = f.tensor_homogeneous(target["p"], tout, x, "vapor")
            his, sis = qi["molar_enthalpy_j_mol"], qi["molar_entropy_j_mol_k"]
            hout, sout = qo["molar_enthalpy_j_mol"], qo["molar_entropy_j_mol_k"]
            row = torch.stack(
                (
                    torch.where(choked, beta - 1, requested - ratio),
                    (sis - sin) / R,
                    (hout - hin - (his - hin) / eta.clamp_min(1e-8)) / scale,
                )
            )
            residual.append(row)
            mass_rate = flow * (source["p"] / unit.card.reference_pressure_pa) / theta.sqrt()
            mw = (x * f.molar_masses).sum()
            molar_rate = mass_rate / mw
            components = molar_rate * x
            phi = self.model.gravity_m_s2 * (
                self.model.nodes[a].elevation_m - self.model.reference_elevation_m
            )
            incoming = torch.cat(
                (components, y.new_zeros(1), (molar_rate * (hin + mw * phi)).reshape(1))
            )
            outgoing = torch.cat(
                (components, y.new_zeros(1), (molar_rate * (hout + mw * phi)).reshape(1))
            )
            net[a] = net.get(a, 0) - incoming
            net[b] = net.get(b, 0) + outgoing
            power, entropy = molar_rate * (hout - hin), molar_rate * (sout - sin)
            admissible = (
                (len(source["phase_n"]) == 1)
                & (speed >= speeds[0])
                & (speed <= speeds[-1])
                & (requested <= surge)
                & (ratio >= 1)
                & (eta > 0)
                & (eta <= 1)
                & (mass_rate > 0)
                & (source["q"]["compressibility_factor"][0] >= unit.minimum_compressibility_factor)
                & (qi["compressibility_factor"] >= unit.minimum_compressibility_factor)
                & (qo["compressibility_factor"] >= unit.minimum_compressibility_factor)
                & ((x - reference).abs().sum() <= unit.maximum_composition_departure)
                & qi["valid"]
                & qo["valid"]
                & (power >= -1e-8)
                & (entropy >= -1e-8)
            )
            if unit.maximum_shaft_power_w is not None:
                admissible = admissible & (power <= unit.maximum_shaft_power_w)
            valid = valid & admissible
            records[unit.name] = {
                "beta": beta,
                "mass_rate_kg_s": mass_rate,
                "shaft_power_w": power,
                "entropy_production_w_k": entropy,
                "isentropic_temperature_k": tis,
                "delivery_temperature_k": tout,
                "pressure_ratio": ratio,
                "efficiency": eta,
                "map_choked": choked,
            }
            traces.append((discharge, tis, target["p"], tout, x))
        return net, residual, records, traces, valid

    def validate(self, traces):
        for pin, tin, pout, tout, x in traces:
            for p, t in ((pin, tin), (pout, tout)):
                q = self.model.fluid.flash_pt(p, t, x)
                if q["phase_count"] != 1:
                    raise PropertyModelError("Dry gas compressor trace crosses a phase boundary")
                homogeneous = self.model.fluid.tensor_homogeneous(p, t, x, "vapor")
                if (
                    float(
                        (
                            homogeneous["log_fugacity_coefficients"]
                            - q["phases"][0]["log_fugacity_coefficients"]
                        )
                        .abs()
                        .max()
                        .cpu()
                    )
                    > 1e-7
                ):
                    raise PropertyModelError(
                        "Dry gas compressor trace uses a metastable vapor root"
                    )

    def serialize(self, records):
        result = {}
        for unit, _a, _b, _offset, enabled, *_ in self.entries:
            row = records[unit.name]
            values = torch.stack(list(row.values())).detach().cpu().tolist()
            result[unit.name] = {
                **{
                    key: bool(value) if row[key].dtype == torch.bool else value
                    for key, value in zip(row, values)
                },
                "enabled": enabled,
            }
        return result


def inventory_guesses(model, state, dt, openings, previous):
    """Seed newly arriving components only; the physical old inventory is exact."""
    f = model.fluid
    guesses = (
        [f.tensor(s.component_moles) for _, s in state.nodes]
        if previous is None
        else list(previous)
    )
    for unit in model.compressors:
        if not openings[unit.name]:
            continue
        a, b = model.names.index(unit.source), model.names.index(unit.target)
        source = state.nodes[a][1]
        x = [v / sum(source.component_moles) for v in source.component_moles]
        mw = sum(v * c.molar_mass_kg_mol for v, c in zip(x, f.components))
        op = unit.card.operating_point(
            dict(state.rotors).get(unit.name, unit.speed_rad_s) or unit.speed_rad_s,
            source.pressure_pa,
            source.temperature_k,
            state.nodes[b][1].pressure_pa,
            strict=False,
        )
        addition = f.tensor(x) * (dt * op["mass_rate_kg_s"] / mw)
        old = f.tensor(state.nodes[b][1].component_moles)
        guesses[b] = torch.where(old == 0, torch.maximum(guesses[b], addition), guesses[b])
    return guesses
