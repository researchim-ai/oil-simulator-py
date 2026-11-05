"""Prescribed-speed liquid pump cards inside finite PR transport cells.

The existing Pump/EquipmentCurve card is reused. Force and shaft power enter
the same momentum/energy Newton. Disabled units retain an explicit passive
loss; motor switching does not erase the fluid inventory or reverse flow.
Rotor inertia, powered reverse quadrants and cavitation kinetics are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .liquid import Pump, _name, _number

G = 9.80665


def serialize_pumps(records, openings):
    result = {}
    for name, row in records.items():
        # One device-to-host transfer per unit, rather than one per scalar.
        values = torch.stack(list(row.values())).detach().cpu().tolist()
        result[name] = {**dict(zip(row, values)), "enabled": openings[name]}
    return result


@dataclass(frozen=True)
class PRPumpUnit:
    node: str
    card: Pump
    minimum_density_kg_m3: float = 400.0
    maximum_compressibility_fraction: float = 0.01
    stopped_loss_coefficient: float = 1.0
    vapor_pressure_pa: float | None = None
    vapor_pressure_temperature_k: float | None = None
    maximum_vapor_pressure_temperature_departure_k: float = 0.5

    @property
    def name(self):
        return self.card.name

    def __post_init__(self):
        _name(self.node)
        if not isinstance(self.card, Pump) or self.card.check_valve:
            raise ValueError("PR pump needs a Pump card without an implicit check valve")
        _number(self.minimum_density_kg_m3, "minimum liquid density", positive=True)
        if not 0 < self.maximum_compressibility_fraction <= 0.05:
            raise ValueError("Liquid pump card requires a small g*H/a^2 envelope")
        _number(self.stopped_loss_coefficient, "stopped pump loss", nonnegative=True)
        _number(
            self.maximum_vapor_pressure_temperature_departure_k,
            "vapor pressure temperature range",
            positive=True,
        )
        if (self.vapor_pressure_pa is None) != (self.vapor_pressure_temperature_k is None):
            raise ValueError("Vapor pressure and its temperature must be specified together")
        if self.vapor_pressure_pa is not None:
            _number(self.vapor_pressure_pa, "vapor pressure", nonnegative=True)
            _number(self.vapor_pressure_temperature_k, "vapor pressure temperature", positive=True)
        if (
            self.card.npsh_required_m is not None or self.card.npsh_curve is not None
        ) and self.vapor_pressure_pa is None:
            raise ValueError("NPSH evaluation needs explicit vapor pressure and temperature")

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "card": Pump(**data["card"])})


class TorchPumpCell:
    def __init__(self, model, unit):
        self.unit = unit
        self.index = model.names.index(unit.node)
        node = model.nodes[self.index]
        card = unit.card
        if (
            node.kind != "pipe_cell"
            or node.drift_flux is not None
            or node.distribution_strength
            or node.buoyancy_relaxation_s
        ):
            raise ValueError("Pump cell must be a homogeneous finite pipe cell")
        if node.face_diameter("left") != node.face_diameter("right"):
            raise ValueError("Pump body requires constant cross-section")
        if (
            card.inlet_diameter_m is not None
            and abs(card.inlet_diameter_m - node.face_diameter("left")) > 1e-12
        ):
            raise ValueError("Pump inlet diameter must match its represented pipe face")
        neighbors = {}
        for link in model.links:
            if link.source == unit.node:
                neighbors[link.source_port] = link.target
            elif link.target == unit.node:
                neighbors[link.target_port] = link.source
        if neighbors.get("left") != card.start or neighbors.get("right") != card.end:
            raise ValueError(
                "Pump card endpoints must match the body's physical left/right neighbors"
            )
        self.inlet = model.names.index(card.start)
        self.area = node.face_area("left")
        self.efficiency = model.fluid.tensor(card.efficiency)
        self.curves = {
            name: None
            if getattr(card, name) is None
            else (
                model.fluid.tensor(getattr(card, name).flow_m3_s),
                model.fluid.tensor(getattr(card, name).values),
            )
            for name in ("head_curve", "efficiency_curve", "npsh_curve")
        }

    def curve(self, name, query):
        x, y = self.curves[name]
        i = (torch.searchsorted(x, query.reshape(1), right=True) - 1).clamp(0, len(x) - 2)
        a, b = x.index_select(0, i).squeeze(0), x.index_select(0, i + 1).squeeze(0)
        ya, yb = y.index_select(0, i).squeeze(0), y.index_select(0, i + 1).squeeze(0)
        # Edge continuation is for Newton only. Accepted flow must stay in the card.
        return ya + (yb - ya) * (query - a) / (b - a)

    def evaluate(self, data, enabled):
        unit, card = self.unit, self.unit.card
        body, inlet = data[self.index], data[self.inlet]
        q = self.area * body["u"]
        rho = body["rho"]
        if not enabled:
            force = (
                -0.5 * self.area * unit.stopped_loss_coefficient * rho * body["u"] * body["u"].abs()
            )
            return (
                force,
                q * 0,
                {
                    "flow_m3_s": q,
                    "head_m": q * 0,
                    "efficiency": self.efficiency,
                    "shaft_power_w": q * 0,
                    "hydraulic_power_w": q * 0,
                    "force_n": force,
                    "npsh_available_m": q * 0,
                "compressibility_fraction": q * 0,
                },
                q.new_ones((), dtype=torch.bool),
            )
        speed = card.speed_ratio
        reduced = q.abs() / speed
        if self.curves["head_curve"] is None:
            head = speed**2 * card.shutoff_head_m - card.curve_coefficient_s2_m5 * q * q.abs()
        else:
            nominal = self.curve("head_curve", reduced)
            sign = torch.where(q >= 0, 1.0, -1.0)
            head = speed**2 * (card.shutoff_head_m - sign * (card.shutoff_head_m - nominal))
        eta = (
            self.efficiency
            if self.curves["efficiency_curve"] is None
            else self.curve("efficiency_curve", reduced)
        )
        hydraulic = rho * G * head * q
        power = hydraulic / eta.clamp_min(1e-8)
        force = self.area * rho * G * head
        compressibility = G * head / body["q"]["frozen_sound_speed_m_s"][0].square()
        valid = (
            (len(body["phase_n"]) == 1)
            & (len(inlet["phase_n"]) == 1)
            & (rho >= unit.minimum_density_kg_m3)
            & (inlet["rho"] >= unit.minimum_density_kg_m3)
            & (q >= -64 * torch.finfo(q.dtype).eps * card.max_flow_m3_s)
            & (q <= speed * card.max_flow_m3_s)
            & (head >= 0)
            & (eta > 0)
            & (eta <= 1)
            & (compressibility <= unit.maximum_compressibility_fraction)
            & torch.isfinite(power)
        )
        if card.max_shaft_power_w is not None:
            valid = valid & (power <= card.max_shaft_power_w)
        npsh = q * 0
        if unit.vapor_pressure_pa is not None:
            npsh = (inlet["p"] - unit.vapor_pressure_pa) / (inlet["rho"] * G) + (
                inlet["u"]
            ).square() / (2 * G)
            valid = valid & (
                (inlet["t"] - unit.vapor_pressure_temperature_k).abs()
                <= unit.maximum_vapor_pressure_temperature_departure_k
            )
            required = q * 0
            if card.npsh_required_m is not None:
                required = required + card.npsh_required_m * speed**2
            if self.curves["npsh_curve"] is not None:
                required = self.curve("npsh_curve", reduced) * speed**2
            valid = valid & (npsh >= required)
        return (
            force,
            power,
            {
                "flow_m3_s": q,
                "head_m": head,
                "efficiency": eta,
                "shaft_power_w": power,
                "hydraulic_power_w": hydraulic,
                "force_n": force,
                "npsh_available_m": npsh,
                "compressibility_fraction": compressibility,
            },
            valid,
        )
