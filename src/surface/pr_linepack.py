"""Horizontal finite-volume PR linepack coupled through quasi-steady Darcy faces.

Cells store real component N and U+PE, with finite walls. No fluid momentum,
kinetic storage, acoustic waves, phase slip or predictive viscosity is claimed.
The same network Newton solves cells, terminal vessels and every face velocity.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from .liquid import ConservationError, _name, _number
from .operation import _digest
from .separator_geometry import SeparatorGeometry
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .stratified_pr_flow import StratifiedPRFlow
from .stratified_pr_network import StratifiedConnection, StratifiedNetworkState, StratifiedPRNetwork
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class HorizontalPRPipe:
    name: str
    source: str
    target: str
    source_height_m: float
    target_height_m: float
    length_m: float
    diameter_m: float
    cells: int
    dynamic_viscosity_pa_s: float
    wall: VesselWall
    roughness_m: float = 0.0
    maximum_velocity_m_s: float = 10.0
    maximum_pressure_drop_fraction: float = 0.05
    allow_reverse: bool = True

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        if self.source == self.target:
            raise ValueError("Pipe terminals must be distinct vessels")
        for key in ("length_m", "diameter_m", "dynamic_viscosity_pa_s", "maximum_velocity_m_s"):
            _number(getattr(self, key), key, positive=True)
        for key in ("source_height_m", "target_height_m", "roughness_m"):
            _number(getattr(self, key), key, nonnegative=True)
        if (
            isinstance(self.cells, bool)
            or not isinstance(self.cells, int)
            or not 1 <= self.cells <= 64
        ):
            raise ValueError("Pipe supports 1..64 cells, subject to network unknown budget")
        if not isinstance(self.wall, VesselWall):
            raise TypeError("Pipe needs total finite-wall properties")
        if not isinstance(self.allow_reverse, bool):
            raise TypeError("allow_reverse must be boolean")
        if (
            self.roughness_m >= self.diameter_m / 2
            or not 0 < _number(self.maximum_pressure_drop_fraction, "pressure-drop domain") <= 1
        ):
            raise ValueError("Invalid pipe roughness/pressure domain")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["wall"] = VesselWall(**data["wall"])
        return cls(**data)


class PRLinepackNetwork:
    def __init__(self, vessels, pipes, **integration):
        if (
            not isinstance(vessels, dict)
            or not vessels
            or any(not isinstance(v, StratifiedPRFlow) for v in vessels.values())
        ):
            raise TypeError("Named terminal PR vessels required")
        self.terminals = dict(sorted(vessels.items()))
        self.pipes = tuple(pipes)
        if not self.pipes or any(not isinstance(p, HorizontalPRPipe) for p in self.pipes):
            raise TypeError("Nonempty HorizontalPRPipe definitions required")
        if len({p.name for p in self.pipes}) != len(self.pipes):
            raise ValueError("Duplicate pipe name")
        compiled = dict(self.terminals)
        connections = []
        self.cell_names = {}
        self.face_names = {}
        for pipe in self.pipes:
            if pipe.source not in self.terminals or pipe.target not in self.terminals:
                raise ValueError("Pipe references an unknown terminal")
            source, target = self.terminals[pipe.source].column, self.terminals[pipe.target].column
            center = source.elevation_m + pipe.source_height_m
            if abs(center - target.elevation_m - pipe.target_height_m) > 1e-9:
                raise ValueError("Horizontal pipe terminal absolute elevations differ")
            dx = pipe.length_m / pipe.cells
            names = tuple(f"pipe:{pipe.name}:cell:{i}" for i in range(pipe.cells))
            faces = tuple(f"pipe:{pipe.name}:face:{i}" for i in range(pipe.cells + 1))
            if set(names) & set(compiled):
                raise ValueError("Compiled pipe cell name collides with a vessel")
            self.cell_names[pipe.name] = names
            self.face_names[pipe.name] = faces
            for name in names:
                column = StratifiedPRColumn(
                    source.fluid,
                    SeparatorGeometry(pipe.diameter_m, dx, "horizontal"),
                    gravity_m_s2=source.gravity_m_s2,
                    elevation_m=center - pipe.diameter_m / 2,
                    quadrature_order=source.quadrature_order,
                    maximum_quadrature_order=source.maximum_quadrature_order,
                    relative_tolerance=source.relative_tolerance,
                    temperature_tolerance_k=source.temperature_tolerance_k,
                )
                wall = replace(
                    pipe.wall,
                    heat_capacity_j_k=pipe.wall.heat_capacity_j_k / pipe.cells,
                    fluid_ua_w_k=pipe.wall.fluid_ua_w_k / pipe.cells,
                    ambient_ua_w_k=pipe.wall.ambient_ua_w_k / pipe.cells,
                )
                compiled[name] = StratifiedPRFlow(column, wall)
            sequence = (pipe.source, *names, pipe.target)
            for i, face in enumerate(faces):
                left, right = sequence[i], sequence[i + 1]
                connections.append(
                    StratifiedConnection(
                        face,
                        left,
                        right,
                        pipe.source_height_m if i == 0 else pipe.diameter_m / 2,
                        pipe.target_height_m if i == pipe.cells else pipe.diameter_m / 2,
                        pipe.diameter_m,
                        0.0,
                        allow_reverse=pipe.allow_reverse if i == 0 else True,
                        maximum_pressure_drop_fraction=pipe.maximum_pressure_drop_fraction,
                        hydraulic_model="darcy_weisbach",
                        length_m=dx / 2 if i in (0, pipe.cells) else dx,
                        dynamic_viscosity_pa_s=pipe.dynamic_viscosity_pa_s,
                        roughness_m=pipe.roughness_m,
                        maximum_velocity_m_s=pipe.maximum_velocity_m_s,
                    )
                )
        self.network = StratifiedPRNetwork(compiled, connections, **integration)
        self.model_id = self.network.model_id
        self.cell_names = MappingProxyType(self.cell_names)
        self.face_names = MappingProxyType(self.face_names)
        self.terminals = MappingProxyType(self.terminals)
        self._definition_id = _digest(self.to_dict())

    def to_dict(self):
        integration = {
            k: v
            for k, v in self.network.to_dict().items()
            if k not in ("model", "vessels", "connections")
        }
        return {
            "model": "quasisteady_pr_linepack_v1",
            "vessels": {n: v.to_dict() for n, v in self.terminals.items()},
            "pipes": [p.to_dict() for p in self.pipes],
            "integration": integration,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if (
            set(data) != {"model", "vessels", "pipes", "integration"}
            or data["model"] != "quasisteady_pr_linepack_v1"
        ):
            raise ValueError("Unknown linepack model/fields")
        return cls(
            {n: StratifiedPRFlow.from_dict(v, device=device) for n, v in data["vessels"].items()},
            [HorizontalPRPipe.from_dict(p) for p in data["pipes"]],
            **data["integration"],
        )

    def initialize(self, terminals, pipe_initials):
        if set(terminals) != set(self.terminals) or set(pipe_initials) != set(self.cell_names):
            raise ValueError("Initial terminal/pipe registry differs")
        states = dict(terminals)
        for pipe in self.pipes:
            spec = pipe_initials[pipe.name]
            if set(spec) - {
                "pressure_pa",
                "temperature_k",
                "mole_fractions",
                "wall_temperature_k",
            } or not {"pressure_pa", "temperature_k", "mole_fractions"} <= set(spec):
                raise ValueError(
                    "Pipe initial state requires PT/composition and optional wall temperature"
                )

            def scalar(key, i, spec=spec, pipe=pipe):
                raw = spec[key]
                if isinstance(raw, (tuple, list)):
                    if len(raw) != pipe.cells:
                        raise ValueError("Initial axial scalar array differs from cell count")
                    raw = raw[i]
                return _number(raw, key, positive=True)

            for i, name in enumerate(self.cell_names[pipe.name]):
                v = self.network.vessels[name]
                p, t = scalar("pressure_pa", i), scalar("temperature_k", i)
                raw = spec["mole_fractions"]
                if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], (tuple, list)):
                    if len(raw) != pipe.cells:
                        raise ValueError("Initial axial composition array differs from cells")
                    raw = raw[i]
                fractions = [_number(x, "mole fraction", nonnegative=True) for x in raw]
                initial = v.column.initialize(p, t, fractions)
                column = StratifiedPRState.from_dict(initial["state"])
                tw = scalar("wall_temperature_k", i) if "wall_temperature_k" in spec else t
                states[name] = v.initialize(column, tw)
        return self.network.initialize(states)

    def inspect(self, state):
        self.network.inspect(state)
        if self.model_id != self.network.model_id or self._definition_id != _digest(self.to_dict()):
            raise ValueError("Linepack identity differs")

    def _inventory(self, state, pipe):
        states = dict(state.vessels)
        f = self.network.fluid
        columns = [states[n].column for n in self.cell_names[pipe.name]]
        n = [math.fsum(c.component_moles[i] for c in columns) for i in range(len(f.names))]
        wall = math.fsum(
            self.network.vessels[name].wall.energy(states[name].wall_temperature_k)
            for name in self.cell_names[pipe.name]
        )
        energy = math.fsum(c.total_energy_j for c in columns)
        return {
            "component_moles": n,
            "fluid_total_energy_j": energy,
            "wall_energy_j": wall,
            "total_energy_j": energy + wall,
            "mass_kg": math.fsum(nn * c.molar_mass_kg_mol for nn, c in zip(n, f.components)),
            "volume_m3": math.pi * pipe.diameter_m**2 / 4 * pipe.length_m,
        }

    def advance(self, state, duration_s, *, openings=None):
        self.inspect(state)
        report = self.network.advance(state, duration_s, openings=openings)
        final = StratifiedNetworkState.from_dict(report["state"])
        inventories = {}
        for pipe in self.pipes:
            old, new = self._inventory(state, pipe), self._inventory(final, pipe)
            incoming = [0.0] * len(new["component_moles"])
            outgoing = incoming.copy()
            ein = eout = ambient = 0.0
            for step in report["steps"]:
                transfers = {p["name"]: p for p in step["transfers"]}
                a, b = (
                    transfers[self.face_names[pipe.name][0]],
                    transfers[self.face_names[pipe.name][-1]],
                )
                dt = step["dt_s"]
                for i in range(len(incoming)):
                    incoming[i] += dt * a["component_rates_mol_s"][i]
                    outgoing[i] += dt * b["component_rates_mol_s"][i]
                ein += dt * a["total_energy_rate_w"]
                eout += dt * b["total_energy_rate_w"]
                for name in self.cell_names[pipe.name]:
                    w = self.network.vessels[name].wall
                    ambient += (
                        dt
                        * w.ambient_ua_w_k
                        * (
                            w.ambient_temperature_k
                            - step["state"]["vessels"][name]["wall_temperature_k"]
                        )
                    )
            nr = [
                b - a - input_n + output_n
                for a, b, input_n, output_n in zip(
                    old["component_moles"], new["component_moles"], incoming, outgoing
                )
            ]
            er = new["total_energy_j"] - old["total_energy_j"] - ein + eout - ambient
            if max(map(abs, nr)) > 1e-8 + 1e-9 * sum(new["component_moles"]) or abs(
                er
            ) > 1e-5 + 1e-9 * max(1, abs(ein), abs(eout), abs(ambient)):
                raise ConservationError("Finite pipe linepack ledger rejected")
            states = dict(final.vessels)
            cells = []
            for i, name in enumerate(self.cell_names[pipe.name]):
                column = states[name].column
                d = self.network.vessels[name]._data(column)
                v = self.network.vessels[name].column
                mean_p = (
                    float(sum((w * p).sum() for w, p in zip(d["weights"], d["pressures"])).cpu())
                    / v.geometry.volume_m3
                )
                cells.append(
                    {
                        "name": name,
                        "axial_center_m": (i + 0.5) * pipe.length_m / pipe.cells,
                        "mean_pressure_pa": mean_p,
                        "temperature_k": column.temperature_k,
                        "component_moles": column.component_moles,
                        "fluid_total_energy_j": column.total_energy_j,
                        "phase_count": len(column.phase_roots),
                    }
                )
            inventories[pipe.name] = {
                **new,
                "cells": cells,
                "source_component_transfer_mol": incoming,
                "target_component_transfer_mol": outgoing,
                "source_energy_transfer_j": ein,
                "target_energy_transfer_j": eout,
                "ambient_heat_j": ambient,
                "component_residuals_mol": nr,
                "energy_residual_j": er,
            }
        return {**report, "pipes": inventories}
