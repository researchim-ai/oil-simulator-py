"""Directed production tree with component mixing and full pressure sensitivities.

One fixed-pressure receiving node, merging flows, no split/reversal/zero-flow
holdup, horizontal/upward pipes. All sources share one isothermal PVT region.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from .blackoil_pipe import BlackOilPipe
from .blackoil_properties import BlackOilMixture
from .blackoil_valve import BlackOilLiquidValve
from .liquid import ConservationError, PropertyModelError, _name, _number
from .process import ComponentStream, StockDensities


@dataclass(frozen=True)
class ProductionNode:
    name: str
    elevation_m: float = 0.0
    pressure_pa: float | None = None

    def __post_init__(self):
        _name(self.name)
        _number(self.elevation_m, "elevation_m")
        if self.pressure_pa is not None:
            _number(self.pressure_pa, "pressure_pa", positive=True)


@dataclass(frozen=True)
class ProductionPipe:
    name: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    roughness_m: float = 0.0
    loss_coefficient: float = 0.0
    valve_opening: float = 1.0

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError("Production pipe cannot connect a node to itself")
        BlackOilPipe(
            self.name, self.length_m, self.diameter_m, 0, self.roughness_m, self.loss_coefficient
        )
        _number(self.valve_opening, "valve_opening", positive=True)
        if self.valve_opening > 1:
            raise ValueError("valve_opening must be in (0, 1]; closed branches need static holdup")
        if self.loss_coefficient == 0 and self.valve_opening != 1:
            raise ValueError("Controllable restriction requires positive loss_coefficient")


@dataclass(frozen=True)
class ProductionValve:
    name: str
    start: str
    end: str
    diameter_m: float
    loss_coefficient: float
    vapor_pressure_pa: float
    pressure_recovery_factor: float
    valve_opening: float = 1.0

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError("Production valve cannot connect a node to itself")
        BlackOilLiquidValve(
            self.name,
            self.diameter_m,
            self.loss_coefficient,
            self.vapor_pressure_pa,
            self.pressure_recovery_factor,
        )
        if not 0 < _number(self.valve_opening, "valve_opening", positive=True) <= 1:
            raise ValueError("valve_opening must be in (0,1]; static holdup is unsupported")


class BlackOilProductionNetwork:
    def __init__(self, model, nodes, pipes, valves=()):
        if not isinstance(model, BlackOilMixture):
            raise TypeError("Expected BlackOilMixture")
        self.model = model
        self.nodes = tuple(nodes)
        pipe_items, valve_items = tuple(pipes), tuple(valves)
        if not all(isinstance(p, ProductionPipe) for p in pipe_items):
            raise TypeError("Expected production pipes")
        if not all(isinstance(v, ProductionValve) for v in valve_items):
            raise TypeError("Expected production valves")
        self.pipes = pipe_items
        self.valves = valve_items
        self.edges = (*pipe_items, *valve_items)
        if not self.nodes or not all(isinstance(n, ProductionNode) for n in self.nodes):
            raise ValueError("Expected production nodes")
        if not self.edges or not all(
            isinstance(p, (ProductionPipe, ProductionValve)) for p in self.edges
        ):
            raise ValueError("Expected production pipes")
        self.by_name = {n.name: n for n in self.nodes}
        if len(self.by_name) != len(self.nodes) or len({p.name for p in self.edges}) != len(
            self.edges
        ):
            raise ValueError("Duplicate production node or pipe names")
        self.outgoing = {}
        self.incoming = {n.name: [] for n in self.nodes}
        self.models = {}
        for pipe in self.edges:
            if pipe.start not in self.by_name or pipe.end not in self.by_name:
                raise ValueError("Unknown production pipe node")
            if pipe.start in self.outgoing:
                raise ValueError("Production tree cannot split flow")
            self.outgoing[pipe.start] = pipe
            self.incoming[pipe.end].append(pipe.start)
            dz = self.by_name[pipe.end].elevation_m - self.by_name[pipe.start].elevation_m
            if isinstance(pipe, ProductionValve):
                if abs(dz) > 1e-9:
                    raise PropertyModelError("Localized valve ports must have equal elevation")
                self.models[pipe.name] = BlackOilLiquidValve(
                    pipe.name,
                    pipe.diameter_m,
                    pipe.loss_coefficient / pipe.valve_opening**2,
                    pipe.vapor_pressure_pa,
                    pipe.pressure_recovery_factor,
                )
            else:
                self.models[pipe.name] = BlackOilPipe(
                    pipe.name,
                    pipe.length_m,
                    pipe.diameter_m,
                    dz,
                    pipe.roughness_m,
                    pipe.loss_coefficient / pipe.valve_opening**2,
                )
        roots = [n.name for n in self.nodes if n.name not in self.outgoing]
        if len(roots) != 1:
            raise ValueError("Production tree requires exactly one receiving node")
        self.root = roots[0]
        if self.by_name[self.root].pressure_pa is None or any(
            n.pressure_pa is not None and n.name != self.root for n in self.nodes
        ):
            raise ValueError("Only receiving node must have fixed pressure")
        model.validate(self.by_name[self.root].pressure_pa, [0.0, 1.0, 0.0])
        degree = {n: len(children) for n, children in self.incoming.items()}
        queue = deque(n for n, d in degree.items() if d == 0)
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            if node in self.outgoing:
                parent = self.outgoing[node].end
                degree[parent] -= 1
                if degree[parent] == 0:
                    queue.append(parent)
        if len(order) != len(self.nodes):
            raise ValueError("Production tree contains a cycle or disconnected cycle")
        self.order = tuple(order)

    def to_dict(self):
        from dataclasses import asdict

        pvt = self.model.pvt
        data = {
            "pressure_MPa": pvt.P.tolist(),
            "Bo": pvt.Bo.tolist(),
            "Bw": pvt.Bw.tolist(),
            "Bg": pvt.Bg.tolist(),
            "mu_o_cP": pvt.mu_o.tolist(),
            "mu_w_cP": pvt.mu_w.tolist(),
            "mu_g_cP": pvt.mu_g.tolist(),
            "Rs_m3m3": pvt.Rs.tolist(),
            "Rv_m3m3": pvt.Rv.tolist(),
        }
        if len(pvt.oil_rs):
            data["extended"] = {
                "oil": {
                    "rs_values": pvt.oil_rs.tolist(),
                    "tables": {
                        str(rs): {key: value.tolist() for key, value in table.items()}
                        for rs, table in pvt.oil_tables.items()
                    },
                }
            }
        return {
            "model": "homogeneous_blackoil_tree",
            "pvt": data,
            "stock_densities": asdict(self.model.densities),
            "nodes": [asdict(n) for n in self.nodes],
            "pipes": [asdict(p) for p in self.edges if isinstance(p, ProductionPipe)],
            "valves": [asdict(p) for p in self.edges if isinstance(p, ProductionValve)],
        }

    @classmethod
    def from_dict(cls, spec):
        allowed = {"model", "pvt", "stock_densities", "nodes", "pipes", "valves"}
        if set(spec) - allowed or spec.get("model") != "homogeneous_blackoil_tree":
            raise ValueError("Expected homogeneous_blackoil_tree specification")
        model = BlackOilMixture(spec["pvt"], StockDensities(**spec["stock_densities"]))
        return cls(
            model,
            [ProductionNode(**n) for n in spec["nodes"]],
            [ProductionPipe(**p) for p in spec["pipes"]],
            [ProductionValve(**v) for v in spec.get("valves", [])],
        )

    def with_controls(self, valve_openings=None):
        """Independent effective tree; changing a control never mutates a live solve.

        Pipes distribute K/opening² along their length. ProductionValve uses
        a separate liquid-only local loss with strict phase/safety checks.
        """
        openings = {} if valve_openings is None else valve_openings
        if not isinstance(openings, dict):
            raise TypeError("valve_openings must map controllable branch names to openings")
        controlled = {p.name for p in self.edges if p.loss_coefficient > 0}
        if set(openings) - controlled:
            raise ValueError("Unknown controllable production restriction")
        return type(self)(
            self.model,
            self.nodes,
            [
                replace(p, valve_opening=openings.get(p.name, p.valve_opening))
                for p in self.edges
                if isinstance(p, ProductionPipe)
            ],
            [
                replace(p, valve_opening=openings.get(p.name, p.valve_opening))
                for p in self.edges
                if isinstance(p, ProductionValve)
            ],
        )

    def solve(self, sources):
        if not isinstance(sources, dict) or set(sources) - set(self.by_name):
            raise ValueError("Sources must map known production nodes to ComponentStreams")
        if not all(isinstance(s, ComponentStream) for s in sources.values()):
            raise TypeError("Expected ComponentStream sources")
        source_names = tuple(n.name for n in self.nodes if n.name in sources)
        count = len(source_names)
        masses = {}
        members = {}
        for node in self.order:
            own = sources.get(node, ComponentStream()).mass_rates(self.model.densities)
            masses[node] = np.array(
                [
                    math.fsum([own[c], *(masses[ch][i] for ch in self.incoming[node])])
                    for i, c in enumerate(("oil", "water", "gas"))
                ]
            )
            members[node] = np.array([float(n == node) for n in source_names])
            for child in self.incoming[node]:
                members[node] += members[child]
            if node != self.root and masses[node].sum() == 0:
                raise PropertyModelError(
                    f"Zero-flow branch at {node}: static phase holdup is not defined"
                )
        pressure = {self.root: self.by_name[self.root].pressure_pa}
        jac = {self.root: np.zeros((count, 3))}
        outlet_derivative = {self.root: 1.0}
        results = {}
        valve_names = tuple(p.name for p in self.edges if p.loss_coefficient > 0)
        control_jac = {self.root: np.zeros(len(valve_names))}
        for node in reversed(self.order):
            if node == self.root:
                continue
            pipe = self.outgoing[node]
            result = self.models[pipe.name].required_inlet(
                pressure[pipe.end], masses[node], self.model
            )
            pressure[node] = result["inlet_pressure_pa"]
            d = result["pressure_derivatives"]
            local = np.array([d[c] for c in ("oil_kg_s", "water_kg_s", "gas_kg_s")])
            jac[node] = (
                d["boundary_pressure_pa"] * jac[pipe.end] + members[node][:, None] * local[None, :]
            )
            outlet_derivative[node] = d["boundary_pressure_pa"] * outlet_derivative[pipe.end]
            control_jac[node] = d["boundary_pressure_pa"] * control_jac[pipe.end]
            if pipe.name in valve_names:
                control_jac[node][valve_names.index(pipe.name)] += (
                    d["loss_coefficient"] * -2 * pipe.loss_coefficient / pipe.valve_opening**3
                )
            results[pipe.name] = result
        residuals = {}
        for node in self.order:
            own = sources.get(node, ComponentStream()).mass_rates(self.model.densities)
            withdrawn = masses[node]
            residuals[node] = {
                c: math.fsum(
                    [own[c], *(masses[ch][i] for ch in self.incoming[node]), -float(withdrawn[i])]
                )
                for i, c in enumerate(("oil", "water", "gas"))
            }
        scale = max(1.0, float(masses[self.root].sum()))
        if any(abs(x) > 1e-12 * scale for r in residuals.values() for x in r.values()):
            raise ConservationError("Production network component imbalance")
        total = ComponentStream.from_mass_rates(*masses[self.root], self.model.densities)
        return {
            "model": "homogeneous_blackoil_tree",
            "receiving_node": self.root,
            "node_pressures_pa": pressure,
            "pipes": {p.name: results[p.name] for p in self.pipes},
            "valves": {v.name: results[v.name] for v in self.valves},
            "receiving_component_mass_rates_kg_s": dict(
                zip(("oil", "water", "gas"), masses[self.root].tolist())
            ),
            "receiving_stream": {
                "oil_m3_s": total.oil_m3_s,
                "water_m3_s": total.water_m3_s,
                "gas_m3_s": total.gas_m3_s,
            },
            "node_component_mass_residuals_kg_s": residuals,
            "pressure_derivatives_per_kg_s": {
                n: {
                    source: dict(zip(("oil", "water", "gas"), row.tolist()))
                    for source, row in zip(source_names, jac[n])
                }
                for n in self.order
            },
            "pressure_derivatives_receiving_pressure": outlet_derivative,
            "operating_controls": {
                "valve_openings": {
                    p.name: p.valve_opening for p in self.edges if p.loss_coefficient > 0
                }
            },
            "pressure_derivatives_valve_opening": {
                n: dict(zip(valve_names, control_jac[n].tolist())) for n in self.order
            },
            "friction_dissipation_w": math.fsum(
                r["friction_dissipation_w"] for r in results.values()
            ),
            "mechanical_balance_error_w": math.fsum(
                r["mechanical_balance_error_w"] for r in results.values()
            ),
        }
