"""FIM adapter to one open, resolved gravitational pipe/equipment Newton."""

import math
from dataclasses import replace

from .pr_transport_boundary import PRTransportBoundary

try:
    from ..surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
    from ..surface.pr_open_transport import PRTransportFeed, _BoundaryProperties, _OpenCoupling
except ImportError:
    from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
    from surface.pr_open_transport import PRTransportFeed, _BoundaryProperties, _OpenCoupling


class PRGravityTransportBoundary(PRTransportBoundary):
    model_tag = "dynamic_pr_gravity_transport_fim_v1"
    scope = "Producer-only two-way FIM coupling to a single open PR pipe/equipment/gravitational-vessel graph. Finite apertures retain joint chemical-profile and quadrature closure. Supported stirred inlet/export nodes, low-dP one-way external ports, steady PVT gathering and explicit thermal characterization. No capillary branch, dynamic injection-well coupling or thermal reservoir inferred."

    def __init__(self, model, inlet_vessel):
        super().__init__(model, inlet_vessel)
        self.graph = model.graph

    def nodes(self, state):
        return dict(state.transport.nodes)

    def fluids(self):
        return self.model.fluids()

    def totals(self, state):
        n, energy = super().totals(state.transport)
        n = [
            math.fsum((n[j], *(s.component_moles[j] for _, s in state.columns)))
            for j in range(len(n))
        ]
        energy += math.fsum(s.total_energy_j for _, s in state.columns)
        energy += math.fsum(self.graph.walls[name].heat_capacity_j_k * t for name, t in state.walls)
        return n, energy

    def commands(self):
        return {
            **self.transport.default_openings,
            **{p.name: True for p in (*self.graph.ports, *self.graph.heat_links)},
        }

    def build(self, streams, height, outlet_openings, connection_openings):
        self.inlet_elevation(height)
        if set(outlet_openings) - self.outlet_names():
            raise ValueError("Unknown gravitational export command")
        if set(connection_openings) - set(self.commands()) or any(
            type(v) is not bool for v in connection_openings.values()
        ):
            raise ValueError("Known binary gravitational connection commands required")
        return PROpenGravityTransportNetwork(
            self.graph,
            [
                PRTransportFeed(f"fim_feed_{i}", self.inlet_name, stream)
                for i, stream in enumerate(streams)
            ],
            [
                replace(p, opening=outlet_openings.get(p.name, p.opening))
                for p in self.model.pressure_ports
            ],
        )

    def step(self, model, state, dt, connection_openings):
        model.inspect(state)
        commands = {**self.commands(), **connection_openings}
        material = _OpenCoupling(
            model,
            _BoundaryProperties(model, {p.name: p.opening for p in model.pressure_ports}),
            state.transport,
            dt,
            commands,
        )
        end, raw = self.graph._step(state, dt, commands, {}, {}, {}, _material=material)
        return end, self.flow_report(model, dt, raw)

    def flow_report(self, model, dt, raw):
        """Account an already accepted graph step without solving it again."""
        records = raw["columns"].get("material", raw["transport"].get("coupling"))["ports"]
        n = len(self.fluid.names)
        return {
            "dt_s": dt,
            "gravity_model": model.to_dict(),
            "gravity_report": raw,
            "feed_component_moles": [
                dt * math.fsum(v.stream.component_molar_rates_mol_s[j] for v in model.feeds)
                for j in range(n)
            ],
            "outlet_component_moles": [
                dt
                * math.fsum(
                    records[p.name]["component_rates_mol_s"][j] for p in model.pressure_ports
                )
                for j in range(n)
            ],
            "feed_energy_j": dt * math.fsum(v.stream.total_energy_rate_w for v in model.feeds),
            "outlet_energy_j": dt
            * math.fsum(records[p.name]["total_energy_rate_w"] for p in model.pressure_ports),
            "ambient_heat_j": raw["total_heat_j"],
            "external_energy_j": raw["total_work_j"],
            "entropy_production_j_k": raw["entropy_production_j_k"],
        }

    def temporal_error(self, coarse, fine, c, height):
        error = max(
            super().temporal_error(coarse.transport, fine.transport, c, height),
            self.graph._error(coarse, fine),
        )
        for (name, x), (_, y) in zip(coarse.columns, fine.columns):
            column = self.graph.columns[name]
            p, q = (column.sample(s, 0.0, side="below")["pressure_pa"] for s in (x, y))
            error = max(
                error,
                abs(p - q)
                / (c.temporal_pressure_tolerance_pa + c.temporal_relative_tolerance * max(p, q)),
            )
        return error
