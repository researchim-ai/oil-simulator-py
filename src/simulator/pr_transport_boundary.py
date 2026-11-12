"""Conservative physical adapter from accepted FIM streams to dynamic PR pipes."""

import math
from contextlib import nullcontext
from dataclasses import replace

try:
    from ..surface.pr_open_transport import (
        PROpenTransportNetwork,
        PRTransportFeed,
        _BoundaryProperties,
        _OpenCoupling,
    )
except ImportError:
    from surface.pr_open_transport import (
        PROpenTransportNetwork,
        PRTransportFeed,
        _BoundaryProperties,
        _OpenCoupling,
    )


class PRTransportBoundary:
    network = True
    model_tag = "dynamic_pr_transport_fim_v1"
    maximum_inlet_height_m = 0.0
    scope = "Producer-only two-way FIM pressure coupling to distributed implicit PR pipes and equipment, with finite supported inlet/export vessels. Steady PVT well gathering, explicit thermal characterization; no thermal reservoir, dynamic injection-well coupling or resolved gravitational separator inlet. External exports use empirical low-dP one-way pressure characteristics."

    def __init__(self, model, inlet_vessel):
        self.model, self.inlet_name = model, inlet_vessel
        self.transport, self.fluid = model.transport, model.fluid
        if model.feeds:
            raise ValueError("FIM-bound distributed model must not prescribe material feeds")
        if inlet_vessel not in self.transport.names:
            raise ValueError("Known distributed inlet vessel required")
        self.inlet = self.transport.nodes[self.transport.names.index(inlet_vessel)]
        if self.inlet.kind != "vessel" or self.inlet.geometry is not None:
            raise ValueError("Distributed inlet requires a supported stirred vessel")
        if any(p.allow_reverse for p in model.pressure_ports):
            raise ValueError("FIM material accounting currently requires one-way surface exports")
        # Stocks/packets use one explicit absolute gravitational energy datum.
        if self.transport.reference_elevation_m != 0.0:
            raise ValueError("FIM energy boundary requires absolute elevation datum zero")

    @property
    def gravity_m_s2(self):
        return self.transport.gravity_m_s2

    def inlet_elevation(self, height):
        if height != 0.0:
            raise ValueError("Stirred inlet height must be zero")
        return self.inlet.elevation_m

    def transaction(self):
        return nullcontext()

    def fluids(self):
        return [self.fluid]

    def nodes(self, state):
        return dict(state.nodes)

    def temperature(self, state):
        return self.nodes(state)[self.inlet_name].temperature_k

    def pressure(self, state, height):
        self.inlet_elevation(height)
        return self.nodes(state)[self.inlet_name].pressure_pa

    def totals(self, state):
        n = [
            math.fsum(v.component_moles[j] for _, v in state.nodes)
            for j in range(len(self.fluid.names))
        ]
        energy = math.fsum(v.total_energy_j for _, v in state.nodes) + self.transport.rotor_energy(
            state
        )
        energy += math.fsum(
            node.wall.heat_capacity_j_k * v.wall_temperature_k
            for node, (_, v) in zip(self.transport.nodes, state.nodes)
            if node.wall
        )
        return n, energy

    def outlet_names(self):
        return {p.name for p in self.model.pressure_ports}

    def build(self, streams, height, outlet_openings, connection_openings):
        self.inlet_elevation(height)
        if set(outlet_openings) - self.outlet_names():
            raise ValueError("Unknown distributed export command")
        if set(connection_openings) - set(self.transport.default_openings) or any(
            type(v) is not bool for v in connection_openings.values()
        ):
            raise ValueError("Known binary distributed connection commands required")
        return PROpenTransportNetwork(
            self.transport,
            [
                PRTransportFeed(f"fim_feed_{i}", self.inlet_name, stream)
                for i, stream in enumerate(streams)
            ],
            [
                replace(p, opening=outlet_openings.get(p.name, p.opening))
                for p in self.model.pressure_ports
            ],
        )

    def rebind(self, state, target):
        # Boundary commands/streams are outside physical graph identity;
        # pipe states, rotor speeds and graph identity stay unchanged.
        return state

    def step(self, model, state, dt, connection_openings):
        t = self.transport
        t.inspect(state)
        commands = {**t.default_openings, **connection_openings}
        ports = {p.name: p.opening for p in model.pressure_ports}
        coupling = _OpenCoupling(model, _BoundaryProperties(model, ports), state, dt, commands)
        end, raw = t._step(
            state,
            dt,
            commands,
            [0.0] * len(t.names),
            [s.temperature_k for _, s in state.nodes],
            _coupling=coupling,
        )
        n = len(self.fluid.names)
        records = raw["coupling"]["ports"]
        feed = [
            dt * math.fsum(v.stream.component_molar_rates_mol_s[j] for v in model.feeds)
            for j in range(n)
        ]
        export = [
            dt
            * math.fsum(records[p.name]["component_rates_mol_s"][j] for p in model.pressure_ports)
            for j in range(n)
        ]
        return end, {
            "dt_s": dt,
            "transport_model": model.to_dict(),
            "transport_report": raw,
            "feed_component_moles": feed,
            "outlet_component_moles": export,
            "feed_energy_j": dt * math.fsum(v.stream.total_energy_rate_w for v in model.feeds),
            "outlet_energy_j": dt
            * math.fsum(records[p.name]["total_energy_rate_w"] for p in model.pressure_ports),
            "ambient_heat_j": raw["ambient_heat_j"],
            "external_energy_j": raw["external_work_j"] + raw["external_heat_j"],
            "entropy_production_j_k": raw["entropy_production_j_k"],
        }

    def temporal_error(self, coarse, fine, c, height):
        error = 0.0
        for (name, x), (_, y) in zip(coarse.nodes, fine.nodes):
            scale = max(1.0, math.fsum(y.component_moles))
            error = max(
                error,
                abs(x.pressure_pa - y.pressure_pa)
                / (
                    c.temporal_pressure_tolerance_pa
                    + c.temporal_relative_tolerance * max(x.pressure_pa, y.pressure_pa)
                ),
                abs(x.temperature_k - y.temperature_k) / c.temporal_temperature_tolerance_k,
                max(abs(a - b) for a, b in zip(x.component_moles, y.component_moles))
                / (1e-9 + c.temporal_relative_tolerance * scale),
                abs(x.total_energy_j - y.total_energy_j)
                / (
                    1e-5
                    + c.temporal_relative_tolerance
                    * max(1.0, abs(y.total_energy_j), scale * 8.314462618 * y.temperature_k)
                ),
            )
            node = self.transport.nodes[self.transport.names.index(name)]
            mass = math.fsum(
                n * v.molar_mass_kg_mol for n, v in zip(y.component_moles, self.fluid.components)
            )
            error = max(
                error,
                abs(x.momentum_kg_m_s - y.momentum_kg_m_s)
                / (1e-9 + mass * self.transport.velocity_tolerance_m_s),
            )
            if node.wall:
                error = max(
                    error,
                    abs(x.wall_temperature_k - y.wall_temperature_k)
                    / c.temporal_temperature_tolerance_k,
                )
            if len(x.phase_roots) != len(y.phase_roots):
                error = max(error, 2.0)
        # Retain drive-speed and all native transport error controls as well.
        return max(error, self.transport._error(coarse, fine))
