"""Open distributed pipes/equipment and resolved gravitational vessel apertures.

Material boundaries act on physical stirred nodes; internal column apertures
retain their complete common Newton/profile/quadrature solve. No sequential
mixing step, fictitious buffer, or duplicate boundary energy is introduced.
"""

from contextlib import ExitStack

from .equilibrium_cache import cached_flashes
from .liquid import PropertyModelError, _number
from .operation import _digest
from .pr_gravity_transport import PRGravityTransportNetwork
from .pr_open_transport import (
    PROpenTransportNetwork,
    PRTransportFeed,
    PRTransportPressurePort,
    _BoundaryProperties,
    _OpenCoupling,
)


class PROpenGravityTransportNetwork(PROpenTransportNetwork):
    def __init__(self, graph, feeds=(), pressure_ports=()):
        if type(graph) is not PRGravityTransportNetwork:
            raise TypeError("Expected explicit resolved gravitational transport graph")
        self.graph = graph
        super().__init__(graph.transport, feeds, pressure_ports)
        reserved = {p.name for p in (*graph.ports, *graph.heat_links)}
        if reserved & {p.name for p in (*self.feeds, *self.pressure_ports)}:
            raise ValueError("Material and gravitational port names must be distinct")

    def to_dict(self):
        base = super().to_dict()
        return {
            "model": "pr_open_gravity_transport_v1",
            "graph": self.graph.to_dict(),
            "feeds": base["feeds"],
            "pressure_ports": base["pressure_ports"],
        }

    @classmethod
    def from_dict(cls, raw, *, device="cpu"):
        if (
            set(raw) != {"model", "graph", "feeds", "pressure_ports"}
            or raw["model"] != "pr_open_gravity_transport_v1"
        ):
            raise ValueError("Unknown open gravitational graph")
        from .component_energy import ComponentEnergyStream

        return cls(
            PRGravityTransportNetwork.from_dict(raw["graph"], device=device),
            [
                PRTransportFeed(p["name"], p["node"], ComponentEnergyStream(**p["stream"]))
                for p in raw["feeds"]
            ],
            [PRTransportPressurePort(**p) for p in raw["pressure_ports"]],
        )

    def inspect(self, state):
        if _digest(self.to_dict()) != self.model_id:
            raise PropertyModelError("Open gravitational model changed")
        self.graph.inspect(state)

    def fluids(self):
        result = [self.fluid]
        for column in self.graph.columns.values():
            if all(column.fluid is not f for f in result):
                result.append(column.fluid)
        return result

    def _commands(self, port_openings):
        commands = {p.name: p.opening for p in self.pressure_ports}
        if set(port_openings or {}) - set(commands):
            raise ValueError("Unknown external port command")
        for name, value in (port_openings or {}).items():
            if not 0 <= _number(value, "port opening") <= 1:
                raise ValueError("Port opening outside [0,1]")
            commands[name] = value
        return commands

    def advance(self, state, duration_s, *, port_openings=None, **controls):
        self.inspect(state)
        if "_material_factory" in controls:
            raise ValueError("Material coupling factory belongs to the open graph")
        commands = self._commands(port_openings)
        with ExitStack() as stack:
            for fluid in self.fluids():
                stack.enter_context(cached_flashes(fluid))
            prepared = _BoundaryProperties(self, commands)
            result = self.graph.advance(
                state,
                duration_s,
                _material_factory=lambda s, dt, o: _OpenCoupling(
                    self, prepared, s.transport, dt, o
                ),
                **controls,
            )
        result.update(
            {
                "open_model_id": self.model_id,
                "port_openings": commands,
                "boundary_component_moles": result["material_component_moles"],
                "boundary_energy_j": result["material_energy_j"],
                "scope": "Open implicit PR pipe/equipment graph with resolved finite gravitational apertures; empirical low-dP external pressure ports on supported stirred nodes. No capillary interface or thermal reservoir inferred.",
            }
        )
        return result
