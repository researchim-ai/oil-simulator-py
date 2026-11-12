"""Physical receiver/network boundary for the transactional FIM outer solve.

Rebinding changes only the command/feed model identity, never stored stocks,
profiles, wall temperatures or clocks. Each trial owns a separate model.
"""

import math
from contextlib import nullcontext
from dataclasses import replace

try:
    from ..surface.liquid import PropertyModelError
    from ..surface.stratified_pr_flow import StratifiedFeed, StratifiedPRFlow
    from ..surface.stratified_pr_network import StratifiedPRNetwork
except ImportError:
    from surface.liquid import PropertyModelError
    from surface.stratified_pr_flow import StratifiedFeed, StratifiedPRFlow
    from surface.stratified_pr_network import StratifiedPRNetwork


class PRSurfaceBoundary:
    def __init__(self, model, inlet_vessel=None):
        self.model, self.inlet_vessel = model, inlet_vessel
        self.network = type(model) is StratifiedPRNetwork
        if self.network:
            if inlet_vessel not in model.vessels:
                raise ValueError("Known inlet vessel required for dynamic network")
            self.vessels = model.vessels
        elif type(model) is StratifiedPRFlow and inlet_vessel is None:
            self.vessels = {"receiver": model}
        else:
            raise TypeError("Expected unforced PR flow vessel or named PR vessel network")
        if any(type(v) is not StratifiedPRFlow for v in self.vessels.values()):
            raise TypeError("Custom vessel physics needs an explicit boundary adapter")
        if any(v.feeds for v in self.vessels.values()):
            raise ValueError("FIM-bound surface model must not contain prescribed material feeds")
        self.inlet_name = inlet_vessel if self.network else "receiver"
        self.inlet = self.vessels[self.inlet_name].column
        self.fluid = self.inlet.fluid
        if len(self.outlet_names()) != sum(len(v.outlets) for v in self.vessels.values()):
            raise ValueError("Ambiguous qualified outlet names")

    @property
    def model_tag(self):
        return "dynamic_pr_vessel_network_fim_v1" if self.network else "dynamic_pr_receiver_fim_v1"

    @property
    def maximum_inlet_height_m(self):
        return self.inlet.geometry.height_m

    @property
    def gravity_m_s2(self):
        return self.inlet.gravity_m_s2

    def inlet_elevation(self, height):
        return self.inlet.elevation_m + height

    def temporal_error(self, coarse, fine, c, height):
        pc, pf = self.pressure(coarse, height), self.pressure(fine, height)
        error = abs(pc - pf) / (
            c.temporal_pressure_tolerance_pa + c.temporal_relative_tolerance * pf
        )
        cn, fn = self.nodes(coarse), self.nodes(fine)
        for name, cx in cn.items():
            fx = fn[name]
            column = self.vessels[name].column
            nc, nf = cx.column.component_moles, fx.column.component_moles
            scale = max(1.0, math.fsum(nf))
            pressure_c, pressure_f = (
                column.sample(cx.column, 0, side="below")["pressure_pa"],
                column.sample(fx.column, 0, side="below")["pressure_pa"],
            )
            local_error = max(
                abs(pressure_c - pressure_f)
                / (c.temporal_pressure_tolerance_pa + c.temporal_relative_tolerance * pressure_f),
                abs(cx.column.temperature_k - fx.column.temperature_k)
                / c.temporal_temperature_tolerance_k,
                abs(cx.wall_temperature_k - fx.wall_temperature_k)
                / c.temporal_temperature_tolerance_k,
                max(abs(a - b) for a, b in zip(nc, nf))
                / (1e-9 + c.temporal_relative_tolerance * scale),
                abs(cx.column.total_energy_j - fx.column.total_energy_j)
                / (
                    1e-5
                    + c.temporal_relative_tolerance
                    * max(
                        1.0,
                        abs(fx.column.total_energy_j),
                        scale * 8.314462618 * fx.column.temperature_k,
                    )
                ),
            )
            error = max(error, local_error)
            if len(cx.column.phase_roots) != len(fx.column.phase_roots):
                error = max(error, 2.0)
            else:
                for a, b in zip(cx.column.interface_heights_m, fx.column.interface_heights_m):
                    error = max(
                        error,
                        abs(a - b)
                        / (1e-8 + c.temporal_relative_tolerance * column.geometry.height_m),
                    )
        return error

    def transaction(self):
        return nullcontext()

    def fluids(self):
        # Equal EOS identities can be represented by distinct immutable
        # snapshots. Keep their AD/storage namespaces, but cover every cache.
        result = []
        seen = set()
        for fluid in [self.fluid, *[v.column.fluid for v in self.vessels.values()]]:
            if id(fluid) not in seen:
                seen.add(id(fluid))
                result.append(fluid)
        return result

    def nodes(self, state):
        return dict(state.vessels) if self.network else {"receiver": state}

    def temperature(self, state):
        return self.nodes(state)[self.inlet_name].column.temperature_k

    def pressure(self, state, height):
        return self.inlet.sample(self.nodes(state)[self.inlet_name].column, height, side="below")[
            "pressure_pa"
        ]

    def totals(self, state):
        nodes = self.nodes(state)
        n = [
            math.fsum(s.column.component_moles[j] for s in nodes.values())
            for j in range(len(self.fluid.names))
        ]
        energy = math.fsum(
            s.column.total_energy_j + self.vessels[name].wall.energy(s.wall_temperature_k)
            for name, s in nodes.items()
        )
        return n, energy

    def outlet_names(self):
        return {
            f"{n}/{p.name}" if self.network else p.name
            for n, v in self.vessels.items()
            for p in v.outlets
        }

    def build(self, streams, height, outlet_openings, connection_openings):
        if set(outlet_openings) - self.outlet_names():
            raise ValueError("Unknown receiver outlet command")
        if not self.network and connection_openings:
            raise ValueError("Connection commands require a PR vessel network")
        vessels = {}
        keys = (
            "initial_step_s",
            "minimum_step_s",
            "maximum_step_s",
            "temperature_tolerance_k",
            "relative_tolerance",
            "maximum_attempts",
        )
        for name, v in self.vessels.items():
            vessels[name] = StratifiedPRFlow(
                v.column,
                v.wall,
                outlets=[
                    replace(
                        p,
                        opening=outlet_openings.get(
                            f"{name}/{p.name}" if self.network else p.name, p.opening
                        ),
                    )
                    for p in v.outlets
                ],
                feeds=[StratifiedFeed(height, s) for s in streams]
                if name == self.inlet_name
                else (),
                **{k: getattr(v, k) for k in keys},
            )
        if not self.network:
            return vessels["receiver"]
        keys = (
            "initial_step_s",
            "minimum_step_s",
            "maximum_step_s",
            "relative_tolerance",
            "temperature_tolerance_k",
            "maximum_attempts",
        )
        model = StratifiedPRNetwork(
            vessels, self.model.connections, **{k: getattr(self.model, k) for k in keys}
        )
        model._commands(connection_openings)
        return model

    def rebind(self, state, target):
        if not self.network:
            return replace(state, model_id=target.model_id)
        return replace(
            state,
            model_id=target.model_id,
            vessels=tuple(
                (name, replace(s, model_id=target.vessels[name].model_id))
                for name, s in state.vessels
            ),
        )

    def step(self, model, state, dt, connection_openings):
        # Validate actual trial feeds against the old active support too.
        # Inspecting only the unforced template would miss species activation.
        model.inspect(state)
        if not self.network:
            return model._step(state, dt)
        end, raw = model._step(state, dt, model._commands(connection_openings))
        incoming, outgoing = raw["feed_component_moles"], raw["outlet_component_moles"]
        if any(v < 0 for v in outgoing):
            raise PropertyModelError("Network export components became negative")
        return end, {
            "dt_s": dt,
            "network_model": model.to_dict(),
            "network_report": raw,
            "feed_component_moles": incoming,
            "outlet_component_moles": outgoing,
            **{
                k: raw[k]
                for k in (
                    "feed_energy_j",
                    "outlet_energy_j",
                    "ambient_heat_j",
                    "entropy_production_j_k",
                )
            },
        }


def make_surface_boundary(model, inlet_vessel):
    try:
        from ..surface.pr_open_transport import PROpenTransportNetwork
    except ImportError:
        from surface.pr_open_transport import PROpenTransportNetwork
    try:
        from ..surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
    except ImportError:
        from surface.pr_open_gravity_transport import PROpenGravityTransportNetwork
    try:
        from ..surface.pr_capillary_surface_plant import PRCapillarySurfacePlant
    except ImportError:
        from surface.pr_capillary_surface_plant import PRCapillarySurfacePlant
    if type(model) is PRCapillarySurfacePlant:
        from .pr_capillary_surface_boundary import PRCapillarySurfaceBoundary

        return PRCapillarySurfaceBoundary(model, inlet_vessel)
    if type(model) is PROpenGravityTransportNetwork:
        from .pr_gravity_transport_boundary import PRGravityTransportBoundary

        return PRGravityTransportBoundary(model, inlet_vessel)
    if type(model) is PROpenTransportNetwork:
        from .pr_transport_boundary import PRTransportBoundary

        return PRTransportBoundary(model, inlet_vessel)
    return PRSurfaceBoundary(model, inlet_vessel)
