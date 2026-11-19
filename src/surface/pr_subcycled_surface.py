"""Conservative local pipe subcycling with controlled operator splitting.

Fast pipes retain their full finite inventories and their original physics.
Finite boundary vessels belong to both operators, with their updated stocks
passed between stages. The outer coarse/fine test measures splitting error.
"""

import inspect
import math
from dataclasses import replace

from .liquid import _number
from .pr_equilibrium_transport import PREquilibriumTransportNetwork, PREquilibriumTransportState
from .pr_gravity_transport import PRGravityTransportNetwork
from .pr_surface_control import PRControlledSurface
from .time_interval import split_step_s


def _transport_partition(original, names, links, *, fast=False, fraction=1.0, reservoir_ports=None):
    # Copy actual objects and the SAME fluid, not a serialized/reconstructed EOS.
    options = {
        name: getattr(original, name)
        for name in inspect.signature(PREquilibriumTransportNetwork).parameters
        if name not in ("fluid", "nodes", "links")
    }
    options["pipe_paths"] = None
    if reservoir_ports is not None:
        options["reservoir_ports"] = reservoir_ports
    if fast:
        for name in (
            "heat_links",
            "pumps",
            "compressors",
            "pump_connections",
            "restrictions",
            "drives",
        ):
            options[name] = ()
        for name in ("relative_tolerance", "temperature_tolerance_k", "velocity_tolerance_m_s"):
            options[name] *= fraction
    return PREquilibriumTransportNetwork(
        original.fluid, [n for n in original.nodes if n.name in names], links, **options
    )


class PRSubcycledSurface(PRControlledSurface):
    _share_probe_quadrature = False

    def __init__(self, surface, pipe_nodes, *, local_error_fraction=0.1, fast_reservoir_ports=None):
        if fast_reservoir_ports not in (None, "riemann", "stagnation_single_phase"):
            raise ValueError("Unknown explicit fast reservoir-port model")
        self.fast_reservoir_ports = fast_reservoir_ports
        if (
            type(surface) is not PRControlledSurface
            or type(surface.plant) is not PRGravityTransportNetwork
        ):
            raise TypeError("Subcycling requires a gravitational controlled surface")
        if surface.trips:
            raise ValueError("Subcycling of protection-event sensors is not yet supported")
        if (
            not isinstance(pipe_nodes, (list, tuple))
            or not pipe_nodes
            or len(set(pipe_nodes)) != len(pipe_nodes)
        ):
            raise ValueError("Declare unique nonempty fast pipe nodes")
        self.pipe_nodes = tuple(sorted(pipe_nodes))
        self.reference_surface = surface
        self.local_error_fraction = _number(
            local_error_fraction, "local error fraction", positive=True
        )
        if self.local_error_fraction > 0.25:
            raise ValueError("Local error budget must be at most one quarter of outer tolerances")
        plant, t = surface.plant, surface.plant.transport
        selected = set(self.pipe_nodes)
        nodes = {n.name: n for n in t.nodes}
        if selected - nodes.keys() or any(nodes[n].kind != "pipe_cell" for n in selected):
            raise ValueError("Fast nodes must be existing pipe cells")
        fast_links = [c for c in t.links if {c.source, c.target} & selected]
        fast_names = selected | {n for c in fast_links for n in (c.source, c.target)}
        boundaries = fast_names - selected
        if not fast_links or not boundaries or any(nodes[n].kind != "vessel" for n in boundaries):
            raise ValueError("Include contiguous fast pipes up to their finite vessel boundaries")
        if any(
            nodes[n].wall is not None
            or nodes[n].geometry is not None
            or nodes[n].droplet_population is not None
            for n in boundaries
        ):
            raise ValueError(
                "Fast boundaries must be plain stirred vessels without wall/population states"
            )
        references = [{p.node} for p in plant.ports]
        references += [{p.node} for p in t.pumps]
        references += [{p.card.start, p.card.end} for p in t.pump_connections]
        references += [
            {p.source, p.target}
            for p in (
                *t.heat_links,
                *t.compressors,
                *t.restrictions,
                *plant.heat_links,
                *plant.darcy_connections,
            )
        ]
        if any(r & selected for r in references):
            raise ValueError(
                "Fast pipe body participates in an external equipment/column/heat connection"
            )
        self.fast_links = frozenset(c.name for c in fast_links)
        if any(
            c.port in self.fast_links or getattr(c, "resistance_node", None) in selected
            for c in surface.channels
        ):
            raise ValueError("Actuated fast-pipe links require event-aware local control")
        controls = {
            name: getattr(surface, name)
            for name in inspect.signature(PRControlledSurface).parameters
            if name not in ("plant", "valves", "controllers")
        }
        super().__init__(plant, surface.valves, surface.controllers, **controls)
        self.fast = _transport_partition(
            t,
            fast_names,
            fast_links,
            fast=True,
            fraction=self.local_error_fraction,
            reservoir_ports=self.fast_reservoir_ports,
        )
        slow_transport = _transport_partition(
            t, set(t.names) - selected, [c for c in t.links if c.name not in self.fast_links]
        )
        slow_plant = PRGravityTransportNetwork(
            slow_transport,
            plant.columns,
            plant.ports,
            walls=plant.walls,
            maximum_transverse_head_fraction=plant.maximum_transverse_head_fraction,
            heat_links=plant.heat_links,
            darcy_connections=plant.darcy_connections,
        )
        self.slow = PRControlledSurface(slow_plant, surface.valves, surface.controllers, **controls)
        self.boundaries = tuple(sorted(boundaries))

    def to_dict(self):
        return {
            **PRControlledSurface.to_dict(self),
            "model": "pr_subcycled_surface_v1",
            "subcycled_pipe_nodes": list(self.pipe_nodes),
            "local_error_fraction": self.local_error_fraction,
            **(
                {"fast_reservoir_ports": self.fast_reservoir_ports}
                if self.fast_reservoir_ports is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "pr_subcycled_surface_v1":
            raise ValueError("Unknown subcycled surface model")
        nodes = data.pop("subcycled_pipe_nodes")
        fraction = data.pop("local_error_fraction", 0.1)
        ports = data.pop("fast_reservoir_ports", None)
        data["model"] = "pr_controlled_surface_v1"
        return cls(
            PRControlledSurface.from_dict(data, device=device),
            nodes,
            local_error_fraction=fraction,
            fast_reservoir_ports=ports,
        )

    def adopt(self, state):
        """Explicitly adopt an original checkpoint without modifying any stocks."""
        self.reference_surface.inspect(state)
        result = replace(state, model_id=self.model_id)
        self.inspect(result)
        return result

    def _commands(self, commands):
        plant = self.plant
        switches = {
            p.name
            for p in (
                *plant.transport.switches,
                *plant.ports,
                *plant.heat_links,
                *plant.darcy_connections,
            )
        }
        openings = commands.get("openings") or {}
        if set(openings) - switches or any(type(v) is not bool for v in openings.values()):
            raise ValueError("Known binary isolation commands required")
        names = set(plant.transport.names) | set(plant.columns)
        heat, utility = commands.get("heat_w") or {}, commands.get("utility_temperature_k") or {}
        if (set(heat) | set(utility)) - names:
            raise ValueError("Unknown thermal node")
        for n, value in heat.items():
            if _number(value, "heat rate") and n not in utility:
                raise ValueError("Heat requires utility temperature")
        for value in utility.values():
            _number(value, "utility temperature", positive=True)
        losses = commands.get("loss_coefficients") or {}
        if set(losses) - {n.name for n in plant.transport.nodes if n.kind == "pipe_cell"}:
            raise ValueError("Unknown resistance cell")
        for value in losses.values():
            _number(value, "loss coefficient", nonnegative=True)
        fast, slow = {}, dict(commands)
        for key, selected in (
            ("openings", self.fast_links),
            ("heat_w", self.pipe_nodes),
            ("utility_temperature_k", self.pipe_nodes),
            ("loss_coefficients", self.pipe_nodes),
        ):
            data = commands.get(key) or {}
            fast[key] = {k: v for k, v in data.items() if k in selected}
            slow[key] = {k: v for k, v in data.items() if k not in selected}
        return fast, slow

    def _fast_stage(self, state, clock, dt, commands):
        full = state.plant.transport
        initial = replace(
            full,
            nodes=tuple((n, s) for n, s in full.nodes if n in self.fast.names),
            time_s=clock,
            model_id=self.fast.model_id,
            rotors=(),
        )
        report = self.fast.advance(initial, dt, **commands)
        local = PREquilibriumTransportState.from_dict(report["state"])
        values = dict(local.nodes)
        merged = replace(
            full,
            nodes=tuple((n, values.get(n, s)) for n, s in full.nodes),
            time_s=local.time_s,
            next_step_s=local.next_step_s,
        )
        result = replace(state, plant=replace(state.plant, transport=merged))
        # Record the actual conservative exchange with every finite boundary.
        old = dict(initial.nodes)
        transfers = {
            n: {
                "component_moles": [
                    b - a for a, b in zip(old[n].component_moles, values[n].component_moles)
                ],
                "total_energy_j": values[n].total_energy_j - old[n].total_energy_j,
            }
            for n in self.boundaries
        }
        return result, {
            "operator": "fast",
            "initial": initial.to_dict(),
            "report": report,
            "boundary_transfers": transfers,
        }

    def _slow_stage(self, state, clock, dt, manual, modes, commands):
        full = state.plant.transport
        local_transport = replace(
            full,
            nodes=tuple((n, s) for n, s in full.nodes if n in self.slow.plant.transport.names),
            time_s=clock,
            model_id=self.slow.plant.transport.model_id,
        )
        initial = replace(
            state,
            plant=replace(
                state.plant, transport=local_transport, model_id=self.slow.plant.model_id
            ),
            model_id=self.slow.model_id,
        )
        local, report = self.slow._step(initial, dt, manual, modes, commands)
        values = dict(local.plant.transport.nodes)
        merged = replace(
            local.plant.transport,
            nodes=tuple((n, values.get(n, s)) for n, s in full.nodes),
            model_id=self.plant.transport.model_id,
            next_step_s=full.next_step_s,
        )
        result = replace(
            local,
            plant=replace(local.plant, transport=merged, model_id=self.plant.model_id),
            model_id=self.model_id,
        )
        return result, {"operator": "slow", "initial": initial.to_dict(), "report": report}

    def _error_openings(self, step):
        # Darcy links belong to the slow operator; include actual actuator
        # isolation, not merely the user's requested command dictionary.
        slow = step["local_subcycling"]["stages"][1]["report"]
        return self.slow._error_openings(slow)

    def _step(self, state, dt, manual, modes, commands, *, _probe_quadrature_order=None):
        if _probe_quadrature_order is not None:
            raise ValueError("Split stages require their own spatial qualification")
        fast_commands, slow_commands = self._commands(commands)
        first, second = split_step_s(state.time_s, dt)
        middle, a = self._fast_stage(state, state.time_s, first, fast_commands)
        middle, b = self._slow_stage(middle, state.time_s, dt, manual, modes, slow_commands)
        result, c = self._fast_stage(middle, state.time_s + first, second, fast_commands)
        self.inspect(result)
        stages = [a, b, c]
        heat = math.fsum(s["report"]["total_heat_j"] for s in stages)
        work = math.fsum(s["report"]["total_work_j"] for s in stages)
        residual = self.plant._ledger(state.plant, result.plant, heat + work)
        return result, {
            "dt_s": dt,
            "state": result.to_dict(),
            "total_heat_j": heat,
            "total_work_j": work,
            "local_subcycling": {
                "stages": stages,
                "conservation_residuals": residual,
                "slow_commands": slow_commands,
                "fast_commands": fast_commands,
            },
        }
