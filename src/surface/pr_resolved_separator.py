"""Axially resolved separating vessels assembled into the common PR graph.

Local phase equilibrium is retained, but redistribution between axial cells
takes finite time. The prescribed linear terminal-drag response is an explicit
constitutive approximation, not an experimental equipment efficiency or a
droplet population balance. No stand-alone separator time integrator exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .liquid import _name, _number
from .pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PRResolvedSeparator:
    name: str
    diameter_m: float
    height_m: float
    cells: int
    terminal_response_s: float
    bottom_elevation_m: float = 0.0
    dynamic_viscosity_pa_s: float = 0.0
    wall: VesselWall | None = None

    def __post_init__(self):
        _name(self.name)
        for key in ("diameter_m", "height_m", "terminal_response_s"):
            _number(getattr(self, key), key, positive=True)
        _number(self.bottom_elevation_m, "bottom elevation")
        _number(self.dynamic_viscosity_pa_s, "mixture viscosity", nonnegative=True)
        if type(self.cells) is not int or not 2 <= self.cells <= 128:
            raise ValueError("Resolved vessel needs 2..128 axial cells")
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected a whole-vessel VesselWall")

    @property
    def cell_height_m(self):
        return self.height_m / self.cells

    def cell_name(self, index):
        if type(index) is not int or not 0 <= index < self.cells:
            raise ValueError("Separator cell index outside mesh")
        return f"{self.name}__{index:04d}"

    def cell_elevation_m(self, index):
        self.cell_name(index)
        return self.bottom_elevation_m + (index + 0.5) * self.cell_height_m

    def build(self, fluid, *, gravity_m_s2=9.80665, **solver):
        _number(gravity_m_s2, "settling gravity", positive=True)
        length = self.cell_height_m
        wall = self.wall
        if wall is not None:
            wall = replace(
                wall,
                heat_capacity_j_k=wall.heat_capacity_j_k / self.cells,
                fluid_ua_w_k=wall.fluid_ua_w_k / self.cells,
                ambient_ua_w_k=wall.ambient_ua_w_k / self.cells,
            )
        nodes = [
            PREquilibriumNode(
                self.cell_name(i),
                math.pi * self.diameter_m**2 / 4 * length,
                "pipe_cell",
                elevation_m=self.cell_elevation_m(i),
                length_m=length,
                diameter_m=self.diameter_m,
                elevation_change_m=length,
                dynamic_viscosity_pa_s=self.dynamic_viscosity_pa_s,
                buoyancy_relaxation_s=self.terminal_response_s,
                wall=wall,
            )
            for i in range(self.cells)
        ]
        links = [
            PREquilibriumLink(
                f"{self.name}__face_{i:04d}",
                self.cell_name(i - 1),
                self.cell_name(i),
                self.diameter_m,
            )
            for i in range(1, self.cells)
        ]
        return PREquilibriumTransportNetwork(
            fluid,
            nodes,
            links,
            gravity_m_s2=gravity_m_s2,
            pipe_paths={self.name: tuple(n.name for n in nodes)},
            **solver,
        )


@dataclass(frozen=True)
class PRLateralNozzle:
    """Finite circular pickup in a cylindrical cell; receiver is a finite body.

    The receiver must have its physical opening at the pickup elevation.
    A nozzle pipe may instead occupy its free left/right face. Two lateral
    axes must be joined through an explicit finite mixing volume.
    """

    name: str
    cell: str
    receiver: str
    diameter_m: float
    height_m: float
    receiver_port: str = "left"

    def __post_init__(self):
        for key in ("name", "cell", "receiver"):
            _name(getattr(self, key))
        _number(self.diameter_m, "nozzle diameter", positive=True)
        _number(self.height_m, "cell-local nozzle height", nonnegative=True)
        if self.receiver_port not in ("left", "right"):
            raise ValueError("A lateral nozzle needs an axial or mixed-volume receiver")

    def link(self):
        return PREquilibriumLink(
            self.name,
            self.cell,
            self.receiver,
            self.diameter_m,
            source_port=f"side:{self.name}",
            target_port=self.receiver_port,
            source_height_m=self.height_m,
        )


def join_pr_transport_graphs(
    graphs, *, links=(), nozzles=(), heat_links=(), pumps=(), compressors=()
):
    """Flatten compatible physical blocks into one Newton and one checkpoint.

    Solver policy, EOS, gravity and energy datum must agree exactly. There is
    no sequential pressure overwrite or hidden boundary stock at a junction.
    """
    graphs = tuple(graphs)
    if not graphs or any(not isinstance(g, PREquilibriumTransportNetwork) for g in graphs):
        raise TypeError("At least one PR transport graph required")
    topology = {
        "model",
        "fluid",
        "nodes",
        "links",
        "heat_links",
        "pumps",
        "compressors",
        "pipe_paths",
    }
    first = graphs[0]
    options = {k: v for k, v in first.to_dict().items() if k not in topology}
    for graph in graphs[1:]:
        if (
            graph.fluid.model_id != first.fluid.model_id
            or graph.fluid.device != first.fluid.device
            or {k: v for k, v in graph.to_dict().items() if k not in topology} != options
        ):
            raise ValueError("Joined graphs require identical EOS/device/gravity/solver policy")
    paths = {}
    for graph in graphs:
        if paths.keys() & graph.pipe_paths.keys():
            raise ValueError("Duplicate grouped physical body")
        paths.update(graph.pipe_paths)
    return PREquilibriumTransportNetwork(
        first.fluid,
        [n for g in graphs for n in g.nodes],
        [*(c for g in graphs for c in g.links), *links, *(n.link() for n in nozzles)],
        heat_links=[*(c for g in graphs for c in g.heat_links), *heat_links],
        pumps=[*(p for g in graphs for p in g.pumps), *pumps],
        compressors=[*(p for g in graphs for p in g.compressors), *compressors],
        pipe_paths=paths,
        **options,
    )
