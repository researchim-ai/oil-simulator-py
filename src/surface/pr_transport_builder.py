"""Compile named finite pipes into the same equilibrium transport graph.

Junctions and equipment vessels remain explicit finite control volumes.
No zero-volume mixing node or hidden pressure boundary is inserted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .liquid import _name, _number
from .pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PREquilibriumPipe:
    name: str
    source: str
    target: str
    length_m: float
    diameter_m: float
    cells: int
    dynamic_viscosity_pa_s: float = 0.0
    roughness_m: float = 0.0
    distribution_strength: float = 0.0
    buoyancy_relaxation_s: float = 0.0
    source_height_m: float | None = None
    target_height_m: float | None = None
    wall: VesselWall | None = None

    def __post_init__(self):
        for n in (self.name, self.source, self.target):
            _name(n)
        _number(self.length_m, "pipe length", positive=True)
        _number(self.diameter_m, "pipe diameter", positive=True)
        if (
            isinstance(self.cells, bool)
            or not isinstance(self.cells, int)
            or not 2 <= self.cells <= 128
        ):
            raise ValueError("Finite pipe requires 2..128 cells subject to joint Newton budget")
        for k in (
            "dynamic_viscosity_pa_s",
            "roughness_m",
            "distribution_strength",
            "buoyancy_relaxation_s",
        ):
            _number(getattr(self, k), k, nonnegative=True)
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected whole-pipe VesselWall")


def build_equilibrium_transport(fluid, vessels, pipes, *, links=(), **solver):
    vessels = tuple(vessels)
    pipes = tuple(pipes)
    by_name = {n.name: n for n in vessels}
    if len(by_name) != len(vessels) or any(n.kind != "vessel" for n in vessels):
        raise ValueError("Unique finite vessels required")
    if len({p.name for p in pipes}) != len(pipes):
        raise ValueError("Unique pipe names required")
    nodes = list(vessels)
    connections = list(links)
    paths = {}
    for pipe in pipes:
        if not isinstance(pipe, PREquilibriumPipe):
            raise TypeError("Expected PREquilibriumPipe")
        if pipe.source not in by_name or pipe.target not in by_name:
            raise ValueError("Unknown pipe terminal vessel")
        a, b = by_name[pipe.source], by_name[pipe.target]
        if abs(b.elevation_m - a.elevation_m) > pipe.length_m:
            raise ValueError("Pipe elevation difference exceeds length")
        names = []
        area = math.pi * pipe.diameter_m**2 / 4
        wall = None
        if pipe.wall is not None:
            data = asdict(pipe.wall)
            for key in ("heat_capacity_j_k", "fluid_ua_w_k", "ambient_ua_w_k"):
                data[key] /= pipe.cells
            wall = VesselWall(**data)
        for i in range(pipe.cells):
            name = f"{pipe.name}__{i:04d}"
            names.append(name)
            nodes.append(
                PREquilibriumNode(
                    name,
                    area * pipe.length_m / pipe.cells,
                    "pipe_cell",
                    elevation_m=a.elevation_m
                    + (b.elevation_m - a.elevation_m) * (i + 0.5) / pipe.cells,
                    length_m=pipe.length_m / pipe.cells,
                    diameter_m=pipe.diameter_m,
                    elevation_change_m=(b.elevation_m - a.elevation_m) / pipe.cells,
                    dynamic_viscosity_pa_s=pipe.dynamic_viscosity_pa_s,
                    roughness_m=pipe.roughness_m,
                    distribution_strength=pipe.distribution_strength,
                    buoyancy_relaxation_s=pipe.buoyancy_relaxation_s,
                    wall=wall,
                )
            )
        chain = [pipe.source, *names, pipe.target]
        for i, (source, target) in enumerate(zip(chain[:-1], chain[1:])):
            connections.append(
                PREquilibriumLink(
                    f"{pipe.name}__face_{i:04d}",
                    source,
                    target,
                    pipe.diameter_m,
                    source_height_m=pipe.source_height_m if i == 0 else None,
                    target_height_m=pipe.target_height_m if i == pipe.cells else None,
                )
            )
        paths[pipe.name] = tuple(names)
    return PREquilibriumTransportNetwork(fluid, nodes, connections, pipe_paths=paths, **solver)


def initialize_grouped(model, initial):
    paths = model.pipe_paths
    claimed = {n for names in paths.values() for n in names}
    expected = (set(model.names) - claimed) | set(paths)
    if set(initial) != expected:
        raise ValueError("Grouped initial states must cover every finite vessel and pipe")
    expanded = {n: spec for n, spec in initial.items() if n not in paths}
    for path, names in paths.items():
        spec = initial[path]
        for i, name in enumerate(names):
            row = {}
            for key, value in spec.items():
                if key == "mole_fractions":
                    axial = (
                        isinstance(value, (list, tuple))
                        and value
                        and isinstance(value[0], (list, tuple))
                    )
                else:
                    axial = isinstance(value, (list, tuple))
                if axial:
                    if len(value) != len(names):
                        raise ValueError("Pipe initial axial field size differs")
                    row[key] = value[i]
                else:
                    row[key] = value
            expanded[name] = row
    return model.initialize(expanded)
