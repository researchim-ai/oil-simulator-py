"""Compile named finite pipes into the same equilibrium transport graph.

Junctions and equipment vessels remain explicit finite control volumes.
No zero-volume mixing node or hidden pressure boundary is inserted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .dispersed_liquid_drift import DispersedOilWaterClosure
from .drift_flux_closure import (
    RELAP7EquilibriumTransition,
    RELAP7VerticalDriftClosure,
    ZuberFindlayClosure,
)
from .holmes_drift import HolmesGasLiquidClosure
from .liquid import _name, _number
from .pr_equilibrium_transport import (
    PREquilibriumLink,
    PREquilibriumNode,
    PREquilibriumTransportNetwork,
)
from .pr_terrain_pipe import PRTerrainPipe
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
    drift_flux: (
        ZuberFindlayClosure
        | RELAP7VerticalDriftClosure
        | RELAP7EquilibriumTransition
        | HolmesGasLiquidClosure
        | DispersedOilWaterClosure
        | None
    ) = None

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
        if self.drift_flux is not None:
            if not isinstance(
                self.drift_flux,
                (
                    ZuberFindlayClosure,
                    RELAP7VerticalDriftClosure,
                    RELAP7EquilibriumTransition,
                    HolmesGasLiquidClosure,
                    DispersedOilWaterClosure,
                ),
            ):
                raise TypeError("Expected a supported drift-flux closure")
            if self.distribution_strength or self.buoyancy_relaxation_s:
                raise ValueError("Select one pipe slip closure")


@dataclass(frozen=True)
class PREquilibriumNozzle:
    """Resolved quasi-1-D smooth contraction/expansion, with fixed walls."""

    name: str
    source: str
    target: str
    face_diameters_m: tuple
    cell_lengths_m: tuple
    dynamic_viscosity_pa_s: float = 0.0
    roughness_m: float = 0.0
    source_height_m: float | None = None
    target_height_m: float | None = None
    wall: VesselWall | None = None

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        object.__setattr__(self, "face_diameters_m", tuple(self.face_diameters_m))
        object.__setattr__(self, "cell_lengths_m", tuple(self.cell_lengths_m))
        if (
            not 2 <= len(self.cell_lengths_m) <= 1024
            or len(self.face_diameters_m) != len(self.cell_lengths_m) + 1
        ):
            raise ValueError("Nozzle requires N lengths and N+1 face diameters")
        for value in self.face_diameters_m + self.cell_lengths_m:
            _number(value, "nozzle dimension", positive=True)
        _number(self.dynamic_viscosity_pa_s, "viscosity", nonnegative=True)
        _number(self.roughness_m, "roughness", nonnegative=True)
        if self.roughness_m >= min(self.face_diameters_m) / 2:
            raise ValueError("Nozzle roughness exceeds hydraulic radius")
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected whole-nozzle VesselWall")

    @property
    def cells(self):
        return len(self.cell_lengths_m)

    @property
    def length_m(self):
        return math.fsum(self.cell_lengths_m)


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
        if not isinstance(pipe, (PREquilibriumPipe, PREquilibriumNozzle, PRTerrainPipe)):
            raise TypeError("Expected finite pipe or resolved nozzle")
        if pipe.source not in by_name or pipe.target not in by_name:
            raise ValueError("Unknown pipe terminal vessel")
        a, b = by_name[pipe.source], by_name[pipe.target]
        terrain = isinstance(pipe, PRTerrainPipe)
        if terrain and (
            abs(pipe.elevations_m[0] - a.elevation_m) > 1e-8
            or abs(pipe.elevations_m[-1] - b.elevation_m) > 1e-8
        ):
            raise ValueError("Survey terminal elevations differ from physical vessels")
        if abs(b.elevation_m - a.elevation_m) > pipe.length_m:
            raise ValueError("Pipe elevation difference exceeds length")
        names = []
        tapered = isinstance(pipe, PREquilibriumNozzle)
        faces = pipe.face_diameters_m if tapered else (pipe.diameter_m,) * (pipe.cells + 1)
        routes = pipe.cell_routes() if terrain else None
        lengths = (
            tuple(row[0] for row in routes)
            if terrain
            else pipe.cell_lengths_m
            if tapered
            else (pipe.length_m / pipe.cells,) * pipe.cells
        )
        wall_areas = [
            math.pi
            * (faces[i] + faces[i + 1])
            / 2
            * math.hypot(lengths[i], (faces[i + 1] - faces[i]) / 2)
            for i in range(pipe.cells)
        ]
        total_wall_area = math.fsum(wall_areas)
        position = 0.0
        for i in range(pipe.cells):
            dl, dr = faces[i : i + 2]
            length = lengths[i]
            wall = None
            if pipe.wall is not None:
                data = asdict(pipe.wall)
                for key in ("heat_capacity_j_k", "fluid_ua_w_k", "ambient_ua_w_k"):
                    data[key] *= wall_areas[i] / total_wall_area
                wall = VesselWall(**data)
            name = f"{pipe.name}__{i:04d}"
            names.append(name)
            nodes.append(
                PREquilibriumNode(
                    name,
                    math.pi * length * (dl * dl + dl * dr + dr * dr) / 12,
                    "pipe_cell",
                    elevation_m=routes[i][1]
                    if terrain
                    else a.elevation_m
                    + (b.elevation_m - a.elevation_m) * (position + length / 2) / pipe.length_m,
                    length_m=length,
                    diameter_m=(dl + dr) / 2,
                    left_diameter_m=dl if tapered else None,
                    right_diameter_m=dr if tapered else None,
                    elevation_change_m=routes[i][2]
                    if terrain
                    else (b.elevation_m - a.elevation_m) * length / pipe.length_m,
                    hydrostatic_reconstruction=terrain,
                    local_loss_coefficient=routes[i][3] if terrain else 0.0,
                    dynamic_viscosity_pa_s=pipe.dynamic_viscosity_pa_s,
                    roughness_m=pipe.roughness_m,
                    distribution_strength=0.0 if tapered else pipe.distribution_strength,
                    buoyancy_relaxation_s=0.0 if tapered else pipe.buoyancy_relaxation_s,
                    drift_flux=None if tapered else pipe.drift_flux,
                    wall=wall,
                )
            )
            position += length
        chain = [pipe.source, *names, pipe.target]
        for i, (source, target) in enumerate(zip(chain[:-1], chain[1:])):
            connections.append(
                PREquilibriumLink(
                    f"{pipe.name}__face_{i:04d}",
                    source,
                    target,
                    faces[i],
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
