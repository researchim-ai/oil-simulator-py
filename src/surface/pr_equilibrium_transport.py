"""Conservative equilibrium transport graph with PR phase variables in Newton.

HEM is the default. Optional density-contrast slip is phenomenological and
requires calibration; relative kinetic energy and actual-phase donor fluxes
are explicit. Geometric pickups are kinematic and currently restricted to g=0.
This is not a validated flow-regime map or a real-water/brine property model.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass, replace

import torch

from .compositional_separator import CompositionalSeparator, CompositionalSeparatorState
from .darcy_hydraulics import darcy_pressure_loss
from .dispersed_liquid_drift import DispersedOilWaterClosure
from .drift_flux_closure import (
    RELAP7EquilibriumTransition,
    RELAP7VerticalDriftClosure,
    ZuberFindlayClosure,
    drift_closure_from_dict,
)
from .equilibrium_inventory import EquilibriumInventory
from .equilibrium_properties import EquilibriumFluid
from .hllc_local import hllc_local
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _name, _number
from .operation import _digest
from .phase_geometry import phase_layout, pickup_fractions, validate_pickup
from .pr_compressor_unit import CompressorBundle, PRCompressorUnit, inventory_guesses
from .pr_droplet_population import DropletPopulations, PRDropletPopulation, number_flux
from .pr_heat_link import PRHeatLink, thermal_exchange, thermal_records
from .pr_hydrostatic_faces import HydrostaticFaces
from .pr_momentum_vessels import hllc_port
from .pr_pump_unit import PRPumpUnit, TorchPumpCell, serialize_pumps
from .pr_stagnation_ports import SinglePhaseReservoirTraces
from .separator_geometry import SeparatorGeometry
from .time_interval import interval_step_s
from .torch_linear import TorchNewton, checked_device
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PREquilibriumNode:
    name: str
    volume_m3: float
    kind: str = "vessel"
    elevation_m: float = 0.0
    length_m: float | None = None
    diameter_m: float | None = None
    elevation_change_m: float = 0.0
    dynamic_viscosity_pa_s: float = 0.0
    roughness_m: float = 0.0
    distribution_strength: float = 0.0
    buoyancy_relaxation_s: float = 0.0
    local_loss_coefficient: float = 0.0
    drift_flux: (
        ZuberFindlayClosure
        | RELAP7VerticalDriftClosure
        | RELAP7EquilibriumTransition
        | DispersedOilWaterClosure
        | None
    ) = None
    wall: VesselWall | None = None
    geometry: SeparatorGeometry | None = None
    left_diameter_m: float | None = None
    right_diameter_m: float | None = None
    hydrostatic_reconstruction: bool = False
    droplet_population: PRDropletPopulation | None = None
    droplet_capture: bool = False

    def __post_init__(self):
        _name(self.name)
        if type(self.hydrostatic_reconstruction) is not bool:
            raise ValueError("Hydrostatic reconstruction must be boolean")
        if self.hydrostatic_reconstruction and self.kind != "pipe_cell":
            raise ValueError("Hydrostatic reconstruction requires a finite pipe cell")
        if type(self.droplet_capture) is not bool:
            raise ValueError("Droplet capture must be explicitly boolean")
        if self.droplet_population is not None and (
            not isinstance(self.droplet_population, PRDropletPopulation)
            or self.kind != "pipe_cell"
            or not isinstance(self.drift_flux, DispersedOilWaterClosure)
            or self.droplet_capture
        ):
            raise ValueError(
                "Droplet population requires a dispersed-liquid pipe closure and no capture flag"
            )
        _number(self.volume_m3, "node volume", positive=True)
        _number(self.elevation_m, "node elevation")
        _number(self.elevation_change_m, "cell elevation change")
        for key in (
            "dynamic_viscosity_pa_s",
            "roughness_m",
            "distribution_strength",
            "buoyancy_relaxation_s",
            "local_loss_coefficient",
        ):
            _number(getattr(self, key), key, nonnegative=True)
        if self.drift_flux is not None:
            if not isinstance(
                self.drift_flux,
                (
                    ZuberFindlayClosure,
                    RELAP7VerticalDriftClosure,
                    RELAP7EquilibriumTransition,
                    DispersedOilWaterClosure,
                ),
            ):
                raise TypeError("Expected a supported drift-flux closure")
            if self.distribution_strength or self.buoyancy_relaxation_s:
                raise ValueError("Select one slip closure, without double-counting drift")
        if self.kind not in ("vessel", "pipe_cell"):
            raise ValueError("Node kind must be vessel or pipe_cell")
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected VesselWall")
        if self.geometry is not None and not isinstance(self.geometry, SeparatorGeometry):
            raise TypeError("Expected SeparatorGeometry")
        if self.kind == "pipe_cell":
            _number(self.length_m, "cell length", positive=True)
            _number(self.diameter_m, "cell diameter", positive=True)
            if (self.left_diameter_m is None) != (self.right_diameter_m is None):
                raise ValueError("Both face diameters must be specified")
            dl, dr = self.face_diameter("left"), self.face_diameter("right")
            _number(dl, "left face diameter", positive=True)
            _number(dr, "right face diameter", positive=True)
            if abs(self.diameter_m - (dl + dr) / 2) > 1e-12:
                raise ValueError("Cell diameter must be the arithmetic face mean")
            volume = math.pi * self.length_m * (dl * dl + dl * dr + dr * dr) / 12
            if abs(self.volume_m3 - volume) > 1e-12 + 1e-10 * volume:
                raise ValueError("Pipe cell volume differs from its conical frustum")
            if (
                self.geometry is not None
                or (
                    abs(self.elevation_change_m) > self.length_m
                    and not math.isclose(abs(self.elevation_change_m), self.length_m, rel_tol=1e-12)
                )
                or self.roughness_m >= self.diameter_m / 2
            ):
                raise ValueError("Invalid pipe cell geometry")
        elif (
            any(
                x is not None
                for x in (
                    self.length_m,
                    self.diameter_m,
                    self.left_diameter_m,
                    self.right_diameter_m,
                )
            )
            or self.elevation_change_m
            or self.dynamic_viscosity_pa_s
            or self.roughness_m
            or self.distribution_strength
            or self.buoyancy_relaxation_s
            or self.local_loss_coefficient
            or self.drift_flux is not None
        ):
            raise ValueError("Pipe parameters cannot be attached to a stirred vessel")
        if (
            self.geometry is not None
            and abs(self.geometry.volume_m3 - self.volume_m3) > 1e-12 + 1e-10 * self.volume_m3
        ):
            raise ValueError("Vessel geometry volume differs")

    def face_diameter(self, side):
        if side not in ("left", "right") or self.kind != "pipe_cell":
            raise ValueError("A pipe left/right face is required")
        value = self.left_diameter_m if side == "left" else self.right_diameter_m
        return self.diameter_m if value is None else value

    def face_area(self, side):
        return math.pi * self.face_diameter(side) ** 2 / 4

    def lateral_geometry(self):
        if self.kind != "pipe_cell" or self.left_diameter_m is not None:
            raise ValueError("Lateral layer pickups require a cylindrical finite volume")
        if abs(self.elevation_change_m) < 1e-12:
            return SeparatorGeometry(self.diameter_m, self.length_m, "horizontal")
        if abs(self.elevation_change_m - self.length_m) < 1e-12:
            return SeparatorGeometry(self.diameter_m, self.length_m, "vertical")
        raise ValueError(
            "Layer-resolved lateral pickups require horizontal or upward vertical cells"
        )


@dataclass(frozen=True)
class PREquilibriumLink:
    name: str
    source: str
    target: str
    diameter_m: float
    source_port: str = "right"
    target_port: str = "left"
    source_height_m: float | None = None
    target_height_m: float | None = None

    def __post_init__(self):
        for n in (self.name, self.source, self.target):
            _name(n)
        _number(self.diameter_m, "link diameter", positive=True)
        if self.source == self.target:
            raise ValueError("Self link requires an explicit pipe/control volume")
        for port in (self.source_port, self.target_port):
            if port not in ("left", "right") and not (
                isinstance(port, str) and port.startswith("side:") and len(port) > 5
            ):
                raise ValueError("Pipe ports must be left/right or named side:<id>")
        if self.source_port.startswith("side:") and self.target_port.startswith("side:"):
            raise ValueError("Two lateral axes require an explicit junction control volume")
        for h in (self.source_height_m, self.target_height_m):
            if h is not None:
                _number(h, "pickup height", nonnegative=True)


@dataclass(frozen=True)
class PREquilibriumCellState:
    component_moles: tuple
    total_energy_j: float
    momentum_kg_m_s: float
    phase_component_moles: tuple
    phase_roots: tuple
    pressure_pa: float
    temperature_k: float
    wall_temperature_k: float | None
    droplet_number: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "component_moles", tuple(self.component_moles))
        object.__setattr__(
            self, "phase_component_moles", tuple(tuple(x) for x in self.phase_component_moles)
        )
        object.__setattr__(self, "phase_roots", tuple(self.phase_roots))
        for n in self.component_moles:
            _number(n, "component inventory", nonnegative=True)
        if not math.fsum(self.component_moles) > 0:
            raise ValueError("Positive cell inventory required")
        for x in self.phase_component_moles:
            for n in x:
                _number(n, "phase inventory", nonnegative=True)
        if len(self.phase_component_moles) != len(self.phase_roots) or not self.phase_roots:
            raise ValueError("Phase topology differs")
        for t in self.phase_roots:
            if t not in ("stable", "liquid", "vapor"):
                raise ValueError("Unknown PR root")
        for k in ("total_energy_j", "momentum_kg_m_s"):
            _number(getattr(self, k), k)
        for k in ("pressure_pa", "temperature_k"):
            _number(getattr(self, k), k, positive=True)
        if self.wall_temperature_k is not None:
            _number(self.wall_temperature_k, "wall temperature", positive=True)
        if self.droplet_number is not None:
            _number(self.droplet_number, "droplet number", nonnegative=True)


@dataclass(frozen=True)
class PREquilibriumTransportState:
    nodes: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "nodes", tuple((name, s) for name, s in self.nodes))
        _number(self.time_s, "time", nonnegative=True)
        _number(self.next_step_s, "step", positive=True)

    def to_dict(self):
        return {
            "nodes": {
                name: {k: v for k, v in asdict(s).items() if k != "droplet_number" or v is not None}
                for name, s in self.nodes
            },
            "time_s": self.time_s,
            "next_step_s": self.next_step_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"nodes", "time_s", "next_step_s", "model_id"}:
            raise ValueError("Unknown transport state fields")
        return cls(
            tuple((n, PREquilibriumCellState(**s)) for n, s in sorted(data["nodes"].items())),
            data["time_s"],
            data["next_step_s"],
            data["model_id"],
        )


class PREquilibriumTransportNetwork:
    def __init__(
        self,
        fluid,
        nodes,
        links=(),
        *,
        gravity_m_s2=0.0,
        reference_elevation_m=0.0,
        wave_speed_bound_m_s=2000.0,
        maximum_velocity_m_s=100.0,
        initial_step_s=1e-5,
        minimum_step_s=1e-10,
        maximum_step_s=0.01,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.002,
        velocity_tolerance_m_s=0.001,
        maximum_attempts=1000,
        maximum_branch_attempts=32,
        pipe_paths=None,
        dense_limit=128,
        maximum_unknowns=100000,
        wave_fan="fixed_bound",
        reservoir_ports="riemann",
        heat_links=(),
        pumps=(),
        compressors=(),
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Expected EquilibriumFluid")
        checked_device(fluid.device)
        self.fluid = fluid
        self.nodes = tuple(sorted(nodes, key=lambda n: n.name))
        self.links = tuple(sorted(links, key=lambda n: n.name))
        self.heat_links = tuple(sorted(heat_links, key=lambda n: n.name))
        self.pumps = tuple(sorted(pumps, key=lambda n: n.name))
        self.compressors = tuple(sorted(compressors, key=lambda n: n.name))
        if any(not isinstance(unit, PRCompressorUnit) for unit in self.compressors):
            raise TypeError("Expected PRCompressorUnit")
        if any(not isinstance(pump, PRPumpUnit) for pump in self.pumps):
            raise TypeError("Expected PRPumpUnit")
        if any(not isinstance(link, PRHeatLink) for link in self.heat_links):
            raise TypeError("Expected PRHeatLink")
        self.switches = (*self.links, *self.heat_links, *self.pumps, *self.compressors)
        self.default_openings = {
            c.name: c.card.open
            if isinstance(c, PRPumpUnit)
            else c.open
            if isinstance(c, PRCompressorUnit)
            else True
            for c in self.switches
        }
        if (
            not self.nodes
            or len({n.name for n in self.nodes}) != len(self.nodes)
            or len({c.name for c in self.links}) != len(self.links)
        ):
            raise ValueError("Unique nonempty topology required")
        self.names = tuple(n.name for n in self.nodes)
        for unit in self.compressors:
            if unit.source not in self.names or unit.target not in self.names:
                raise ValueError("Unknown compressor plenum")
            a, b = [self.nodes[self.names.index(name)] for name in (unit.source, unit.target)]
            if (
                a.kind != "vessel"
                or b.kind != "vessel"
                or abs(a.elevation_m - b.elevation_m) > 1e-8
            ):
                raise ValueError("Mapped compressor needs two finite plenums at the same elevation")
            if len(unit.reference_mole_fractions) != len(fluid.components):
                raise ValueError("Compressor reference gas differs from the EOS component list")
        for node in self.nodes:
            closure = node.drift_flux
            if isinstance(closure, DispersedOilWaterClosure) and max(
                closure.gas_component_index,
                closure.oil_component_index,
                closure.water_component_index,
            ) >= len(fluid.components):
                raise ValueError("Drift marker index outside fluid component list")
        if len({c.name for c in self.switches}) != len(self.switches):
            raise ValueError("Material and thermal connections need unique names")
        if any(c.source not in self.names or c.target not in self.names for c in self.heat_links):
            raise ValueError("Unknown thermal endpoint")
        self.heat_ends = tuple(
            (self.names.index(c.source), self.names.index(c.target)) for c in self.heat_links
        )
        self.pipe_paths = {name: tuple(path) for name, path in sorted((pipe_paths or {}).items())}
        claimed = set()
        for name, path in self.pipe_paths.items():
            _name(name)
            if name in self.names or not path or len(set(path)) != len(path):
                raise ValueError("Invalid compiled pipe group")
            for cell in path:
                if (
                    cell not in self.names
                    or cell in claimed
                    or self.nodes[self.names.index(cell)].kind != "pipe_cell"
                ):
                    raise ValueError("Pipe groups must name disjoint finite pipe cells")
                claimed.add(cell)
        for k, v in locals().copy().items():
            if k in (
                "wave_speed_bound_m_s",
                "maximum_velocity_m_s",
                "initial_step_s",
                "minimum_step_s",
                "maximum_step_s",
                "relative_tolerance",
                "temperature_tolerance_k",
                "velocity_tolerance_m_s",
            ):
                setattr(self, k, _number(v, k, positive=True))
        self.gravity_m_s2 = _number(gravity_m_s2, "gravity", nonnegative=True)
        self.reference_elevation_m = _number(reference_elevation_m, "reference elevation")
        for k, v in [
            ("maximum_attempts", maximum_attempts),
            ("maximum_branch_attempts", maximum_branch_attempts),
            ("maximum_unknowns", maximum_unknowns),
        ]:
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError("Positive integer budget required")
            setattr(self, k, v)
        if isinstance(dense_limit, bool) or not isinstance(dense_limit, int) or dense_limit < 0:
            raise ValueError("Nonnegative dense limit required")
        if wave_fan not in ("fixed_bound", "local_single_phase", "pressure_single_phase"):
            raise ValueError("Unknown transport wave fan")
        if reservoir_ports not in ("riemann", "stagnation_single_phase"):
            raise ValueError("Unknown reservoir port model")
        self.reservoir_ports = reservoir_ports
        self.wave_fan = wave_fan
        self.dense_limit = dense_limit
        self._patterns = {}
        if (
            not minimum_step_s <= initial_step_s <= maximum_step_s
            or maximum_velocity_m_s >= wave_speed_bound_m_s
        ):
            raise ValueError("Invalid transport bounds")
        if gravity_m_s2 and any(n.geometry is not None for n in self.nodes):
            raise ValueError(
                "Finite-aperture homogeneous vessels require g=0; use gravitational profiles separately"
            )
        self.ends = []
        used = set()
        for link in self.links:
            if link.source not in self.names or link.target not in self.names:
                raise ValueError("Unknown link endpoint")
            pair = (self.names.index(link.source), self.names.index(link.target))
            self.ends.append(pair)
            heights = []
            for side, i in zip(("source", "target"), pair):
                n = self.nodes[i]
                h = getattr(link, side + "_height_m")
                port = getattr(link, side + "_port")
                if n.kind == "pipe_cell":
                    lateral = port.startswith("side:")
                    if n.droplet_population is not None and h is not None:
                        raise ValueError("Dispersed droplets require mixed, not layered, sampling")
                    if lateral and h is not None:
                        validate_pickup(n.lateral_geometry(), h, link.diameter_m)
                    if (
                        (h is not None and not lateral)
                        or (not lateral and abs(link.diameter_m - n.face_diameter(port)) > 1e-12)
                        or (
                            lateral
                            and math.pi * link.diameter_m**2 / 4
                            > math.pi * n.diameter_m * n.length_m
                        )
                        or (i, port) in used
                    ):
                        raise ValueError("Pipe ends require unique matching-area connections")
                    used.add((i, port))
                    heights.append(
                        n.elevation_m
                        + (
                            h - n.lateral_geometry().height_m / 2
                            if lateral and h is not None
                            else 0.0
                            if lateral
                            else n.elevation_change_m * (0.5 if port == "right" else -0.5)
                        )
                    )
                else:
                    validate_pickup(n.geometry, h, link.diameter_m if h is not None else None)
                    heights.append(n.elevation_m)
            if self.gravity_m_s2 and abs(heights[0] - heights[1]) > 1e-8:
                raise ValueError("Connected endpoints must share a physical elevation")
            na, nb = (self.nodes[j] for j in pair)
            pa, pb = na.droplet_population, nb.droplet_population
            if pa is not None and pb is not None:
                if (
                    pa.family != pb.family
                    or na.drift_flux.dispersed_liquid != nb.drift_flux.dispersed_liquid
                    or (
                        na.drift_flux.gas_component_index,
                        na.drift_flux.oil_component_index,
                        na.drift_flux.water_component_index,
                    )
                    != (
                        nb.drift_flux.gas_component_index,
                        nb.drift_flux.oil_component_index,
                        nb.drift_flux.water_component_index,
                    )
                ):
                    raise ValueError(
                        "Direct particle transfer requires the same droplet family and markers"
                    )
            elif (pa is not None and not nb.droplet_capture) or (
                pb is not None and not na.droplet_capture
            ):
                raise ValueError(
                    "Droplets need an explicit finite capture/injection volume at a model boundary"
                )
        self._used = used
        if len({pump.node for pump in self.pumps}) != len(self.pumps):
            raise ValueError("One pump card per finite body required")
        self._pump_cells = tuple(TorchPumpCell(self, pump) for pump in self.pumps)
        for node in self.nodes:
            if isinstance(
                node.drift_flux, (RELAP7VerticalDriftClosure, RELAP7EquilibriumTransition)
            ) and (
                self.gravity_m_s2 <= 0 or abs(abs(node.elevation_change_m) - node.length_m) > 1e-12
            ):
                raise ValueError("RELAP-7 pipe drift requires a vertical pipe and nonzero gravity")
        self._pickup_nodes = {
            index
            for link, pair in zip(self.links, self.ends)
            for side, index in zip(("source", "target"), pair)
            if getattr(link, side + "_height_m") is not None
        }
        # Reuse the monolithic phase-equilibrium residual, not a nested flash.
        self.cells = tuple(
            CompositionalSeparator(
                fluid, n.volume_m3, maximum_dt_s=maximum_step_s, minimum_dt_s=minimum_step_s
            )
            for n in self.nodes
        )
        self.model_id = _digest(self.to_dict())
        self._verified = {}

    def to_dict(self):
        return {
            "model": "pr_equilibrium_transport_graph_v1",
            "fluid": self.fluid.to_dict(),
            "nodes": [
                {
                    k: v
                    for k, v in asdict(n).items()
                    if k
                    not in ("hydrostatic_reconstruction", "droplet_capture", "droplet_population")
                    or v
                }
                for n in self.nodes
            ],
            "links": [asdict(c) for c in self.links],
            "heat_links": [asdict(c) for c in self.heat_links],
            **({"pumps": [asdict(pump) for pump in self.pumps]} if self.pumps else {}),
            **(
                {"compressors": [asdict(unit) for unit in self.compressors]}
                if self.compressors
                else {}
            ),
            "pipe_paths": self.pipe_paths,
            "reservoir_ports": self.reservoir_ports,
            "wave_fan": self.wave_fan,
            "dense_limit": self.dense_limit,
            "maximum_unknowns": self.maximum_unknowns,
            **{
                k: getattr(self, k)
                for k in (
                    "gravity_m_s2",
                    "reference_elevation_m",
                    "wave_speed_bound_m_s",
                    "maximum_velocity_m_s",
                    "initial_step_s",
                    "minimum_step_s",
                    "maximum_step_s",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "velocity_tolerance_m_s",
                    "maximum_attempts",
                    "maximum_branch_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "pr_equilibrium_transport_graph_v1":
            raise ValueError("Unknown equilibrium transport model")
        data["fluid"] = EquilibriumFluid.from_dict(data["fluid"], device=device)
        nodes = []
        for n in data["nodes"]:
            n = dict(n)
            if n.get("drift_flux") is not None:
                n["drift_flux"] = drift_closure_from_dict(n["drift_flux"])
            if n.get("droplet_population") is not None:
                n["droplet_population"] = PRDropletPopulation(**n["droplet_population"])
            if n.get("wall") is not None:
                n["wall"] = VesselWall(**n["wall"])
            if n.get("geometry") is not None:
                n["geometry"] = SeparatorGeometry(**n["geometry"])
            nodes.append(PREquilibriumNode(**n))
        data["nodes"] = nodes
        data["links"] = [PREquilibriumLink(**c) for c in data["links"]]
        data["heat_links"] = [PRHeatLink(**c) for c in data.get("heat_links", [])]
        data["pumps"] = [PRPumpUnit.from_dict(c) for c in data.get("pumps", [])]
        data["compressors"] = [PRCompressorUnit.from_dict(c) for c in data.get("compressors", [])]
        return cls(**data)

    def _population(self, index, phase_n, x, q, number):
        node = self.nodes[index]
        if node.droplet_population is None:
            return None
        result = node.droplet_population.properties(
            node.drift_flux, phase_n * q["molar_volume_m3_mol"], x, number, q
        )
        result["valid"] = result["valid"] & (
            result["number"] <= node.droplet_population.maximum_number_density_m3 * node.volume_m3
        )
        return result

    def _kinematics(self, index, phase_n, x, q, u, population=None):
        node, f = self.nodes[index], self.fluid
        masses = phase_n * (x * f.molar_masses).sum(-1)
        drift = torch.zeros_like(phase_n)
        if node.kind == "pipe_cell" and (node.distribution_strength or node.buoyancy_relaxation_s):
            density = masses.sum() / (phase_n * q["molar_volume_m3_mol"]).sum()
            scale = (
                node.distribution_strength * u
                + node.buoyancy_relaxation_s
                * self.gravity_m_s2
                * node.elevation_change_m
                / node.length_m
            )
            drift = scale * (density / q["density_kg_m3"] - 1.0)
        velocities = u + drift
        if node.drift_flux is not None:
            velocities, _valid = node.drift_flux.velocities(
                phase_n * q["molar_volume_m3_mol"],
                q["density_kg_m3"],
                u,
                diameter_m=node.diameter_m,
                axial_gravity_m_s2=self.gravity_m_s2 * node.elevation_change_m / node.length_m,
                phase_mole_fractions=x,
                gravity_m_s2=self.gravity_m_s2,
                **(
                    {
                        "droplet_diameter_m": population["diameter_m"],
                        "phase_roles": population["roles"],
                    }
                    if population is not None
                    else {}
                ),
            )
            if population is not None:
                velocities = torch.where(population["present"], velocities, u.expand_as(velocities))
            drift = velocities - u
        return (
            drift,
            velocities,
            0.5 * (masses * velocities.square()).sum(),
            (masses * drift.square()).sum() / node.volume_m3,
        )

    def _decode(self, index, amounts, p, t, u, active, tags, droplet_number=None):
        f = self.fluid
        n = self.nodes[index]
        phase_n, x, q = self.cells[index]._phase_values(p, t, amounts, active, tags)
        totals = f.tensor([0.0] * len(f.names)).index_copy(0, active, amounts.sum(0))
        volume = (phase_n * q["molar_volume_m3_mol"]).sum()
        weights = phase_n * q["molar_volume_m3_mol"] / n.volume_m3
        mass = (totals * f.molar_masses).sum()
        rho = mass / n.volume_m3
        internal = (phase_n * q["molar_internal_energy_j_mol"]).sum()
        population = self._population(index, phase_n, x, q, droplet_number)
        surface = population["surface_energy_j"] if population is not None else p * 0
        drift, velocities, kinetic, stress = self._kinematics(index, phase_n, x, q, u, population)
        potential = mass * self.gravity_m_s2 * (n.elevation_m - self.reference_elevation_m)
        entropy = (phase_n * q["molar_entropy_j_mol_k"]).sum()
        return {
            "amounts": amounts,
            "n": totals,
            "p": p,
            "t": t,
            "u": u,
            "phase_n": phase_n,
            "x": x,
            "q": q,
            "volume": volume,
            "alpha": weights,
            "mass": mass,
            "rho": rho,
            "internal": internal,
            "kinetic": kinetic,
            "potential": potential,
            "entropy": entropy,
            "energy": internal + kinetic + potential + surface,
            "surface": surface,
            "population": population,
            "momentum": mass * u,
            "drift": drift,
            "phase_velocity": velocities,
            "drift_stress": stress,
            "tags": tuple(tags),
        }

    def _state_data(self, index, s):
        f = self.fluid
        active = torch.nonzero(f.tensor(s.component_moles) > 0).flatten()
        a = f.tensor(s.phase_component_moles)
        if a.shape != (len(s.phase_roots), len(f.names)):
            raise ValueError("Phase inventory dimensions differ")
        mass = (f.tensor(s.component_moles) * f.molar_masses).sum()
        u = f.tensor(s.momentum_kg_m_s) / mass
        return self._decode(
            index,
            a[:, active],
            f.tensor(s.pressure_pa),
            f.tensor(s.temperature_k),
            u,
            active,
            s.phase_roots,
            None if s.droplet_number is None else f.tensor(s.droplet_number),
        )

    def _continuation_flash(self, data):
        """Audit an already converged branch if a cold PT allocation fails.

        Incipient phases make PT allocation ill-conditioned. The common Newton
        already provides equal fugacities and stocks: check those and run the
        same multistart TPD search, never accept an unaudited branch.
        """
        f = self.fluid
        z = data["n"] / data["n"].sum()
        active = torch.nonzero(z > 0).flatten()
        chemical = data["x"].clamp_min(1e-300).log() + data["q"]["log_fugacity_coefficients"]
        if float((chemical[:, active] - chemical[-1, active]).abs().max().cpu()) > 1e-8:
            raise HydraulicConvergenceError("Continuation branch has unequal fugacities")
        minimum, trial, trial_q = f._stability(data["p"], data["t"], z, chemical[-1], active)
        beta = data["phase_n"] / data["n"].sum()
        phases = [
            {"molar_fraction": beta[j], **{k: v[j] for k, v in data["q"].items()}}
            for j in range(len(beta))
        ]
        if float(minimum.cpu()) < -1e-8:
            if len(phases) >= f.max_phases:
                raise PropertyModelError(
                    "Continuation requires additional phases beyond max_phases"
                )
            fraction = torch.minimum(
                f.tensor(1e-4), 0.5 * (z[active] / trial[active].clamp_min(1e-300)).min()
            )
            phases = [
                {**phase, "molar_fraction": phase["molar_fraction"] * (1 - fraction)}
                for phase in phases
            ]
            phases.append({**trial_q, "mole_fractions": trial, "molar_fraction": fraction})
        return {
            "pressure_pa": data["p"],
            "temperature_k": data["t"],
            "phase_count": len(phases),
            "phases": phases,
            "molar_volume_m3_mol": (beta * data["q"]["molar_volume_m3_mol"]).sum(),
            "molar_internal_energy_j_mol": (beta * data["q"]["molar_internal_energy_j_mol"]).sum(),
        }

    def _stable(self, index, data):
        f = self.fluid
        n = self.nodes[index]
        z = data["n"] / data["n"].sum()
        try:
            q = f.flash_pt(data["p"], data["t"], z)
        except HydraulicConvergenceError:
            q = self._continuation_flash(data)
        except PropertyModelError as exc:
            if "Pure-fluid PT coexistence" not in str(exc):
                raise
            q = f.flash_uv(
                data["n"],
                float(data["internal"].cpu()),
                n.volume_m3,
                pressure_guess_pa=float(data["p"].cpu()),
                temperature_guess_k=float(data["t"].cpu()),
            )
        if q["phase_count"] != len(data["phase_n"]):
            return q
        checks = [
            (q["molar_volume_m3_mol"] * data["n"].sum() - n.volume_m3).abs()
            / (1e-12 + 1e-8 * n.volume_m3),
            (q["molar_internal_energy_j_mol"] * data["n"].sum() - data["internal"]).abs()
            / (1e-6 + 1e-8 * max(1.0, abs(float(data["internal"].cpu())))),
        ]
        if max(float(v.cpu()) for v in checks) > 1:
            return q
        active = torch.nonzero(z > 0).flatten()
        mu = (
            torch.log(data["x"][-1, active].clamp_min(1e-300))
            + data["q"]["log_fugacity_coefficients"][-1, active]
        )
        refphase = q["phases"][0]
        reference = (
            torch.log(refphase["mole_fractions"][active].clamp_min(1e-300))
            + refphase["log_fugacity_coefficients"][active]
        )
        if float((mu - reference).abs().max().cpu()) > 1e-7:
            return q
        if n.drift_flux is not None:
            population = data["population"]
            _, valid = n.drift_flux.velocities(
                data["phase_n"] * data["q"]["molar_volume_m3_mol"],
                data["q"]["density_kg_m3"],
                data["u"],
                diameter_m=n.diameter_m,
                axial_gravity_m_s2=self.gravity_m_s2 * n.elevation_change_m / n.length_m,
                phase_mole_fractions=data["x"],
                gravity_m_s2=self.gravity_m_s2,
                **(
                    {
                        "droplet_diameter_m": population["diameter_m"],
                        "phase_roles": population["roles"],
                    }
                    if population is not None
                    else {}
                ),
            )
            if isinstance(n.drift_flux, DispersedOilWaterClosure):
                valid = valid & n.drift_flux.pressure_valid(
                    data["x"],
                    data["p"],
                    None if population is None else population["diameter_m"],
                    None if population is None else population["roles"],
                )
            if population is not None:
                valid = torch.where(
                    population["present"],
                    valid & population["valid"],
                    (population["number"] == 0) & population["valid"],
                )
            if not bool(valid.cpu()):
                raise PropertyModelError("Drift-flux closure outside declared regime envelope")
        if float(data["phase_velocity"].abs().max().cpu()) > self.maximum_velocity_m_s:
            raise PropertyModelError("Phase velocity exceeds transport domain")
        if index in self._pickup_nodes:
            if n.kind == "pipe_cell":
                layout = phase_layout(data["phase_n"], data["q"], n.lateral_geometry())
                if not bool(layout["density_order_resolved"].cpu()):
                    raise PropertyModelError("Lateral pickup phase density order is unresolved")
                # Cell-centred gravity and a sharp subcell reconstruction are
                # a low-head approximation, not a hydrostatic PR column.
                if (
                    float(
                        (
                            data["q"]["density_kg_m3"].max()
                            * self.gravity_m_s2
                            * n.lateral_geometry().height_m
                            / data["p"]
                        ).cpu()
                    )
                    > 1e-3
                ):
                    raise PropertyModelError("Lateral layer pickup needs rho*g*height/P <= 1e-3")
            for x, tag in zip(data["x"], data["tags"]):
                pt = f.tensor([float(data["p"].cpu()), float(data["t"].cpu())])

                def properties(v, x=x, tag=tag):
                    phase = f.tensor_homogeneous(v[0], v[1], x, tag)
                    return torch.stack((phase["density_kg_m3"], phase["molar_entropy_j_mol_k"]))

                jac = torch.func.jacfwd(properties)(pt)
                den = jac[0, 0] - jac[0, 1] * jac[1, 0] / jac[1, 1]
                if (
                    not bool((torch.isfinite(jac).all() & (den > 0) & (jac[1, 1] > 0)).cpu())
                    or float(torch.rsqrt(den).cpu()) > self.wave_speed_bound_m_s
                ):
                    raise PropertyModelError(
                        "Wave bound below a phase-selective aperture acoustic speed"
                    )
        data["characteristic_speeds"] = self._characteristics(index, data, active)
        return None

    def _characteristics(self, index, data, active):
        """Exact constrained flux tangent on the accepted fixed phase branch.

        The algebraic phase/volume constraints are differentiated together with
        conservative N/M/E, avoiding a frozen-phase sound-speed substitution.
        """
        f = self.fluid
        node = self.nodes[index]
        m, c = data["amounts"].shape
        y0 = torch.cat(
            (
                data["amounts"].log().flatten(),
                data["p"].log().reshape(1),
                (data["t"] / f.tref).reshape(1),
                (data["u"] / self.maximum_velocity_m_s).reshape(1),
            )
        )
        pop = data["population"]
        particle_active = pop is not None and bool(pop["present"].cpu())
        number_scale = max(1.0, float(pop["number"].cpu())) if particle_active else 1.0
        if particle_active:
            y0 = torch.cat((y0, (pop["number"] / number_scale).reshape(1)))

        def quantities(y):
            d = self._decode(
                index,
                y[: m * c].exp().reshape(m, c),
                y[m * c].exp(),
                y[m * c + 1] * f.tref,
                y[m * c + 2] * self.maximum_velocity_m_s,
                active,
                data["tags"],
                y[m * c + 3] * number_scale
                if particle_active
                else None
                if pop is None
                else pop["number"],
            )
            mu = (
                torch.log(d["x"][:, active].clamp_min(1e-300))
                + d["q"]["log_fugacity_coefficients"][:, active]
            )
            uc = torch.cat(
                (
                    d["n"][active] / node.volume_m3,
                    (d["momentum"] / node.volume_m3).reshape(1),
                    (d["energy"] / node.volume_m3).reshape(1),
                    *(
                        ((d["population"]["number"] / number_scale / node.volume_m3).reshape(1),)
                        if particle_active
                        else ()
                    ),
                    (mu[:-1] - mu[-1]).flatten(),
                    (d["volume"] / node.volume_m3 - 1).reshape(1),
                )
            )
            w = d["drift"]
            phase_mass = (d["x"] * f.molar_masses).sum(-1)
            component = d["n"] / node.volume_m3 * d["u"] + (
                d["phase_n"][:, None] / node.volume_m3 * d["x"] * w[:, None]
            ).sum(0)
            drift_energy = (
                d["phase_n"]
                / node.volume_m3
                * (
                    d["q"]["molar_enthalpy_j_mol"]
                    + phase_mass
                    * (
                        0.5 * w.square()
                        + self.gravity_m_s2 * (node.elevation_m - self.reference_elevation_m)
                    )
                )
                * w
            ).sum() + d["u"] * d["drift_stress"]
            if particle_active:
                drop_velocity = (d["phase_velocity"] * d["population"]["drop_mask"]).sum()
                drift_energy = drift_energy + d["surface"] / node.volume_m3 * (
                    drop_velocity - d["u"]
                )
            flux = torch.cat(
                (
                    component[active],
                    (d["rho"] * d["u"].square() + d["p"] + d["drift_stress"]).reshape(1),
                    ((d["energy"] / node.volume_m3 + d["p"]) * d["u"] + drift_energy).reshape(1),
                    *(
                        (
                            (
                                d["population"]["number"]
                                / number_scale
                                / node.volume_m3
                                * drop_velocity
                            ).reshape(1),
                        )
                        if particle_active
                        else ()
                    ),
                )
            )
            return uc, flux

        a = torch.func.jacfwd(lambda y: quantities(y)[0])(y0)
        b = torch.func.jacfwd(lambda y: quantities(y)[1])(y0)
        count = c + 2 + int(particle_active)
        rhs = y0.new_zeros((len(y0), count))
        rhs[:count] = torch.eye(count, dtype=y0.dtype, device=y0.device)
        try:
            tangent = torch.linalg.solve(a, rhs)
            speeds = torch.linalg.eigvals(b @ tangent)
        except RuntimeError as exc:
            raise HydraulicConvergenceError(
                "Singular equilibrium transport characteristic tangent"
            ) from exc
        if not bool(torch.isfinite(speeds).all().cpu()) or float(
            speeds.imag.abs().max().cpu()
        ) > 1e-7 * max(1.0, float(speeds.real.abs().max().cpu())):
            raise PropertyModelError("Transport closure loses local hyperbolicity")
        if float(speeds.real.abs().max().cpu()) > self.wave_speed_bound_m_s:
            raise PropertyModelError("Transport wave bound below equilibrium characteristics")
        return speeds.real.detach()

    def inspect(self, state):
        if (
            not isinstance(state, PREquilibriumTransportState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Transport graph state/model differs")
        if (
            tuple(name for name, _ in state.nodes) != self.names
            or not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s
        ):
            raise ValueError("Transport restart topology or timestep differs")
        key = _digest(state.to_dict())
        if key in self._verified:
            return
        for i, (node, (_, s)) in enumerate(zip(self.nodes, state.nodes)):
            if (node.droplet_population is None) != (s.droplet_number is None):
                raise ValueError("Droplet checkpoint registry differs from physical model")
            f = self.fluid
            d = self._state_data(i, s)
            if (
                not f.min_pressure_pa <= s.pressure_pa <= f.max_pressure_pa
                or not f.min_temperature_k <= s.temperature_k <= f.max_temperature_k
                or abs(float(d["u"].cpu())) > self.maximum_velocity_m_s
            ):
                raise PropertyModelError("Transport primitive outside domain")
            if node.kind == "vessel" and s.momentum_kg_m_s != 0:
                raise ConservationError("Stirred node cannot retain axial momentum")
            if node.wall is None:
                if s.wall_temperature_k is not None:
                    raise ValueError("Unexpected wall state")
            elif (
                s.wall_temperature_k is None
                or not node.wall.min_temperature_k
                <= s.wall_temperature_k
                <= node.wall.max_temperature_k
            ):
                raise PropertyModelError("Wall outside domain")
            active = torch.nonzero(f.tensor(s.component_moles) > 0).flatten()
            mu = (
                torch.log(d["x"][:, active].clamp_min(1e-300))
                + d["q"]["log_fugacity_coefficients"][:, active]
            )
            if not bool(
                (
                    d["q"]["valid"].all()
                    & torch.isfinite(d["energy"])
                    & ((d["n"] - f.tensor(s.component_moles)).abs().max() < 1e-9)
                    & ((d["volume"] - node.volume_m3).abs() < 1e-12 + 1e-9 * node.volume_m3)
                    & (
                        (d["energy"] - s.total_energy_j).abs()
                        < 1e-6 + 1e-9 * max(1, abs(s.total_energy_j))
                    )
                    & (
                        (mu[:-1] - mu[-1]).abs().max() < 1e-8
                        if len(mu) > 1
                        else mu.new_tensor(True, dtype=torch.bool)
                    )
                ).cpu()
            ):
                raise ConservationError("Transport equilibrium or stock checkpoint differs")
            if node.geometry is not None and not bool(
                phase_layout(d["phase_n"], d["q"], node.geometry)["density_order_resolved"].cpu()
            ):
                raise PropertyModelError(
                    "Vessel phase densities do not resolve a unique layer order"
                )
            if self._stable(i, d) is not None:
                raise PropertyModelError("Serialized transport phase branch is unstable")
        self._verified[key] = True
        while len(self._verified) > 16:
            self._verified.pop(next(iter(self._verified)))

    def initialize(self, initial):
        if set(initial) != set(self.names):
            raise ValueError("Initial node registry differs")
        states = []
        f = self.fluid
        for i, (node, cell) in enumerate(zip(self.nodes, self.cells)):
            spec = dict(initial[node.name])
            velocity = _number(spec.pop("velocity_m_s", 0.0), "initial velocity")
            tw = spec.pop("wall_temperature_k", None)
            particle_number = spec.pop("droplet_number", None)
            if particle_number is not None and node.droplet_population is None:
                raise ValueError("Initial particle stock without a droplet model")
            if node.kind == "vessel" and velocity:
                raise ValueError("Stirred vessels have zero bulk velocity")
            seed = cell.initialize(**spec)
            q = cell.inspect(seed)
            tags = cell._tags(q)
            total = f.tensor(seed.inventory.component_moles).sum()
            a = torch.stack(
                [phase["molar_fraction"] * phase["mole_fractions"] * total for phase in q["phases"]]
            )
            active = torch.nonzero(f.tensor(seed.inventory.component_moles) > 0).flatten()
            d = self._decode(
                i,
                a[:, active],
                f.tensor(seed.inventory.pressure_pa),
                f.tensor(seed.inventory.temperature_k),
                f.tensor(velocity),
                active,
                tags,
                None if particle_number is None else f.tensor(particle_number),
            )
            if node.wall is not None:
                tw = seed.inventory.temperature_k if tw is None else tw
            elif tw is not None:
                raise ValueError("Initial wall temperature without wall")
            states.append(
                (
                    node.name,
                    PREquilibriumCellState(
                        tuple(d["n"].cpu().tolist()),
                        float(d["energy"].cpu()),
                        float(d["momentum"].cpu()),
                        tuple(map(tuple, a.cpu().tolist())),
                        tuple(tags),
                        float(d["p"].cpu()),
                        float(d["t"].cpu()),
                        tw,
                        None if d["population"] is None else float(d["population"]["number"].cpu()),
                    ),
                )
            )
        state = PREquilibriumTransportState(tuple(states), 0.0, self.initial_step_s, self.model_id)
        self.inspect(state)
        return state

    def _endpoint(self, index, data, side, link):
        node = self.nodes[index]
        data = data.get("hydrostatic_faces", {}).get(getattr(link, side + "_port"), data)
        weights = data["alpha"]
        h = getattr(link, side + "_height_m")
        if h is not None:
            weights = pickup_fractions(
                phase_layout(
                    data["phase_n"],
                    data["q"],
                    node.lateral_geometry() if node.kind == "pipe_cell" else node.geometry,
                ),
                h,
                link.diameter_m,
            )
        port = getattr(link, side + "_port")
        normal = 0.0 if port.startswith("side:") else (1.0 if port == "right" else -1.0)
        projection = normal if side == "source" else -normal
        velocity = data["u"] * projection if node.kind == "pipe_cell" else data["u"] * 0
        if "port_velocity" in data:
            velocity = data["port_velocity"]
        q = data["q"]
        concentration = (weights[:, None] * data["x"] / q["molar_volume_m3_mol"][:, None]).sum(0)
        rho = (weights * q["density_kg_m3"]).sum()
        internal = (weights * q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]).sum()
        population = data.get("population")
        surface_phase = weights * 0
        if population is not None:
            surface_phase = (
                weights
                * population["drop_mask"]
                * population["surface_energy_j"]
                / population["drop_volume_m3"].clamp_min(1e-30)
            )
            internal = internal + surface_phase.sum()
        # A lateral opening retains axial kinetic energy as a tangential
        # passive quantity. Incoming perpendicular-jet energy thermalizes;
        # outgoing flow also removes the donor's axial momentum below.
        if normal == 0.0 and node.kind == "pipe_cell":
            internal = (
                internal
                + 0.5 * (weights * q["density_kg_m3"] * data["phase_velocity"].square()).sum()
            )
        entropy = (weights * q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]).sum()
        # Both sides of a link use its physical endpoint potential.
        z = node.elevation_m + (
            h - node.lateral_geometry().height_m / 2
            if node.kind == "pipe_cell" and normal == 0.0 and h is not None
            else node.elevation_change_m * normal / 2
            if node.kind == "pipe_cell"
            else 0.0
        )
        e = (
            internal
            + 0.5 * rho * velocity.square()
            + rho * self.gravity_m_s2 * (z - self.reference_elevation_m)
        )
        u = torch.cat((concentration, (rho * velocity).reshape(1), e.reshape(1)))
        flux = torch.cat(
            (
                concentration * velocity,
                (rho * velocity.square() + data["p"]).reshape(1),
                ((e + data["p"]) * velocity).reshape(1),
            )
        )
        drift = (
            data["drift"] * projection if node.kind == "pipe_cell" else torch.zeros_like(weights)
        )
        concentration_phase = weights[:, None] * data["x"] / q["molar_volume_m3_mol"][:, None]
        phase_mass_density = weights * q["density_kg_m3"]
        phase_mass = (data["x"] * self.fluid.molar_masses).sum(-1)
        phase_entropy = weights * q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
        stress = (phase_mass_density * drift.square()).sum()
        u = torch.cat((u[:-1], (u[-1] + 0.5 * stress).reshape(1)))
        flux = torch.cat((flux[:-1], (flux[-1] + 0.5 * stress * velocity).reshape(1)))
        phi = self.gravity_m_s2 * (z - self.reference_elevation_m)
        actual_velocity = velocity + drift
        actual = torch.cat(
            (
                concentration_phase * actual_velocity[:, None],
                (phase_mass_density * actual_velocity.square())[:, None],
                (
                    weights
                    / q["molar_volume_m3_mol"]
                    * (
                        q["molar_enthalpy_j_mol"]
                        + phase_mass * (0.5 * actual_velocity.square() + phi)
                    )
                    * actual_velocity
                )[:, None],
            ),
            -1,
        )
        bulk = torch.cat(
            (
                concentration_phase * velocity,
                (phase_mass_density * velocity.square())[:, None],
                (
                    weights
                    / q["molar_volume_m3_mol"]
                    * (
                        q["molar_enthalpy_j_mol"]
                        + phase_mass * (0.5 * (velocity.square() + drift.square()) + phi)
                    )
                    * velocity
                )[:, None],
            ),
            -1,
        )
        if population is not None:
            actual = torch.cat(
                (actual[:, :-1], (actual[:, -1] + surface_phase * actual_velocity)[:, None]), -1
            )
            bulk = torch.cat((bulk[:, :-1], (bulk[:, -1] + surface_phase * velocity)[:, None]), -1)
        extra = {
            "phase_z": q["compressibility_factor"],
            "phase_densities": q["density_kg_m3"],
            "phase_x": data["x"],
            "phase_weights": weights,
            "phase_concentration": concentration_phase,
            "sound_speed": data["q"].get("frozen_sound_speed_m_s", None),
            "tangential_velocity": (weights * q["density_kg_m3"] * data["phase_velocity"]).sum()
            / rho,
            "actual_velocity": actual_velocity,
            "bulk_velocity": velocity,
            "actual": actual,
            "bulk": bulk,
            "actual_entropy": phase_entropy * actual_velocity,
            "bulk_entropy": phase_entropy * velocity,
        }
        return (u, flux, data["p"], velocity, rho, entropy), weights, normal, extra

    def _fluxes(self, data, openings, external_ports=(), traces=None):
        f = self.fluid
        nc = len(f.names)
        net = [data[0]["p"].new_zeros(nc + 2) for _ in self.nodes]
        records = []
        valid = data[0]["p"].new_ones((), dtype=torch.bool)
        occupied = set(external_ports)
        for link, (a, b) in zip(self.links, self.ends):
            area = math.pi * link.diameter_m**2 / 4
            left_data = data[a] if traces is None else traces.get((link.name, "source"), data[a])
            right_data = data[b] if traces is None else traces.get((link.name, "target"), data[b])
            la, wa, na, da = self._endpoint(a, left_data, "source", link)
            rb, wb, nb, db = self._endpoint(b, right_data, "target", link)
            _fallback = data[a]["p"].new_zeros((), dtype=torch.bool)
            if openings[link.name]:
                if (
                    self.wave_fan in ("local_single_phase", "pressure_single_phase")
                    and len(data[a]["phase_n"]) == 1
                    and len(data[b]["phase_n"]) == 1
                ):
                    flux, entropy, contact, pressure, ok, _fallback = hllc_local(
                        la,
                        rb,
                        da["sound_speed"][0],
                        db["sound_speed"][0],
                        self.wave_speed_bound_m_s,
                        estimate="pvrs" if self.wave_fan == "pressure_single_phase" else "davis",
                    )
                else:
                    flux, entropy, contact, pressure, ok = hllc_port(
                        la, rb, self.wave_speed_bound_m_s
                    )
                valid = valid & ok
                # Correct bulk HLLC by actual-phase FVS minus bulk FVS.
                base_flux = flux
                # Donors follow the absolute phase velocity, not the relative drift.
                correction = torch.zeros_like(flux)
                entropy_correction = entropy * 0
                for packet, forward in ((da, True), (db, False)):
                    actual_mask = (
                        packet["actual_velocity"] > 0 if forward else packet["actual_velocity"] < 0
                    )
                    bulk_mask = (
                        packet["bulk_velocity"] > 0 if forward else packet["bulk_velocity"] < 0
                    )
                    correction = (
                        correction
                        + torch.where(
                            actual_mask[:, None],
                            packet["actual"],
                            torch.zeros_like(packet["actual"]),
                        ).sum(0)
                        - torch.where(
                            bulk_mask, packet["bulk"], torch.zeros_like(packet["bulk"])
                        ).sum(0)
                    )
                    entropy_correction = (
                        entropy_correction
                        + torch.where(
                            actual_mask,
                            packet["actual_entropy"],
                            torch.zeros_like(packet["actual_entropy"]),
                        ).sum()
                        - torch.where(
                            bulk_mask,
                            packet["bulk_entropy"],
                            torch.zeros_like(packet["bulk_entropy"]),
                        ).sum()
                    )
                flux = flux + correction
                entropy = entropy + entropy_correction
            else:
                flux = data[a]["p"].new_zeros(nc + 2)
                base_flux = flux
                entropy = data[a]["p"].new_zeros(())
                contact = data[a]["p"].new_zeros(())
                pressure = (la[2] + rb[2]) / 2
            for side, index, normal, sign, endpoint in [
                ("source", a, na, -1.0, la),
                ("target", b, nb, 1.0, rb),
            ]:
                node = self.nodes[index]
                update = torch.cat(
                    (
                        sign * area * flux[:-2],
                        (
                            flux[-2] * (-normal * area)
                            if node.kind == "pipe_cell"
                            else flux[-2] * 0
                        ).reshape(1),
                        (sign * area * flux[-1]).reshape(1),
                    )
                )
                if node.kind == "pipe_cell":
                    if normal == 0.0:
                        mass_flux = (flux[:-2] * f.molar_masses).sum()
                        outward_mass = -sign * mass_flux
                        tangent = (
                            da["tangential_velocity"]
                            if side == "source"
                            else db["tangential_velocity"]
                        )
                        axial = -area * outward_mass.clamp_min(0) * tangent
                        update = update + torch.cat(
                            (update.new_zeros(nc), axial.reshape(1), update.new_zeros(1))
                        )
                    occupied.add((index, getattr(link, side + "_port")))
                    if not openings[link.name]:
                        # A closed end reflects the pipe's own velocity and pressure.
                        velocity = data[index]["u"]
                        wall = (
                            data[index]
                            .get("hydrostatic_faces", {})
                            .get(getattr(link, side + "_port"), data[index])
                        )
                        rho = (
                            wall["q"]["density_kg_m3"][0]
                            if "hydrostatic_faces" in data[index]
                            else data[index]["rho"]
                        )
                        wallforce = area * (
                            wall["p"]
                            + data[index]["drift_stress"]
                            + rho * velocity.square()
                            + normal * self.wave_speed_bound_m_s * rho * velocity
                        )
                        update = update + torch.cat(
                            (
                                update.new_zeros(nc),
                                (-normal * wallforce).reshape(1),
                                update.new_zeros(1),
                            )
                        )
                net[index] = net[index] + update
            particle_fluxes = {}
            populations = [
                (node.droplet_population, node.drift_flux)
                for node in (self.nodes[a], self.nodes[b])
                if node.droplet_population is not None
            ]
            if populations:
                config, law = populations[0]
                particle_fluxes[config.family] = area * number_flux(
                    base_flux,
                    contact,
                    da,
                    db,
                    law,
                    data[a].get("population"),
                    data[b].get("population"),
                    config,
                )
                rate = particle_fluxes[config.family]
                left_pop, right_pop = data[a].get("population"), data[b].get("population")
                dl = (
                    left_pop["diameter_m"]
                    if left_pop is not None
                    else rate.new_tensor(law.droplet_diameter_m)
                )
                dr = (
                    right_pop["diameter_m"]
                    if right_pop is not None
                    else rate.new_tensor(law.droplet_diameter_m)
                )
                valid = valid & torch.where(
                    rate > 0, dl / link.diameter_m <= law.maximum_confinement_ratio, True
                )
                valid = valid & torch.where(
                    rate < 0, dr / link.diameter_m <= law.maximum_confinement_ratio, True
                )
            records.append(
                {
                    "name": link.name,
                    "source_index": a,
                    "target_index": b,
                    "flux": area * flux,
                    "entropy_flux": area * entropy,
                    "contact_velocity": contact,
                    "pressure": pressure,
                    "source_weights": wa,
                    "target_weights": wb,
                    "local_fan_fixed_fallback": _fallback,
                    "droplet_number_fluxes": particle_fluxes,
                }
            )
        for i, node in enumerate(self.nodes):
            if node.kind != "pipe_cell":
                continue
            area = math.pi * node.diameter_m**2 / 4
            rho = data[i]["rho"]
            v = data[i]["u"]
            p = data[i]["p"]
            for port, normal in [("left", -1.0), ("right", 1.0)]:
                if (i, port) not in occupied:
                    wall = data[i].get("hydrostatic_faces", {}).get(port, data[i])
                    pressure = wall["p"]
                    wall_rho = (
                        wall["q"]["density_kg_m3"][0] if "hydrostatic_faces" in data[i] else rho
                    )
                    force = node.face_area(port) * (
                        pressure
                        + data[i]["drift_stress"]
                        + wall_rho * v.square()
                        + normal * self.wave_speed_bound_m_s * wall_rho * v
                    )
                    net[i] = net[i] + torch.cat(
                        (p.new_zeros(nc), (-normal * force).reshape(1), p.new_zeros(1))
                    )
        return net, records, valid

    def _seed(self, index, s):
        d = self._state_data(index, s)
        total = d["n"].sum()
        return {
            "pressure_pa": d["p"],
            "temperature_k": d["t"],
            "phase_count": len(s.phase_roots),
            "phases": [
                {"molar_fraction": d["phase_n"][j] / total, **{k: v[j] for k, v in d["q"].items()}}
                for j in range(len(s.phase_roots))
            ],
        }

    def _active(self, state, openings):
        masks = [self.fluid.tensor(s.component_moles) > 0 for _, s in state.nodes]
        for _ in self.nodes:
            for link, (a, b) in zip(self.links, self.ends):
                if openings[link.name]:
                    union = masks[a] | masks[b]
                    masks[a] = union
                    masks[b] = union
            for unit in self.compressors:
                if openings[unit.name]:
                    a, b = self.names.index(unit.source), self.names.index(unit.target)
                    masks[b] = masks[b] | masks[a]
        return [torch.nonzero(mask).flatten() for mask in masks]

    def _problem(
        self,
        state,
        seeds,
        dt,
        openings,
        heat_w,
        loss_coefficients=None,
        *,
        external_ports=(),
        active_override=None,
        guesses_override=None,
    ):
        f = self.fluid
        old = [s for _, s in state.nodes]
        active = self._active(state, openings) if active_override is None else active_override
        losses = loss_coefficients or {}
        problems = []
        offsets = [0]
        lo = []
        hi = []
        initial = []
        olddata = [self._state_data(i, s) for i, s in enumerate(old)]
        if self.compressors:
            guesses_override = inventory_guesses(self, state, dt, openings, guesses_override)
        for i, (node, cell, s, seed) in enumerate(zip(self.nodes, self.cells, old, seeds)):
            inv = EquilibriumInventory(
                s.component_moles,
                float(olddata[i]["internal"].cpu()),
                node.volume_m3,
                s.pressure_pa,
                s.temperature_k,
                f.model_id,
                state.time_s,
                0.0,
                0.0,
            )
            fake = CompositionalSeparatorState(inv, None, state.next_step_s, cell.model_id)
            problem = cell._branch_step(
                fake,
                seed,
                tuple(range(seed["phase_count"])),
                dt,
                (f.tensor([0.0] * len(f.names)), f.tensor(0.0), f.tensor(0.0), f.tensor(math.inf)),
                0.0,
                f.tensor(s.temperature_k),
                {},
                _capture=True,
                _network_active=active[i],
                _network_guess=f.tensor(s.component_moles)
                if guesses_override is None
                else guesses_override[i],
            )
            problems.append(problem)
            initial.append(
                torch.cat(
                    (problem["initial"], (olddata[i]["u"] / self.maximum_velocity_m_s).reshape(1))
                )
            )
            lo.append(torch.cat((problem["lower"], f.tensor([-1.0]))))
            hi.append(torch.cat((problem["upper"], f.tensor([1.0]))))
            offsets.append(offsets[-1] + len(initial[-1]))
        trace_bundle = SinglePhaseReservoirTraces(
            self, state, seeds, olddata, openings, offsets[-1]
        )
        initial.extend(trace_bundle.initial)
        lo.extend(trace_bundle.lower)
        hi.extend(trace_bundle.upper)
        compressor_bundle = CompressorBundle(
            self, state, openings, offsets[-1] + 3 * len(trace_bundle.entries)
        )
        initial.extend(compressor_bundle.initial)
        lo.extend(compressor_bundle.lower)
        hi.extend(compressor_bundle.upper)
        face_bundle = HydrostaticFaces(
            self,
            state,
            seeds,
            olddata,
            active,
            offsets[-1] + 3 * (len(trace_bundle.entries) + len(compressor_bundle.entries)),
        )
        initial.extend(face_bundle.initial)
        lo.extend(face_bundle.lower)
        hi.extend(face_bundle.upper)
        population_bundle = DropletPopulations(self, state, seeds, sum(map(len, initial)))
        initial.extend(population_bundle.initial)
        lo.extend(population_bundle.lower)
        hi.extend(population_bundle.upper)
        if sum(map(len, initial)) > self.maximum_unknowns:
            raise ValueError("Transport graph exceeds configured unknown budget")

        phase_tags = [self.cells[i]._tags(seed) for i, seed in enumerate(seeds)]
        mass_scales = [
            max(1e-12, float(d["mass"].cpu())) * self.maximum_velocity_m_s for d in olddata
        ]

        def evaluate(y, external_fluxes=None):
            numbers = population_bundle.numbers(y)
            blocks = []
            data = []
            walls = []
            heat = []
            ambient = []
            for i, (node, p, seed, s) in enumerate(zip(self.nodes, problems, seeds, old)):
                local = y[offsets[i] : offsets[i + 1]]
                r, raw = p["evaluate"](local[:-1])
                amounts, totals, pp, t, _, phase_n, x, q, internal = raw[:9]
                velocity = local[-1] * self.maximum_velocity_m_s
                mass = (totals * f.molar_masses).sum()
                population = self._population(i, phase_n, x, q, numbers.get(i))
                surface = population["surface_energy_j"] if population is not None else pp * 0
                drift, velocities, kinetic, stress = self._kinematics(
                    i, phase_n, x, q, velocity, population
                )
                potential = (
                    mass * self.gravity_m_s2 * (node.elevation_m - self.reference_elevation_m)
                )
                d = {
                    "amounts": amounts,
                    "n": totals,
                    "p": pp,
                    "t": t,
                    "u": velocity,
                    "phase_n": phase_n,
                    "x": x,
                    "q": q,
                    "volume": raw[19],
                    "alpha": phase_n * q["molar_volume_m3_mol"] / node.volume_m3,
                    "mass": mass,
                    "rho": mass / node.volume_m3,
                    "internal": internal,
                    "kinetic": kinetic,
                    "potential": potential,
                    "entropy": (phase_n * q["molar_entropy_j_mol_k"]).sum(),
                    "energy": internal + kinetic + potential + surface,
                    "surface": surface,
                    "population": population,
                    "momentum": mass * velocity,
                    "drift": drift,
                    "phase_velocity": velocities,
                    "drift_stress": stress,
                    "tags": phase_tags[i],
                }
                if node.wall is None:
                    tw = t
                    qh = t * 0
                    qa = t * 0
                else:
                    w = node.wall
                    tw = (
                        w.heat_capacity_j_k * s.wall_temperature_k
                        + dt * (w.fluid_ua_w_k * t + w.ambient_ua_w_k * w.ambient_temperature_k)
                    ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
                    qh = w.fluid_ua_w_k * (tw - t)
                    qa = w.ambient_ua_w_k * (w.ambient_temperature_k - tw)
                blocks.append(r)
                data.append(d)
                walls.append(tw)
                heat.append(qh)
                ambient.append(qa)
            face_rows, face_records, face_valid = face_bundle.evaluate(y, data)
            trace_points, trace_residual, trace_details = trace_bundle.evaluate(y, data)
            net, records, valid = self._fluxes(data, openings, external_ports, trace_points)
            valid = valid & face_valid
            for d in data:
                if d["population"] is not None:
                    valid = valid & d["population"]["valid"]
            population_rows, population_records = population_bundle.balances(data, records, dt)
            thermal_rates, heat_records = thermal_exchange(
                self.heat_links, {n: d["t"] for n, d in zip(self.names, data)}, openings
            )
            for name, rate in thermal_rates.items():
                i = self.names.index(name)
                net[i] = net[i] + torch.cat((net[i].new_zeros(len(f.names) + 1), rate.reshape(1)))
            pump_records, pump_forces = {}, {}
            for pump in self._pump_cells:
                force, power, row, admissible = pump.evaluate(data, openings[pump.unit.name])
                i = pump.index
                pump_forces[i] = force
                net[i] = net[i] + torch.cat((net[i].new_zeros(len(f.names) + 1), power.reshape(1)))
                pump_records[pump.unit.name] = row
                valid = valid & admissible
            compressor_rows, compressor_records, compressor_traces = [], {}, []
            if self.compressors:
                (
                    compressor_net,
                    compressor_rows,
                    compressor_records,
                    compressor_traces,
                    compressor_valid,
                ) = compressor_bundle.evaluate(y, data)
                for i, rate in compressor_net.items():
                    net[i] = net[i] + rate
                valid = valid & compressor_valid
            trace_residual, trace_details = trace_bundle.complete(trace_details, records)
            for _entry, point, _source, _v, _h, _row in trace_details:
                valid = valid & point["q"]["valid"].all()
            exchange = None
            if external_fluxes is not None:
                exchange = external_fluxes(data)
                net = [a + b for a, b in zip(net, exchange["net"])]
                valid = valid & exchange["valid"]
            residual = []
            sources = []
            for i, (node, d, p, r) in enumerate(zip(self.nodes, data, problems, blocks)):
                erow = p["energy_row"]
                force = d["p"] * 0
                if node.kind == "pipe_cell":
                    force = -d[
                        "mass"
                    ] * self.gravity_m_s2 * node.elevation_change_m / node.length_m + d["p"] * (
                        node.face_area("right") - node.face_area("left")
                    )
                    if "hydrostatic_faces" in d:
                        pl = d["hydrostatic_faces"]["left"]["p"]
                        pr = d["hydrostatic_faces"]["right"]["p"]
                        al, ar = node.face_area("left"), node.face_area("right")
                        force = (al + ar) / 2 * (pr - pl) + (pl + pr) / 2 * (ar - al)
                    if node.dynamic_viscosity_pa_s:
                        force = (
                            force
                            - node.volume_m3
                            * darcy_pressure_loss(
                                d["u"],
                                d["rho"],
                                1.0,
                                node.diameter_m,
                                node.dynamic_viscosity_pa_s,
                                node.roughness_m,
                            )[0]
                        )
                    coefficient = losses.get(node.name, node.local_loss_coefficient)
                    force = (
                        force
                        - 0.5
                        * math.pi
                        * node.diameter_m**2
                        / 4
                        * coefficient
                        * d["rho"]
                        * d["u"]
                        * d["u"].abs()
                    )
                    if i in pump_forces:
                        force = force + pump_forces[i]
                    mr = (d["momentum"] - old[i].momentum_kg_m_s - dt * (net[i][-2] + force)) / (
                        mass_scales[i]
                    )
                else:
                    mr = d["u"] / self.maximum_velocity_m_s
                # Original residual contains U_new-U_old. Add K/PE and actual shared fluxes.
                extra = (
                    d["kinetic"]
                    + d["potential"]
                    - olddata[i]["kinetic"]
                    - olddata[i]["potential"]
                    + d["surface"]
                    - olddata[i]["surface"]
                )
                energy = (
                    r[erow] + (extra - dt * (net[i][-1] + heat[i] + heat_w[i])) / p["energy_scale"]
                )
                residual.append(
                    torch.cat(
                        (
                            r[: len(p["active"])] - dt * net[i][p["active"]] / p["nscale"],
                            r[len(p["active"]) : erow],
                            energy.reshape(1),
                            mr.reshape(1),
                        )
                    )
                )
                sources.append(force)
            return torch.cat(
                (*residual, *trace_residual, *compressor_rows, *face_rows, *population_rows)
            ), {
                "droplet_populations": population_records,
                "hydrostatic_faces": face_records,
                "reservoir_trace_data": trace_details,
                "heat_links": heat_records,
                "pumps": pump_records,
                "compressors": compressor_records,
                "compressor_traces": compressor_traces,
                "data": data,
                "net": net,
                "links": records,
                "valid": valid,
                "walls": walls,
                "heat": heat,
                "ambient": ambient,
                "forces": sources,
                "exchange": exchange,
            }

        evaluate.reservoir_traces = trace_bundle
        evaluate.compressors = compressor_bundle
        evaluate.hydrostatic_faces = face_bundle
        evaluate.droplet_populations = population_bundle
        return (
            torch.cat(initial),
            torch.cat(lo),
            torch.cat(hi),
            evaluate,
            offsets,
            problems,
            olddata,
        )

    def _sparsity(
        self,
        offsets,
        openings,
        trace_entries=(),
        compressor_entries=(),
        hydrostatic_entries=(),
        population_entries=(),
    ):
        """Exact block graph support, rebuilt on phase/support/isolation changes."""
        key = (
            tuple(offsets),
            tuple(openings[c.name] for c in self.switches),
            tuple((e.vessel, e.pipe, e.offset) for e in trace_entries),
            tuple((e[1], e[2], e[3]) for e in compressor_entries),
            tuple((e.cell, e.port, e.offset, e.count) for e in hydrostatic_entries),
            tuple((i, offset) for i, offset, *_ in population_entries),
        )
        if key not in self._patterns:
            neighbours = [{i} for i in range(len(self.nodes))]
            for link, (a, b) in zip(self.links, self.ends):
                if openings[link.name]:
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            for link, (a, b) in zip(self.heat_links, self.heat_ends):
                if openings[link.name]:
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            for unit in self.compressors:
                if openings[unit.name]:
                    a, b = self.names.index(unit.source), self.names.index(unit.target)
                    neighbours[a].add(b)
                    neighbours[b].add(a)
            rows, cols = [], []
            for i, connected in enumerate(neighbours):
                for row in range(offsets[i], offsets[i + 1]):
                    for j in sorted(connected):
                        for col in range(offsets[j], offsets[j + 1]):
                            rows.append(row)
                            cols.append(col)
            for e in trace_entries:
                bodies = [
                    *range(offsets[e.vessel], offsets[e.vessel + 1]),
                    *range(offsets[e.pipe], offsets[e.pipe + 1]),
                ]
                for row in bodies:
                    for col in (e.offset, e.offset + 1, e.offset + 2):
                        rows.append(row)
                        cols.append(col)
                for row in (e.offset, e.offset + 1, e.offset + 2):
                    for col in [*bodies, e.offset, e.offset + 1, e.offset + 2]:
                        rows.append(row)
                        cols.append(col)
            # Bound topology-cache growth in long schedules with many switches.
            for _unit, a, b, offset, *_ in compressor_entries:
                bodies = [*range(offsets[a], offsets[a + 1]), *range(offsets[b], offsets[b + 1])]
                aux = list(range(offset, offset + 3))
                for row in bodies:
                    for col in aux:
                        rows.append(row)
                        cols.append(col)
                for row in aux:
                    for col in [*bodies, *aux]:
                        rows.append(row)
                        cols.append(col)
            if len(self._patterns) >= 16:
                self._patterns.clear()
            extra = HydrostaticFaces.pattern(self, offsets, hydrostatic_entries, openings)
            extra |= DropletPopulations.pattern(self, offsets, population_entries, openings)
            support = sorted(set(zip(rows, cols)) | extra)
            self._patterns[key] = ([p[0] for p in support], [p[1] for p in support])
        return self._patterns[key]

    def _step(
        self, state, dt, openings, heat_w, utility, loss_coefficients=None, *, _coupling=None
    ):
        f = self.fluid
        seeds = [self._seed(i, s) for i, (_, s) in enumerate(state.nodes)]
        queue = [seeds]
        seen = set()
        failures = []
        while queue and len(seen) < self.maximum_branch_attempts:
            seeds = queue.pop(0)
            signature = tuple(
                (
                    q["phase_count"],
                    round(float(q["temperature_k"].cpu()), 7),
                    round(float(q["pressure_pa"].cpu()), 5),
                    tuple(
                        round(float(x), 9)
                        for phase in q["phases"]
                        for x in phase["mole_fractions"].cpu().tolist()
                    ),
                )
                for q in seeds
            )
            if signature in seen:
                continue
            seen.add(signature)
            y, lo, hi, evaluate, _offsets, problems, olddata = self._problem(
                state,
                seeds,
                dt,
                openings,
                heat_w,
                loss_coefficients,
                external_ports=() if _coupling is None else _coupling.external_ports,
                active_override=None if _coupling is None else _coupling.active,
                guesses_override=None if _coupling is None else _coupling.ordinary_guesses,
            )
            trace_bundle = evaluate.reservoir_traces
            compressor_bundle = evaluate.compressors
            face_bundle = evaluate.hydrostatic_faces
            population_bundle = evaluate.droplet_populations
            pattern = (
                self._sparsity(
                    _offsets,
                    openings,
                    trace_bundle.entries,
                    compressor_bundle.entries,
                    face_bundle.entries,
                    population_bundle.entries,
                )
                if len(y) > self.dense_limit
                else None
            )
            if _coupling is not None:
                y, lo, hi, evaluate, pattern = _coupling.prepare(
                    y, lo, hi, evaluate, _offsets, dt, pattern
                )
            solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=self.dense_limit)
            solver.host_convergence = True
            with torch.no_grad():
                y, _, ok, evaluations = solver.solve(
                    lambda z, evaluate=evaluate: evaluate(z)[0],
                    y,
                    lo,
                    hi,
                    pattern=pattern,
                    tolerance=3e-12,
                )
                r, result = evaluate(y)
            accepted = bool(
                (ok & result["valid"] & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()
            )
            replacements = []
            if accepted:
                for i, d in enumerate(result["data"]):
                    if not bool(d["q"]["valid"].all().cpu()):
                        accepted = False
                        break
                    post = self._stable(i, d)
                    if post is not None:
                        replacements.append((i, post))
                if not replacements and accepted:
                    trace_bundle.validate(result["reservoir_trace_data"])
                    compressor_bundle.validate(result["compressor_traces"])
                    face_bundle.validate(result["hydrostatic_faces"])
                    coupled_state = None if _coupling is None else _coupling.accept(y, result)
                    states = []
                    entropy = 0.0
                    local_ledgers = []
                    for i, (node, (_, old), d) in enumerate(
                        zip(self.nodes, state.nodes, result["data"])
                    ):
                        target_n = f.tensor(old.component_moles) + dt * result["net"][i][:-2]
                        n = d["n"]
                        energy = old.total_energy_j + dt * (
                            result["net"][i][-1] + result["heat"][i] + heat_w[i]
                        )
                        momentum = (
                            (
                                old.momentum_kg_m_s
                                + dt * (result["net"][i][-2] + result["forces"][i])
                            )
                            if node.kind == "pipe_cell"
                            else d["p"] * 0
                        )
                        if not bool((n[f.tensor(old.component_moles) > 0] > 0).all().cpu()):
                            raise HydraulicConvergenceError("HEM step exhausts active component")
                        a = d["amounts"]
                        active = problems[i]["active"]
                        full = a.new_zeros((len(a), len(f.names))).index_copy(-1, active, a)
                        tw = float(result["walls"][i].cpu()) if node.wall is not None else None
                        if node.wall is not None and (
                            not node.wall.min_temperature_k <= tw <= node.wall.max_temperature_k
                            or abs(tw - old.wall_temperature_k) > node.wall.max_temperature_change_k
                        ):
                            raise HydraulicConvergenceError("HEM wall step/domain rejected")
                        states.append(
                            (
                                node.name,
                                PREquilibriumCellState(
                                    tuple(n.cpu().tolist()),
                                    float(energy.cpu()),
                                    # Keep the converged primitive and the
                                    # recorded Riemann flux consistent near
                                    # rest. The independently checked momentum
                                    # defect remains in local_ledgers below.
                                    float(d["momentum"].cpu()) if node.kind == "pipe_cell" else 0.0,
                                    tuple(map(tuple, full.cpu().tolist())),
                                    tuple(self.cells[i]._tags(seeds[i])),
                                    float(d["p"].cpu()),
                                    float(d["t"].cpu()),
                                    tw,
                                    None
                                    if d["population"] is None
                                    else float(d["population"]["number"].cpu()),
                                ),
                            )
                        )
                        entropy += float((d["entropy"] - olddata[i]["entropy"]).cpu())
                        if node.wall:
                            entropy += (
                                node.wall.heat_capacity_j_k * math.log(tw / old.wall_temperature_k)
                                - dt
                                * float(result["ambient"][i].cpu())
                                / node.wall.ambient_temperature_k
                            )
                        if heat_w[i]:
                            entropy -= dt * heat_w[i] / utility[i]
                        if heat_w[i] * (utility[i] - float(d["t"].cpu())) < -1e-10:
                            raise PropertyModelError(
                                "Heat source utility violates passive direction"
                            )
                        local_ledgers.append(
                            {
                                "component_residuals_mol": (d["n"] - target_n).cpu().tolist(),
                                "energy_residual_j": float((d["energy"] - energy).cpu()),
                                "momentum_residual_kg_m_s": float((d["momentum"] - momentum).cpu()),
                            }
                        )
                    if result["exchange"] is not None:
                        entropy -= dt * float(result["exchange"]["incoming_entropy"].cpu())
                    if entropy < -1e-6 - 1e-10 * max(
                        1.0, sum(abs(float(q["entropy"].cpu())) for q in olddata)
                    ):
                        raise ConservationError("HEM graph total entropy rejected")
                    new = PREquilibriumTransportState(
                        tuple(states), state.time_s + dt, state.next_step_s, self.model_id
                    )
                    self.inspect(new)
                    ambient = dt * sum(float(v.cpu()) for v in result["ambient"])
                    external = dt * sum(heat_w)
                    powers = [
                        row["shaft_power_w"]
                        for row in [*result["pumps"].values(), *result["compressors"].values()]
                    ]
                    work = dt * float(torch.stack(powers).sum().cpu()) if powers else 0.0
                    exchange_n = None
                    exchange_e = 0.0
                    if result["exchange"] is not None:
                        exchange_n = (
                            (
                                dt
                                * sum(
                                    (v[:-2] for v in result["exchange"]["net"]),
                                    f.tensor([0.0] * len(f.names)),
                                )
                            )
                            .cpu()
                            .tolist()
                        )
                        exchange_e = dt * float(sum(v[-1] for v in result["exchange"]["net"]).cpu())
                    ledger = self._ledger(
                        state, new, ambient + external + work + exchange_e, incoming_n=exchange_n
                    )
                    return new, {
                        "dt_s": dt,
                        "state": new.to_dict(),
                        **(
                            {
                                "coupling": coupled_state,
                                "exchange_component_moles": exchange_n,
                                "exchange_energy_j": exchange_e,
                            }
                            if _coupling is not None
                            else {}
                        ),
                        "reservoir_traces": trace_bundle.serialize(result["reservoir_trace_data"]),
                        "heat_links": thermal_records(result["heat_links"]),
                        "pumps": serialize_pumps(result["pumps"], openings),
                        "compressors": compressor_bundle.serialize(result["compressors"]),
                        "hydrostatic_faces": face_bundle.serialize(result["hydrostatic_faces"]),
                        "droplet_populations": population_bundle.serialize(
                            result["droplet_populations"]
                        ),
                        "links": {
                            rec["name"]: {
                                "flux": rec["flux"].cpu().tolist(),
                                **(
                                    {
                                        "droplet_number_fluxes": {
                                            k: float(v.cpu())
                                            for k, v in rec["droplet_number_fluxes"].items()
                                        }
                                    }
                                    if rec["droplet_number_fluxes"]
                                    else {}
                                ),
                                "local_fan_fixed_fallback": bool(
                                    rec["local_fan_fixed_fallback"].cpu()
                                ),
                                "entropy_flux_w_k": float(rec["entropy_flux"].cpu()),
                                "contact_velocity_m_s": float(rec["contact_velocity"].cpu()),
                                "source_fluid_reaction_impulse_kg_m_s": dt
                                * float(rec["flux"][-2].cpu())
                                if self.nodes[rec["source_index"]].kind == "vessel"
                                else 0.0,
                                "target_fluid_reaction_impulse_kg_m_s": -dt
                                * float(rec["flux"][-2].cpu())
                                if self.nodes[rec["target_index"]].kind == "vessel"
                                else 0.0,
                                "source_area_fractions": rec["source_weights"].cpu().tolist(),
                                "target_area_fractions": rec["target_weights"].cpu().tolist(),
                            }
                            for rec in result["links"]
                        },
                        "nodes": {
                            n.name: {
                                "phase_count": len(d["phase_n"]),
                                "phase_volume_fractions": d["alpha"].cpu().tolist(),
                                "phase_density_kg_m3": d["q"]["density_kg_m3"].cpu().tolist(),
                                "velocity_m_s": float(d["u"].cpu()),
                                "phase_velocity_m_s": d["phase_velocity"].cpu().tolist(),
                                "characteristic_speeds_m_s": d["characteristic_speeds"]
                                .cpu()
                                .tolist(),
                                "kinetic_energy_j": float(d["kinetic"].cpu()),
                                "potential_energy_j": float(d["potential"].cpu()),
                                "ambient_heat_j": dt * float(result["ambient"][i].cpu()),
                                "fluid_heat_j": dt * float(result["heat"][i].cpu()),
                                "force_impulse_kg_m_s": dt * float(result["forces"][i].cpu()),
                                **local_ledgers[i],
                            }
                            for i, (n, d) in enumerate(zip(self.nodes, result["data"]))
                        },
                        "conservation_residuals": ledger,
                        "ambient_heat_j": ambient,
                        "external_heat_j": external,
                        "external_work_j": work,
                        "entropy_production_j_k": entropy,
                        "phase_branch_attempts": len(seen),
                        "nonlinear_evaluations": evaluations,
                        "max_scaled_residual": float(r.abs().max().cpu()),
                    }
            for i, post in replacements:
                changed = list(seeds)
                changed[i] = post
                queue.append(changed)
            # A conserved UV predictor can discover phase birth without solving a PT flash in Newton.
            for i, (node, (_, s)) in enumerate(zip(self.nodes, state.nodes)):
                d = result["data"][i]
                n = f.tensor(s.component_moles) + dt * result["net"][i][:-2]
                if not bool((n[f.tensor(s.component_moles) > 0] > 0).all().cpu()):
                    continue
                target = s.total_energy_j + dt * (
                    result["net"][i][-1] + result["heat"][i] + heat_w[i]
                )
                u = (
                    target
                    - d["kinetic"]
                    - d["surface"]
                    - (n * f.molar_masses).sum()
                    * self.gravity_m_s2
                    * (node.elevation_m - self.reference_elevation_m)
                )
                try:
                    post = f.flash_uv(
                        n,
                        float(u.cpu()),
                        node.volume_m3,
                        pressure_guess_pa=s.pressure_pa,
                        temperature_guess_k=s.temperature_k,
                    )
                    if post["phase_count"] != seeds[i]["phase_count"]:
                        changed = list(seeds)
                        changed[i] = post
                        queue.append(changed)
                except (PropertyModelError, HydraulicConvergenceError) as exc:
                    failures.append(str(exc))
                if seeds[i]["phase_count"] > 1:
                    for keep in itertools.combinations(
                        range(seeds[i]["phase_count"]), seeds[i]["phase_count"] - 1
                    ):
                        changed = list(seeds)
                        changed[i] = {
                            **seeds[i],
                            "phase_count": len(keep),
                            "phases": [seeds[i]["phases"][j] for j in keep],
                        }
                        queue.append(changed)
            failures.append("joint Newton, stable phase branch, or HLLC domain rejected")
        raise HydraulicConvergenceError(
            "HEM phase branch budget exhausted: "
            + (failures[-1] if failures else "no admissible branch")
        )

    def _ledger(self, before, after, heat, *, incoming_n=None):
        f = self.fluid
        n = [
            math.fsum(s.component_moles[i] for _, s in after.nodes)
            - math.fsum(s.component_moles[i] for _, s in before.nodes)
            - (incoming_n[i] if incoming_n is not None else 0.0)
            for i in range(len(f.names))
        ]
        energy = (
            math.fsum(
                b.total_energy_j
                - a.total_energy_j
                + (
                    node.wall.heat_capacity_j_k * (b.wall_temperature_k - a.wall_temperature_k)
                    if node.wall
                    else 0.0
                )
                for node, (_, a), (_, b) in zip(self.nodes, before.nodes, after.nodes)
            )
            - heat
        )
        if any(
            abs(x) > 1e-8 + 1e-9 * math.fsum(s.component_moles[i] for _, s in before.nodes)
            for i, x in enumerate(n)
        ) or abs(energy) > 1e-5 + 1e-9 * max(1, abs(heat)):
            raise ConservationError("HEM graph global N/E balance rejected")
        return n + [energy]

    def _error(self, a, b):
        error = 0.0
        for i, ((_, x), (_, y)) in enumerate(zip(a.nodes, b.nodes)):
            if x.droplet_number is not None:
                error = max(
                    error,
                    abs(x.droplet_number - y.droplet_number)
                    / (1e-8 + self.relative_tolerance * max(x.droplet_number, y.droplet_number)),
                )
            error = max(
                error,
                abs(x.temperature_k - y.temperature_k) / self.temperature_tolerance_k,
                abs(x.pressure_pa - y.pressure_pa)
                / (1 + self.relative_tolerance * max(x.pressure_pa, y.pressure_pa)),
                max(
                    abs(u - v) / (1e-10 + self.relative_tolerance * max(abs(u), abs(v)))
                    for u, v in zip(x.component_moles, y.component_moles)
                ),
                abs(x.total_energy_j - y.total_energy_j)
                / (
                    1e-6
                    + self.relative_tolerance * max(1, abs(x.total_energy_j), abs(y.total_energy_j))
                ),
            )
            dx = self._state_data(i, x)
            dy = self._state_data(i, y)
            error = max(error, float((dx["u"] - dy["u"]).abs().cpu()) / self.velocity_tolerance_m_s)
            if self.nodes[i].wall:
                error = max(
                    error,
                    abs(x.wall_temperature_k - y.wall_temperature_k) / self.temperature_tolerance_k,
                )
        return error

    def advance(
        self,
        state,
        duration_s,
        *,
        openings=None,
        heat_w=None,
        utility_temperature_k=None,
        loss_coefficients=None,
    ):
        self.inspect(state)
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable HEM interval")
        losses = {n.name: n.local_loss_coefficient for n in self.nodes if n.kind == "pipe_cell"}
        if loss_coefficients is not None:
            if set(loss_coefficients) - set(losses):
                raise ValueError("Loss command requires known pipe cells")
            losses.update(
                {
                    name: _number(value, "local loss coefficient", nonnegative=True)
                    for name, value in loss_coefficients.items()
                }
            )
        commands = dict(self.default_openings)
        if openings is not None:
            if set(openings) - set(commands) or any(type(v) is not bool for v in openings.values()):
                raise ValueError("Known binary link openings required")
            commands.update(openings)
        heat_w = dict(heat_w or {})
        utility_temperature_k = dict(utility_temperature_k or {})
        if set(heat_w) - set(self.names) or set(utility_temperature_k) - set(self.names):
            raise ValueError("Unknown thermal node")
        heat = [_number(heat_w.get(n, 0.0), "heat rate") for n in self.names]
        utility = [
            _number(
                utility_temperature_k.get(n, state.nodes[i][1].temperature_k),
                "utility temperature",
                positive=True,
            )
            for i, n in enumerate(self.names)
        ]
        if any(q and n not in utility_temperature_k for n, q in zip(self.names, heat)):
            raise ValueError("Nonzero heat requires explicit utility temperature")
        current = state
        dt = min(state.next_step_s, duration_s)
        steps = []
        attempts = 0
        while current.time_s < end:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("HEM interval budget exhausted")
            dt = interval_step_s(current.time_s, end, dt)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("HEM substep below time resolution")
            try:
                coarse, _ = self._step(current, dt, commands, heat, utility, losses)
                half, one = self._step(current, dt / 2, commands, heat, utility, losses)
                fine, two = self._step(half, dt / 2, commands, heat, utility, losses)
                error = self._error(coarse, fine)
                if error > 1:
                    raise HydraulicConvergenceError("HEM temporal error exceeded")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                dt /= 2
                if dt < self.minimum_step_s:
                    raise HydraulicConvergenceError("HEM minimum step exhausted") from exc
                continue
            nextdt = min(
                self.maximum_step_s, max(self.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            )
            current = replace(
                fine,
                time_s=end if dt >= end - current.time_s else min(end, fine.time_s),
                next_step_s=nextdt,
            )
            steps.extend((one, two))
            dt = nextdt
        heat = math.fsum(s["ambient_heat_j"] + s["external_heat_j"] for s in steps)
        work = math.fsum(s["external_work_j"] for s in steps)
        residual = self._ledger(state, current, heat + work)
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "conservation_residuals": residual,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
            "openings": commands,
            "loss_coefficients": losses,
            "total_heat_j": heat,
            "total_work_j": work,
        }
