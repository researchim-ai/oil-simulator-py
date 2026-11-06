"""Piecewise surveyed centreline, compiled into ordinary PR finite volumes."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .liquid import _name, _number
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PRTerrainPipe:
    name: str
    source: str
    target: str
    diameter_m: float
    segment_lengths_m: tuple
    elevations_m: tuple
    cells_per_segment: tuple
    dynamic_viscosity_pa_s: float = 0.0
    roughness_m: float = 0.0
    wall: VesselWall | None = None
    bend_loss_coefficients: tuple = ()

    def __post_init__(self):
        for key in ("name", "source", "target"):
            _name(getattr(self, key))
        _number(self.diameter_m, "terrain pipe diameter", positive=True)
        for key in ("dynamic_viscosity_pa_s", "roughness_m"):
            _number(getattr(self, key), key, nonnegative=True)
        for key in (
            "segment_lengths_m",
            "elevations_m",
            "cells_per_segment",
            "bend_loss_coefficients",
        ):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        lengths, heights, counts = self.segment_lengths_m, self.elevations_m, self.cells_per_segment
        if not lengths or len(heights) != len(lengths) + 1 or len(counts) != len(lengths):
            raise ValueError("Survey needs one more elevation than segment lengths and cell counts")
        for length, a, b, n in zip(lengths, heights, heights[1:], counts):
            _number(length, "survey segment length", positive=True)
            _number(a, "survey elevation")
            _number(b, "survey elevation")
            if (
                (abs(b - a) > length and not math.isclose(abs(b - a), length, rel_tol=1e-12))
                or type(n) is not int
                or n < 1
            ):
                raise ValueError(
                    "Survey length must cover elevation change with positive cell count"
                )
        if not 2 <= sum(counts) <= 1024 or self.roughness_m >= self.diameter_m / 2:
            raise ValueError("Terrain route needs 2..1024 cells and admissible roughness")
        if self.bend_loss_coefficients and len(self.bend_loss_coefficients) != len(lengths) - 1:
            raise ValueError("One explicit loss coefficient per interior survey vertex required")
        for value in self.bend_loss_coefficients:
            _number(value, "bend loss coefficient", nonnegative=True)
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected whole-route VesselWall")

    @property
    def cells(self):
        return sum(self.cells_per_segment)

    @property
    def length_m(self):
        return math.fsum(self.segment_lengths_m)

    source_height_m = None
    target_height_m = None
    distribution_strength = 0.0
    buoyancy_relaxation_s = 0.0
    drift_flux = None

    def cell_routes(self):
        rows = []
        losses = self.bend_loss_coefficients or (0.0,) * (len(self.segment_lengths_m) - 1)
        for segment, (length, a, b, n) in enumerate(
            zip(
                self.segment_lengths_m,
                self.elevations_m,
                self.elevations_m[1:],
                self.cells_per_segment,
            )
        ):
            dl, dz = length / n, (b - a) / n
            for j in range(n):
                rows.append(
                    (dl, a + (j + 0.5) * dz, dz, losses[segment - 1] if segment and j == 0 else 0.0)
                )
        return rows

    def split_at_levels(self, levels_m):
        """Place survey vertices at known initial interfaces, retaining bend losses.

        Splitting changes only the mesh. Levels are not held fixed after initialisation.
        """
        levels = sorted({_number(h, "interface elevation") for h in levels_m})
        lengths, heights, counts, losses = [], [self.elevations_m[0]], [], []
        original = self.bend_loss_coefficients or (0.0,) * (len(self.segment_lengths_m) - 1)
        for i, (length, a, b, n) in enumerate(
            zip(
                self.segment_lengths_m,
                self.elevations_m,
                self.elevations_m[1:],
                self.cells_per_segment,
            )
        ):
            cuts = sorted((h - a) / (b - a) for h in levels if min(a, b) < h < max(a, b))
            fractions = [0.0, *cuts, 1.0]
            for j, (left, right) in enumerate(zip(fractions, fractions[1:])):
                if lengths:
                    losses.append(original[i - 1] if j == 0 else 0.0)
                lengths.append(length * (right - left))
                heights.append(a + right * (b - a))
                counts.append(max(1, math.ceil(n * (right - left) - 1e-12)))
        return replace(
            self,
            segment_lengths_m=tuple(lengths),
            elevations_m=tuple(heights),
            cells_per_segment=tuple(counts),
            bend_loss_coefficients=tuple(losses),
        )


def initialize_from_gravity_profile(model, column, state):
    """Sample a resolved gravitational equilibrium into the actual finite volumes.

    Mesh faces must resolve interfaces; a centre sample is never used to silently
    represent an interface crossing a pipe cell.
    """
    if model.fluid.to_dict() != column.fluid.to_dict():
        raise ValueError("Initial gravity profile must use the same EOS")
    if not math.isclose(model.gravity_m_s2, column.gravity_m_s2, abs_tol=1e-12):
        raise ValueError("Initial gravity profile must use the same gravity")
    column.inspect(state)
    interfaces = [column.elevation_m + h for h in state.interface_heights_m]
    specs = {}
    shared_pressures = {}
    for node in model.nodes:
        low = node.elevation_m - abs(node.elevation_change_m) / 2
        high = node.elevation_m + abs(node.elevation_change_m) / 2
        if any(low + 1e-10 < h < high - 1e-10 for h in interfaces):
            raise ValueError("Split pipe cells at initial phase interfaces")
        point = column.sample(state, node.elevation_m - column.elevation_m)
        specs[node.name] = {k: point[k] for k in ("pressure_pa", "temperature_k", "mole_fractions")}
        if node.hydrostatic_reconstruction and node.elevation_change_m:
            faces = []
            for port, normal in (("left", -1), ("right", 1)):
                height = round(node.elevation_m + normal * node.elevation_change_m / 2, 12)
                sample = column.sample(
                    state,
                    height - column.elevation_m,
                    side="below" if node.elevation_m < height else "above",
                )
                pressure = shared_pressures.setdefault(height, sample["pressure_pa"])
                faces.append((port, pressure, sample["mole_fractions"]))
            specs[node.name]["hydrostatic_face_guesses"] = faces
    return model.initialize(specs)


def initialize_hydrostatic_transport(
    model, pressure_pa, temperature_k, mole_fractions, *, anchor_elevation_m=0.0
):
    """Chemical equilibrium initial data; never a reference profile during stepping.

    Supports a single homogeneous phase per centre, including separated pure
    liquid/vapour layers when the anchor is a saturation interface. Rejects
    centres whose equilibrium needs several phases instead of guessing levels.
    """
    from .liquid import PropertyModelError
    from .peng_robinson import R
    from .pr_inclined_pipe import solve_chemical_nodes

    f = model.fluid
    pressure = _number(pressure_pa, "anchor pressure", positive=True)
    temperature = _number(temperature_k, "hydrostatic temperature", positive=True)
    anchor = _number(anchor_elevation_m, "anchor elevation")
    values = tuple(_number(x, "mole fraction", nonnegative=True) for x in mole_fractions)
    if len(values) != len(f.names) or not math.isclose(
        sum(values), 1.0, abs_tol=1e-12, rel_tol=0.0
    ):
        raise ValueError("Hydrostatic composition must match the EOS and sum to one")
    if (
        not f.min_pressure_pa <= pressure <= f.max_pressure_pa
        or not f.min_temperature_k <= temperature <= f.max_temperature_k
    ):
        raise ValueError("Hydrostatic anchor outside EOS domain")
    x = f.tensor(values)
    active = (x > 0).nonzero().flatten()
    q = f.tensor_homogeneous(f.tensor(pressure), f.tensor(temperature), x, "stable")
    mu = x[active].log() + q["log_fugacity_coefficients"][active] + math.log(pressure / f.pref)
    ql = f.tensor_homogeneous(f.tensor(pressure), f.tensor(temperature), x, "liquid")
    qg = f.tensor_homogeneous(f.tensor(pressure), f.tensor(temperature), x, "vapor")
    specs = {}
    batch_size = max(1, 128 // len(active))
    for start in range(0, len(model.nodes), batch_size):
        nodes = model.nodes[start : start + batch_size]
        dz = f.tensor([anchor - node.elevation_m for node in nodes])
        targets = mu[None, :] + f.molar_masses[active][None, :] * model.gravity_m_s2 * dz[
            :, None
        ] / (R * temperature)
        density = torch.where(dz >= 0, ql["density_kg_m3"], qg["density_kg_m3"])
        pp = (pressure + density * model.gravity_m_s2 * dz).clamp(
            f.min_pressure_pa, f.max_pressure_pa
        )
        ratios = (x[active][:-1] / x[active][-1]).log()
        initial = torch.cat((pp.log()[:, None], ratios[None, :].expand(len(nodes), -1)), 1)
        solved = solve_chemical_nodes(
            f, initial, f.tensor([temperature] * len(nodes)), targets, active
        )
        for node, row in zip(nodes, solved):
            fractions = row.new_zeros(len(f.names)).index_copy(
                0, active, torch.cat((row[1:], row.new_zeros(1))).softmax(0)
            )
            p = float(row[0].exp().cpu())
            equilibrium = f.flash_pt(p, temperature, fractions)
            if equilibrium["phase_count"] != 1:
                raise PropertyModelError(
                    "Hydrostatic centre needs resolved layers; refine/split the mesh"
                )
            specs[node.name] = {
                "pressure_pa": p,
                "temperature_k": temperature,
                "mole_fractions": fractions.cpu().tolist(),
            }
    return model.initialize(specs)
