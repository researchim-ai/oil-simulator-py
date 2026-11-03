"""Conservative thermally coupled three-phase separator trains in Torch.

Frozen immiscible phases; common temperature, stratified hydrostatics,
finite wall storage, pressure-driven phase ports and optional pickup affinities.
No hydrocarbon flash or automatic coupling to black-oil stock components.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    _name,
    _number,
)
from .operation import _digest
from .separator_geometry import SeparatorGeometry
from .three_phase_properties import (
    ThreePhaseFluid,
    ThreePhaseStream,
    disk_area_moment,
    frozen_nozzle,
    level_tensor,
)
from .torch_linear import TorchNewton, checked_device
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class ThreePhaseVessel:
    name: str
    geometry: SeparatorGeometry
    min_pressure_pa: float
    max_pressure_pa: float
    elevation_m: float = 0.0
    minimum_gas_volume_fraction: float = 1e-4
    wall: VesselWall | None = None

    def __post_init__(self):
        _name(self.name)
        if not isinstance(self.geometry, SeparatorGeometry):
            raise TypeError("Expected SeparatorGeometry")
        if self.wall is not None and not isinstance(self.wall, VesselWall):
            raise TypeError("Expected VesselWall")
        _number(self.elevation_m, "elevation_m")
        for key in ("min_pressure_pa", "max_pressure_pa", "minimum_gas_volume_fraction"):
            _number(getattr(self, key), key, positive=True)
        if self.min_pressure_pa >= self.max_pressure_pa or self.minimum_gas_volume_fraction >= 1:
            raise ValueError("Invalid separator pressure or gas-space domain")


@dataclass(frozen=True)
class ThreePhasePort:
    name: str
    source: str
    phase: str
    height_m: float
    diameter_m: float
    discharge_coefficient: float
    target: str | None = None
    target_phase: str | None = None
    target_height_m: float = 0.0
    back_pressure_pa: float | None = None
    opening: float = 1.0
    allow_reverse: bool = False
    pickup_affinities: tuple | None = None
    target_pickup_affinities: tuple | None = None
    control_metric: str | None = None
    setpoint: float | None = None
    opening_gain: float = 0.0
    opening_bias: float = 0.0
    max_mass_rate_kg_s: float | None = None

    def __post_init__(self):
        for key in ("name", "source"):
            _name(getattr(self, key))
        if self.phase not in ("oil", "water", "gas"):
            raise ValueError("Unknown separator port phase")
        if self.target_phase is None:
            object.__setattr__(self, "target_phase", self.phase)
        if self.target_phase not in ("oil", "water", "gas"):
            raise ValueError("Unknown reverse port phase")
        if self.target is not None:
            _name(self.target)
        if self.target == self.source:
            raise ValueError("Separator port cannot connect a vessel to itself")
        if (self.target is None) == (self.back_pressure_pa is None):
            raise ValueError("Choose target vessel OR back pressure")
        if self.back_pressure_pa is not None:
            _number(self.back_pressure_pa, "back_pressure_pa", positive=True)
        if not isinstance(self.allow_reverse, bool) or self.allow_reverse and self.target is None:
            raise ValueError("Reverse flow requires two finite vessels")
        for key in ("height_m", "target_height_m", "opening", "opening_gain", "opening_bias"):
            _number(getattr(self, key), key, nonnegative=True)
        for key in ("diameter_m", "discharge_coefficient"):
            _number(getattr(self, key), key, positive=True)
        if max(self.opening, self.opening_bias, self.discharge_coefficient) > 1:
            raise ValueError("Opening and discharge coefficient must not exceed one")
        if self.max_mass_rate_kg_s is not None:
            _number(self.max_mass_rate_kg_s, "max_mass_rate_kg_s", positive=True)
        if self.control_metric not in (None, "pressure_pa", "liquid_level_m", "interface_level_m"):
            raise ValueError("Unknown implicit valve control metric")
        if self.control_metric is None:
            if self.setpoint is not None or self.opening_gain or self.opening_bias:
                raise ValueError("Feedback parameters require a control metric")
        elif self.setpoint is None:
            raise ValueError("Feedback requires a setpoint")
        else:
            _number(self.setpoint, "setpoint", nonnegative=True)
        for key, phase in (
            ("pickup_affinities", self.phase),
            ("target_pickup_affinities", self.target_phase),
        ):
            values = getattr(self, key)
            if values is None:
                values = tuple(float(p == phase) for p in ("oil", "water", "gas"))
            values = tuple(_number(v, key, nonnegative=True) for v in values)
            if len(values) != 3 or values[("oil", "water", "gas").index(phase)] <= 0:
                raise ValueError(
                    "Port needs three pickup affinities and a positive carrier affinity"
                )
            object.__setattr__(self, key, values)


@dataclass(frozen=True)
class ThreePhaseState:
    vessels: tuple
    time_s: float
    model_id: str

    def __post_init__(self):
        _number(self.time_s, "time_s", nonnegative=True)
        if not isinstance(self.model_id, str):
            raise TypeError("State model identity must be a string")
        items = (
            tuple(self.vessels.items()) if isinstance(self.vessels, dict) else tuple(self.vessels)
        )
        if len({n for n, _ in items}) != len(items):
            raise ValueError("Duplicate vessel in state")
        result = []
        for name, values in items:
            _name(name)
            if len(values) != 5:
                raise ValueError("State needs oil/water/gas masses, fluid T, wall T")
            values = tuple(
                _number(v, "phase_mass_kg", nonnegative=True)
                if i < 3
                else _number(v, "temperature_k", positive=True)
                for i, v in enumerate(values)
            )
            result.append((name, values))
        object.__setattr__(self, "vessels", tuple(sorted(result)))

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"vessels", "time_s", "model_id"}:
            raise ValueError("Invalid three-phase state fields")
        return cls(**data)

    def to_dict(self):
        return {
            "vessels": {n: list(v) for n, v in self.vessels},
            "time_s": self.time_s,
            "model_id": self.model_id,
        }


class ThreePhaseSeparatorNetwork:
    def __init__(
        self,
        fluid,
        vessels,
        ports=(),
        *,
        device="cpu",
        gravity_m_s2=9.80665,
        maximum_dt_s=0.1,
        minimum_dt_s=1e-7,
        max_fractional_inventory_change=0.1,
        max_temperature_change_k=2.0,
        temporal_error_control=True,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.001,
    ):
        if not isinstance(fluid, ThreePhaseFluid):
            raise TypeError("Expected ThreePhaseFluid")
        self.fluid = fluid
        self.vessels = tuple(vessels)
        self.ports = tuple(ports)
        if not self.vessels or not all(isinstance(v, ThreePhaseVessel) for v in self.vessels):
            raise TypeError("Expected nonempty ThreePhaseVessel sequence")
        if not all(isinstance(p, ThreePhasePort) for p in self.ports):
            raise TypeError("Expected ThreePhasePort sequence")
        self.names = tuple(v.name for v in self.vessels)
        self.index = {n: i for i, n in enumerate(self.names)}
        if len(self.index) != len(self.names) or len({p.name for p in self.ports}) != len(
            self.ports
        ):
            raise ValueError("Duplicate vessel/port name")
        self.device = checked_device(device)
        self.dtype = torch.float64
        self.gravity_m_s2 = _number(gravity_m_s2, "gravity_m_s2", nonnegative=True)
        for key, value in (
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("max_fractional_inventory_change", max_fractional_inventory_change),
            ("max_temperature_change_k", max_temperature_change_k),
            ("relative_tolerance", relative_tolerance),
            ("temperature_tolerance_k", temperature_tolerance_k),
        ):
            setattr(self, key, _number(value, key, positive=True))
        if (
            minimum_dt_s > maximum_dt_s
            or max_fractional_inventory_change > 1
            or not isinstance(temporal_error_control, bool)
        ):
            raise ValueError("Invalid separator integration controls")
        self.temporal_error_control = temporal_error_control
        for v in self.vessels:
            if (
                self.gravity_m_s2
                * v.geometry.height_m
                / (fluid.gas_constant_j_kg_k * fluid.min_temperature_k)
                > 0.1
            ):
                raise ValueError("Separator gas hydrostatic exponent exceeds model domain")
        for p in self.ports:
            if p.source not in self.index or p.target is not None and p.target not in self.index:
                raise ValueError("Unknown port vessel")
            for node, height in ((p.source, p.height_m), (p.target, p.target_height_m)):
                if (
                    node is not None
                    and not p.diameter_m / 2
                    <= height
                    <= self.vessels[self.index[node]].geometry.height_m - p.diameter_m / 2
                ):
                    raise ValueError("Port disk must fit within separator height")
            if (
                p.control_metric is not None
                and "level" in p.control_metric
                and p.setpoint > self.vessels[self.index[p.source]].geometry.height_m
            ):
                raise ValueError("Level setpoint outside geometry")
        self.cv = self.tensor(fluid.heat_capacities)
        self.rho_l = self.tensor((fluid.oil_density_kg_m3, fluid.water_density_kg_m3))
        # Positive Gaussian quadrature for the small barometric gas correction.
        x, w = np.polynomial.legendre.leggauss(32)
        self.quad_x = self.tensor((x + 1) / 2)
        self.quad_w = self.tensor(w / 2)
        pairs = set()
        adjacency = {i: {i} for i in range(len(self.names))}
        for port in self.ports:
            if port.target is not None:
                a, b = self.index[port.source], self.index[port.target]
                adjacency[a].add(b)
                adjacency[b].add(a)
        for a, neighbors in adjacency.items():
            for b in neighbors:
                pairs.update((5 * a + i, 5 * b + j) for i in range(5) for j in range(5))
        ordered = sorted(pairs)
        self._pattern = tuple(a for a, b in ordered), tuple(b for a, b in ordered)
        affinities = {
            a
            for port in self.ports
            for a in (port.pickup_affinities, port.target_pickup_affinities)
        }
        self._pickup = {a: self.tensor(a) for a in affinities}
        constants = {self.fluid.min_temperature_k, self.fluid.gas_min_temperature_k}
        for port in self.ports:
            constants.update((port.opening, port.height_m, port.target_height_m))
            if port.back_pressure_pa is not None:
                constants.add(port.back_pressure_pa)
            if port.target is not None:
                constants.add(
                    port.target_height_m + self.vessels[self.index[port.target]].elevation_m
                )
            constants.add(port.height_m + self.vessels[self.index[port.source]].elevation_m)
        self._scalars = {float(v): self.tensor(v) for v in constants}
        self._mass_floor = self.tensor(
            [
                [
                    v.geometry.volume_m3 * fluid.oil_density_kg_m3 * 0.001,
                    v.geometry.volume_m3 * fluid.water_density_kg_m3 * 0.001,
                    0.001,
                ]
                for v in self.vessels
            ]
        )
        self._temperature_scale = torch.full(
            (len(self.names), 2),
            fluid.reference_temperature_k,
            dtype=self.dtype,
            device=self.device,
        )
        self._lower_coords = self.tensor(
            [
                [
                    0,
                    0,
                    0,
                    fluid.min_temperature_k,
                    v.wall.min_temperature_k if v.wall else fluid.min_temperature_k,
                ]
                for v in self.vessels
            ]
        )
        self._upper_coords = self.tensor(
            [
                [
                    v.geometry.volume_m3 * fluid.oil_density_kg_m3,
                    v.geometry.volume_m3 * fluid.water_density_kg_m3,
                    v.max_pressure_pa
                    * v.geometry.volume_m3
                    / (fluid.gas_constant_j_kg_k * fluid.min_temperature_k),
                    fluid.max_temperature_k,
                    v.wall.max_temperature_k if v.wall else fluid.max_temperature_k,
                ]
                for v in self.vessels
            ]
        )
        self._temporal_mass_floor = self.tensor([1.0, 1.0, 0.001])
        self._change_limits = self.tensor(
            [
                [
                    self.max_temperature_change_k,
                    min(self.max_temperature_change_k, v.wall.max_temperature_change_k)
                    if v.wall
                    else self.max_temperature_change_k,
                ]
                for v in self.vessels
            ]
        )
        self.newton = TorchNewton(self.device, maximum_iterations=18)
        self.newton.host_convergence = True
        self._compiled_id = self.model_id

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    @property
    def model_id(self):
        return _digest({"model": "immiscible_thermal_separator_network_v1", **self.to_dict()})

    def to_dict(self):
        return {
            "fluid": asdict(self.fluid),
            "vessels": [asdict(v) for v in self.vessels],
            "ports": [asdict(p) for p in self.ports],
            "device": str(self.device),
            **{
                k: getattr(self, k)
                for k in (
                    "gravity_m_s2",
                    "maximum_dt_s",
                    "minimum_dt_s",
                    "max_fractional_inventory_change",
                    "max_temperature_change_k",
                    "temporal_error_control",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                )
            },
        }

    @classmethod
    def from_dict(cls, spec):
        data = dict(spec)
        fluid = ThreePhaseFluid(**data.pop("fluid"))
        vessels = []
        for item in data.pop("vessels"):
            item = dict(item)
            item["geometry"] = SeparatorGeometry(**item["geometry"])
            if item.get("wall") is not None:
                item["wall"] = VesselWall(**item["wall"])
            vessels.append(ThreePhaseVessel(**item))
        ports = [ThreePhasePort(**p) for p in data.pop("ports", ())]
        return cls(fluid, vessels, ports, **data)

    def _geometry(self, coords):
        result = []
        f = self.fluid
        g = self.gravity_m_s2
        for v, c in zip(self.vessels, coords):
            masses, t, tw = c[:3], c[3], c[4]
            vo, vw = masses[0] / f.oil_density_kg_m3, masses[1] / f.water_density_kg_m3
            volume = v.geometry.volume_m3
            hw = level_tensor(vw, v.geometry)
            hl = level_tensor(vo + vw, v.geometry)
            vg = volume - vo - vw
            safe_vg = vg.clamp_min(volume * 1e-12)
            height = v.geometry.height_m
            z = hl + (height - hl) * self.quad_x
            if v.geometry.orientation == "vertical":
                cross = torch.ones_like(z) * volume / height
                mw = 0.5 * vw * hw
                ml = 0.5 * (vo + vw) * hl
            else:
                cross = 2 * v.geometry.length_m * torch.sqrt((z * (height - z)).clamp_min(1e-30))
                _, mw = disk_area_moment(hw, height)
                mw = mw * v.geometry.length_m
                _, ml = disk_area_moment(hl, height)
                ml = ml * v.geometry.length_m
            exponent = g * (height - z) / (f.gas_constant_j_kg_k * t)
            density_weights = cross * self.quad_w * torch.exp(exponent)
            norm = density_weights.sum().clamp_min(1e-30)
            zg = (z * density_weights).sum() / norm
            weighted_volume = safe_vg * norm / (cross * self.quad_w).sum().clamp_min(1e-30)
            pressure = masses[2] * f.gas_constant_j_kg_k * t / weighted_volume
            interface_pressure = pressure * torch.exp(
                g * (height - hl) / (f.gas_constant_j_kg_k * t)
            )
            potential = g * (
                f.water_density_kg_m3 * mw
                + f.oil_density_kg_m3 * (ml - mw)
                + masses[2] * zg
                + v.elevation_m * masses.sum()
            )
            energy = (masses * self.cv).sum() * (t - f.reference_temperature_k) + potential
            entropy = (masses[:2] * self.cv[:2]).sum() * torch.log(
                t / f.reference_temperature_k
            ) + masses[2] * (
                (f.gas_cv_j_kg_k + f.gas_constant_j_kg_k) * torch.log(t / f.reference_temperature_k)
                - f.gas_constant_j_kg_k
                * torch.log(pressure.clamp_min(1e-30) / f.reference_pressure_pa)
                - g * (height - zg) / t
            )
            wall_energy = (
                v.wall.heat_capacity_j_k * (tw - v.wall.reference_temperature_k)
                if v.wall
                else t * 0
            )
            result.append(
                {
                    "mass": masses,
                    "temperature": t,
                    "wall_temperature": tw,
                    "pressure_pa": pressure,
                    "interface_pressure_pa": interface_pressure,
                    "bottom_pressure_pa": interface_pressure
                    + g * (f.oil_density_kg_m3 * (hl - hw) + f.water_density_kg_m3 * hw),
                    "oil_volume_m3": vo,
                    "water_volume_m3": vw,
                    "interface_level_m": hw,
                    "liquid_level_m": hl,
                    "gas_volume_m3": vg,
                    "potential_energy_j": potential,
                    "energy_j": energy,
                    "entropy_j_k": entropy,
                    "wall_energy_j": wall_energy,
                }
            )
        return result

    def _pressure_at(self, node, height, values):
        v = self.vessels[self.index[node]]
        q = values[self.index[node]]
        f = self.fluid
        g = self.gravity_m_s2
        hl, hw = q["liquid_level_m"], q["interface_level_m"]
        gas = q["pressure_pa"] * torch.exp(
            g * (v.geometry.height_m - height) / (f.gas_constant_j_kg_k * q["temperature"])
        )
        liquid = q["interface_pressure_pa"] + g * (
            f.oil_density_kg_m3 * (hl - torch.maximum(hw, height)).clamp_min(0)
            + f.water_density_kg_m3 * (hw - height).clamp_min(0)
        )
        return torch.where(height >= hl, gas, liquid)

    def _donor(self, port, node, phase, height, affinities, values):
        q = values[self.index[node]]
        v = self.vessels[self.index[node]]
        limits = (
            q["temperature"] * 0,
            q["interface_level_m"],
            q["liquid_level_m"],
            q["temperature"] * 0 + v.geometry.height_m,
        )
        k = {"water": 0, "oil": 1, "gas": 2}[phase]
        bottom = height - port.diameter_m / 2
        alo, mlo = disk_area_moment(limits[k] - bottom, port.diameter_m)
        ahi, mhi = disk_area_moment(limits[k + 1] - bottom, port.diameter_m)
        area = (ahi - alo).clamp_min(0)
        zeff = bottom + (mhi - mlo) / torch.where(area > 0, area, torch.ones_like(area))
        zeff = torch.where(area > 0, zeff, q["temperature"] * 0 + height)
        p = self._pressure_at(node, zeff, values)
        sampled = q["mass"] * self._pickup[affinities]
        total = sampled.sum()
        weights = sampled / torch.where(total > 0, total, torch.ones_like(total))
        h = (weights * self.fluid.phase_enthalpies(p, q["temperature"])).sum()
        s = (weights * self.fluid.phase_entropies(p, q["temperature"])).sum()
        return p, q["temperature"], area, weights, h, s, zeff + v.elevation_m

    def _rates(self, coords, values, feeds, heat, openings):
        mass = [c[:3] * 0 for c in coords]
        energy = [c[3] * 0 for c in coords]
        ext_mass = coords[0, :3] * 0
        ext_energy = coords[0, 3] * 0
        ext_entropy = ext_energy
        flow_records = []
        valid = torch.ones((), dtype=torch.bool, device=self.device)
        for i, streams in enumerate(feeds):
            for rates, e, s, p, height in streams:
                receiver = self._pressure_at(self.names[i], height, values)
                valid = valid & ((rates.sum() == 0) | (p >= receiver))
                mass[i] = mass[i] + rates
                energy[i] = energy[i] + e
                ext_mass = ext_mass + rates
                ext_energy = ext_energy + e
                ext_entropy = ext_entropy + s
        for port in self.ports:
            source = self.index[port.source]
            manual = openings.get(port.name)
            if manual is not None:
                opening = manual
            elif port.control_metric:
                metric = values[source][port.control_metric]
                opening = (port.opening_bias + port.opening_gain * (metric - port.setpoint)).clamp(
                    0, 1
                )
            else:
                opening = self._scalars[port.opening]
            donor = self._donor(
                port, port.source, port.phase, port.height_m, port.pickup_affinities, values
            )
            if port.target is None:
                back = self._scalars[port.back_pressure_pa]
                target_z = self._scalars[port.target_height_m]
            else:
                back = self._pressure_at(port.target, self._scalars[port.target_height_m], values)
                target_z = self._scalars[
                    port.target_height_m + self.vessels[self.index[port.target]].elevation_m
                ]
            p, t, area, w, h, s, z = donor
            flux, throat, tt, choked = frozen_nozzle(
                p,
                t,
                back,
                w,
                self.fluid,
                target_z - z,
                self.gravity_m_s2,
                mixed=sum(a > 0 for a in port.pickup_affinities) > 1,
            )
            rate = opening * port.discharge_coefficient * area * flux
            component = rate * w
            total_e = rate * (h + self.gravity_m_s2 * z)
            entropy = rate * s
            minimum_t = torch.where(
                w[:2].sum() > 1e-12,
                self._scalars[self.fluid.min_temperature_k],
                self._scalars[self.fluid.gas_min_temperature_k],
            )
            valid = valid & (
                (rate == 0) | ((tt >= minimum_t) & (tt <= self.fluid.max_temperature_k))
            )
            reverse = rate * 0
            if port.allow_reverse:
                rp, rt, ra, rw, rh, rs, rz = self._donor(
                    port,
                    port.target,
                    port.target_phase,
                    port.target_height_m,
                    port.target_pickup_affinities,
                    values,
                )
                rback = self._pressure_at(port.source, self._scalars[port.height_m], values)
                rf, rth, rtt, rch = frozen_nozzle(
                    rp,
                    rt,
                    rback,
                    rw,
                    self.fluid,
                    self._scalars[port.height_m + self.vessels[source].elevation_m] - rz,
                    self.gravity_m_s2,
                    mixed=sum(a > 0 for a in port.target_pickup_affinities) > 1,
                )
                reverse = opening * port.discharge_coefficient * ra * rf
                valid = valid & ~((rate > 1e-10) & (reverse > 1e-10))
                rate = rate - reverse
                component = component - reverse * rw
                total_e = total_e - reverse * (rh + self.gravity_m_s2 * rz)
                entropy = entropy - reverse * rs
                valid = valid & (
                    (reverse == 0)
                    | (
                        rtt
                        >= torch.where(
                            rw[:2].sum() > 1e-12,
                            self._scalars[self.fluid.min_temperature_k],
                            self._scalars[self.fluid.gas_min_temperature_k],
                        )
                    )
                )
                throat = torch.where(reverse > 0, rth, throat)
                tt = torch.where(reverse > 0, rtt, tt)
                choked = torch.where(reverse > 0, rch, choked)
            if port.allow_reverse:
                valid = valid & ((reverse == 0) | (rtt <= self.fluid.max_temperature_k))
            if port.max_mass_rate_kg_s is not None:
                valid = valid & (rate.abs() <= port.max_mass_rate_kg_s)
            mass[source] = mass[source] - component
            energy[source] = energy[source] - total_e
            if port.target is not None:
                target = self.index[port.target]
                mass[target] = mass[target] + component
                energy[target] = energy[target] + total_e
            else:
                ext_mass = ext_mass - component
                ext_energy = ext_energy - total_e
                ext_entropy = ext_entropy - entropy
            flow_records.append(
                torch.cat(
                    (
                        rate.reshape(1),
                        component,
                        total_e.reshape(1),
                        entropy.reshape(1),
                        opening.reshape(1),
                        throat.reshape(1),
                        tt.reshape(1),
                        choked.to(self.dtype).reshape(1),
                    )
                )
            )
        wall_rates = []
        ambient_entropy = coords[0, 3] * 0
        for i, (v, c) in enumerate(zip(self.vessels, coords)):
            energy[i] = energy[i] + heat[i]
            ext_energy = ext_energy + heat[i]
            if v.wall:
                exchange = v.wall.fluid_ua_w_k * (c[4] - c[3])
                ambient = v.wall.ambient_ua_w_k * (v.wall.ambient_temperature_k - c[4])
                energy[i] = energy[i] + exchange
                wall_rates.append(-exchange + ambient)
                ext_energy = ext_energy + ambient
                ambient_entropy = ambient_entropy + ambient / v.wall.ambient_temperature_k
            else:
                wall_rates.append(c[3] * 0)
        return (
            torch.stack(mass),
            torch.stack(energy),
            torch.stack(wall_rates),
            ext_mass,
            ext_energy,
            ext_entropy + ambient_entropy,
            flow_records,
            valid,
        )

    def _validate(self, coords, values):
        valid = torch.isfinite(coords).all() & (coords[:, :3] >= 0).all()
        f = self.fluid
        valid = (
            valid
            & (coords[:, 3] >= f.min_temperature_k).all()
            & (coords[:, 3] <= f.max_temperature_k).all()
        )
        for v, c, q in zip(self.vessels, coords, values):
            valid = (
                valid
                & (q["pressure_pa"] >= v.min_pressure_pa)
                & (q["bottom_pressure_pa"] <= v.max_pressure_pa)
            )
            valid = valid & (
                q["gas_volume_m3"] >= v.minimum_gas_volume_fraction * v.geometry.volume_m3
            )
            valid = valid & (
                q["interface_pressure_pa"] / (f.gas_constant_j_kg_k * c[3]) < f.oil_density_kg_m3
            )
            if v.wall:
                valid = (
                    valid & (c[4] >= v.wall.min_temperature_k) & (c[4] <= v.wall.max_temperature_k)
                )
            else:
                valid = valid & ((c[4] - c[3]).abs() <= 1e-8)
        return valid

    def initialize(self, states, time_s=0.0):
        if set(states) != set(self.names):
            raise ValueError("Initial conditions require every vessel")
        rows = []
        for v in self.vessels:
            d = dict(states[v.name])
            required = {"pressure_pa", "temperature_k", "oil_volume_m3", "water_volume_m3"}
            if not required <= set(d) or set(d) - required - {"wall_temperature_k"}:
                raise ValueError("Unknown or missing separator initial field")
            _number(d["pressure_pa"], "pressure_pa", positive=True)
            t = _number(d["temperature_k"], "temperature_k", positive=True)
            vw = _number(d["water_volume_m3"], "water_volume_m3", nonnegative=True)
            vo = _number(d["oil_volume_m3"], "oil_volume_m3", nonnegative=True)
            if vo + vw > (1 - v.minimum_gas_volume_fraction) * v.geometry.volume_m3:
                raise EquipmentOperatingError("Initial liquid volumes leave insufficient gas space")
            row = self.tensor(
                [
                    vo * self.fluid.oil_density_kg_m3,
                    vw * self.fluid.water_density_kg_m3,
                    1.0,
                    t,
                    d.get("wall_temperature_k", t),
                ]
            )
            rows.append(row)
        coords = torch.stack(rows)
        info = self._geometry(coords)
        for i, d in enumerate(states[n] for n in self.names):
            coords[i, 2] = float(d["pressure_pa"]) / info[i]["pressure_pa"]
        state = ThreePhaseState(
            dict(zip(self.names, coords.detach().cpu().tolist())), time_s, self.model_id
        )
        self.inspect(state)
        return state

    def _coords(self, state):
        if (
            not isinstance(state, ThreePhaseState)
            or state.model_id != self.model_id
            or self.model_id != self._compiled_id
        ):
            raise ValueError("Three-phase state/model mismatch")
        if tuple(n for n, _ in state.vessels) != tuple(sorted(self.names)):
            raise ValueError("State vessel coordinates mismatch")
        return self.tensor([dict(state.vessels)[n] for n in self.names])

    def inspect(self, state):
        coords = self._coords(state)
        values = self._geometry(coords)
        if not bool(self._validate(coords, values).detach().cpu()):
            raise EquipmentOperatingError(
                "Three-phase vessel outside pressure/temperature/inventory domain"
            )
        result = {}
        for v, c, q in zip(self.vessels, coords, values):
            result[v.name] = {
                k: float(value.detach().cpu()) for k, value in q.items() if value.numel() == 1
            }
            result[v.name]["phase_masses_kg"] = c[:3].detach().cpu().tolist()
        return result

    def _step(self, old, dt, feeds, heat, openings, utility):
        old_values = self._geometry(old)
        mass_scale = torch.maximum(old[:, :3], self._mass_floor)
        scale = torch.cat((mass_scale, self._temperature_scale), dim=1)
        energy_scale = (
            (old[:, :3] * self.cv).sum(-1) * self.fluid.reference_temperature_k
        ).clamp_min(1.0)
        lower = self._lower_coords / scale
        upper = self._upper_coords / scale

        def evaluate(x):
            c = x.reshape(old.shape) * scale
            q = self._geometry(c)
            rates = self._rates(c, q, feeds, heat, openings)
            mr, er, wr = rates[:3]
            rm = (c[:, :3] - old[:, :3] - dt * mr) / mass_scale
            re = (
                torch.stack([v["energy_j"] - o["energy_j"] for v, o in zip(q, old_values)])
                - dt * er
            )
            rw = torch.stack(
                [
                    (v.wall.heat_capacity_j_k * (c[i, 4] - old[i, 4]) - dt * wr[i])
                    / energy_scale[i]
                    if v.wall
                    else (c[i, 4] - c[i, 3]) / self.fluid.reference_temperature_k
                    for i, v in enumerate(self.vessels)
                ]
            )
            residual = torch.cat((rm, (re / energy_scale)[:, None], rw[:, None]), dim=1).reshape(-1)
            return residual, c, q, rates, re

        fn = lambda x: evaluate(x)[0]
        with torch.no_grad():
            x, residual, ok, evaluations = self.newton.solve(
                fn,
                (old / scale).reshape(-1),
                lower.reshape(-1),
                upper.reshape(-1),
                pattern=self._pattern,
                tolerance=1e-13,
            )
            residual, c, q, rates, _re = evaluate(x)
        if not bool(
            (
                ok
                & (residual.abs().max() <= 1e-11)
                & self._validate(c, q)
                & rates[-1]
                & (heat * (utility - c[:, 3]) >= -1e-10).all()
            )
            .detach()
            .cpu()
        ):
            raise HydraulicConvergenceError("Three-phase nonlinear solve/domain/port rejection")
        _mr, _er, _wr, em, ee, es, flows, _ = rates
        mass_error = (c[:, :3] - old[:, :3]).sum(0) - dt * em
        stored_change = torch.stack(
            [
                a["energy_j"] + a["wall_energy_j"] - b["energy_j"] - b["wall_energy_j"]
                for a, b in zip(q, old_values)
            ]
        ).sum()
        energy_error = stored_change - dt * ee
        entropy_change = torch.stack(
            [a["entropy_j_k"] - b["entropy_j_k"] for a, b in zip(q, old_values)]
        ).sum()
        for i, v in enumerate(self.vessels):
            if v.wall:
                entropy_change = entropy_change + v.wall.heat_capacity_j_k * torch.log(
                    c[i, 4] / old[i, 4]
                )
        entropy_error = entropy_change - dt * es
        # Utility and ambient entropy fluxes use their declared temperatures.
        entropy_error = entropy_error - dt * (heat / utility).sum()
        local_mass = c[:, :3] - old[:, :3] - dt * _mr
        local_wall = torch.stack(
            [
                v.wall.heat_capacity_j_k * (c[i, 4] - old[i, 4]) - dt * _wr[i]
                if v.wall
                else c[i, 3] * 0
                for i, v in enumerate(self.vessels)
            ]
        )
        local = torch.cat((local_mass, _re[:, None], local_wall[:, None]), dim=1)
        # Global cancellation must never hide a failed individual vessel balance.
        local_energy_change = torch.stack(
            [(a["energy_j"] - b["energy_j"]).abs() for a, b in zip(q, old_values)]
        )
        local_energy_tol = 1e-5 + 1e-9 * torch.maximum(local_energy_change, (dt * _er).abs())
        local_wall_tol = 1e-5 + 1e-9 * (dt * _wr).abs()
        local_mass_tol = 1e-8 + 1e-10 * (c[:, :3] - old[:, :3]).abs()
        if not bool(
            (
                (local_mass.abs() <= local_mass_tol).all()
                & (_re.abs() <= local_energy_tol).all()
                & (local_wall.abs() <= local_wall_tol).all()
            )
            .detach()
            .cpu()
        ):
            raise ConservationError("Three-phase local vessel balance rejected")
        diagnostic_keys = (
            "pressure_pa",
            "interface_pressure_pa",
            "bottom_pressure_pa",
            "interface_level_m",
            "liquid_level_m",
            "oil_volume_m3",
            "water_volume_m3",
            "gas_volume_m3",
            "potential_energy_j",
            "energy_j",
            "wall_energy_j",
            "entropy_j_k",
        )
        diagnostics = torch.stack([torch.stack([v[k] for k in diagnostic_keys]) for v in q])
        packet = torch.cat(
            (
                mass_error,
                energy_error.reshape(1),
                entropy_error.reshape(1),
                stored_change.reshape(1),
                torch.stack((dt * ee, residual.abs().max())),
                (dt * em).reshape(-1),
                c.reshape(-1),
                local.reshape(-1),
                diagnostics.reshape(-1),
                torch.cat(flows) if flows else c.new_empty(0),
            )
        )
        host = packet.detach().cpu().numpy()
        mass_tol = 1e-8 + 1e-10 * float((c[:, :3] - old[:, :3]).abs().sum().detach().cpu())
        energy_tol = 1e-5 + 1e-9 * max(1.0, abs(host[5]), abs(host[6]))
        if max(abs(host[:3])) > mass_tol or abs(host[3]) > energy_tol or host[4] < -1e-5:
            raise ConservationError("Three-phase mass/energy/entropy audit rejected")
        fraction = ((c[:, :3] - old[:, :3]).abs() / mass_scale).max()
        if (
            float(fraction.detach().cpu()) > self.max_fractional_inventory_change
            or float(((c[:, 3:] - old[:, 3:]).abs() / self._change_limits).max().detach().cpu()) > 1
        ):
            raise HydraulicConvergenceError("Three-phase state-change guard")
        report = {
            "dt_s": dt,
            "phase_mass_residuals_kg": host[:3].tolist(),
            "energy_residual_j": float(host[3]),
            "entropy_production_j_k": float(host[4]),
            "stored_energy_change_j": float(host[5]),
            "external_energy_in_j": float(host[6]),
            "max_scaled_residual": float(host[7]),
            "nonlinear_evaluations": evaluations,
            "external_phase_mass_in_kg": host[8:11].tolist(),
            "ports": {
                p.name: dict(
                    zip(
                        (
                            "mass_rate_kg_s",
                            "oil_mass_rate_kg_s",
                            "water_mass_rate_kg_s",
                            "gas_mass_rate_kg_s",
                            "total_energy_rate_w",
                            "entropy_rate_w_k",
                            "opening",
                            "throat_pressure_pa",
                            "throat_temperature_k",
                            "choked",
                        ),
                        host[
                            11 + 2 * c.numel() + diagnostics.numel() + i * 10 : 11
                            + 2 * c.numel()
                            + diagnostics.numel()
                            + (i + 1) * 10
                        ].tolist(),
                    )
                )
                for i, p in enumerate(self.ports)
            },
        }
        hcoords = host[11 : 11 + c.numel()].reshape(-1, 5)
        hlocal = host[11 + c.numel() : 11 + 2 * c.numel()].reshape(-1, 5)
        hdiagnostic = host[11 + 2 * c.numel() : 11 + 2 * c.numel() + diagnostics.numel()].reshape(
            len(self.names), -1
        )
        report["vessels"] = {
            n: {
                **dict(zip(diagnostic_keys, map(float, hdiagnostic[i]))),
                "phase_masses_kg": hcoords[i, :3].tolist(),
                "temperature_k": float(hcoords[i, 3]),
                "wall_temperature_k": float(hcoords[i, 4]),
                "phase_mass_residuals_kg": hlocal[i, :3].tolist(),
                "energy_residual_j": float(hlocal[i, 3]),
                "wall_energy_residual_j": float(hlocal[i, 4]),
            }
            for i, n in enumerate(self.names)
        }
        return c.detach().clone(), report

    def advance(
        self,
        state,
        duration_s,
        *,
        feeds=None,
        heat_duties_w=None,
        utility_temperatures_k=None,
        openings=None,
        max_substeps=100000,
    ):
        old = self._coords(state)
        self.inspect(state)
        duration = _number(duration_s, "duration_s", positive=True)
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Separator clock cannot advance")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Positive integer work budget required")
        feeds = dict(feeds or {})
        heat = dict(heat_duties_w or {})
        utility = dict(utility_temperatures_k or {})
        openings = dict(openings or {})
        if (
            set(feeds) - set(self.names)
            or set(heat) - set(self.names)
            or set(utility) - set(self.names)
            or set(openings) - {p.name for p in self.ports}
        ):
            raise ValueError("Unknown separator feed, heat source or port")
        for feed_node, streams in feeds.items():
            if not isinstance(streams, (list, tuple)):
                raise TypeError("Each vessel feed must be a sequence")
            for stream in streams:
                if (
                    not isinstance(stream, ThreePhaseStream)
                    or stream.fluid_id != self.fluid.model_id
                ):
                    raise TypeError(
                        "Feed requires matching three-phase PHASE stream; stock/PR streams need explicit conversion"
                    )
                if (
                    not self.fluid.min_temperature_k
                    <= stream.temperature_k
                    <= self.fluid.max_temperature_k
                ):
                    raise ValueError("Feed temperature outside model")
                vessel = self.vessels[self.index[feed_node]]
                if (
                    not vessel.elevation_m
                    <= stream.elevation_m
                    <= vessel.elevation_m + vessel.geometry.height_m
                ):
                    raise ValueError("Feed elevation must lie on the receiving vessel")
        for value in heat.values():
            _number(value, "heat_duty_w")
        for value in openings.values():
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Opening outside [0,1]")
        for name, value in utility.items():
            _number(value, "utility_temperature_k", positive=True)
        if any(heat.get(n, 0) != 0 and n not in utility for n in self.names):
            raise ValueError("Every nonzero heat duty needs an explicit utility temperature")
        feed_rows = []
        for name, vessel in zip(self.names, self.vessels):
            row = []
            for stream in feeds.get(name, ()):
                rates = self.tensor(stream.phase_mass_rates_kg_s)
                p = self.tensor(stream.pressure_pa)
                t = self.tensor(stream.temperature_k)
                energy = (
                    rates * self.fluid.phase_enthalpies(p, t)
                ).sum() + rates.sum() * self.gravity_m_s2 * stream.elevation_m
                entropy = (rates * self.fluid.phase_entropies(p, t)).sum()
                row.append(
                    (
                        rates,
                        energy,
                        entropy,
                        p,
                        self.tensor(stream.elevation_m - vessel.elevation_m),
                    )
                )
            feed_rows.append(row)
        heat_tensor = self.tensor([heat.get(n, 0.0) for n in self.names])
        utility_tensor = self.tensor([utility.get(n, 1.0) for n in self.names])
        opening_tensors = {n: self.tensor(v) for n, v in openings.items()}
        initial_values = self._geometry(old)
        admission = []
        for i, row in enumerate(feed_rows):
            for rates, _e, _s, p, height in row:
                admission.append(
                    (rates.sum() == 0)
                    | (p >= self._pressure_at(self.names[i], height, initial_values))
                )
        if admission and not bool(torch.stack(admission).all().detach().cpu()):
            raise EquipmentOperatingError(
                "Passive feed pressure is below receiving nozzle pressure"
            )
        if not bool((heat_tensor * (utility_tensor - old[:, 3]) >= -1e-10).all().detach().cpu()):
            raise EquipmentOperatingError(
                "Heat duty direction disagrees with its utility temperature"
            )
        elapsed = 0.0
        hnext = self.maximum_dt_s
        steps = []
        reasons = {}
        while elapsed < duration:
            if len(steps) + sum(reasons.values()) >= max_substeps:
                raise HydraulicConvergenceError("Three-phase work budget exhausted")
            h = min(hnext, duration - elapsed)
            if elapsed + h <= elapsed:
                raise ValueError("Three-phase substep cannot advance time")
            try:
                coarse = self._step(old, h, feed_rows, heat_tensor, opening_tensors, utility_tensor)
                accepted = [coarse]
                error = None
                if self.temporal_error_control:
                    a = self._step(
                        old, h / 2, feed_rows, heat_tensor, opening_tensors, utility_tensor
                    )
                    b = self._step(
                        a[0], h / 2, feed_rows, heat_tensor, opening_tensors, utility_tensor
                    )
                    ms = torch.maximum(
                        torch.maximum(old[:, :3].abs(), b[0][:, :3].abs()),
                        self._temporal_mass_floor,
                    )
                    me = (
                        (b[0][:, :3] - coarse[0][:, :3]).abs()
                        / (1e-9 + self.relative_tolerance * ms)
                    ).max()
                    te = (
                        (b[0][:, 3:] - coarse[0][:, 3:]).abs()
                        / (
                            self.temperature_tolerance_k
                            + self.relative_tolerance
                            * torch.maximum(old[:, 3:].abs(), b[0][:, 3:].abs())
                        )
                    ).max()
                    error = float(torch.maximum(me, te).detach().cpu())
                    if not math.isfinite(error) or error > 1:
                        raise HydraulicConvergenceError("Three-phase temporal error control")
                    accepted = [a, b]
            except HydraulicConvergenceError as exc:
                reasons[str(exc)] = reasons.get(str(exc), 0) + 1
                if h / 2 < self.minimum_dt_s:
                    raise
                hnext = h / 2
                continue
            for candidate, record in accepted:
                if len(steps) + sum(reasons.values()) >= max_substeps:
                    raise HydraulicConvergenceError("Three-phase work budget exhausted")
                record = {"time_s": state.time_s + elapsed, **record, "temporal_error_ratio": error}
                old = candidate
                steps.append(record)
                elapsed = math.fsum((elapsed, record["dt_s"]))
            if abs(duration - elapsed) <= 16 * math.ulp(max(1.0, duration)):
                elapsed = duration
            hnext = min(
                self.maximum_dt_s, h * (1.5 if not error else min(1.5, max(0.5, 0.9 * error**-0.5)))
            )
        final = ThreePhaseState(
            dict(zip(self.names, old.detach().cpu().tolist())),
            state.time_s + duration,
            self.model_id,
        )
        return {
            "state": final.to_dict(),
            "properties": self.inspect(final),
            "steps": steps,
            "failed_steps": sum(reasons.values()),
            "retry_reasons": reasons,
        }
