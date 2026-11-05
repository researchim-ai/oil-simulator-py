"""Monolithic component/equilibrium/energy dynamics of a lumped PR separator.

Uniform pressure and temperature, ideal phase-selective linear outlets with
holdup-weighted conductance, optional finite wall. No distributed hydrostatic
column, nozzle choking, or automatic thermal-reservoir coupling is inferred.
All nonlinear residuals/Jacobians/linear solves use Torch on the selected device.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import torch

from .component_energy import ComponentEnergyStream
from .equilibrium_inventory import EquilibriumInventory
from .equilibrium_properties import EquilibriumFluid
from .liquid import (
    ConservationError,
    EquipmentOperatingError,
    HydraulicConvergenceError,
    PropertyModelError,
    _name,
    _number,
)
from .operation import _digest
from .phase_geometry import EquilibriumLevelControl, phase_layout, pickup_fractions, validate_pickup
from .separator_geometry import SeparatorGeometry
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class EquilibriumOutlet:
    name: str
    back_pressure_pa: float
    volume_conductance_m3_s_pa: float
    selection: str = "bulk"
    component: str | None = None
    minimum_mole_fraction: float = 0.5
    opening: float = 1.0
    maximum_pressure_drop_fraction: float = 0.05
    pressure_setpoint_pa: float | None = None
    opening_gain_per_pa: float = 0.0
    opening_bias: float = 0.0
    height_m: float | None = None
    diameter_m: float | None = None
    level_control: EquilibriumLevelControl | None = None

    def __post_init__(self):
        _name(self.name)
        _number(self.back_pressure_pa, "back_pressure_pa", positive=True)
        _number(self.volume_conductance_m3_s_pa, "volume_conductance_m3_s_pa", nonnegative=True)
        if self.selection not in ("bulk", "densest", "lightest", "component_rich"):
            raise ValueError("Unknown equilibrium outlet selection")
        if (self.component is not None) != (self.selection == "component_rich"):
            raise ValueError("component_rich requires exactly one named component")
        if self.component is not None:
            _name(self.component)
        for key in ("minimum_mole_fraction", "opening", "opening_bias"):
            if not 0 <= _number(getattr(self, key), key) <= 1:
                raise ValueError(f"{key} outside [0,1]")
        if (
            not 0
            < _number(self.maximum_pressure_drop_fraction, "maximum_pressure_drop_fraction")
            <= 1
        ):
            raise ValueError("Invalid calibrated pressure-drop domain")
        _number(self.opening_gain_per_pa, "opening_gain_per_pa", nonnegative=True)
        if self.pressure_setpoint_pa is None:
            if self.opening_gain_per_pa or self.opening_bias:
                raise ValueError("Pressure feedback requires a setpoint")
        else:
            _number(self.pressure_setpoint_pa, "pressure_setpoint_pa", positive=True)
        if (self.height_m is None) != (self.diameter_m is None):
            raise ValueError("Pickup needs height and diameter")
        if self.height_m is not None:
            _number(self.height_m, "height_m", nonnegative=True)
            _number(self.diameter_m, "diameter_m", positive=True)
        if self.level_control is not None:
            if not isinstance(self.level_control, EquilibriumLevelControl):
                raise TypeError("Expected EquilibriumLevelControl")
            if self.pressure_setpoint_pa is not None:
                raise ValueError("Choose pressure or level feedback")


@dataclass(frozen=True)
class CompositionalSeparatorState:
    inventory: EquilibriumInventory
    wall_temperature_k: float | None
    next_dt_s: float
    model_id: str

    def __post_init__(self):
        if not isinstance(self.inventory, EquilibriumInventory):
            raise TypeError("Expected EquilibriumInventory")
        _number(self.next_dt_s, "next_dt_s", positive=True)
        if self.wall_temperature_k is not None:
            _number(self.wall_temperature_k, "wall_temperature_k", positive=True)
        if not isinstance(self.model_id, str) or len(self.model_id) != 64:
            raise PropertyModelError("Missing separator model identity")

    def to_dict(self):
        return {
            "inventory": self.inventory.to_dict(),
            "wall_temperature_k": self.wall_temperature_k,
            "next_dt_s": self.next_dt_s,
            "model_id": self.model_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"inventory", "wall_temperature_k", "next_dt_s", "model_id"}:
            raise ValueError("Unexpected separator state fields")
        return cls(
            EquilibriumInventory.from_dict(data["inventory"]),
            data["wall_temperature_k"],
            data["next_dt_s"],
            data["model_id"],
        )


class CompositionalSeparator:
    """Equilibrium is solved inside BE Newton, not by nested flash per trial.

    Material resides at a common elevation datum; internal gravitational
    stratification/head is outside this spatially lumped model. Outlet
    conductances are effective operating-point data, not nozzle correlations.
    """

    def __init__(
        self,
        fluid,
        volume_m3,
        outlets=(),
        *,
        wall=None,
        geometry=None,
        elevation_m=0.0,
        gravity_m_s2=0.0,
        maximum_dt_s=0.1,
        minimum_dt_s=1e-7,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.001,
        temporal_error_control=True,
        maximum_fractional_inventory_change=0.1,
        maximum_temperature_change_k=2.0,
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Expected EquilibriumFluid")
        self.fluid = fluid
        self.volume_m3 = _number(volume_m3, "volume_m3", positive=True)
        self.outlets = tuple(outlets)
        if not all(isinstance(p, EquilibriumOutlet) for p in self.outlets) or len(
            {p.name for p in self.outlets}
        ) != len(self.outlets):
            raise ValueError("Expected uniquely named EquilibriumOutlets")
        for p in self.outlets:
            if p.component is not None and p.component not in fluid.names:
                raise ValueError("Outlet component missing from fluid registry")
        if wall is not None and not isinstance(wall, VesselWall):
            raise TypeError("Expected VesselWall")
        self.wall = wall
        self.elevation_m = _number(elevation_m, "elevation_m")
        self.gravity_m_s2 = _number(gravity_m_s2, "gravity_m_s2", nonnegative=True)
        if geometry is not None:
            if not isinstance(geometry, SeparatorGeometry):
                raise TypeError("Expected SeparatorGeometry")
            if abs(geometry.volume_m3 - self.volume_m3) > 1e-12 + 1e-10 * self.volume_m3:
                raise ValueError("Geometry volume differs from rigid vessel volume")
            if self.gravity_m_s2 != 0:
                raise ValueError("Uniform-pressure PR geometry currently requires gravity_m_s2=0")
        self.geometry = geometry
        for port in self.outlets:
            validate_pickup(geometry, port.height_m, port.diameter_m)
            self._validate_level_control(port.level_control)
        for key, value in (
            ("maximum_dt_s", maximum_dt_s),
            ("minimum_dt_s", minimum_dt_s),
            ("relative_tolerance", relative_tolerance),
            ("temperature_tolerance_k", temperature_tolerance_k),
            ("maximum_fractional_inventory_change", maximum_fractional_inventory_change),
            ("maximum_temperature_change_k", maximum_temperature_change_k),
        ):
            setattr(self, key, _number(value, key, positive=True))
        if minimum_dt_s > maximum_dt_s or maximum_fractional_inventory_change > 1:
            raise ValueError("Invalid separator timestep/inventory domain")
        if not isinstance(temporal_error_control, bool):
            raise TypeError("temporal_error_control must be bool")
        self.temporal_error_control = temporal_error_control
        self.newton = TorchNewton(fluid.device, maximum_iterations=30)
        self.newton.host_convergence = True
        self.model_id = _digest({"model": "monolithic_pr_separator_v1", **self.to_dict()})

    def to_dict(self):
        return {
            "fluid": self.fluid.to_dict(),
            "volume_m3": self.volume_m3,
            "outlets": [asdict(p) for p in self.outlets],
            "wall": asdict(self.wall) if self.wall else None,
            "geometry": asdict(self.geometry) if self.geometry else None,
            "elevation_m": self.elevation_m,
            "gravity_m_s2": self.gravity_m_s2,
            **{
                k: getattr(self, k)
                for k in (
                    "maximum_dt_s",
                    "minimum_dt_s",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "temporal_error_control",
                    "maximum_fractional_inventory_change",
                    "maximum_temperature_change_k",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        data["fluid"] = EquilibriumFluid.from_dict(data["fluid"], device=device)
        outlets = []
        for port in data.get("outlets", ()):
            port = dict(port)
            if port.get("level_control") is not None:
                port["level_control"] = EquilibriumLevelControl(**port["level_control"])
            outlets.append(EquilibriumOutlet(**port))
        data["outlets"] = tuple(outlets)
        data["geometry"] = (
            SeparatorGeometry(**data["geometry"]) if data.get("geometry") is not None else None
        )
        data["wall"] = VesselWall(**data["wall"]) if data.get("wall") is not None else None
        return cls(**data)

    def initialize(
        self, pressure_pa, temperature_k, mole_fractions, *, wall_temperature_k=None, time_s=0.0
    ):
        q = self.fluid.flash_pt(pressure_pa, temperature_k, mole_fractions)
        total = self.volume_m3 / float(q["molar_volume_m3_mol"].cpu())
        inv = EquilibriumInventory(
            tuple((total * q["mole_fractions"]).cpu().tolist()),
            total * float(q["molar_internal_energy_j_mol"].cpu()),
            self.volume_m3,
            pressure_pa,
            temperature_k,
            self.fluid.model_id,
            time_s,
            self.elevation_m,
            self.gravity_m_s2,
        )
        if self.wall:
            wall_temperature_k = temperature_k if wall_temperature_k is None else wall_temperature_k
            if (
                not self.wall.min_temperature_k
                <= _number(wall_temperature_k, "wall_temperature_k", positive=True)
                <= self.wall.max_temperature_k
            ):
                raise PropertyModelError("Wall temperature outside domain")
        elif wall_temperature_k is not None:
            raise ValueError("Wall temperature requires a wall")
        return CompositionalSeparatorState(
            inv, wall_temperature_k, self.maximum_dt_s, self.model_id
        )

    def inspect(self, state):
        if _digest({"model": "monolithic_pr_separator_v1", **self.to_dict()}) != self.model_id:
            raise PropertyModelError("Separator configuration changed; construct a new model")
        if not isinstance(state, CompositionalSeparatorState) or state.model_id != self.model_id:
            raise PropertyModelError("Separator state/configuration mismatch")
        inv = state.inventory
        if (
            inv.volume_m3 != self.volume_m3
            or inv.elevation_m != self.elevation_m
            or inv.gravity_m_s2 != self.gravity_m_s2
        ):
            raise PropertyModelError("Separator volume/elevation/gravity mismatch")
        if self.wall:
            if (
                state.wall_temperature_k is None
                or not self.wall.min_temperature_k
                <= state.wall_temperature_k
                <= self.wall.max_temperature_k
            ):
                raise PropertyModelError("Missing/invalid wall state")
        elif state.wall_temperature_k is not None:
            raise PropertyModelError("Wall state without wall model")
        if not self.minimum_dt_s <= state.next_dt_s <= self.maximum_dt_s:
            raise PropertyModelError("Invalid carried timestep")
        q = inv.inspect(self.fluid)
        if self.geometry is not None:
            n = torch.stack([p["molar_fraction"] for p in q["phases"]]) * math.fsum(
                inv.component_moles
            )
            props = {
                k: torch.stack([p[k] for p in q["phases"]])
                for k in ("molar_volume_m3_mol", "density_kg_m3")
            }
            if not bool(self._layout(n, props)["density_order_resolved"].cpu()):
                raise PropertyModelError("Phase densities do not resolve a unique layer order")
        return q

    def _phase_values(self, p, t, np_, active, tags):
        phase_moles = np_.sum(-1)
        xa = np_ / phase_moles[:, None]
        x = np_.new_zeros((len(tags), len(self.fluid.names))).index_copy(-1, active, xa)
        phases = [self.fluid.tensor_homogeneous(p, t, x[i], tag) for i, tag in enumerate(tags)]
        q = {key: torch.stack([row[key] for row in phases]) for key in phases[0]}
        return phase_moles, x, q

    def _validate_level_control(self, control):
        if control is not None:
            if self.geometry is None:
                raise ValueError("Level feedback requires vessel geometry")
            if control.component not in self.fluid.names:
                raise ValueError("Unknown level-control component")
            if control.setpoint_m > self.geometry.height_m:
                raise ValueError("Level setpoint outside geometry")

    def _layout(self, phase_moles, q):
        return phase_layout(phase_moles, q, self.geometry) if self.geometry else None

    def _pickup(self, phase_moles, q, height, diameter):
        if height is None:
            return phase_moles * q["molar_volume_m3_mol"] / self.volume_m3
        return pickup_fractions(self._layout(phase_moles, q), height, diameter)

    def _fluxes(self, p, phase_moles, x, q, openings):
        volumes = phase_moles * q["molar_volume_m3_mol"]
        holdup = volumes / self.volume_m3
        rates = []
        energy = []
        entropy = []
        records = []
        valid = p.new_tensor(True, dtype=torch.bool)
        layout = self._layout(phase_moles, q)
        if layout is not None:
            valid = valid & layout["density_order_resolved"]
        for port in self.outlets:
            opening = openings.get(port.name)
            if opening is None:
                opening = (
                    p.new_tensor(port.opening)
                    if port.pressure_setpoint_pa is None
                    else (
                        port.opening_bias
                        + port.opening_gain_per_pa * (p - port.pressure_setpoint_pa)
                    ).clamp(0, 1)
                )
                if port.level_control is not None:
                    opening = port.level_control.value(layout, x, self.fluid.names)[0]
            mask = torch.ones_like(holdup)
            if port.selection in ("densest", "lightest"):
                index = (
                    q["density_kg_m3"].argmax()
                    if port.selection == "densest"
                    else q["density_kg_m3"].argmin()
                )
                mask = torch.nn.functional.one_hot(index, len(holdup)).to(holdup.dtype)
            elif port.selection == "component_rich":
                column = x[:, self.fluid.names.index(port.component)]
                index = column.argmax()
                mask = torch.nn.functional.one_hot(index, len(holdup)).to(holdup.dtype) * (
                    column.max() >= port.minimum_mole_fraction
                )
            dp = (p - port.back_pressure_pa).clamp_min(0)
            weights = (
                pickup_fractions(layout, port.height_m, port.diameter_m)
                if port.height_m is not None
                else holdup
            )
            vdot = port.volume_conductance_m3_s_pa * opening * dp * weights * mask
            ndot = vdot / q["molar_volume_m3_mol"]
            component = (ndot[:, None] * x).sum(0)
            mdot = (component * self.fluid.molar_masses).sum()
            e = (
                ndot * q["molar_enthalpy_j_mol"]
            ).sum() + mdot * self.gravity_m_s2 * self.elevation_m
            s = (ndot * q["molar_entropy_j_mol_k"]).sum()
            valid = valid & ((vdot.sum() == 0) | (dp / p <= port.maximum_pressure_drop_fraction))
            rates.append(component)
            energy.append(e)
            entropy.append(s)
            records.append(
                torch.cat(
                    (
                        component,
                        e.reshape(1),
                        s.reshape(1),
                        vdot.sum().reshape(1),
                        opening.reshape(1),
                    )
                )
            )
        zeros = p.new_zeros(len(self.fluid.names))
        return (
            torch.stack(rates).sum(0) if rates else zeros,
            torch.stack(energy).sum() if energy else p * 0,
            torch.stack(entropy).sum() if entropy else p * 0,
            records,
            valid,
        )

    def _tags(self, seed):
        p, t = seed["pressure_pa"], seed["temperature_k"]
        if seed["phase_count"] == 1:
            return ("stable",)
        tags = []
        for phase in seed["phases"]:
            x = phase["mole_fractions"]
            a = self.fluid.tensor_homogeneous(p, t, x, "liquid")["compressibility_factor"]
            b = self.fluid.tensor_homogeneous(p, t, x, "vapor")["compressibility_factor"]
            z = phase["compressibility_factor"]
            tags.append(
                "liquid" if float((z - a).abs().cpu()) <= float((z - b).abs().cpu()) else "vapor"
            )
        return tuple(tags)

    def _branch_step(
        self,
        old,
        seed,
        keep,
        dt,
        feed,
        heat,
        utility,
        openings,
        *,
        _capture=False,
        _solution=None,
        _network_incoming=None,
        _network_active=None,
        _network_guess=None,
    ):
        f = self.fluid
        inv = old.inventory
        n0 = f.tensor(inv.component_moles)
        predicted = n0 + dt * feed[0]
        active = (
            torch.nonzero(predicted > 0).flatten() if _network_active is None else _network_active
        )
        if _network_active is not None:
            # Extend only the guess/scaling, never the physical old inventory.
            predicted = predicted if _network_guess is None else _network_guess
            predicted = predicted.clone().index_copy(
                # Preserve existing trace components: raising them to a floor
                # destroys chemical equilibrium in separated liquid layers.
                # Only newly activated/invalid guesses need a positive seed.
                0,
                active,
                torch.where(predicted[active] > 0, predicted[active], n0.sum() * 1e-8),
            )
        tags0 = self._tags(seed)
        tags = tuple(tags0[i] for i in keep) if len(keep) > 1 else ("stable",)
        weights = torch.stack(
            [
                seed["phases"][i]["molar_fraction"] * seed["phases"][i]["mole_fractions"][active]
                for i in keep
            ]
        )
        weights = weights + predicted[active][None, :] / predicted.sum() * 1e-30
        amounts = weights / weights.sum(0) * predicted[active]
        m, c = amounts.shape
        energy_scale = max(
            1.0,
            abs(inv.internal_energy_j),
            float(n0.sum().cpu()) * 8.31446261815324 * inv.temperature_k,
        )
        nscale = torch.maximum(predicted[active], predicted.sum() * 1e-10)
        ref = f.tref
        initial = torch.cat(
            (
                amounts.log().reshape(-1),
                torch.log(seed["pressure_pa"]).reshape(1),
                (seed["temperature_k"] / ref).reshape(1),
                f.tensor([old.wall_temperature_k / ref]) if self.wall else f.tensor([]),
            )
        )
        low = torch.cat(
            (
                f.tensor([-700.0] * (m * c)),
                f.tensor([math.log(f.min_pressure_pa), f.min_temperature_k / ref]),
                f.tensor([self.wall.min_temperature_k / ref]) if self.wall else f.tensor([]),
            )
        )
        upper_n = math.log(max(1.0, float(predicted.sum().cpu())) * 1e4)
        high = torch.cat(
            (
                f.tensor([upper_n] * (m * c)),
                f.tensor([math.log(f.max_pressure_pa), f.max_temperature_k / ref]),
                f.tensor([self.wall.max_temperature_k / ref]) if self.wall else f.tensor([]),
            )
        )
        old_potential = inv.mass_kg(f) * self.gravity_m_s2 * self.elevation_m
        incoming = (
            (torch.zeros_like(n0), f.tensor(0.0), f.tensor(0.0))
            if _network_incoming is None
            else _network_incoming
        )

        def evaluate(y):
            np_ = torch.exp(y[: m * c]).reshape(m, c)
            p = torch.exp(y[m * c])
            t = y[m * c + 1] * ref
            tw = y[-1] * ref if self.wall else t
            totals = f.tensor([0.0] * len(f.names)).index_copy(0, active, np_.sum(0))
            phase_n, x, q = self._phase_values(p, t, np_, active, tags)
            volume = (phase_n * q["molar_volume_m3_mol"]).sum()
            internal = (phase_n * q["molar_internal_energy_j_mol"]).sum()
            potential = (totals * f.molar_masses).sum() * self.gravity_m_s2 * self.elevation_m
            out_n, out_e, out_s, port_rows, port_valid = self._fluxes(p, phase_n, x, q, openings)
            exchange = self.wall.fluid_ua_w_k * (tw - t) if self.wall else t * 0
            ambient = (
                self.wall.ambient_ua_w_k * (self.wall.ambient_temperature_k - tw)
                if self.wall
                else t * 0
            )
            mass = totals - n0 - dt * (feed[0] + incoming[0] - out_n)
            energy = (
                internal
                + potential
                - inv.internal_energy_j
                - old_potential
                - dt * (feed[1] + incoming[1] - out_e + heat + exchange)
            )
            wall_error = (
                (
                    self.wall.heat_capacity_j_k * (tw - old.wall_temperature_k)
                    - dt * (ambient - exchange)
                )
                if self.wall
                else t * 0
            )
            logx = torch.log(np_) - torch.log(phase_n)[:, None]
            mu = logx + q["log_fugacity_coefficients"][:, active]
            chemical = (mu[:-1] - mu[-1]).reshape(-1)
            rows = (
                mass[active] / nscale,
                chemical,
                (volume / self.volume_m3 - 1).reshape(1),
                (energy / energy_scale).reshape(1),
            )
            residual = (
                torch.cat(
                    (*rows, (wall_error / max(1.0, self.wall.heat_capacity_j_k * ref)).reshape(1))
                )
                if self.wall
                else torch.cat(rows)
            )
            return residual, (
                np_,
                totals,
                p,
                t,
                tw,
                phase_n,
                x,
                q,
                internal,
                potential,
                ambient,
                out_n,
                out_e,
                out_s,
                port_rows,
                port_valid,
                mass,
                energy,
                wall_error,
                volume,
            )

        if _capture:
            return {
                "initial": initial,
                "lower": low,
                "upper": high,
                "evaluate": evaluate,
                "active": active,
                "nscale": nscale,
                "energy_scale": energy_scale,
                "energy_row": m * c + 1,
            }
        residual_fn = lambda y: evaluate(y)[0]
        with torch.no_grad():
            if _solution is None:
                y, r, ok, evaluations = self.newton.solve(
                    residual_fn, initial, low, high, tolerance=2e-13
                )
            else:
                y, ok, evaluations = _solution
            r, data = evaluate(y)
        (
            _np,
            totals,
            p,
            t,
            tw,
            phase_n,
            _x,
            q,
            internal,
            _potential,
            ambient,
            out_n,
            out_e,
            out_s,
            port_rows,
            port_valid,
            mass,
            energy,
            wall_error,
            volume,
        ) = data
        valid = (
            ok
            & torch.isfinite(r).all()
            & (r.abs().max() < 1e-10)
            & q["valid"].all()
            & port_valid
            & (heat * (utility - t) >= -1e-10)
            & (p <= feed[3])
        )
        if not bool(valid.detach().cpu()):
            raise HydraulicConvergenceError("Monolithic separator branch/domain rejected")
        candidate = EquilibriumInventory(
            tuple(totals.cpu().tolist()),
            float(internal.cpu()),
            self.volume_m3,
            float(p.cpu()),
            float(t.cpu()),
            f.model_id,
            inv.time_s + dt,
            self.elevation_m,
            self.gravity_m_s2,
        )
        # Independent post-step phase stability; never run a flash in residual_fn.
        try:
            stable = f.flash_pt(p, t, totals / totals.sum())
        except PropertyModelError as exc:
            if "Pure-fluid PT coexistence" not in str(exc) or len(active) != 1 or m != 2:
                raise
            stable = {"phase_count": m, "minimum_tpd": f.tensor(0.0)}
        if stable["phase_count"] != m:
            return None, stable
        if "phases" in stable:
            energy_match = (stable["molar_internal_energy_j_mol"] * totals.sum() - internal).abs()
            volume_match = (stable["molar_volume_m3_mol"] * totals.sum() - self.volume_m3).abs()
            refphase = stable["phases"][0]
            chemical_match = (
                (
                    torch.log(_x[-1, active])
                    + q["log_fugacity_coefficients"][-1, active]
                    - torch.log(refphase["mole_fractions"][active])
                    - refphase["log_fugacity_coefficients"][active]
                )
                .abs()
                .max()
            )
            if (
                float(energy_match.cpu())
                > 0.5 * (1e-6 + 1e-9 * max(1.0, abs(float(internal.cpu()))))
                or float(volume_match.cpu()) > 0.5 * (1e-12 + 1e-9 * self.volume_m3)
                or float(chemical_match.cpu()) > 1e-8
            ):
                return None, stable
        mass_tol = 1e-9 + 1e-10 * max(1.0, float((totals - n0).abs().sum().cpu()))
        energy_tol = 1e-5 + 1e-9 * max(
            1.0,
            abs(dt * float((feed[1] + incoming[1] - out_e + heat).cpu())),
            abs(float(internal.cpu()) - inv.internal_energy_j),
        )
        if (
            float(mass.abs().max().cpu()) > mass_tol
            or abs(float(energy.cpu())) > energy_tol
            or abs(float(wall_error.cpu())) > 1e-5 + 1e-9 * max(1.0, abs(float(ambient.cpu()) * dt))
        ):
            raise ConservationError("Monolithic separator local component/energy balance rejected")
        old_entropy = seed.get("old_entropy_j_k")
        if old_entropy is None:
            raise ValueError("Missing accepted initial entropy")
        entropy = (phase_n * q["molar_entropy_j_mol_k"]).sum()
        wall_entropy = (
            self.wall.heat_capacity_j_k * torch.log(tw / old.wall_temperature_k)
            if self.wall
            else t * 0
        )
        ambient_entropy = ambient / self.wall.ambient_temperature_k if self.wall else t * 0
        production = (
            entropy
            - old_entropy
            + wall_entropy
            - dt * (feed[2] + incoming[2] - out_s + heat / utility + ambient_entropy)
        )
        if float(production.cpu()) < -1e-6 - 1e-10 * max(1.0, abs(float(entropy.cpu()))):
            raise ConservationError("Monolithic separator entropy balance rejected")
        # A newly injected component must be able to appear from exact zero.
        # Temporal control resolves its composition error; this guard bounds
        # absolute component transfer relative to the whole stored inventory.
        fraction = (totals - n0).abs().max() / n0.sum()
        if (
            float(fraction.cpu()) > self.maximum_fractional_inventory_change
            or abs(float(t.cpu()) - inv.temperature_k) > self.maximum_temperature_change_k
        ):
            raise HydraulicConvergenceError("Monolithic separator state-change guard")
        if (
            self.wall
            and abs(float(tw.cpu()) - old.wall_temperature_k) > self.wall.max_temperature_change_k
        ):
            raise HydraulicConvergenceError("Monolithic separator wall-change guard")
        result = CompositionalSeparatorState(
            candidate, float(tw.cpu()) if self.wall else None, old.next_dt_s, self.model_id
        )
        phases = [
            {
                **{
                    key: value[i].detach()
                    for key, value in q.items()
                    if key not in ("valid", "coexistence_degenerate")
                },
                "molar_fraction": phase_n[i] / totals.sum(),
            }
            for i in range(m)
        ]
        accepted_seed = {
            "pressure_pa": p.detach(),
            "temperature_k": t.detach(),
            "phase_count": m,
            "phases": phases,
            "old_entropy_j_k": entropy.detach(),
        }
        packet = torch.cat(
            (
                mass,
                energy.reshape(1),
                wall_error.reshape(1),
                production.reshape(1),
                volume.reshape(1),
                ambient.reshape(1),
                out_n,
                out_e.reshape(1),
                r.abs().max().reshape(1),
            )
        )
        host = packet.cpu().tolist()
        nc = len(f.names)
        report = {
            "pressure_pa": float(p.cpu()),
            "temperature_k": float(t.cpu()),
            "wall_temperature_k": float(tw.cpu()) if self.wall else None,
            "phase_moles": phase_n.cpu().tolist(),
            "phase_mole_fractions": q["mole_fractions"].cpu().tolist(),
            "phase_volume_fractions": (phase_n * q["molar_volume_m3_mol"] / self.volume_m3)
            .cpu()
            .tolist(),
            "dt_s": dt,
            "phase_count": m,
            "component_residuals_mol": host[:nc],
            "fluid_energy_residual_j": host[nc],
            "wall_energy_residual_j": host[nc + 1],
            "entropy_production_j_k": host[nc + 2],
            "volume_m3": host[nc + 3],
            "ambient_heat_w": host[nc + 4],
            "outlet_component_rates_mol_s": host[nc + 5 : 2 * nc + 5],
            "outlet_energy_rate_w": host[2 * nc + 5],
            "max_scaled_residual": host[2 * nc + 6],
            "nonlinear_evaluations": evaluations,
            "minimum_tpd": float(stable["minimum_tpd"].cpu()),
            "ports": {
                port.name: {
                    "component_molar_rates_mol_s": row[:nc],
                    "energy_rate_w": row[nc],
                    "entropy_rate_w_k": row[nc + 1],
                    "volume_rate_m3_s": row[nc + 2],
                    "opening": row[nc + 3],
                }
                for port, row in zip(self.outlets, [row.cpu().tolist() for row in port_rows])
            },
        }
        if self.geometry is not None:
            layout = self._layout(phase_n, q)
            report["geometry"] = {key: value.cpu().tolist() for key, value in layout.items()}
            for port in self.outlets:
                if port.height_m is not None:
                    report["ports"][port.name]["pickup_area_fractions"] = (
                        self._pickup(phase_n, q, port.height_m, port.diameter_m).cpu().tolist()
                    )
                if port.level_control is not None:
                    _, level, present = port.level_control.value(layout, _x, f.names)
                    report["ports"][port.name]["controlled_level_m"] = float(level.cpu())
                    report["ports"][port.name]["controlled_phase_present"] = bool(present.cpu())
        return (result, accepted_seed, report), stable

    def _step(self, old, seed, dt, feed, heat, utility, openings):
        accepted_entropy = seed["old_entropy_j_k"]
        predicted = self.fluid.tensor(old.inventory.component_moles) + dt * feed[0]
        seed_active = torch.stack([p["mole_fractions"] for p in seed["phases"]]).sum(0)
        if bool(((predicted > 0) & (seed_active == 0)).any().cpu()):
            seed = self.fluid.flash_pt(
                old.inventory.pressure_pa, old.inventory.temperature_k, predicted / predicted.sum()
            )
        seed = {**seed, "old_entropy_j_k": accepted_entropy}
        attempted = set()
        queue = [(seed, tuple(range(seed["phase_count"])))]
        errors = []
        pure_predictor_tried = False
        while queue and len(attempted) < 12:
            guess, keep = queue.pop(0)
            signature = (
                tuple(keep),
                tuple(
                    round(float(value), 12)
                    for phase in guess["phases"]
                    for value in phase["mole_fractions"].cpu().tolist()
                ),
                round(float(guess["pressure_pa"].cpu()), 6),
                round(float(guess["temperature_k"].cpu()), 10),
            )
            if signature in attempted:
                continue
            attempted.add(signature)
            try:
                solved, post = self._branch_step(
                    old, guess, keep, dt, feed, heat, utility, openings
                )
                if solved is not None:
                    solved[2]["phase_branch_attempts"] = len(attempted)
                    return solved
                if "phases" in post:
                    post = {**post, "old_entropy_j_k": accepted_entropy}
                    queue.append((post, tuple(range(post["phase_count"]))))
            except (HydraulicConvergenceError, PropertyModelError) as exc:
                errors.append(str(exc))
            if not pure_predictor_tried and int((predicted > 0).sum().cpu()) == 1:
                pure_predictor_tried = True
                try:
                    f = self.fluid
                    n0 = f.tensor(old.inventory.component_moles)
                    phase_n = torch.stack([p["molar_fraction"] for p in seed["phases"]]) * n0.sum()
                    x = torch.stack([p["mole_fractions"] for p in seed["phases"]])
                    values = [
                        f.tensor_homogeneous(seed["pressure_pa"], seed["temperature_k"], xi, tag)
                        for xi, tag in zip(x, self._tags(seed))
                    ]
                    q = {key: torch.stack([row[key] for row in values]) for key in values[0]}
                    out_n, out_e, _, _, _ = self._fluxes(
                        seed["pressure_pa"], phase_n, x, q, openings
                    )
                    next_n = predicted - dt * out_n
                    if bool(((next_n >= 0).all() & (next_n.sum() > 0)).cpu()):
                        exchange = (
                            self.wall.fluid_ua_w_k
                            * (old.wall_temperature_k - old.inventory.temperature_k)
                            if self.wall
                            else 0.0
                        )
                        dp = (
                            ((next_n - n0) * f.molar_masses).sum()
                            * self.gravity_m_s2
                            * self.elevation_m
                        )
                        target = (
                            old.inventory.internal_energy_j
                            + dt * (feed[1] - out_e + heat + exchange)
                            - dp
                        )
                        post = f.flash_uv(
                            next_n,
                            float(target.cpu()),
                            self.volume_m3,
                            pressure_guess_pa=old.inventory.pressure_pa,
                            temperature_guess_k=old.inventory.temperature_k,
                        )
                        queue.append(
                            (
                                {**post, "old_entropy_j_k": accepted_entropy},
                                tuple(range(post["phase_count"])),
                            )
                        )
                except (HydraulicConvergenceError, PropertyModelError) as exc:
                    errors.append(str(exc))
            if len(keep) > 1:
                queue.extend((guess, sub) for sub in itertools.combinations(keep, len(keep) - 1))
        raise HydraulicConvergenceError(
            "Separator active-phase solve failed: "
            + (errors[-1] if errors else "unresolved stability")
        )

    def advance(
        self,
        state,
        duration_s,
        *,
        feeds=(),
        heat_w=0.0,
        utility_temperature_k=None,
        openings=None,
        max_substeps=10000,
    ):
        seed = self.inspect(state)
        old = state
        total = math.fsum(state.inventory.component_moles)
        seed = {**seed, "old_entropy_j_k": self.fluid.tensor(total) * seed["molar_entropy_j_mol_k"]}
        duration = _number(duration_s, "duration_s", positive=True)
        if (
            not math.isfinite(state.inventory.time_s + duration)
            or state.inventory.time_s + duration <= state.inventory.time_s
        ):
            raise ValueError("Separator clock cannot advance")
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Positive work budget required")
        feeds = tuple(feeds)
        opening_values = dict(openings or {})
        if set(opening_values) - {p.name for p in self.outlets}:
            raise ValueError("Unknown outlet opening")
        for value in opening_values.values():
            if not 0 <= _number(value, "opening") <= 1:
                raise ValueError("Opening outside [0,1]")
        opening_values = {name: self.fluid.tensor(value) for name, value in opening_values.items()}
        initial_phase_n = self.fluid.tensor(
            [total * float(p["molar_fraction"].cpu()) for p in seed["phases"]]
        )
        initial_x = torch.stack([p["mole_fractions"] for p in seed["phases"]])
        initial_properties = {
            key: torch.stack([p[key] for p in seed["phases"]])
            for key in (
                "molar_volume_m3_mol",
                "molar_enthalpy_j_mol",
                "molar_entropy_j_mol_k",
                "density_kg_m3",
            )
        }
        if not bool(
            self._fluxes(
                self.fluid.tensor(state.inventory.pressure_pa),
                initial_phase_n,
                initial_x,
                initial_properties,
                opening_values,
            )[-1].cpu()
        ):
            raise EquipmentOperatingError(
                "Initial outlet outside its calibrated pressure-drop domain"
            )
        heat = self.fluid.tensor(_number(heat_w, "heat_w"))
        if heat_w and utility_temperature_k is None:
            raise ValueError("Nonzero heat requires explicit utility temperature")
        utility = self.fluid.tensor(
            _number(utility_temperature_k, "utility_temperature_k", positive=True)
            if utility_temperature_k is not None
            else 1.0
        )
        if heat_w * (float(utility.cpu()) - state.inventory.temperature_k) < 0:
            raise EquipmentOperatingError("Heat direction incompatible with utility")
        fn = self.fluid.tensor([0.0] * len(self.fluid.names))
        fe = self.fluid.tensor(0.0)
        fs = self.fluid.tensor(0.0)
        minp = self.fluid.max_pressure_pa
        for stream in feeds:
            if not isinstance(stream, ComponentEnergyStream):
                raise TypeError("Separator feed must be ComponentEnergyStream")
            stream.validate_fluid(self.fluid)
            if stream.gravity_m_s2 != self.gravity_m_s2:
                raise PropertyModelError("Feed gravity reference differs from separator")
            if stream.molar_rate_mol_s:
                if stream.pressure_pa < state.inventory.pressure_pa:
                    raise EquipmentOperatingError("Feed pressure below separator pressure")
                eq = stream.equilibrate(
                    self.fluid, temperature_guess_k=state.inventory.temperature_k
                )
                fs = fs + stream.molar_rate_mol_s * eq["molar_entropy_j_mol_k"]
                minp = min(minp, stream.pressure_pa)
            fn = fn + self.fluid.tensor(stream.component_molar_rates_mol_s)
            fe = fe + stream.total_energy_rate_w
        feed = (fn, fe, fs, self.fluid.tensor(minp))
        elapsed = 0.0
        hnext = min(state.next_dt_s, self.maximum_dt_s)
        steps = []
        reasons = {}
        while elapsed < duration:
            if len(steps) + sum(reasons.values()) >= max_substeps:
                raise HydraulicConvergenceError("Separator work budget exhausted")
            h = min(hnext, duration - elapsed)
            if elapsed + h <= elapsed:
                raise ValueError("Separator substep cannot advance")
            try:
                coarse = self._step(old, seed, h, feed, heat, utility, opening_values)
                accepted = [coarse]
                error = 0.0
                if self.temporal_error_control:
                    a = self._step(old, seed, h / 2, feed, heat, utility, opening_values)
                    b = self._step(a[0], a[1], h / 2, feed, heat, utility, opening_values)
                    na = self.fluid.tensor(coarse[0].inventory.component_moles)
                    nb = self.fluid.tensor(b[0].inventory.component_moles)
                    scale = torch.maximum(na.abs(), nb.abs()).clamp_min(nb.sum() * 1e-8)
                    me = ((na - nb).abs() / (1e-10 + self.relative_tolerance * scale)).max()
                    tc, tb = coarse[0].inventory.temperature_k, b[0].inventory.temperature_k
                    error = max(
                        float(me.cpu()),
                        abs(tc - tb)
                        / (
                            self.temperature_tolerance_k
                            + self.relative_tolerance * max(abs(tc), abs(tb))
                        ),
                    )
                    pc, pb = coarse[0].inventory.pressure_pa, b[0].inventory.pressure_pa
                    error = max(
                        error,
                        abs(pc - pb) / (1.0 + self.relative_tolerance * max(abs(pc), abs(pb))),
                    )
                    if self.wall:
                        tc, tb = coarse[0].wall_temperature_k, b[0].wall_temperature_k
                        error = max(
                            error,
                            abs(tc - tb)
                            / (
                                self.temperature_tolerance_k
                                + self.relative_tolerance * max(abs(tc), abs(tb))
                            ),
                        )
                    if not math.isfinite(error) or error > 1:
                        raise HydraulicConvergenceError("Separator temporal error control")
                    accepted = [a, b]
            except HydraulicConvergenceError as exc:
                reasons[str(exc)] = reasons.get(str(exc), 0) + 1
                if h / 2 < self.minimum_dt_s:
                    raise
                hnext = h / 2
                continue
            for candidate, next_seed, record in accepted:
                if len(steps) + sum(reasons.values()) >= max_substeps:
                    raise HydraulicConvergenceError("Separator work budget exhausted")
                record = {
                    **record,
                    "time_s": state.inventory.time_s + elapsed,
                    "temporal_error_ratio": error,
                }
                old, seed = candidate, next_seed
                steps.append(record)
                elapsed = math.fsum((elapsed, record["dt_s"]))
            if abs(duration - elapsed) <= 16 * math.ulp(max(1.0, duration)):
                elapsed = duration
            hnext = min(
                self.maximum_dt_s,
                max(
                    self.minimum_dt_s,
                    h * min(1.5, max(0.5, 0.9 * error**-0.5)) if error else h * 1.5,
                ),
            )
        inv = old.inventory
        # Report intervals land at their exact requested endpoint.
        inv = EquilibriumInventory(
            inv.component_moles,
            inv.internal_energy_j,
            inv.volume_m3,
            inv.pressure_pa,
            inv.temperature_k,
            inv.model_id,
            state.inventory.time_s + duration,
            inv.elevation_m,
            inv.gravity_m_s2,
        )
        final = CompositionalSeparatorState(inv, old.wall_temperature_k, hnext, self.model_id)
        feed_n = (duration * fn).cpu().tolist()
        out_n = [
            math.fsum(s["dt_s"] * s["outlet_component_rates_mol_s"][i] for s in steps)
            for i in range(len(self.fluid.names))
        ]
        external_n = [a - b for a, b in zip(feed_n, out_n)]
        mass_residual = [
            a - b - c
            for a, b, c in zip(inv.component_moles, state.inventory.component_moles, external_n)
        ]
        stored_change = inv.total_energy_j(self.fluid) - state.inventory.total_energy_j(self.fluid)
        if self.wall:
            stored_change += self.wall.energy(final.wall_temperature_k) - self.wall.energy(
                state.wall_temperature_k
            )
        external_e = (
            float((duration * fe).cpu())
            + duration * heat_w
            + math.fsum(
                s["dt_s"] * (s["ambient_heat_w"] - s["outlet_energy_rate_w"]) for s in steps
            )
        )
        energy_residual = stored_change - external_e
        if max(map(abs, mass_residual)) > 1e-8 + 1e-10 * max(1.0, sum(map(abs, external_n))) or abs(
            energy_residual
        ) > 1e-5 + 1e-9 * max(1.0, abs(external_e), abs(stored_change)):
            raise ConservationError("Separator report-interval balance rejected")
        return {
            "state": final.to_dict(),
            "steps": steps,
            "failed_steps": sum(reasons.values()),
            "retry_reasons": reasons,
            "feed_component_transfer_mol": feed_n,
            "outlet_component_transfer_mol": out_n,
            "external_component_transfer_mol": external_n,
            "component_residuals_mol": mass_residual,
            "stored_energy_change_j": stored_change,
            "external_energy_in_j": external_e,
            "energy_residual_j": energy_residual,
            "feed_energy_j": float((duration * fe).cpu()),
            "utility_heat_j": duration * heat_w,
        }
