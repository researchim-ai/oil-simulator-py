"""Conservative horizontal thermal homogeneous pipe with frozen liquid/ideal gas.

One velocity and temperature, incompressible liquid, explicit constant Cv.
Wall friction converts kinetic to internal energy; no artificial energy sink.
No phase transfer, real-gas calorics, or energy closure for finite slip.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .thermal_flux import (
    entropy_density,
    entropy_stable_flux,
    entropy_viscosity_sensor,
    physical_entropy_flux,
)
from .transient_pipe import DriftFluxPipe, PipeInventory


@dataclass(frozen=True)
class ThermalPipeInventory(PipeInventory):
    total_energy_j: tuple

    def __post_init__(self):
        super().__post_init__()
        values = tuple(_number(v, "total_energy_j", positive=True) for v in self.total_energy_j)
        if len(values) != len(self.liquid_mass_kg):
            raise ValueError("Thermal pipe energy array must match cells")
        object.__setattr__(self, "total_energy_j", values)


@dataclass(frozen=True)
class ThermalHomogeneousPipe(DriftFluxPipe):
    liquid_cv_j_kg_k: float = 2000.0
    gas_cv_j_kg_k: float = 1500.0
    min_temperature_k: float = 150.0
    max_temperature_k: float = 700.0
    max_temperature_change_k: float = 5.0
    heat_transfer_w_m_k: float = 0.0
    ambient_temperature_k: float = 298.15
    numerical_flux: str = "entropy_stable"
    thermal_entropy_dissipation: float = 1.4
    adaptive_entropy_dissipation: bool = True

    def __post_init__(self):
        super().__post_init__()
        for key in (
            "liquid_cv_j_kg_k",
            "gas_cv_j_kg_k",
            "min_temperature_k",
            "max_temperature_k",
            "max_temperature_change_k",
            "ambient_temperature_k",
        ):
            _number(getattr(self, key), key, positive=True)
        _number(self.heat_transfer_w_m_k, "heat_transfer_w_m_k", nonnegative=True)
        if self.numerical_flux not in ("rusanov", "entropy_stable"):
            raise ValueError("Unknown thermal pipe numerical flux")
        _number(self.thermal_entropy_dissipation, "thermal_entropy_dissipation", nonnegative=True)
        if not isinstance(self.adaptive_entropy_dissipation, bool):
            raise TypeError("adaptive_entropy_dissipation must be boolean")
        if (
            not self.min_temperature_k < self.max_temperature_k
            or not self.min_temperature_k <= self.temperature_k <= self.max_temperature_k
        ):
            raise ValueError("Invalid thermal pipe temperature range")
        if self.distribution_parameter != 1 or self.drift_velocity_m_s != 0:
            raise PropertyModelError("Thermal pipe requires homogeneous C0=1 and Vd=0")
        if self.elevation_change_m != 0:
            raise PropertyModelError("Thermal reference currently requires a horizontal pipe")

    def _decode(self, u, validate=True):
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != 4:
            raise PropertyModelError("Thermal pipe requires four conservative variables")
        alpha = 1 - u[..., 0] / self.liquid_density_kg_m3
        rho = u[..., 0] + u[..., 1]
        if np.any(alpha <= 0) or np.any(alpha >= 1) or np.any(u[..., 1] <= 0):
            raise PropertyModelError("Thermal pipe loses a phase")
        v = u[..., 2] / rho
        cv = u[..., 0] * self.liquid_cv_j_kg_k + u[..., 1] * self.gas_cv_j_kg_k
        kinetic = 0.5 * u[..., 2] * v
        t = (u[..., 3] - kinetic) / cv
        p = u[..., 1] * self.gas_constant_j_kg_k * t / alpha
        if not np.isfinite(np.stack([alpha, rho, v, t, p])).all() or np.any(t <= 0):
            raise PropertyModelError("Thermal pipe has invalid thermodynamic state")
        if validate and (
            np.any(p < self.min_pressure_pa)
            or np.any(p > self.max_pressure_pa)
            or np.any(alpha < self.minimum_void_fraction)
            or np.any(alpha > 1 - self.minimum_void_fraction)
            or np.any(t < self.min_temperature_k)
            or np.any(t > self.max_temperature_k)
        ):
            raise PropertyModelError("Thermal pipe outside pressure/temperature/void domain")
        return p, alpha, v, t, cv

    def primitive(self, pressure_pa, gas_void_fraction, mixture_velocity_m_s, temperature_k=None):
        p, a, v, t = np.broadcast_arrays(
            pressure_pa,
            gas_void_fraction,
            mixture_velocity_m_s,
            self.temperature_k if temperature_k is None else temperature_k,
        )
        ml = self.liquid_density_kg_m3 * (1 - a)
        mg = p * a / (self.gas_constant_j_kg_k * t)
        rho = ml + mg
        energy = (ml * self.liquid_cv_j_kg_k + mg * self.gas_cv_j_kg_k) * t + 0.5 * rho * v * v
        u = np.stack([ml, mg, rho * v, energy], axis=-1)
        self._decode(u)
        return u

    def _state(self, u, time_s):
        return ThermalPipeInventory(
            *(tuple(u[:, k] * self.cell_volume_m3) for k in range(3)),
            time_s,
            self.model_id,
            tuple(u[:, 3] * self.cell_volume_m3),
        )

    def _conservative(self, state):
        if (
            not isinstance(state, ThermalPipeInventory)
            or state.model_id != self.model_id
            or len(state.liquid_mass_kg) != self.cells
        ):
            raise PropertyModelError("Thermal pipe inventory/model mismatch")
        return (
            np.array(
                [
                    state.liquid_mass_kg,
                    state.gas_mass_kg,
                    state.momentum_kg_m_s,
                    state.total_energy_j,
                ]
            ).T
            / self.cell_volume_m3
        )

    def initialize(
        self,
        pressure_pa,
        gas_void_fraction,
        mixture_velocity_m_s=0.0,
        time_s=0.0,
        temperature_k=None,
    ):
        u = np.broadcast_to(
            self.primitive(pressure_pa, gas_void_fraction, mixture_velocity_m_s, temperature_k),
            (self.cells, 4),
        ).copy()
        state = self._state(u, time_s)
        self.inspect(state)
        return state

    def inspect(self, state):
        u = self._conservative(state)
        p, a, v, t, cv = self._decode(u)
        # Explicit entropy references: Tref=1 K and gas density ref=1 kg/m3.
        entropy = cv * np.log(t) - u[:, 1] * self.gas_constant_j_kg_k * np.log(u[:, 1] / a)
        return {
            "pressure_pa": p.tolist(),
            "gas_void_fraction": a.tolist(),
            "liquid_holdup": (1 - a).tolist(),
            "mixture_velocity_m_s": v.tolist(),
            "liquid_velocity_m_s": v.tolist(),
            "gas_velocity_m_s": v.tolist(),
            "temperature_k": t.tolist(),
            "internal_energy_j": ((u[:, 3] - 0.5 * u[:, 2] * v) * self.cell_volume_m3).tolist(),
            "kinetic_energy_j": (0.5 * u[:, 2] * v * self.cell_volume_m3).tolist(),
            "entropy_j_k": (entropy * self.cell_volume_m3).tolist(),
        }

    def _flux(self, u):
        p, _a, v, _t, _cv = self._decode(u, validate=False)
        return np.stack(
            [u[..., 0] * v, u[..., 1] * v, u[..., 2] * v + p, (u[..., 3] + p) * v], axis=-1
        )

    def _wave_speed(self, u):
        p, a, v, _t, cv = self._decode(u, validate=False)
        gamma = 1 + u[..., 1] * self.gas_constant_j_kg_k / cv
        return abs(v) + np.sqrt(gamma * p / ((u[..., 0] + u[..., 1]) * a))

    def _boundary(self, spec):
        if spec is None or spec == "closed":
            return None
        if not isinstance(spec, dict) or set(spec) != {
            "pressure_pa",
            "gas_void_fraction",
            "mixture_velocity_m_s",
            "temperature_k",
        }:
            raise ValueError("Thermal boundary needs explicit P, void, velocity and temperature")
        u = self.primitive(**spec)
        if u.shape != (4,):
            raise ValueError("Thermal boundary state must be scalar")
        return u

    def _interface_flux(self, left, right, dissipation=1.0):
        if self.numerical_flux == "entropy_stable":
            return entropy_stable_flux(self, left, right, dissipation)
        speed = np.maximum(self._wave_speed(left), self._wave_speed(right))
        return 0.5 * (self._flux(left) + self._flux(right)) - 0.5 * speed[..., None] * (
            right - left
        )

    def _step(self, state, dt, left, right):
        old = self._conservative(state)
        # Lag the sensor, not the conservative flux: all state-dependent fluxes
        # below are still evaluated at the new implicit state. Retry recomputes
        # this deterministic sensor from the same accepted state.
        factor = (
            entropy_viscosity_sensor(self, old, left, right)
            if self.numerical_flux == "entropy_stable" and self.adaptive_entropy_dissipation
            else np.ones(self.cells - 1)
        )
        p, a, _v, t, _cv = self._decode(old)
        ps = max(1.0, float(np.mean(p)))
        ts = max(1.0, float(np.mean(t)))
        ms = max(1.0, float(np.max(abs(old[:, 2]))), math.sqrt(ps * self.liquid_density_kg_m3))
        initial = np.stack([p / ps, a, old[:, 2] / ms, t / ts], axis=-1)
        lower = np.tile(
            [
                self.min_pressure_pa / ps,
                self.minimum_void_fraction,
                -np.inf,
                self.min_temperature_k / ts,
            ],
            (self.cells, 1),
        ).flatten()
        upper = np.tile(
            [
                self.max_pressure_pa / ps,
                1 - self.minimum_void_fraction,
                np.inf,
                self.max_temperature_k / ts,
            ],
            (self.cells, 1),
        ).flatten()
        scale = np.maximum(abs(old), [1.0, 1e-3, ms, 1.0])
        pattern = lil_matrix((4 * self.cells, 4 * self.cells), dtype=int)
        for i in range(self.cells):
            pattern[4 * i : 4 * i + 4, 4 * max(0, i - 1) : 4 * min(self.cells, i + 2)] = 1
        cache = {}

        def evaluate(x):
            key = tuple(x)
            if key in cache:
                return cache[key]
            z = x.reshape(-1, 4)
            alpha = z[:, 1]
            temp = z[:, 3] * ts
            ml = self.liquid_density_kg_m3 * (1 - alpha)
            mg = z[:, 0] * ps * alpha / (self.gas_constant_j_kg_k * temp)
            mom = z[:, 2] * ms
            rho = ml + mg
            energy = (
                ml * self.liquid_cv_j_kg_k + mg * self.gas_cv_j_kg_k
            ) * temp + 0.5 * mom * mom / rho
            u = np.stack([ml, mg, mom, energy], axis=-1)
            flux = self._flux(u)
            wave = self._wave_speed(u)
            faces = np.empty((self.cells + 1, 4))
            faces[1:-1] = self._interface_flux(u[:-1], u[1:], factor)
            for side, cell, ghost in ((0, 0, left), (-1, -1, right)):
                if ghost is None:
                    if self.numerical_flux == "rusanov":
                        sign = -1 if side == 0 else 1
                        faces[side] = [0, 0, flux[cell, 2] + sign * wave[cell] * u[cell, 2], 0]
                    else:
                        reflected = np.array(u[cell], copy=True)
                        reflected[2] *= -1
                        l, r = (reflected, u[cell]) if side == 0 else (u[cell], reflected)
                        faces[side] = self._interface_flux(l, r)
                else:
                    l, r = (ghost, u[cell]) if side == 0 else (u[cell], ghost)
                    faces[side] = self._interface_flux(l, r)
            source = np.zeros_like(u)
            velocity = mom / rho
            source[:, 2] = (
                -self.darcy_factor * rho * velocity * abs(velocity) / (2 * self.diameter_m)
            )
            # Stationary-wall friction dissipates mechanical energy INSIDE
            # the fluid. It is not an external sink in total energy.
            source[:, 3] = (
                self.heat_transfer_w_m_k * (self.ambient_temperature_k - temp) / self.area_m2
            )
            residual = (
                u - old + dt * (faces[1:] - faces[:-1]) / (self.length_m / self.cells) - dt * source
            )
            value = (residual / scale, u, faces, source)
            if len(cache) > 64:
                cache.clear()
            cache[key] = value
            return value

        fit = least_squares(
            lambda x: evaluate(x)[0].flatten(),
            initial.flatten(),
            bounds=(lower, upper),
            jac_sparsity=pattern.tocsr(),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=150,
        )
        residual, u, faces, source = evaluate(fit.x)
        if not fit.success or np.max(abs(residual)) > 1e-9:
            raise HydraulicConvergenceError("Thermal pipe nonlinear solve failed")
        _, _, _, new_t, _ = self._decode(u)
        if (
            np.max(abs(u[:, :2] - old[:, :2]) / old[:, :2]) > self.max_fractional_inventory_change
            or np.max(abs(new_t - t)) > self.max_temperature_change_k
        ):
            raise HydraulicConvergenceError("Thermal pipe state-change accuracy guard")
        end = self._state(u, state.time_s + dt)
        self.inspect(end)
        delta = np.sum(u - old, axis=0) * self.cell_volume_m3
        boundary = dt * self.area_m2 * (faces[0] - faces[-1])
        body = dt * self.cell_volume_m3 * np.sum(source, axis=0)
        error = delta - boundary - body
        norm = np.maximum(1.0, np.sum(abs(old), axis=0) * self.cell_volume_m3)
        if np.max(abs(error) / norm) > 1e-8:
            raise ConservationError("Thermal pipe conservative balance failed")
        energy_throughput = dt * self.area_m2 * (abs(faces[0, 3]) + abs(faces[-1, 3])) + abs(
            body[3]
        )
        if abs(error[3]) > 1e-6 + 1e-8 * max(1.0, energy_throughput):
            raise ConservationError("Thermal pipe energy transaction balance failed")
        entropy_production = None
        if self.numerical_flux == "entropy_stable":
            entropy_left = (
                0.0 if left is None else physical_entropy_flux(self, left, u[0], faces[0])
            )
            entropy_right = (
                0.0 if right is None else physical_entropy_flux(self, u[-1], right, faces[-1])
            )
            entropy_boundary = dt * self.area_m2 * (entropy_left - entropy_right)
            entropy_production = float(
                self.cell_volume_m3 * np.sum(entropy_density(self, u) - entropy_density(self, old))
                - entropy_boundary
                - body[3] / self.ambient_temperature_k
            )
            entropy_scale = dt * self.area_m2 * (abs(entropy_left) + abs(entropy_right)) + abs(
                body[3] / self.ambient_temperature_k
            )
            if entropy_production < -1e-7 - 1e-9 * max(1.0, entropy_scale):
                raise ConservationError("Thermal pipe discrete entropy balance failed")
        return end, {
            "dt_s": dt,
            "left_boundary_transfer": (dt * self.area_m2 * faces[0]).tolist(),
            "right_boundary_transfer": (dt * self.area_m2 * faces[-1]).tolist(),
            "body_impulse_kg_m_s": float(body[2]),
            "heat_j": float(body[3]),
            "mass_residuals_kg": error[:2].tolist(),
            "momentum_residual_kg_m_s": float(error[2]),
            "energy_residual_j": float(error[3]),
            "nonlinear_evaluations": fit.nfev,
            "numerical_flux": self.numerical_flux,
            "thermal_entropy_dissipation": self.thermal_entropy_dissipation,
            "adaptive_entropy_dissipation": self.adaptive_entropy_dissipation,
            "entropy_dissipation_min": float(np.min(factor)),
            "entropy_dissipation_max": float(np.max(factor)),
            "entropy_production_j_k": entropy_production,
        }

    def advance(
        self,
        state,
        duration_s,
        left="closed",
        right="closed",
        initial_dt_s=0.01,
        maximum_dt_s=None,
        minimum_dt_s=1e-7,
        max_substeps=10000,
    ):
        self.inspect(state)
        duration = _number(duration_s, "duration_s", positive=True)
        dt = _number(initial_dt_s, "initial_dt_s", positive=True)
        maximum = _number(
            maximum_dt_s if maximum_dt_s is not None else dt, "maximum_dt_s", positive=True
        )
        minimum = _number(minimum_dt_s, "minimum_dt_s", positive=True)
        if isinstance(max_substeps, bool) or not isinstance(max_substeps, int) or max_substeps < 1:
            raise ValueError("Invalid thermal pipe substep budget")
        left, right = self._boundary(left), self._boundary(right)
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Thermal interval cannot advance representable time")
        current = state
        elapsed = 0.0
        steps = []
        failures = 0
        while elapsed < duration:
            if len(steps) >= max_substeps:
                raise HydraulicConvergenceError("Thermal pipe substep budget exhausted")
            step = min(dt, maximum, duration - elapsed)
            try:
                new, record = self._step(current, step, left, right)
            except (HydraulicConvergenceError, PropertyModelError) as error:
                failures += 1
                if step / 2 < minimum:
                    raise HydraulicConvergenceError(f"Thermal pipe minimum dt: {error}") from error
                dt = step / 2
                continue
            following = math.fsum([elapsed, step])
            if following <= elapsed or state.time_s + following <= current.time_s:
                raise ValueError("Thermal pipe cannot advance representable time")
            elapsed = following
            current = ThermalPipeInventory(
                new.liquid_mass_kg,
                new.gas_mass_kg,
                new.momentum_kg_m_s,
                state.time_s + elapsed,
                new.model_id,
                new.total_energy_j,
            )
            steps.append(record)
            dt = min(maximum, step * 1.5)
        return {
            "state": current.to_dict(),
            "properties": self.inspect(current),
            "steps": steps,
            "failed_steps": failures,
            "duration_s": duration,
            "component_order": ["liquid", "free_gas"],
            "model": f"thermal_homogeneous_frozen_two_phase_{self.numerical_flux}_backward_Euler",
        }
