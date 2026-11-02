"""Conservative isothermal two-phase drift-flux finite-volume CPU reference.

Liquid is incompressible, gas ideal, with no dissolution or phase change.
Explicit C0, drift and Darcy coefficients require independent calibration.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest


@dataclass(frozen=True)
class PipeInventory:
    liquid_mass_kg: tuple
    gas_mass_kg: tuple
    momentum_kg_m_s: tuple
    time_s: float
    model_id: str

    def __post_init__(self):
        for key in ("liquid_mass_kg", "gas_mass_kg", "momentum_kg_m_s"):
            values = tuple(_number(x, key) for x in getattr(self, key))
            object.__setattr__(self, key, values)
        if (
            not self.liquid_mass_kg
            or len({len(self.liquid_mass_kg), len(self.gas_mass_kg), len(self.momentum_kg_m_s)})
            != 1
        ):
            raise ValueError("Pipe inventory arrays must match")
        if min(self.liquid_mass_kg + self.gas_mass_kg) <= 0:
            raise ValueError("Both phases must have positive inventories")
        _number(self.time_s, "time_s", nonnegative=True)
        if not isinstance(self.model_id, str):
            raise TypeError("Pipe inventory needs model_id")

    def to_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class DriftFluxPipe:
    length_m: float
    diameter_m: float
    cells: int
    liquid_density_kg_m3: float
    gas_constant_j_kg_k: float
    temperature_k: float
    distribution_parameter: float
    drift_velocity_m_s: float
    darcy_factor: float
    elevation_change_m: float = 0.0
    min_pressure_pa: float = 1e4
    max_pressure_pa: float = 1e8
    minimum_void_fraction: float = 1e-4
    max_fractional_inventory_change: float = 0.2

    def __post_init__(self):
        for key in (
            "length_m",
            "diameter_m",
            "liquid_density_kg_m3",
            "gas_constant_j_kg_k",
            "temperature_k",
            "distribution_parameter",
            "min_pressure_pa",
            "max_pressure_pa",
            "minimum_void_fraction",
            "max_fractional_inventory_change",
        ):
            _number(getattr(self, key), key, positive=True)
        for key in ("drift_velocity_m_s", "elevation_change_m"):
            _number(getattr(self, key), key)
        _number(self.darcy_factor, "darcy_factor", nonnegative=True)
        if (
            isinstance(self.cells, bool)
            or not isinstance(self.cells, int)
            or not 2 <= self.cells <= 1000
        ):
            raise ValueError("CPU reference supports 2..1000 finite-volume cells")
        if (
            self.min_pressure_pa >= self.max_pressure_pa
            or self.minimum_void_fraction >= 0.5
            or self.max_fractional_inventory_change > 1
            or abs(self.elevation_change_m) > self.length_m
        ):
            raise ValueError("Invalid pipe domain")

    @property
    def area_m2(self):
        return math.pi * self.diameter_m**2 / 4

    @property
    def cell_volume_m3(self):
        return self.area_m2 * self.length_m / self.cells

    @property
    def model_id(self):
        return _digest(asdict(self))

    def _decode(self, u, validate=True):
        u = np.asarray(u, dtype=float)
        alpha = 1 - u[..., 0] / self.liquid_density_kg_m3
        if np.any(alpha <= 0) or np.any(alpha >= 1) or np.any(u[..., 1] <= 0):
            raise PropertyModelError("Drift-flux state loses a phase")
        p = u[..., 1] * self.gas_constant_j_kg_k * self.temperature_k / alpha
        rg = p / (self.gas_constant_j_kg_k * self.temperature_k)
        denominator = (
            self.liquid_density_kg_m3
            + (rg - self.liquid_density_kg_m3) * alpha * self.distribution_parameter
        )
        if np.any(denominator <= 0):
            raise PropertyModelError("Drift-flux momentum/velocity closure is singular")
        j = (
            u[..., 2] - (rg - self.liquid_density_kg_m3) * alpha * self.drift_velocity_m_s
        ) / denominator
        jg = alpha * (self.distribution_parameter * j + self.drift_velocity_m_s)
        jl = j - jg
        if not np.isfinite(np.stack([p, alpha, j, jl, jg])).all():
            raise PropertyModelError("Nonfinite drift-flux properties")
        if validate and (
            np.any(p < self.min_pressure_pa)
            or np.any(p > self.max_pressure_pa)
            or np.any(alpha < self.minimum_void_fraction)
            or np.any(alpha > 1 - self.minimum_void_fraction)
        ):
            raise PropertyModelError("Drift-flux pressure/void fraction outside declared domain")
        return p, alpha, j, jl, jg, rg

    def primitive(self, pressure_pa, gas_void_fraction, mixture_velocity_m_s):
        p, alpha, j = np.broadcast_arrays(pressure_pa, gas_void_fraction, mixture_velocity_m_s)
        rg = p / (self.gas_constant_j_kg_k * self.temperature_k)
        jg = alpha * (self.distribution_parameter * j + self.drift_velocity_m_s)
        u = np.stack(
            [
                self.liquid_density_kg_m3 * (1 - alpha),
                rg * alpha,
                self.liquid_density_kg_m3 * (j - jg) + rg * jg,
            ],
            axis=-1,
        )
        self._decode(u)
        return u

    def initialize(self, pressure_pa, gas_void_fraction, mixture_velocity_m_s=0.0, time_s=0.0):
        u = np.broadcast_to(
            self.primitive(pressure_pa, gas_void_fraction, mixture_velocity_m_s), (self.cells, 3)
        ).copy()
        state = PipeInventory(
            *(tuple(u[:, k] * self.cell_volume_m3) for k in range(3)), time_s, self.model_id
        )
        self.inspect(state)
        return state

    def inspect(self, state):
        if state.model_id != self.model_id or len(state.liquid_mass_kg) != self.cells:
            raise PropertyModelError("Pipe inventory/model mismatch")
        u = (
            np.array([state.liquid_mass_kg, state.gas_mass_kg, state.momentum_kg_m_s]).T
            / self.cell_volume_m3
        )
        p, a, j, jl, jg, _ = self._decode(u)
        return {
            "pressure_pa": p.tolist(),
            "gas_void_fraction": a.tolist(),
            "liquid_holdup": (1 - a).tolist(),
            "mixture_velocity_m_s": j.tolist(),
            "liquid_velocity_m_s": (jl / (1 - a)).tolist(),
            "gas_velocity_m_s": (jg / a).tolist(),
        }

    def _flux(self, u):
        p, alpha, _j, jl, jg, rg = self._decode(u, validate=False)
        return np.stack(
            [
                self.liquid_density_kg_m3 * jl,
                rg * jg,
                self.liquid_density_kg_m3 * jl**2 / (1 - alpha) + rg * jg**2 / alpha + p,
            ],
            axis=-1,
        )

    def _wave_speed(self, u):
        # Local flux Jacobian in CONSERVATIVE coordinates, including slip.
        matrix = np.empty((3, 3))
        for k in range(3):
            h = 1e-5 * max(abs(u[k]), 1e-3)
            if k == 0:
                h = min(h, 0.1 * min(u[0], self.liquid_density_kg_m3 - u[0]))
            if k == 1:
                h = min(h, 0.1 * u[1])
            plus, minus = u.copy(), u.copy()
            plus[k] += h
            minus[k] -= h
            matrix[:, k] = (self._flux(plus) - self._flux(minus)) / (2 * h)
        eigenvalues = np.linalg.eigvals(matrix)
        if np.max(abs(eigenvalues.imag)) > 1e-6 * max(1, np.max(abs(eigenvalues.real))):
            raise PropertyModelError("Specified drift-flux closure is not locally hyperbolic")
        return max(abs(eigenvalues.real))

    def _boundary(self, spec):
        if spec is None or spec == "closed":
            return None
        if not isinstance(spec, dict) or set(spec) != {
            "pressure_pa",
            "gas_void_fraction",
            "mixture_velocity_m_s",
        }:
            raise ValueError("Boundary must be closed or an explicit reservoir state")
        u = self.primitive(**spec)
        if u.shape != (3,):
            raise ValueError("Boundary reservoir state must be scalar")
        return u

    def _step(self, state, dt, left, right):
        self.inspect(state)
        old = (
            np.array([state.liquid_mass_kg, state.gas_mass_kg, state.momentum_kg_m_s]).T
            / self.cell_volume_m3
        )
        p, alpha, _j, _, _, _ = self._decode(old)
        pscale = max(1.0, float(np.mean(p)))
        mscale = max(
            1.0, float(np.max(abs(old[:, 2]))), math.sqrt(pscale * self.liquid_density_kg_m3)
        )
        x = np.stack([p / pscale, alpha, old[:, 2] / mscale], axis=-1)
        lo = np.tile(
            [self.min_pressure_pa / pscale, self.minimum_void_fraction, -np.inf], (self.cells, 1)
        ).flatten()
        hi = np.tile(
            [self.max_pressure_pa / pscale, 1 - self.minimum_void_fraction, np.inf], (self.cells, 1)
        ).flatten()
        scale = np.maximum(abs(old), np.array([1.0, 1e-3, mscale]))
        dx = self.length_m / self.cells
        pattern = lil_matrix((3 * self.cells, 3 * self.cells), dtype=int)
        for i in range(self.cells):
            pattern[3 * i : 3 * i + 3, 3 * max(0, i - 1) : 3 * min(self.cells, i + 2)] = 1
        cache = {}

        def evaluate(y):
            key = tuple(y)
            if key in cache:
                return cache[key]
            z = y.reshape(-1, 3)
            a = z[:, 1]
            u = np.stack(
                [
                    self.liquid_density_kg_m3 * (1 - a),
                    z[:, 0] * pscale * a / (self.gas_constant_j_kg_k * self.temperature_k),
                    z[:, 2] * mscale,
                ],
                axis=-1,
            )
            _p, a, j, _, _, _ = self._decode(u)
            flux = self._flux(u)
            wave = np.array([self._wave_speed(row) for row in u])
            faces = np.empty((self.cells + 1, 3))
            speed = np.maximum(wave[:-1], wave[1:])
            faces[1:-1] = 0.5 * (flux[:-1] + flux[1:]) - 0.5 * speed[:, None] * (u[1:] - u[:-1])
            for side, cell, ghost in ((0, 0, left), (-1, -1, right)):
                if ghost is None:
                    # Impermeable wall: exactly zero component flux; reflected
                    # momentum adds the wall reaction, including acoustic drag.
                    sign = -1 if side == 0 else 1
                    faces[side] = [0, 0, flux[cell, 2] + sign * wave[cell] * u[cell, 2]]
                else:
                    l, r = (ghost, u[cell]) if side == 0 else (u[cell], ghost)
                    speed_b = max(wave[cell], self._wave_speed(ghost))
                    faces[side] = 0.5 * (self._flux(l) + self._flux(r)) - 0.5 * speed_b * (r - l)
            rho = u[:, 0] + u[:, 1]
            source = np.zeros_like(u)
            source[:, 2] = (
                -rho * 9.80665 * self.elevation_change_m / self.length_m
                - self.darcy_factor * rho * j * abs(j) / (2 * self.diameter_m)
            )
            residual = u - old + dt * (faces[1:] - faces[:-1]) / dx - dt * source
            result = (residual / scale, u, faces, source)
            # A bounded cache avoids retaining every nonlinear probe on long lines.
            if len(cache) > 64:
                cache.clear()
            cache[key] = result
            return result

        fit = least_squares(
            lambda y: evaluate(y)[0].flatten(),
            x.flatten(),
            bounds=(lo, hi),
            jac_sparsity=pattern.tocsr(),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=100,
        )
        residual, u, faces, source = evaluate(fit.x)
        if not fit.success or np.max(abs(residual)) > 1e-9:
            raise HydraulicConvergenceError("Transient drift-flux Newton solve failed")
        if np.max(abs(u[:, :2] - old[:, :2]) / old[:, :2]) > self.max_fractional_inventory_change:
            raise HydraulicConvergenceError("Pipe step inventory change exceeds accuracy guard")
        end = PipeInventory(
            *(tuple(u[:, k] * self.cell_volume_m3) for k in range(3)),
            state.time_s + dt,
            self.model_id,
        )
        self.inspect(end)
        delta = np.sum(u - old, axis=0) * self.cell_volume_m3
        boundary = dt * self.area_m2 * (faces[0] - faces[-1])
        force = dt * self.cell_volume_m3 * np.sum(source, axis=0)
        error = delta - boundary - force
        if (
            np.max(abs(error) / np.maximum(1.0, np.sum(abs(old), axis=0) * self.cell_volume_m3))
            > 1e-8
        ):
            raise ConservationError("Transient pipe conservative balance failed")
        return end, {
            "dt_s": dt,
            "left_boundary_transfer": (dt * self.area_m2 * faces[0]).tolist(),
            "right_boundary_transfer": (dt * self.area_m2 * faces[-1]).tolist(),
            "body_impulse_kg_m_s": float(force[2]),
            "mass_residuals_kg": error[:2].tolist(),
            "momentum_residual_kg_m_s": float(error[2]),
            "nonlinear_evaluations": fit.nfev,
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
            raise ValueError("Invalid pipe substep budget")
        left, right = self._boundary(left), self._boundary(right)
        if not math.isfinite(state.time_s + duration) or state.time_s + duration <= state.time_s:
            raise ValueError("Pipe duration cannot advance representable time")
        current = state
        steps, elapsed, failed = [], 0.0, 0
        while elapsed < duration:
            if len(steps) >= max_substeps:
                raise HydraulicConvergenceError("Pipe substep budget exhausted")
            step = min(dt, maximum, duration - elapsed)
            try:
                new, record = self._step(current, step, left, right)
            except (HydraulicConvergenceError, PropertyModelError) as error:
                failed += 1
                if step / 2 < minimum:
                    raise HydraulicConvergenceError(
                        f"Pipe interval failed at minimum dt: {error}"
                    ) from error
                dt = step / 2
                continue
            following = math.fsum([elapsed, step])
            if following <= elapsed or state.time_s + following <= current.time_s:
                raise ValueError("Pipe substep cannot advance representable time")
            elapsed = following
            current = PipeInventory(
                new.liquid_mass_kg,
                new.gas_mass_kg,
                new.momentum_kg_m_s,
                state.time_s + elapsed,
                new.model_id,
            )
            steps.append(record)
            dt = min(maximum, step * 1.5)
        return {
            "state": current.to_dict(),
            "properties": self.inspect(current),
            "steps": steps,
            "failed_steps": failed,
            "duration_s": duration,
            "component_order": ["liquid", "free_gas"],
            "model": "isothermal_frozen_two_phase_drift_flux_Rusanov_backward_Euler",
        }
