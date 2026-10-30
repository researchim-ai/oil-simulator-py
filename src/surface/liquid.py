"""Steady, isothermal, incompressible single-liquid hydraulics in SI.

Elevation is positive upwards; pressure is absolute; sources inject mass.
Darcy-Weisbach uses 64/Re, Swamee-Jain, and a C1 cubic transition between
Re=2000 and 4000. Unknown pressures and signed pipe rates are solved together.
There is no gas, multiphase holdup, temperature solve, or automatic cavitation cap.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

G = 9.80665


def _number(value, name, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name}: invalid sign")
    return float(value)


def _name(value):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("Names must be nonempty strings without surrounding whitespace")


class HydraulicConvergenceError(RuntimeError):
    """The hydraulic equations did not converge; no passing result is returned."""


class InvalidPressureError(ValueError):
    """The incompressible liquid solution violates its pressure domain."""


@dataclass(frozen=True)
class Liquid:
    density_kg_m3: float
    viscosity_pa_s: float
    vapor_pressure_pa: float = 0.0

    def __post_init__(self):
        _number(self.density_kg_m3, "density_kg_m3", positive=True)
        _number(self.viscosity_pa_s, "viscosity_pa_s", positive=True)
        _number(self.vapor_pressure_pa, "vapor_pressure_pa", nonnegative=True)


@dataclass(frozen=True)
class Node:
    name: str
    elevation_m: float = 0.0
    pressure_pa: float | None = None
    source_mass_rate_kg_s: float = 0.0

    def __post_init__(self):
        _name(self.name)
        _number(self.elevation_m, "elevation_m")
        _number(self.source_mass_rate_kg_s, "source_mass_rate_kg_s")
        if self.pressure_pa is not None:
            _number(self.pressure_pa, "pressure_pa", positive=True)


@dataclass(frozen=True)
class Pipe:
    name: str
    start: str
    end: str
    length_m: float
    diameter_m: float
    roughness_m: float = 0.0
    minor_loss_coefficient: float = 0.0
    open: bool = True

    def __post_init__(self):
        for name in (self.name, self.start, self.end):
            _name(name)
        if self.start == self.end:
            raise ValueError(f"{self.name}: pipe endpoints must differ")
        _number(self.length_m, "length_m", positive=True)
        _number(self.diameter_m, "diameter_m", positive=True)
        _number(self.roughness_m, "roughness_m", nonnegative=True)
        _number(self.minor_loss_coefficient, "minor_loss_coefficient", nonnegative=True)
        if self.roughness_m / self.diameter_m > 0.05:
            raise ValueError(f"{self.name}: relative roughness exceeds 0.05")
        if not isinstance(self.open, bool):
            raise TypeError(f"{self.name}: open must be boolean")

    def loss_and_gradient(self, q_m3_s, liquid):
        """Signed dissipative pressure loss and d(loss)/dq, including q=0."""
        q = _number(q_m3_s, "q_m3_s")
        rho, mu = liquid.density_kg_m3, liquid.viscosity_pa_s
        d, length, k = self.diameter_m, self.length_m, self.minor_loss_coefficient
        area = math.pi * d * d / 4
        re_per_q = rho * d / (mu * area)
        re = abs(q) * re_per_q
        c = rho / (2 * area * area)
        laminar = 128 * mu * length / (math.pi * d**4)
        if re <= 2000:
            return laminar * q + c * k * q * abs(q), laminar + 2 * c * k * abs(q)

        def turbulent(r):
            term = 5.74 * r ** (-0.9)
            a = self.roughness_m / (3.7 * d) + term
            loga = math.log10(a)
            f = 0.25 / loga**2
            df = -0.5 * (-0.9 * term / (r * a * math.log(10))) / loga**3
            return f, df

        if re >= 4000:
            f, df = turbulent(re)
        else:
            # Cubic Hermite form of the continuous transition in EPANET 2.2.
            t = (re - 2000) / 2000
            a, da = 0.032, -64 / 2000**2
            b, db = turbulent(4000.0)
            f = (
                (2 * t**3 - 3 * t * t + 1) * a
                + (t**3 - 2 * t * t + t) * 2000 * da
                + (-2 * t**3 + 3 * t * t) * b
                + (t**3 - t * t) * 2000 * db
            )
            df = (
                (6 * t * t - 6 * t) * a / 2000
                + (3 * t * t - 4 * t + 1) * da
                + (-6 * t * t + 6 * t) * b / 2000
                + (3 * t * t - 2 * t) * db
            )
        loss = c * (f * length / d + k) * q * abs(q)
        gradient = c * abs(q) * (2 * (f * length / d + k) + length / d * df * re)
        return loss, gradient


@dataclass(frozen=True)
class NetworkResult:
    pressures_pa: dict
    pipe_mass_rates_kg_s: dict
    pipe_volume_rates_m3_s: dict
    boundary_mass_rates_kg_s: dict
    node_mass_residuals_kg_s: dict
    pipe_equation_residuals_pa: dict
    pipe_dissipation_w: dict
    power_balance_error_w: float
    iterations: int

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


class LiquidNetwork:
    """Arbitrary pipe graph with at least one pressure boundary per component.

    Positive source and pressure-boundary rates supply the network. Pipe
    rates are positive start->end. Fixed-pressure nodes can exchange the
    mass necessary to hold pressure; those rates are returned explicitly.
    """

    def __init__(self, fluid, nodes, pipes):
        if not isinstance(fluid, Liquid):
            raise TypeError("This network requires one incompressible Liquid")
        self.fluid, self.nodes, self.pipes = fluid, tuple(nodes), tuple(pipes)
        if not self.nodes or not all(isinstance(n, Node) for n in self.nodes):
            raise ValueError("Provide at least one Node")
        if not all(isinstance(p, Pipe) for p in self.pipes):
            raise ValueError("All branches must be Pipe objects")
        self._index = {n.name: i for i, n in enumerate(self.nodes)}
        if len(self._index) != len(self.nodes):
            raise ValueError("Duplicate node names")
        if len({p.name for p in self.pipes}) != len(self.pipes):
            raise ValueError("Duplicate pipe names")
        # Geometry and endpoints remain valid when closed branches reopen.
        for p in self.pipes:
            if p.start not in self._index or p.end not in self._index:
                raise ValueError(f"{p.name}: unknown endpoint")
            a, b = self._index[p.start], self._index[p.end]
            if abs(self.nodes[a].elevation_m - self.nodes[b].elevation_m) > p.length_m + 1e-9:
                raise ValueError(f"{p.name}: length is less than its elevation difference")
        self._active = tuple(p for p in self.pipes if p.open)
        rows, cols, values = [], [], []
        for e, p in enumerate(self._active):
            a, b = self._index[p.start], self._index[p.end]
            rows.extend((a, b))
            cols.extend((e, e))
            values.extend((1.0, -1.0))
        self._B = sp.csc_matrix((values, (rows, cols)), shape=(len(self.nodes), len(self._active)))
        graph = abs(self._B) @ abs(self._B.T)
        _, self._components = sp.csgraph.connected_components(graph, directed=False)

    @classmethod
    def from_dict(cls, data):
        unknown = set(data) - {"model", "fluid", "nodes", "pipes"}
        if unknown or data.get("model", "incompressible_liquid") != "incompressible_liquid":
            raise ValueError(f"Unsupported surface model or fields: {sorted(unknown)}")
        return cls(
            Liquid(**data["fluid"]),
            [Node(**n) for n in data["nodes"]],
            [Pipe(**p) for p in data["pipes"]],
        )

    def to_dict(self):
        return {
            "model": "incompressible_liquid",
            "fluid": asdict(self.fluid),
            "nodes": [asdict(n) for n in self.nodes],
            "pipes": [asdict(p) for p in self.pipes],
        }

    def solve(
        self,
        source_mass_rates_kg_s=None,
        boundary_pressures_pa=None,
        initial=None,
        pressure_tolerance_pa=1e-3,
        mass_tolerance_kg_s=1e-8,
        max_iterations=60,
    ):
        """Solve without mutating the model or the previous result.

        Sources and pressures may be overridden by node name at each call.
        A pressure override of None releases that pressure boundary. This is
        the interface for subsequent well/facility coupling, not a coupled
        reservoir solve in this first single-liquid implementation.
        """
        ptol = _number(pressure_tolerance_pa, "pressure_tolerance_pa", positive=True)
        mtol = _number(mass_tolerance_kg_s, "mass_tolerance_kg_s", positive=True)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        rho = self.fluid.density_kg_m3
        z = np.array([n.elevation_m for n in self.nodes])
        sources = np.array([n.source_mass_rate_kg_s for n in self.nodes], dtype=float)
        pressures = np.array(
            [np.nan if n.pressure_pa is None else n.pressure_pa for n in self.nodes], dtype=float
        )
        for mapping, target, pressure in [
            (source_mass_rates_kg_s, sources, False),
            (boundary_pressures_pa, pressures, True),
        ]:
            for name, value in (mapping or {}).items():
                if name not in self._index:
                    raise ValueError(f"Unknown boundary node: {name}")
                target[self._index[name]] = (
                    np.nan
                    if pressure and value is None
                    else _number(value, name, positive=pressure)
                )
        fixed = np.isfinite(pressures)
        if np.any(pressures[fixed] <= self.fluid.vapor_pressure_pa):
            raise InvalidPressureError("Pressure boundary is at or below vapor pressure")
        for component in np.unique(self._components):
            if not fixed[self._components == component].any():
                names = [
                    n.name for i, n in enumerate(self.nodes) if self._components[i] == component
                ]
                raise ValueError(f"No pressure boundary in connected component: {names}")
        free = np.flatnonzero(~fixed)
        Bf, B = self._B[free].tocsc(), self._B
        H = np.where(fixed, pressures + rho * G * z, 0.0)
        if initial is None:
            conductance = np.array(
                [
                    math.pi * p.diameter_m**4 / (128 * self.fluid.viscosity_pa_s * p.length_m)
                    for p in self._active
                ]
            )
            if len(free):
                A = Bf @ sp.diags(conductance) @ Bf.T
                H[free] = splu(A.tocsc()).solve(
                    sources[free] / rho - Bf @ (conductance * (B.T @ H))
                )
            q = conductance * (B.T @ H)
        else:
            if set(initial.pressures_pa) != set(self._index) or set(
                initial.pipe_mass_rates_kg_s
            ) != {p.name for p in self.pipes}:
                raise ValueError("Initial result does not match network node/pipe names")
            H[free] = [
                _number(initial.pressures_pa[self.nodes[i].name], "initial pressure", positive=True)
                + rho * G * z[i]
                for i in free
            ]
            q = np.array(
                [
                    _number(initial.pipe_mass_rates_kg_s[p.name], "initial rate") / rho
                    for p in self._active
                ]
            )
        pscale = max(1e5, float(np.max(np.abs(H))))
        qscale = max(
            1e-8,
            float(np.abs(sources).sum()) / rho,
            max((math.pi * p.diameter_m**2 / 4 for p in self._active), default=0.0),
        )
        mscale = max(1.0, rho * qscale)

        def residual(head, rate):
            losses = np.array(
                [p.loss_and_gradient(float(v), self.fluid) for p, v in zip(self._active, rate)]
            ).reshape(-1, 2)
            re = B.T @ head - losses[:, 0]
            rm = rho * (Bf @ rate) - sources[free]
            return losses, re, rm, np.concatenate((re / pscale, rm / mscale))

        for iteration in range(max_iterations + 1):
            losses, re, rm, r = residual(H, q)
            if not np.isfinite(r).all():
                raise HydraulicConvergenceError("Nonfinite hydraulic residual")
            if np.max(np.abs(re), initial=0.0) <= ptol and np.max(np.abs(rm), initial=0.0) <= mtol:
                break
            if iteration == max_iterations:
                raise HydraulicConvergenceError(
                    f"Iteration limit: pressure={abs(re).max(initial=0.0)} Pa, "
                    f"mass={abs(rm).max(initial=0.0)} kg/s"
                )
            J = sp.bmat(
                [
                    [Bf.T, -sp.diags(losses[:, 1] * qscale / pscale)],
                    [sp.csc_matrix((len(free), len(free))), Bf * (rho * qscale / mscale)],
                ],
                format="csc",
            )
            try:
                step = splu(J).solve(-r)
            except RuntimeError as error:
                raise HydraulicConvergenceError("Singular hydraulic Jacobian") from error
            if not np.isfinite(step).all():
                raise HydraulicConvergenceError("Nonfinite hydraulic update")
            norm = np.linalg.norm(r)
            alpha = 1.0
            for _ in range(35):
                trial_H = H.copy()
                trial_H[free] += alpha * pscale * step[: len(free)]
                trial_q = q + alpha * qscale * step[len(free) :]
                _, _, _, trial_r = residual(trial_H, trial_q)
                if (
                    np.isfinite(trial_r).all()
                    and np.linalg.norm(trial_r) <= (1 - 1e-4 * alpha) * norm
                ):
                    H, q = trial_H, trial_q
                    break
                alpha *= 0.5
            else:
                raise HydraulicConvergenceError("Hydraulic line search failed")
        solved_pressures = H - rho * G * z
        solved_pressures[fixed] = pressures[fixed]
        pressures = solved_pressures
        if np.any(pressures <= self.fluid.vapor_pressure_pa):
            bad = [
                n.name
                for i, n in enumerate(self.nodes)
                if pressures[i] <= self.fluid.vapor_pressure_pa
            ]
            raise InvalidPressureError(f"Liquid pressure is at or below vapor pressure: {bad}")
        boundary = np.where(fixed, rho * (B @ q) - sources, 0.0)
        mass_residual = rho * (B @ q) - sources - boundary
        mass_rates = dict.fromkeys((p.name for p in self.pipes), 0.0)
        mass_rates.update({p.name: float(v * rho) for p, v in zip(self._active, q)})
        dissipation = {p.name: float(v * loss) for p, v, loss in zip(self._active, q, losses[:, 0])}
        return NetworkResult(
            pressures_pa={n.name: float(v) for n, v in zip(self.nodes, pressures)},
            pipe_mass_rates_kg_s=mass_rates,
            pipe_volume_rates_m3_s={name: v / rho for name, v in mass_rates.items()},
            boundary_mass_rates_kg_s={
                n.name: float(boundary[i]) for i, n in enumerate(self.nodes) if fixed[i]
            },
            node_mass_residuals_kg_s={n.name: float(v) for n, v in zip(self.nodes, mass_residual)},
            pipe_equation_residuals_pa={p.name: float(v) for p, v in zip(self._active, re)},
            pipe_dissipation_w=dissipation,
            power_balance_error_w=float(H @ (sources + boundary) / rho - sum(dissipation.values())),
            iterations=iteration,
        )
