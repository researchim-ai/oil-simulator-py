"""Conservative horizontal single-phase PR Euler pipe with viscous wall drag.

Stores component moles, axial momentum and U+K. Fixed-bound Rusanov faces
and implicit Euler share a Torch Newton. No gravity, phase slip, finite-wall
heat, vessel/nozzle coupling or universal shock/phase-change validity is claimed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .darcy_hydraulics import darcy_pressure_loss
from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .torch_linear import TorchNewton, checked_device


@dataclass(frozen=True)
class PRDirichletBoundary:
    pressure_pa: float
    temperature_k: float
    mole_fractions: tuple
    fluid_id: str

    def __post_init__(self):
        _number(self.pressure_pa, "boundary pressure", positive=True)
        _number(self.temperature_k, "boundary temperature", positive=True)
        object.__setattr__(self, "mole_fractions", tuple(self.mole_fractions))
        for x in self.mole_fractions:
            _number(x, "boundary fraction", nonnegative=True)
        if abs(math.fsum(self.mole_fractions) - 1) > 1e-10:
            raise ValueError("Boundary fractions must sum to one")

    def to_dict(self):
        data = asdict(self)
        data["mole_fractions"] = list(self.mole_fractions)
        return data


@dataclass(frozen=True)
class PRMomentumState:
    component_moles: tuple
    momentum_kg_m_s: tuple
    total_energy_j: tuple
    solution_variables: tuple
    time_s: float
    next_step_s: float
    model_id: str

    def __post_init__(self):
        object.__setattr__(
            self, "component_moles", tuple(tuple(row) for row in self.component_moles)
        )
        for key in ("momentum_kg_m_s", "total_energy_j", "solution_variables"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for row in self.component_moles:
            for n in row:
                _number(n, "inventory", nonnegative=True)
        for key in ("momentum_kg_m_s", "total_energy_j", "solution_variables"):
            for value in getattr(self, key):
                _number(value, key)
        _number(self.time_s, "time", nonnegative=True)
        _number(self.next_step_s, "step", positive=True)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown momentum state fields")
        return cls(**data)


class PRMomentumPipe:
    def __init__(
        self,
        fluid,
        *,
        length_m,
        diameter_m,
        cells,
        wave_speed_bound_m_s,
        maximum_velocity_m_s=100.0,
        dynamic_viscosity_pa_s=0.0,
        roughness_m=0.0,
        boundary_mode="ends",
        initial_step_s=1e-5,
        minimum_step_s=1e-10,
        maximum_step_s=0.01,
        relative_tolerance=1e-4,
        temperature_tolerance_k=0.002,
        velocity_tolerance_m_s=1e-3,
        maximum_attempts=1000,
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Expected EquilibriumFluid")
        checked_device(fluid.device)
        self.fluid = fluid
        for key, value in locals().copy().items():
            if key in (
                "length_m",
                "diameter_m",
                "wave_speed_bound_m_s",
                "maximum_velocity_m_s",
                "initial_step_s",
                "minimum_step_s",
                "maximum_step_s",
                "relative_tolerance",
                "temperature_tolerance_k",
                "velocity_tolerance_m_s",
            ):
                setattr(self, key, _number(value, key, positive=True))
            if key in ("dynamic_viscosity_pa_s", "roughness_m"):
                setattr(self, key, _number(value, key, nonnegative=True))
        if isinstance(cells, bool) or not isinstance(cells, int) or not 2 <= cells <= 128:
            raise ValueError("Pipe requires 2..128 cells subject to 768-unknown budget")
        if boundary_mode not in ("periodic", "ends"):
            raise ValueError("Unknown pipe boundary mode")
        if (
            not minimum_step_s <= initial_step_s <= maximum_step_s
            or maximum_velocity_m_s >= wave_speed_bound_m_s
        ):
            raise ValueError("Invalid time/velocity bounds")
        if roughness_m >= diameter_m / 2 or (not dynamic_viscosity_pa_s and roughness_m):
            raise ValueError("Invalid roughness for pipe drag")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1
        ):
            raise ValueError("Positive attempt budget required")
        self.cells, self.boundary_mode, self.maximum_attempts = (
            cells,
            boundary_mode,
            maximum_attempts,
        )
        self.area = math.pi * diameter_m**2 / 4
        self.dx = length_m / cells
        self.volume = self.area * self.dx
        self.model_id = _digest(self.to_dict())
        self._verified = {}

    def to_dict(self):
        return {
            "model": "single_phase_pr_momentum_v1",
            "fluid": self.fluid.to_dict(),
            **{
                key: getattr(self, key)
                for key in (
                    "length_m",
                    "diameter_m",
                    "cells",
                    "wave_speed_bound_m_s",
                    "maximum_velocity_m_s",
                    "dynamic_viscosity_pa_s",
                    "roughness_m",
                    "boundary_mode",
                    "initial_step_s",
                    "minimum_step_s",
                    "maximum_step_s",
                    "relative_tolerance",
                    "temperature_tolerance_k",
                    "velocity_tolerance_m_s",
                    "maximum_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "single_phase_pr_momentum_v1":
            raise ValueError("Unknown momentum pipe model")
        data["fluid"] = EquilibriumFluid.from_dict(data["fluid"], device=device)
        return cls(**data)

    def _support(self, n):
        f = self.fluid
        n = f.tensor(n)
        if n.shape != (self.cells, len(f.names)) or not bool(
            (torch.isfinite(n).all() & (n >= 0).all() & (n.sum(-1) > 0).all()).cpu()
        ):
            raise ValueError("Invalid momentum pipe component inventory")
        active = torch.nonzero(n[0] > 0).flatten()
        if not bool(((n > 0) == (n[0] > 0)).all().cpu()):
            raise PropertyModelError("Pipe currently requires common positive component support")
        return active

    def _decode(self, y, active):
        f = self.fluid
        c = len(active)
        rows = y.reshape(self.cells, c + 2)
        p = rows[:, 0].exp()
        xa = torch.softmax(torch.cat((rows[:, 1:c], rows.new_zeros((self.cells, 1))), -1), -1)
        x = rows.new_zeros((self.cells, len(f.names))).index_copy(-1, active, xa)
        t = rows[:, c] * f.tref
        v = rows[:, c + 1] * self.maximum_velocity_m_s
        q = torch.vmap(lambda pp, tt, xx: f.tensor_homogeneous(pp, tt, xx, "stable"))(p, t, x)
        concentration = x / q["molar_volume_m3_mol"][:, None]
        momentum = q["density_kg_m3"] * v
        internal = q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
        kinetic = 0.5 * q["density_kg_m3"] * v.square()
        conserved = torch.cat((concentration, momentum[:, None], (internal + kinetic)[:, None]), -1)
        flux = torch.cat(
            (
                concentration * v[:, None],
                (momentum * v + p)[:, None],
                ((internal + kinetic + p) * v)[:, None],
            ),
            -1,
        )
        entropy = q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
        return {
            "p": p,
            "t": t,
            "x": x,
            "v": v,
            "q": q,
            "u": conserved,
            "flux": flux,
            "entropy": entropy,
            "internal": internal,
            "kinetic": kinetic,
        }

    def sound_speed(self, p, t, x):
        f = self.fluid
        pt = f.tensor([p, t])
        x = f.tensor(x)

        def properties(pt):
            q = f.tensor_homogeneous(pt[0], pt[1], x, "stable")
            return torch.stack((q["density_kg_m3"], q["molar_entropy_j_mol_k"]))

        jac = torch.func.jacfwd(properties)(pt)
        denominator = jac[0, 0] - jac[0, 1] * jac[1, 0] / jac[1, 1]
        if not bool((torch.isfinite(jac).all() & (jac[1, 1] > 0) & (denominator > 0)).cpu()):
            raise PropertyModelError("Invalid stable acoustic derivative")
        return float(torch.rsqrt(denominator).cpu())

    def _stable(self, p, t, x):
        q = self.fluid.flash_pt(p, t, x)
        if len(q["phases"]) != 1:
            raise PropertyModelError("Momentum pipe currently requires a stable single phase")
        return self.sound_speed(p, t, x)

    def _validate_data(self, data):
        if not bool(data["q"]["valid"].all().cpu()):
            raise PropertyModelError("Pipe outside homogeneous EOS domain")
        speeds = []
        for p, t, x, v in zip(
            data["p"].cpu().tolist(), data["t"].cpu().tolist(), data["x"], data["v"].cpu().tolist()
        ):
            speed = self._stable(p, t, x)
            if speed + abs(v) > self.wave_speed_bound_m_s * (1 + 1e-12):
                raise PropertyModelError("Rusanov speed bound below physical characteristic speed")
            speeds.append(speed)
        return speeds

    def _bounds(self, c):
        f = self.fluid
        lo = f.tensor(
            [math.log(f.min_pressure_pa)]
            + [-100.0] * (c - 1)
            + [f.min_temperature_k / f.tref, -1.0]
        ).repeat(self.cells)
        hi = f.tensor(
            [math.log(f.max_pressure_pa)] + [100.0] * (c - 1) + [f.max_temperature_k / f.tref, 1.0]
        ).repeat(self.cells)
        return lo, hi

    def inspect(self, state):
        if (
            not isinstance(state, PRMomentumState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Momentum state/model mismatch")
        key = _digest(state.to_dict())
        if key in self._verified:
            return
        f = self.fluid
        active = self._support(state.component_moles)
        y = f.tensor(state.solution_variables)
        if (
            len(y) != self.cells * (len(active) + 2)
            or len(y) > 768
            or len(state.momentum_kg_m_s) != self.cells
            or len(state.total_energy_j) != self.cells
        ):
            raise ValueError("Momentum collocation dimensions differ")
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise ValueError("Restart step outside pipe domain")
        lo, hi = self._bounds(len(active))
        if not bool(((y >= lo) & (y <= hi)).all().cpu()):
            raise PropertyModelError("Pipe primitive outside declared bounds")
        data = self._decode(y, active)
        target = self._conserved(state)
        scale = self._scales(target)
        if not bool(
            (
                torch.isfinite(data["u"]).all()
                & (((data["u"] * self.volume - target).abs() / scale).max() < 1e-9)
            ).cpu()
        ):
            raise ConservationError("Serialized primitive/conserved pipe state differs")
        self._validate_data(data)
        self._remember(state)

    def _remember(self, state):
        self._verified[_digest(state.to_dict())] = True
        while len(self._verified) > 16:
            self._verified.pop(next(iter(self._verified)))

    def _conserved(self, state):
        f = self.fluid
        return torch.cat(
            (
                f.tensor(state.component_moles),
                f.tensor(state.momentum_kg_m_s)[:, None],
                f.tensor(state.total_energy_j)[:, None],
            ),
            -1,
        )

    def _scales(self, stock):
        n = stock[:, : len(self.fluid.names)].sum(-1)
        mass = (stock[:, : len(self.fluid.names)] * self.fluid.molar_masses).sum(-1)
        e = torch.maximum(stock[:, -1].abs(), n * 8.31446261815324 * self.fluid.tref).clamp_min(
            1e-8
        )
        return torch.cat(
            (
                stock[:, : len(self.fluid.names)].abs().clamp_min(n[:, None] * 1e-12),
                (mass * self.maximum_velocity_m_s).clamp_min(1e-12)[:, None],
                e[:, None],
            ),
            -1,
        )

    def initialize(self, pressure_pa, temperature_k, mole_fractions, velocity_m_s=0.0):
        f = self.fluid

        def scalar(values, name):
            if isinstance(values, (tuple, list)):
                if len(values) != self.cells:
                    raise ValueError("Axial array differs from cells")
                return [_number(v, name) for v in values]
            return [_number(values, name)] * self.cells

        pp, tt, vv = (
            scalar(pressure_pa, "pressure"),
            scalar(temperature_k, "temperature"),
            scalar(velocity_m_s, "velocity"),
        )
        xx = mole_fractions
        if not isinstance(xx, (tuple, list)) or not xx:
            raise ValueError("Composition required")
        if not isinstance(xx[0], (tuple, list)):
            xx = [xx] * self.cells
        if len(xx) != self.cells or any(len(row) != len(f.names) for row in xx):
            raise ValueError("Composition registry differs")
        for row in xx:
            for value in row:
                _number(value, "mole fraction", nonnegative=True)
            if abs(math.fsum(row) - 1) > 1e-10:
                raise ValueError("Fractions must sum to one")
        x = f.tensor(xx)
        active = self._support(x)
        xa = x[:, active]
        y = torch.cat(
            (
                f.tensor(pp).log()[:, None],
                (xa[:, :-1] / xa[:, -1:]).log(),
                f.tensor(tt)[:, None] / f.tref,
                f.tensor(vv)[:, None] / self.maximum_velocity_m_s,
            ),
            -1,
        ).flatten()
        if not bool(torch.isfinite(y).all().cpu()):
            raise ValueError("Invalid pipe initial state")
        data = self._decode(y, active)
        u = data["u"] * self.volume
        state = PRMomentumState(
            tuple(map(tuple, u[:, : len(f.names)].cpu().tolist())),
            tuple(u[:, -2].cpu().tolist()),
            tuple(u[:, -1].cpu().tolist()),
            tuple(y.cpu().tolist()),
            0.0,
            self.initial_step_s,
            self.model_id,
        )
        self.inspect(state)
        return state

    def _prepare(self, state, left, right):
        if self.boundary_mode == "periodic":
            if left is not None or right is not None:
                raise ValueError("Periodic pipe does not accept end boundaries")
            return None, None
        active = self._support(state.component_moles)
        prepared = []
        f = self.fluid
        for spec in (left, right):
            if spec is None or spec == "wall":
                prepared.append(None)
                continue
            if (
                not isinstance(spec, PRDirichletBoundary)
                or spec.fluid_id != f.model_id
                or len(spec.mole_fractions) != len(f.names)
            ):
                raise PropertyModelError("Pressure boundary registry/reference differs")
            if any(
                spec.mole_fractions[i] > 0
                for i in range(len(f.names))
                if i not in active.cpu().tolist()
            ):
                raise PropertyModelError("Boundary activates an absent component")
            q = f.tensor_homogeneous(
                f.tensor(spec.pressure_pa),
                f.tensor(spec.temperature_k),
                f.tensor(spec.mole_fractions),
                "stable",
            )
            sound = self._stable(spec.pressure_pa, spec.temperature_k, spec.mole_fractions)
            if not bool(q["valid"].cpu()) or sound > self.wave_speed_bound_m_s:
                raise PropertyModelError("Pressure boundary outside acoustic/EOS domain")
            prepared.append((spec, q, sound))
        return tuple(prepared)

    def _faces(self, data, prepared):
        f = self.fluid
        if self.boundary_mode == "periodic":
            left = torch.cat((data["u"][-1:], data["u"]), 0)
            right = torch.cat((data["u"], data["u"][:1]), 0)
            fl = torch.cat((data["flux"][-1:], data["flux"]), 0)
            fr = torch.cat((data["flux"], data["flux"][:1]), 0)
            sl = torch.cat((data["entropy"][-1:], data["entropy"]), 0)
            sr = torch.cat((data["entropy"], data["entropy"][:1]), 0)
            gl = torch.cat(((data["entropy"] * data["v"])[-1:], data["entropy"] * data["v"]), 0)
            gr = torch.cat((data["entropy"] * data["v"], (data["entropy"] * data["v"])[:1]), 0)
        else:
            ghosts = []
            for index, bound in zip((0, -1), prepared):
                if bound is None:
                    u = data["u"][index]
                    v = -data["v"][index]
                    p = data["p"][index]
                    s = data["entropy"][index]
                    ghost = torch.cat((u[:-2], (-u[-2]).reshape(1), u[-1:]))
                else:
                    spec, q, _sound = bound
                    v = data["v"][index]
                    p = f.tensor(spec.pressure_pa)
                    s = q["molar_entropy_j_mol_k"] / q["molar_volume_m3_mol"]
                    n = f.tensor(spec.mole_fractions) / q["molar_volume_m3_mol"]
                    rho = q["density_kg_m3"]
                    energy = (
                        q["molar_internal_energy_j_mol"] / q["molar_volume_m3_mol"]
                        + 0.5 * rho * v * v
                    )
                    ghost = torch.cat((n, (rho * v).reshape(1), energy.reshape(1)))
                flux = torch.cat(
                    (
                        ghost[:-2] * v,
                        (ghost[-2] * v + p).reshape(1),
                        ((ghost[-1] + p) * v).reshape(1),
                    )
                )
                ghosts.append((ghost, flux, s, s * v))
            a, b = ghosts
            left = torch.cat((a[0][None], data["u"]), 0)
            right = torch.cat((data["u"], b[0][None]), 0)
            fl = torch.cat((a[1][None], data["flux"]), 0)
            fr = torch.cat((data["flux"], b[1][None]), 0)
            sl = torch.cat((a[2].reshape(1), data["entropy"]), 0)
            sr = torch.cat((data["entropy"], b[2].reshape(1)), 0)
            gl = torch.cat((a[3].reshape(1), data["entropy"] * data["v"]), 0)
            gr = torch.cat((data["entropy"] * data["v"], b[3].reshape(1)), 0)
        return (fl + fr) / 2 - self.wave_speed_bound_m_s * (right - left) / 2, (
            gl + gr
        ) / 2 - self.wave_speed_bound_m_s * (sr - sl) / 2

    def _problem(self, state, dt, prepared):
        f = self.fluid
        old = self._conserved(state)
        active = self._support(state.component_moles)
        scale = self._scales(old)
        initial = f.tensor(state.solution_variables)
        lo, hi = self._bounds(len(active))

        def evaluate(y):
            data = self._decode(y, active)
            flux, sflux = self._faces(data, prepared)
            drag = torch.zeros_like(data["v"])
            if self.dynamic_viscosity_pa_s:
                drag = darcy_pressure_loss(
                    data["v"],
                    data["q"]["density_kg_m3"],
                    1.0,
                    self.diameter_m,
                    self.dynamic_viscosity_pa_s,
                    self.roughness_m,
                )[0]
            target = old + dt * self.area * (flux[:-1] - flux[1:])
            source = torch.cat(
                (
                    torch.zeros_like(target[:, :-2]),
                    (-self.volume * drag)[:, None],
                    torch.zeros_like(drag)[:, None],
                ),
                -1,
            )
            target = target + dt * source
            residual = (data["u"] * self.volume - target) / scale
            # Inactive registry entries are identically zero, not extra equations.
            selected = torch.cat((active, active.new_tensor([len(f.names), len(f.names) + 1])))
            return residual[:, selected].flatten(), {
                "data": data,
                "target": target,
                "flux": flux,
                "entropy_flux": sflux,
                "drag": drag,
            }

        return initial, lo, hi, evaluate

    def _check_ledger(self, residual, stock):
        absolute = residual.new_tensor([1e-9] * len(self.fluid.names) + [1e-9, 1e-6])
        allowed = absolute + 1e-9 * self._scales(stock).sum(0)
        if not bool((torch.isfinite(residual).all() & (residual.abs() <= allowed).all()).cpu()):
            raise ConservationError("Pipe component/momentum/energy ledger rejected")

    def _step(self, state, dt, prepared):
        f = self.fluid
        y, lo, hi, evaluate = self._problem(state, dt, prepared)
        if len(y) > 768:
            raise ValueError("Momentum pipe exceeds 768-unknown dense budget")
        solver = TorchNewton(f.device, maximum_iterations=35, dense_limit=768)
        solver.host_convergence = True
        with torch.no_grad():
            y, r, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0], y, lo, hi, tolerance=3e-12
            )
            r, result = evaluate(y)
        if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Conservative PR momentum Newton rejected")
        data = result["data"]
        for index, bound in zip((0, -1), prepared):
            if (
                bound is not None
                and bound[2] + abs(float(data["v"][index].cpu())) > self.wave_speed_bound_m_s
            ):
                raise PropertyModelError("Boundary characteristic speed exceeds Rusanov bound")
        target = result["target"]
        active = self._support(state.component_moles)
        if not bool((target[:, active] > 0).all().cpu()):
            raise HydraulicConvergenceError("Pipe exhausts active component")
        new = PRMomentumState(
            tuple(map(tuple, target[:, : len(f.names)].cpu().tolist())),
            tuple(target[:, -2].cpu().tolist()),
            tuple(target[:, -1].cpu().tolist()),
            tuple(y.cpu().tolist()),
            state.time_s + dt,
            state.next_step_s,
            self.model_id,
        )
        self.inspect(new)
        olddata = self._decode(f.tensor(state.solution_variables), active)
        production = float(
            (
                (data["entropy"] - olddata["entropy"]).sum() * self.volume
                - dt * self.area * (result["entropy_flux"][0] - result["entropy_flux"][-1])
            ).cpu()
        )
        sold = float((olddata["entropy"].sum() * self.volume).cpu())
        if production < -1e-8 - 1e-9 * max(1, abs(sold)):
            raise ConservationError("Pipe numerical entropy balance rejected")
        boundary = dt * self.area * (result["flux"][0] - result["flux"][-1])
        friction = -dt * self.volume * result["drag"].sum()
        delta = (target - self._conserved(state)).sum(0) - boundary
        delta = delta - torch.cat(
            (torch.zeros_like(delta[:-2]), friction.reshape(1), delta.new_zeros(1))
        )
        self._check_ledger(delta, self._conserved(state))
        self._remember(new)
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "left_flux": (self.area * result["flux"][0]).cpu().tolist(),
            "right_flux": (self.area * result["flux"][-1]).cpu().tolist(),
            "friction_impulse_kg_m_s": float(friction.cpu()),
            "conservation_residuals": delta.cpu().tolist(),
            "entropy_production_j_k": production,
            "max_scaled_residual": float(r.abs().max().cpu()),
            "nonlinear_evaluations": evaluations,
        }

    def describe(self, state):
        self.inspect(state)
        data = self._decode(
            self.fluid.tensor(state.solution_variables), self._support(state.component_moles)
        )
        return {
            "pressure_pa": data["p"].cpu().tolist(),
            "temperature_k": data["t"].cpu().tolist(),
            "velocity_m_s": data["v"].cpu().tolist(),
            "mole_fractions": data["x"].cpu().tolist(),
            "density_kg_m3": data["q"]["density_kg_m3"].cpu().tolist(),
            "internal_energy_j": (self.volume * data["internal"]).cpu().tolist(),
            "kinetic_energy_j": (self.volume * data["kinetic"]).cpu().tolist(),
            "entropy_j_k": float((self.volume * data["entropy"].sum()).cpu()),
        }

    def _error(self, a, b):
        f = self.fluid
        active = self._support(a.component_moles)
        x = self._decode(f.tensor(a.solution_variables), active)
        y = self._decode(f.tensor(b.solution_variables), active)
        stock_a, stock_b = self._conserved(a), self._conserved(b)
        scale = self._scales(stock_a)
        return max(
            float(
                ((stock_a - stock_b).abs() / (1e-10 + self.relative_tolerance * scale)).max().cpu()
            ),
            float(((x["t"] - y["t"]).abs() / self.temperature_tolerance_k).max().cpu()),
            float(((x["v"] - y["v"]).abs() / self.velocity_tolerance_m_s).max().cpu()),
            float(((x["p"] - y["p"]).abs() / (1.0 + self.relative_tolerance * x["p"])).max().cpu()),
        )

    def advance(self, state, duration_s, *, left=None, right=None):
        self.inspect(state)
        prepared = self._prepare(state, left, right)
        duration = _number(duration_s, "duration", positive=True)
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable pipe end time")
        current = state
        dt = min(state.next_step_s, duration)
        steps = []
        attempts = 0
        while current.time_s < end:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("Momentum integration budget exhausted")
            dt = min(dt, end - current.time_s)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Substep below floating time resolution")
            try:
                coarse, _ = self._step(current, dt, prepared)
                half, one = self._step(current, dt / 2, prepared)
                fine, two = self._step(half, dt / 2, prepared)
                error = self._error(coarse, fine)
                if error > 1:
                    raise HydraulicConvergenceError("Momentum temporal error exceeds tolerance")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                dt /= 2
                if dt < self.minimum_step_s:
                    raise HydraulicConvergenceError("Momentum minimum step exhausted") from exc
                continue
            steps.extend((one, two))
            nextdt = min(
                self.maximum_step_s, max(self.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            )
            current = PRMomentumState(
                fine.component_moles,
                fine.momentum_kg_m_s,
                fine.total_energy_j,
                fine.solution_variables,
                end if dt >= end - current.time_s else min(fine.time_s, end),
                nextdt,
                self.model_id,
            )
            dt = nextdt
        delta = (self._conserved(current) - self._conserved(state)).sum(0)
        boundary = math.fsum(step["friction_impulse_kg_m_s"] for step in steps)
        totals = self.fluid.tensor(
            [
                math.fsum(s["dt_s"] * (s["left_flux"][i] - s["right_flux"][i]) for s in steps)
                for i in range(len(delta))
            ]
        )
        residual = (
            delta
            - totals
            - torch.cat((torch.zeros_like(delta[:-2]), delta.new_tensor([boundary, 0.0])))
        )
        self._check_ledger(residual, self._conserved(state))
        self._remember(current)
        return {
            "initial_summary": self.describe(state),
            "final_summary": self.describe(current),
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "conservation_residuals": residual.cpu().tolist(),
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
            "left_boundary": left.to_dict() if isinstance(left, PRDirichletBoundary) else left,
            "right_boundary": right.to_dict() if isinstance(right, PRDirichletBoundary) else right,
        }
