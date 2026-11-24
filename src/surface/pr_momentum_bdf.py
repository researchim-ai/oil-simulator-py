"""Opt-in conservative variable-step BDF2 for a horizontal single-phase PR pipe.

One nonlinear solve per BDF trial; BE step doubling is used only to start or
restart history. The estimator uses a quadratic history predictor, not a second
integration. Fixed phase support, EOS validity, acoustic bounds and inventory
ledgers are mandatory. This does not qualify multiphase, thermal-wall or gravity
extensions, nor claim BE's per-step discrete entropy inequality for BDF2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .cuda_execution import declare_runtime_residual
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_momentum_pipe import PRMomentumPipe, PRMomentumState
from .torch_nonlinear import TorchNewton


@dataclass(frozen=True)
class PRBDFPoint:
    state: PRMomentumState
    left_exchange: tuple
    right_exchange: tuple
    friction_impulse: float
    entropy_exchange: float
    order: int

    def __post_init__(self):
        for key in ("left_exchange", "right_exchange"):
            object.__setattr__(self, key, tuple(_number(v, key) for v in getattr(self, key)))
        _number(self.friction_impulse, "BDF friction impulse")
        _number(self.entropy_exchange, "BDF entropy exchange")
        if type(self.order) is not int or self.order not in (0, 1, 2):
            raise ValueError("Unknown BDF history order")

    def to_dict(self):
        return {
            "state": self.state.to_dict(),
            "left_exchange": self.left_exchange,
            "right_exchange": self.right_exchange,
            "friction_impulse": self.friction_impulse,
            "entropy_exchange": self.entropy_exchange,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(**{**data, "state": PRMomentumState.from_dict(data["state"])})


@dataclass(frozen=True)
class PRMomentumBDFState:
    history: tuple
    boundary_id: str | None
    integrator_id: str

    def __post_init__(self):
        object.__setattr__(self, "history", tuple(self.history))
        if len(self.history) not in (1, 3) or any(
            not isinstance(p, PRBDFPoint) for p in self.history
        ):
            raise ValueError("BDF checkpoint needs one to three history points")

    @property
    def current(self):
        return self.history[-1].state

    def to_dict(self):
        return {
            "history": [p.to_dict() for p in self.history],
            "boundary_id": self.boundary_id,
            "integrator_id": self.integrator_id,
        }

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"history", "boundary_id", "integrator_id"}:
            raise ValueError("Unknown BDF checkpoint fields")
        return cls(
            tuple(PRBDFPoint.from_dict(p) for p in data["history"]),
            data["boundary_id"],
            data["integrator_id"],
        )


class PRMomentumBDF:
    def __init__(self, pipe):
        # Subclasses may add gravity, reconstruction unknowns or thermal stores.
        # Their continuous operators/ledgers must be adapted explicitly first.
        if type(pipe) is not PRMomentumPipe:
            raise TypeError("BDF adapter currently supports the base horizontal PRMomentumPipe")
        self.pipe = pipe
        self.integrator_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "single_phase_pr_momentum_bdf2_v1",
            "pipe": self.pipe.to_dict(),
            "maximum_step_ratio": 1.5,
            "error_safety": 0.85,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if (
            set(data) != {"model", "pipe", "maximum_step_ratio", "error_safety"}
            or data["model"] != "single_phase_pr_momentum_bdf2_v1"
            or data["maximum_step_ratio"] != 1.5
            or data["error_safety"] != 0.85
        ):
            raise ValueError("Unknown BDF model/policy")
        return cls(PRMomentumPipe.from_dict(data["pipe"], device=device))

    def _first(self, state):
        zero = (0.0,) * (len(self.pipe.fluid.names) + 2)
        return PRBDFPoint(state, zero, zero, 0.0, 0.0, 0)

    def initialize(self, *args, **kwargs):
        return PRMomentumBDFState(
            (self._first(self.pipe.initialize(*args, **kwargs)),), None, self.integrator_id
        )

    def inspect(self, checkpoint):
        p = self.pipe
        if (
            not isinstance(checkpoint, PRMomentumBDFState)
            or checkpoint.integrator_id != self.integrator_id
            or _digest(self.to_dict()) != self.integrator_id
        ):
            raise PropertyModelError("BDF checkpoint/model mismatch")
        nc = len(p.fluid.names) + 2
        if checkpoint.boundary_id is None and len(checkpoint.history) != 1:
            raise ValueError("Unbound BDF checkpoint cannot contain a trajectory")
        for point in checkpoint.history:
            p.inspect(point.state)
            if len(point.left_exchange) != nc or len(point.right_exchange) != nc:
                raise ValueError("BDF exchange registry differs")
            if [v > 0 for v in point.state.component_moles[0]] != [
                v > 0 for v in checkpoint.current.component_moles[0]
            ]:
                raise ValueError("BDF history active supports differ")
        for old, new in zip(checkpoint.history, checkpoint.history[1:]):
            if new.state.time_s <= old.state.time_s or new.order == 0:
                raise ValueError("BDF history clocks/orders differ")
            delta = (p._conserved(new.state) - p._conserved(old.state)).sum(0)
            delta -= p.fluid.tensor(new.left_exchange) - p.fluid.tensor(new.right_exchange)
            delta[-2] -= new.friction_impulse
            p._check_ledger(delta, p._conserved(old.state))

    def _check_history_equation(self, points, prepared):
        if len(points) < 3:
            return
        p, f = self.pipe, self.pipe.fluid
        a, b, c = points
        h = c.state.time_s - b.state.time_s
        hp = b.state.time_s - a.state.time_s
        if c.order == 2:
            r = h / hp
            if r > 1.5 * (1 + 1e-10):
                raise ValueError("Checkpoint BDF step ratio rejected")
            beta, gamma = r * r / (1 + 2 * r), (1 + r) / (1 + 2 * r)
        else:
            beta, gamma = 0.0, 1.0
        _, _, _, continuous = p.continuous_problem(c.state, prepared)
        _, data = continuous(f.tensor(c.state.solution_variables))
        old = p._conserved(b.state)
        expected = old + beta * (old - p._conserved(a.state)) + gamma * h * data["rate"]
        error = (p._conserved(c.state) - expected) / p._scales(old)
        if not bool((torch.isfinite(error).all() & (error.abs().max() < 1e-9)).cpu()):
            raise ConservationError("BDF checkpoint does not satisfy its time equation")
        allowed = f.tensor([1e-9] * len(f.names) + [1e-9, 1e-6]) + 1e-9 * p._scales(old).sum(0)
        for key, index in [("left_exchange", 0), ("right_exchange", -1)]:
            reference = beta * f.tensor(getattr(b, key)) + gamma * h * p.area * data["flux"][index]
            if not bool(((f.tensor(getattr(c, key)) - reference).abs() <= allowed).all().cpu()):
                raise ConservationError("BDF checkpoint port quadrature differs")

    def _packet(self, state):
        p = self.pipe
        data = p._decode(
            p.fluid.tensor(state.solution_variables), p._support(state.component_moles)
        )
        return {"stock": p._conserved(state), "p": data["p"], "t": data["t"], "v": data["v"]}

    def _error(self, points, new, h, packet):
        a, b, c = points
        hp, hpp = c.state.time_s - b.state.time_s, b.state.time_s - a.state.time_s
        r = h / hp
        a0 = (1 + 2 * r) / (1 + r)
        # BDF2 LTE coefficient h^2(h+hp)/(6*a0) times a third derivative.
        # A quadratic extrapolation defect estimates the third divided
        # difference. This deliberately uses the uncorrected (larger) factor.
        coefficient = h / (a0 * (h + hp + hpp))
        pa, pb, pc, pn = (packet(s) for s in (a.state, b.state, c.state, new))
        errors = {}
        for key in pc:
            first = (pc[key] - pb[key]) / hp
            second = (first - (pb[key] - pa[key]) / hpp) / (hp + hpp)
            prediction = pc[key] + h * first + h * (h + hp) * second
            errors[key] = coefficient * (pn[key] - prediction).abs()
        p = self.pipe
        return max(
            float(
                (errors["stock"] / (1e-10 + p.relative_tolerance * p._scales(pn["stock"])))
                .max()
                .cpu()
            ),
            float((errors["p"] / (1.0 + p.relative_tolerance * pn["p"])).max().cpu()),
            float((errors["t"] / p.temperature_tolerance_k).max().cpu()),
            float((errors["v"] / p.velocity_tolerance_m_s).max().cpu()),
        )

    def _be_point(self, state, row):
        p = self.pipe
        h = row["dt_s"]
        # BE reports entropy production separately from boundary exchange.
        old = row.pop("_old")
        delta_s = p.describe(state)["entropy_j_k"] - p.describe(old)["entropy_j_k"]
        return PRBDFPoint(
            state,
            tuple(h * v for v in row["left_flux"]),
            tuple(h * v for v in row["right_flux"]),
            row["friction_impulse_kg_m_s"],
            delta_s - row["entropy_production_j_k"],
            1,
        )

    def _bdf_step(self, points, h, prepared):
        p, f = self.pipe, self.pipe.fluid
        previous, current = points[-2:]
        hp = current.state.time_s - previous.state.time_s
        r = h / hp
        if r > 1.5 * (1 + 1e-12):
            raise ValueError("BDF history step ratio exceeds the supported bound")
        beta, gamma = r * r / (1 + 2 * r), (1 + r) / (1 + 2 * r)
        old, older = p._conserved(current.state), p._conserved(previous.state)
        base = old + beta * (old - older)
        scale = p._scales(old)
        interval_factor = f.tensor(gamma * h)
        active = p._support(current.state.component_moles)
        nc = len(f.names)
        selected = torch.cat((active, active.new_tensor([nc, nc + 1])))
        initial, lo, hi, continuous = p.continuous_problem(current.state, prepared)
        if initial.numel() > 768:
            raise ValueError("BDF momentum pipe exceeds the existing 768-unknown budget")
        initial = initial + r * (initial - f.tensor(previous.state.solution_variables))

        def evaluate(y):
            _, result = continuous(y)
            target = base + interval_factor * result["rate"]
            return ((result["inventory"] - target) / scale)[:, selected].flatten(), result, target

        solver = TorchNewton(f.device, maximum_iterations=35, dense_limit=768)
        solver.host_convergence = True

        def residual(y):
            return evaluate(y)[0]

        declare_runtime_residual(residual)
        with torch.no_grad():
            y, row, ok, evaluations = solver.solve(residual, initial, lo, hi, tolerance=3e-12)
            row, result, target = evaluate(y)
        if not bool((ok & torch.isfinite(row).all() & (row.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("BDF momentum Newton rejected")
        if not bool((target[:, active] > 0).all().cpu()):
            raise HydraulicConvergenceError("BDF exhausts an active component")
        new = PRMomentumState(
            tuple(map(tuple, target[:, :nc].cpu().tolist())),
            tuple(target[:, -2].cpu().tolist()),
            tuple(target[:, -1].cpu().tolist()),
            tuple(y.cpu().tolist()),
            current.state.time_s + h,
            current.state.next_step_s,
            p.model_id,
        )
        p.inspect(new)
        for index, bound in zip((0, -1), prepared):
            if (
                bound is not None
                and bound[2] + abs(float(result["data"]["v"][index].cpu())) > p.wave_speed_bound_m_s
            ):
                raise PropertyModelError("BDF boundary characteristic exceeds acoustic bound")
        left = beta * f.tensor(current.left_exchange) + gamma * h * p.area * result["flux"][0]
        right = beta * f.tensor(current.right_exchange) + gamma * h * p.area * result["flux"][-1]
        friction = beta * current.friction_impulse - gamma * h * p.volume * float(
            result["drag"].sum().cpu()
        )
        entropy = beta * current.entropy_exchange + gamma * h * p.area * float(
            (result["entropy_flux"][0] - result["entropy_flux"][-1]).cpu()
        )
        point = PRBDFPoint(
            new, tuple(left.cpu().tolist()), tuple(right.cpu().tolist()), friction, entropy, 2
        )
        residual = (target - old).sum(0) - left + right
        residual[-2] -= friction
        p._check_ledger(residual, old)
        return point, {
            "nonlinear_evaluations": evaluations,
            "jacobian_builds": solver.jacobian_builds,
            "max_scaled_residual": float(row.abs().max().cpu()),
        }

    def advance(self, checkpoint, duration_s, *, left=None, right=None):
        self.inspect(checkpoint)
        p, f = self.pipe, self.pipe.fluid
        start = checkpoint.current
        duration = _number(duration_s, "BDF interval", positive=True)
        end = start.time_s + duration
        if not math.isfinite(end) or end <= start.time_s:
            raise ValueError("Unrepresentable BDF end time")
        prepared = p._prepare(start, left, right)
        boundary_id = _digest(
            {
                "left": left.to_dict() if hasattr(left, "to_dict") else left,
                "right": right.to_dict() if hasattr(right, "to_dict") else right,
            }
        )
        reset = checkpoint.boundary_id != boundary_id
        points = list(checkpoint.history) if not reset else [self._first(start)]
        if not reset:
            self._check_history_equation(points, prepared)
        h = min(start.next_step_s, duration)
        if len(points) > 1:
            h = min(h, 1.5 * (points[-1].state.time_s - points[-2].state.time_s))
        accepted, attempts, rejected = [], 0, 0
        cache = {}

        def packet(state):
            key = id(state)
            if key not in cache:
                cache[key] = self._packet(state)
            return cache[key]

        while points[-1].state.time_s < end:
            attempts += 1
            if attempts > p.maximum_attempts:
                raise HydraulicConvergenceError("BDF integration budget exhausted")
            old = points[-1].state
            h = min(h, end - old.time_s)
            if old.time_s + h / 2 <= old.time_s:
                raise ValueError("BDF substep below floating time resolution")
            try:
                if len(points) < 3:
                    coarse, coarse_row = p._step(old, h, prepared)
                    half, one = p._step(old, h / 2, prepared)
                    fine, two = p._step(half, h / 2, prepared)
                    fine = replace(fine, time_s=old.time_s + h)
                    error = p._error(coarse, fine)
                    one["_old"], two["_old"] = old, half
                    trial = [self._be_point(half, one), self._be_point(fine, two)]
                    work = {
                        "nonlinear_solves": 3,
                        "nonlinear_evaluations": coarse_row["nonlinear_evaluations"]
                        + one["nonlinear_evaluations"]
                        + two["nonlinear_evaluations"],
                    }
                else:
                    point, work = self._bdf_step(points, h, prepared)
                    work["nonlinear_solves"] = 1
                    error = self._error(points, point.state, h, packet)
                    trial = [point]
                if not math.isfinite(error) or error > 1:
                    raise HydraulicConvergenceError("BDF temporal error exceeds tolerance")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                rejected += 1
                cache.clear()
                h *= 0.5
                if h < p.minimum_step_s:
                    raise HydraulicConvergenceError("BDF minimum step exhausted") from exc
                continue
            # Publish history only after all endpoint, temporal and ledger checks.
            for point in trial:
                accepted.append(
                    {
                        "dt_s": point.state.time_s - points[-1].state.time_s,
                        "point": point.to_dict(),
                        "temporal_error": error,
                        "work": work if point is trial[-1] else {},
                    }
                )
                points.append(point)
            points = points[-3:]
            last_h = points[-1].state.time_s - points[-2].state.time_s
            factor = min(1.5, max(0.5, 0.85 / max(error, 1e-12) ** (1 / 3)))
            h = min(p.maximum_step_s, max(p.minimum_step_s, last_h * factor))
            state = replace(points[-1].state, next_step_s=h)
            points[-1] = replace(points[-1], state=state)
            # No rejected-point or object-id cache entry survives publication.
            cache.clear()
        final = PRMomentumBDFState(tuple(points), boundary_id, self.integrator_id)
        self.inspect(final)
        left_total = f.tensor(
            [
                math.fsum(s["point"]["left_exchange"][i] for s in accepted)
                for i in range(len(f.names) + 2)
            ]
        )
        right_total = f.tensor(
            [
                math.fsum(s["point"]["right_exchange"][i] for s in accepted)
                for i in range(len(f.names) + 2)
            ]
        )
        friction = math.fsum(s["point"]["friction_impulse"] for s in accepted)
        residual = (
            (p._conserved(final.current) - p._conserved(start)).sum(0) - left_total + right_total
        )
        residual[-2] -= friction
        p._check_ledger(residual, p._conserved(start))
        entropy = (
            p.describe(final.current)["entropy_j_k"]
            - p.describe(start)["entropy_j_k"]
            - math.fsum(s["point"]["entropy_exchange"] for s in accepted)
        )
        return {
            "initial_state": checkpoint.to_dict(),
            "model": self.to_dict(),
            "requested_duration_s": duration,
            "left_boundary": left.to_dict() if hasattr(left, "to_dict") else left,
            "right_boundary": right.to_dict() if hasattr(right, "to_dict") else right,
            "state": final.to_dict(),
            "steps": accepted,
            "attempts": attempts,
            "rejected_steps": rejected,
            "history_reset": reset,
            "final_summary": p.describe(final.current),
            "conservation_residuals": residual.cpu().tolist(),
            "entropy_balance_diagnostic_j_k": entropy,
            "entropy_policy": "Diagnostic only; BE per-step inequality is not asserted for BDF2",
        }
