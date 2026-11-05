"""Distributed finite wall storage coupled implicitly to a PR Euler pipe.

Wall parameters describe the WHOLE pipe and are divided equally among cells.
Fluid total energy is U+K; stationary-wall drag remains an internal conversion.
No axial wall conduction, gravity or multiphase closure is implied.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .pr_momentum_pipe import PRMomentumPipe, PRMomentumState
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class PRThermalMomentumState:
    pipe: PRMomentumState
    wall_temperature_k: tuple
    model_id: str

    def __post_init__(self):
        object.__setattr__(self, "wall_temperature_k", tuple(self.wall_temperature_k))
        for t in self.wall_temperature_k:
            _number(t, "wall temperature", positive=True)

    @property
    def time_s(self):
        return self.pipe.time_s

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown thermal momentum state fields")
        return cls(**{**data, "pipe": PRMomentumState.from_dict(data["pipe"])})


class PRThermalMomentumPipe:
    def __init__(self, pipe, wall):
        if not isinstance(pipe, PRMomentumPipe) or not isinstance(wall, VesselWall):
            raise TypeError("Expected PRMomentumPipe and VesselWall")
        self.pipe, self.wall = pipe, wall
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "thermal_pr_momentum_v1",
            "pipe": self.pipe.to_dict(),
            "wall": asdict(self.wall),
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if set(data) != {"model", "pipe", "wall"} or data["model"] != "thermal_pr_momentum_v1":
            raise ValueError("Unknown thermal momentum model")
        return cls(
            PRMomentumPipe.from_dict(data["pipe"], device=device), VesselWall(**data["wall"])
        )

    def inspect(self, state):
        if (
            not isinstance(state, PRThermalMomentumState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Thermal momentum model/state mismatch")
        self.pipe.inspect(state.pipe)
        if len(state.wall_temperature_k) != self.pipe.cells or any(
            not self.wall.min_temperature_k <= t <= self.wall.max_temperature_k
            for t in state.wall_temperature_k
        ):
            raise PropertyModelError("Distributed wall temperature outside domain")

    def initialize(self, pipe_state, wall_temperature_k):
        ts = wall_temperature_k
        if not isinstance(ts, (list, tuple)):
            ts = [ts] * self.pipe.cells
        state = PRThermalMomentumState(pipe_state, ts, self.model_id)
        self.inspect(state)
        return state

    def _problem(self, state, dt, prepared, *, face_operator=None):
        p, w = self.pipe, self.wall
        initial, lo, hi, base = p._problem(state.pipe, dt, prepared, face_operator=face_operator)
        capacity = w.heat_capacity_j_k / p.cells
        ua, ambient_ua = w.fluid_ua_w_k / p.cells, w.ambient_ua_w_k / p.cells
        old_wall = p.fluid.tensor(state.wall_temperature_k)
        scale = p._scales(p._conserved(state.pipe))[:, -1]
        width = len(p._support(state.pipe.component_moles)) + 2

        def evaluate(y, context=None):
            residual, data = base(y, context)
            temperature = data["data"]["t"]
            wall = (
                capacity * old_wall + dt * (ua * temperature + ambient_ua * w.ambient_temperature_k)
            ) / (capacity + dt * (ua + ambient_ua))
            heat = dt * ua * (wall - temperature)
            ambient = dt * ambient_ua * (w.ambient_temperature_k - wall)
            # Algebraic wall elimination retains its derivative in the SAME Newton.
            correction = torch.cat(
                (heat.new_zeros((p.cells, width - 1)), (heat / scale)[:, None]), -1
            )
            target = data["target"] + torch.cat(
                (heat.new_zeros((p.cells, len(p.fluid.names) + 1)), heat[:, None]), -1
            )
            correction = torch.cat(
                (correction.flatten(), residual.new_zeros(residual.numel() - correction.numel()))
            )
            return residual - correction, {
                **data,
                "target": target,
                "wall": wall,
                "heat_j": heat,
                "ambient_j": ambient,
            }

        return initial, lo, hi, evaluate

    def _step(self, state, dt, prepared):
        f = self.pipe.fluid
        initial, lo, hi, evaluate = self._problem(state, dt, prepared)
        if len(initial) > 768:
            raise ValueError("Thermal momentum pipe exceeds dense Newton budget")
        solver = TorchNewton(f.device, maximum_iterations=35, dense_limit=768)
        solver.host_convergence = True
        with torch.no_grad():
            y, _, ok, evaluations = solver.solve(
                lambda z: evaluate(z)[0], initial, lo, hi, tolerance=3e-12
            )
            r, result = evaluate(y)
        if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
            raise HydraulicConvergenceError("Thermal momentum Newton rejected")
        return self._accept_step(state, dt, prepared, y, r, result, evaluations)

    def _accept_step(self, state, dt, prepared, y, r, result, evaluations):
        p, w, f = self.pipe, self.wall, self.pipe.fluid
        data, target = result["data"], result["target"]
        active = p._support(state.pipe.component_moles)
        if not bool((torch.isfinite(target).all() & (target[:, active] > 0).all()).cpu()):
            raise HydraulicConvergenceError("Thermal momentum step exhausts active component")
        for index, bound in zip((0, -1), prepared):
            if (
                bound is not None
                and bound[2] + abs(float(data["v"][index].cpu())) > p.wave_speed_bound_m_s
            ):
                raise PropertyModelError("Boundary acoustic bound exceeded")
        wall = tuple(result["wall"].cpu().tolist())
        if (
            max(abs(a - b) for a, b in zip(wall, state.wall_temperature_k))
            > w.max_temperature_change_k
        ):
            raise HydraulicConvergenceError("Wall temperature step exceeds operating limit")
        next_pipe = PRMomentumState(
            tuple(map(tuple, target[:, : len(f.names)].cpu().tolist())),
            tuple(target[:, -2].cpu().tolist()),
            tuple(target[:, -1].cpu().tolist()),
            tuple(y.cpu().tolist()),
            state.time_s + dt,
            state.pipe.next_step_s,
            p.model_id,
        )
        new = PRThermalMomentumState(next_pipe, wall, self.model_id)
        self.inspect(new)
        old = p._decode(
            f.tensor(state.pipe.solution_variables), p._support(state.pipe.component_moles)
        )
        entropy = (
            (data["entropy"] - old["entropy"]).sum() * p.volume
            + w.heat_capacity_j_k
            / p.cells
            * torch.log(result["wall"] / f.tensor(state.wall_temperature_k)).sum()
            - result["ambient_j"].sum() / w.ambient_temperature_k
            - dt * p.area * (result["entropy_flux"][0] - result["entropy_flux"][-1])
        )
        sold = abs(float((old["entropy"].sum() * p.volume).cpu())) + math.fsum(
            abs(w.heat_capacity_j_k / p.cells * math.log(t / w.reference_temperature_k))
            for t in state.wall_temperature_k
        )
        if float(entropy.cpu()) < -1e-8 - 1e-9 * max(1, sold):
            raise ConservationError("Fluid/wall/ambient entropy balance rejected")
        friction = -dt * p.volume * result["drag"].sum()
        external = p._external_impulse(result, dt)
        boundary = dt * p.area * (result["flux"][0] - result["flux"][-1])
        residual = (target - p._conserved(state.pipe)).sum(0) - boundary
        residual = residual - torch.cat(
            (
                residual.new_zeros(len(f.names)),
                (friction + external).reshape(1),
                result["heat_j"].sum().reshape(1),
            )
        )
        p._check_ledger(residual, p._conserved(state.pipe))
        wall_delta = (
            w.heat_capacity_j_k / p.cells * (result["wall"] - f.tensor(state.wall_temperature_k))
        )
        wall_residual = wall_delta + result["heat_j"] - result["ambient_j"]
        if not bool(
            (
                wall_residual.abs()
                <= 1e-7
                + 1e-10 * (wall_delta.abs() + result["heat_j"].abs() + result["ambient_j"].abs())
            )
            .all()
            .cpu()
        ):
            raise ConservationError("Local wall energy ledger rejected")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "left_flux": (p.area * result["flux"][0]).cpu().tolist(),
            "right_flux": (p.area * result["flux"][-1]).cpu().tolist(),
            "fluid_heat_j": result["heat_j"].cpu().tolist(),
            "ambient_heat_j": result["ambient_j"].cpu().tolist(),
            "wall_energy_residual_j": wall_residual.cpu().tolist(),
            "friction_impulse_kg_m_s": float(friction.cpu()),
            "body_force_impulse_kg_m_s": float(external.cpu()),
            "conservation_residuals": residual.cpu().tolist(),
            "entropy_production_j_k": float(entropy.cpu()),
            "max_scaled_residual": float(r.abs().max().cpu()),
            "nonlinear_evaluations": evaluations,
        }

    def advance(self, state, duration_s, *, left=None, right=None):
        self.inspect(state)
        p, w = self.pipe, self.wall
        prepared = p._prepare(state.pipe, left, right)
        end = state.time_s + _number(duration_s, "duration", positive=True)
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Unrepresentable thermal momentum interval")
        current, steps, attempts = state, [], 0
        dt = min(state.pipe.next_step_s, duration_s)
        while current.time_s < end:
            attempts += 1
            if attempts > p.maximum_attempts:
                raise HydraulicConvergenceError("Thermal momentum integration budget exhausted")
            dt = min(dt, end - current.time_s)
            if current.time_s + dt / 2 <= current.time_s:
                raise ValueError("Substep below floating time resolution")
            try:
                coarse, _ = self._step(current, dt, prepared)
                half, first = self._step(current, dt / 2, prepared)
                fine, second = self._step(half, dt / 2, prepared)
                error = max(
                    p._error(coarse.pipe, fine.pipe),
                    max(
                        abs(a - b) / p.temperature_tolerance_k
                        for a, b in zip(coarse.wall_temperature_k, fine.wall_temperature_k)
                    ),
                )
                if error > 1:
                    raise HydraulicConvergenceError("Thermal momentum temporal error exceeded")
            except (HydraulicConvergenceError, ConservationError, PropertyModelError) as exc:
                dt /= 2
                if dt < p.minimum_step_s:
                    raise HydraulicConvergenceError(
                        "Thermal momentum minimum step exhausted"
                    ) from exc
                continue
            nextdt = min(p.maximum_step_s, max(p.minimum_step_s, dt * (1.5 if error < 0.25 else 1)))
            final_time = end if dt >= end - current.time_s else min(fine.time_s, end)
            current = replace(fine, pipe=replace(fine.pipe, time_s=final_time, next_step_s=nextdt))
            steps.extend((first, second))
            dt = nextdt
        oldstock, newstock = p._conserved(state.pipe), p._conserved(current.pipe)
        delta = (newstock - oldstock).sum(0)
        boundary = p.fluid.tensor(
            [
                math.fsum(s["dt_s"] * (s["left_flux"][i] - s["right_flux"][i]) for s in steps)
                for i in range(len(delta))
            ]
        )
        momentum_impulse = math.fsum(
            s["friction_impulse_kg_m_s"] + s.get("body_force_impulse_kg_m_s", 0.0) for s in steps
        )
        heat = math.fsum(q for s in steps for q in s["fluid_heat_j"])
        residual = (
            delta - boundary - p.fluid.tensor([0.0] * len(p.fluid.names) + [momentum_impulse, heat])
        )
        p._check_ledger(residual, oldstock)
        ambient = math.fsum(q for s in steps for q in s["ambient_heat_j"])
        wall_delta = (
            w.heat_capacity_j_k
            / p.cells
            * math.fsum(b - a for a, b in zip(state.wall_temperature_k, current.wall_temperature_k))
        )
        energy_residual = float((delta[-1] - boundary[-1]).cpu()) + wall_delta - ambient
        if abs(energy_residual) > 1e-6 + 1e-9 * max(1, abs(wall_delta), abs(ambient)):
            raise ConservationError("Combined pipe/wall interval energy balance rejected")
        return {
            "state": current.to_dict(),
            "steps": steps,
            "attempts": attempts,
            "conservation_residuals": residual.cpu().tolist(),
            "combined_energy_residual_j": energy_residual,
            "ambient_heat_j": ambient,
            "entropy_production_j_k": math.fsum(s["entropy_production_j_k"] for s in steps),
        }
