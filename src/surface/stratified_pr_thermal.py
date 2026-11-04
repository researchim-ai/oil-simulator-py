"""Quasi-static thermal evolution of a closed, rigid gravitational PR vessel.

The fluid profile and wall exchange are implicit in the same Torch residual.
There is no material flow, phase-transfer kinetics or fluid momentum equation.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .stratified_pr import StratifiedPRColumn, StratifiedPRState
from .torch_linear import TorchNewton
from .vessel_wall import VesselWall


@dataclass(frozen=True)
class StratifiedThermalState:
    column: StratifiedPRState
    wall_temperature_k: float
    time_s: float
    next_step_s: float
    model_id: str

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unknown thermal state fields")
        return cls(**{**data, "column": StratifiedPRState.from_dict(data["column"])})


class StratifiedPRThermal:
    """Closed-vessel BE integration with step doubling and atomic publication."""

    def __init__(
        self,
        column,
        wall,
        *,
        initial_step_s=1.0,
        minimum_step_s=1e-5,
        maximum_step_s=60.0,
        temperature_tolerance_k=0.002,
        relative_tolerance=1e-5,
        maximum_attempts=1000,
    ):
        if not isinstance(column, StratifiedPRColumn) or not isinstance(wall, VesselWall):
            raise TypeError("Expected StratifiedPRColumn and VesselWall")
        self.column, self.wall = column, wall
        for key, value in locals().copy().items():
            if key in (
                "initial_step_s",
                "minimum_step_s",
                "maximum_step_s",
                "temperature_tolerance_k",
                "relative_tolerance",
            ):
                setattr(self, key, _number(value, key, positive=True))
        if not minimum_step_s <= initial_step_s <= maximum_step_s:
            raise ValueError("Invalid thermal step interval")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1
        ):
            raise ValueError("Positive integer attempt budget required")
        self.maximum_attempts = maximum_attempts
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "closed_stratified_pr_thermal_v1",
            "column": self.column.to_dict(),
            "wall": asdict(self.wall),
            **{
                k: getattr(self, k)
                for k in (
                    "initial_step_s",
                    "minimum_step_s",
                    "maximum_step_s",
                    "temperature_tolerance_k",
                    "relative_tolerance",
                    "maximum_attempts",
                )
            },
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        data = dict(data)
        if data.pop("model", None) != "closed_stratified_pr_thermal_v1":
            raise ValueError("Unknown thermal model")
        data["column"] = StratifiedPRColumn.from_dict(data["column"], device=device)
        data["wall"] = VesselWall(**data["wall"])
        return cls(**data)

    def inspect(self, state):
        if (
            not isinstance(state, StratifiedThermalState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Thermal state/model mismatch")
        self.column.inspect(state.column)
        _number(state.time_s, "time", nonnegative=True)
        _number(state.next_step_s, "next step", positive=True)
        _number(state.wall_temperature_k, "wall temperature", positive=True)
        if (
            not self.wall.min_temperature_k
            <= state.wall_temperature_k
            <= self.wall.max_temperature_k
        ):
            raise PropertyModelError("Wall temperature outside domain")
        if not self.minimum_step_s <= state.next_step_s <= self.maximum_step_s:
            raise PropertyModelError("Restart step outside domain")

    def initialize(self, column_state, wall_temperature_k):
        state = StratifiedThermalState(
            column_state, wall_temperature_k, 0.0, self.initial_step_s, self.model_id
        )
        self.inspect(state)
        return state

    def _data(self, state):
        c, f = self.column, self.column.fluid
        seed = (
            [f.tensor(x) for x in state.phase_seeds],
            list(state.phase_roots),
            f.tensor(state.interface_heights_m),
            math.exp(state.solution_variables[0]),
            state.temperature_k,
            f.tensor(state.chemical_constants),
        )
        return c._problem(
            f.tensor(state.component_moles), state.total_energy_j, seed, state.quadrature_order
        )[3](f.tensor(state.solution_variables))[1]

    def _step(self, state, dt):
        c, f, w = self.column, self.column.fluid, self.wall
        n = f.tensor(state.column.component_moles)
        old = state.column

        def exchanges(t):
            tw = (
                w.heat_capacity_j_k * state.wall_temperature_k
                + dt * (w.fluid_ua_w_k * t + w.ambient_ua_w_k * w.ambient_temperature_k)
            ) / (w.heat_capacity_j_k + dt * (w.fluid_ua_w_k + w.ambient_ua_w_k))
            return tw, w.fluid_ua_w_k * (tw - t), w.ambient_ua_w_k * (w.ambient_temperature_k - tw)

        # UV predictor may change phase count. It is only a seed; no inventory is changed.
        predicted = old.total_energy_j + dt * exchanges(old.temperature_k)[1]
        seed = c._seed(n, predicted, math.exp(old.solution_variables[0]), old.temperature_k)
        previous = None
        order = c.quadrature_order
        while order <= c.maximum_quadrature_order:
            initial, lower, upper, evaluate, active, roots = c._problem(
                n, old.total_energy_j, seed, order
            )
            mass = float((n * f.molar_masses).sum().cpu())
            escale = max(
                1.0,
                abs(old.total_energy_j - mass * c.gravity_m_s2 * c.elevation_m),
                float(n.sum().cpu()) * R * seed[4],
            )

            def residual(y, evaluate=evaluate, escale=escale):
                r, data = evaluate(y)
                exchange = exchanges(data["temperature"])[1]
                return torch.cat((r[:-1], (r[-1] - dt * exchange / escale).reshape(1)))

            solver = TorchNewton(f.device, maximum_iterations=40, dense_limit=768)
            solver.host_convergence = True
            with torch.no_grad():
                y, r, ok, _ = solver.solve(residual, initial, lower, upper, tolerance=3e-12)
                r = residual(y)
                _, data = evaluate(y)
            if not bool((ok & torch.isfinite(r).all() & (r.abs().max() < 1e-9)).cpu()):
                raise HydraulicConvergenceError("Implicit thermal column Newton rejected")
            t = float(data["temperature"].cpu())
            tw, q, ambient = exchanges(t)
            if (
                not w.min_temperature_k <= tw <= w.max_temperature_k
                or abs(tw - state.wall_temperature_k) > w.max_temperature_change_k
            ):
                raise HydraulicConvergenceError("Wall step/domain limit exceeded")
            constants = f.tensor([0.0] * len(f.names)).index_copy(0, active, data["constants"])
            solved_seed = (
                [x[0] for x in data["compositions"][: len(roots)]],
                roots,
                data["interfaces"],
                float(data["pressures"][0][0].cpu()),
                t,
                constants,
            )
            energy = old.total_energy_j + dt * q
            # Independent fixed-N/E residual and local stability checks before acceptance.
            candidate, report = c._solve(n, energy, solved_seed, order)
            if abs(candidate.temperature_k - t) > 1e-6:
                raise ConservationError("Thermal/equilibrium temperatures differ")
            if previous is not None:
                prior, prior_report = previous
                if len(prior.phase_roots) != len(candidate.phase_roots):
                    raise HydraulicConvergenceError("Quadrature changed phase topology")
                errors = [
                    abs(candidate.temperature_k - prior.temperature_k)
                    / (c.temperature_tolerance_k + c.relative_tolerance * t),
                    abs(report["mean_pressure_pa"] - prior_report["mean_pressure_pa"])
                    / (1 + c.relative_tolerance * report["mean_pressure_pa"]),
                    abs(
                        report["relative_potential_energy_j"]
                        - prior_report["relative_potential_energy_j"]
                    )
                    / (
                        1e-5
                        + c.relative_tolerance * max(1, abs(report["relative_potential_energy_j"]))
                    ),
                ]
                errors += [
                    abs(a - b) / (1e-8 + c.relative_tolerance * c.geometry.height_m)
                    for a, b in zip(candidate.interface_heights_m, prior.interface_heights_m)
                ]
                errors.append(
                    float(
                        (
                            (
                                f.tensor(report["phase_component_moles"])
                                - f.tensor(prior_report["phase_component_moles"])
                            ).abs()
                            / (1e-9 + c.relative_tolerance * n.sum())
                        )
                        .max()
                        .cpu()
                    )
                )
                if max(errors) <= 1:
                    entropy_old = float(self._data(old)["entropy_j_k"].cpu())
                    production = (
                        report["entropy_j_k"]
                        - entropy_old
                        + w.heat_capacity_j_k * math.log(tw / state.wall_temperature_k)
                        - dt * ambient / w.ambient_temperature_k
                    )
                    tolerance = 1e-6 + 1e-9 * max(1, abs(entropy_old), abs(report["entropy_j_k"]))
                    if production < -tolerance:
                        raise ConservationError("Thermal step lowered total entropy")
                    total_delta = (
                        candidate.total_energy_j
                        - old.total_energy_j
                        + w.heat_capacity_j_k * (tw - state.wall_temperature_k)
                    )
                    imbalance = total_delta - dt * ambient
                    if abs(imbalance) > 1e-5 + 1e-9 * max(1, abs(dt * ambient), abs(dt * q)):
                        raise ConservationError("Fluid/wall energy ledger rejected")
                    return StratifiedThermalState(
                        candidate, tw, state.time_s + dt, state.next_step_s, self.model_id
                    ), {
                        "dt_s": dt,
                        "ambient_heat_j": dt * ambient,
                        "fluid_heat_j": dt * q,
                        "energy_residual_j": imbalance,
                        "entropy_production_j_k": production,
                        "quadrature_error_ratio": max(errors),
                        "phase_count": len(roots),
                    }
            previous = candidate, report
            seed = solved_seed
            order *= 2
        raise HydraulicConvergenceError("Thermal quadrature budget exhausted")

    def advance(self, state, duration_s):
        self.inspect(state)
        duration = _number(duration_s, "duration", positive=True)
        target = state.time_s + duration
        if not math.isfinite(target) or target <= state.time_s:
            raise ValueError("Unrepresentable end time")
        current, history = state, []
        dt = min(state.next_step_s, duration)
        attempts = 0
        while current.time_s < target:
            attempts += 1
            if attempts > self.maximum_attempts:
                raise HydraulicConvergenceError("Thermal integration attempt budget exhausted")
            dt = min(dt, target - current.time_s)
            try:
                coarse, _ = self._step(current, dt)
                half, first = self._step(current, dt / 2)
                fine, second = self._step(half, dt / 2)
                error = (
                    max(
                        abs(coarse.column.temperature_k - fine.column.temperature_k),
                        abs(coarse.wall_temperature_k - fine.wall_temperature_k),
                    )
                    / self.temperature_tolerance_k
                )
                coarse_n = self.column.fluid.tensor(coarse.column.component_moles)
                fine_n = self.column.fluid.tensor(fine.column.component_moles)
                error = max(
                    error,
                    float(
                        (
                            (coarse_n - fine_n).abs()
                            / (1e-9 + self.relative_tolerance * fine_n.sum())
                        )
                        .max()
                        .cpu()
                    ),
                )
                energy_scale = max(
                    1.0,
                    abs(
                        fine.column.total_energy_j
                        - float(
                            (
                                self.column.fluid.tensor(fine.column.component_moles)
                                * self.column.fluid.molar_masses
                            )
                            .sum()
                            .cpu()
                        )
                        * self.column.gravity_m_s2
                        * self.column.elevation_m
                    ),
                )
                error = max(
                    error,
                    abs(coarse.column.total_energy_j - fine.column.total_energy_j)
                    / (1e-5 + self.relative_tolerance * energy_scale),
                )
                error = max(
                    error,
                    *(
                        [
                            abs(a - b)
                            / (1e-8 + self.relative_tolerance * self.column.geometry.height_m)
                            for a, b in zip(
                                coarse.column.interface_heights_m, fine.column.interface_heights_m
                            )
                        ]
                        or [0.0]
                    ),
                )
                coarse_data, fine_data = self._data(coarse.column), self._data(fine.column)

                def mean_pressure(data):
                    return (
                        float(
                            torch.stack(
                                [(a * b).sum() for a, b in zip(data["weights"], data["pressures"])]
                            )
                            .sum()
                            .cpu()
                        )
                        / self.column.geometry.volume_m3
                    )

                pc, pf = mean_pressure(coarse_data), mean_pressure(fine_data)
                error = max(error, abs(pc - pf) / (1.0 + self.relative_tolerance * max(pc, pf)))
                if len(coarse.column.phase_roots) == len(fine.column.phase_roots):
                    amounts = (
                        coarse_data["phase_component_moles"] - fine_data["phase_component_moles"]
                    ).abs()
                    error = max(
                        error,
                        float(
                            (
                                amounts
                                / (
                                    1e-9
                                    + self.relative_tolerance * fine_data["component_moles"].sum()
                                )
                            )
                            .max()
                            .cpu()
                        ),
                    )
                if len(coarse.column.phase_roots) != len(fine.column.phase_roots):
                    error = max(error, 2.0)
                if error > 1:
                    raise HydraulicConvergenceError("Thermal temporal error above tolerance")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError) as exc:
                dt *= 0.5
                if dt < self.minimum_step_s:
                    raise HydraulicConvergenceError("Thermal minimum step exhausted") from exc
                continue
            history.extend((first, second))
            next_dt = min(
                self.maximum_step_s, max(self.minimum_step_s, dt * (1.5 if error < 0.25 else 1.0))
            )
            current = StratifiedThermalState(
                fine.column,
                fine.wall_temperature_k,
                target if dt >= target - current.time_s else min(fine.time_s, target),
                next_dt,
                self.model_id,
            )
            dt = next_dt
        ambient = math.fsum(h["ambient_heat_j"] for h in history)
        feed_energy = math.fsum(h.get("feed_energy_j", 0.0) for h in history)
        outlet_energy = math.fsum(h.get("outlet_energy_j", 0.0) for h in history)
        energy_error = (
            current.column.total_energy_j
            - state.column.total_energy_j
            + self.wall.heat_capacity_j_k * (current.wall_temperature_k - state.wall_temperature_k)
            - ambient
            - feed_energy
            + outlet_energy
        )
        if abs(energy_error) > 1e-5 + 1e-9 * max(
            1.0,
            abs(ambient),
            abs(feed_energy),
            abs(outlet_energy),
            abs(current.column.total_energy_j - state.column.total_energy_j),
        ):
            raise ConservationError("Interval energy ledger rejected")
        component_error = [
            b
            - a
            - math.fsum(
                h.get("feed_component_moles", [0.0] * len(state.column.component_moles))[i]
                - h.get("outlet_component_moles", [0.0] * len(state.column.component_moles))[i]
                for h in history
            )
            for i, (a, b) in enumerate(
                zip(state.column.component_moles, current.column.component_moles)
            )
        ]
        if max(map(abs, component_error)) > 1e-8 + 1e-9 * sum(current.column.component_moles):
            raise ConservationError("Interval component ledger rejected")
        return {
            "component_residuals_mol": component_error,
            "feed_energy_j": feed_energy,
            "outlet_energy_j": outlet_energy,
            "state": current.to_dict(),
            "steps": history,
            "attempts": attempts,
            "ambient_heat_j": ambient,
            "energy_residual_j": energy_error,
            "entropy_production_j_k": math.fsum(h["entropy_production_j_k"] for h in history),
        }
