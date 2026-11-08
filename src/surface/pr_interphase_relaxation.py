"""Finite-inertia, pressure and thermal relaxation of closed PR phase stocks.

An embeddable local source block for a nonequilibrium two/three-fluid model.
No spatial fluxes, chemical mass transfer, phase birth/death or regime map.
All per-phase component stocks remain fixed in this subproblem.
"""

import math
from dataclasses import asdict, dataclass, replace

import torch

from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .phase_momentum_exchange import PhaseMomentumExchange, symmetric_coefficients
from .time_interval import interval_step_s, split_step_s
from .torch_linear import TorchNewton


@dataclass(frozen=True)
class PRRelaxationPhase:
    component_moles: tuple
    volume_m3: float
    pressure_pa: float
    temperature_k: float
    momentum_kg_m_s: float
    total_energy_j: float

    def __post_init__(self):
        object.__setattr__(self, "component_moles", tuple(self.component_moles))
        if not self.component_moles or not any(self.component_moles):
            raise ValueError("Finite nonempty phase required")
        for v in self.component_moles:
            _number(v, "phase component stock", nonnegative=True)
        for name in ("volume_m3", "pressure_pa", "temperature_k"):
            _number(getattr(self, name), name, positive=True)
        for name in ("momentum_kg_m_s", "total_energy_j"):
            _number(getattr(self, name), name)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class PRRelaxationState:
    phases: tuple
    time_s: float
    model_id: str
    next_step_s: float = 0.001

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        _number(self.time_s, "relaxation clock", nonnegative=True)
        _number(self.next_step_s, "next step", positive=True)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"phases", "time_s", "model_id", "next_step_s"}:
            raise ValueError("Unknown relaxation checkpoint fields")
        return cls(
            tuple(PRRelaxationPhase.from_dict(p) for p in data["phases"]),
            data["time_s"],
            data["model_id"],
            data["next_step_s"],
        )


class PRInterphaseRelaxation:
    def __init__(
        self,
        fluid,
        phase_roots,
        pair_drag_kg_s,
        *,
        wall_drag_kg_s=None,
        pressure_conductance_m3_pa_s=None,
        heat_conductance_w_k=None,
    ):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Caloric PR EOS required")
        self.fluid = fluid
        self.phase_roots = tuple(phase_roots)
        n = len(self.phase_roots)
        if n not in (2, 3) or any(tag not in ("liquid", "vapor") for tag in self.phase_roots):
            raise ValueError("Two/three explicit mechanical root branches required")
        self.momentum = PhaseMomentumExchange(
            pair_drag_kg_s, (0.0,) * n if wall_drag_kg_s is None else wall_drag_kg_s
        )
        if len(self.momentum.wall_drag_kg_s) != n:
            raise ValueError("Drag phase registry differs")
        zero = ((0.0,) * n,) * n
        self.pressure = symmetric_coefficients(
            zero if pressure_conductance_m3_pa_s is None else pressure_conductance_m3_pa_s,
            n,
            "pressure conductance",
        )
        self.thermal = symmetric_coefficients(
            zero if heat_conductance_w_k is None else heat_conductance_w_k, n, "heat conductance"
        )
        self.pressure_edges = tuple(
            (i, j, self.pressure[i][j])
            for i in range(n)
            for j in range(i + 1, n)
            if self.pressure[i][j] > 0
        )
        self.thermal_edges = tuple(
            (i, j, self.thermal[i][j])
            for i in range(n)
            for j in range(i + 1, n)
            if self.thermal[i][j] > 0
        )
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_interphase_relaxation_v1",
            "fluid": self.fluid.to_dict(),
            "phase_roots": self.phase_roots,
            "pair_drag_kg_s": self.momentum.pair_drag_kg_s,
            "wall_drag_kg_s": self.momentum.wall_drag_kg_s,
            "pressure_conductance_m3_pa_s": self.pressure,
            "heat_conductance_w_k": self.thermal,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        raw = dict(data)
        if raw.pop("model", None) != "pr_interphase_relaxation_v1":
            raise ValueError("Unknown relaxation model")
        raw["fluid"] = EquilibriumFluid.from_dict(raw["fluid"], device=device)
        return cls(**raw)

    def _phase(self, n, p, t, tag):
        f = self.fluid
        amount = n.sum()
        q = f.tensor_homogeneous(p, t, n / amount, tag, derivative_order=2)
        return (
            q,
            amount * q["molar_volume_m3_mol"],
            amount * q["molar_internal_energy_j_mol"],
            amount * q["molar_entropy_j_mol_k"],
        )

    def initialize(self, phases):
        if len(phases) != len(self.phase_roots):
            raise ValueError("Initial phase registry differs")
        result = []
        f = self.fluid
        for spec, tag in zip(phases, self.phase_roots):
            if set(spec) != {
                "pressure_pa",
                "temperature_k",
                "volume_m3",
                "mole_fractions",
                "velocity_m_s",
            }:
                raise ValueError("Explicit phase P/T/V/composition/velocity required")
            p = _number(spec["pressure_pa"], "phase pressure", positive=True)
            t = _number(spec["temperature_k"], "phase temperature", positive=True)
            volume = _number(spec["volume_m3"], "phase volume", positive=True)
            velocity = _number(spec["velocity_m_s"], "phase velocity")
            x = f.tensor(spec["mole_fractions"])
            f._validate(f.tensor(p), f.tensor(t), x)
            q = f.tensor_homogeneous(f.tensor(p), f.tensor(t), x, tag, derivative_order=2)
            if not bool(q["valid"].cpu()):
                raise PropertyModelError("Initial homogeneous branch is mechanically invalid")
            n = x * volume / q["molar_volume_m3_mol"]
            mass = (n * f.molar_masses).sum()
            energy = n.sum() * q["molar_internal_energy_j_mol"] + 0.5 * mass * velocity**2
            result.append(
                PRRelaxationPhase(
                    tuple(n.cpu().tolist()),
                    volume,
                    p,
                    t,
                    float((mass * velocity).cpu()),
                    float(energy.cpu()),
                )
            )
        state = PRRelaxationState(tuple(result), 0.0, self.model_id)
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRRelaxationState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.phases) != len(self.phase_roots)
        ):
            raise PropertyModelError("Relaxation model/state mismatch")
        f = self.fluid
        for phase, tag in zip(state.phases, self.phase_roots):
            n = f.tensor(phase.component_moles)
            p, t = f.tensor(phase.pressure_pa), f.tensor(phase.temperature_k)
            f._validate(p, t, n / n.sum())
            q, v, u, _s = self._phase(n, p, t, tag)
            mass = (n * f.molar_masses).sum()
            residual = torch.stack(
                (
                    (v - phase.volume_m3) / phase.volume_m3,
                    (u + 0.5 * phase.momentum_kg_m_s**2 / mass - phase.total_energy_j)
                    / max(1.0, abs(phase.total_energy_j)),
                )
            )
            if not bool(
                (q["valid"] & torch.isfinite(residual).all() & (residual.abs().max() < 1e-9)).cpu()
            ):
                raise PropertyModelError("Phase EOS/volume/energy checkpoint disagrees")

    def _problem(self, state, dt):
        self.inspect(state)
        f, nph = self.fluid, len(state.phases)
        stocks = [f.tensor(s.component_moles) for s in state.phases]
        mass = torch.stack([(n * f.molar_masses).sum() for n in stocks])
        pold = f.tensor([s.pressure_pa for s in state.phases])
        told = f.tensor([s.temperature_k for s in state.phases])
        vold = f.tensor([s.volume_m3 for s in state.phases])
        uold = f.tensor([s.momentum_kg_m_s for s in state.phases]) / mass
        eold = f.tensor([s.total_energy_j for s in state.phases]) - 0.5 * mass * uold.square()
        _vguess, momentum_guess = self.momentum.solve(mass, uold, dt)
        total_volume = vold.sum()
        pscale = pold.max()
        velocity_scale = max(1.0, float(uold.abs().max().cpu()))
        old_thermo = [
            self._phase(n, p, t, tag) for n, p, t, tag in zip(stocks, pold, told, self.phase_roots)
        ]
        energy_scale = torch.stack(
            [
                n.sum() * q[0]["homogeneous_molar_cv_j_mol_k"] * t
                for n, q, t in zip(stocks, old_thermo, told)
            ]
        ).clamp_min(1.0)
        old_entropy = torch.stack([q[3] for q in old_thermo]).sum()
        volumes = len(self.pressure_edges)
        heats = len(self.thermal_edges)
        initial = torch.cat(
            (
                pold.log(),
                told / f.tref,
                momentum_guess["velocity_coordinates"] / velocity_scale,
                f.tensor([0.0] * (volumes + heats)),
            )
        )
        lower = f.tensor(
            [math.log(f.min_pressure_pa)] * nph
            + [f.min_temperature_k / f.tref] * nph
            + [-1e6 / velocity_scale] * nph
            + [-1.0] * volumes
            + [-1e6] * heats
        )
        upper = f.tensor(
            [math.log(f.max_pressure_pa)] * nph
            + [f.max_temperature_k / f.tref] * nph
            + [1e6 / velocity_scale] * nph
            + [1.0] * volumes
            + [1e6] * heats
        )
        heat_scale = energy_scale.sum()

        def evaluate(y):
            p, t, coordinates = (
                y[:nph].exp(),
                y[nph : 2 * nph] * f.tref,
                y[2 * nph : 3 * nph] * velocity_scale,
            )
            velocity, _fluctuation = self.momentum.kinematics(mass, coordinates)
            q, volume, internal, entropy = [], [], [], []
            for i, tag in enumerate(self.phase_roots):
                qi, vi, ei, si = self._phase(stocks[i], p[i], t[i], tag)
                q.append(qi)
                volume.append(vi)
                internal.append(ei)
                entropy.append(si)
            volume, internal, entropy = map(torch.stack, (volume, internal, entropy))
            drag = self.momentum.terms(mass, uold, velocity, dt, coordinates=coordinates)
            dv = [y.new_zeros(()) for _ in range(nph)]
            work = [y.new_zeros(()) for _ in range(nph)]
            heat = [y.new_zeros(()) for _ in range(nph)]
            rows, pressure_entropy, heat_entropy = [], y.new_zeros(()), y.new_zeros(())
            transfers_v = y[3 * nph : 3 * nph + volumes] * total_volume
            transfers_q = y[3 * nph + volumes :] * heat_scale
            for k, (i, j, c) in enumerate(self.pressure_edges):
                delta = transfers_v[k]
                dv[i] = dv[i] + delta
                dv[j] = dv[j] - delta
                w = 0.5 * (p[i] + p[j]) * delta
                work[i] = work[i] + w
                work[j] = work[j] - w
                rows.append(((p[i] - p[j]) - delta / (dt * c)) / pscale)
                pressure_entropy = pressure_entropy + 0.5 * (p[i] - p[j]) * delta * (
                    1 / t[i] + 1 / t[j]
                )
            for k, (i, j, c) in enumerate(self.thermal_edges):
                delta = transfers_q[k]
                heat[i] = heat[i] + delta
                heat[j] = heat[j] - delta
                rows.append(((t[j] - t[i]) - delta / (dt * c)) / f.tref)
                heat_entropy = heat_entropy + delta * (1 / t[i] - 1 / t[j])
            dv, work, heat = map(torch.stack, (dv, work, heat))
            er = (internal - eold + work - heat - drag["internal_energy_gain_j"]) / energy_scale
            mr = drag["momentum_residual"] / (mass * velocity_scale)
            residual = torch.cat(
                ((volume - vold - dv) / vold, er, mr, torch.stack(rows) if rows else y.new_empty(0))
            )
            lower_entropy = (
                pressure_entropy + heat_entropy + (drag["internal_energy_gain_j"] / t).sum()
            )
            return residual, {
                "pressure": p,
                "temperature": t,
                "velocity": velocity,
                "volume": volume,
                "internal": internal,
                "entropy": entropy,
                "old_entropy": old_entropy,
                "mass": mass,
                "drag": drag,
                "volume_transfer": transfers_v,
                "heat_transfer": transfers_q,
                "volume_change": dv,
                "pressure_work": work,
                "heat": heat,
                "entropy_lower_bound": lower_entropy,
                "pressure_entropy": pressure_entropy,
                "heat_entropy": heat_entropy,
                "valid": torch.stack([v["valid"] for v in q]).all()
                & drag["valid"]
                & torch.isfinite(residual).all(),
            }

        return initial, lower, upper, evaluate

    def step(self, state, dt_s):
        dt = _number(dt_s, "relaxation step", positive=True)
        if not math.isfinite(state.time_s + dt) or state.time_s + dt <= state.time_s:
            raise ValueError("Step below clock resolution")
        y, lo, hi, evaluate = self._problem(state, dt)
        solver = TorchNewton(self.fluid.device, maximum_iterations=40, dense_limit=32)
        solver.host_convergence = True
        with torch.no_grad():
            y, _r, ok, count = solver.solve(lambda z: evaluate(z)[0], y, lo, hi, tolerance=2e-12)
            residual, d = evaluate(y)
            if not ok or not bool(d["valid"].cpu()) or float(residual.abs().max().cpu()) > 1e-10:
                raise HydraulicConvergenceError("Finite phase relaxation Newton failed")
            phases = tuple(
                PRRelaxationPhase(
                    s.component_moles,
                    float(d["volume"][i].cpu()),
                    float(d["pressure"][i].cpu()),
                    float(d["temperature"][i].cpu()),
                    float((d["mass"][i] * d["velocity"][i]).cpu()),
                    float((d["internal"][i] + 0.5 * d["mass"][i] * d["velocity"][i] ** 2).cpu()),
                )
                for i, s in enumerate(state.phases)
            )
            new = PRRelaxationState(phases, state.time_s + dt, self.model_id, state.next_step_s)
            self.inspect(new)
            old_s = d["old_entropy"]
            entropy = float((d["entropy"].sum() - old_s).cpu())
            bound = float(d["entropy_lower_bound"].cpu())
            volume_error = math.fsum(s.volume_m3 for s in new.phases) - math.fsum(
                s.volume_m3 for s in state.phases
            )
            energy_error = math.fsum(s.total_energy_j for s in new.phases) - math.fsum(
                s.total_energy_j for s in state.phases
            )
            wall_impulse = dt * float(d["drag"]["wall_force_n"].sum().cpu())
            momentum_error = (
                math.fsum(s.momentum_kg_m_s for s in new.phases)
                - math.fsum(s.momentum_kg_m_s for s in state.phases)
                - wall_impulse
            )
            if (
                abs(volume_error) > 1e-12 + 1e-9 * sum(s.volume_m3 for s in state.phases)
                or abs(energy_error)
                > 1e-8 + 1e-10 * sum(abs(s.total_energy_j) for s in state.phases)
                or abs(momentum_error)
                > 1e-10 + 1e-9 * sum(abs(s.momentum_kg_m_s) for s in state.phases)
            ):
                raise ConservationError("Phase relaxation V/E/momentum ledger failed")
            if entropy < -1e-8 or entropy < bound - 1e-8 - 1e-10 * max(1.0, abs(entropy)):
                raise ConservationError("Phase relaxation entropy inequality failed")
        return new, {
            "dt_s": dt,
            "state": new.to_dict(),
            "volume_residual_m3": volume_error,
            "energy_residual_j": energy_error,
            "momentum_residual_kg_m_s": momentum_error,
            "wall_impulse_kg_m_s": wall_impulse,
            "entropy_change_j_k": entropy,
            "entropy_source_lower_bound_j_k": bound,
            "thermodynamic_temporal_entropy_j_k": entropy - bound,
            "physical_dissipation_j": d["drag"]["physical_dissipation_j"].cpu().tolist(),
            "temporal_dissipation_j": d["drag"]["temporal_dissipation_j"].cpu().tolist(),
            "volume_transfers_m3": d["volume_transfer"].cpu().tolist(),
            "heat_transfers_j": d["heat_transfer"].cpu().tolist(),
            "pressure_work_j": d["pressure_work"].cpu().tolist(),
            "net_heat_j": d["heat"].cpu().tolist(),
            "pressure_relaxation_entropy_j_k": float(d["pressure_entropy"].cpu()),
            "heat_relaxation_entropy_j_k": float(d["heat_entropy"].cpu()),
            "drag_physical_entropy_j_k": float(
                (d["drag"]["physical_dissipation_j"] / d["temperature"]).sum().cpu()
            ),
            "drag_temporal_entropy_j_k": float(
                (d["drag"]["temporal_dissipation_j"] / d["temperature"]).sum().cpu()
            ),
            "maximum_scaled_residual": float(residual.abs().max().cpu()),
            "velocity_coordinates_m_s": d["drag"]["velocity_coordinates"].cpu().tolist(),
            "pair_impulses_kg_m_s": (dt * d["drag"]["pair_force_n"]).cpu().tolist(),
            "evaluations": count,
        }

    def ledger(self, old, new, wall_impulse):
        components = [
            math.fsum(s.component_moles[i] for s in new.phases)
            - math.fsum(s.component_moles[i] for s in old.phases)
            for i in range(len(self.fluid.names))
        ]
        return components + [
            math.fsum(s.volume_m3 for s in new.phases) - math.fsum(s.volume_m3 for s in old.phases),
            math.fsum(s.total_energy_j for s in new.phases)
            - math.fsum(s.total_energy_j for s in old.phases),
            math.fsum(s.momentum_kg_m_s for s in new.phases)
            - math.fsum(s.momentum_kg_m_s for s in old.phases)
            - wall_impulse,
        ]

    def advance(
        self,
        state,
        duration_s,
        *,
        relative_tolerance=1e-4,
        velocity_tolerance_m_s=1e-4,
        maximum_step_s=0.01,
        minimum_step_s=1e-10,
        maximum_attempts=1000,
    ):
        self.inspect(state)
        duration = _number(duration_s, "relaxation duration", positive=True)
        tolerance = _number(relative_tolerance, "relative tolerance", positive=True)
        velocity_tolerance = _number(
            velocity_tolerance_m_s, "absolute velocity tolerance", positive=True
        )
        maximum = _number(maximum_step_s, "maximum step", positive=True)
        minimum = _number(minimum_step_s, "minimum step", positive=True)
        if maximum < minimum or type(maximum_attempts) is not int or maximum_attempts < 1:
            raise ValueError("Ordered step bounds and positive attempt budget required")
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Interval below clock resolution")
        initial = state
        dt, steps = min(duration, maximum, state.next_step_s), []
        for attempt in range(maximum_attempts):
            dt = interval_step_s(state.time_s, end, dt)
            reached_end = dt >= end - state.time_s
            a, b = split_step_s(state.time_s, dt)
            try:
                coarse, _ = self.step(state, dt)
                half, ra = self.step(state, a)
                fine, rb = self.step(half, b)
                error = 0.0
                for ca, cb in zip(coarse.phases, fine.phases):
                    for x, y, minimum_scale in (
                        (ca.pressure_pa, cb.pressure_pa, 1.0),
                        (ca.temperature_k, cb.temperature_k, 1.0),
                        (ca.volume_m3, cb.volume_m3, 1e-12),
                    ):
                        error = max(
                            error, abs(x - y) / (tolerance * max(minimum_scale, abs(x), abs(y)))
                        )
                    mass = sum(
                        n * c.molar_mass_kg_mol
                        for n, c in zip(ca.component_moles, self.fluid.components)
                    )
                    va, vb = ca.momentum_kg_m_s / mass, cb.momentum_kg_m_s / mass
                    error = max(
                        error,
                        abs(va - vb) / (velocity_tolerance + tolerance * max(abs(va), abs(vb))),
                    )
                if error > 1:
                    raise HydraulicConvergenceError("Phase relaxation temporal error exceeded")
            except (HydraulicConvergenceError, PropertyModelError, ConservationError):
                dt /= 2
                if dt < minimum:
                    raise
                continue
            nextdt = min(maximum, max(minimum, dt * (1.5 if error < 0.25 else 1.0)))
            state = replace(fine, time_s=end if reached_end else fine.time_s, next_step_s=nextdt)
            if reached_end:
                rb = {**rb, "state": replace(fine, time_s=end).to_dict()}
            steps.extend((ra, rb))
            dt = nextdt
            if state.time_s >= end:
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": self.ledger(
                        initial, state, math.fsum(r["wall_impulse_kg_m_s"] for r in steps)
                    ),
                }
        raise HydraulicConvergenceError("Phase relaxation interval budget exhausted")
