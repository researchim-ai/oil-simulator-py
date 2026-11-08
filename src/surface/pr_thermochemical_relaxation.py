"""Finite-time-constant, conservative extensive BGK source for PR phases.

Independent phase P/T/u and compositions persist at finite relaxation time.
The target is the entropy-maximizing UV equilibrium with the same N/V/E/M.
This phenomenological HRM source is not a calibrated nucleation/area law and
is not a vacuum Riemann solver for spatial Baer--Nunziato phase fronts.
"""

import itertools
import math
from dataclasses import replace

from .liquid import ConservationError, HydraulicConvergenceError, PropertyModelError, _number
from .operation import _digest
from .peng_robinson import R
from .pr_canonical_equilibrium import canonical_equilibrium_target
from .pr_homogeneous_uv import caloric_tv, recover_homogeneous_phases
from .pr_phase_equilibration import PRPhaseEquilibration, PRPhaseEquilibriumState
from .pr_stock_balances import balance_limits, entropy_allowance


def phase_families(fluid, state):
    """Classify cubic mechanical branches without a fitted density cutoff.

    The PR critical reduced volume solves w^3-3w^2-3w-3=0.
    Below the critical attraction ratio both branches are one continuous fluid.
    Above it mechanically stable volumes on either side are liquid/vapor.
    """
    critical_v = 1 + (4 + math.sqrt(8)) ** (1 / 3) + (4 - math.sqrt(8)) ** (1 / 3)
    critical_a = (critical_v**2 + 2 * critical_v - 1) ** 2 / (
        2 * (critical_v + 1) * (critical_v - 1) ** 2
    )
    n = fluid.tensor([p.component_moles for p in state.phases])
    amount = n.sum(-1)
    t = fluid.tensor([p.temperature_k for p in state.phases])
    v = fluid.tensor([p.volume_m3 for p in state.phases]) / amount
    q = caloric_tv(fluid, t, v, n / amount[:, None])
    ratios = (q["a"] / (R * t * q["b"])).cpu().tolist()
    reduced = (v / q["b"]).cpu().tolist()
    return tuple(
        "stable"
        if ratio <= critical_a * (1 + 1e-12)
        else "liquid"
        if volume < critical_v
        else "vapor"
        for ratio, volume in zip(ratios, reduced)
    )


def phase_assignment(old, target, old_families, target_families):
    """Maximum compatible-root matching, then minimum composition distance.

    Unmatched old phases decay; unmatched target phases are genuinely created.
    No different cubic branches are merged just to fit a fixed slot registry.
    """
    old_x = [tuple(v / math.fsum(p.component_moles) for v in p.component_moles) for p in old.phases]
    new_x = [
        tuple(v / math.fsum(p.component_moles) for v in p.component_moles) for p in target.phases
    ]
    best = None
    for choices in itertools.product(range(-1, len(target.phases)), repeat=len(old.phases)):
        assigned = [v for v in choices if v >= 0]
        if len(assigned) != len(set(assigned)):
            continue
        if any(
            j >= 0
            and old_families[i] != target_families[j]
            and "stable" not in (old_families[i], target_families[j])
            for i, j in enumerate(choices)
        ):
            continue
        distance = math.fsum(
            math.fsum(abs(a - b) for a, b in zip(old_x[i], new_x[j]))
            for i, j in enumerate(choices)
            if j >= 0
        )
        intensive_distance = math.fsum(
            math.log(
                (old.phases[i].volume_m3 / math.fsum(old.phases[i].component_moles))
                / (target.phases[j].volume_m3 / math.fsum(target.phases[j].component_moles))
            )
            ** 2
            + math.log(old.phases[i].temperature_k / target.phases[j].temperature_k) ** 2
            for i, j in enumerate(choices)
            if j >= 0
        )
        key = (-len(assigned), distance, intensive_distance, choices)
        if best is None or key < best[0]:
            best = (key, choices)
    assigned = list(best[1])
    # Multiple material populations may approach the same equilibrium phase.
    # Share its extensive target rather than privileging one duplicate slot.
    for i, j in enumerate(assigned):
        if j >= 0:
            continue
        compatible = [
            k
            for k, r in enumerate(target_families)
            if old_families[i] == r or "stable" in (old_families[i], r)
        ]
        if compatible:
            assigned[i] = min(
                compatible, key=lambda k: math.fsum(abs(a - b) for a, b in zip(old_x[i], new_x[k]))
            )
    return tuple(assigned)


class PRThermochemicalRelaxation:
    def __init__(self, fluid, relaxation_time_s, *, maximum_active_phases=6):
        self.equilibrium = PRPhaseEquilibration(fluid)
        self.fluid = fluid
        self.relaxation_time_s = _number(
            relaxation_time_s, "thermochemical relaxation time", nonnegative=True
        )
        if type(maximum_active_phases) is not int or not 1 <= maximum_active_phases <= 6:
            raise ValueError("Explicit active phase budget in 1..6 required")
        self.maximum_active_phases = maximum_active_phases
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {
            "model": "pr_thermochemical_relaxation_v1",
            "fluid": self.fluid.to_dict(),
            "relaxation_time_s": self.relaxation_time_s,
            "maximum_active_phases": self.maximum_active_phases,
        }

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        from .equilibrium_properties import EquilibriumFluid

        if (
            set(data) != {"model", "fluid", "relaxation_time_s", "maximum_active_phases"}
            or data["model"] != "pr_thermochemical_relaxation_v1"
        ):
            raise ValueError("Unknown thermochemical relaxation model")
        return cls(
            EquilibriumFluid.from_dict(data["fluid"], device=device),
            data["relaxation_time_s"],
            maximum_active_phases=data["maximum_active_phases"],
        )

    def initialize(self, phases, phase_roots, *, time_s=0.0):
        state = PRPhaseEquilibriumState(tuple(phases), tuple(phase_roots), self.model_id, time_s)
        self.inspect(state)
        return state

    def inspect(self, state):
        if (
            not isinstance(state, PRPhaseEquilibriumState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
            or len(state.phases) > self.maximum_active_phases
        ):
            raise PropertyModelError("Thermochemical phase registry/model differs")
        cached = getattr(self, "_inspection_cache", None)
        if cached is not None and cached[0] is state:
            return cached[1]
        adapted = PRPhaseEquilibriumState(
            state.phases, state.phase_roots, self.equilibrium.model_id, state.time_s
        )
        entropy = self.equilibrium.inspect(adapted)
        self._inspection_cache = (state, entropy)
        return entropy

    def _totals(self, state):
        return [
            *(
                math.fsum(p.component_moles[j] for p in state.phases)
                for j in range(len(self.fluid.names))
            ),
            math.fsum(p.volume_m3 for p in state.phases),
            math.fsum(p.total_energy_j for p in state.phases),
            math.fsum(p.momentum_kg_m_s for p in state.phases),
        ]

    def step(self, state, dt_s):
        entropy_old = self.inspect(state)
        dt = _number(dt_s, "thermochemical source step", positive=True)
        end = state.time_s + dt
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Source step below clock resolution")
        dt = end - state.time_s
        source = PRPhaseEquilibriumState(
            state.phases, state.phase_roots, self.equilibrium.model_id, state.time_s
        )
        cached = getattr(self, "_target_cache", None)
        if cached is not None and cached[0] is state:
            flash_target = replace(cached[1], time_s=state.time_s)
            target = replace(cached[2], time_s=state.time_s)
        else:
            conserved = self._totals(state)
            nc = len(self.fluid.names)
            flash_target = self.equilibrium.recover(
                conserved[:nc],
                conserved[nc + 1],
                conserved[nc + 2],
                conserved[nc],
                pressure_guess_pa=math.fsum(p.pressure_pa * p.volume_m3 for p in state.phases)
                / conserved[nc],
                temperature_guess_k=math.fsum(
                    p.temperature_k * math.fsum(p.component_moles) for p in state.phases
                )
                / math.fsum(conserved[:nc]),
                time_s=state.time_s,
            )
            target = canonical_equilibrium_target(self.fluid, source, flash_target)
        target_entropy = self.equilibrium.inspect(target) - entropy_old
        if self.relaxation_time_s == 0.0:
            weight_old, weight_target = 0.0, 1.0
        else:
            ratio = dt / self.relaxation_time_s
            weight_old = math.exp(-ratio)
            weight_target = -math.expm1(-ratio)
            # Underflow cannot silently turn finite-rate kinetics into an
            # instantaneous equilibrium switch. Select tau=0 explicitly.
            if weight_old == 0.0 or weight_target == 0.0:
                raise PropertyModelError(
                    "Relaxation weights outside representable precision; select an explicit instantaneous source or a resolved step"
                )
        old_families = phase_families(self.fluid, state)
        target_families = phase_families(self.fluid, target)
        assignment = phase_assignment(state, target, old_families, target_families)
        used = {j for j in assignment if j >= 0}
        pairs = []
        if weight_old:
            for i, j in enumerate(assignment):
                total = (
                    math.fsum(
                        math.fsum(state.phases[k].component_moles)
                        for k, v in enumerate(assignment)
                        if v == j
                    )
                    if j >= 0
                    else 1.0
                )
                share = math.fsum(state.phases[i].component_moles) / total if j >= 0 else 0.0
                pairs.append((i, j if j >= 0 else None, share))
        pairs += [
            (None, j, 1.0) for j in range(len(target.phases)) if j not in used or not weight_old
        ]
        if len(pairs) > self.maximum_active_phases:
            raise ValueError("Thermochemical transition exceeds declared active-phase budget")
        amounts = []
        volumes = []
        momenta = []
        energies = []
        roots = []
        temperatures = []
        for i, j, share in pairs:
            old = None if i is None else state.phases[i]
            eq = None if j is None else target.phases[j]

            def interpolate(a, b):
                if a is None:
                    return weight_target * b
                if b is None:
                    return weight_old * a
                # Exact fixed points and small increments; keep the decaying
                # exponential separately when the target fraction rounds to 1.
                return (
                    math.fsum((a, weight_target * (b - a)))
                    if weight_target <= 0.5
                    else math.fsum((b, weight_old * (a - b)))
                )

            def blend(field, old=old, eq=eq, share=share):
                return interpolate(
                    getattr(old, field) if old else None, share * getattr(eq, field) if eq else None
                )

            n = tuple(
                interpolate(
                    old.component_moles[k] if old else None,
                    share * eq.component_moles[k] if eq else None,
                )
                for k in range(len(self.fluid.names))
            )
            if math.fsum(n) <= 0.0 or blend("volume_m3") <= 0.0:
                raise PropertyModelError("Finite active phase fell below representable precision")
            amounts.append(n)
            volumes.append(blend("volume_m3"))
            momenta.append(blend("momentum_kg_m_s"))
            energies.append(blend("total_energy_j"))
            # Keep a selected homogeneous branch; a stable root denotes a
            # single-root state and may be continued to the matched branch.
            root = state.phase_roots[i] if old else target.phase_roots[j]
            if old and old_families[i] == "stable" and eq:
                root = target_families[j]
            elif root == "stable":
                root = old_families[i] if old else target_families[j]
            roots.append(root)
            temperatures.append(
                (
                    math.fsum(
                        (
                            [weight_old * math.fsum(old.component_moles) * old.temperature_k]
                            if old
                            else []
                        )
                        + (
                            [
                                weight_target
                                * share
                                * math.fsum(eq.component_moles)
                                * eq.temperature_k
                            ]
                            if eq
                            else []
                        )
                    )
                )
                / math.fsum(n)
            )
        if not weight_old:
            phases = target.phases
            roots = target.phase_roots
        else:
            phases = recover_homogeneous_phases(
                self.fluid, amounts, volumes, momenta, energies, roots, temperatures
            )
        new = PRPhaseEquilibriumState(tuple(phases), tuple(roots), self.model_id, end)
        entropy_new = self.inspect(new)
        production = entropy_new - entropy_old
        residual = [b - a for a, b in zip(self._totals(state), self._totals(new))]
        limits = balance_limits(
            state.phases, new.phases, len(self.fluid.names), include_volume=True
        )
        entropy_limit = entropy_allowance(entropy_old, entropy_new, target_entropy + entropy_old)
        if any(abs(r) > tol for r, tol in zip(residual, limits)) or production < -entropy_limit:
            raise ConservationError("Thermochemical N/V/E/M or entropy balance failed")
        # Concavity of phase entropy provides a stronger lower bound than
        # nonnegative total production on every accepted interpolation.
        lower_bound = weight_target * target_entropy
        if production < lower_bound - entropy_limit:
            raise ConservationError(
                "Thermochemical convex relaxation violates its entropy lower bound"
            )
        self._target_cache = (new, flash_target, target)
        return new, {
            "state": new.to_dict(),
            "dt_s": dt,
            "old_weight": weight_old,
            "equilibrium_weight": weight_target,
            "equilibrium_target": target.to_dict(),
            "flash_target": flash_target.to_dict(),
            "old_families": old_families,
            "target_families": target_families,
            "assignment": assignment,
            "phase_pairs": pairs,
            "entropy_production_j_k": production,
            "entropy_lower_bound_j_k": lower_bound,
            "conservation_residuals": residual,
            "conservation_tolerances": limits,
            "entropy_tolerance_j_k": entropy_limit,
        }

    def advance(
        self,
        state,
        duration_s,
        *,
        maximum_step_s=None,
        minimum_step_s=1e-12,
        maximum_attempts=10000,
    ):
        self.inspect(state)
        duration = _number(duration_s, "source interval", positive=True)
        maximum = (
            duration
            if maximum_step_s is None
            else _number(maximum_step_s, "source maximum step", positive=True)
        )
        minimum = _number(minimum_step_s, "source minimum step", positive=True)
        if type(maximum_attempts) is not int or maximum_attempts < 1 or minimum > maximum:
            raise ValueError("Invalid source integration budget")
        end = state.time_s + duration
        if not math.isfinite(end) or end <= state.time_s:
            raise ValueError("Source interval below clock resolution")
        first = state
        steps = []
        dt = min(maximum, duration)
        for attempt in range(maximum_attempts):
            proposed = min(dt, end - state.time_s)
            target = min(end, state.time_s + proposed)
            if target - state.time_s > proposed:
                target = math.nextafter(target, state.time_s)
            dt = target - state.time_s
            if dt <= 0:
                raise ValueError("Source interval resolution exhausted")
            try:
                new, report = self.step(state, dt)
            except (PropertyModelError, HydraulicConvergenceError, ConservationError):
                dt *= 0.5
                if dt < minimum:
                    raise
                continue
            steps.append(report)
            state = new
            dt = min(maximum, dt * 1.5)
            if state.time_s >= end:
                return {
                    "state": state.to_dict(),
                    "steps": steps,
                    "attempts": attempt + 1,
                    "conservation_residuals": [
                        b - a for a, b in zip(self._totals(first), self._totals(state))
                    ],
                    "entropy_production_j_k": math.fsum(r["entropy_production_j_k"] for r in steps),
                }
        raise HydraulicConvergenceError("Thermochemical source interval budget exhausted")
