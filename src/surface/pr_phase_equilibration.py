"""Conservative UV equilibration of finite moving PR phase inventories.

This is an instantaneous homogeneous-equilibrium closure, not finite-rate
nucleation. Phase number may change; no artificial residual phase is stored.
Spatial multifluid active-set transport remains a separate responsibility.
"""

import math
from dataclasses import dataclass

from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, PropertyModelError, _number
from .operation import _digest
from .pr_interphase_relaxation import PRRelaxationPhase


@dataclass(frozen=True)
class PRPhaseEquilibriumState:
    phases: tuple
    phase_roots: tuple
    model_id: str
    time_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "phase_roots", tuple(self.phase_roots))
        if not self.phases or len(self.phases) != len(self.phase_roots):
            raise ValueError("Explicit nonempty active phase registry required")
        if any(not isinstance(p, PRRelaxationPhase) for p in self.phases) or any(
            r not in ("vapor", "liquid", "stable") for r in self.phase_roots
        ):
            raise ValueError("Typed finite phase stocks and PR roots required")
        _number(self.time_s, "equilibrium clock", nonnegative=True)

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if set(data) != {"phases", "phase_roots", "model_id", "time_s"}:
            raise ValueError("Unknown phase-equilibrium checkpoint fields")
        return cls(
            tuple(PRRelaxationPhase.from_dict(p) for p in data["phases"]),
            tuple(data["phase_roots"]),
            data["model_id"],
            data["time_s"],
        )


class PRPhaseEquilibration:
    def __init__(self, fluid):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Caloric PR fluid required")
        self.fluid = fluid
        self.model_id = _digest(self.to_dict())

    def to_dict(self):
        return {"model": "pr_phase_equilibration_v1", "fluid": self.fluid.to_dict()}

    @classmethod
    def from_dict(cls, data, *, device="cpu"):
        if set(data) != {"model", "fluid"} or data["model"] != "pr_phase_equilibration_v1":
            raise ValueError("Unknown phase-equilibration model")
        return cls(EquilibriumFluid.from_dict(data["fluid"], device=device))

    def _mass(self, phase):
        return math.fsum(
            n * c.molar_mass_kg_mol for n, c in zip(phase.component_moles, self.fluid.components)
        )

    def inspect(self, state):
        if (
            not isinstance(state, PRPhaseEquilibriumState)
            or state.model_id != self.model_id
            or _digest(self.to_dict()) != self.model_id
        ):
            raise PropertyModelError("Phase-equilibrium model/state differs")
        entropy = 0.0
        for phase, root in zip(state.phases, state.phase_roots):
            if len(phase.component_moles) != len(self.fluid.components):
                raise PropertyModelError("Phase component registry differs")
            amount = math.fsum(phase.component_moles)
            q = self.fluid.homogeneous(
                phase.pressure_pa,
                phase.temperature_k,
                [n / amount for n in phase.component_moles],
                root,
            )
            volume = amount * float(q["molar_volume_m3_mol"].cpu())
            energy = amount * float(
                q["molar_internal_energy_j_mol"].cpu()
            ) + 0.5 * phase.momentum_kg_m_s**2 / self._mass(phase)
            if abs(volume - phase.volume_m3) > 1e-12 + 1e-9 * volume or abs(
                energy - phase.total_energy_j
            ) > 1e-6 + 1e-9 * abs(energy):
                raise ConservationError("Phase checkpoint disagrees with caloric EOS")
            entropy += amount * float(q["molar_entropy_j_mol_k"].cpu())
        return entropy

    def initialize(self, phases, phase_roots, *, time_s=0.0):
        state = PRPhaseEquilibriumState(tuple(phases), tuple(phase_roots), self.model_id, time_s)
        self.inspect(state)
        return state

    def project(self, state, *, heat_j=0.0, heat_temperature_k=None):
        initial_entropy = self.inspect(state)
        heat = _number(heat_j, "external heat")
        if heat and heat_temperature_k is None:
            raise ValueError("Prescribed heat requires its reservoir temperature")
        bath = (
            None
            if heat_temperature_k is None
            else _number(heat_temperature_k, "heat reservoir temperature", positive=True)
        )
        nc = len(self.fluid.components)
        n = tuple(math.fsum(p.component_moles[j] for p in state.phases) for j in range(nc))
        momentum = math.fsum(p.momentum_kg_m_s for p in state.phases)
        volume = math.fsum(p.volume_m3 for p in state.phases)
        energy = math.fsum(p.total_energy_j for p in state.phases) + heat
        new = self.recover(
            n,
            energy,
            momentum,
            volume,
            pressure_guess_pa=math.fsum(p.pressure_pa * p.volume_m3 for p in state.phases) / volume,
            temperature_guess_k=math.fsum(
                p.temperature_k * math.fsum(p.component_moles) for p in state.phases
            )
            / math.fsum(n),
            time_s=state.time_s,
        )
        entropy = self.inspect(new) - initial_entropy - (0.0 if bath is None else heat / bath)
        residual = [math.fsum(p.component_moles[j] for p in new.phases) - n[j] for j in range(nc)]
        residual += [
            math.fsum(p.volume_m3 for p in new.phases) - volume,
            math.fsum(p.total_energy_j for p in new.phases) - energy,
            math.fsum(p.momentum_kg_m_s for p in new.phases) - momentum,
        ]
        if (
            max(map(abs, residual[:nc])) > 1e-8
            or abs(residual[nc]) > 1e-12 + 1e-9 * volume
            or abs(residual[nc + 1]) > 1e-6 + 1e-9 * abs(energy)
            or abs(residual[-1]) > 1e-9
            or entropy < -1e-7
        ):
            raise ConservationError("UV phase equilibration failed component/V/E/P/entropy audit")
        return new, {
            "state": new.to_dict(),
            "phase_count_before": len(state.phases),
            "phase_count_after": len(new.phases),
            "heat_j": heat,
            "heat_temperature_k": bath,
            "entropy_production_j_k": entropy,
            "conservation_residuals": residual,
        }

    def recover(
        self,
        component_moles,
        total_energy_j,
        momentum_kg_m_s,
        volume_m3,
        *,
        pressure_guess_pa,
        temperature_guess_k,
        time_s=0.0,
    ):
        """Recover active equilibrium phases from transported bulk N/E/P/V.

        Unlike project(), no temporary off-EOS phase parcels are fabricated.
        This is the conservative interface for a spatial equilibrium closure.
        """
        n = tuple(_number(v, "component inventory", nonnegative=True) for v in component_moles)
        if len(n) != len(self.fluid.components) or not math.fsum(n) > 0:
            raise PropertyModelError("Nonempty component registry required")
        energy = _number(total_energy_j, "total energy")
        momentum = _number(momentum_kg_m_s, "momentum")
        volume = _number(volume_m3, "cell volume", positive=True)
        _number(time_s, "recovery clock", nonnegative=True)
        mass = math.fsum(v * c.molar_mass_kg_mol for v, c in zip(n, self.fluid.components))
        velocity = momentum / mass
        q = self.fluid.flash_uv(
            n,
            energy - 0.5 * mass * velocity**2,
            volume,
            pressure_guess_pa=pressure_guess_pa,
            temperature_guess_k=temperature_guess_k,
        )
        pressure = float(q["pressure_pa"].cpu())
        temperature = float(q["temperature_k"].cpu())
        phases = []
        for phase in q["phases"]:
            numbers = tuple(
                math.fsum(n) * float(phase["molar_fraction"].cpu()) * float(x)
                for x in phase["mole_fractions"].cpu().tolist()
            )
            amount = math.fsum(numbers)
            if amount <= 0:
                raise ConservationError("Flash returned an empty active phase")
            vm = float(phase["molar_volume_m3_mol"].cpu())
            m = math.fsum(x * c.molar_mass_kg_mol for x, c in zip(numbers, self.fluid.components))
            errors = []
            for root in ["liquid", "vapor"]:
                homogeneous = self.fluid.homogeneous(
                    pressure, temperature, [x / amount for x in numbers], root
                )
                errors.append(abs(float(homogeneous["molar_volume_m3_mol"].cpu()) - vm) / vm)
            if min(errors) > 1e-7:
                raise ConservationError("Flash phase root does not match homogeneous EOS")
            root = (
                "stable" if max(errors) < 1e-9 else ("liquid" if errors[0] < errors[1] else "vapor")
            )
            stock = PRRelaxationPhase(
                numbers,
                amount * vm,
                pressure,
                temperature,
                m * velocity,
                amount * float(phase["molar_internal_energy_j_mol"].cpu()) + 0.5 * m * velocity**2,
            )
            phases.append((m / stock.volume_m3, root, stock))

        # Density ordering alone permutes symmetric liquid phases under roundoff
        # across devices. Composition provides a deterministic active registry.
        def phase_key(row):
            density, root, stock = row
            amount = math.fsum(stock.component_moles)
            x = tuple(n / amount for n in stock.component_moles)
            dominant = max(range(len(x)), key=x.__getitem__)
            return (root != "vapor", dominant, tuple(round(v, 12) for v in x), density)

        phases.sort(key=phase_key)
        new = PRPhaseEquilibriumState(
            tuple(p[2] for p in phases), tuple(p[1] for p in phases), self.model_id, time_s
        )

        self.inspect(new)
        residual = [
            math.fsum(p.component_moles[j] for p in new.phases) - n[j] for j in range(len(n))
        ]
        dv = math.fsum(p.volume_m3 for p in new.phases) - volume
        de = math.fsum(p.total_energy_j for p in new.phases) - energy
        dp = math.fsum(p.momentum_kg_m_s for p in new.phases) - momentum
        if (
            max(map(abs, residual)) > 1e-8
            or abs(dv) > 1e-12 + 1e-9 * volume
            or abs(de) > 1e-6 + 1e-9 * abs(energy)
            or abs(dp) > 1e-9
        ):
            raise ConservationError("Transported N/E/P/V recovery failed")
        return new
