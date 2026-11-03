"""Conservative N/U/V thermodynamic cells, independent of a flow solver.

Explicit accepted boundary fluxes update components and total stored energy;
UV flash recovers equilibrium. No pressure-network/flow closure is invented.
"""

import math
from dataclasses import asdict, dataclass

from .component_energy import ComponentEnergyStream
from .liquid import ConservationError, EquipmentOperatingError, PropertyModelError, _number


@dataclass(frozen=True)
class EquilibriumInventory:
    component_moles: tuple
    internal_energy_j: float
    volume_m3: float
    pressure_pa: float
    temperature_k: float
    model_id: str
    time_s: float = 0.0
    elevation_m: float = 0.0
    gravity_m_s2: float = 9.80665

    def __post_init__(self):
        values = tuple(
            _number(v, "component_moles", nonnegative=True) for v in self.component_moles
        )
        if (
            not 2 <= len(values) <= 12
            or not math.isfinite(math.fsum(values))
            or not math.fsum(values) > 0
        ):
            raise PropertyModelError("Inventory needs positive total component moles")
        object.__setattr__(self, "component_moles", values)
        for name in ("volume_m3", "pressure_pa", "temperature_k"):
            _number(getattr(self, name), name, positive=True)
        _number(self.internal_energy_j, "internal_energy_j")
        _number(self.elevation_m, "elevation_m")
        _number(self.time_s, "time_s", nonnegative=True)
        _number(self.gravity_m_s2, "gravity_m_s2", nonnegative=True)
        if not isinstance(self.model_id, str) or len(self.model_id) != 64:
            raise PropertyModelError("Missing inventory thermodynamic identity")

    def _validate(self, fluid):
        if self.model_id != fluid.model_id or len(self.component_moles) != len(fluid.components):
            raise PropertyModelError("Inventory component/EOS/reference mismatch")

    def mass_kg(self, fluid):
        self._validate(fluid)
        return math.fsum(
            n * c.molar_mass_kg_mol for n, c in zip(self.component_moles, fluid.components)
        )

    def total_energy_j(self, fluid):
        return self.internal_energy_j + self.mass_kg(fluid) * self.gravity_m_s2 * self.elevation_m

    @classmethod
    def from_pt(
        cls,
        fluid,
        component_moles,
        pressure_pa,
        temperature_k,
        *,
        time_s=0.0,
        elevation_m=0.0,
        gravity_m_s2=9.80665,
    ):
        values = tuple(_number(n, "component_moles", nonnegative=True) for n in component_moles)
        total = math.fsum(values)
        if not total > 0:
            raise ValueError("Positive inventory required")
        q = fluid.flash_pt(pressure_pa, temperature_k, [n / total for n in values])
        return cls(
            values,
            total * float(q["molar_internal_energy_j_mol"].cpu()),
            total * float(q["molar_volume_m3_mol"].cpu()),
            pressure_pa,
            temperature_k,
            fluid.model_id,
            time_s,
            elevation_m,
            gravity_m_s2,
        )

    def inspect(self, fluid):
        self._validate(fluid)
        total = math.fsum(self.component_moles)
        try:
            q = fluid.flash_pt(
                self.pressure_pa, self.temperature_k, [n / total for n in self.component_moles]
            )
        except PropertyModelError as exc:
            if "Pure-fluid PT coexistence" not in str(exc):
                raise
            q = fluid.flash_uv(
                self.component_moles,
                self.internal_energy_j,
                self.volume_m3,
                pressure_guess_pa=self.pressure_pa,
                temperature_guess_k=self.temperature_k,
            )
            if (
                abs(float(q["pressure_pa"].cpu()) - self.pressure_pa) > 1e-7 * self.pressure_pa
                or abs(float(q["temperature_k"].cpu()) - self.temperature_k) > 1e-6
            ):
                raise ConservationError("Pure-fluid inventory P/T disagrees with conserved U/V")
        u = total * float(q["molar_internal_energy_j_mol"].cpu())
        v = total * float(q["molar_volume_m3_mol"].cpu())
        if (
            abs(u - self.internal_energy_j) > 1e-6 + 1e-9 * max(1.0, abs(u))
            or abs(v - self.volume_m3) > 1e-12 + 1e-9 * self.volume_m3
        ):
            raise ConservationError("Inventory P/T disagrees with conserved U/V")
        return q

    def phase_stream(self, fluid, phase_index, molar_rate_mol_s):
        q = self.inspect(fluid)
        if (
            isinstance(phase_index, bool)
            or not isinstance(phase_index, int)
            or not 0 <= phase_index < q["phase_count"]
        ):
            raise ValueError("Unknown equilibrium phase")
        rate = _number(molar_rate_mol_s, "molar_rate_mol_s", nonnegative=True)
        phase = q["phases"][phase_index]
        rates = tuple(rate * float(x) for x in phase["mole_fractions"].cpu())
        mass = math.fsum(n * c.molar_mass_kg_mol for n, c in zip(rates, fluid.components))
        energy = (
            rate * float(phase["molar_enthalpy_j_mol"].cpu())
            + mass * self.gravity_m_s2 * self.elevation_m
        )
        return ComponentEnergyStream(
            rates, self.pressure_pa, energy, self.model_id, self.elevation_m, 0.0, self.gravity_m_s2
        )

    def advance_fluxes(
        self,
        fluid,
        duration_s,
        *,
        inputs=(),
        outputs=(),
        heat_w=0.0,
        shaft_power_w=0.0,
        utility_temperature_k=None,
    ):
        """Transactional conservative update from supplied accepted fluxes.

        Explicit thermodynamic cell update, not an implicit dynamic flow solve.
        Flux closure, temporal accuracy and admissible donor extraction belong
        to the calling flow solver. Failed UV flashes publish no partial state.
        """
        old_q = self.inspect(fluid)
        dt = _number(duration_s, "duration_s", positive=True)
        if not math.isfinite(self.time_s + dt) or self.time_s + dt <= self.time_s:
            raise ValueError("Inventory clock cannot advance")
        heat = _number(heat_w, "heat_w")
        work = _number(shaft_power_w, "shaft_power_w")
        if heat and utility_temperature_k is None:
            raise ValueError("Heat flux needs explicit utility temperature")
        utility = (
            _number(utility_temperature_k, "utility_temperature_k", positive=True)
            if utility_temperature_k is not None
            else 1.0
        )
        inputs, outputs = tuple(inputs), tuple(outputs)
        for s in (*inputs, *outputs):
            s.validate_fluid(fluid)
            if s.gravity_m_s2 != self.gravity_m_s2:
                raise PropertyModelError("Inventory/stream gravity references differ")
        dn = [
            dt
            * (
                math.fsum(s.component_molar_rates_mol_s[i] for s in inputs)
                - math.fsum(s.component_molar_rates_mol_s[i] for s in outputs)
            )
            for i in range(len(self.component_moles))
        ]
        new_n = tuple(n + d for n, d in zip(self.component_moles, dn))
        if min(new_n) < 0 or math.fsum(new_n) <= 0:
            raise EquipmentOperatingError("Boundary flux exceeds component inventory")
        de = dt * (
            math.fsum(s.total_energy_rate_w for s in inputs)
            - math.fsum(s.total_energy_rate_w for s in outputs)
            + heat
            + work
        )
        new_mass = math.fsum(n * c.molar_mass_kg_mol for n, c in zip(new_n, fluid.components))
        new_u = self.total_energy_j(fluid) + de - new_mass * self.gravity_m_s2 * self.elevation_m
        q = fluid.flash_uv(
            new_n,
            new_u,
            self.volume_m3,
            pressure_guess_pa=self.pressure_pa,
            temperature_guess_k=self.temperature_k,
        )
        temp = float(q["temperature_k"].cpu())
        if heat * (utility - self.temperature_k) < 0 or heat * (utility - temp) < 0:
            raise EquipmentOperatingError("Heat direction incompatible with utility temperature")
        state = EquilibriumInventory(
            new_n,
            new_u,
            self.volume_m3,
            float(q["pressure_pa"].cpu()),
            temp,
            self.model_id,
            self.time_s + dt,
            self.elevation_m,
            self.gravity_m_s2,
        )
        mass_error = [a - b - d for a, b, d in zip(state.component_moles, self.component_moles, dn)]
        energy_error = state.total_energy_j(fluid) - self.total_energy_j(fluid) - de
        entropy_change = math.fsum(new_n) * float(q["molar_entropy_j_mol_k"].cpu()) - math.fsum(
            self.component_moles
        ) * float(old_q["molar_entropy_j_mol_k"].cpu())
        entropy_flux = 0.0
        for sign, streams in ((1.0, inputs), (-1.0, outputs)):
            for s in streams:
                if s.molar_rate_mol_s:
                    eq = s.equilibrate(fluid, temperature_guess_k=self.temperature_k)
                    entropy_flux += (
                        sign * s.molar_rate_mol_s * float(eq["molar_entropy_j_mol_k"].cpu())
                    )
        production = entropy_change - dt * (entropy_flux + heat / utility)
        if production < -1e-6 - 1e-9 * max(1.0, abs(entropy_change)):
            raise ConservationError("Inventory entropy audit rejected accepted boundary fluxes")
        return {
            "state": state,
            "component_residuals_mol": mass_error,
            "energy_residual_j": energy_error,
            "entropy_production_j_k": production,
            "equilibrium": q,
        }

    def to_dict(self):
        data = asdict(self)
        data["component_moles"] = list(self.component_moles)
        return data

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unexpected equilibrium-inventory fields")
        return cls(**data)
