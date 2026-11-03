"""Conserved component/total-energy boundary shared by thermodynamic models.

All components have explicit identities. Stream energy includes enthalpy,
kinetic and gravitational energy. Stock assays are explicit characterization,
never inferred from PVT, densities or phase labels.
"""

import math
from dataclasses import asdict, dataclass

import torch

from .equilibrium_properties import EquilibriumFluid
from .liquid import ConservationError, EquipmentOperatingError, PropertyModelError, _number
from .operation import _digest


@dataclass(frozen=True)
class ComponentEnergyStream:
    component_molar_rates_mol_s: tuple
    pressure_pa: float
    total_energy_rate_w: float
    model_id: str
    elevation_m: float = 0.0
    velocity_m_s: float = 0.0
    gravity_m_s2: float = 9.80665

    def __post_init__(self):
        rates = tuple(
            _number(x, "component_molar_rate", nonnegative=True)
            for x in self.component_molar_rates_mol_s
        )
        if not 2 <= len(rates) <= 12 or not math.isfinite(math.fsum(rates)):
            raise PropertyModelError("Invalid component-rate registry")
        object.__setattr__(self, "component_molar_rates_mol_s", rates)
        _number(self.pressure_pa, "pressure_pa", positive=True)
        _number(self.total_energy_rate_w, "total_energy_rate_w")
        _number(self.elevation_m, "elevation_m")
        _number(self.velocity_m_s, "velocity_m_s", nonnegative=True)
        _number(self.gravity_m_s2, "gravity_m_s2", nonnegative=True)
        if not any(rates) and self.total_energy_rate_w != 0:
            raise ConservationError("Zero material flow cannot carry advective energy")
        if (
            not isinstance(self.model_id, str)
            or len(self.model_id) != 64
            or any(c not in "0123456789abcdef" for c in self.model_id)
        ):
            raise PropertyModelError("Missing thermodynamic component/reference identity")

    def validate_fluid(self, fluid):
        if not isinstance(fluid, EquilibriumFluid):
            raise TypeError("Expected EquilibriumFluid")
        if self.model_id != fluid.model_id or len(self.component_molar_rates_mol_s) != len(
            fluid.components
        ):
            raise PropertyModelError("Component registry/EOS/caloric reference mismatch")
        if not fluid.min_pressure_pa <= self.pressure_pa <= fluid.max_pressure_pa:
            raise PropertyModelError("Stream pressure outside EOS domain")

    @property
    def molar_rate_mol_s(self):
        return math.fsum(self.component_molar_rates_mol_s)

    def mass_rate_kg_s(self, fluid):
        self.validate_fluid(fluid)
        return math.fsum(
            n * c.molar_mass_kg_mol
            for n, c in zip(self.component_molar_rates_mol_s, fluid.components)
        )

    def enthalpy_rate_w(self, fluid):
        return self.total_energy_rate_w - self.mass_rate_kg_s(fluid) * (
            self.gravity_m_s2 * self.elevation_m + 0.5 * self.velocity_m_s**2
        )

    @classmethod
    def from_pt(
        cls,
        fluid,
        component_molar_rates_mol_s,
        pressure_pa,
        temperature_k,
        *,
        elevation_m=0.0,
        velocity_m_s=0.0,
        gravity_m_s2=9.80665,
    ):
        # Validate the packet before an EOS solve (negative rates are not guesses).
        candidate = cls(
            tuple(component_molar_rates_mol_s),
            pressure_pa,
            0.0,
            fluid.model_id,
            elevation_m,
            velocity_m_s,
            gravity_m_s2,
        )
        candidate.validate_fluid(fluid)
        total = candidate.molar_rate_mol_s
        z = (
            [n / total for n in candidate.component_molar_rates_mol_s]
            if total
            else [1 / len(fluid.components)] * len(fluid.components)
        )
        if not total:
            fluid._validate(fluid.tensor(pressure_pa), fluid.tensor(temperature_k), fluid.tensor(z))
            return candidate
        q = fluid.flash_pt(pressure_pa, temperature_k, z)
        h = total * float(q["molar_enthalpy_j_mol"].cpu())
        energy = h + candidate.mass_rate_kg_s(fluid) * (
            gravity_m_s2 * elevation_m + 0.5 * velocity_m_s**2
        )
        return cls(
            candidate.component_molar_rates_mol_s,
            pressure_pa,
            energy,
            fluid.model_id,
            elevation_m,
            velocity_m_s,
            gravity_m_s2,
        )

    def equilibrate(self, fluid, *, temperature_guess_k=None):
        self.validate_fluid(fluid)
        total = self.molar_rate_mol_s
        if not total:
            return None
        return fluid.flash_ph(
            self.pressure_pa,
            self.enthalpy_rate_w(fluid) / total,
            [n / total for n in self.component_molar_rates_mol_s],
            temperature_guess_k,
        )

    def transport(
        self,
        fluid,
        pressure_pa,
        *,
        elevation_m=None,
        velocity_m_s=None,
        heat_w=0.0,
        shaft_power_w=0.0,
        utility_temperature_k=None,
        temperature_guess_k=None,
    ):
        """Conservative transfer, positive Q and W enter the fluid.

        Pressure is a boundary condition, not a claimed hydraulic solution.
        Adiabatic zero-work upward pressure changes are refused.
        """
        self.validate_fluid(fluid)
        p = _number(pressure_pa, "pressure_pa", positive=True)
        q, w = _number(heat_w, "heat_w"), _number(shaft_power_w, "shaft_power_w")
        if p > self.pressure_pa and w <= 0:
            raise EquipmentOperatingError("Pressure raising requires explicit positive shaft work")
        out = ComponentEnergyStream(
            self.component_molar_rates_mol_s,
            p,
            self.total_energy_rate_w + q + w,
            self.model_id,
            self.elevation_m if elevation_m is None else elevation_m,
            self.velocity_m_s if velocity_m_s is None else velocity_m_s,
            self.gravity_m_s2,
        )
        old_q = self.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
        new_q = out.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
        if q and utility_temperature_k is None:
            raise ValueError("Heat transfer requires explicit utility temperature")
        utility = (
            _number(utility_temperature_k, "utility_temperature_k", positive=True)
            if utility_temperature_k is not None
            else 1.0
        )
        if old_q is not None:
            ti, tf = float(old_q["temperature_k"].cpu()), float(new_q["temperature_k"].cpu())
            if q * (utility - ti) < 0 or q * (utility - tf) < 0:
                raise EquipmentOperatingError(
                    "Heat direction incompatible with utility temperature"
                )
            production = (
                self.molar_rate_mol_s
                * float((new_q["molar_entropy_j_mol_k"] - old_q["molar_entropy_j_mol_k"]).cpu())
                - q / utility
            )
            if production < -1e-6:
                raise ConservationError(
                    "Transfer violates entropy balance; insufficient work or invalid heat duty"
                )
        return out

    @classmethod
    def from_thermal_stream(
        cls, stream, fluid, *, elevation_m=0.0, velocity_m_s=0.0, gravity_m_s2=9.80665
    ):
        """Lossless bridge for the existing explicit-composition PR contract."""
        from .peng_robinson_mixture import PengRobinsonMixture
        from .thermal_process import ThermalStream

        if not isinstance(stream, ThermalStream):
            raise TypeError("Expected legacy ThermalStream")
        reference = PengRobinsonMixture(fluid.components, fluid.binary_interactions)
        stream._validate_fluid(reference)
        mass = math.fsum(
            n * c.molar_mass_kg_mol
            for n, c in zip(stream.component_molar_rates_mol_s, fluid.components)
        )
        result = cls(
            stream.component_molar_rates_mol_s,
            stream.pressure_pa,
            stream.enthalpy_rate_w + mass * (gravity_m_s2 * elevation_m + 0.5 * velocity_m_s**2),
            fluid.model_id,
            elevation_m,
            velocity_m_s,
            gravity_m_s2,
        )
        result.validate_fluid(fluid)
        return result

    def to_thermal_stream(self, fluid, *, temperature_guess_k=None):
        """Legacy PR VLE has no three-phase/zero-species representation."""
        from .peng_robinson_mixture import PengRobinsonMixture
        from .thermal_process import ThermalStream

        self.validate_fluid(fluid)
        q = self.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
        if q is not None and q["phase_count"] > 2:
            raise PropertyModelError(
                "Legacy ThermalStream backend cannot represent three-phase equilibrium"
            )
        reference = PengRobinsonMixture(fluid.components, fluid.binary_interactions)
        return ThermalStream(
            self.component_molar_rates_mol_s,
            self.pressure_pa,
            self.enthalpy_rate_w(fluid) / self.molar_rate_mol_s if self.molar_rate_mol_s else 0.0,
            _digest(asdict(reference)),
        )

    def split(self, fraction):
        f = _number(fraction, "fraction", nonnegative=True)
        if f > 1:
            raise ValueError("Split fraction exceeds one")
        return tuple(
            ComponentEnergyStream(
                tuple(a * n for n in self.component_molar_rates_mol_s),
                self.pressure_pa,
                a * self.total_energy_rate_w,
                self.model_id,
                self.elevation_m,
                self.velocity_m_s,
                self.gravity_m_s2,
            )
            for a in (f, 1 - f)
        )

    def oriented_flux(self, fluid, orientation=1):
        self.validate_fluid(fluid)
        if isinstance(orientation, bool) or orientation not in (-1, 1):
            raise ValueError("Orientation must be +1 or -1")
        return (
            fluid.tensor((*self.component_molar_rates_mol_s, self.total_energy_rate_w))
            * orientation
        )

    def to_dict(self):
        result = asdict(self)
        result["component_molar_rates_mol_s"] = list(self.component_molar_rates_mol_s)
        return result

    @classmethod
    def from_dict(cls, data):
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("Unexpected component-energy stream fields")
        return cls(**data)


def separate_equilibrium_stream(stream, fluid, *, temperature_guess_k=None):
    """Adiabatic equilibrium products with the same pressure and energy datum."""
    q = stream.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
    if q is None:
        return {
            "outputs": (),
            "equilibrium": None,
            "component_residuals_mol_s": [0.0] * len(fluid.components),
            "energy_residual_w": 0.0,
        }
    outputs = []
    for phase in q["phases"]:
        amount = stream.molar_rate_mol_s * float(phase["molar_fraction"].cpu())
        rates = tuple(amount * float(x) for x in phase["mole_fractions"].cpu())
        mass = math.fsum(n * c.molar_mass_kg_mol for n, c in zip(rates, fluid.components))
        energy = amount * float(phase["molar_enthalpy_j_mol"].cpu()) + mass * (
            stream.gravity_m_s2 * stream.elevation_m + 0.5 * stream.velocity_m_s**2
        )
        outputs.append(
            ComponentEnergyStream(
                rates,
                stream.pressure_pa,
                energy,
                stream.model_id,
                stream.elevation_m,
                stream.velocity_m_s,
                stream.gravity_m_s2,
            )
        )
    audit = audit_component_energy((stream,), outputs, fluid)
    return {"outputs": tuple(outputs), "equilibrium": q, **audit}


def audit_component_energy(inputs, outputs, fluid, *, heat_w=0.0, shaft_power_w=0.0):
    inputs, outputs = tuple(inputs), tuple(outputs)
    for stream in (*inputs, *outputs):
        stream.validate_fluid(fluid)
    residual = [
        math.fsum(s.component_molar_rates_mol_s[i] for s in outputs)
        - math.fsum(s.component_molar_rates_mol_s[i] for s in inputs)
        for i in range(len(fluid.components))
    ]
    energy = (
        math.fsum(s.total_energy_rate_w for s in outputs)
        - math.fsum(s.total_energy_rate_w for s in inputs)
        - _number(heat_w, "heat_w")
        - _number(shaft_power_w, "shaft_power_w")
    )
    scale = max(
        1.0,
        math.fsum(abs(s.total_energy_rate_w) for s in (*inputs, *outputs)),
        abs(heat_w),
        abs(shaft_power_w),
    )
    mass_scale = max(1.0, math.fsum(s.molar_rate_mol_s for s in inputs))
    if max(map(abs, residual)) > 1e-10 * mass_scale or abs(energy) > 1e-6 + 1e-9 * scale:
        raise ConservationError("Component/total-energy transfer audit failed")
    return {"component_residuals_mol_s": residual, "energy_residual_w": energy}


def mix_component_energy_streams(
    streams, fluid, pressure_pa, *, elevation_m=0.0, velocity_m_s=0.0, temperature_guess_k=None
):
    streams = tuple(streams)
    if not streams:
        raise ValueError("Mixer needs at least one stream")
    p = _number(pressure_pa, "pressure_pa", positive=True)
    for stream in streams:
        stream.validate_fluid(fluid)
        if stream.molar_rate_mol_s and stream.pressure_pa < p:
            raise EquipmentOperatingError("Passive mixing pressure exceeds a feed pressure")
        if stream.gravity_m_s2 != streams[0].gravity_m_s2:
            raise PropertyModelError("Gravity references differ")
    rates = tuple(
        math.fsum(s.component_molar_rates_mol_s[i] for s in streams)
        for i in range(len(fluid.components))
    )
    out = ComponentEnergyStream(
        rates,
        p,
        math.fsum(s.total_energy_rate_w for s in streams),
        fluid.model_id,
        elevation_m,
        velocity_m_s,
        streams[0].gravity_m_s2,
    )
    final_q = out.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
    incoming_entropy = []
    for stream in streams:
        if stream.molar_rate_mol_s:
            q = stream.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
            incoming_entropy.append(
                stream.molar_rate_mol_s * float(q["molar_entropy_j_mol_k"].cpu())
            )
    outgoing_entropy = (
        out.molar_rate_mol_s * float(final_q["molar_entropy_j_mol_k"].cpu()) if final_q else 0.0
    )
    if outgoing_entropy - math.fsum(incoming_entropy) < -1e-6:
        raise ConservationError("Passive mixer violates entropy balance")
    audit_component_energy(streams, (out,), fluid)
    return out


@dataclass(frozen=True)
class StockComponentAssay:
    """Explicit kg-species/kg-stock fractions for oil, water, gas stock groups.

    A characterization, not information recoverable from a black-oil PVT table.
    Reservoir energy is NEVER synthesized silently from isothermal FIM output.
    """

    component_names: tuple
    stock_mass_fractions: tuple
    fluid_id: str

    def __post_init__(self):
        names = tuple(self.component_names)
        if (
            len(set(names)) != len(names)
            or not 2 <= len(names) <= 12
            or not all(isinstance(n, str) and n for n in names)
        ):
            raise ValueError("Invalid assay component registry")
        rows = tuple(
            tuple(_number(v, "assay_fraction", nonnegative=True) for v in row)
            for row in self.stock_mass_fractions
        )
        if len(rows) != 3 or any(
            len(row) != len(names) or abs(math.fsum(row) - 1) > 1e-12 for row in rows
        ):
            raise PropertyModelError("Three oil/water/gas mass-fraction rows must each sum to one")
        if not isinstance(self.fluid_id, str) or len(self.fluid_id) != 64:
            raise PropertyModelError("Assay must bind a thermodynamic model")
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "stock_mass_fractions", rows)

    @property
    def model_id(self):
        return _digest(asdict(self))

    def component_molar_rates(self, stock_component_mass_rates_kg_s, fluid):
        if self.fluid_id != fluid.model_id or self.component_names != fluid.names:
            raise PropertyModelError("Assay component order/model mismatch")
        stock = fluid.tensor(stock_component_mass_rates_kg_s)
        if stock.shape != (3,) or not bool(
            (torch.isfinite(stock).all() & (stock >= 0).all()).cpu()
        ):
            raise PropertyModelError(
                "Assay expects nonnegative stock-component masses, not phase masses"
            )
        masses = stock @ fluid.tensor(self.stock_mass_fractions)
        if not bool(((masses.sum() - stock.sum()).abs() <= 1e-10 * stock.sum().clamp_min(1)).cpu()):
            raise ConservationError("Stock assay mass balance failed")
        return masses / fluid.molar_masses

    def from_energy_boundary(
        self,
        stock_component_mass_rates_kg_s,
        fluid,
        pressure_pa,
        total_energy_rate_w,
        *,
        elevation_m=0.0,
        velocity_m_s=0.0,
        temperature_guess_k=None,
    ):
        rates = self.component_molar_rates(stock_component_mass_rates_kg_s, fluid)
        stream = ComponentEnergyStream(
            tuple(rates.cpu().tolist()),
            pressure_pa,
            total_energy_rate_w,
            fluid.model_id,
            elevation_m,
            velocity_m_s,
        )
        stream.equilibrate(fluid, temperature_guess_k=temperature_guess_k)
        return stream

    def from_isothermal_boundary(
        self,
        stock_component_mass_rates_kg_s,
        fluid,
        pressure_pa,
        temperature_k,
        *,
        elevation_m=0.0,
        velocity_m_s=0.0,
    ):
        rates = self.component_molar_rates(stock_component_mass_rates_kg_s, fluid)
        stream = ComponentEnergyStream.from_pt(
            fluid,
            rates.cpu().tolist(),
            pressure_pa,
            temperature_k,
            elevation_m=elevation_m,
            velocity_m_s=velocity_m_s,
        )
        return {
            "stream": stream,
            "boundary_type": "prescribed_temperature_external_energy",
            "energy_supplied_by_thermal_boundary_w": stream.total_energy_rate_w,
            "reservoir_energy_balance_available": False,
            "assay_id": self.model_id,
        }
