"""Dry ideal-gas equipment, explicit caloric assumptions and steady energy audits.

No condensation, real-gas Z correction, composition change or surge prediction.
Pressures/temperatures are absolute. Port kinetic energy is neglected for
compressors, coolers and mixers (plenum ports), but retained for pipe momentum.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from scipy.optimize import brentq

from .gas_valve import GasExportValve
from .liquid import EquipmentOperatingError, PropertyModelError, _name, _number


@dataclass(frozen=True)
class IdealGas:
    specific_gas_constant_j_kg_k: float
    heat_capacity_ratio: float
    viscosity_pa_s: float = 1e-5

    def __post_init__(self):
        _number(self.specific_gas_constant_j_kg_k, "specific_gas_constant", positive=True)
        _number(self.heat_capacity_ratio, "heat_capacity_ratio", positive=True)
        _number(self.viscosity_pa_s, "viscosity_pa_s", positive=True)
        if self.heat_capacity_ratio <= 1:
            raise ValueError("Ideal gas heat capacity ratio must exceed 1")

    @property
    def cp_j_kg_k(self):
        k = self.heat_capacity_ratio
        return k / (k - 1) * self.specific_gas_constant_j_kg_k

    def density(self, pressure_pa, temperature_k):
        p = _number(pressure_pa, "pressure_pa", positive=True)
        t = _number(temperature_k, "temperature_k", positive=True)
        return p / (self.specific_gas_constant_j_kg_k * t)


@dataclass(frozen=True)
class GasState:
    mass_rate_kg_s: float
    pressure_pa: float
    temperature_k: float

    def __post_init__(self):
        _number(self.mass_rate_kg_s, "mass_rate_kg_s", nonnegative=True)
        _number(self.pressure_pa, "pressure_pa", positive=True)
        _number(self.temperature_k, "temperature_k", positive=True)

    def volume_rate_m3_s(self, gas):
        return self.mass_rate_kg_s / gas.density(self.pressure_pa, self.temperature_k)


@dataclass(frozen=True)
class GasEquipmentResult:
    name: str
    kind: str
    inlet: GasState
    outlet: GasState
    gas_power_w: float = 0.0
    shaft_power_w: float = 0.0
    drive_power_w: float = 0.0
    rejected_heat_w: float = 0.0
    drive_loss_w: float = 0.0
    energy_residual_w: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class GasCompressor:
    name: str
    outlet_pressure_pa: float
    isentropic_efficiency: float = 0.75
    mechanical_efficiency: float = 0.95
    max_pressure_ratio: float = 4.0
    max_discharge_temperature_k: float = 500.0
    max_drive_power_w: float | None = None
    max_inlet_volume_m3_s: float | None = None

    def __post_init__(self):
        _name(self.name)
        for key in (
            "outlet_pressure_pa",
            "isentropic_efficiency",
            "mechanical_efficiency",
            "max_pressure_ratio",
            "max_discharge_temperature_k",
        ):
            _number(getattr(self, key), key, positive=True)
        if (
            self.isentropic_efficiency > 1
            or self.mechanical_efficiency > 1
            or self.max_pressure_ratio < 1
        ):
            raise ValueError("Invalid compressor efficiency or pressure-ratio bound")
        for key in ("max_drive_power_w", "max_inlet_volume_m3_s"):
            if getattr(self, key) is not None:
                _number(getattr(self, key), key, positive=True)

    def run(self, inlet, gas):
        if self.outlet_pressure_pa < inlet.pressure_pa:
            raise EquipmentOperatingError(f"{self.name}: compressor cannot reduce pressure")
        ratio = self.outlet_pressure_pa / inlet.pressure_pa
        if ratio > self.max_pressure_ratio * (1 + 1e-12):
            raise EquipmentOperatingError(f"{self.name}: pressure ratio exceeds limit")
        exponent = (gas.heat_capacity_ratio - 1) / gas.heat_capacity_ratio
        delta_t = (
            inlet.temperature_k
            * math.expm1(exponent * math.log(ratio))
            / self.isentropic_efficiency
        )
        temperature = inlet.temperature_k + delta_t
        if temperature > self.max_discharge_temperature_k:
            raise EquipmentOperatingError(f"{self.name}: discharge temperature limit")
        gas_power = inlet.mass_rate_kg_s * gas.cp_j_kg_k * delta_t
        shaft = gas_power / self.mechanical_efficiency
        drive = shaft  # Required mechanical drive load; motor electrical losses are not modeled.
        if self.max_drive_power_w is not None and drive > self.max_drive_power_w:
            raise EquipmentOperatingError(f"{self.name}: drive power limit")
        if (
            self.max_inlet_volume_m3_s is not None
            and inlet.volume_rate_m3_s(gas) > self.max_inlet_volume_m3_s
        ):
            raise EquipmentOperatingError(f"{self.name}: inlet volume capacity")
        outlet = GasState(inlet.mass_rate_kg_s, self.outlet_pressure_pa, temperature)
        residual = (
            inlet.mass_rate_kg_s * gas.cp_j_kg_k * (temperature - inlet.temperature_k) - gas_power
        )
        return GasEquipmentResult(
            self.name,
            "adiabatic_ideal_gas_compressor",
            inlet,
            outlet,
            gas_power_w=gas_power,
            shaft_power_w=shaft,
            drive_power_w=drive,
            drive_loss_w=shaft - gas_power,
            energy_residual_w=residual,
        )


@dataclass(frozen=True)
class GasCooler:
    name: str
    outlet_temperature_k: float
    pressure_loss_fraction: float = 0.0
    max_heat_duty_w: float | None = None

    def __post_init__(self):
        _name(self.name)
        _number(self.outlet_temperature_k, "outlet_temperature_k", positive=True)
        loss = _number(self.pressure_loss_fraction, "pressure_loss_fraction", nonnegative=True)
        if loss >= 1:
            raise ValueError("Cooler pressure loss must be < 1")
        if self.max_heat_duty_w is not None:
            _number(self.max_heat_duty_w, "max_heat_duty_w", positive=True)

    def run(self, inlet, gas):
        if self.outlet_temperature_k > inlet.temperature_k:
            raise EquipmentOperatingError(f"{self.name}: cooler cannot heat gas")
        heat = (
            inlet.mass_rate_kg_s * gas.cp_j_kg_k * (inlet.temperature_k - self.outlet_temperature_k)
        )
        if self.max_heat_duty_w is not None and heat > self.max_heat_duty_w:
            raise EquipmentOperatingError(f"{self.name}: cooling duty limit")
        outlet = GasState(
            inlet.mass_rate_kg_s,
            inlet.pressure_pa * (1 - self.pressure_loss_fraction),
            self.outlet_temperature_k,
        )
        residual = (
            inlet.mass_rate_kg_s * gas.cp_j_kg_k * (outlet.temperature_k - inlet.temperature_k)
            + heat
        )
        return GasEquipmentResult(
            self.name,
            "ideal_gas_cooler",
            inlet,
            outlet,
            rejected_heat_w=heat,
            energy_residual_w=residual,
        )


@dataclass(frozen=True)
class GasLetdown:
    name: str
    outlet_pressure_pa: float

    def __post_init__(self):
        _name(self.name)
        _number(self.outlet_pressure_pa, "outlet_pressure_pa", positive=True)

    def run(self, inlet, gas):
        if self.outlet_pressure_pa > inlet.pressure_pa:
            raise EquipmentOperatingError(f"{self.name}: letdown cannot boost pressure")
        outlet = GasState(inlet.mass_rate_kg_s, self.outlet_pressure_pa, inlet.temperature_k)
        return GasEquipmentResult(self.name, "ideal_gas_isenthalpic_letdown", inlet, outlet)


def mix_gas_streams(streams, gas):
    streams = tuple(streams)
    if not streams:
        raise ValueError("Gas mixer requires streams")
    pressure = streams[0].pressure_pa
    if any(not math.isclose(s.pressure_pa, pressure, rel_tol=1e-12, abs_tol=1e-6) for s in streams):
        raise EquipmentOperatingError(
            "Gas mixing requires equal port pressures; condition streams first"
        )
    mass = math.fsum(s.mass_rate_kg_s for s in streams)
    if mass <= 0:
        raise ValueError("Zero-flow gas mixer temperature is undetermined")
    temperature = math.fsum(s.mass_rate_kg_s * s.temperature_k for s in streams) / mass
    return GasState(mass, pressure, temperature)


class GasTrain:
    def __init__(self, equipment):
        self.equipment = tuple(equipment)
        if not self.equipment or not all(
            isinstance(e, (GasCompressor, GasCooler, GasLetdown)) for e in self.equipment
        ):
            raise ValueError("Gas train requires supported equipment")
        if len({e.name for e in self.equipment}) != len(self.equipment):
            raise ValueError("Duplicate gas equipment names")

    def run(self, inlet, gas):
        current = inlet
        results = []
        for equipment in self.equipment:
            result = equipment.run(current, gas)
            results.append(result)
            current = result.outlet
        gas_power = math.fsum(r.gas_power_w for r in results)
        shaft = math.fsum(r.shaft_power_w for r in results)
        heat = math.fsum(r.rejected_heat_w for r in results)
        drive = math.fsum(r.drive_power_w for r in results)
        losses = math.fsum(r.drive_loss_w for r in results)
        residual = (
            inlet.mass_rate_kg_s * gas.cp_j_kg_k * (current.temperature_k - inlet.temperature_k)
            - gas_power
            + heat
        )
        return {
            "inlet": asdict(inlet),
            "outlet": asdict(current),
            "equipment": [r.to_dict() for r in results],
            "gas_power_w": gas_power,
            "shaft_power_w": shaft,
            "drive_power_w": drive,
            "rejected_heat_w": heat,
            "drive_loss_w": losses,
            "energy_residual_w": residual,
            "drive_energy_residual_w": drive - gas_power - losses,
            "mass_residual_kg_s": inlet.mass_rate_kg_s - current.mass_rate_kg_s,
        }


@dataclass(frozen=True)
class IsothermalGasPipe:
    """Horizontal constant-area dry ideal-gas pipe with momentum acceleration.

    Darcy friction (Colebrook turbulent / laminar) is constant along the pipe
    for constant mass flow, T and viscosity. Only the subcritical isothermal
    branch is returned. This is a single edge, not a gas network solver.
    """

    name: str
    length_m: float
    diameter_m: float
    roughness_m: float = 0.0

    def __post_init__(self):
        _name(self.name)
        _number(self.length_m, "length_m", positive=True)
        _number(self.diameter_m, "diameter_m", positive=True)
        _number(self.roughness_m, "roughness_m", nonnegative=True)
        if self.roughness_m / self.diameter_m > 0.05:
            raise ValueError("Gas pipe relative roughness exceeds .05")

    def friction_factor(self, mass_rate_kg_s, gas):
        mass = _number(mass_rate_kg_s, "mass_rate_kg_s", nonnegative=True)
        re = 4 * mass / (math.pi * self.diameter_m * gas.viscosity_pa_s)
        if re == 0:
            return 0.0
        if re <= 2000:
            return 64 / re
        relative = self.roughness_m / self.diameter_m

        def colebrook(r):
            # Solve for 1/sqrt(f); no explicit correlation approximation.
            u = brentq(
                lambda u: u + 2 * math.log10(relative / 3.7 + 2.51 * u / r), 1, 100, xtol=1e-13
            )
            return 1 / u**2

        if re >= 4000:
            return colebrook(re)
        t = (re - 2000) / 2000
        blend = t * t * (3 - 2 * t)
        return (1 - blend) * 64 / re + blend * colebrook(re)

    def run(self, inlet, gas):
        mass = inlet.mass_rate_kg_s
        area = math.pi * self.diameter_m**2 / 4
        rt = gas.specific_gas_constant_j_kg_k * inlet.temperature_k
        flux = mass / area
        a = flux**2 * rt
        f = self.friction_factor(mass, gas)
        if mass == 0:
            return {
                "name": self.name,
                "inlet": asdict(inlet),
                "outlet": asdict(inlet),
                "friction_factor": f,
                "momentum_residual_pa2": 0.0,
                "heat_added_w": 0.0,
                "outlet_mach": 0.0,
                "outlet_isothermal_mach": 0.0,
                "momentum_relative_residual": 0.0,
                "mass_residual_kg_s": 0.0,
            }
        sonic_pressure = math.sqrt(a)
        pin = inlet.pressure_pa
        if pin <= sonic_pressure:
            raise EquipmentOperatingError(
                f"{self.name}: inlet outside subcritical isothermal branch"
            )
        friction_term = f * self.length_m / self.diameter_m * a

        def momentum(pout):
            return (
                (pin - pout) * (pin + pout) + 2 * a * math.log1p((pout - pin) / pin) - friction_term
            )

        if momentum(sonic_pressure) <= 0:
            raise EquipmentOperatingError(
                f"{self.name}: prescribed flow reaches isothermal choking"
            )
        pout = brentq(momentum, sonic_pressure, pin, xtol=1e-8, rtol=1e-14)
        outlet = GasState(mass, pout, inlet.temperature_k)
        vin = flux * rt / pin
        vout = flux * rt / pout
        return {
            "name": self.name,
            "inlet": asdict(inlet),
            "outlet": asdict(outlet),
            "friction_factor": f,
            "momentum_residual_pa2": momentum(pout),
            "heat_added_w": mass * (vout**2 - vin**2) / 2,
            "outlet_mach": vout / math.sqrt(gas.heat_capacity_ratio * rt),
            "outlet_isothermal_mach": vout / math.sqrt(rt),
            "momentum_relative_residual": momentum(pout) / max(pin**2, friction_term, 1.0),
            "mass_residual_kg_s": 0.0,
        }


def intercooled_train(
    name,
    inlet,
    gas,
    outlet_pressure_pa,
    cooling_temperature_k,
    max_stage_pressure_ratio=4.0,
    cooler_pressure_loss_fraction=0.0,
    isentropic_efficiency=0.75,
    mechanical_efficiency=0.95,
    max_discharge_temperature_k=500.0,
    max_stages=20,
):
    """Equal compressor ratios, including every cooler's pressure loss.

    This selects stage count from ratio and temperature bounds, not an economic
    optimization or a compressor performance map/surge-control algorithm.
    """
    _name(name)
    target = _number(outlet_pressure_pa, "outlet_pressure_pa", positive=True)
    cool = _number(cooling_temperature_k, "cooling_temperature_k", positive=True)
    ratio_limit = _number(max_stage_pressure_ratio, "max_stage_pressure_ratio", positive=True)
    loss = _number(cooler_pressure_loss_fraction, "cooler_pressure_loss_fraction", nonnegative=True)
    if ratio_limit <= 1 or loss >= 1 or ratio_limit * (1 - loss) <= 1:
        raise ValueError("Compression ratio must overcome cooler pressure loss")
    if isinstance(max_stages, bool) or not isinstance(max_stages, int) or max_stages < 1:
        raise ValueError("max_stages must be a positive integer")
    if target < inlet.pressure_pa:
        return GasTrain([GasLetdown(f"{name}_letdown", target)])
    if target == inlet.pressure_pa:
        return GasTrain([GasLetdown(f"{name}_bypass", target)])
    count = max(
        1, math.ceil(math.log(target / inlet.pressure_pa) / math.log(ratio_limit * (1 - loss)))
    )
    if count > max_stages:
        raise EquipmentOperatingError("Gas train exceeds maximum stage count")
    eta = _number(isentropic_efficiency, "isentropic_efficiency", positive=True)
    mechanical = _number(mechanical_efficiency, "mechanical_efficiency", positive=True)
    max_temp = _number(max_discharge_temperature_k, "max_discharge_temperature_k", positive=True)
    if eta > 1 or mechanical > 1:
        raise ValueError("Compressor efficiencies must be <= 1")
    for candidate in range(count, max_stages + 1):
        equipment = []
        current = inlet
        for i in range(candidate):
            after = (
                target
                if i == candidate - 1
                else inlet.pressure_pa * (target / inlet.pressure_pa) ** ((i + 1) / candidate)
            )
            discharge = after / (1 - loss)
            exponent = (gas.heat_capacity_ratio - 1) / gas.heat_capacity_ratio
            predicted = current.temperature_k * (
                1 + math.expm1(exponent * math.log(discharge / current.pressure_pa)) / eta
            )
            if predicted > max_temp:
                break
            compressor = GasCompressor(
                f"{name}_compressor_{i + 1}", discharge, eta, mechanical, ratio_limit, max_temp
            )
            compressed = compressor.run(current, gas).outlet
            cooler = GasCooler(f"{name}_cooler_{i + 1}", min(cool, compressed.temperature_k), loss)
            equipment.extend([compressor, cooler])
            current = cooler.run(compressed, gas).outlet
        else:
            return GasTrain(equipment)
    raise EquipmentOperatingError(
        "Gas train cannot meet discharge temperature within maximum stage count"
    )


@dataclass(frozen=True)
class GasExportSystem:
    """Prescribed-flow dry-gas conditioning from each separator to one header.

    PVT phase gas density must agree with the ideal-gas model at source ports.
    A pressure-changing train precedes mixing, never an unaccounted pressure jump.
    """

    header_pressure_pa: float
    inlet_temperature_k: float = 300.0
    cooling_temperature_k: float = 300.0
    standard_pressure_pa: float = 100000.0
    standard_temperature_k: float = 300.0
    heat_capacity_ratio: float = 1.3
    viscosity_pa_s: float = 1e-5
    isentropic_efficiency: float = 0.75
    mechanical_efficiency: float = 0.95
    max_stage_pressure_ratio: float = 4.0
    cooler_pressure_loss_fraction: float = 0.02
    max_discharge_temperature_k: float = 500.0
    max_stages: int = 20
    export_pipe: IsothermalGasPipe | None = None
    export_valve: GasExportValve | None = None

    def __post_init__(self):
        for name in (
            "header_pressure_pa",
            "inlet_temperature_k",
            "cooling_temperature_k",
            "standard_pressure_pa",
            "standard_temperature_k",
            "viscosity_pa_s",
            "max_discharge_temperature_k",
        ):
            _number(getattr(self, name), name, positive=True)
        IdealGas(1.0, self.heat_capacity_ratio, self.viscosity_pa_s)
        for name in ("isentropic_efficiency", "mechanical_efficiency"):
            if not 0 < _number(getattr(self, name), name, positive=True) <= 1:
                raise ValueError("Gas export efficiencies must be <= 1")
        loss = _number(
            self.cooler_pressure_loss_fraction, "cooler_pressure_loss_fraction", nonnegative=True
        )
        ratio = _number(self.max_stage_pressure_ratio, "max_stage_pressure_ratio", positive=True)
        if loss >= 1 or ratio * (1 - loss) <= 1:
            raise ValueError("Gas export stage ratio cannot overcome cooler losses")
        if (
            isinstance(self.max_stages, bool)
            or not isinstance(self.max_stages, int)
            or self.max_stages < 1
        ):
            raise ValueError("max_stages must be a positive integer")
        if isinstance(self.export_valve, dict):
            object.__setattr__(self, "export_valve", GasExportValve(**self.export_valve))
        if self.export_valve is not None and not isinstance(self.export_valve, GasExportValve):
            raise TypeError("export_valve must be GasExportValve")
        if isinstance(self.export_pipe, dict):
            object.__setattr__(self, "export_pipe", IsothermalGasPipe(**self.export_pipe))
        if self.export_pipe is not None and not isinstance(self.export_pipe, IsothermalGasPipe):
            raise TypeError("export_pipe must be IsothermalGasPipe")

    def run(self, separator_results, densities):
        gas = IdealGas(
            self.standard_pressure_pa / (densities.gas_kg_m3 * self.standard_temperature_k),
            self.heat_capacity_ratio,
            self.viscosity_pa_s,
        )
        routes = []
        header_inputs = []
        source_enthalpies = []
        for stage in separator_results:
            stream = stage.gas
            if stream.oil_m3_s or stream.water_m3_s:
                raise EquipmentOperatingError(
                    "Gas export requires dry gas without oil/water carryover"
                )
            mass = stream.total_mass_rate(densities)
            if mass == 0:
                routes.append({"source": stage.name, "status": "no_flow", "mass_rate_kg_s": 0.0})
                continue
            inlet = GasState(mass, stage.pressure_pa, self.inlet_temperature_k)
            reference_density = stage.phase_density_kg_m3["gas"]
            if not math.isclose(
                reference_density, gas.density(inlet.pressure_pa, inlet.temperature_k), rel_tol=1e-6
            ):
                raise PropertyModelError(
                    f"{stage.name}: PVT/ideal-gas density mismatch; real-gas model required"
                )
            train = intercooled_train(
                stage.name,
                inlet,
                gas,
                self.header_pressure_pa,
                self.cooling_temperature_k,
                self.max_stage_pressure_ratio,
                self.cooler_pressure_loss_fraction,
                self.isentropic_efficiency,
                self.mechanical_efficiency,
                self.max_discharge_temperature_k,
                self.max_stages,
            )
            result = train.run(inlet, gas)
            routes.append({"source": stage.name, "status": "flowing", **result})
            header_inputs.append(GasState(**result["outlet"]))
            source_enthalpies.append(mass * gas.cp_j_kg_k * inlet.temperature_k)
        active = [r for r in routes if r["status"] == "flowing"]
        gas_power = math.fsum(r["gas_power_w"] for r in active)
        shaft = math.fsum(r["shaft_power_w"] for r in active)
        drive = math.fsum(r["drive_power_w"] for r in active)
        losses = math.fsum(r["drive_loss_w"] for r in active)
        heat = math.fsum(r["rejected_heat_w"] for r in active)
        if not header_inputs:
            return {
                "model": "dry_ideal_gas_prescribed_flow",
                "routes": routes,
                "header": None,
                "outlet": None,
                "pipe": None,
                "valve": None,
                "mass_rate_kg_s": 0.0,
                "gas_power_w": 0.0,
                "shaft_power_w": 0.0,
                "drive_power_w": 0.0,
                "rejected_heat_w": 0.0,
                "drive_loss_w": 0.0,
                "mass_residual_kg_s": 0.0,
                "energy_residual_w": 0.0,
                "drive_energy_residual_w": 0.0,
            }
        header = mix_gas_streams(header_inputs, gas)
        pipe_result = self.export_pipe.run(header, gas) if self.export_pipe is not None else None
        outlet = GasState(**pipe_result["outlet"]) if pipe_result else header
        valve_result = self.export_valve.run(outlet, gas) if self.export_valve else None
        if valve_result is not None:
            outlet = GasState(**valve_result["outlet"])
        energy = (
            outlet.mass_rate_kg_s * gas.cp_j_kg_k * outlet.temperature_k
            - math.fsum(source_enthalpies)
            - gas_power
            + heat
        )
        return {
            "model": "dry_ideal_gas_prescribed_flow",
            "routes": routes,
            "header": asdict(header),
            "outlet": asdict(outlet),
            "pipe": pipe_result,
            "valve": valve_result,
            "gas_properties": asdict(gas),
            "mass_rate_kg_s": outlet.mass_rate_kg_s,
            "gas_power_w": gas_power,
            "shaft_power_w": shaft,
            "drive_power_w": drive,
            "drive_loss_w": losses,
            "rejected_heat_w": heat,
            "mass_residual_kg_s": math.fsum(s.mass_rate_kg_s for s in header_inputs)
            - outlet.mass_rate_kg_s,
            # Pipe T is fixed: its kinetic-energy increase is exactly its external heat input.
            "energy_residual_w": energy,
            "drive_energy_residual_w": drive - gas_power - losses,
        }
