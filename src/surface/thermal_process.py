"""Conservative explicit-composition steady thermal unit operations.

Same PR/PH/PS reference as the nozzle. Stream basis is mol/s per component;
energy flux is total molar rate times molar enthalpy. No black-oil conversion,
water phase, inventories, machine map, or pressure-network solve is inferred.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from .liquid import ConservationError, EquipmentOperatingError, PropertyModelError, _name, _number
from .operation import _digest
from .peng_robinson_mixture import PengRobinsonMixture


def _fluid(fluid):
    if not isinstance(fluid, PengRobinsonMixture):
        raise TypeError("Thermal process requires explicit PengRobinsonMixture")


@dataclass(frozen=True)
class ThermalStream:
    component_molar_rates_mol_s: tuple
    pressure_pa: float
    molar_enthalpy_j_mol: float
    model_id: str

    def __post_init__(self):
        rates = tuple(
            _number(v, "component_molar_rate", nonnegative=True)
            for v in self.component_molar_rates_mol_s
        )
        if not 2 <= len(rates) <= 12:
            raise PropertyModelError("Stream requires 2..12 component rates")
        if any(rates) and not all(v > 0 for v in rates):
            raise PropertyModelError(
                "Remove absent species from the model; mixed zero rates unsupported"
            )
        object.__setattr__(self, "component_molar_rates_mol_s", rates)
        if (
            not isinstance(self.model_id, str)
            or len(self.model_id) != 64
            or any(c not in "0123456789abcdef" for c in self.model_id)
        ):
            raise PropertyModelError("Stream requires a valid model fingerprint")
        _number(self.pressure_pa, "pressure_pa", positive=True)
        _number(self.molar_enthalpy_j_mol, "molar_enthalpy_j_mol")
        if not math.isfinite(sum(rates)):
            raise PropertyModelError("Nonfinite total molar rate")

    @property
    def molar_rate_mol_s(self):
        return math.fsum(self.component_molar_rates_mol_s)

    @property
    def enthalpy_rate_w(self):
        return self.molar_rate_mol_s * self.molar_enthalpy_j_mol

    @property
    def mole_fractions(self):
        n = self.molar_rate_mol_s
        return tuple(x / n for x in self.component_molar_rates_mol_s) if n else None

    def _validate_fluid(self, fluid):
        _fluid(fluid)
        if self.model_id != _digest(asdict(fluid)):
            raise PropertyModelError("Stream thermodynamic model/order/reference mismatch")
        if len(self.component_molar_rates_mol_s) != len(fluid.components):
            raise PropertyModelError("Stream component count differs from model")

    def state(self, fluid):
        self._validate_fluid(fluid)
        return (
            fluid.flash_ph(self.pressure_pa, self.molar_enthalpy_j_mol, self.mole_fractions)
            if self.molar_rate_mol_s
            else None
        )

    def component_mass_rates(self, fluid):
        self._validate_fluid(fluid)
        return tuple(
            n * c.molar_mass_kg_mol
            for n, c in zip(self.component_molar_rates_mol_s, fluid.components)
        )

    def to_dict(self):
        result = asdict(self)
        result["component_molar_rates_mol_s"] = list(self.component_molar_rates_mol_s)
        return result

    @classmethod
    def from_pt(cls, fluid, pressure_pa, temperature_k, molar_rate_mol_s, mole_fractions):
        _fluid(fluid)
        n = _number(molar_rate_mol_s, "molar_rate_mol_s", nonnegative=True)
        z = fluid._composition(mole_fractions)
        for component in fluid.components:
            component._validate(pressure_pa, temperature_k)
        h = fluid.flash_pt(pressure_pa, temperature_k, z)["molar_enthalpy_j_mol"] if n else 0.0
        return cls(tuple(n * z), pressure_pa, h, _digest(asdict(fluid)))

    def split(self, fraction):
        f = _number(fraction, "fraction", nonnegative=True)
        if f > 1:
            raise ValueError("Split fraction must not exceed one")
        return tuple(
            ThermalStream(
                tuple(n * a for n in self.component_molar_rates_mol_s),
                self.pressure_pa,
                self.molar_enthalpy_j_mol,
                self.model_id,
            )
            for a in (f, 1 - f)
        )


def _result(
    inputs,
    outputs,
    fluid,
    heat_w=0.0,
    shaft_power_w=0.0,
    utility_temperature_k=None,
    heat_entropy_transfer_w_k=None,
):
    """Common checked mass/energy/entropy accounting; positive Q/W enter fluid."""
    for stream in (*inputs, *outputs):
        stream._validate_fluid(fluid)
    incoming = np.sum([s.component_molar_rates_mol_s for s in inputs], axis=0)
    outgoing = np.sum([s.component_molar_rates_mol_s for s in outputs], axis=0)
    residual = incoming - outgoing
    energy = (
        math.fsum(s.enthalpy_rate_w for s in outputs)
        - math.fsum(s.enthalpy_rate_w for s in inputs)
        - heat_w
        - shaft_power_w
    )
    scale = max(1.0, float(np.max(np.abs(incoming))))
    energy_scale = max(
        1.0, abs(heat_w), abs(shaft_power_w), *(abs(s.enthalpy_rate_w) for s in (*inputs, *outputs))
    )
    if np.max(np.abs(residual)) > scale * 1e-9 or abs(energy) > 1e-6 + 1e-9 * energy_scale:
        raise ConservationError("Thermal operation violates component or energy balance")

    def entropy_rate(stream):
        state = stream.state(fluid)
        return stream.molar_rate_mol_s * state["molar_entropy_j_mol_k"] if state else 0.0

    entropy = math.fsum(entropy_rate(s) for s in outputs) - math.fsum(
        entropy_rate(s) for s in inputs
    )
    if heat_entropy_transfer_w_k is not None:
        entropy -= heat_entropy_transfer_w_k
    elif heat_w:
        if utility_temperature_k is None:
            raise PropertyModelError("Nonzero heat needs an explicit utility temperature")
        entropy -= heat_w / utility_temperature_k
    if not all(math.isfinite(v) for v in [energy, entropy, heat_w, shaft_power_w]):
        raise ConservationError("Nonfinite thermal balance")
    if entropy < -1e-6:
        raise ConservationError("Thermal operation has negative entropy generation")
    return {
        "outputs": outputs,
        "heat_duty_w": heat_w,
        "shaft_power_w": shaft_power_w,
        "energy_residual_w": energy,
        "component_molar_residuals_mol_s": residual.tolist(),
        "entropy_generation_w_k": entropy,
        "heat_entropy_transfer_w_k": (
            heat_entropy_transfer_w_k
            if heat_entropy_transfer_w_k is not None
            else heat_w / utility_temperature_k
            if heat_w
            else 0.0
        ),
    }


def mix_thermal_streams(streams, fluid):
    streams = tuple(streams)
    if not streams:
        raise ValueError("Mixer requires at least one stream")
    for s in streams:
        s._validate_fluid(fluid)
    p = streams[0].pressure_pa
    if any(s.pressure_pa != p for s in streams):
        raise PropertyModelError("Mixing needs equal declared pressures; let down first")
    rates = tuple(
        math.fsum(s.component_molar_rates_mol_s[i] for s in streams)
        for i in range(len(fluid.components))
    )
    n = math.fsum(rates)
    h = math.fsum(s.enthalpy_rate_w for s in streams) / n if n else 0.0
    out = ThermalStream(rates, p, h, streams[0].model_id)
    return _result(streams, (out,), fluid)


@dataclass(frozen=True)
class MixtureHeatExchanger:
    name: str
    outlet_temperature_k: float
    utility_temperature_k: float
    pressure_drop_pa: float = 0.0
    max_abs_duty_w: float = None

    def __post_init__(self):
        _name(self.name)
        for k in ("outlet_temperature_k", "utility_temperature_k"):
            _number(getattr(self, k), k, positive=True)
        _number(self.pressure_drop_pa, "pressure_drop_pa", nonnegative=True)
        if self.max_abs_duty_w is not None:
            _number(self.max_abs_duty_w, "max_abs_duty_w", positive=True)

    def run(self, stream, fluid):
        stream._validate_fluid(fluid)
        p = stream.pressure_pa - self.pressure_drop_pa
        _number(p, "outlet_pressure_pa", positive=True)
        if not stream.molar_rate_mol_s:
            return _result(
                (stream,),
                (ThermalStream(stream.component_molar_rates_mol_s, p, 0.0, stream.model_id),),
                fluid,
            )
        inlet = stream.state(fluid)
        outstate = fluid.flash_pt(p, self.outlet_temperature_k, stream.mole_fractions)
        out = ThermalStream(
            stream.component_molar_rates_mol_s, p, outstate["molar_enthalpy_j_mol"], stream.model_id
        )
        heat = out.enthalpy_rate_w - stream.enthalpy_rate_w
        lo, hi = sorted([inlet["temperature_k"], self.outlet_temperature_k])
        if (
            heat > 1e-8
            and self.utility_temperature_k < hi
            or heat < -1e-8
            and self.utility_temperature_k > lo
        ):
            raise EquipmentOperatingError(
                "Utility cannot cross the specified stream endpoint temperature"
            )
        if self.max_abs_duty_w is not None and abs(heat) > self.max_abs_duty_w:
            raise EquipmentOperatingError("Heat duty exceeds equipment limit")
        return _result(
            (stream,), (out,), fluid, heat_w=heat, utility_temperature_k=self.utility_temperature_k
        )


@dataclass(frozen=True)
class MixturePressureRaiser:
    """Single-phase thermodynamic work model, not a calibrated compressor map."""

    name: str
    outlet_pressure_pa: float
    isentropic_efficiency: float
    max_power_w: float = None
    max_outlet_temperature_k: float = None

    def __post_init__(self):
        _name(self.name)
        _number(self.outlet_pressure_pa, "outlet_pressure_pa", positive=True)
        e = _number(self.isentropic_efficiency, "isentropic_efficiency", positive=True)
        if e > 1:
            raise ValueError("Isentropic efficiency must not exceed one")
        for k in ("max_power_w", "max_outlet_temperature_k"):
            if getattr(self, k) is not None:
                _number(getattr(self, k), k, positive=True)

    def run(self, stream, fluid):
        stream._validate_fluid(fluid)
        if self.outlet_pressure_pa < stream.pressure_pa:
            raise PropertyModelError("Pressure raiser cannot expand the stream")
        if not stream.molar_rate_mol_s:
            return _result(
                (stream,),
                (
                    ThermalStream(
                        stream.component_molar_rates_mol_s,
                        self.outlet_pressure_pa,
                        0.0,
                        stream.model_id,
                    ),
                ),
                fluid,
            )
        inlet = stream.state(fluid)
        if inlet["phase_count"] != 1:
            raise EquipmentOperatingError("Pressure raiser rejects two-phase inlet")
        ideal = fluid.flash_ps(
            self.outlet_pressure_pa, inlet["molar_entropy_j_mol_k"], stream.mole_fractions
        )
        if ideal["phase_count"] != 1:
            raise EquipmentOperatingError("Pressure raiser rejects two-phase isentropic endpoint")
        gain = ideal["molar_enthalpy_j_mol"] - stream.molar_enthalpy_j_mol
        if gain < -1e-6:
            raise ConservationError("Isentropic compression decreases enthalpy")
        h = stream.molar_enthalpy_j_mol + max(0.0, gain) / self.isentropic_efficiency
        out = ThermalStream(
            stream.component_molar_rates_mol_s, self.outlet_pressure_pa, h, stream.model_id
        )
        actual = out.state(fluid)
        if actual["phase_count"] != 1:
            raise EquipmentOperatingError("Pressure raiser rejects two-phase outlet")
        power = out.enthalpy_rate_w - stream.enthalpy_rate_w
        if self.max_power_w is not None and power > self.max_power_w:
            raise EquipmentOperatingError("Shaft power exceeds equipment limit")
        if (
            self.max_outlet_temperature_k is not None
            and actual["temperature_k"] > self.max_outlet_temperature_k
        ):
            raise EquipmentOperatingError("Outlet temperature exceeds equipment limit")
        return _result((stream,), (out,), fluid, shaft_power_w=power)


@dataclass(frozen=True)
class MixtureFlashSeparator:
    """Adiabatic PH vessel; no sizing, entrainment, levels or residence kinetics."""

    name: str
    pressure_pa: float
    single_phase_outlet: int = None

    def __post_init__(self):
        _name(self.name)
        _number(self.pressure_pa, "pressure_pa", positive=True)
        if self.single_phase_outlet is not None and (
            isinstance(self.single_phase_outlet, bool) or self.single_phase_outlet not in (0, 1)
        ):
            raise ValueError("single_phase_outlet must be explicitly 0 or 1")

    def run(self, stream, fluid):
        stream._validate_fluid(fluid)
        if self.pressure_pa > stream.pressure_pa:
            raise PropertyModelError("Separator cannot boost pressure")
        if not stream.molar_rate_mol_s:
            empty = ThermalStream(
                stream.component_molar_rates_mol_s, self.pressure_pa, 0.0, stream.model_id
            )
            return dict(
                _result((stream,), (empty, empty), fluid),
                phase_count=0,
                phase_roles=["absent", "absent"],
            )
        equilibrium = fluid.flash_ph(
            self.pressure_pa, stream.molar_enthalpy_j_mol, stream.mole_fractions
        )
        outputs = []
        for phase in equilibrium["phases"]:
            rates = tuple(
                stream.molar_rate_mol_s * phase["molar_fraction"] * x
                for x in phase["mole_fractions"]
            )
            outputs.append(
                ThermalStream(
                    rates, self.pressure_pa, phase["molar_enthalpy_j_mol"], stream.model_id
                )
            )
        roles = ["liquid", "vapor"]
        if len(outputs) == 1:
            if self.single_phase_outlet is None:
                raise PropertyModelError(
                    "Single-phase separator needs an explicit single_phase_outlet"
                )
            roles = ["single", "absent"]
            outputs.append(
                ThermalStream(
                    tuple(0.0 for _ in fluid.components), self.pressure_pa, 0.0, stream.model_id
                )
            )
        if equilibrium["phase_count"] == 1 and self.single_phase_outlet == 1:
            outputs.reverse()
            roles.reverse()
        return dict(
            _result((stream,), tuple(outputs), fluid),
            phase_count=equilibrium["phase_count"],
            phase_roles=roles,
        )


def let_down_thermal_stream(stream, pressure_pa, fluid):
    if pressure_pa > stream.pressure_pa:
        raise PropertyModelError("Letdown cannot boost pressure")
    out = ThermalStream(
        stream.component_molar_rates_mol_s,
        pressure_pa,
        stream.molar_enthalpy_j_mol,
        stream.model_id,
    )
    return _result((stream,), (out,), fluid)
